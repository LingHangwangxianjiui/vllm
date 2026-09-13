# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：Executor 抽象基类与「按配置选执行器」的工厂。
# [CN] 链路位置：EngineCore 持有唯一一个 Executor；Executor 之下是若干 Worker。
# [CN]   SchedulerOutput --Executor.execute_model--> ModelRunnerOutput。
# [CN] 核心类：Executor（ABC）。子类有三条主线：
# [CN]   - UniProcExecutor：单进程，调试/TP=1；
# [CN]   - MultiprocExecutor：多进程，默认生产路径；
# [CN]   - RayDistributedExecutor / RayExecutorV2：Ray 编排。
# [CN] 设计要点：
# [CN]   1) 几乎所有能力都是 collective_rpc 的语法糖（广播给所有 worker）；
# [CN]   2) 返回值一律是「每 worker 一个元素」的列表，基类多数方法取 [0]；
# [CN]   3) 文件末尾为了向后兼容又导入了 UniProcExecutor，形成轻微循环引用，
# [CN]      所以 import 被放在文件底部并标注 noqa: E402。
# [CN] 最容易看错的点：collective_rpc 的 docstring 明确警告「只用于传控制消息」，
# [CN]   高频张量数据必须走数据面（共享内存 / NCCL），否则性能会崩。

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


# [CN] 执行器抽象：屏蔽「单进程 / 多进程 / Ray」的差异，向上提供统一 API。
class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """

    # [CN] 两个能力标志，供 EngineCore 判断能否走某些优化路径。
    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    supports_pp: bool = False  # whether the executor supports PP

    # [CN] 工厂方法：根据 parallel_config.distributed_executor_backend 选子类。
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        # [CN] 允许直接传类对象（便于测试与自定义后端注入）。
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        # [CN] Ray 有 v1/v2 两代实现，由环境变量 VLLM_USE_RAY_V2_EXECUTOR_BACKEND 切换。
        elif distributed_executor_backend == "ray":
            if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

                executor_class = RayExecutorV2
            else:
                from vllm.v1.executor.ray_executor import RayDistributedExecutor

                executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        # [CN] external_launcher 用 torchrun 语义启动，需要调度是确定性的。
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        # [CN] 字符串形式的自定义后端：按全限定名动态解析。
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    # [CN] 构造即完成初始化（_init_executor 由子类实现）。
    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        # [CN] 子类在此拉起 worker 进程 / 建 Ray actor / 直接建 worker 对象。
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self.kv_output_aggregator: KVOutputAggregator | None = None
        self.ec_output_aggregator: ECOutputAggregator | None = None

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    # [CN] 把调度器算好的 KVCacheConfig 下发给每个 worker 真正分配显存。
    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Initialize the KV caches on the underlying workers."""
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))

    # [CN] 触发 torch.compile 与 CUDA Graph 捕获（耗时大头，只在启动阶段做一次）。
    def compile_or_warm_up_model(self) -> None:
        """Compile/warm up the model and capture cudagraphs on workers."""
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        # [CN] 编译发生在 worker 进程内，主进程的 config 不会被更新，必须回传。
        # [CN] 取 max 而非 sum：各 worker 并行编译，墙钟时间取决于最慢的那个。
        # Propagate compilation time from workers back to the main process.
        # With TP>1, compilation happens in worker processes, so the main
        # process config is never updated. Use max across workers since they
        # compile in parallel.
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    # [CN] 显存画像：每个 worker 返回自己可用的 KV cache 字节数。
    def determine_available_memory(self) -> list[int]:  # in bytes
        return self.collective_rpc("determine_available_memory")

    # [CN] 收集各 worker 的 KVCacheSpec，供调度器决定 block 大小与分组。
    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")

    # [CN] 各后端支持的 KV cache 布局（NHD/HND 等），按偏好排序。
    def get_supported_kv_cache_layouts(self) -> list[list[str]]:
        """Layouts each worker's backends support, most preferred first."""
        return self.collective_rpc("get_supported_kv_cache_layouts")

    # [CN] 把最终选定的布局广播给 worker，保证跨进程一致。
    def set_kv_cache_layout(self, layout_name: str) -> None:
        """Publish the resolved KV cache layout to the workers."""
        self.collective_rpc("set_kv_cache_layout", args=(layout_name,))

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        pass

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        pass

    # [CN] 唯一的真抽象：所有广播最终都落到这里，由子类决定怎么送达。
    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        raise NotImplementedError

    # [CN] KV connector 握手元数据（P/D 分离场景下 P 与 D 需要对齐的信息）。
    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[tuple[int, int], KVConnectorHandshakeMetadata]]:
        return self.collective_rpc("get_kv_connector_handshake_metadata")

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        pass

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        pass

    # [CN] 执行一步前向，取 driver worker（下标 0）的结果。
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        pass

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        pass

    # [CN] 仅做采样（前向已在上一步完成），配合 async scheduling 使用。
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    # [CN] 跑一个空批次：用于 DP 下的 wave 同步或保持 NCCL 通信活跃。
    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.collective_rpc("shutdown")

    # [CN] KV connector 的输出在多 worker 间需要聚合（如 TP 下每个 rank 持有分片）。
    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """Init KVOutputAggregator"""
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    def init_ec_output_aggregator(self) -> None:
        self.ec_output_aggregator = ECOutputAggregator()

    # [CN] cached_property：避免每次查询都发一轮 RPC。
    @cached_property  # Avoid unnecessary RPC calls
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def supports_draft_weight_updates(self) -> bool:
        worker_support: list[bool] = self.collective_rpc(
            "supports_draft_weight_updates"
        )
        return all(worker_support)

    # [CN] LoRA 相关：要求所有 worker 都成功才算成功。
    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        sets: list[set[int]] = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache in each worker."""
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        self.collective_rpc("reset_encoder_cache")

    # [CN] 休眠：level 越大释放越多（1=仅 KV cache，2=连同权重一起卸载）。
    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    # [CN] 唤醒：可按 tag 部分唤醒；只有全部 tag 都唤醒后才清除 is_sleeping。
    def wake_up(self, tags: list[str] | None = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        raise NotImplementedError

    # [CN] 是否支持 async scheduling。多进程执行器会覆写为 True。
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        Whether the executor supports async scheduling.
        """
        return False


# [CN] 底部延迟导入：打破 abstract <-> uniproc 的循环引用，同时保留旧名字。
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)

# For backwards compatibility.
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
