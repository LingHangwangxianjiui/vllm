# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：单进程执行器（UniprocExecutor）。
# [CN] 链路位置：EngineCore -> Executor.collective_rpc/execute_model -> Worker。
# [CN]   本执行器不 fork 子进程，driver_worker 就活在当前进程里，
# [CN]   因此 collective_rpc 退化为一次普通函数调用（run_method）。
# [CN] 核心类：
# [CN]   - AsyncOutputFuture：把 AsyncModelRunnerOutput 包装成 Future，
# [CN]     让 non_block=True 的调用在多进程与单进程下接口一致。
# [CN]   - UniProcExecutor：调试首选（断点可直达模型内部），也用于 TP=1 场景。
# [CN]   - ExecutorWithExternalLauncher：torchrun 兼容模式，见文件末尾说明。
# [CN] 最容易看错的点：collective_rpc 返回值恒为「长度 1 的列表」语义，
# [CN]   子类里 single_value=True 才取标量；抽象基类默认返回 output[0]。

import os
from collections.abc import Callable
from concurrent.futures import Future
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    aiter_requires_tcp_store,
    get_distributed_init_method,
    get_file_store_init_method,
    get_ip,
    get_open_port,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


# [CN] 把「异步产出的 ModelRunnerOutput」伪装成 concurrent Future。
# [CN] 存在意义：多进程执行器天然返回 Future，单进程也要能返回 Future，
# [CN] 否则上层的 async scheduling 代码要写两套分支。
class AsyncOutputFuture(Future):
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    # [CN] 惰性求值：只有真的调用 result() 时才去阻塞取输出。
    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not super().done():
            try:
                output = self.async_output.get_output()
                self.set_result(output if self.single_value else [output])
            except Exception as e:
                self.set_exception(e)
        return super().result()


# [CN] 单进程执行器：worker 与 EngineCore 同进程、同 GIL。
class UniProcExecutor(Executor):
    # [CN] 构造 WorkerWrapperBase 并完成「初始化设备 + 加载权重」两步。
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        set_worker_net_device(local_rank, self.vllm_config)

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        # [CN] 弹性 EP 扩容时走专属入口，保证 tp/pp 组重建后再加载权重。
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            self.driver_worker.load_model()
        current_platform.update_block_size_for_backend(self.vllm_config)

    # [CN] 单进程下 rank 恒为 0；local_rank 从 device 字符串（如 cuda:1）解析。
    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        if aiter_requires_tcp_store():
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port()
            )
        else:
            distributed_init_method = get_file_store_init_method()
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    # [CN] 单进程版 RPC：没有序列化、没有消息队列，直接反射调用 worker 方法。
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # [CN] 同步路径：直接拿到结果；若 worker 返回的是异步句柄就地展开。
        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            return result if single_value else [result]

        # [CN] 异步路径：把结果塞进一个已完成的 Future，异常也塞进 Future，
        # [CN] 这样调用方无需区分「跑完了」和「报错了」两种时序。
        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, single_value)
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    # [CN] 执行一步模型前向。single_value=True 表示取标量而非长度 1 列表。
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # In non-blocking mode, surface any exception as early as possible.
        if non_block and output.done():
            # Raise the exception in-line if the task failed.
            output.result()
        return output

    # [CN] 独立的采样阶段：async scheduling 下前向与采样被拆成两次 RPC，
    # [CN] 采样可在本步调度下一批之前完成，从而重叠 CPU 与 GPU。
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    # [CN] 取走投机解码产生的草稿 token（取走即清空，避免重复消费）。
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    # [CN] 同进程执行器不会「部分失败」：进程活着就是健康的。
    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()

    # [CN] 单进程天然支持 async scheduling（无跨进程同步开销问题）。
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True


# [CN] torchrun 兼容执行器：每个 executor 只建 1 个 worker，
# [CN] 由外部启动器拉起多个引擎进程共同完成 TP。
class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/features/torchrun/torchrun_example_offline.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    # [CN] 用 env:// 初始化，rank/local_rank 直接取自 torchrun 注入的环境变量。
    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    # [CN] 多引擎各测各的显存，必须取全局最小值，否则某个 rank 会 OOM。
    def determine_available_memory(self) -> list[int]:  # in bytes
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
