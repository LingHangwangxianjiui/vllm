# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：基于 Ray 的分布式执行器（旧版 / V1 实现）。
# [CN] 与 MultiprocExecutor 的对比：
# [CN]   - 多进程执行器自己 fork/spawn 进程、用共享内存 MQ 通信；
# [CN]   - Ray 执行器把每个 worker 包装成 Ray actor，用 Ray Compiled Graph 通信。
# [CN] 核心机制：
# [CN]   1) Ray actor + PlacementGroup 做资源编排与跨节点调度；
# [CN]   2) pp_tp_workers 二维结构（先 PP 后 TP），直接映射成 Compiled DAG 的拓扑；
# [CN]   3) execute_model 与 sample_tokens 被拆成两阶段，中间靠 self.scheduler_output 传递。
# [CN] 最容易看错的点：
# [CN]   1) actor 的创建顺序是随机的，真正的 rank 要在拿到 IP 之后重新排序（adjust_rank）；
# [CN]   2) 本文件是「旧版 Ray 执行器」，新实现是 ray_executor_v2 的 RayExecutorV2，
# [CN]      由 VLLM_USE_RAY_V2_EXECUTOR_BACKEND 切换；
# [CN]   3) EC connector 在 world_size > 1 时明确不支持，会直接抛错而不是静默出错。


import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cloudpickle

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.ray_env_utils import (
    update_runtime_env_for_worker_import,
)
from vllm.v1.executor.ray_utils import (
    WORKER_SPECIFIC_ENV_VARS,
    FutureWrapper,
    RayWorkerWrapper,
    detach_zero_copy_from_model_runner_output,
    initialize_ray_cluster,
    ray,
)
from vllm.v1.outputs import ModelRunnerOutput

if ray is not None:
    from ray.actor import ActorHandle
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

# [CN] 预置的「已完成且结果为 None」的 Future，用于 non_block 模式下返回空结果，
# [CN] 避免每次都新建 Future 对象。
COMPLETED_NONE_FUTURE: Future[ModelRunnerOutput | None] = Future()
COMPLETED_NONE_FUTURE.set_result(None)


# [CN] Ray worker 的元数据。created_rank 是创建顺序，
# [CN] adjusted_rank 是「按节点/IP 重排后」的最终 rank，两者常不相同。
@dataclass
class RayWorkerMetaData:
    """
    Metadata for a Ray worker.
    The order of ray worker creation can be random,
    and we need to reset the rank after creating all workers.
    """

    worker: ActorHandle
    created_rank: int
    adjusted_rank: int = -1
    ip: str = ""


# [CN] Ray 分布式执行器。
class RayDistributedExecutor(Executor):
    """Ray-based distributed executor"""

    uses_ray: bool = True
    supports_pp: bool = True

    # [CN] 初始化：起 Ray、建 placement group、拉起 actor、完成分布式握手。
    def _init_executor(self) -> None:
        self.forward_dag: ray.dag.CompiledDAG | None = None

        # [CN] TPU/XPU 上没有 NCCL，Compiled DAG 必须走共享内存通道。
        # For TPU or XPU, avoid compiling NVIDIA's NCCL
        if current_platform.is_tpu() or current_platform.is_xpu():
            os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"

        assert self.uses_ray
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        # [CN] 只有配了 KV connector 才需要「从所有 worker 收集输出」。
        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        # [CN] 旧版 Ray 执行器在 world_size>1 时无法聚合 EC connector 状态，
        # [CN] 与其静默丢数据，不如直接报错引导用户切到 V2 或多进程执行器。
        if (
            self.vllm_config.ec_transfer_config is not None
            and self.parallel_config.world_size > 1
        ):
            raise NotImplementedError(
                "EC connector worker metadata is not supported with the "
                "legacy Ray executor when world_size > 1: only the output "
                "of a single worker is fetched, silently dropping the "
                "other workers' EC connector state. Set "
                "VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 to use RayExecutorV2, "
                "or use the multiprocessing executor instead."
            )

        # [CN] pooling 模型不做采样；EC producer 也不做采样（采样在 consumer 侧）。
        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            or self.vllm_config.ec_transfer_config.is_ec_consumer
        )

        self.scheduler_output: SchedulerOutput | None = None

    # [CN] 关停：先拆 DAG，再 kill 所有 actor。
    def shutdown(self) -> None:
        if logger:
            # Somehow logger can be None here.
            logger.info(
                "Shutting down Ray distributed executor. If you see error log "
                "from logging.cc regarding SIGTERM received, please ignore "
                "because this is the expected termination process in Ray."
            )
        # [CN] 只有建过 Compiled DAG 才需要 teardown 与 kill。
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray

            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None

    # [CN] nsight 性能剖析需要写进 runtime_env，Ray 会在 actor 启动时生效。
    def _configure_ray_workers_use_nsight(self, ray_remote_kwargs) -> dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        runtime_env.update(
            {
                "nsight": {
                    "t": "cuda,cudnn,cublas",
                    "o": "'worker_process_%p'",
                    "cuda-graph-trace": "node",
                }
            }
        )

        return ray_remote_kwargs

    # [CN] 告诉 Ray 不要替我们设置可见设备 —— vLLM 自己用 local_rank 索引 GPU。
    def _update_noset_device_env_vars(self, ray_remote_kwargs):
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars.update(
            {env_var: "1" for env_var in current_platform.ray_noset_device_env_vars}
        )
        update_runtime_env_for_worker_import(runtime_env)
        return ray_remote_kwargs

    # child class could overwrite this to return actual env vars.
    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    # [CN] 建 Ray actor 的全过程：选 bundle -> 建 actor -> 重排 rank -> 下发配置 -> 初始化。
    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # [CN] driver dummy worker 不占实际资源，只为占住 driver 的那份资源配额。
        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: RayWorkerWrapper | None = None
        # The remaining workers are the actual ray actors.
        self.workers: list[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        # [CN] pp_tp_workers[pp_rank][tp_rank]：直接对应 Compiled DAG 的层与节点。
        self.pp_tp_workers: list[list[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs
            )

        # [CN] 与 mp 模式一致：不设 CUDA_VISIBLE_DEVICES，改用 local_rank 索引。
        # The way ray actors are setup in vllm is that the visible devices are
        # not set by actors, they are left unset by ray. Internally we index
        # the right gpu with local_rank. This is similar to how mp mode works.
        self._update_noset_device_env_vars(ray_remote_kwargs)

        # Create the workers.
        # [CN] 决定 actor 落在 placement group 的哪些 bundle 上。
        bundle_indices: list[int]
        # [CN] 用户可用 VLLM_RAY_BUNDLE_INDICES 显式指定 bundle，否则自动挑前 N 个带 GPU 的。
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            bundle_indices = list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            assert len(bundle_indices) == self.parallel_config.world_size, (
                "VLLM_RAY_BUNDLE_INDICES must have the same size"
                f" as the world size, but got {bundle_indices=} "
                f"and {self.parallel_config.world_size=}"
            )
            assert len(set(bundle_indices)) == len(bundle_indices), (
                "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
                f" but got {bundle_indices=}"
            )
        else:
            # use the first N bundles that have GPU resources.
            bundle_indices = []
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                if bundle.get(current_platform.ray_device_key, 0):
                    bundle_indices.append(bundle_id)
            bundle_indices = bundle_indices[: self.parallel_config.world_size]

        # [CN] 先按创建顺序建 actor，此时 rank 还不是最终的。
        worker_metadata: list[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if current_platform.ray_device_key == "GPU":
                # NV+AMD GPUs, and Intel XPUs
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_gpus},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)

            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))

        # [CN] 批量 ray.get 拿所有 actor 的节点 IP，用于后续重排。
        worker_ips = ray.get(
            [
                each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
                for each in worker_metadata
            ]
        )

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        logger.debug("workers: %s", worker_metadata)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)

        ip_counts: dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        # [CN] 排序规则：driver 所在节点优先 -> 同节点上 worker 少的优先 -> IP 小的优先。
        # [CN] 目的是让同一节点的 worker 的 rank 连续，通信局部性更好。
        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return 0 if ip == driver_ip else 1, ip_counts[ip], ip

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        # [CN] 重排后下发 rerank_mapping，让每个 actor 更新自己的 rank。
        sorted_worker_metadata = sorted(
            worker_metadata, key=sort_by_driver_then_worker_ip
        )
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank for item in sorted_worker_metadata
        }
        self.collective_rpc("adjust_rank", args=(rerank_mapping,))

        # [CN] 收集每个节点上实际用到的物理 GPU id，用于正确设置设备映射。
        # Get the set of physical GPU IDs used on each node.
        worker_node_and_physical_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_physical_gpu_ids.append(
                ray.get(worker.get_node_and_physical_gpu_ids.remote())  # type: ignore[attr-defined]
            )

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_physical_gpu_ids = defaultdict(list)  # node id -> physical GPU IDs

        for i, (node_id, physical_gpu_ids) in enumerate(
            worker_node_and_physical_gpu_ids
        ):
            node_workers[node_id].append(i)
            # `physical_gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: physical GPU IDs can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            # [CN] 物理 GPU id 必须转成整数再排序：字符串排序下 16 会排在 9 前面（见 issue #5590）。
            physical_gpu_ids = [
                current_platform.device_control_id_to_physical_device_id(str(x))
                for x in physical_gpu_ids
            ]
            node_physical_gpu_ids[node_id].extend(physical_gpu_ids)
        for node_id, physical_gpu_ids in node_physical_gpu_ids.items():
            node_physical_gpu_ids[node_id] = sorted(physical_gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        # [CN] 节点数必须等于唯一 IP 数，否则说明 VLLM_HOST_IP 配重了，通信会连错。
        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP`"
                " environment variable, make sure it is unique for"
                " each node."
            )

        # [CN] 把 driver 的环境变量复制给所有 worker（排除各 worker 独有的那些）。
        all_args_to_update_environment_variables: list[dict[str, str]] = [
            {} for _ in worker_node_and_physical_gpu_ids
        ]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars),
            destination="workers",
        )

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = all_args_to_update_environment_variables

        self.collective_rpc(
            "update_environment_variables", args=(self._get_env_vars_to_be_updated(),)
        )

        # [CN] 单节点时用 127.0.0.1 最稳：get_ip() 可能返回某个不通的网卡地址，
        # [CN] 而 loopback 在节点内一定可达。
        if len(node_physical_gpu_ids) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        # [CN] 分布式初始化地址：driver_ip + 一个空闲端口。
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port()
        )

        # Initialize the actual workers inside worker wrapper.
        # [CN] 为每个 actor 组装初始化参数，local_rank 取「本节点内的序号」。
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_physical_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            kwargs = dict(
                vllm_config=self.vllm_config,
                assigned_physical_gpu_ids=sorted(node_physical_gpu_ids[node_id]),
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        # [CN] 依次完成：init_worker -> init_device -> load_model -> 回填 block size。
        self.collective_rpc("init_worker", args=(all_kwargs,))

        self.collective_rpc("init_device")
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.collective_rpc("elastic_ep_execute", args=("load_model",))
        else:
            self.collective_rpc("load_model")

        def _update_block_size(worker):
            current_platform.update_block_size_for_backend(worker.vllm_config)

        self.collective_rpc(_update_block_size)

        # [CN] 按 PP/TP 二维填充 pp_tp_workers，供 Compiled DAG 使用。
        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            self.pp_tp_workers.append([])
            for tp_rank in range(self.parallel_config.tensor_parallel_size):
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                rank = (pp_rank * self.parallel_config.tensor_parallel_size) + tp_rank
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                assert pp_rank < len(self.pp_tp_workers)
                self.pp_tp_workers[pp_rank].append(self.workers[rank])

    # [CN] 弹性伸缩：重建分布式组；若是「关停当前 rank」则顺带关掉执行器。
    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.collective_rpc("reinitialize_distributed", args=(reconfig_request,))
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()

    # [CN] 两阶段执行之第一步：只把 scheduler_output 存起来，真正执行推迟到 sample_tokens。
    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # [CN] 状态机保护：上一次的输出还没取走（返回 None 后必须调 sample_tokens）。
        if self.scheduler_output is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # [CN] 不需要采样（pooling 或空批次）时直接同步执行完。
        if not self.uses_sampler or not scheduler_output.total_num_scheduled_tokens:
            # Model will not execute, call model runner immediately.
            return self._execute_dag(scheduler_output, None, non_block)

        # Model will execute, defer to sample_tokens() call.
        self.scheduler_output = scheduler_output
        return COMPLETED_NONE_FUTURE if non_block else None

    # [CN] 两阶段执行之第二步：真正驱动 DAG 执行并取回结果。
    def sample_tokens(  # type: ignore[override]
        self,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """Execute the model on the Ray workers.

        The scheduler output to use should have been provided in
        a prior call to execute_model().

        Args:
            grammar_output: The structured outputs grammar bitmask, if applicable.
            non_block: If True, the method will return a Future.

        Returns:
            The model runner output.
        """
        scheduler_output = self.scheduler_output
        if scheduler_output is None:
            return COMPLETED_NONE_FUTURE if non_block else None

        self.scheduler_output = None

        return self._execute_dag(scheduler_output, grammar_output, non_block)

    # [CN] 驱动 Compiled DAG 执行；首次调用时惰性编译。
    def _execute_dag(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # Build the compiled DAG for the first time.
        # [CN] Compiled DAG 编译开销大，只在首次执行时构建一次。
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        refs = self.forward_dag.execute((scheduler_output, grammar_output))  # type: ignore

        # [CN] 无 connector 时只需一个 worker 的输出；有 connector 时必须收集全部再聚合。
        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            # [CN] 不用 PP 时直接阻塞取结果；用 PP 时返回 Future，让调度器可以去排下一批。
            if not non_block:
                output = refs[0].get()
                # [CN] detach_zero_copy：把输出从 Ray 的零拷贝缓冲区里摘出来，避免持有大块共享内存。
                detach_zero_copy_from_model_runner_output(output)
                return output

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return FutureWrapper(refs[0])

        # Get output from all workers when connector is present
        assert self.kv_output_aggregator is not None
        if not non_block:
            # Block and get results from all workers
            outputs = ray.get(refs)
            for output in outputs:
                detach_zero_copy_from_model_runner_output(output)
            return self.kv_output_aggregator.aggregate(outputs)

        # Return a future that will aggregate outputs from all workers
        return FutureWrapper(refs, self.kv_output_aggregator)

    # [CN] Ray 版 collective_rpc：直接对每个 actor 发 execute_method。
    # [CN] 注意这里不走 Compiled DAG —— 控制消息用普通 actor 调用即可。
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any] | Future[list[Any]]:
        """Runs the given method on all workers."""
        # [CN] 方法名走字符串，可调用对象走 cloudpickle。
        sent_method = method if isinstance(method, str) else cloudpickle.dumps(method)
        del method

        if kwargs is None:
            kwargs = {}
        ray_worker_outputs = [
            worker.execute_method.remote(  # type: ignore[attr-defined]
                sent_method, *args, **kwargs
            )
            for worker in self.workers
        ]

        # Get the results of the ray workers.
        if non_block:
            return FutureWrapper(ray_worker_outputs)

        return ray.get(ray_worker_outputs, timeout=timeout)

    # [CN] 前置检查：Ray 版本、compiled_dag_ref、以及 nccl 通道所需的 cupy。
    def _check_ray_cgraph_installation(self):
        import importlib.metadata

        from packaging import version

        required_version = version.parse("2.43.0")
        current_version = version.parse(importlib.metadata.version("ray"))
        if current_version < required_version:
            raise ValueError(
                f"Ray version {required_version} is "
                f"required, but found {current_version}"
            )

        import importlib.util

        cgraph_spec = importlib.util.find_spec("ray.experimental.compiled_dag_ref")
        if cgraph_spec is None:
            raise ValueError(
                "Ray Compiled Graph is not installed. "
                "Run `pip install ray[cgraph]` to install it."
            )

        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE == "nccl":
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE is set to 'nccl'. "
                "Run `pip install ray[cgraph]` and check cupy installation."
            )

    # [CN] 构建 Ray Compiled DAG：PP 各 stage 串行、TP 组内 SPMD 并行。
    def _compiled_ray_dag(self, enable_asyncio: bool):
        assert self.parallel_config.use_ray
        self._check_ray_cgraph_installation()
        # [CN] 必须在 import ray.dag 之前设置该环境变量，否则不生效。
        # [CN] 默认 10 秒太短，大模型一次前向很容易超时。
        # Enlarge the default value of "RAY_CGRAPH_get_timeout" to 300 seconds
        # (it is 10 seconds by default). This is a Ray environment variable to
        # control the timeout of getting result from a compiled graph execution,
        # i.e., the distributed execution that includes model forward runs and
        # intermediate tensor communications, in the case of vllm.
        # Note: we should set this env var before importing
        # ray.dag, otherwise it will not take effect.
        os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")  # noqa: SIM112
        from ray.dag import InputNode, MultiOutputNode

        logger.info(
            "RAY_CGRAPH_get_timeout is set to %s",
            os.environ["RAY_CGRAPH_get_timeout"],  # noqa: SIM112
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE,
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )

        channel_type = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
        if channel_type not in ("auto", "nccl", "shm"):
            raise ValueError(
                "Invalid value for VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: "
                f"{channel_type}. Valid values are: 'auto', 'nccl', or 'shm'."
            )

        # [CN] InputNode 作为 DAG 输入；下面按 PP 逐层串联。
        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            #
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501

            # All workers in the first TP group will take in the
            # ExecuteModelRequest as input.
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            # [CN] 每个 PP stage 接收上一层的输出；同一层内各 TP rank 并行执行。
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker,
                # and the TP group executes in SPMD fashion.
                outputs = [
                    worker.execute_model_ray.bind(outputs[i])  # type: ignore[attr-defined]
                    for i, worker in enumerate(tp_group)
                ]

                # [CN] 只有中间 stage 需要指定中间张量的传输方式；最后一层和 shm 通道不需要。
                last_pp_rank = len(self.pp_tp_workers) - 1
                if (
                    pp_rank < last_pp_rank
                    and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE != "shm"
                ):
                    # Specify how intermediate tensors should be passed
                    # between pp stages, no need to specify for the last
                    # pp stage or when using shared memory (the default).
                    transport = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
                    outputs = [
                        output.with_tensor_transport(transport=transport)
                        for output in outputs
                    ]

            # [CN] MultiOutputNode：把最后一个 PP stage 的所有 TP 输出作为 DAG 的输出。
            forward_dag = MultiOutputNode(outputs)

        # [CN] 可选：用 vLLM 自己的 PP 通信器替代 Ray 的 NCCL 通信器。
        if envs.VLLM_USE_RAY_WRAPPED_PP_COMM:
            from ray.experimental.channel.accelerator_context import (
                register_accelerator_context,
            )

            from vllm.distributed.device_communicators.ray_communicator import (
                RayPPCommunicator,
            )

            register_accelerator_context(
                torch_module_name="cuda", communicator_cls=RayPPCommunicator
            )
            logger.info(
                "Using RayPPCommunicator "
                "(which wraps vLLM _PP GroupCoordinator) "
                "for Ray Compiled Graph communication."
            )
        else:
            logger.info(
                "Using Ray's NCCL communicator for Ray Compiled Graph communication."
            )

        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            _overlap_gpu_communication=envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )

    # [CN] 析构时关停，避免 Ray actor 泄漏。
    def __del__(self):
        self.shutdown()

    # [CN] 当前实现假定 Ray worker 始终健康，不做实际检查。
    def check_health(self) -> None:
        # Assume that the Ray workers are healthy.
        # TODO: check the health of the Ray workers
        return
