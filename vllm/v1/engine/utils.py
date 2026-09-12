# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# [CN] 文件总览：**引擎进程的启动、握手与生命周期管理**。
#
#     这个文件回答的是「vLLM 启动时到底起了几个进程、谁先谁后、它们怎么找到
#     彼此」。它是 V1 多进程架构的**装配车间**。
#
#     ===================== 三条并行的部署路径 =====================
#     ① **本地多进程**（CoreEngineProcManager）：用 multiprocessing 起
#        N 个 EngineCore 进程，每个进程一个 DP rank，ZMQ 走 IPC 或 TCP。
#     ② **Ray 后端**（CoreEngineActorManager）：把 EngineCore 包成 Ray
#        actor，可跨节点调度，用 placement group 精确控制 device 亲和性。
#     ③ **Elastic EP**：在 Ray 路径上支持运行时扩缩容（scale_up / down）。
#
#     ===================== 启动时序（最容易看晕的地方）=====================
#     1. get_engine_zmq_addresses()：**先分配地址**。此时 TCP 端口可能还是 0，
#        真正的端口要等 bind 之后才知道（见下面的「端口回填」）。
#     2. launch_core_engines()：起 DP 协调器 → 起 N 个引擎进程 → yield。
#        **注意它是 contextmanager**：yield 之后的代码（等待就绪）
#        要等调用方的 with 块结束才执行。
#     3. wait_for_engine_startup()：在 ROUTER 套接字上收 HELLO / READY，
#        回握手元数据（把地址表交给引擎），所有引擎 READY 才返回。
#
#     ===================== 三个易错设计点 =====================
#     · **端口回填**：TCP 地址先写成 host:0，由真正 bind 的一方在 bind 之后
#       用 getsockopt(zmq.LAST_ENDPOINT) 取回内核分配的端口写回。
#       提前是不知道端口的，硬猜必然撞车。
#     · **配置深拷贝**：DP 下所有 rank 名义上共享一个 VllmConfig，但每个 rank
#       要有自己的 instance_id 和 GPU 分片。所以是按 rank 逐个 deepcopy 再改，
#       绝不能在循环里原地改同一份配置。
#     · **ROCm 关停宽限**：shutdown_timeout=0 表示「收到 SIGTERM 立刻中止请求」，
#       但进程仍需一点时间释放显存。强制 kill 会把显存留在卡上，
#       所以 ROCm 平台额外给 15 秒清理窗口（见下面的函数）。
import contextlib
import os
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import TYPE_CHECKING, cast

import msgspec
import zmq

from vllm import envs
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils import numa_utils
from vllm.utils.network_utils import (
    get_open_port,
    get_open_zmq_ipc_path,
    get_tcp_uri,
    zmq_socket_ctx,
)
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.executor import Executor
from vllm.v1.executor.ray_utils import WORKER_SPECIFIC_ENV_VARS
from vllm.v1.utils import _SubprocessWrapper, get_engine_client_zmq_addr, shutdown

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

# [CN] 启动阶段的轮询周期（毫秒）。用于等待引擎进程握手。
STARTUP_POLL_PERIOD_MS = 10000
# [CN] ROCm 平台强制 kill 前的额外清理窗口（秒）。
ROCM_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S = 15.0


# [CN] 计算「进程管理器」的关停超时。
#      注意区分两个超时：request_timeout 控制在途请求能排空多久；
#      process_timeout 控制进程被强杀前有多少时间释放设备资源。
#      只有两者都为 0（即「立即中止」）且是 ROCm 时，才额外给宽限，
#      因为 ROCm 的 teardown 较慢，强杀会留下驻留显存。
def get_engine_process_shutdown_timeout(
    request_timeout: float | None,
    process_timeout: float | None,
) -> float | None:
    """Return the EngineCore process-manager shutdown timeout.

    ``VllmConfig.shutdown_timeout`` controls how long in-flight requests may
    drain. A value of zero therefore tells EngineCore to abort requests as soon
    as it receives SIGTERM. The parent process manager still needs a separate
    window in which the EngineCore can release device resources before it is
    force-killed. ROCm teardown can take longer than the generic best-effort
    window, and force-killing during teardown can leave VRAM resident.

    ``process_timeout`` may be a remaining budget computed by an outer process
    manager. Keep it unchanged unless both values are zero: a zero remaining
    budget for a positive request timeout must not receive a fresh grace period
    because EngineCore relies on that deadline to enforce request draining.
    """
    if request_timeout == 0 and process_timeout == 0 and current_platform.is_rocm():
        return ROCM_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S
    return process_timeout


# [CN] 单个引擎进程在握手过程中的状态机：NEW → CONNECTED → READY。
class CoreEngineState(Enum):
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


# [CN] 一个 DP rank 对应的引擎描述，仅用于握手期跟踪状态。
class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    # [CN] local=True 表示这个引擎与当前前端进程同机（可走 IPC）。
    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        # [CN] ZMQ ROUTER 用的身份标识：2 字节小端 rank。握手时靠它认人。
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


# [CN] 引擎与前端通信用的 ZMQ 地址集合。
#      它是**先分配、后回填**的：创建时端口可能是 0，真正 bind 之后再写回。
@dataclass
class EngineZmqAddresses:
    # [CN] 前端 → 引擎的请求套接字地址（每个 API server 一个）。
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # [CN] 引擎 → 前端的响应套接字地址（每个 API server 一个）。
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # [CN] DP 协调器的输入地址；非 DP 场景为 None。
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None
    # [CN] 前端订阅协调器负载统计的发布地址。
    #      引擎自己不用它，只是在握手时**转交**给前端；
    #      只有外部 DP 负载均衡模式才需要。
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None


# [CN] 握手时发给引擎进程的元数据：主要是「前端地址表」。
#      引擎拿到它才知道该去连哪些套接字。
@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]


# [CN] 构造 placement group 中的「控制 bundle」（只占 1 个 CPU）。
#      为什么要带 node 亲和：engine actor 调度在最后一个 CPU-only bundle 上，
#      必须把它钉在组内第一个 GPU bundle 所在的节点，否则 actor 可能飘到别的
#      节点，导致 worker rank 顺序与对外通告的 DP bootstrap 主机不一致。
def _make_control_bundle(node_ip: str) -> dict[str, float]:
    # The engine actor is scheduled on the final CPU-only bundle. Keep that
    # bundle colocated with the group's first GPU bundle so the actor does not
    # float to an unrelated node and reorder worker ranks away from the
    # advertised DP bootstrap host.
    return {"CPU": 1.0, "node:" + node_ip: 0.001}


# [CN] 从 bundle 里解出 node:<ip> 形式的节点亲和地址。
def _get_bundle_node_ip(bundle: dict[str, float]) -> str:
    for key in bundle:
        if key.startswith("node:"):
            return key.split(":", 1)[1]
    raise ValueError(f"Missing node affinity in placement bundle: {bundle}")


# [CN] 从 Ray 的节点资源字典里解出节点 IP。
#      要跳过 node:__internal_head__（head 节点标记）和带 _group_ 的键。
def _node_ip_from_resources(node_resources: dict) -> str | None:
    """Return the node IP encoded in a Ray per-node resource dict, or None.

    Ray advertises each node's IP as a ``node:<ip>`` resource key. The head node
    also carries ``node:__internal_head__``, and placement groups add
    ``..._group_...`` keys; both are ignored.
    """
    for key in node_resources:
        if (
            key.startswith("node:")
            and key != "node:__internal_head__"
            and "_group_" not in key
        ):
            return key.split(":", 1)[1]
    return None


# [CN] **本地多进程**路径的引擎进程管理器：负责启动、探活、关停。
class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        self._request_shutdown_timeout = vllm_config.shutdown_timeout
        # [CN] 取多进程上下文（spawn / fork，取决于平台配置）。
        context = get_mp_context()
        # [CN] 所有 rank 共用的启动参数，逐 rank 再叠加 dp_rank / local_dp_rank。
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        # [CN] 本地前端进程另有一个握手地址时一并传给引擎。
        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        # [CN] 只有 DP>1 时才需要给进程名加 rank 后缀。
        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        # [CN] 延迟导入：避免循环依赖（core.py 反过来也引用本模块）。
        from vllm.v1.engine.core import EngineCoreProc

        self.processes: list[BaseProcess] = []
        local_dp_ranks = []
        # [CN] 逐个 rank 建进程对象。**注意此时还没 start()**。
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            # Start EngineCore in background process.
            local_dp_ranks.append(local_index)
            # [CN] 每个进程跑 EngineCoreProc.run_engine_core。
            self.processes.append(
                context.Process(
                    target=EngineCoreProc.run_engine_core,
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )

        # [CN] 注册弱引用 finalizer：即使调用方忘记 shutdown，
        #      对象被 GC 时也会兜底把子进程收掉，避免孤儿进程。
        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        # [CN] 主动关停标志位。探活线程与 shutdown 都看它，避免重复关停。
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        # All ranks share this config object: capture the user-provided
        # --device-ids list before the per-rank shard overwrites it. Mutating
        # the config before each proc.start() works because the spawn method
        # pickles process args at start() time, sequentially per rank.
        # [CN] 关键：先把用户原始的 --device-ids 列表抓下来。
        #      因为下面会**逐 rank 覆写** config 里的这个字段，不先存就丢了。
        user_assigned_gpu_ids = vllm_config.parallel_config.assigned_physical_gpu_ids
        try:
            # [CN] 逐个 rank 启动。config 是共享对象，在每次 start() 前就地修改，
            #      依赖 spawn 在 start() 时**逐个** pickle 参数的特性才安全。
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # Populate the logical-to-physical GPU mapping in DP for
                # platforms that cannot rely on
                # torch.accelerator.set_device_index(), and for Ray.
                # [CN] 判断是否需要靠环境变量做设备隔离。CUDA / XPU 可以直接
                #      用 torch.accelerator.set_device_index()，其余平台只能靠 env var。
                needs_device_env_isolation = not (
                    current_platform.is_cuda_alike() or current_platform.is_xpu()
                )
                if is_dp and (
                    needs_device_env_isolation or vllm_config.parallel_config.use_ray
                ):
                    # [CN] 为这个 DP rank 计算并设置它独占的物理 GPU 列表。
                    set_assigned_physical_gpu_ids_for_dp_rank(
                        vllm_config, local_dp_rank, user_assigned_gpu_ids
                    )

                # [CN] NUMA 绑定。EngineCore 自身没有 TP/PP local rank，所以传 0，
                #      含义是「本 DP 分片内的第一张本地卡」。
                #      真正的 TP/PP worker 由 executor 另行绑定各自的 local_rank。
                with numa_utils.configure_subprocess(
                    # EngineCore itself does not have a TP/PP-local rank.
                    # When DP is enabled, set_assigned_physical_gpu_ids_for_dp_rank()
                    # populates the logical-to-physical mapping for this DP
                    # shard, so local_rank=0 means "the first local GPU in
                    # this shard". The actual TP/PP worker processes spawned
                    # by the executor are bound separately with their own
                    # local_rank values.
                    vllm_config,
                    local_rank=0,
                    dp_local_rank=local_dp_rank,
                    process_kind="EngineCore",
                ):
                    proc.start()
        # [CN] 只要有一个进程已经退出就立刻收摊，避免留下半启动状态的进程组。
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.shutdown()

    # [CN] 关停所有引擎进程。
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine core processes with configurable timeout."""
        # [CN] 先置停止标志，让探活线程自然退出，避免它把正常关停误判成崩溃。
        self.manager_stopped.set()
        # [CN] detach 掉 finalizer：既然已经显式关停，就不需要 GC 兜底了。
        #      detach 返回 None 说明已经被触发过（重复关停），直接跳过。
        if self._finalizer.detach() is not None:
            process_timeout = get_engine_process_shutdown_timeout(
                self._request_shutdown_timeout, timeout
            )
            if process_timeout != timeout:
                logger.info(
                    "[shutdown] EngineCore process manager: using %ss ROCm "
                    "cleanup grace after immediate request abort",
                    process_timeout,
                )
            shutdown(self.processes, timeout=process_timeout)

    # [CN] 探活：在**独立线程**里等子进程退出，发现异常退出就记录名字并关停。
    def monitor_engine_liveness(self) -> None:
        """Monitor engine core process liveness."""

        # [CN] 建立 sentinel(fd) → 进程的映射。sentinel 在进程退出时变为可读。
        sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
        sentinels = set(sentinel_to_proc.keys())

        # [CN] 每秒轮询一次。用 wait 而不是 join，是为了能同时响应停止标志。
        while sentinels and not self.manager_stopped.is_set():
            died_sentinels = connection.wait(sentinels, timeout=1)

            for sentinel in died_sentinels:
                proc = sentinel_to_proc.pop(cast(int, sentinel))
                exitcode = proc.exitcode
                # [CN] 非零退出码才算失败；正常关停时不记录。
                if exitcode != 0 and not self.manager_stopped.is_set():
                    self.failed_proc_name = proc.name
            if died_sentinels:
                break

        self.shutdown()

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    # [CN] 返回已结束进程的 {名字: 退出码}，用于启动失败时给出可读的错误信息。
    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


# [CN] 把信号处理器里的回调**挪到专用线程**执行。
#      为什么：信号处理器运行在内核交付的上下文中，里面做复杂操作
#      （加锁、IO、抛异常）极易死锁。这里只 set 一个 Event，
#      真正的回落在独立线程里跑。
class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""

    def __init__(self, callback: Callable[[], None]):
        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        # [CN] daemon 线程：不阻塞主进程退出。
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    def _run(self):
        self._event.wait()
        # [CN] 再次检查停止标志：stop() 与 trigger() 都可能唤醒这个 Event。
        if not self._stopped:
            self._callback()

    def trigger(self):
        self._event.set()

    def stop(self):
        self._stopped = True
        self._event.set()


# [CN] 为指定 DP rank 计算它独占的物理 GPU 列表并写回 config。
def set_assigned_physical_gpu_ids_for_dp_rank(
    vllm_config: VllmConfig,
    local_dp_rank: int,
    user_assigned_gpu_ids: list[int] | None = None,
) -> None:
    """
    Populate assigned_physical_gpu_ids on the config for the given DP rank.

    user_assigned_gpu_ids is the full (un-sharded) --device-ids list, if the
    user provided one; this DP rank's shard is sliced from it. It is passed
    explicitly rather than read from the config because callers may reuse
    one config object across DP ranks, overwriting the field each time.
    """
    # [CN] world_size 是**每个 DP rank 内部**的 TP×PP 大小，
    #      也就是一个 rank 要占几张卡。
    world_size = vllm_config.parallel_config.world_size
    local_world_size = vllm_config.parallel_config.local_world_size
    evar = current_platform.device_control_env_var

    # [CN] 优先用用户显式指定的 --device-ids 分片；否则从 env var 推导。
    physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
        evar,
        local_dp_rank,
        world_size,
        local_world_size,
        user_assigned_gpu_ids=user_assigned_gpu_ids,
    )
    vllm_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids


# [CN] 计算某个 DP rank 应该拿到哪几张物理卡。
#      例：world_size=2、local_dp_rank=1、共 4 张卡 → [2, 3]。
def get_physical_gpu_ids_for_local_dp_rank(
    device_control_env_var: str,
    local_dp_rank: int,
    world_size: int,
    local_world_size: int | None = None,
    user_assigned_gpu_ids: list[int] | None = None,
) -> list[int]:
    """
    Returns list of physical GPU IDs for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will return [2, 3] for local_dp_rank=1.

    If user_assigned_gpu_ids is provided (e.g. from --device-ids), this DP
    rank's shard is sliced from it instead of being derived from the
    device-control env var.
    """
    if local_world_size is None:
        local_world_size = world_size
    # [CN] 用户指定了 --device-ids：直接从里面切片。
    if user_assigned_gpu_ids is not None:
        start = local_dp_rank * world_size
        stop = start + local_world_size
        # [CN] 设备不够分就明确报错，而不是静默复用别人的卡。
        if stop > len(user_assigned_gpu_ids):
            raise ValueError(
                f"--device-ids provides {len(user_assigned_gpu_ids)} devices, "
                f"but DP rank {local_dp_rank} needs devices [{start}, {stop})"
            )
        return user_assigned_gpu_ids[start:stop]
    # [CN] 未指定：按 rank 顺序把「逻辑序号」映射到「物理卡号」。
    #      这层映射很关键 —— CUDA_VISIBLE_DEVICES 会重排可见序号，
    #      逻辑第 i 张卡未必是物理第 i 张。
    try:
        return [
            current_platform.device_id_to_physical_device_id(i)
            for i in range(
                local_dp_rank * world_size,
                local_dp_rank * world_size + local_world_size,
            )
        ]
    except IndexError as e:
        raise Exception(
            f"Error computing device indices for "
            f"{device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e


# [CN] 给 DP rank 的配置加上身份后缀（instance_id、KV connector engine_id）。
#      为什么必须唯一：Ray actor 名和 KV connector 的 engine_id 都是
#      全局注册的，兄弟 DP 引擎重名会直接冲突。
#      用**全局** rank 而不是节点内 rank，因为兄弟引擎可能跨节点。
def _apply_dp_identity_suffix(dp_vllm_config, dp_rank: int) -> None:
    # Ray actor names (RayExecutorV2) and KV-connector engine_ids must
    # be unique across sibling DP engines or registration collides.
    # Use the global DP rank, not a node-local rank, since sibling DP
    # engines can span multiple nodes.
    dp_vllm_config.instance_id = f"{dp_vllm_config.instance_id}_dp{dp_rank}"
    if dp_vllm_config.kv_transfer_config is not None:
        dp_vllm_config.kv_transfer_config.engine_id = (
            f"{dp_vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
        )


# [CN] **Ray 路径**的引擎管理器：可同时管理本地与远端节点上的引擎 actor。
#      与 CoreEngineProcManager 的区别：后者只管本机子进程。
class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        placement_groups: list["PlacementGroup"] | None = None,
        local_dp_ranks: list[int] | None = None,
    ):
        # [CN] Ray 相关导入全部延迟到函数内，保证不装 Ray 也能用其它路径。
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        # [CN] MoE + DP 场景要用 DPMoEEngineCoreActor（多了 wave 同步逻辑）。
        dp_size = vllm_config.parallel_config.data_parallel_size
        actor_class = (
            DPMoEEngineCoreActor
            if dp_size > 1 and vllm_config.model_config.is_moe
            else EngineCoreActor
        )

        # [CN] 本地 / 远端 actor 分开保存：只有本地的能走共享内存等快速通道。
        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []

        # [CN] 收集需要透传给 actor 的环境变量（排除 worker 专属的那些）。
        env_vars_list = get_env_vars_to_copy(
            destination=actor_class.__name__,
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
        )
        self.env_vars_dict = {
            name: os.environ[name] for name in env_vars_list if name in os.environ
        }
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)

        self.addresses = addresses
        self.executor_class = executor_class
        self.log_stats = log_stats
        local_engine_count = vllm_config.parallel_config.data_parallel_size_local
        world_size = vllm_config.parallel_config.world_size
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        # [CN] Ray 已初始化则复用，否则自己 init。
        if ray.is_initialized():
            logger.info("Ray is already initialized. Skipping Ray initialization.")
        else:
            ray.init()

        parallel_config = vllm_config.parallel_config
        # [CN] 弹性 EP 需要一个 TCP store 做 rank 间的协调与发现。
        if parallel_config.enable_elastic_ep:
            from vllm.distributed.utils import create_tcp_store

            ip = parallel_config.data_parallel_master_ip
            store = create_tcp_store(
                ip,
                0,
                is_master=True,
                world_size=-1,
                wait_for_workers=False,
            )
            parallel_config._coord_store_port = store.port
            self._coord_store = store

        # [CN] 调用方可以传入预先创建好的 placement groups（例如复用上次的结果）。
        if placement_groups is not None:
            assert local_dp_ranks is not None, (
                "local_dp_ranks must be provided if placement_groups is provided"
            )
            assert len(placement_groups) == len(local_dp_ranks), (
                "placement_groups and local_dp_ranks must have the same length"
            )
            logger.info("Using provided placement groups")
            # TODO(rui): validate passed-in placement groups
            self.created_placement_groups = []
        else:
            # [CN] 否则自己按 DP 拓扑创建。
            placement_groups, local_dp_ranks = (
                CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            )
            self.created_placement_groups = placement_groups
        # [CN] 每个 DP rank 一个 placement group，数量必须对齐。
        assert len(placement_groups) == dp_size, (
            "Number of placement groups must match data parallel size"
        )

        self.placement_group_is_local = []
        refs = []
        # [CN] 逐 rank：深拷贝配置 → 打身份后缀 → 绑定 placement group → 起 actor。
        for index, local_index, pg in zip(
            range(dp_size), local_dp_ranks, placement_groups
        ):
            # [CN] 深拷贝是必须的：config 要被逐 rank 修改，共享对象会互相污染。
            dp_vllm_config = copy.deepcopy(vllm_config)
            if dp_size > 1:
                _apply_dp_identity_suffix(dp_vllm_config, index)
            dp_vllm_config.parallel_config.placement_group = pg
            # [CN] index < local_engine_count 的 rank 与当前前端同机。
            local_client = index < local_engine_count

            # [CN] Ray + XPU 的已知问题：dpctl 过早初始化 GPU runtime，
            #      导致在 actor 的 __init__ 里设 env var 已经来不及影响设备选择。
            #      只能把设备变量塞进 runtime_env，让 Ray 在**启动前**就设好。
            # Ray XPU known issue: dpctl initializes the GPU runtime early, so
            # setting device env vars in Ray actor's initialization method
            # will not affect device selection. See:
            # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
            if current_platform.is_xpu():
                device_evar = current_platform.device_control_env_var
                physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
                    device_evar, local_index, world_size
                )
                actor_env_vars = self.env_vars_dict.copy()
                actor_env_vars[device_evar] = ",".join(str(d) for d in physical_gpu_ids)
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)

            # [CN] 把 actor 调度到 placement group 的第 world_size 个 bundle 上 ——
            #      也就是那个 CPU-only 的控制 bundle（GPU bundle 占前 world_size 个）。
            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    local_client=local_client,
                    addresses=addresses,
                    dp_rank=index,
                    local_dp_rank=local_index,
                )
            )
            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.placement_group_is_local.append(local_client)
            # [CN] 先等所有 actor 完成 __init__（模型加载、显存分配都在这一步）。
            refs.append(actor.wait_for_init.remote())

        ray.get(refs)
        self.run_refs = []
        self.actor_run_ref_dict = dict()
        # [CN] 再让所有 actor 开始 run 主循环，并保存 ref 用于后续探活。
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ref = actor.run.remote()
            self.run_refs.append(ref)
            self.actor_run_ref_dict[actor] = ref

    # [CN] 为 DP 创建 placement groups：这是 Ray 路径下最复杂的一段。
    @staticmethod
    def create_dp_placement_groups(
        vllm_config: VllmConfig,
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Create placement groups for data parallel.
        """

        import ray
        from ray._private.state import available_resources_per_node

        logger.info("Creating placement groups for data parallel")
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local

        # [CN] 查询集群里每个节点的可用资源。
        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        placement_groups: list[PlacementGroup] = []
        local_dp_ranks: list[int] = []

        # [CN] 节点排序：把 DP master 节点排在最前，保证它优先承载本地 rank。
        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(
            available_resources.values(), key=lambda x: dp_master_ip_key not in x
        )
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], (
            f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        )

        # [CN] 可选：通过环境变量把 DP 限制在调用方指定的节点集合内。
        # optionally restrict DP placement to a caller-provided node set.
        requested_node_ips = {
            ip.strip()
            for ip in envs.VLLM_RAY_DP_PLACEMENT_NODE_IPS.split(",")
            if ip.strip()
        }
        if requested_node_ips:
            allowed_node_ips = set(requested_node_ips)
            # The master node must host the local ranks, so it has to be allowed.
            # [CN] master 节点必须承载 local rank，所以它一定要在允许列表里。
            if dp_master_ip not in allowed_node_ips:
                allowed_node_ips.add(dp_master_ip)
            filtered_nodes = [
                node_resources
                for node_resources in nodes
                if _node_ip_from_resources(node_resources) in allowed_node_ips
            ]
            logger.info(
                "VLLM_RAY_DP_PLACEMENT_NODE_IPS set; restricting DP placement "
                "from %d to %d node(s): %s",
                len(nodes),
                len(filtered_nodes),
                sorted(allowed_node_ips),
            )
            nodes = filtered_nodes

        # [CN] 各节点的 GPU 数量。
        device_str = current_platform.ray_device_key
        n_node_devices: list[int] = [
            int(node_resources[device_str])
            for node_resources in nodes
            if device_str in node_resources
        ]
        assert n_node_devices, f"No {device_str} found in Ray cluster."
        max_device_per_node = max(n_node_devices)

        # [CN] 打包策略：strict（每节点必须放满 world_size 整数倍）/
        #      fill（有多少用多少）/ span（允许一个 DP rank 跨多个节点）。
        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
        _supported_pack_strategies = ("strict", "fill", "span")
        if pack_strategy not in _supported_pack_strategies:
            raise ValueError(
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                f"to one of {_supported_pack_strategies}"
            )

        # [CN] DeepEP 的 all-to-all 要求 EP rank [0,7]（以及 [8,15] 等）
        #      必须同节点，而 fill 策略不保证这点，所以直接拒绝。
        all2all_backend = vllm_config.parallel_config.all2all_backend
        if pack_strategy == "fill" and (
            all2all_backend == "deepep_high_throughput"
            or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                "does not guarantee that. "
                "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        # [CN] strict / fill 用 STRICT_PACK：所有 bundle 必须挤在同一节点。
        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            # [CN] span 用 PACK：允许跨节点，此时要求集群是同构的，
            #      且 world_size 必须是单节点卡数的整数倍。
            placement_strategy = "PACK"
            assert world_size > max_device_per_node, (
                f"World size {world_size} is smaller than the "
                "maximum number of devices per node "
                f"{max_device_per_node}. Make sure to set "
                "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
            )

            # if we need multiple nodes per dp group, we require for now that
            # available nodes are homogeneous
            assert set(n_node_devices) == {max_device_per_node}, (
                f"Nodes are not homogeneous, {nodes}"
            )
            assert world_size % max_device_per_node == 0, (
                f"For multi-node data parallel groups, world_size ({world_size}) must "
                f"be a multiple of number of devices per node ({max_device_per_node})."
            )
            assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (
                f"Not enough total available nodes ({len(n_node_devices)}) "
                f"and devices per node ({max_device_per_node}) "
                f"to satisfy required world size {world_size} and data parallel size "
                f"{dp_size}"
            )
            assert dp_size_local == 1, (
                f"data-parallel-size-local {dp_size_local} should be set as the "
                "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                "The actual data-parallel-size-local will be auto determined."
            )

        # [CN] 开始逐节点分配 bundle。
        # bundles collected for a single DP rank from multiple nodes,
        # for "span" pack strategy
        collected_bundles = []
        # [CN] 遍历节点，为每个节点算出能放几个 DP rank。
        for node_resources in nodes:
            node_ip = _node_ip_from_resources(node_resources)
            assert node_ip is not None, (
                f"No node IP key found in node resources: {node_resources}"
            )

            n_device_on_node = int(node_resources.get(device_str, 0))
            # [CN] span 模式下按整节点收集（下面会凑够 world_size 个再建组）。
            if pack_strategy == "span" and n_device_on_node != 0:
                # Strictly speaking,
                # dp_size_available = n_device_on_node / world_size
                # and is a fraction, but we use 1 for easier processing
                dp_size_available = 1
            else:
                dp_size_available = n_device_on_node // world_size

            # [CN] master 节点要放 dp_size_local 个本地 rank，放不下就直接报错。
            if node_ip == dp_master_ip:
                if dp_size_available < dp_size_local:
                    raise ValueError(
                        f"Not enough resources to allocate {dp_size_local} DP ranks "
                        f"on DP master node {dp_master_ip}, possible to fit "
                        f"{dp_size_available} DP ranks."
                    )
                dp_size_to_allocate = dp_size_local
            # [CN] strict 模式下非 master 节点放不下就跳过，继续找下一个节点。
            elif pack_strategy == "strict":
                if dp_size_available < dp_size_local:
                    logger.info(
                        "Skipping node %s as %s DP ranks could not fit, "
                        "possible to fit %s DP ranks",
                        node_ip,
                        dp_size_local,
                        dp_size_available,
                    )
                    continue
                dp_size_to_allocate = dp_size_local
            # [CN] fill / span 模式：能吃多少吃多少。
            else:
                # for "pack_strategy" in "fill" and "span"
                # we always take everything that's available
                dp_size_to_allocate = dp_size_available

            # [CN] 每个 DP rank 一组 bundle：world_size 个 GPU bundle + 1 个控制 bundle。
            for i in range(dp_size_to_allocate):
                device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]
                # [CN] span 特殊处理：把整节点的卡累积起来，凑够 world_size 才建组。
                if pack_strategy == "span":
                    collected_bundles += device_bundle * n_device_on_node
                    assert len(collected_bundles) <= world_size, (
                        "collected_bundles should be <= world_size, "
                        f"but got {len(collected_bundles)=} and {world_size=}"
                    )

                    # we only create a placement group if we collected enough devices
                    if len(collected_bundles) < world_size:
                        continue

                    # [CN] 控制 bundle 钉在**第一个** bundle 所在节点，保证 actor 不飘走。
                    control_node_ip = _get_bundle_node_ip(collected_bundles[0])
                    bundles = collected_bundles + [
                        _make_control_bundle(control_node_ip)
                    ]
                    collected_bundles = []
                # [CN] STRICT_PACK 本身已保证同节点，这里仍然显式写亲和：
                #      一是与 span 路径保持一致，二是将来换调度策略时不至于悄悄坏掉。
                else:
                    # STRICT_PACK already keeps every bundle in the placement
                    # group on one node, so the explicit node affinity on the
                    # control bundle is redundant for correctness here. Keep it
                    # anyway for consistency with the span path and to preserve
                    # intent if this scheduling strategy changes later.
                    bundles = device_bundle * world_size + [
                        _make_control_bundle(node_ip)
                    ]

                # [CN] 真正创建 placement group。注意此时只是**申请**，未必立即就绪。
                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy=placement_strategy,
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
                # [CN] 够了就收工。
                if len(placement_groups) == dp_size:
                    break

            if len(placement_groups) == dp_size:
                break

        # [CN] 资源不够 → 明确报错，把可用资源一并打出来方便排查。
        if len(placement_groups) < dp_size:
            raise ValueError(
                f"Not enough resources to allocate {dp_size} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            f"local_dp_ranks length {len(local_dp_ranks)} does not match "
            f"expected {dp_size}"
        )
        return placement_groups, local_dp_ranks

    # [CN] 弹性扩容时**增量**创建 placement groups（只建新增的那几个）。
    @staticmethod
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Add placement groups for new data parallel size.
        """
        import ray
        from ray._private.state import (
            available_resources_per_node,
            total_resources_per_node,
        )
        from ray.util.state import list_nodes

        # [CN] 只需要新增的部分。
        old_dp_size = old_vllm_config.parallel_config.data_parallel_size
        num_pg_to_create = new_data_parallel_size - old_dp_size

        if num_pg_to_create <= 0:
            return [], []

        dp_master_ip = old_vllm_config.parallel_config.data_parallel_master_ip
        world_size = old_vllm_config.parallel_config.world_size

        # [CN] 按「是否 master 节点」排序，优先把新 rank 放在 master 上。
        nodes = list_nodes()
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)
        assert nodes[0].node_ip == dp_master_ip, "The first node must be the head node"
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            "There can only be one head node"
        )

        available_resources = available_resources_per_node()
        total_resources = total_resources_per_node()

        placement_groups = []
        local_dp_ranks = []
        num_pg_created = 0

        device_str = current_platform.ray_device_key
        for node in nodes:
            if num_pg_created >= num_pg_to_create:
                break

            node_ip = node.node_ip
            node_id = node.node_id
            # [CN] 没有 GPU 的节点直接跳过。
            if device_str not in available_resources[node_id]:
                continue
            available_gpus = int(available_resources[node_id][device_str])

            # Get total GPUs on this node from the node's resources
            # Ray stores node resources with node ID as key
            # [CN] 用「总卡数 - 可用卡数」推出本节点已经跑了多少个引擎，
            #      新 rank 的 local_rank 要接着往后排。
            total_gpus = int(total_resources[node_id][device_str])

            # Calculate used GPUs and used engines on this node
            used_gpus = max(0, total_gpus - available_gpus)
            used_engines_on_node = used_gpus // world_size

            # Calculate how many new engines this node can accommodate
            available_engine_count = available_gpus // world_size

            # Create placement groups for new engines on this node
            # [CN] 逐个新建，直到凑够数量。
            for i in range(available_engine_count):
                if num_pg_created >= num_pg_to_create:
                    break

                rank = old_dp_size + num_pg_created

                # Create bundles with node constraint for master node
                # [CN] master 节点的 bundle 带 node 亲和；其它节点不限制。
                if node_ip == dp_master_ip:
                    bundles = [
                        {device_str: 1.0, "node:" + dp_master_ip: 0.001}
                    ] * world_size + [{"CPU": 1.0}]
                else:
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]

                # [CN] 弹性扩容一律用 STRICT_PACK，保证一个 rank 不跨节点。
                pg = ray.util.placement_group(
                    name=f"dp_rank_{rank}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)

                # Local rank starts from the number of engines already used
                # on this node
                # [CN] local_rank 从本节点已占用的引擎数往后接。
                local_rank = used_engines_on_node + i
                local_dp_ranks.append(local_rank)
                num_pg_created += 1

        return placement_groups, local_dp_ranks

    # [CN] 弹性 EP 扩容：新建 placement groups + 拉起新的 engine actor。
    def scale_up_elastic_ep(
        self,
        cur_vllm_config: VllmConfig,
        new_data_parallel_size: int,
        num_redundant_experts: int,
    ) -> None:
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        actor_class = (
            DPMoEEngineCoreActor
            if cur_vllm_config.model_config.is_moe
            else EngineCoreActor
        )

        # [CN] 当前实际的 DP 规模 = 本地 + 远端 actor 总数。
        cur_data_parallel_size = len(self.local_engine_actors) + len(
            self.remote_engine_actors
        )

        assert new_data_parallel_size > cur_data_parallel_size, (
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up"
        )

        placement_groups, local_dp_ranks = self.add_dp_placement_groups(
            cur_vllm_config, new_data_parallel_size
        )

        world_size = cur_vllm_config.parallel_config.world_size
        dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip
        new_local_engines = 0

        # [CN] 新起的 actor 需要知道自己是「扩容进来的」，靠这个环境变量传递。
        runtime_env = RuntimeEnv(
            env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}
        )
        for i, (pg, local_rank) in enumerate(zip(placement_groups, local_dp_ranks)):
            rank = cur_data_parallel_size + i
            # [CN] 同样要深拷贝 + 打身份后缀 + 更新目标 DP 规模。
            dp_vllm_config = copy.deepcopy(cur_vllm_config)
            if new_data_parallel_size > 1:
                _apply_dp_identity_suffix(dp_vllm_config, rank)
            dp_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            dp_vllm_config.parallel_config.eplb_config.num_redundant_experts = (
                num_redundant_experts
            )
            dp_vllm_config.parallel_config.placement_group = pg

            # Check if this placement group is on the head node
            # [CN] 通过 bundle 里是否带 master 节点亲和，判断这个 PG 是否在本地。
            local_client = any(
                bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs
            )

            if local_client:
                new_local_engines += 1
                # Update data_parallel_size_local
                dp_vllm_config.parallel_config.data_parallel_size_local = (
                    cur_vllm_config.parallel_config.data_parallel_size_local
                    + new_local_engines
                )

            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=self.executor_class,
                    log_stats=self.log_stats,
                    local_client=local_client,
                    addresses=self.addresses,
                    dp_rank=rank,
                    local_dp_rank=local_rank,
                )
            )

            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.created_placement_groups.append(pg)
            self.placement_group_is_local.append(local_client)

        # [CN] 只对**本次新增**的 actor 等 init，避免阻塞已有引擎。
        actors = (
            self.local_engine_actors[-new_local_engines:]
            if new_local_engines > 0
            else []
        ) + self.remote_engine_actors[-(len(placement_groups) - new_local_engines) :]

        ray.get([actor.wait_for_init.remote() for actor in actors])
        for actor in actors:
            ref = actor.run.remote()
            self.run_refs.append(ref)
            self.actor_run_ref_dict[actor] = ref

    # [CN] 弹性缩容：从尾部开始摘掉 actor 并释放对应的 placement group。
    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        import ray

        assert cur_data_parallel_size > new_data_parallel_size, (
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down"
        )
        # [CN] 后进先出：最后加的最先删。
        for _ in range(cur_data_parallel_size - new_data_parallel_size):
            pg = self.created_placement_groups.pop()
            is_local = self.placement_group_is_local.pop()
            if is_local:
                self.local_engine_actors.pop()
            else:
                self.remote_engine_actors.pop()
            ray.util.remove_placement_group(pg)

    # [CN] 缩容后把这些 actor 的 run ref 从探活列表里移除，
    #      否则监控线程会把「正常摘除」误判成异常退出。
    def remove_run_refs_for_scale_down(self, removed_dp_size: int) -> None:
        if removed_dp_size <= 0:
            return
        flags = self.placement_group_is_local[-removed_dp_size:]
        li = len(self.local_engine_actors) - 1
        ri = len(self.remote_engine_actors) - 1
        for is_local in reversed(flags):
            if is_local:
                actor = self.local_engine_actors[li]
                li -= 1
            else:
                actor = self.remote_engine_actors[ri]
                ri -= 1
            ref = self.actor_run_ref_dict.pop(actor)
            self.run_refs.remove(ref)

    def get_run_refs(self):
        return self.run_refs

    # [CN] Ray 路径的探活：周期性 wait 所有 actor 的 run ref。
    def monitor_engine_liveness(self) -> None:
        import ray

        while not self.manager_stopped.is_set():
            actor_run_refs = list(self.get_run_refs())
            if not actor_run_refs:
                logger.info(
                    "There are no actors to monitor currently. "
                    "The monitoring function is about to terminate."
                )
                break
            # [CN] 最多等 5 秒就返回，以便响应停止标志。
            actor_done_refs, _ = ray.wait(actor_run_refs, timeout=5)
            unexpected_failure = False
            for actor_ref in actor_done_refs:
                if self.manager_stopped.is_set():
                    break
                # [CN] ref 可能已被缩容移走，跳过即可（不是故障）。
                if actor_ref not in self.get_run_refs():
                    # The run refs may have been updated by elastic scale-down.
                    continue
                try:
                    ray.get(actor_ref)
                # [CN] RayActorError 表示 actor 真的挂了。
                except ray.exceptions.RayActorError:
                    self.failed_proc_name = f"Actor {actor_ref}"
                    unexpected_failure = True

            if unexpected_failure:
                break

        self.shutdown()

    # [CN] 关停：杀掉所有 actor、释放所有 placement group。
    def shutdown(self, timeout: float | None = None) -> None:
        import ray

        self.manager_stopped.set()
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ray.kill(actor)
        for pg in self.created_placement_groups:
            ray.util.remove_placement_group(pg)


# [CN] 分配引擎与前端通信用的 ZMQ 地址。
#      核心技巧：TCP 地址先写成 host:0，由真正 bind 的一方在 bind 之后
#      用 getsockopt(zmq.LAST_ENDPOINT) 取回内核分配的真实端口并回填。
#      这样才能既避免端口冲突，又不用提前确定端口。
def get_engine_zmq_addresses(
    vllm_config: VllmConfig,
    num_api_servers: int = 1,
    *,
    defer_api_server_ports: bool = True,
) -> EngineZmqAddresses:
    """Allocate ZMQ addresses for engine-client communication.

    By default each TCP address is a ``tcp://host:0`` placeholder; the
    consumer (API-server child or single-process ``MPClient``) binds, then
    recovers the kernel-assigned port via ``getsockopt(zmq.LAST_ENDPOINT)``
    and writes it back into ``addresses`` before the engine handshake.

    Set ``defer_api_server_ports=False`` only when the consumer cannot
    report a bound port back (e.g. the Rust front-end). IPC paths are
    unaffected."""
    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_size = parallel_config.data_parallel_size
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    # [CN] 离线模式下每个 DP rank 有一个独立 LLM 实例，各自一个引擎。
    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/features/data_parallel/data_parallel_offline.py.
    offline_mode = local_start_index is not None

    # [CN] 只有本前端只跟同机引擎通信时，才能用 IPC（更快，且不用占端口）。
    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )
    # NOTE(yongji): handling scaling from intra-node to inter-node
    # [CN] 弹性 EP 可能从节点内扩到跨节点，所以不能假定只走本地。
    if parallel_config.enable_elastic_ep:
        client_local_only = False

    # [CN] 生成单个地址：能走本地就用 IPC 路径，否则用 TCP。
    def _addr() -> str:
        if client_local_only:
            return get_open_zmq_ipc_path()
        # [CN] defer=False 时才**预先**占一个确定端口（例如 Rust 前端无法回填）。
        return get_tcp_uri(host, 0 if defer_api_server_ports else get_open_port())

    return EngineZmqAddresses(
        inputs=[_addr() for _ in range(num_api_servers)],
        outputs=[_addr() for _ in range(num_api_servers)],
    )


FrontendProcess = BaseProcess | _SubprocessWrapper


# [CN] 启动产出的资源集合 + 启动屏障。
@dataclass
class CoreEngineLaunch:
    """Resources and startup barrier for launched engine processes."""

    # [CN] 引擎管理器：本地多进程 / Ray actor 二选一。
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None
    coordinator: DPCoordinator | None
    addresses: EngineZmqAddresses
    tensor_queue: Queue | None
    # [CN] 启动期间要一并监视的前端进程（它们挂了也要报错，而不是干等）。
    # Frontend processes to watch during engine startup; may be assigned by
    # the caller before the startup barrier runs on context manager exit.
    watched_frontend_processes: Sequence[FrontendProcess] = ()


# [CN] 启动引擎与 DP 协调器。**contextmanager**。
#      进入时：起协调器 → 起引擎进程 → yield（此时引擎还没就绪）。
#      退出时：才执行 wait_for_engine_startup 等所有引擎握手完成。
#      这个设计让调用方能在 yield 之后、等待之前插入自己的初始化逻辑。
@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
) -> Iterator[CoreEngineLaunch]:
    """Launch engine and DP coordinator processes as needed."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    # [CN] 多模态张量的共享内存队列（只支持 DP=1，见 tensor_ipc.py）。
    # Create a single tensor IPC queue for sharing multimodal tensors between
    # API servers and engine core. Returns a single queue since we only support
    # DP=1 for this data flow.
    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = get_mp_context().Queue()

    # [CN] 只有在线 DP 模式、且是 rank 0 时才起协调器。
    #      它负责两件事：① 负载统计发布（供 LB 决策）；
    #      ② MoE 场景下的 wave 同步（见 coordinator.py）。
    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    # [CN] Ray 后端：起 actor 管理器后直接 yield 并返回，
    #      Ray 路径不走下面的 ZMQ 握手流程。
    if parallel_config.data_parallel_backend == "ray":
        logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield CoreEngineLaunch(
            engine_actor_manager, coordinator, addresses, tensor_queue
        )
        return

    # [CN] 确定「本进程要跟哪些引擎握手」：
    #      离线模式只管自己；rank 0 管**全部**（它持有协调器）；
    #      rank > 0 只管自己名下的本地引擎。
    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        # Rank > 0 handshakes with just the local cores it is managing.
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # [CN] 外部 DP LB 模式下，rank>0 要同时跟本地前端和 rank0 前端握手，
    #      所以不能限定为「仅本地」。
    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only,
        host,
        parallel_config.data_parallel_rpc_port,
    )

    # [CN] rank>0 时另开一个本地握手地址，让本地前端能连上来。
    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    # [CN] 建 ROUTER 套接字做握手。用 ROUTER 是因为要同时跟多个引擎对话，
    #      靠 identity 区分彼此。
    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        # Start local engines.
        # [CN] 启动本地引擎进程。
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        launch = CoreEngineLaunch(
            local_engine_manager, coordinator, addresses, tensor_queue
        )
        # [CN] yield 之后（即 with 块结束时）才真正等待所有引擎就绪。
        yield launch
        wait_for_engine_startup(
            handshake_socket,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            launch,
        )


# [CN] 等待所有引擎进程完成握手：收 HELLO → 回元数据 → 收 READY。
def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    coordinated_dp: bool,
    cache_config: CacheConfig,
    launch: CoreEngineLaunch,
):
    # Wait for engine core process(es) to send ready messages.
    # [CN] 本地 / 远端分开计数，因为它们的就绪要求可能不同。
    local_count = parallel_config.data_parallel_size_local
    remote_count = len(core_engines) - local_count
    # [local, remote] counts
    # [CN] 两组计数器：[local, remote]，分别是「已连接待启动」和「已启动待就绪」。
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    # [CN] 非 hybrid / 非 external LB 模式下，远端引擎必须是 headless 的。
    remote_should_be_headless = (
        not parallel_config.data_parallel_hybrid_lb
        and not parallel_config.data_parallel_external_lb
    )

    # 1. Engine processes
    # [CN] 把三类 fd 都注册进 poller：引擎进程、协调器进程、被监视的前端进程。
    #      任何一类提前退出都能立刻被发现，不会变成「干等到超时」。
    if isinstance(launch.engine_manager, CoreEngineProcManager):
        for sentinel in launch.engine_manager.sentinels():
            poller.register(sentinel, zmq.POLLIN)
    # 2. DP Coordinator process, if present
    coord_process = launch.coordinator.proc if launch.coordinator else None
    if coord_process is not None:
        poller.register(coord_process.sentinel, zmq.POLLIN)
    # 3. Watched frontend processes, if any
    frontend_process_by_fd: dict[int, FrontendProcess] = {}
    for proc in launch.watched_frontend_processes:
        fd = proc.sentinel if isinstance(proc.sentinel, int) else proc.sentinel.fileno()
        frontend_process_by_fd[fd] = proc
        poller.register(fd, zmq.POLLIN)

    # [CN] 主循环：还有引擎没连上或没就绪就继续等。
    while any(conn_pending) or any(start_pending):
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        # [CN] 超时没有事件：打日志继续等（启动期模型加载可能很慢）。
        if not events:
            if any(conn_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue
        # [CN] 有非握手套接字可读 == 某个进程退出了。
        #      这里把所有已结束进程的信息收集起来，拼成可读的错误抛出。
        if len(events) > 1 or events[0][0] != handshake_socket:
            # One of the local core, coordinator, or watched frontend processes exited.
            if isinstance(launch.engine_manager, CoreEngineProcManager):
                finished = launch.engine_manager.finished_procs()
            else:
                finished = {}
            if coord_process is not None and coord_process.exitcode is not None:
                finished[coord_process.name] = coord_process.exitcode
            failed_frontend_procs = {
                proc.name: proc.exitcode
                for fd, proc in frontend_process_by_fd.items()
                if proc.exitcode is not None
                or any(event_fd == fd for event_fd, _ in events)
            }
            # [CN] 只有前端进程挂了、引擎进程没挂：说明是前端自己的问题。
            if failed_frontend_procs and not finished:
                raise RuntimeError(
                    "Frontend process failed during engine core initialization. "
                    "See root cause above. "
                    f"Failed frontend proc(s): {failed_frontend_procs}"
                )
            raise RuntimeError(
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"
                + (
                    f", failed frontend proc(s): {failed_frontend_procs}"
                    if failed_frontend_procs
                    else ""
                )
            )

        # [CN] 收到握手消息：两帧（identity + 载荷）。
        # Receive HELLO and READY messages from the input socket.
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        # [CN] 从 identity 反解出 rank，据此找到对应的 CoreEngine 记录。
        eng_index = int.from_bytes(eng_identity, "little")
        engine = next((e for e in core_engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        msg = msgspec.msgpack.decode(ready_msg_bytes)
        # [CN] 解析状态、是否本地、是否 headless。
        status, local, headless = msg["status"], msg["local"], msg["headless"]
        # [CN] 引擎自报的「本地/远端」必须与我们预期一致，否则拓扑有问题。
        if local != engine.local:
            raise RuntimeError(
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # [CN] 远端引擎是否 headless 必须匹配当前 LB 模式，防止配置错误。
        # Remote engines must be headless iff we aren't in hybrid dp lb mode.
        if not local and headless != remote_should_be_headless:
            if headless:
                raise RuntimeError(
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:
                raise RuntimeError(
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        # [CN] HELLO → 回握手元数据（把地址表和 DP 配置交给引擎）。
        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # Send init message with DP config info.
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=launch.addresses,
                    # [CN] 只有需要协调的 DP 场景才下发并行配置；否则给空 dict 省体积。
                    parallel_config={
                        k: getattr(parallel_config, k)
                        for k in (
                            "data_parallel_master_ip",
                            "data_parallel_master_port",
                            "_data_parallel_master_port_list",
                            "data_parallel_size",
                        )
                    }
                    if coordinated_dp
                    else {},
                )
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            # [CN] 连接计数减一、启动计数加一，进入 CONNECTED 状态。
            conn_pending[0 if local else 1] -= 1
            start_pending[0 if local else 1] += 1
            engine.state = CoreEngineState.CONNECTED
        # [CN] READY → 校验配置一致性后标记就绪。
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # Validate config hash consistency across DP workers for MoE models.
            # [CN] 关键校验：**所有 DP worker 的配置必须完全一致**。
            #      只校验影响集合通信的那些参数（如 enable_eplb），
            #      不一致会导致 all-reduce 挂死或结果错乱，属于必须提前拦住的错误。
            if coordinated_dp:
                worker_config_hash = msg.get("parallel_config_hash")
                expected_hash = parallel_config.compute_hash()
                if worker_config_hash != expected_hash:
                    raise RuntimeError(
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            start_pending[0 if local else 1] -= 1
            engine.state = CoreEngineState.READY
        # [CN] 其它状态组合都是协议错误，直接抛异常而不是静默忽略。
        else:
            raise RuntimeError(
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        logger.debug(
            "%s from %s core engine process %s.",
            status,
            "local" if local else "remote",
            eng_index,
        )
