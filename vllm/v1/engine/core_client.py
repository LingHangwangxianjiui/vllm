# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# 本文件职责：实现"前端进程 ↔ EngineCore 进程"之间的**通信客户端**。
#   如果说 v1/engine/__init__.py 定义的是"报文格式"，那本文件定义的就是
#   "怎么把这些报文送过去、收回来"。它是 V1 里**进程边界**的全部实现。
#
# 在系统链路中的位置：
#   前端进程                                    │  EngineCore 进程
#   --------------------------------------------┼-------------------------------
#   LLMEngine  ──> SyncMPClient   ──┐           │
#   AsyncLLM   ──> AsyncMPClient  ──┤ ZMQ 消息  │  EngineCoreProc 主循环
#   LLM(进程内) ──> InprocClient ───┘           │   （v1/engine/core.py）
#                                   直接函数调用 │
#
# 三/四种客户端实现（本文件的主体）：
#   - InprocClient   ：EngineCore 就在本进程，**没有 IPC**，直接函数调用。
#                      只用于进程内模式（LLM 默认在 V1 下也是多进程，所以
#                      它主要用于测试与 v0 兼容路径）。
#   - SyncMPClient   ：ZMQ + **后台线程**收输出 + 阻塞 get_output()。给 LLM 用。
#   - AsyncMPClient  ：ZMQ + **asyncio 任务**收输出 + await get_output_async()。
#                      给 AsyncLLM（在线服务）用。
#   - DPAsyncMPClient / DPLBAsyncMPClient：数据并行下的两种变体
#                      （每个 rank 一个引擎 / 客户端自己做负载均衡）。
#
# 核心内容速查：
#   - EngineCoreClient（ABC）    : 全部客户端的接口，含 make_client 工厂
#   - InprocClient               : 进程内直连
#   - BackgroundResources        : ZMQ socket / 子进程 / 后台任务的"回收袋"
#   - MPClient                   : 多进程客户端**基类**，负责建 ZMQ、握手、拉引擎
#   - SyncMPClient               : 同步版（线程 + queue.Queue）
#   - AsyncMPClient              : 异步版（asyncio.Task + asyncio.Queue）
#   - _process_utility_output    : 控制类 RPC 的"应答归位"逻辑
#
# 阅读提示（几个容易踩的点）：
#   1. **为什么要绕开 self**：SyncMPClient 的收包线程、AsyncMPClient 的收包协程、
#      MPClient 的引擎监控线程，都**刻意只捕获局部变量**（socket、decoder、queue），
#      绝不捕获 self。原因与 async_llm.py 的 output_handler 一样：
#      后台线程/任务 -> self 的强引用会让客户端永远无法被 GC，
#      进而 EngineCore 子进程也退不干净。看到 `ctx = self.ctx` 这种"多余"的
#      局部变量赋值，答案都在这里。
#   2. **BackgroundResources + weakref.finalize**：把所有需要清理的资源集中在一个
#      独立 dataclass 里，再用 weakref.finalize 注册成终结器。
#      这样即使 __init__ 中途抛异常（模型加载失败很常见），已经建好的 socket 和
#      子进程也会被回收（见 __init__ 里的 try/finally + success 标志）。
#   3. **握手协议**：ROUTER 输入 socket 上收到的第一帧是 EngineCore 的"身份"，
#      第二帧是 EngineCoreReadyResponse。客户端必须收齐所有 rank 的 ready 才能开工，
#      并把响应里的真实配置（max_model_len、num_gpu_blocks、block_size）**回写**
#      到前端的 VllmConfig —— 所以前端看到的配置在握手后可能变了。
#   4. **两类返回通道要分清**：
#      数据面输出 -> outputs_queue（由 get_output 消费）；
#      控制面应答 -> utility_results: dict[call_id, Future]（谁发起谁等）。
# ==============================================================================
import asyncio
import contextlib
import queue
import sys
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from threading import Thread
from typing import Any, TypeAlias, TypeVar

import msgspec
import msgspec.msgpack
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.envs import VLLM_ENGINE_READY_TIMEOUT_S
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.renderers import BaseRenderer
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.async_utils import in_loop
from vllm.utils.network_utils import (
    close_sockets,
    get_open_zmq_inproc_path,
    make_zmq_socket,
)
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    FT_STATUS_CALL_ID,
    EEPNotificationType,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
)
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.tensor_ipc import TensorIpcSender
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    get_engine_zmq_addresses,
    launch_core_engines,
)
from vllm.v1.executor import Executor
from vllm.v1.fault_tolerance.engine_core_sentinel import FT_UTILITY_METHOD
from vllm.v1.fault_tolerance.utils import (
    FaultToleranceRequest,
    FaultToleranceResult,
)
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

_R = TypeVar("_R")  # Return type for collective_rpc

EngineIdentity = bytes


class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)

    [CN] 这个 ABC 有个特点：**大部分方法不是 @abstractmethod 而是
         raise NotImplementedError**。这是刻意的 —— 抽象基类要求子类全部实现，
        而这里同步/异步客户端各自只需要实现一半（同步的不需要 *_async，
        异步的不需要同步版）。用 raise 可以让子类"按需实现"，
        真正强制实现的只有 shutdown 一个。
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        renderer: BaseRenderer | None = None,
    ) -> "EngineCoreClient":
        # renderer is passed through to the multiprocess clients, which start
        # the frontend MM warmup (renderer.start_mm_warmup_in_background) once
        # the engine-core processes have been forked, so warmup overlaps
        # engine-core load without a live thread at fork() time. In-process
        # clients do not take a renderer: the engine core is built in this
        # process, so there is no other process to overlap with and the MM
        # warmup runs synchronously inside renderer.warmup() instead.
        # TODO: support this for debugging purposes.
        # [CN] 唯一**不支持**的组合：异步 + 进程内。
        #      原因：进程内模式下模型推理就在本进程跑，一定会阻塞事件循环，
        #      "异步"就名不副实了。所以直接拒绝，而不是悄悄退化成同步。
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config,
                executor_class,
                log_stats,
                renderer=renderer,
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(
                vllm_config,
                executor_class,
                log_stats,
                renderer=renderer,
            )

        return InprocClient(vllm_config, executor_class, log_stats)

    @staticmethod
    @instrument(span_name="Overall Loading")
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        renderer: BaseRenderer | None = None,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        # [CN] DP 场景分两种负载均衡模式，选的客户端实现不同：
        #   - 外部 LB（data_parallel_external_lb）：由**外部**（如 router / k8s
        #     service）把请求分到某个前端进程，每个前端固定对接一个 DP rank，
        #     所以一个客户端只管一个引擎（DPAsyncMPClient）。
        #   - 内部 LB：前端进程自己持有到**所有** rank 的连接，并在客户端里
        #     按负载挑一个 rank 发（DPLBAsyncMPClient）。
        # 非 DP 就是普通的 AsyncMPClient。
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(
                    *client_args,
                    renderer=renderer,
                )
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(
                *client_args,
                renderer=renderer,
            )
        return AsyncMPClient(
            *client_args,
            renderer=renderer,
        )

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None: ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        raise NotImplementedError

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    def reset_encoder_cache(self) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    def set_weight_version(self, weight_version: str) -> None:
        raise NotImplementedError

    def get_weight_version(self) -> str:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    async def set_weight_version_async(self, weight_version: str) -> None:
        raise NotImplementedError

    async def get_weight_version_async(self) -> str:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        raise NotImplementedError

    async def commit_elastic_ep(self) -> None:
        raise NotImplementedError

    async def prepare_elastic_ep(self, new_data_parallel_size: int) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        raise NotImplementedError

    async def reset_mm_cache_async(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    async def reset_encoder_cache_async(self) -> None:
        raise NotImplementedError

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping_async(self) -> bool:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        raise NotImplementedError

    async def get_status(self):
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore

    [CN] 最简单的一种：EngineCore 对象就活在本进程里（self.engine_core），
        所有方法都是**直接转发**（一两层调用而已），没有任何序列化与 IPC。
        它的价值在于：调试时可以在一个进程里打断点看完整链路；
        单卡小模型跑离线批处理时也省掉跨进程开销。
        代价：模型推理会阻塞调用方，且无法与事件循环共存。
    """

    # Takes no renderer: this class has no _start_mm_warmup (only MPClient
    # does), so a renderer would never be used here. EngineCore is built in
    # this process and MM warmup runs synchronously inside renderer.warmup().
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        *,
        executor_fail_callback: Callable | None = None,
    ):
        self.engine_core = EngineCore(
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback=executor_fail_callback,
        )

    def get_output(self) -> EngineCoreOutputs:
        """[CN] 推进引擎一步并返回输出。

        两步是固定的组合拳：
          - step_fn()   ：调度 + 执行模型，返回 (输出, 本轮是否真的跑了模型)；
          - post_step() ：执行后的收尾（异步调度下的记账、DP 波次推进等）。
        返回 `outputs.get(0)` 是因为进程内模式下"引擎编号"恒为 0；
        如果引擎没产出任何东西（比如本轮被跳过），返回一个**空帧**而不是 None ——
        调用方（LLMEngine.step）可以无脑访问 .outputs / .scheduler_stats。
        """
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def add_request(self, request: EngineCoreRequest) -> None:
        # [CN] 分两步：preprocess（判断属于哪个 DP 波次、要不要唤醒引擎）
        #      + add_request（真正入队）。多进程客户端里这两步分别在
        #      前后端完成（preprocess 在前端，入队在引擎进程）。
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    def abort_requests(self, request_ids: list[str]) -> None:
        # [CN] 空列表不发：省一次跨进程调用；多进程客户端里这个判断同样存在，
        #      因为 abort 是同步阻塞 RPC，空调用白等一个 RTT。
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.engine_core.profile(is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # [CN] 进程内模式不支持 "wait" 模式：等待在途请求跑完需要有人**继续驱动**
        #      引擎（调 step），而进程内模式下没有后台循环去做这件事，
        #      一旦进入 wait 就永远等不到完成 —— 会挂死。所以直接报错。
        if mode == "wait":
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
        result = self.engine_core.sleep(level, mode)
        assert result is None

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def set_weight_version(self, weight_version: str) -> None:
        self.engine_core.set_weight_version(weight_version)

    def get_weight_version(self) -> str:
        return self.engine_core.get_weight_version()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def dp_engines_running(self) -> bool:
        """[CN] 进程内模式**永远是 False**：因为没有别的引擎进程，
        "其他 DP 引擎还在跑吗"这个问题不存在。"""
        return False


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""
    # [CN] 这是一个"资源回收袋"，把所有跨进程/跨线程的资源集中放一处：
    #   - ctx / 各种 socket：ZMQ
    #   - engine_manager ：EngineCore 子进程（或 Ray actor）管理器
    #   - coordinator    ：DP 协调器（也是一个独立进程）
    #   - output_queue_task / stats_update_task：后台 asyncio 任务
    #   - shutdown_path  ：给同步收包线程用的"退出信号"通道
    #   - engine_dead    ：**跨线程可见**的死亡标志
    #
    # 为什么单独抽成一个 dataclass（而不是放在 client 上）：
    # 为了让 **weakref.finalize** 能持有它 —— 终结器的回调不能引用被终结的对象
    # （否则对象永远不会被回收）。抽出来之后，清理逻辑可以在 client 已经"半死"
    # 的情况下独立执行。
    # engine_dead 放在这里而不是 client 上的原因同理：收包线程要能读到它，
    # 但**不能**因此持有 client 的强引用。

    ctx: zmq.Context
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None
    coordinator: DPCoordinator | None = None
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    first_req_send_socket: zmq.asyncio.Socket | None = None
    first_req_rcv_socket: zmq.asyncio.Socket | None = None
    stats_update_socket: zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None
    stats_update_task: asyncio.Task | None = None
    shutdown_path: str | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""

        # [CN] 清理顺序：先置 engine_dead（让后续操作快速失败，不再往死引擎发消息），
        #      再停子进程/协调器，最后关 socket 与取消任务。
        logger.debug_once("[shutdown] MPClient: background resource cleanup start")
        # [CN] 先立死亡标志：清理过程可能耗时（等子进程退出有超时），
        #      期间不该再有人往引擎发消息。
        self.engine_dead = True
        if self.engine_manager is not None:
            self.engine_manager.shutdown(
                timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
            )
        if self.coordinator is not None:
            self.coordinator.shutdown()

        if isinstance(self.output_socket, zmq.asyncio.Socket):
            # Async case.
            loop = self.output_queue_task._loop if self.output_queue_task else None

            sockets = (
                self.output_socket,
                self.input_socket,
                self.first_req_send_socket,
                self.first_req_rcv_socket,
                self.stats_update_socket,
            )

            tasks = (self.output_queue_task, self.stats_update_task)

            def close_sockets_and_tasks():
                close_sockets(sockets)
                for task in tasks:
                    if task is not None and not task.done():
                        with contextlib.suppress(Exception):
                            task.cancel()

            # [CN] ZMQ socket 与 asyncio task 都**绑定在创建它们的事件循环**上，
            #      从别的线程直接关会出未定义行为。所以这里判断三种情形：
            #        - 当前就在这个 loop 里：直接关；
            #        - 在别的线程、loop 还活着：call_soon_threadsafe 丢回去执行；
            #        - loop 已经关了：只能硬关 socket，并手动 del 掉 task 引用
            #          （task 已经没法取消了，解开引用让它被 GC）。
            if loop is not None:
                if in_loop(loop):
                    close_sockets_and_tasks()
                elif not loop.is_closed():
                    loop.call_soon_threadsafe(close_sockets_and_tasks)
            else:
                # Loop has been closed, try to clean up directly.
                del tasks
                del close_sockets_and_tasks
                close_sockets(sockets)
                del self.output_queue_task
                del self.stats_update_task
        else:
            # Sync case.

            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            close_sockets((self.output_socket, self.input_socket))

            if self.shutdown_path is not None:
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                # [CN] 用一对 PAIR socket 当"唤醒信号"：收包线程正阻塞在
                #      poller.poll() 上，必须有东西把它唤醒才能退出。
                #      不能直接从外部关 socket —— 那会让正在 poll 的线程炸掉。
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(self.shutdown_path)
                    # Send shutdown signal.
                    shutdown_sender.send(b"")

        logger.debug_once("[shutdown] MPClient: background resource cleanup complete")

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        """[CN] 收包时先检查是不是"引擎死亡通知"：
        EngineCore 进程在退出前会发一个固定内容的哨兵帧（ENGINE_CORE_DEAD）。
        单帧是这个特征（正常输出是多帧/有结构的），据此判定并置位 + 抛异常。
        """
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


@dataclass
class ElasticScalingCache:
    """[CN] 弹性扩缩容期间暂存的状态：
    existing_core_engines  : 扩容前已经存在的引擎身份列表；
    num_new_core_engines   : 本次要新增多少个引擎；
    pending_notifications  : 还在等待哪些引擎回哪种通知（如 RECONFIGURE_FINISHED）。
    之所以要"暂存"：扩缩容是多步异步流程，中途引擎身份会变，
    必须记住"谁还没回话"，否则无法判断流程是否完成。
    """

    existing_core_engines: list[EngineIdentity]
    num_new_core_engines: int
    pending_notifications: dict[EEPNotificationType, set[int]]


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage

    [CN] 注意"谁来驱动引擎"这件事在两种模式下完全不同：
      多进程模式下，EngineCore 有自己的**忙循环**（busy loop，见 core.py 的
      EngineCoreProc.run_engine_core），它自己在那儿不停地 step；
      客户端只管发请求、收输出。
      而进程内模式（InprocClient）里引擎是"被动"的，靠客户端调 get_output()
      才推进一步 —— 这是理解两者行为差异的关键。
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        renderer: BaseRenderer | None = None,
    ):
        self.vllm_config = vllm_config
        self._renderer: BaseRenderer | None = renderer

        # ZMQ setup.
        # [CN] io_threads=2：ZMQ 用两个 IO 线程分别处理收发，
        #      避免"收输出"被"发请求"阻塞（两者都是高频操作）。
        sync_ctx = zmq.Context(io_threads=2)
        # [CN] 异步模式要把 sync Context 包一层（zmq.asyncio.Context 支持 await）。
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        # [CN] 这是**异常安全**的关键设计：构造过程很长（建 socket、fork 子进程、
        #      等待握手），任何一步抛异常（模型加载失败最常见）都要把已经建好的
        #      资源收干净，否则会留下僵尸子进程和占用端口的 socket。
        #      做法：先注册 finalizer，再用 try/finally + success 标志兜底
        #      （见本函数末尾的 finally）。
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            # State used for data parallel.
            self.engines_running = False
            parallel_config = vllm_config.parallel_config
            # Elastic EP can remove a rank and later add it back with the same
            # identity. The client input ROUTER needs handover to allow the new
            # engine to replace the dead connection.
            # [CN] ZMQ_ROUTER_HANDOVER：默认情况下 ROUTER 拒绝第二个连接使用
            #      相同的身份标识（identity）。而弹性扩缩容会"删掉 rank 3、
            #      过一会儿又加回来且还是 rank 3" —— 身份相同但是全新的进程。
            #      开启 handover 后，新连接可以直接接管旧身份。
            enable_input_socket_handover = parallel_config.enable_elastic_ep

            self.stats_update_address: str | None = None
            tensor_queue: Queue | None = None
            # [CN] 两条分支，决定"引擎进程由谁拉起"：
            #   A) client_addresses 非空：**外部管理**（多 API server 共享引擎，
            #      或 Ray/Serverless 场景），本进程只是连上去；
            #   B) 否则：**本客户端自己拉起** EngineCore 子进程。
            if client_addresses:
                # Engines are managed externally to this client.
                input_address = client_addresses["input_address"]
                output_address = client_addresses["output_address"]
                self.stats_update_address = client_addresses.get("stats_update_address")
                # Tensor queues passed via client_addresses for multi-API-server case
                tensor_queue = client_addresses.get("tensor_queue")
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    input_address,
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )

                # Report bound endpoints back so the parent can forward
                # them to engines (mirrors the DPCoordinator pattern).
                actual_address_pipe: Connection | None = client_addresses.get(
                    "actual_address_pipe"
                )
                if actual_address_pipe is not None:
                    try:
                        actual_input = self.input_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_output = self.resources.output_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_address_pipe.send(
                            {
                                "input_address": actual_input,
                                "output_address": actual_output,
                            }
                        )
                    finally:
                        actual_address_pipe.close()
                # Engines are managed externally: this process does not fork
                # them, so there is no fork race; start the MM warmup now.
                self._start_mm_warmup()
            else:
                # Engines are managed by this client.
                addresses = get_engine_zmq_addresses(vllm_config)
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    addresses.inputs[0],
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, addresses.outputs[0], zmq.PULL
                )

                # Resolve ``tcp://host:0`` placeholders to bound endpoints
                # before engines DEALER-connect. No-op for IPC.
                # [CN] 端口写 0 表示"让操作系统随便挑一个空闲端口"。
                #      挑完之后必须用 LAST_ENDPOINT 问出**实际端口**，
                #      再把真实地址交给引擎去 connect（引擎是 DEALER，主动连）。
                addresses.inputs[0] = self.input_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                addresses.outputs[0] = self.resources.output_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()

                with launch_core_engines(
                    vllm_config, executor_class, log_stats, addresses
                ) as engine_launch:
                    self.resources.coordinator = engine_launch.coordinator
                    self.resources.engine_manager = engine_launch.engine_manager
                    coordinator = engine_launch.coordinator
                    addresses = engine_launch.addresses
                    tensor_queue = engine_launch.tensor_queue
                    # Engine-core processes have now all been forked/started
                    # (CoreEngineProcManager.proc.start()). It is now safe to
                    # launch the frontend background MM warmup: it must not
                    # run while fork() is in flight (a live thread holding a
                    # lock would deadlock the forked child), but the
                    # engine-core model load (minutes) that follows is exactly
                    # what the warmup should overlap with.
                    self._start_mm_warmup()

                self.stats_update_address = addresses.frontend_stats_publish_address
                if coordinator is not None:
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address()
                    )

            # Serialization setup with tensor queues for multimodal tensor IPC.
            tensor_ipc_sender: TensorIpcSender | None = None
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is not None and model_config.multimodal_config is not None:
                mm_tensor_ipc = model_config.multimodal_config.mm_tensor_ipc
                if mm_tensor_ipc == "torch_shm" and tensor_queue is not None:
                    tensor_ipc_sender = TensorIpcSender(tensor_queue)

            self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)

            dp_size = parallel_config.data_parallel_size
            dp_rank = parallel_config.data_parallel_index
            dp_local_size = parallel_config.data_parallel_size_local
            offline_mode = parallel_config.data_parallel_rank_local is not None
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
            self.engine_ranks_managed = (
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
            )
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )

            # ZMQ identity of each engine that this client will talk to.
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]

            # Wait for ready messages from each engine on the input socket.
            # [CN] 握手：必须收齐**每一个**引擎的 ready 才能开工，
            #      否则会出现"前端以为 8 个 rank 都好了，实际只有 5 个"的
            #      诡异状态（后续请求发到还没起来的 rank 会丢）。
            identities = set(self.core_engines)
            # [CN] Socket.shadow：在同一个底层 socket 上造一个**同步视图**。
            #      握手阶段是阻塞的（此时事件循环可能还没跑起来），
            #      所以不能用 async socket 的 await 版本。
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
                ):
                    raise TimeoutError(
                        f"Timed out waiting for engine core processes to "
                        f"start. This is often caused by slow weight loading "
                        f"for large models. Waited "
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                        f"timeout, set the environment variable: "
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                    )
                identity, payload = sync_input_socket.recv_multipart()
                identities.remove(identity)
                self._apply_ready_response(payload)

            self.core_engine: EngineIdentity = self.core_engines[0]
            # [CN] call_id -> Future 的等待表。控制类 RPC（abort/profile/sleep...）
            #      发出后把 Future 存这里，收到 UtilityOutput 时按 call_id 归位。
            self.utility_results: dict[int, AnyFuture] = {}

            # Start monitoring engine core processes for unexpected failures
            self.start_engine_core_monitor()

            success = True
        finally:
            # [CN] 构造失败就**立刻**跑一遍清理（不等 GC）：
            #      僵尸子进程会一直占着 GPU 显存，这是很痛的故障模式。
            if not success:
                self._finalizer()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        # [CN] finalizer.detach() 返回 None 表示**已经清理过了**（幂等保护）：
        #      __del__ 和显式 shutdown 都可能被调用，重复停子进程会报错。
        if self._finalizer.detach() is not None:
            timeout_str = "default" if timeout is None else f"{timeout}s"
            logger.info("[shutdown] MPClient: start timeout=%s", timeout_str)
            if self.resources.engine_manager is not None:
                logger.info_once("[shutdown] MPClient: stopping engine manager")
                self.resources.engine_manager.shutdown(timeout=timeout)
                logger.info_once("[shutdown] MPClient: engine manager stopped")
            logger.info_once("[shutdown] MPClient: cleaning up background resources")
            self.resources()
            logger.info_once("[shutdown] MPClient: complete")

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        # [CN] 很实用的一处设计：引擎已经死了的时候，所有等待中的 RPC 都应该
        #      报 **EngineDeadError** 而不是真正的底层错误（比如 "socket closed"）。
        #      否则用户看到的是一堆莫名其妙的 ZMQ 异常，而不是"引擎崩了"这个根因。
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    def dp_engines_running(self) -> bool:
        return self.engines_running

    def _start_mm_warmup(self) -> None:
        # Called once the engine-core process(es) have been created (forked or
        # externally managed). Overlap the frontend MM warmup with the
        # engine-core model load. This is a no-op when no renderer was passed
        # (e.g. text-only serving or tests).
        if self._renderer is not None:
            self._renderer.start_mm_warmup_in_background()

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        # [CN] 后台守护线程：盯着 EngineCore 子进程还活着没。
        #      为什么需要它：子进程可能**静默死亡**（OOM kill、CUDA 故障、段错误），
        #      此时前端不会收到任何异常消息，只会一直等输出。
        engine_manager = self.resources.engine_manager
        if engine_manager is None:
            # No engine processes to monitor
            # [CN] 外部管理的引擎（client_addresses 分支）没有 manager，无从监控。
            return

        # [CN] 同样的"避免循环引用"套路：监控线程只持有 **weakref**，
        #      否则线程活着 => client 活着 => 线程被引用，谁都退不掉。
        self_ref = weakref.ref(self)

        # Monitor engine core process liveness. If any die unexpectedly,
        # marks the engine as dead, and shuts down the client.
        def monitor_engine_cores():
            # [CN] 阻塞调用：等到任一引擎进程退出才返回。
            engine_manager.monitor_engine_liveness()
            _self = self_ref()
            if not _self or not _self._finalizer.alive or _self.resources.engine_dead:
                return
            _self.resources.engine_dead = True
            logger.warning_once(
                "[shutdown] MPClient: engine core exited unexpectedly; starting cleanup"
            )
            _self.shutdown()
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()

    def _apply_ready_response(self, payload: bytes) -> None:
        """Decode an EngineCoreReadyResponse and sync any post-initialization
        config changes (e.g. auto-fitted max_model_len) back to the frontend."""
        # [CN] **握手回写**：这一步解释了"为什么前端的配置会变"。
        #      引擎在初始化时做了很多前端算不出来的事（显存测量、block 对齐、
        #      dtype 决定），于是把真实值塞进 EngineCoreReadyResponse 发回来，
        #      前端在这里覆盖自己的 VllmConfig。
        if not payload:
            return
        vllm_config = self.vllm_config
        response = msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
        # [CN] 取 min：DP 下每个 rank 的实际 max_model_len 可能不同
        #      （显存大小不一样），必须按**最小的**那个来，否则小显存的 rank 会炸。
        vllm_config.model_config.max_model_len = min(
            vllm_config.model_config.max_model_len, response.max_model_len
        )

        # Setup KV cache config with initialization state from
        # engine core process. Sum num_gpu_blocks from all engines in DP case.
        # [CN] num_gpu_blocks 要**累加**（每个 DP rank 都有自己的显存和 block），
        #      所以总容量是各 rank 之和；
        #      而 block_size / kv_cache_size_tokens 是"每 rank 的值"，不能累加
        #      （下面单独处理）。这个区别很容易搞错。
        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        num_gpu_blocks += response.num_gpu_blocks
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

        # Sync block_size: may be enlarged by _align_hybrid_block_size in the
        # worker for hybrid Mamba models.
        cache_config = vllm_config.cache_config
        cache_config.block_size = response.block_size
        cache_config.mamba_block_size = response.mamba_block_size
        # Keep these as per-engine cache_config_info values; do not sum across DP.
        cache_config.kv_cache_size_tokens = (
            getattr(cache_config, "kv_cache_size_tokens", None)
            if getattr(cache_config, "kv_cache_size_tokens", None) is not None
            else response.kv_cache_size_tokens
        )
        cache_config.kv_cache_max_concurrency = (
            getattr(cache_config, "kv_cache_max_concurrency", None)
            if getattr(cache_config, "kv_cache_max_concurrency", None) is not None
            else response.kv_cache_max_concurrency
        )

        # In external DP LB mode, the coordinator address that the
        # front-end procs connect to is obtained by each engine via it's
        # initial handshake with the rank 0 front-end.
        if response.dp_stats_address is not None:
            if self.stats_update_address is None:
                self.stats_update_address = response.dp_stats_address
            else:
                assert response.dp_stats_address == self.stats_update_address


def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    # [CN] 控制类 RPC 的"应答归位"：按 call_id 找到发起方留下的 Future，
    #      把结果（或异常）set 进去，从而唤醒那个正在 await 的协程/阻塞等待的线程。
    #      这是典型的 **请求-响应关联（correlation）** 模式。
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""
    # [CN] 同步版的做法：**开一个后台线程专门收输出**，主线程要输出时从
    #      queue.Queue 里 get（阻塞）。
    #      这样设计的意义：引擎的输出是**随时会来**的（引擎在另一个进程自己循环），
    #      如果等调用方要的时候才去 recv，就会因为"没人及时收"而让 socket 缓冲区堆积，
    #      引擎那边写阻塞。后台线程收 + 队列缓冲，把"引擎节奏"和"调用方节奏"解耦。

    @instrument(span_name="SyncMPClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        renderer: BaseRenderer | None = None,
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            renderer=renderer,
        )

        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        # [CN] 老规矩：线程里只捕获局部变量，绝不出现 self。
        #      注意 `resources` 是个**例外** —— 它是独立的 dataclass，
        #      持有它不会形成回到 client 的引用链（它不指向 client）。
        ctx = self.ctx
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        # [CN] inproc 路径用于"关闭信号"：进程内的 PAIR socket，
        #      不走网络，开销极小，专门用来唤醒阻塞在 poll() 上的收包线程。
        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            """[CN] 收包线程主循环：poll(输出 socket, 关闭 socket) 二选一。"""
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        # [CN] 两个 socket 同时就绪时也当关闭处理
                        #      （关闭优先，避免关到一半还在收）。
                        break

                    # [CN] copy=False：零拷贝接收，帧直接引用 ZMQ 的缓冲区。
                    #      输出的量很大（每步每请求都有），省一次 memcpy 很值。
                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    # [CN] **分路**：控制类应答 -> Future；数据面输出 -> 队列。
                    #      这是理解整个客户端的关键分叉点。
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                # [CN] 收包线程里的异常**不能就地抛**（没人能接住），
                #      而是塞进队列：调用方 get_output() 拿到后重新抛出，
                #      从而让异常在**正确的线程**里被处理（进而关掉整个服务）。
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                # [CN] linger=0：立即关闭，不要等未发送完的消息。
                #      关闭阶段再等就是白白拖延退出时间。
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        # [CN] daemon=True：主线程退出时不等待它（否则收包线程阻塞在 poll 上
        #      会让进程永远退不出去）。
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        # [CN] 所有权转移：socket 由**收包线程**关闭（上面 finally 里），
        #      resources 里置 None，避免 BackgroundResources 再关一次导致重复关闭。
        self.resources.output_socket = None

    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        # [CN] 阻塞等待，直到引擎产出一帧（或收到异常）。
        outputs = self.outputs_queue.get()

        if isinstance(outputs, Exception):
            # [CN] from None：切断原始异常上下文，避免打印出一长串
            #      线程内部的堆栈（对排查"引擎为什么死"没有帮助）。
            raise self._format_exception(outputs) from None
        # [CN] DP 波次结束信号：把"引擎还在跑"标志复位。
        #      这个标志被 has_unfinished_requests 用来判断还要不要继续 step。
        if outputs.wave_complete is not None:
            self.engines_running = False
        return outputs

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        # (Identity, RequestType, SerializedRequest)
        # [CN] ROUTER socket 的发送格式：**第一帧必须是目标身份**，
        #      第二帧是请求类型（那个单字节枚举），后面才是序列化后的请求体。
        #      因为 input_socket 是 ROUTER，可以同时连多个引擎，靠身份路由。
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        # [CN] 这个注释解释了一个很容易怀疑的点：copy=False 发送后，
        #      Python 侧的 tensor 是不是可以被 GC？
        #      答案是安全的 —— ZMQ 内部持有 memoryview，而 memoryview 又持有
        #      原始 tensor 的引用，所以引用链不断，不会被提前释放。
        self.input_socket.send_multipart(msg, copy=False)

    def call_utility(self, method: str, *args) -> Any:
        """[CN] 发起一次**控制类 RPC 并同步等待结果**。

        流程：生成 call_id -> 建 Future 并登记 -> 发消息 -> future.result() 阻塞等待。
        归位由收包线程里的 _process_utility_output 完成。
        """
        # [CN] uuid1 取高 64 位做 call_id：比自增计数器简单（不用加锁），
        #      且多前端/多进程也不会撞。
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        # [CN] 注意消息体里的第一个 0 是"引擎编号"（非 DP 场景恒为 0）。
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        # [CN] **没有超时**：控制类 RPC 默认一直等。
        #      代价是引擎卡住时这里会永久阻塞（生产环境要注意这一点）。
        return future.result()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.call_utility("get_supported_tasks")

    def add_request(self, request: EngineCoreRequest) -> None:
        # [CN] DP 场景下"刚发了请求"就意味着引擎要跑起来（至少跑完这一波），
        #      所以这里乐观地把 engines_running 置 True ——
        #      它会被 wave_complete 信号复位（见 get_output）。
        if self.is_dp:
            self.engines_running = True
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        # [CN] 两重短路：空列表不发；引擎已死也不发
        #      （发了也没人回，还会走一遍 ensure_alive 抛异常，把 abort 变成报错）。
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.call_utility("profile", is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.call_utility("reset_mm_cache")

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.call_utility(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.call_utility("reset_encoder_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        self.call_utility("sleep", level, mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self) -> None:
        self.call_utility("execute_dummy_batch")

    def set_weight_version(self, weight_version: str) -> None:
        self.call_utility("set_weight_version", weight_version)

    def get_weight_version(self) -> str:
        return self.call_utility("get_weight_version")

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""
    # [CN] 与 SyncMPClient 的对照：
    #   同步版：后台**线程** + queue.Queue + 阻塞 get()；
    #   异步版：后台 **asyncio.Task** + asyncio.Queue + await get()。
    #   两者都是"后台收、前台取"，只是并发原语不同。
    #   另一个重要差异：异步版支持**多引擎**（DP）—— _send_input 可以指定
    #   目标 engine（ROUTER 身份），而同步版固定发给 core_engine。

    @instrument(span_name="AsyncMPClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        renderer: BaseRenderer | None = None,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
            renderer=renderer,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()

        # locally-cached engine status
        self._engine_status: dict[int, dict] = {}
        if self.vllm_config.parallel_config.enable_fault_tolerance:
            self._engine_status = {
                rank: {"id": rank, "status": "healthy"}
                for rank in self.engine_ranks_managed
            }
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            # [CN] 与 AsyncLLM 的 output_handler 一样是**惰性启动**，
            #      但这里额外强调了一个理由：如果收包任务没起来，
            #      引擎发来的 EXECUTOR_FAILED（执行器崩溃）消息就没人接，
            #      前端会一直以为引擎活着 —— 所以每个可能"用之前"的入口
            #      （get_output_async / add_request_async / call_utility_async）
            #      都会补一次 _ensure_output_queue_task()。
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def _ensure_output_queue_task(self):
        """[CN] 确保收包协程已启动（幂等）。"""
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        # [CN] 这里有个巧妙的"钩子"设计：output_handler 从 **类属性**
        #      process_engine_outputs 上取（子类可以覆盖），而不是硬编码。
        #      这样 DP 子类可以插入自己的处理逻辑（波次推进、负载均衡统计），
        #      而基类不用知道。取到之后通过 weakref 拿 self 调用它。
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self)
        output_socket = resources.output_socket
        assert output_socket is not None

        notification_callback_handler: (
            Callable[[AsyncMPClient, Sequence[Any]], Any] | None
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)

        async def process_outputs_socket():
            """[CN] 收包协程：与同步版同构，但用 await recv 代替 poll。"""
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        # [CN] utility 应答有三类，按 call_id 区分：
                        #   -1（EEP_NOTIFICATION_CALL_ID）：弹性扩缩容通知；
                        #   -2（FT_STATUS_CALL_ID）      ：容错状态更新；
                        #   其他（>=0）                   ：普通 RPC 应答 -> Future。
                        if (
                            outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID
                            and notification_callback_handler is not None
                        ):
                            assert _self_ref is not None
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is None:
                                continue
                            notification_data = outputs.utility_output.result.result
                            assert isinstance(notification_data, Sequence)
                            assert len(notification_data) == 2
                            asyncio.create_task(
                                notification_callback_handler(_self, notification_data)
                            )
                        elif outputs.utility_output.call_id == FT_STATUS_CALL_ID:
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is not None:
                                _self._engine_status[outputs.engine_index] = (
                                    outputs.utility_output.result.result
                                )
                        else:
                            _process_utility_output(
                                outputs.utility_output, utility_results
                            )
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    # [CN] **空帧不入队**：只有真的带了输出或调度统计才唤醒消费方。
                    #      否则每步都会往队列塞一个空对象，白白唤醒 get_output_async
                    #      （DP 场景下很多引擎会持续发空心跳帧）。
                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                # [CN] 收包任务被取消（引擎关闭流程）时，往队列塞一个
                #      EngineDeadError —— 让等在 get_output_async 的协程
                #      收到明确信号而不是永远挂起。
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        """[CN] 注意返回值是 **Awaitable 而不是 None**，且本方法不是 async：
        构造好消息后返回 send_multipart(...) 这个协程对象，由调用方 await。
        好处：调用方可以在 await 之前先做别的事（比如登记 Future），
        避免"消息已发出但没人等应答"的竞态。
        """
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine)

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity
    ) -> Awaitable[Any]:
        self.ensure_alive()
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        return self.input_socket.send_multipart((engine,) + message, copy=False)

    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        call_id = uuid.uuid1().int >> 64
        # [CN] 顺序很重要：**先登记 Future，再发消息**。
        #      反过来（先发后登记）就有可能应答先到，
        #      那时 utility_results 里还没有这个 call_id -> 应答被丢弃，永久等待。
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            # [CN] 这里比同步版多带了 client_index：多前端场景下引擎要靠它
            #      把应答发给正确的前端（否则 A 前端会收到 B 前端的应答）。
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine)
        self._ensure_output_queue_task()
        return await future

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return await self.call_utility_async("get_supported_tasks")

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # [CN] 打上本前端的编号：引擎据此把这条请求的输出**原路送回**。
        #      多前端共享引擎时，没有它输出就会串到别的前端去。
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def pause_scheduler_async(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> None:
        await self.call_utility_async("pause_scheduler", mode, clear_cache)

    async def resume_scheduler_async(self) -> None:
        await self.call_utility_async("resume_scheduler")

    async def is_scheduler_paused_async(self) -> bool:
        return await self.call_utility_async("is_scheduler_paused")

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        await self.call_utility_async("profile", is_start, profile_prefix)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    async def reset_encoder_cache_async(self) -> None:
        await self.call_utility_async("reset_encoder_cache")

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        await self.call_utility_async("sleep", level, mode)

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def set_weight_version_async(self, weight_version: str) -> None:
        await self.call_utility_async("set_weight_version", weight_version)

    async def get_weight_version_async(self) -> str:
        return await self.call_utility_async("get_weight_version")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )

    async def handle_fault(
        self, ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """[CN] 发一条容错指令（注入/清除故障），并把结果记进本地状态表
        （供 get_status 暴露给运维接口）。"""
        res = await self.call_utility_async(FT_UTILITY_METHOD, ft_request)
        # [CN] msgspec.convert：引擎回的是通用结构，这里转成强类型
        #      FaultToleranceResult（跨进程序列化后类型信息会丢）。
        result = msgspec.convert(res, FaultToleranceResult)
        if not result.success:
            status = self._engine_status.get(self.engine_ranks_managed[0])
            if status is not None:
                status["last_ft_request_id"] = result.request_id
                status["ft_error"] = result.reason
        return result

    async def get_status(self):
        return {
            "schema_version": 1,
            "total_engines": len(self.engine_ranks_managed),
            "engines": list(self._engine_status.values()),
        }


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""
    # [CN] DP 客户端要解决的核心问题是**波次（wave）同步**：
    #   所有 DP rank 必须每轮跑**相同形状**的批次，否则 NCCL 集合通信会对不上
    #   （一个 rank 做 all-reduce、另一个没做 => 死锁）。
    #   所以引擎按波次推进：一波请求全部完成后才能开下一波，
    #   期间各 rank 通过 start_wave / wave_complete 信号互相通知。
    #   本类负责在前端侧维护 current_wave 并在需要时广播 START_DP_WAVE。
    #
    #   另一个机制是 **first_req 通知**：本前端刚收到一个新请求时，
    #   要通过 inproc socket 通知 stats 任务（它会去唤醒/推进波次）。

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        renderer: BaseRenderer | None = None,
    ):
        # [CN] 必须在 super().__init__ **之前**初始化：父类构造期间会启动收包任务，
        #      而收包回调里可能立刻读到 current_wave（顺序错了会 AttributeError）。
        self.current_wave = 0

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
            renderer,
        )

        # List of [waiting, running, kv_cache_usage] per engine.
        # Used only by DPLBAsyncMPClient subclass.
        # [CN] 每个引擎的负载快照，用于"内部负载均衡"挑最闲的 rank。
        self.lb_engines: list[list[int | float]] = [
            [0, 0, 0.0] for _ in self.core_engines
        ]

        self.eep_scaling_cache: ElasticScalingCache | None = None

        # [CN] 进程内 PAIR 通道：add_request 时通知 stats 任务"有新请求来了"。
        #      用 inproc 而不是跨进程，因为这是**本进程内**两个协程之间的通信。
        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    def _ensure_stats_update_task(self):
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0

        async def run_engine_stats_update_task():
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                # [CN] XSUB 子套接字的订阅协议：第一个字节 0x01 表示"订阅所有"。
                #      （ZMQ PUB/SUB 的订阅前缀是长度+内容，\x01 + 空 = 订阅全部主题）
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                        not self.engines_running
                        and len(events) == 2
                        or (events[0][0] == first_req_rcv_socket)
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                            and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            # Extract new engine count from the decoded message
                            new_engine_count = decoded[1]
                            # Update engine_ranks_managed and count_slice
                            parallel_config = self.vllm_config.parallel_config
                            dp_size = parallel_config.data_parallel_size
                            dp_rank = parallel_config.data_parallel_rank
                            assert dp_rank == 0
                            assert dp_size == new_engine_count
                            assert not (
                                parallel_config.data_parallel_hybrid_lb
                                or parallel_config.data_parallel_external_lb
                            )
                            num_ranks = dp_size
                            self.engine_ranks_managed = list(
                                range(dp_rank, dp_rank + num_ranks)
                            )
                            if len(self.lb_engines) < new_engine_count:
                                self.lb_engines = self.lb_engines + [
                                    [0, 0, 0.0]
                                    for _ in range(
                                        new_engine_count - len(self.lb_engines)
                                    )
                                ]
                            else:
                                self.lb_engines = self.lb_engines[:new_engine_count]
                            # Send scale up notification to coordinator
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        # [CN] 统计信息是"快照"，每 100ms 一版，**只有最新的有意义**。
                        #      所以这里把积压的消息全部读掉，只保留最后一条。
                        #      不这么做的话，处理延迟会让负载均衡用上过期数据。
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    counts, wave, running = msgspec.msgpack.decode(buf)
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        ranks = self.engine_ranks_managed
                        count_slice = slice(ranks[0], ranks[-1] + 1)
                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()

        # [CN] 打上当前波次：引擎收到后会判断"这个请求属于老波次还是新波次"，
        #      从而决定要不要触发 start_wave（见 EngineCoreRequest.current_wave 的注释）。
        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        # [CN] 先构造发送协程（**还没发**），再处理"唤醒"逻辑，最后才 await 发送。
        #      顺序的原因：唤醒消息（FIRST_REQ）必须先于请求到达协调器，
        #      否则协调器会看到"引擎已暂停却来了请求"这种矛盾状态。
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            # [CN] 引擎都暂停了（上一波结束）却又来了新请求：
            #      必须通知协调器去**唤醒其他 rank** 开新的一波，
            #      否则只有目标引擎醒着、其他 rank 还在等 -> 集合通信死锁。
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""
    # [CN] 与父类 DPAsyncMPClient 的区别只有一个：**谁来挑 rank**。
    #   父类（外部 LB）：请求里已经指定了 data_parallel_rank，直接用；
    #   本类（内部 LB）：客户端自己按负载挑，核心是 get_core_engine_for_request()
    #   里的打分逻辑（下面有详细注释）。
    #   代价：要给每个请求记账（发给了哪个 engine），abort 时才能找对引擎。

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        renderer: BaseRenderer | None = None,
    ):
        self.client_count = client_count

        # To route aborts to the correct engine.
        # [CN] request_id -> 它实际被发到了哪个引擎。abort 时必须发给同一个引擎
        #      （广播 abort 到所有引擎也可以，但代价大且会产生"未知请求"日志）。
        self.reqs_in_flight: dict[str, EngineIdentity] = {}

        # Exact per-engine count of this client's unfinished requests.
        # [CN] 协调器的负载快照是**采样**的（100ms 一次），可能过时；
        #      而本客户端自己发出的请求数是**精确**的。两者结合打分（见下）。
        self.engine_inflight: Counter[EngineIdentity] = Counter()

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
            renderer,
        )

        assert len(self.core_engines) > 1
        self._prepared_elastic_ep: tuple[int, int] | None = None

        # [CN] 多前端时，让每个前端从**不同的引擎**开始扫描。
        #      否则所有前端都从 rank 0 开始挑，会把突发流量全堆到 rank 0。
        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        """[CN] 挑一个引擎发这个请求。三种情形，优先级从高到低：
        1) 请求显式指定了 data_parallel_rank -> 直接用（可预测/可复现）；
        2) late-interaction 池化任务 -> 按专门规则选（见 get_late_interaction_engine_index）；
        3) 否则 -> 按负载打分选最闲的（下面的 for 循环）。
        """
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score: float = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running, kv_cache_usage = current_counts[idx]
                # Estimate engine load as the greater of the coordinator's
                # latest (waiting + running) snapshot and this client's own
                # in-flight count (scaled by the number of clients). The
                # in-flight floor is exact and can't be erased by a snapshot
                # rebind, so a burst spreads round-robin even when snapshots
                # race with routing decisions; the snapshot raises the score
                # when other clients or stale requests load the engine.
                inflight = self.engine_inflight[self.core_engines[idx]]
                score: float = max(self.client_count * inflight, waiting + running)
                if waiting:
                    # Waiting requests are penalized in proportion to KV cache
                    # pressure: a queue on a KV-bound engine drains slowly, so
                    # new requests should strongly prefer other engines. With
                    # low KV usage the queue is transient (e.g. mid-burst) and
                    # the penalty stays off, preserving exact round-robin.
                    # Ramps from 0 at <=50% usage to 3x waiting at 100%.
                    score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count
            # Rotate the scan start so that ties (equal scores, e.g. right
            # after a coordinator stats reset when engines look equally loaded)
            # don't systematically favor the same engine. This removes the
            # fixed tie-break bias without affecting load-aware decisions when
            # scores actually differ.
            self.eng_start_index = (self.eng_start_index + 1) % num_engines

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        # [CN] 自己发的请求先+1（不等协调器快照）：这样"连续发 10 个请求"
        #      不会因为快照没更新而全发到同一个引擎。
        self.engine_inflight[chosen_engine] += 1
        return chosen_engine

    async def call_utility_async(self, method: str, *args) -> Any:
        # [CN] 控制类调用要**广播到所有引擎**（比如 sleep 不能只让一个 rank 睡），
        #      但只返回第一个引擎的结果（各引擎返回内容一样）。
        # Only the result from the first engine is returned.
        return (
            await asyncio.gather(
                *[
                    self._call_utility_async(method, *args, engine=engine)
                    for engine in self.core_engines
                ]
            )
        )[0]

    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        """[CN] 这是前面说的"钩子"：基类收包时回调到这里。
        作用：请求结束时把记的账销掉（reqs_in_flight / engine_inflight），
        否则负载统计会只增不减，最后所有请求都堆到"看起来最闲"的那个引擎。
        """
        if outputs.finished_requests and self.reqs_in_flight:
            for req_id in outputs.finished_requests:
                if (engine := self.reqs_in_flight.pop(req_id, None)) is not None:
                    self.engine_inflight[engine] -= 1

    @staticmethod
    async def eep_process_engine_core_notification(
        self: "DPLBAsyncMPClient", notification_data: tuple[str, int]
    ):
        """[CN] 处理弹性扩缩容期间来自引擎的通知（第二个"钩子"）。

        两类通知：
          - RECONFIGURE_FINISHED：重配置完成 -> 造一个假 UtilityOutput 去
            resolve 那个在等 RECONFIGURE 的 Future（见
            _eep_wait_for_setup_switch_complete）。这个"借道 utility 通道"
            的技巧很巧妙：复用了现成的 call_id -> Future 机制。
          - SHUTDOWN_COMPLETE：某 rank 已安全退出 -> 收集齐了才真正缩容
            （scale_down_elastic_ep）。必须等所有要裁掉的 rank 都回话。
        """
        cache = self.eep_scaling_cache
        notification_type_str, dp_rank = notification_data
        try:
            notification_type = EEPNotificationType(notification_type_str)
        except ValueError as e:
            raise ValueError(
                f"Unknown EEP notification type: {notification_type_str}"
            ) from e

        if notification_type == EEPNotificationType.RECONFIGURE_FINISHED:
            from vllm.v1.engine import UtilityResult

            # NOTE(yongji): process a dummy UtilityOutput to resolve the future
            # awaited in _eep_wait_for_setup_switch_complete(), signaling that
            # all engine cores have completed reconfiguration.
            dummy_output = UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID, result=UtilityResult(None)
            )
            _process_utility_output(dummy_output, self.utility_results)
            return
        assert cache is not None
        if notification_type not in cache.pending_notifications:
            cache.pending_notifications[notification_type] = set()
        if dp_rank in cache.pending_notifications[notification_type]:
            raise ValueError(
                f"Duplicate notification {notification_type} from dp_rank {dp_rank}"
            )
        cache.pending_notifications[notification_type].add(dp_rank)
        if len(cache.pending_notifications[notification_type]) >= abs(
            cache.num_new_core_engines
        ):
            engine_manager = self.resources.engine_manager
            assert isinstance(engine_manager, CoreEngineActorManager)
            assert cache.num_new_core_engines < 0
            old_dp_size = len(cache.existing_core_engines)
            new_dp_size = old_dp_size + cache.num_new_core_engines
            engine_manager.scale_down_elastic_ep(old_dp_size, new_dp_size)
            self.vllm_config.parallel_config.data_parallel_size_local = len(
                engine_manager.local_engine_actors
            )
            self.eep_scaling_cache = None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """[CN] abort 要**按引擎分组**发（每个请求发到它当初去的那个引擎），
        这是内部 LB 模式特有的复杂度。注意查不到的请求直接跳过：
        它可能已经结束（reqs_in_flight 里被清理了），广播没有意义。
        """
        if not request_ids or self.resources.engine_dead:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine[engine].append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    async def commit_elastic_ep(self) -> None:
        """Commit prepared elastic EP scaling."""
        # [CN] 两阶段提交的"提交"阶段。prepare 阶段已经把新引擎拉起来并配好了，
        #      这里只做切换 + 更新 EPLB 的冗余专家数（专家数变了，冗余数也要变）。
        prepared = self._prepared_elastic_ep
        if prepared is None:
            raise RuntimeError("Elastic EP scaling has not been prepared")
        new_data_parallel_size, num_redundant_experts = prepared
        cur_data_parallel_size = len(self.core_engines)
        if new_data_parallel_size > cur_data_parallel_size:
            await self._commit_scale_up_elastic_ep(new_data_parallel_size)
        else:
            await self._commit_scale_down_elastic_ep(new_data_parallel_size)
        self.vllm_config.parallel_config.eplb_config.num_redundant_experts = (
            num_redundant_experts
        )
        self._prepared_elastic_ep = None

    async def prepare_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Prepare elastic EP scaling without routing requests to new engines."""
        # [CN] "准备"阶段：把新引擎拉起来、配置好，但**还不给它派请求**。
        #      这样切换瞬间就能完成（真正 commit 时几乎无停顿）。
        #      核心计算是"新的冗余专家数"：EPLB 会把热门专家复制多份，
        #      DP 规模变了，冗余数必须跟着变，否则专家放不下。
        #      注意约束：只有 ray DP backend 支持（进程管理要能动态增删）。
        if (prepared := self._prepared_elastic_ep) is not None:
            if prepared[0] == new_data_parallel_size:
                return
            raise RuntimeError("Elastic EP scaling is already prepared")
        cur_data_parallel_size = len(self.core_engines)
        assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (
            "Only ray DP backend supports scaling elastic EP"
        )
        parallel_config = self.vllm_config.parallel_config
        num_experts = self.vllm_config.model_config.get_num_experts()
        num_physical_experts = (
            num_experts + parallel_config.eplb_config.num_redundant_experts
        )
        num_redundant_experts = (
            num_physical_experts * new_data_parallel_size // cur_data_parallel_size
            - num_experts
        )
        if num_redundant_experts < 0:
            # Scaling keeps physical experts per engine fixed, so below this
            # size the logical experts no longer fit.
            raise ValueError(
                f"Cannot scale to data_parallel_size {new_data_parallel_size}, "
                f"minimum is "
                f"{-(-num_experts * cur_data_parallel_size // num_physical_experts)}"
            )
        if new_data_parallel_size < cur_data_parallel_size:
            await self._prepare_scale_down_elastic_ep(new_data_parallel_size)
        else:
            await self._prepare_scale_up_elastic_ep(
                new_data_parallel_size, num_redundant_experts
            )
        self._prepared_elastic_ep = new_data_parallel_size, num_redundant_experts

    def _eep_wait_for_setup_switch_complete(self) -> asyncio.Future:
        """
        Wait for core engines to switch to the new setup.

        In eep_process_engine_core_notification(), a dummy UtilityOutput with
        EEP_NOTIFICATION_CALL_ID will be set when RECONFIGURE_FINISHED
        notification is received from engine 0. We create a future with
        that call_id and wait for it to be resolved.
        """
        future = asyncio.get_running_loop().create_future()
        self.utility_results[EEP_NOTIFICATION_CALL_ID] = future
        self._ensure_output_queue_task()
        return future

    def _wait_for_new_engine_ready(self, new_core_engines: list[bytes]) -> None:
        new_engine_identities = set(new_core_engines)
        sync_input_socket = zmq.Socket.shadow(self.input_socket)
        while new_engine_identities:
            if not sync_input_socket.poll(timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000):
                raise TimeoutError(
                    f"Timed out waiting for new engine core processes to "
                    f"start. Waited "
                    f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                    f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                    f"timeout, set the environment variable: "
                    f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                )
            identity, payload = sync_input_socket.recv_multipart()
            new_engine_identities.discard(identity)
            self._apply_ready_response(payload)

    def _setup_elastic_ep_reconfig_bootstrap(self) -> None:
        from vllm.distributed.utils import create_tcp_store
        from vllm.utils.network_utils import get_open_ports_list

        parallel_config = self.vllm_config.parallel_config
        parallel_config._data_parallel_master_port_list = get_open_ports_list(5)
        parallel_config.data_parallel_master_port = (
            parallel_config._data_parallel_master_port_list.pop()
        )

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

    def _make_reconfig_request(
        self,
        new_data_parallel_size: int,
        rank_type: ReconfigureRankType = ReconfigureRankType.KEEP_CURRENT_RANK,
    ) -> ReconfigureDistributedRequest:
        parallel_config = self.vllm_config.parallel_config
        return ReconfigureDistributedRequest(
            new_data_parallel_size=new_data_parallel_size,
            new_data_parallel_rank=rank_type,
            new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_master_ip=parallel_config.data_parallel_master_ip,
            new_data_parallel_master_port=parallel_config.data_parallel_master_port,
            new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
            coord_store_port=parallel_config._coord_store_port,
        )

    async def _prepare_scale_up_elastic_ep(
        self,
        new_data_parallel_size: int,
        num_redundant_experts: int,
    ) -> None:
        """Prepare scale up by creating new engine cores and reconfiguring
        existing ones."""
        self._setup_elastic_ep_reconfig_bootstrap()

        # Phase 1: Send reconfig messages to existing engines
        reconfig_futures = []
        for engine in self.core_engines:
            reconfig_request = self._make_reconfig_request(new_data_parallel_size)
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # Phase 2: Create new engines
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        start_new_worker_future = asyncio.to_thread(
            self.resources.engine_manager.scale_up_elastic_ep,
            self.vllm_config,
            new_data_parallel_size,
            num_redundant_experts,
        )

        # Phase 3: Wait for new engines to be created
        # and reconfig messages to be received
        await asyncio.gather(start_new_worker_future, *reconfig_futures)
        ready_keys = [future.result() for future in reconfig_futures]
        ready_keys.extend(
            f"eep_ready/{rank}"
            for rank in range(len(self.core_engines), new_data_parallel_size)
        )
        await asyncio.to_thread(self._coord_store.wait, ready_keys)
        logger.info("[Elastic EP] Successfully started new engines")

    async def _commit_scale_up_elastic_ep(self, new_data_parallel_size: int) -> None:
        new_core_engines = [
            rank.to_bytes(2, "little")
            for rank in range(len(self.core_engines), new_data_parallel_size)
        ]

        await self.pause_scheduler_async(mode="keep", clear_cache=False)
        wait_future = self._eep_wait_for_setup_switch_complete()
        finish_futures = [
            asyncio.create_task(
                self._call_utility_async("commit_prepared_elastic_ep", engine=engine)
            )
            for engine in self.core_engines
        ]
        try:
            await asyncio.gather(*finish_futures)
            await wait_future
            self._wait_for_new_engine_ready(new_core_engines)
        except Exception:
            wait_future.cancel()
            raise

        self.core_engines.extend(new_core_engines)
        # Update the parallel config
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = new_data_parallel_size
        if isinstance(self.resources.engine_manager, CoreEngineActorManager):
            parallel_config.data_parallel_size_local = len(
                self.resources.engine_manager.local_engine_actors
            )
        # Notify coordinator about scale up through existing
        # stats_update_task connection
        self._ensure_stats_update_task()
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_up_marker)

        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )
        await self.resume_scheduler_async()

    async def _prepare_scale_down_elastic_ep(self, new_data_parallel_size: int) -> None:
        self._setup_elastic_ep_reconfig_bootstrap()

        reconfig_futures = []
        for engine in self.core_engines[:new_data_parallel_size]:
            reconfig_request = self._make_reconfig_request(new_data_parallel_size)
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        ready_keys = await asyncio.gather(*reconfig_futures)
        await asyncio.to_thread(self._coord_store.wait, ready_keys)

    async def _commit_scale_down_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        cur_data_parallel_size = len(self.core_engines)

        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=self.core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
        )

        old_core_engines = self.core_engines
        # NOTE(yongji): Immediately stop sending requests to the removing engines.
        self.core_engines = old_core_engines[:new_data_parallel_size]
        self.lb_engines = self.lb_engines[:new_data_parallel_size]
        removed_dp_size = cur_data_parallel_size - new_data_parallel_size
        pause_modes = ["keep"] * new_data_parallel_size + ["abort"] * removed_dp_size
        pause_futures = [
            self._call_utility_async("pause_scheduler", mode, False, engine=engine)
            for mode, engine in zip(pause_modes, old_core_engines)
        ]
        await asyncio.gather(*pause_futures)
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        self.resources.engine_manager.remove_run_refs_for_scale_down(removed_dp_size)
        wait_future = self._eep_wait_for_setup_switch_complete()
        reconfig_futures = []
        for cur_dp_rank, engine in enumerate(old_core_engines):
            if cur_dp_rank < new_data_parallel_size:
                coro = self._call_utility_async(
                    "commit_prepared_elastic_ep", engine=engine
                )
            else:
                reconfig_request = self._make_reconfig_request(
                    new_data_parallel_size,
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK,
                )
                coro = self._call_utility_async(
                    "reinitialize_distributed", reconfig_request, engine=engine
                )
            reconfig_futures.append(asyncio.create_task(coro))

        try:
            await asyncio.gather(*reconfig_futures)

            self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            self._ensure_stats_update_task()
            scale_down_marker = msgspec.msgpack.encode(
                ("SCALE_ELASTIC_EP", new_data_parallel_size)
            )
            await self.first_req_send_socket.send(scale_down_marker)
            await wait_future
            await self.resume_scheduler_async()
        except Exception:
            wait_future.cancel()
            raise

        logger.info(
            "[Elastic EP] Scale down completed, new data parallel size: %s",
            new_data_parallel_size,
        )
