# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：多进程执行器（默认生产路径）。
# [CN] 链路：EngineCore --共享内存 MQ--> N 个 WorkerProc 子进程 --> Worker --> ModelRunner。
# [CN] 核心设计：
# [CN]   1) 一条广播 MQ（rpc_broadcast_mq）下发 RPC 请求，所有 worker 收到同一条消息；
# [CN]   2) 每个 worker 一条回传 MQ（worker_response_mq），只让需要的 rank 回结果；
# [CN]   3) 两条普通 Pipe：ready_pipe（子->父，汇报就绪）与 death_pipe（父->子，父进程猝死通知）。
# [CN] 核心类：
# [CN]   - FutureWrapper：异步 RPC 的 Future，带 FIFO 顺序保证；
# [CN]   - MultiprocExecutor：父进程侧的控制器（建进程、发 RPC、监控、关停）；
# [CN]   - WorkerProc：子进程侧的实现（初始化、忙循环、输出回传）。
# [CN] 最容易看错的点：
# [CN]   1) MQ 的「就绪等待」顺序是一处隐式协议，父与子必须完全一致，改顺序就会死锁；
# [CN]   2) fork 启动方式下子进程会继承前面 worker 的 socket fd，
# [CN]      必须显式 close（inherited_fds），否则 EOF 永远收不到、进程无法退出；
# [CN]   3) output_rank 决定「谁回结果」—— TP 内只有 rank 0 回，PP 只看最后一个 stage。

import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.utils import numa_utils
from vllm.utils.network_utils import (
    aiter_requires_tcp_store,
    get_distributed_init_method,
    get_file_store_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.ompmultiprocessing import OMPProcessManager
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


# [CN] 带 FIFO 顺序的 Future：RPC 结果必须按发出顺序返回，
# [CN] 因此每个 Future 都登记进 futures_queue，取结果时先把排在前面的全部 drain 掉。
class FutureWrapper(Future):
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response: Callable[[], Any],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    # [CN] 关键：不是等自己，而是「从队尾依次弹出、逐个等」，从而保证顺序。
    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        # [CN] 若前面的 Future 还没完成，就替它们把响应取回来（pop 从右端即队尾）。
        # Drain any futures ahead of us in the queue.
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    # [CN] suppress(InvalidStateError)：同一个 Future 可能被前面的 drain 顺带设置过，
    # [CN] 重复设置会抛异常，这里静默忽略。
    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


# [CN] 多进程执行器：vLLM 的默认执行路径。
class MultiprocExecutor(Executor):
    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    # [CN] 建进程、建 MQ、等就绪、起监控线程 —— 任何一步失败都要清理干净。
    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        # [CN] weakref.finalize：即使调用方忘了 shutdown，解释器退出时也会兜底杀掉 worker。
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        # [CN] 不变式：world_size == TP * PP * PCP。三者缺一说明配置错了。
        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # [CN] 本节点上的进程数要乘上「本节点的 DP 份数」（多 DP 共存于同一节点时）。
        num_local_procs = self.local_world_size * max(
            1, self.parallel_config.data_parallel_size_local
        )
        set_multiprocessing_worker_envs(num_local_procs)

        if aiter_requires_tcp_store():
            distributed_init_method = get_distributed_init_method(
                get_loopback_ip(), get_open_port()
            )
        else:
            distributed_init_method = get_file_store_init_method()
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        # [CN] 只有每个 DP 组的 leader 节点才建广播 MQ 与连接；follower 节点不参与 RPC 下发。
        if self.parallel_config.node_rank_within_dp == 0:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            # [CN] 单条消息的分片上限，直接影响大 SchedulerOutput 能否一次发完。
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        # [CN] 用 spawn 还是 fork 由平台决定（CUDA 通常强制 spawn）。
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # [CN] fork 模式下子进程会继承父进程所有 fd，包括前面 worker 的 pipe；
            # [CN] 记录下来是为了在后续 worker 里显式关闭，否则 EOF 检测失效。
            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            # For CPU backend only, to setup OpenMP threads affinity
            cpu_omp_manager = OMPProcessManager(self.vllm_config)
            # [CN] 逐个拉起本节点的 worker 进程。
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                with cpu_omp_manager.configure_omp_envs(
                    rank=global_rank, local_rank=local_rank
                ):
                    unready_worker_handle = WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                        is_driver_worker=is_driver_worker,
                        inherited_fds=inherited_fds,
                    )
                unready_workers.append(unready_worker_handle)
                # [CN] 把本 worker 的 pipe fd 也记进去，供后续 worker 关闭。
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # [CN] 必须先建完所有进程再 wait_for_ready：worker 的 init_device 会做设备同步，
            # [CN] 若改成「建一个等一个」，多卡场景下会死锁。
            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # [CN] worker 已继承了自己的线程数设置；本进程只做调度，
            # [CN] 保留 torch intra-op 并行只会和 worker 抢 CPU。
            # The workers have inherited their thread count (see
            # set_multiprocessing_worker_envs); this process only schedules, so
            # it gets no benefit from torch intra-op parallelism, just CPU
            # contention with them.
            set_torch_threads_for_runtime()

            # [CN] 后台监控线程：任一 worker 猝死则整体关停并回调 EngineCore。
            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            # [CN] 回传 MQ 列表。跨节点时远端 worker 的 MQ 由 worker[0] 代理持有。
            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # [CN] 就绪等待顺序是父子间的隐式协议（先输入 MQ 后输出 MQ），
            # [CN] 改动必须与 WorkerProc.worker_main 中的顺序严格对应，否则死锁。
            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            # [CN] FIFO 队列：RPC 结果按发出顺序消费。
            self.futures_queue = deque[FutureWrapper]()

            self._post_init_executor()

            success = True
        # [CN] 失败清理：先关 death_writer 通知子进程退出，再强制终止。
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        # [CN] 只有 output_rank 这个 worker 会回传 ModelRunnerOutput。
        self.output_rank = self._get_output_rank()

    # [CN] -1 表示「所有 rank 都要回」，否则只取指定 rank 的 MQ。
    def get_response_mqs(self, unique_reply_rank: int = -1) -> list[MessageQueue]:
        assert unique_reply_rank >= -1 and unique_reply_rank < self.world_size, (
            f"unique_reply_rank must be -1 or < world_size,"
            f"unique_reply_rank = {unique_reply_rank}, "
            f"world_size={self.world_size}"
        )
        ranks = (
            [unique_reply_rank] if unique_reply_rank != -1 else range(self.world_size)
        )
        return [self.workers[rank].worker_response_mq for rank in ranks]

    # [CN] 解析 TP/PP/PCP 规模并算出本节点 world_size。
    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    # [CN] 每个 TP 组的第一个 rank 是 driver（负责采样等只需一份的工作）。
    def _is_driver_worker(self, rank: int) -> bool:
        return rank % self.parallel_config.tensor_parallel_size == 0

    # [CN] 监控线程用 weakref 持有 self，避免因为被线程引用而无法回收。
    def start_worker_monitor(self, inline=False) -> None:
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        # [CN] 等任一进程 sentinel 变为 ready，即代表有子进程退出。
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            # [CN] 正在正常关停时不算故障，直接返回。
            if not _self or getattr(_self, "shutting_down", False):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            _self.is_failed = True
            proc = next(h.proc for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly (exit code: %s), "
                "shutting down executor.",
                proc.name,
                proc.exitcode,
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        # [CN] inline 模式：同步执行监控逻辑（测试或单步调试用）。
        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    # [CN] 若已处于失败态，回调立即触发，而不是存起来永不执行。
    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    # [CN] 执行一步。带超时（VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS）防止 worker 挂死拖垮引擎。
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
            ec_output_aggregator=self.ec_output_aggregator,
        )

    # [CN] 采样阶段：async scheduling 下与 execute_model 分离，可重叠 CPU 调度。
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
            ec_output_aggregator=self.ec_output_aggregator,
        )

    # [CN] 跑空批次：DP wave 同步或保持通信活跃，不需要结果。
    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    # [CN] 草稿 token 只需从 output_rank 取一份（所有 TP rank 内容相同）。
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    # [CN] 多进程 RPC 主实现：广播请求 -> 收集响应 -> （可选）聚合。
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
        ec_output_aggregator: ECOutputAggregator | None = None,
    ) -> Any:
        """Returns single result if unique_reply_rank and/or an output
        aggregator is provided, otherwise list."""
        # [CN] follower 节点没有广播 MQ，不该走到这里。
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        # [CN] deadline 用 monotonic 时钟，避免系统时间跳变影响超时判定。
        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # [CN] 有聚合器时改为「所有 rank 都回，再聚合」，因为 KV/EC 状态分散在各 worker。
        aggregators = [a for a in (kv_output_aggregator, ec_output_aggregator) if a]
        aggregate: Callable[[Any], Any]
        if aggregators:
            output_rank = None

            # [CN] 多个聚合器串行作用在 outputs[rank] 上（各自就地合并），因此链式调用是安全的。
            def _aggregate(outputs: Any) -> Any:
                # Each aggregator merges its own connector's output onto
                # outputs[output_rank] in place and returns it, so chaining is safe.
                rank = unique_reply_rank or 0
                result = outputs[rank]
                for a in aggregators:
                    result = a.aggregate(outputs, output_rank=rank)
                return result

            aggregate = _aggregate
        else:
            output_rank = unique_reply_rank
            aggregate = lambda x: x

        # [CN] 方法名走字符串；若传的是可调用对象则用 cloudpickle 序列化后发过去。
        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        # [CN] 一条消息广播给所有 worker；output_rank 随消息下发，worker 据此决定是否回传。
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        # [CN] 优化：指定了 output_rank 时只监听那一个 MQ，省去 N-1 次 dequeue。
        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        # [CN] 惰性取响应：non_block 模式下这段逻辑被包进 Future 延后执行。
        def get_response():
            responses = []
            for mq in response_mqs:
                # [CN] 每次 dequeue 都重算剩余超时，保证总耗时不超 deadline。
                dequeue_timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                # [CN] worker 侧异常以 FAILURE 状态回传，这里转成 RuntimeError 抛出。
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        # [CN] 统一走 FutureWrapper：同步模式就立刻 result()，异步模式把 Future 交回上层。
        future = FutureWrapper(
            self.futures_queue, get_response=get_response, aggregate=aggregate
        )

        return future if non_block else future.result()

    # [CN] 三级关停：优雅等待 -> SIGTERM -> SIGKILL，确保不留僵尸进程。
    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        # [CN] 解释器后期可能把 time 置为 None（模块清理），这里做兼容判断。
        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        initial_count = len(active_procs())

        # [CN] 先给 VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS 秒优雅退出时间。
        # Give processes time to clean themselves up properly first
        logger.info(
            "[shutdown] Executor: waiting for worker exit count=%d",
            initial_count,
        )
        if wait_for_termination(
            active_procs(), timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        ):
            logger.info_once("[shutdown] Executor: all workers exited gracefully")
            return

        # [CN] 优雅期过后发 SIGTERM，再等 4 秒。
        # Send SIGTERM if still running
        remaining = active_procs()
        logger.warning(
            "[shutdown] Executor: workers still running after grace period; "
            "sending SIGTERM count=%d",
            len(remaining),
        )
        for p in remaining:
            p.terminate()
        # [CN] 仍不退出才 SIGKILL —— 强杀可能留下未清理的共享内存，是最后的兜底。
        if not wait_for_termination(active_procs(), 4):
            # Send SIGKILL if still running
            remaining = active_procs()
            logger.warning(
                "[shutdown] Executor: workers still running after SIGTERM; "
                "sending SIGKILL count=%d",
                len(remaining),
            )
            for p in remaining:
                p.kill()

    # [CN] 关停：先关 death_writer（通知子进程）-> 等终止 -> 逐个关 MQ。
    # [CN] 顺序不能反：先关 MQ 会让子进程在读取时报错。
    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, "shutting_down", False):
            worker_count = len(getattr(self, "workers", None) or [])
            logger.debug(
                "[shutdown] Executor: start worker_count=%d",
                worker_count,
            )
            self.shutting_down = True

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                for w in workers:
                    # Shutdown response queues
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

        logger.debug_once("[shutdown] Executor: complete")

    # [CN] 健康检查也是一次 RPC：worker 卡住时这里会超时。
    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    # [CN] 取「最后一个 PP stage 的第一个 TP worker」作为输出 rank。
    # [CN] 例：TP=8、PP=4 时 world_size=32，输出 rank = 32 - 8 = 24（即 PP=3 的 TP=0）。
    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True


# [CN] 就绪前的进程句柄：只有 proc / rank / ready_pipe / death_writer。
@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


# [CN] 就绪后的进程句柄：额外持有回传 MQ。
@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    # [CN] worker_response_mq：单节点模式下本进程写、执行器读。
    # The worker process writes to this MQ in single-node mode
    worker_response_mq: MessageQueue | None
    # [CN] peer_worker_response_mqs：多节点时，远端 rank i 的 MQ 由 driver 节点代理持有。
    # This is only non empty on driver node,
    # the peer worker process i writes to MQ
    # `peer_worker_response_mqs[i]`
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


# [CN] 子进程侧：真正跑 Worker 的地方。以下所有代码都在 worker 进程内执行。
class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    # [CN] 建 MQ。单节点用共享内存直连；多节点要借助 DP world group 做跨节点广播。
    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        # [CN] 多节点：广播器需要等真正的 writer 就绪，这里用 blocking=False 避免卡死。
        else:
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # Since there is external_writer_handle from executor proc,
                # where the ready signal from actual writer is sent out of the
                # create_mq_broadcaster method and after this setup, we make it
                # non blocking. The handshake will be triggered when
                # worker.rpc_broadcast_mq.wait_until_ready() is called
                blocking=False,
            )
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            # that include handles for all ranks
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    # [CN] worker 初始化：建 wrapper -> init_device -> load_model -> 建 MQ。
    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        self.rank = rank
        # [CN] 注意 rpc_rank 用 local_rank，global_rank 用全局 rank（多节点下不同）。
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # [CN] init_device 之后再更新进程名，因为并行组此时才建立。
        # Load model
        self.worker.init_device()
        # Update process title now that parallel groups are initialized
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            self.worker.load_model()

        # [CN] async scheduling：起一个专门的线程把输出拷贝回主进程，
        # [CN] 让 worker 主循环不必等待拷贝完成即可继续下一步。
        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # [CN] 块大小依赖 attention 后端，必须在后端确定后再回填。
        # Set block size based on the attention backends
        current_platform.update_block_size_for_backend(vllm_config)

        # [CN] MQ 必须在 init_device 之后建：多节点场景需要先完成分布式组初始化。
        # Initialize message queues after init_device() since multi-node setups
        # (nnodes_within_dp > 1) require distributed groups to be initialized
        self._init_message_queues(input_shm_handle, vllm_config)

        # [CN] 此后不再允许改环境变量，开启缓存以消除 envs 查询开销。
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    # [CN] 父进程侧：拉起一个 worker 子进程并返回未就绪句柄。
    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # [CN] ready_pipe：子 -> 父，用于汇报「我加载完了」并回传 MQ handle。
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # [CN] death_pipe：父 -> 子。父进程一旦退出，pipe 关闭，子进程收到 EOFError 即自杀。
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        # [CN] 把本 worker 的 fd 也加进继承列表：让「后建的 worker」能关掉「先建 worker」的 fd。
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # Run EngineCore busy loop in background process.
        # [CN] daemon=True：父进程退出时子进程会被强制回收，避免残留。
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        # [CN] NUMA 绑定：在 start() 前后设置 CPU 亲和性，对多路 CPU 性能影响明显。
        # Apply NUMA binding if configured
        with numa_utils.configure_subprocess(
            vllm_config, local_rank, process_kind="worker"
        ):
            proc.start()

        # [CN] 父进程关掉子进程的那一端；death_writer 必须留着（父死则 pipe 关闭）。
        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    # [CN] 从就绪消息里取出 MQ handle 并重建 MessageQueue。
    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    # [CN] 等待所有 worker 就绪：用 connection.wait 同时监听多个 pipe，避免顺序等待。
    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        # [CN] 循环直到所有 pipe 都收到消息（或某个 pipe EOF，说明子进程崩了）。
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                # [CN] EOFError 说明子进程在就绪前就退出了，把原始异常透出（抑制上下文避免噪音）。
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    # [CN] worker 侧关停：关 MQ -> 关 worker -> 销毁并行组与分布式环境。
    def shutdown(self):
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    # [CN] 监听 death_pipe：父进程消失时主动关掉自己的 MQ 并请求退出。
    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        if death_pipe is None:
            return

        # [CN] recv() 会一直阻塞；父进程退出导致 pipe 关闭时抛 EOFError，即触发清理。
        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                # This will block until parent process exits (pipe closes)
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        # [CN] 直接传 queue 引用而不传 self：避免线程持有 self 导致的 gc 问题。
        # Pass queue references directly to avoid gc issues if passing self
        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    # [CN] 子进程入口（static method，作为 Process 的 target）。
    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker initialization and execution loops.
        This runs a background process"""

        # [CN] 信号处理：SIGTERM/SIGINT 转成 SystemExit，且只抛一次（避免关停流程被打断）。
        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # [CN] 尽早发布逻辑->物理 GPU 映射，后面的 set_worker_net_device 依赖它。
        # Publish the logical-to-physical mapping early so topology helpers
        # work before init_device (needed by set_worker_net_device below).
        assigned_physical_gpu_ids = kwargs[
            "vllm_config"
        ].parallel_config.assigned_physical_gpu_ids
        if assigned_physical_gpu_ids is not None:
            from vllm.platforms.interface import set_assigned_physical_gpu_ids

            set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        set_worker_net_device(kwargs.get("local_rank", 0), kwargs["vllm_config"])

        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # [CN] 显式关闭继承来的 fd。fork 下若不关，隐藏引用会让 pipe 永不 EOF，进程退不出去。
        # Close inherited pipes from parent (incl. other worker pipes)
        # Explicitly passing in existing pipes and closing them makes the pipe
        # behave when using fork. Otherwise, a hidden reference to the pipes
        # exist in the child process and prevents EOF closure.
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        # [CN] 主体：构造 WorkerProc -> 发 READY -> 等 MQ 就绪 -> 进入忙循环。
        try:
            # Initialize tracer
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None
            if kwargs["vllm_config"].parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(f"Worker_{worker.rank}")

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # [CN] 只有全部加载完成才发 READY，避免父进程过早开始下发请求。
            # Send READY once we know everything is loaded
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # [CN] 与执行器侧的等待顺序严格对应：先输入 MQ，后输出 MQ。改顺序即死锁。
            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        # [CN] 忙循环异常：通过 MQ 回传 FAILURE 通知执行器，触发整体关停。
        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            # [CN] ready_writer 还没关说明「启动阶段就失败了」，属于启动错误而非运行错误。
            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: exiting after shutdown request"
                )
            else:
                logger.exception("WorkerProc failed.")

            # [CN] 置位 shutdown_requested：避免 __del__ 里再抛 SystemExit 引发 zmq 异常。
            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested.set()

        # [CN] SystemExit 必须继续往外抛，不能被吞掉。
        except SystemExit as e:
            # SystemExit is raised on SIGTERM or SIGKILL, which usually indicates that
            # the graceful shutdown process did not succeed
            if shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: terminated by shutdown signal"
                )
            else:
                logger.warning("WorkerProc was terminated")
            # SystemExit must never be ignored
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    # [CN] 响应状态：worker 侧异常以 FAILURE + 字符串形式回传（异常对象无法跨进程序列化）。
    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    # [CN] 输出入队：异步句柄就地取结果，异常转成 FAILURE 元组。
    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as e:
                logger.exception("Error getting async model runner output")
                output = e

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    # [CN] async scheduling 下交给后台线程拷贝，否则直接入队（阻塞当前步）。
    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    # [CN] 输出拷贝线程的入口。
    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""

        # [CN] 新线程不会继承主线程的 CUDA 上下文，若不在 worker 设备上创建上下文，
        # [CN] 会在 device 0 上隐式建一份，白白多占显存。
        # set device to the worker device for the thread.
        # a thread will not inherit the context of the main thread.
        # when calling any cuda runtime functions, it will implicitly
        # create a new cuda context on device 0, consuming extra memory.
        # here we set the device to the worker device for the thread,
        # enforcing the context to be the same as the main thread.
        from vllm.platforms import current_platform

        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    # [CN] 忙循环：无限取 RPC 请求并执行。indefinite=True 表示无超时阻塞。
    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            self._execute_worker_rpc(self.rpc_broadcast_mq.dequeue(indefinite=True))

    # [CN] 执行单条 RPC。单独抽成方法是为了让 dequeue 循环保持简单（便于排查栈）。
    def _execute_worker_rpc(
        self,
        rpc_request: tuple[str | bytes, tuple[Any, ...], dict[str, Any], int | None],
    ) -> None:
        """Execute one RPC in a separate frame from the dequeue loop."""
        method, args, kwargs, output_rank = rpc_request
        try:
            # [CN] 字符串走 getattr；bytes 走 cloudpickle 反序列化并注入 self（即 worker）。
            if isinstance(method, str):
                func = getattr(self.worker, method)
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), self.worker)

            output = func(*args, **kwargs)

            # [CN] 只有 output_rank 匹配（或未指定）时才回传，避免 N 份重复输出。
            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)
        # [CN] 异常也要回传（转成 FAILURE），否则执行器会一直等下去直到超时。
        except Exception as e:
            # Notes have been introduced in python 3.11
            # [CN] add_note 是 Python 3.11+ 的能力：把完整栈附加到异常上再回传。
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            logger.exception("WorkerProc hit an exception.")
            # enqueue_output converts the exception to a FAILURE response
            # containing its string representation before transport.
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)

    # [CN] 设置进程名与日志前缀，便于在多 worker 日志里区分（Worker_TP0/PP1 等）。
    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        # [CN] 并行组未初始化时拿不到 rank 信息，只能用通用名。
        # Check if parallel groups are initialized first
        if not model_parallel_is_initialized():
            # Parallel groups not yet initialized, use default process name
            set_process_title(name="Worker")
            decorate_logs("Worker")
            return

        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


# [CN] 在 fork/spawn 之前设置好 worker 的线程数环境变量。
def set_multiprocessing_worker_envs(local_world_size: int = 1):
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    if current_platform.is_cpu() or "OMP_NUM_THREADS" in os.environ:
        return

    # [CN] 线程数必须「启动前」定好：torch.set_num_threads 会立即建线程池，
    # [CN] 若在 worker 启动到一半时调用，会与 dlopen 竞争，fork 模式下甚至死锁（libgomp 非 fork-safe）。
    # Choose the workers' thread count here, before they start, since a worker
    # must not set its own: `torch.set_num_threads()` spawns the thread pool
    # eagerly, and doing that part way through a worker's startup either races
    # the dlopen of shared objects or, in a forked worker, deadlocks (libgomp
    # is not fork-safe).
    num_threads = startup_omp_num_threads(local_world_size)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ[OMP_NUM_THREADS_SET_BY_VLLM] = "1"

    # [CN] spawn 的子进程 import torch 时从环境变量读；fork 的子进程直接继承父进程的设置。
    # [CN] 只要父进程在 fork 前没有真正跑过并行区，就不会死锁。
    # A spawned worker picks the count up from the environment when it imports
    # torch. A forked worker instead inherits it from this process, so set it
    # here too. This is safe as long as we don't *use* the pool before forking:
    # a forked child whose parent had run a parallel region deadlocks, whereas
    # one whose parent merely sized the pool does not.
    torch.set_num_threads(num_threads)
    logger.debug(
        "Set OMP_NUM_THREADS=%d for %d worker process(es).",
        num_threads,
        local_world_size,
    )
