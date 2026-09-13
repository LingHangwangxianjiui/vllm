# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：DBO（Dual Batch Overlap，双批次重叠）的线程同步原语。
# [CN] 要解决的问题：小 batch 时 GPU 算力吃不满，因为 attention 之外的算子
# [CN]   （尤其是 TP/EP 下的 all-reduce / all-to-all）会让 GPU 空等通信。
# [CN] 解法：把一步拆成 2 个微批次（ubatch），跑在 2 个 CPU 线程上，
# [CN]   线程 A 做计算时线程 B 做通信，从而把通信藏在计算背后。
# [CN] 关键实现：
# [CN]   - 两个 CUDA stream：compute_stream（算）与 comm_stream（通信）；
# [CN]   - 两个 threading.Event 做 CPU 侧「接力棒」，保证同一时刻只有一个线程跑；
# [CN]   - 两个 torch.Event 做 GPU 侧跨 stream 依赖（comm_done / compute_done）。
# [CN] 本文件是全局状态的中心：
# [CN]   _THREAD_ID_TO_CONTEXT：线程 id -> ubatch id，供算子查询「我在哪个 ubatch」；
# [CN]   _CURRENT_CONTEXTS：ubatch id -> UBatchContext；
# [CN]   _NUM_UBATCHES：默认 2，由 make_ubatch_contexts 写入。
# [CN] 最容易看错的点：dbo_* 系列函数在「非 DBO 模式」下是彻底的 no-op
# [CN]   （_register_ubatch_function 里 if 不成立就什么都不做），
# [CN]   所以同一个模型代码可以不用改就能在 DBO 开/关两种模式下运行。

import threading

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

# [CN] 三个模块级全局量。注意 _NUM_UBATCHES 硬编码为 2 只是为了初始化，
# [CN] 真正的值在 make_ubatch_contexts 里被覆盖。
_THREAD_ID_TO_CONTEXT: dict = {}
# Here we hardcode the number of microbatches to 2 for default.
_NUM_UBATCHES: int = 2
_CURRENT_CONTEXTS: list["UBatchContext | None"] = []


# [CN] 一个 ubatch 的运行时上下文：持有自己的 stream、ForwardContext 与同步原语。
# [CN] 每个 ubatch 有独立的 ForwardContext，因为 attention metadata 不同。
class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """

    def __init__(
        self,
        id: int,
        comm_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        gpu_comm_done_event: torch.Event,
        gpu_compute_done_event: torch.Event,
        schedule: str = "default",
    ):
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.forward_context = forward_context
        self.ready_barrier = ready_barrier
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.current_stream = compute_stream
        self.gpu_comm_done_event = gpu_comm_done_event
        self.gpu_compute_done_event = gpu_compute_done_event
        self.schedule = schedule
        self.recv_hook = None

    # [CN] 进入上下文：登记线程->ubatch 映射，然后等 ready_barrier（两个线程都就位）。
    # [CN] 之后阻塞在 cpu_wait_event 上，等前一个 ubatch 把接力棒递过来。
    def __enter__(self):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        # _NUM_UBATCHES is set in make_ubatch_contexts
        self.ready_barrier.wait()

        # [CN] 拿到接力棒后立刻 clear，为下一轮做准备（Event 是一次性的）。
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        # Assume we want to start on the compute stream
        self.update_stream(self.compute_stream)
        return self

    # [CN] 退出上下文：先把可能挂着的 recv_hook 跑掉，再把接力棒交给下一个线程。
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    # [CN] 关键：forward_context._forward_context 是模块级全局，
    # [CN] 两个线程轮番上台时必须各自恢复自己的那份，否则 metadata 会串。
    def _restore_context(self):
        forward_context._forward_context = self.forward_context

    # [CN] 切换当前 CUDA stream（仅在真的不同时才调 set_stream，省一次驱动调用）。
    def update_stream(self, stream):
        self.current_stream = stream
        if current_stream() != self.current_stream:
            torch.cuda.set_stream(self.current_stream)

    # [CN] 四个 GPU 事件原语：record 打点、wait_event 等待，构成跨 stream 依赖边。
    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    # [CN] CPU 侧让出：把接力棒交给另一个线程，自己睡下。
    # [CN] 三个 assert 是正确性护栏 —— DBO 的正确性前提是「同一时刻只有一个线程在跑」，
    # [CN] 如果 assert 触发，说明有代码路径绕过了 yield 直接并发执行。
    def _cpu_yield(self):
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    # [CN] 只切 stream 不打点：调用方自己保证没有跨 stream 依赖。
    def switch_to_comm(self):
        self.update_stream(self.comm_stream)

    def switch_to_compute(self):
        self.update_stream(self.compute_stream)

    # [CN] 切 stream 并插入依赖边：先 record 自己完成，再让对方 stream 等这个事件。
    # [CN] 顺序很重要 —— 必须先 update_stream 之外的 record，否则点打错 stream 上。
    def switch_to_comm_sync(self):
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self):
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    # [CN] recv_hook：由通信库注册的「真正开始收数据」回调，
    # [CN] 推迟到下一个 ubatch 上台时才触发，从而把通信延迟藏进别人的计算里。
    def maybe_run_recv_hook(self):
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    # [CN] 纯让出（不换 stream）：让另一个线程在同一 stream 上继续。
    def yield_(self):
        self.current_stream = current_stream()
        self._cpu_yield()
        self.update_stream(self.current_stream)

    # [CN] 计算->通信的切换并让出：典型用法是 all-reduce 前把控制权交给对方。
    def yield_and_switch_from_compute_to_comm(self):
        assert current_stream() == self.compute_stream
        self._signal_compute_done()
        self._cpu_yield()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        assert current_stream() == self.comm_stream
        self._signal_comm_done()
        self._cpu_yield()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        self._wait_comm_done()


# [CN] 判断是否处于 DBO 模式：只要登记表非空就说明有 ubatch 在跑。
def dbo_enabled() -> bool:
    return len(_THREAD_ID_TO_CONTEXT) > 0


# [CN] 查询「当前线程属于第几个 ubatch」。非 DBO 模式返回 0。
def dbo_current_ubatch_id() -> int:
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


# [CN] 装饰器工厂：把 UBatchContext 的方法包装成「按当前线程自动派发」的全局函数。
# [CN] 非 DBO 模式下静默跳过 —— 这是让模型代码无需分支的关键。
def _register_ubatch_function(func):
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)

    return wrapper


# [CN] 导出的 dbo_* 全局函数族：模型/通信代码只需调用它们，不感知 UBatchContext。
dbo_maybe_run_recv_hook = _register_ubatch_function(UBatchContext.maybe_run_recv_hook)
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_compute_to_comm
)
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_comm_to_compute
)
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
dbo_switch_to_compute = _register_ubatch_function(UBatchContext.switch_to_compute)
dbo_switch_to_comm_sync = _register_ubatch_function(UBatchContext.switch_to_comm_sync)
dbo_switch_to_compute_sync = _register_ubatch_function(
    UBatchContext.switch_to_compute_sync
)


# [CN] 注册到「下一个」ubatch 上（取模回环），实现「我算的时候你收数据」。
def dbo_register_recv_hook(recv_hook):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
        next_ctx.recv_hook = recv_hook


# [CN] 在正确的 stream 上执行事件操作：离开当前上下文时调用方可能拿错 stream。
def dbo_get_previous_event(func, *args, **kwargs):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        # execute callable on the ubatch compute stream to record/wait events there
        with torch.cuda.stream(ctx.compute_stream):
            return func(*args, **kwargs)


# [CN] 构造 N 个 UBatchContext。cpu_signal_event 取「下一个」的 wait_event，
# [CN] 从而形成一个环形的接力链：0 叫醒 1，1 叫醒 0。
def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[UBatchContext]:
    global _NUM_UBATCHES, _CURRENT_CONTEXTS
    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"

    _NUM_UBATCHES = num_micro_batches
    # Ensure the global context list is large enough
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    """
    Create a context manager for micro-batching synchronization.
    """
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
        )
        ctxs.append(ctx)

    return ctxs
