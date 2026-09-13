# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：**V1 层的通用工具箱**。
#
#     它不承担某一块业务逻辑，而是被 engine 各处反复复用的零散能力。
#     按用途可以分成四组：
#
#     ---------------- ① 进程管理（本文件的重头）----------------
#     · APIServerProcessManager：起 N 个 API server 子进程并监控；
#     · RustFrontendProcessManager：起一个 Rust 前端子进程；
#     · _SubprocessWrapper：把 subprocess.Popen 包装成 BaseProcess 的模样，
#       这样监控代码可以统一处理两种进程；
#     · shutdown / _shutdown_subprocesses：优雅关停（SIGTERM → 等待 → 强杀）；
#     · wait_for_completion_or_failure：主进程的主循环，任一子进程挂了就退出。
#
#     ---------------- ② 数据结构 ----------------
#     · ConstantList：只读列表视图，任何写操作都抛 TypeError。
#       用途是把内部可变 list 安全地暴露给下游（防止被调用方改坏）。
#     · CpuGpuBuffer：常驻的 CPU(pinned) + GPU 双缓冲，用于高频小拷贝。
#     · IterationDetails / compute_iteration_details：一步调度的统计口径。
#
#     ---------------- ③ 小工具 ----------------
#     · record_function_or_nullcontext：可开关的 profiler 打点；
#     · tensor_data：把张量摊成 uint8 内存视图，用于哈希/序列化；
#     · copy_slice / get_engine_client_zmq_addr 等。
#
#     ---------------- ④ 使用统计上报 ----------------
#     · report_usage_stats：把匿名配置信息上报（受环境变量开关控制）。
#
#     ===================== 一个贯穿全文件的技巧 =====================
#     **端口回填**：ZMQ 地址里的端口常常先写成 0，等子进程真正 bind 之后
#     才知道内核分配了哪个端口。子进程通过 Pipe 把真实地址回报给父进程，
#     父进程在 gather_actual_addresses() 里收集。
import argparse
import contextlib
import json
import multiprocessing
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
    overload,
)

import torch
import uvloop
from torch.autograd.profiler import record_function

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils.network_utils import get_open_zmq_ipc_path, get_tcp_uri
from vllm.utils.system_utils import decorate_logs, kill_process_tree, set_process_title
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    import numpy as np

    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.engine.utils import CoreEngineActorManager, CoreEngineProcManager

logger = init_logger(__name__)

T = TypeVar("T")


# [CN] 只读列表视图：包一层 list，把所有会改动内容的操作都变成抛异常。
#      典型用法是「把内部状态安全地暴露出去」—— 下游能读能遍历，
#      但改不动，避免被调用方无意写坏。
class ConstantList(Generic[T], Sequence):
    # [CN] 注意是**持有引用**而不是拷贝：外部改原 list 仍会影响这里。
    def __init__(self, x: list[T]) -> None:
        self._x = x

    # [CN] 以下所有 mutator 一律抛 TypeError（与 tuple 的行为对齐）。
    def append(self, item):
        raise TypeError("Cannot append to a constant list")

    def extend(self, item):
        raise TypeError("Cannot extend a constant list")

    def insert(self, item):
        raise TypeError("Cannot insert into a constant list")

    def pop(self, item):
        raise TypeError("Cannot pop from a constant list")

    def remove(self, item):
        raise TypeError("Cannot remove from a constant list")

    def clear(self):
        raise TypeError("Cannot clear a constant list")

    # [CN] 只读但允许查询的方法正常转发。
    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T): ...

    @overload
    def __setitem__(self, s: slice, value: T, /): ...

    def __setitem__(self, item: int | slice, value: T | list[T]):
        raise TypeError("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise TypeError("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"

    # [CN] copy() 是**唯一**被允许的「出口」：想要可变副本就显式拷一份。
    def copy(self) -> list[T]:
        return self._x.copy()


# [CN] CPU(pinned) + GPU 常驻双缓冲。
#      为什么要有它：每步调度都要往 GPU 搬一些小张量（block table、
#      slot mapping 等）。反复临时分配既慢又会产生显存碎片，
#      所以预先分配好，之后只做 copy_。
class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    # [CN] 构造：一次性把两侧缓冲都分好。
    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = PIN_MEMORY,
        with_numpy: bool = True,
    ) -> None:
        # [CN] 这两个缓冲是**可变运行时状态**，必须在 inference_mode 之外分配，
        #      否则会拿到不可写的 inference tensor。
        # these buffers are mutable runtime state, so allocate them as normal
        with torch.inference_mode(False):
            self.cpu = torch.zeros(
                *size, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self.gpu = torch.zeros_like(self.cpu, device=device)
        # [CN] numpy 视图是**可选**的：为了不让类型注解变复杂（避免泛型/子类），
        #      这里按需创建属性。with_numpy=False 时访问 .np 会 AttributeError。
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        if with_numpy:
            # [CN] bfloat16 无法直接转 numpy，明确报错而不是留给后面崩。
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            # [CN] numpy 视图与 cpu 张量**共享内存**，一边改另一边可见（零拷贝）。
            self.np = self.cpu.numpy()

    # [CN] CPU → GPU。n 可指定只搬前 n 个（避免搬整块）。
    #      non_blocking=True：异步拷贝，需要时调用方自己做同步。
    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    # [CN] GPU → CPU。同样是非阻塞的，返回后数据**未必**已就绪，
    #      必须显式同步（torch.cuda.synchronize 等）才能读。
    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


# [CN] 生成引擎 ↔ 客户端的 ZMQ 地址：能走本机就用 IPC（更快，不占端口），
#      否则用 TCP。port=0 表示让内核在 bind 时分配。
def get_engine_client_zmq_addr(
    local_only: bool,
    host: str,
    port: int = 0,
) -> str:
    """Return an IPC path (``local_only=True``) or ``tcp://host:port``.

    ``port=0`` lets the kernel assign the port at ``bind()`` time; the
    caller must recover it via ``getsockopt(zmq.LAST_ENDPOINT)``."""
    if local_only:
        return get_open_zmq_ipc_path()
    return get_tcp_uri(host, port)


# [CN] **API server 进程组管理器**：起 N 个 API server 子进程并监控其存活。
class APIServerProcessManager:
    """Manages a group of API server processes.

    Handles creation, monitoring, and termination of API server worker
    processes. Also monitors extra processes to check if they are healthy.
    """

    def __init__(
        self,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        target_server_fn: Callable | None = None,
        stats_update_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        """Initialize and start API server worker processes.

        ``input_addresses``/``output_addresses`` may contain
        ``tcp://host:0`` placeholders; each child must report the actual
        bound endpoint over its ``actual_address_pipe`` in ``client_config``
        and the parent collects them via
        :py:meth:`gather_actual_addresses`.

        Args:
            target_server_fn: Override function to call for each API server process
            listen_address: Address to listen for client connections
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            input_addresses: Input addresses for each API server
            output_addresses: Output addresses for each API server
            stats_update_address: Optional stats update address
            tensor_queue: Optional tensor IPC queue for sharing MM tensors
        """
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        # [CN] 显式用 spawn：fork 与 CUDA/多线程并存时容易出问题。
        spawn_context = multiprocessing.get_context("spawn")
        self.processes: list[BaseProcess] = []
        # [CN] 每个子进程一条 Pipe，用于回报它实际 bind 的地址。
        self._address_pipes: list[connection.Connection] = []

        # [CN] 逐个 API server 建进程。client_index 让子进程知道自己是第几个。
        for i, in_addr, out_addr in zip(
            range(num_servers), input_addresses, output_addresses
        ):
            client_config: dict[str, Any] = {
                "input_address": in_addr,
                "output_address": out_addr,
                "client_count": num_servers,
                "client_index": i,
            }
            # [CN] 可选字段：统计订阅地址、张量 IPC 队列。
            if stats_update_address is not None:
                client_config["stats_update_address"] = stats_update_address
            if tensor_queue is not None:
                client_config["tensor_queue"] = tensor_queue

            # [CN] 单向 Pipe：子写父读。
            parent_recv, child_send = spawn_context.Pipe(duplex=False)
            self._address_pipes.append(parent_recv)
            client_config["actual_address_pipe"] = child_send

            # [CN] 子进程入口是 run_api_server_worker_proc。
            proc = spawn_context.Process(
                target=target_server_fn or run_api_server_worker_proc,
                name=f"ApiServer_{i}",
                args=(listen_address, sock, args, client_config),
            )
            self.processes.append(proc)
            proc.start()

            # [CN] 关键：父进程立刻关掉自己的写端。
            #      否则子进程即使挂了，父进程读 Pipe 也不会收到 EOF，
            #      只能一直干等到超时。
            # Drop parent's write end so reader sees EOF on child death.
            child_send.close()

        logger.info("Started %d API server processes", len(self.processes))

        # Shutdown only the API server processes on garbage collection
        # The extra processes are managed by their owners
        # [CN] GC 兜底关停。注意只管 API server 自己起的进程，
        #      外部传进来的进程由各自的 owner 负责。
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    # [CN] 收集各子进程回报的**真实**绑定地址（端口回填的那一环）。
    #      同时在子进程提前退出时立刻报错，而不是傻等到超时。
    def gather_actual_addresses(
        self,
        timeout: float = envs.VLLM_ENGINE_READY_TIMEOUT_S,
    ) -> tuple[list[str], list[str]]:
        """Return (inputs, outputs) reported by each child, indexed by
        ``client_index``. Raises ``RuntimeError`` on timeout or premature
        child exit."""
        n = len(self._address_pipes)
        inputs: list[str | None] = [None] * n
        outputs: list[str | None] = [None] * n
        # [CN] pending: Pipe → 下标，表示「还在等谁回报」。
        pending: dict[connection.Connection, int] = {
            pipe: i for i, pipe in enumerate(self._address_pipes)
        }
        # [CN] sentinel → 下标，用于发现「子进程直接退出了」的情况。
        sentinel_to_idx: dict[Any, int] = {
            proc.sentinel: i for i, proc in enumerate(self.processes)
        }

        # [CN] 统一截止时间，所有等待共享同一个预算。
        deadline = time.monotonic() + timeout
        try:
            # [CN] 主循环：还有没回报的就继续等。
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [self.processes[i].name for i in pending.values()]
                    raise RuntimeError(
                        f"Timed out after {timeout:.1f}s waiting for "
                        f"API server(s) to report bound ZMQ addresses: "
                        f"{missing}"
                    )
                # [CN] 同时等两类对象：Pipe（有消息）和 sentinel（进程退出）。
                waitables: list[Any] = list(pending.keys()) + list(
                    sentinel_to_idx.keys()
                )
                ready = connection.wait(waitables, timeout=remaining)
                # [CN] **先处理 Pipe 再处理 sentinel**（顺序很关键）：
                #      一个子进程可能「发完消息后立刻退出」，此时同一次 poll 里两个事件
                #      都会就绪。必须先记下它的成功回报，否则会被误判成「未回报就退出」。
                # Drain pipes before checking sentinels: a child that sent
                # its message and then exited can surface both events in
                # the same poll, and we must record the success first.
                for item in ready:
                    if isinstance(item, connection.Connection) and item in pending:
                        idx = pending.pop(item)
                        try:
                            msg: dict[str, str] = item.recv()
                        # [CN] EOF 说明子进程没写任何东西就关了 Pipe（通常是启动就崩了）。
                        except EOFError as e:
                            raise RuntimeError(
                                f"API server {self.processes[idx].name} "
                                f"closed its address pipe without "
                                f"reporting its bound ZMQ addresses"
                            ) from e
                        inputs[idx] = msg["input_address"]
                        outputs[idx] = msg["output_address"]
                        item.close()
                # [CN] 处理退出事件：如果这个进程还没回报过地址，就是启动失败。
                for item in ready:
                    if item in sentinel_to_idx:
                        idx = sentinel_to_idx.pop(item)
                        pipe = self._address_pipes[idx]
                        # [CN] 已经回报过地址了才退出 —— 那是正常退出，不报错。
                        if pipe in pending:
                            proc = self.processes[idx]
                            raise RuntimeError(
                                f"API server process {proc.name} exited "
                                f"(code={proc.exitcode}) before reporting "
                                f"its bound ZMQ addresses"
                            )
        # [CN] 无论成功失败，都把剩余 Pipe 关掉，避免 fd 泄漏。
        finally:
            for pipe in pending:
                with contextlib.suppress(Exception):
                    pipe.close()

        return inputs, outputs  # type: ignore[return-value]

    # [CN] 关停：先关 Pipe，再走通用的进程关停流程。
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown API server processes with configurable timeout"""
        for pipe in self._address_pipes:
            with contextlib.suppress(Exception):
                pipe.close()
        self._address_pipes = []

        # [CN] detach 掉 GC 兜底，避免重复关停。
        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)


# [CN] **Rust 前端进程管理器**：拉起 vllm-rs 二进制的 frontend 子命令。
#      对外暴露与 APIServerProcessManager 相同的接口，便于上层统一处理。
class RustFrontendProcessManager:
    """Manages a single Rust frontend subprocess.

    Launches the Rust vllm-rs binary in 'frontend' mode, passing the
    listening socket fd and ZMQ transport addresses. Provides the same
    interface as APIServerProcessManager for process monitoring.
    """

    def __init__(
        self,
        binary_path: str,
        sock: Any,
        args: argparse.Namespace,
        input_address: str,
        output_address: str,
        engine_start_index: int,
        engine_count: int,
        data_parallel_size: int,
        stats_update_address: str | None = None,
    ):
        import os
        import subprocess

        # [CN] 把监听 socket 的 fd 设为可继承，直接传给子进程（fd 传递）。
        #      这样父进程可以先把端口 bind 好、甚至做完 graceful restart，
        #      再把 fd 交给子进程，避免端口抢占和连接丢失。
        fd = sock.fileno()
        os.set_inheritable(fd, True)

        cmd = [
            binary_path,
            "frontend",
            "--listen-fd",
            str(fd),
            "--input-address",
            input_address,
            "--output-address",
            output_address,
            "--engine-start-index",
            str(engine_start_index),
            "--engine-count",
            str(engine_count),
            "--data-parallel-size",
            str(data_parallel_size),
        ]
        if stats_update_address is not None:
            cmd.extend(["--coordinator-address", stats_update_address])
        from vllm.entrypoints.serve.utils.api_utils import jsonify_non_default_args

        args_dict = jsonify_non_default_args(
            args,
            exclude={
                "api_server_count",
                # Python passes the bootstrapped engine range explicitly.
                "data_parallel_rank",
                "data_parallel_external_lb",
                "data_parallel_hybrid_lb",
            },
        )
        # [CN] Rust 侧用 serde_json 解析 --args-json，会**绕过 clap**，
        #      因此 `#[arg(env = ...)]` 声明的环境变量默认值不会被应用。
        #      所以这里把几个由环境变量决定的值**显式**塞进 JSON，
        #      保证 Python 前端与 Rust 前端行为一致。
        # The Rust `frontend` subcommand parses --args-json via serde_json,
        # which bypasses clap and therefore ignores any `#[arg(env = ...)]`
        # declarations on SharedRuntimeArgs fields. Forward the env-driven
        # values explicitly so VLLM_ENGINE_READY_TIMEOUT_S and
        # VLLM_HTTP_TIMEOUT_KEEP_ALIVE behave the same on both Python and Rust
        # frontends.
        args_dict["engine_ready_timeout_secs"] = envs.VLLM_ENGINE_READY_TIMEOUT_S
        args_dict["http_timeout_keep_alive"] = envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE
        args_json = json.dumps(args_dict, sort_keys=True)
        cmd.extend(["--args-json", args_json])

        # [CN] 传给子进程的是真实参数，但**日志里必须脱敏**（api_key、hf_token 等）。
        # The subprocess needs the real values, but the log must not carry
        # credentials such as api_key or hf_token.
        from vllm.entrypoints.serve.utils.api_utils import redact_sensitive_args

        redacted_json = json.dumps(redact_sensitive_args(args_dict), sort_keys=True)
        logger.info("Launching Rust frontend: %s", " ".join(cmd[:-1] + [redacted_json]))
        # [CN] pass_fds 让子进程继承监听 fd。
        self._proc = subprocess.Popen(cmd, pass_fds=(fd,))

        # [CN] 包一层 wrapper，让它有 sentinel，从而能参与统一的进程监控。
        # Create a process wrapper with a sentinel fd for monitoring
        self.processes: list[_SubprocessWrapper] = [
            _SubprocessWrapper(self._proc, "RustFrontend")
        ]

        self._finalizer = weakref.finalize(self, _shutdown_subprocesses, self.processes)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._finalizer.detach() is not None:
            _shutdown_subprocesses(self.processes, timeout=timeout)


# [CN] 把 subprocess.Popen 包装成 BaseProcess 的模样。
#      目的：Popen 没有 sentinel，而监控代码（connection.wait）依赖它，
#      所以这里自己造一个「Pipe 版 sentinel」。
class _SubprocessWrapper:
    """Wraps subprocess.Popen to provide the BaseProcess-like interface
    needed by wait_for_completion_or_failure."""

    def __init__(self, proc, name: str):
        self._proc = proc
        self.name = name
        self.pid = proc.pid
        self._sentinel_conn: connection.Connection | None = None
        self._sentinel_send: connection.Connection | None = None

        # [CN] 用 Pipe 造 sentinel：写端一关，读端就可读。
        #      相比直接用 pid，这种做法跨平台且能被 connection.wait 统一等待。
        # Use a Pipe-based sentinel so subprocess monitoring works uniformly
        # across platforms with multiprocessing.connection.wait().
        recv, send = connection.Pipe(duplex=False)
        self._sentinel_conn = recv
        self._sentinel_send = send

        # [CN] 监控线程：等子进程结束，然后关掉写端 —— 读端随即变为可读。
        def monitor_subprocess() -> None:
            try:
                proc.wait()
            finally:
                with contextlib.suppress(Exception):
                    send.close()

        # [CN] daemon 线程，不阻塞主进程退出。
        threading.Thread(
            target=monitor_subprocess, daemon=True, name=f"{name}Monitor"
        ).start()

    # [CN] 对外暴露的 sentinel 就是那个 Pipe 的读端。
    @property
    def sentinel(self):
        return self._sentinel_conn

    @property
    def exitcode(self) -> int | None:
        return self._proc.returncode if self._proc.poll() is not None else None

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self):
        self._proc.terminate()

    def join(self, timeout=None):
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=timeout)

    # [CN] 析构时把 Pipe 两端都关掉，避免 fd 泄漏。
    def __del__(self):
        with contextlib.suppress(Exception):
            if self._sentinel_conn is not None:
                self._sentinel_conn.close()
            if self._sentinel_send is not None:
                self._sentinel_send.close()


# [CN] 关停 subprocess 包装的进程（shutdown() 的 Popen 版本）。
def _shutdown_subprocesses(
    procs: list[_SubprocessWrapper], timeout: float | None = None
) -> None:
    """Shutdown subprocess wrappers (mirrors the shutdown() function)."""
    # [CN] 给一个最小的宽限时间，但至少 5 秒 —— 太短会让进程来不及清理。
    if timeout is None:
        timeout = 0.0
    timeout = max(timeout, 5.0)

    logger.debug(
        "[shutdown] Subprocess manager: start process_count=%d timeout=%ss",
        len(procs),
        timeout,
    )

    # [CN] 先给所有进程发 SIGTERM（让它们有机会优雅退出）。
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # [CN] 再在剩余时间内逐个 join。**共享同一个 deadline**，
    #      不会因为前面进程慢而让总时间无限拉长。
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    # [CN] 超时还没退的，直接杀进程树（连同它自己起的孙进程）。
    remaining_pids = [
        proc.pid for proc in procs if proc.is_alive() and proc.pid is not None
    ]
    if remaining_pids:
        logger.warning(
            "[shutdown] Subprocess manager: force killing remaining processes count=%d",
            len(remaining_pids),
        )
    for pid in remaining_pids:
        kill_process_tree(pid)

    logger.debug_once("[shutdown] Subprocess manager: complete")


# [CN] API server 子进程的入口函数。
def run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Entrypoint for individual API server worker processes."""

    from vllm.entrypoints.launchers.api_server.entry import run_server_worker

    client_config = client_config or {}
    server_index = client_config.get("client_index", 0)

    # [CN] 改进程名 + 给日志加前缀：多进程下排查问题时能分清是谁打的日志。
    # Set process title and add process-specific prefix to stdout and stderr.
    set_process_title("APIServer", str(server_index))
    decorate_logs()

    # [CN] 用 uvloop 跑 asyncio 服务（比默认事件循环快）。
    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs)
    )


# [CN] 主进程的主循环：**等任一进程退出，非正常退出就抛异常**。
#      这是「一个挂了就全部收摊」的 fail-fast 策略 —— 半死不活的
#      服务比直接挂掉更难排查。
def wait_for_completion_or_failure(
    api_server_manager: "APIServerProcessManager | RustFrontendProcessManager",
    engine_manager: Union["CoreEngineProcManager", "CoreEngineActorManager"]
    | None = None,
    coordinator: "DPCoordinator | None" = None,
) -> None:
    """Wait for all processes to complete or detect if any fail.

    Raises an exception if any process exits with a non-zero status.

    Args:
        api_server_manager: The manager for API servers.
        engine_manager: The manager for engine processes.
            If CoreEngineProcManager, it manages local engines;
            if CoreEngineActorManager, it manages all engines.
        coordinator: The coordinator for data parallel.
    """

    try:
        logger.info("Waiting for API servers to complete ...")
        # Create a mapping of sentinels to their corresponding processes
        # for efficient lookup
        # [CN] 建立 sentinel → 进程的映射，一次 wait 就能盯住所有进程。
        sentinel_to_proc: dict[Any, BaseProcess | _SubprocessWrapper | None] = {
            proc.sentinel: proc for proc in api_server_manager.processes
        }

        if coordinator:
            sentinel_to_proc[coordinator.proc.sentinel] = coordinator.proc

        # [CN] 引擎进程另起线程监控（它有自己的探活逻辑）。
        #      这里用一根 Pipe 把「引擎监控结束」也变成一个可等待的 sentinel。
        if engine_manager:
            core_shutdown_recv, core_shutdown_send = connection.Pipe(duplex=False)

            def monitor_engines():
                try:
                    engine_manager.monitor_engine_liveness()
                finally:
                    core_shutdown_send.close()
                    core_shutdown_recv.close()

            # start monitor for engine liveness
            threading.Thread(target=monitor_engines, daemon=True).start()
            # [CN] 值设成 None 是刻意的：它只是个「引擎侧结束」的信号，
            #      不代表某个具体进程退出。
            sentinel_to_proc[core_shutdown_recv] = None  # type: ignore[assignment]

        # Check if any process terminates
        # [CN] 只要还有进程在跑就继续等。
        while sentinel_to_proc:
            # Wait for any process to terminate (or engine shutdown signal)
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc)

            # Process any terminated processes
            for sentinel in ready_sentinels:
                proc = sentinel_to_proc.pop(sentinel)

                # Check if process exited with error
                # [CN] 非零退出码 → 认为失败，立刻抛异常终止整个服务。
                if proc is not None and proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) "
                        f"died with exit code {proc.exitcode}"
                    )
                # [CN] 引擎侧的失败由 manager 记录名字，这里转成异常。
                if engine_manager and engine_manager.failed_proc_name is not None:
                    raise RuntimeError(
                        f"Engine core process {engine_manager.failed_proc_name} "
                        "died unexpectedly."
                    )

    # [CN] Ctrl-C 属正常退出，不抛异常。
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        logger.exception("Exception occurred while running API servers: %s", str(e))
        raise


# [CN] 注意：shutdown **不能写成实例方法**，否则实例会被 finalizer 的引用
#      链一直拽着，GC 永远回收不了这个对象（英文原注释就是这个意思）。
# Note(rob): shutdown function cannot be a bound method,
# else the gc cannot collect the object.
def shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
    """Shutdown processes with timeout.

    Args:
        procs: List of processes to shutdown
        timeout: Maximum time in seconds to wait for graceful shutdown
    """
    # [CN] 未指定超时时给 5 秒，作为尽力而为的清理窗口。
    if timeout is None:
        # Keep a small grace period for best-effort cleanup paths that do not
        # have a user-configured shutdown timeout.
        timeout = 5.0

    logger.debug(
        "[shutdown] Process manager: start process_count=%d timeout=%ss names=%s",
        len(procs),
        timeout,
        (",").join([proc.name for proc in procs]),
    )

    # Shutdown the process.
    # [CN] 第一步：给所有还活着的进程发 SIGTERM。
    for proc in procs:
        if proc.is_alive():
            logger.info(
                "[shutdown] Process manager: send sigterm to process %s", proc.name
            )
            proc.terminate()

    # Allow time for remaining procs to terminate.
    # [CN] 第二步：在共享 deadline 内逐个 join。
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    # [CN] 第三步：仍存活的强制杀进程树。
    remaining_procs = [
        (proc.pid, proc.name)
        for proc in procs
        if proc.is_alive() and proc.pid is not None
    ]
    if remaining_procs:
        logger.warning(
            "[shutdown] Process manager: force killing remaining processes count=%d",
            len(remaining_procs),
        )
    for pid, proc_name in remaining_procs:
        logger.warning(
            "[shutdown] Process manager: force killing remaining process %s pid %d",
            proc_name,
            pid,
        )
        kill_process_tree(pid)

    logger.debug_once("[shutdown] Process manager: complete")


# [CN] 把源张量的前 length 个元素**异步**拷进目标张量。
#      典型场景：pinned CPU 张量 → 预分配的 GPU 张量。
def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    # [CN] non_blocking 需要 pin_memory 才真正异步；返回的是目标张量的切片。
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


# [CN] 上报匿名使用统计（可通过环境变量关闭）。
def report_usage_stats(
    vllm_config, usage_context: UsageContext = UsageContext.ENGINE_CONTEXT
) -> None:
    """Report usage statistics if enabled."""

    # [CN] 未开启统计就直接返回，一行都不多做。
    if not is_usage_stats_enabled():
        return

    from vllm.model_executor.model_loader import get_architecture_class_name

    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config
    parallel_config = vllm_config.parallel_config
    attention_config = vllm_config.attention_config
    compilation_config = vllm_config.compilation_config
    speculative_config = vllm_config.speculative_config

    # [CN] KV connector 是可选功能，没启用就是 None。
    # Prepare KV connector string if applicable
    kv_connector = None
    if vllm_config.kv_transfer_config is not None:
        kv_connector = vllm_config.kv_transfer_config.kv_connector

    # [CN] backend=None 表示 auto（运行时按平台选择），如实上报。
    # Attention backend is None when set to "auto" (resolved at runtime per platform).
    attention_backend = (
        attention_config.backend.name if attention_config.backend is not None else None
    )

    # CompilationMode is an IntEnum; report the name for readability in dashboards.
    compilation_mode = (
        compilation_config.mode.name if compilation_config.mode is not None else None
    )

    # [CN] 未启用推测解码时这些字段为 None。
    # Speculative decoding fields default to None when spec decode is disabled.
    spec_decode_method = (
        speculative_config.method if speculative_config is not None else None
    )
    num_speculative_tokens = (
        speculative_config.num_speculative_tokens
        if speculative_config is not None
        else None
    )

    # [CN] Transformers 后端要额外显示被包装的原始架构名，便于区分。
    if model_config.using_transformers_backend():
        backend_cls = model_config._model_info.architecture
        # Show what was wrapped e.g. TransformersForCausalLM(Starcoder2ForCausalLM)
        architecture = f"{backend_cls}({model_config.architectures[0]})"
    else:
        architecture = get_architecture_class_name(model_config)

    # [CN] 真正上报。所有字段都刻意选了「运维常调的旋钮」，
    #      既有用又不涉及任何用户数据。
    usage_message.report_usage(
        architecture,
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype": str(model_config.dtype),
            "block_size": vllm_config.cache_config.block_size,
            "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,
            "kv_cache_memory_bytes": vllm_config.cache_config.kv_cache_memory_bytes,
            # Quantization
            "quantization": model_config.quantization,
            "kv_cache_dtype": str(vllm_config.cache_config.cache_dtype),
            # Feature flags
            "enable_lora": bool(vllm_config.lora_config),
            "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager": model_config.enforce_eager,
            "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
            # Distributed parallelism settings
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "enable_expert_parallel": parallel_config.enable_expert_parallel,
            # All2All backend for MoE expert parallel
            "all2all_backend": parallel_config.all2all_backend,
            # KV connector used
            "kv_connector": kv_connector,
            # Batching limits — tuning knobs operators commonly override
            "max_model_len": model_config.max_model_len,
            "max_num_seqs": scheduler_config.max_num_seqs,
            "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
            # Attention backend (user-requested; None = auto-selected at runtime)
            "attention_backend": attention_backend,
            # torch.compile mode (e.g. NONE, STOCK_TORCH_COMPILE, VLLM_COMPILE)
            "compilation_mode": compilation_mode,
            # Speculative decoding configuration
            "spec_decode_method": spec_decode_method,
            "num_speculative_tokens": num_speculative_tokens,
            # Wide expert parallel: load balancer + redundant/total expert counts
            "enable_eplb": parallel_config.enable_eplb,
            "num_redundant_experts": parallel_config.eplb_config.num_redundant_experts,
            "num_experts": model_config.get_num_experts(),
        },
    )


# [CN] 全局缓存：profiler 打点函数只在首次调用时决定一次。
_PROFILER_FUNC = None


# [CN] 返回一个「可能打点、也可能是空上下文」的上下文管理器。
#      为什么这么绕：打点代码会散布在热路径上，
#      必须保证**未开启 profiler 时开销几乎为零** —— 所以用缓存 + nullcontext，
#      避免每次都去查环境变量。
def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    global _PROFILER_FUNC

    # [CN] 快路径：已经确定过了，直接用缓存的 func。
    # fast path assume it is set
    if _PROFILER_FUNC is not None:
        return _PROFILER_FUNC(name)

    # [CN] 默认什么都不做（nullcontext 的开销可以忽略）。
    func = contextlib.nullcontext
    # [CN] 按环境变量选择 torch profiler 或 NVTX。
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        func = record_function
    elif envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        import nvtx

        func = nvtx.annotate

    _PROFILER_FUNC = func
    return func(name)


# [CN] 取张量的原始字节视图（uint8），常用于序列化与哈希。
def tensor_data(tensor: torch.Tensor) -> memoryview:
    """Get the raw data of a tensor as a uint8 memoryview, useful for
    serializing and hashing.

    Args:
        tensor: The input tensor.

    Returns:
        A memoryview of the tensor data as uint8.
    """
    # [CN] flatten → 搬到 CPU → 连续化 → 按 uint8 重新解释 → numpy → memoryview。
    #      每一步都是视图操作（不算拷贝），只有 .cpu() 会真正搬数据。
    return tensor.flatten().cpu().contiguous().view(torch.uint8).numpy().data


# [CN] 一步调度的统计口径：区分「prefill（ctx）」和「decode（生成）」。
@dataclass
class IterationDetails:
    num_ctx_requests: int
    num_ctx_tokens: int
    num_generation_requests: int
    num_generation_tokens: int
    num_encoder_inputs: int = 0
    num_encoder_output_tokens: int = 0

    def __repr__(self) -> str:
        return f"IterationDetails(num_ctx_requests={self.num_ctx_requests},\
                 num_ctx_tokens={self.num_ctx_tokens}, \
                 num_generation_requests={self.num_generation_requests}, \
                 num_generation_tokens={self.num_generation_tokens}, \
                 num_encoder_inputs={self.num_encoder_inputs}, \
                 num_encoder_output_tokens={self.num_encoder_output_tokens})"


# [CN] 统计一步里有多少请求/token 属于 prefill、多少属于 decode。
def compute_iteration_details(scheduler_output: SchedulerOutput) -> IterationDetails:
    """
    Compute the number of context/generation requests and tokens
    for the current iteration's scheduler output. A requests is regarded
    as a context request if its output tokens are still 0, an extended chunk
    of chunked prefill falls into this category.

    Args:
        scheduler_output: The scheduler output for the current iteration.

    Returns:
        An IterationDetails object containing the number of
        context/generation requests and tokens.
    """
    num_context_requests = 0
    num_context_tokens = 0
    num_generation_requests = 0
    num_generation_tokens = 0
    # [CN] 新请求一定处于 prefill 阶段。
    new_req_ids = {new_req.req_id for new_req in scheduler_output.scheduled_new_reqs}
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        # [CN] 关键：正在**分块 prefill** 的旧请求也仍算 ctx，
        #      不能因为「它不是新请求」就误判成 decode，否则统计会失真。
        if scheduler_output.scheduled_cached_reqs.is_context_phase(req_id) or (
            req_id in new_req_ids
        ):
            num_context_requests += 1
            num_context_tokens += num_tokens
        else:
            num_generation_requests += 1
            num_generation_tokens += num_tokens
    # [CN] 多模态 encoder 的输入/输出另外单独统计。
    scheduled_encoder_input_stats = scheduler_output.scheduled_encoder_input_stats
    num_encoder_inputs = 0
    num_encoder_output_tokens = 0
    if scheduled_encoder_input_stats is not None:
        num_encoder_inputs = scheduled_encoder_input_stats.num_inputs
        num_encoder_output_tokens = scheduled_encoder_input_stats.output_tokens

    return IterationDetails(
        num_context_requests,
        num_context_tokens,
        num_generation_requests,
        num_generation_tokens,
        num_encoder_inputs,
        num_encoder_output_tokens,
    )
