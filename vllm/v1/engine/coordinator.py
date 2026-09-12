# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

# [CN] 文件总览：**数据并行（DP>1）协调器**。
#
#     为什么需要它：DP 部署下有 N 个独立的 EngineCore 进程，每个都有自己的
#     调度器和 KV cache。前端要把请求发给「最闲」的那个，就得先知道各引擎的
#     负载 —— 但前端和引擎之间没有两两连接，于是需要一个中间人。
#
#     ========================= 它的两件事 =========================
#     ① **负载统计中转**：收集各引擎的 [waiting, running, kv_usage]，
#        聚合成一张表，定期广播给所有前端，供负载均衡决策。
#     ② **wave（波）协调**：这是 DP 下最容易被忽略的正确性机制。
#
#     ========================= 什么是 wave =========================
#     MoE 模型的 DP 是**同步（lockstep）**的：所有 rank 必须一起做 prefill
#     或一起做 decode，因为 MoE 的 all-to-all 需要所有 rank 同时参与。
#     于是引擎们整体在「运行 / 暂停」两个全局状态之间切换，
#     每次「运行 → 暂停」记一次，计数就是 wave 号。
#     这个切换是靠 DPEngineCoreProc._has_global_unfinished_reqs 里的
#     all-reduce 来同步的。
#
#     需要协调器介入的**竞态**：所有引擎都暂停了，但某个前端正要给
#     rank 3 发一个新请求 —— 如果只唤醒 rank 3，其他 rank 会一直等，
#     形成死锁。所以前端在发请求时会顺带通知协调器，由它广播
#     START_DP_WAVE 把所有引擎一起叫醒。
#
#     代码组织：DPCoordinator（父进程侧的句柄）→ EngineState →
#     DPCoordinatorProc（真正跑在独立进程里的主体）。

import multiprocessing
import multiprocessing.connection
import time
import weakref

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


# [CN] 父进程侧的**句柄**：负责拉起协调器进程、拿到它的 ZMQ 地址，
#      并把地址暴露给前端与引擎。真正的逻辑在 DPCoordinatorProc 里。
class DPCoordinator:
    """Coordinator process used for data-parallel deployments (DP>1).

    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * Collects stats from each DP engine (currently just waiting and running
      queue lengths), and publishes these to all front-ends for use in
      load-balancing decisions.

    * Keeps track of the current DP "request wave" number and running state
      of the engines. This is received from the DP rank 0 engine and published
      to the front-end processes along with the current load stats.

      The engines alternate between a global running/paused state. The global
      "request wave" number is a count of the number of times that the workers
      collectively move from a running state to a paused state. This transition
      is synchronized via the all-reduce operation performed in the
      DPEngineCoreProc._has_global_unfinished_reqs method.

    * Broadcasts the START_DP_WAVE message to engines to move them from paused
      to running state when one engine receives a new request. This can happen
      in two cases:
      1) A front-end sending a new request while the engines are paused will
         concurrently notify the coordinator.
      2) An engine receiving a request for a stale request wave while in paused
         state will notify the coordinator.

    Engines will move into running state when receiving a new request or
    START_DP_WAVE message.

    Note that when deployed in External LB mode, no stats will be published by
    the engines and thus updates will only be sent to front-ends when the
    request wave / running state changes.
    """

    # [CN] 等子进程把实际绑定的 ZMQ 地址通过 pipe 送回来。
    #      为什么要这一步：地址可能是 IPC 路径或自动分配的 TCP 端口，
    #      只有 bind 之后才知道，所以必须由子进程回传。
    #      同时监听 proc.sentinel —— 子进程启动就挂了的话能立刻发现。
    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
        try:
            timeout = 120
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=timeout
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    f"within timeout={timeout} seconds during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    # [CN] 构造并拉起协调器进程。
    def __init__(
        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True
    ):
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        # [CN] 三个地址分别用于：前端订阅统计（XPUB）、
        #      引擎推送输出（PULL）、协调器广播控制消息（XPUB）。
        #      local_only 决定是否走 IPC（同机）还是 TCP（跨机）。
        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        local_only = not parallel_config.local_engines_only
        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        # [CN] 弹性 EP 会从「节点内」扩展到「跨节点」，
        #      所以引擎侧的地址不能限定为本机。
        # NOTE(yongji): handling scaling from intra-node to inter-node
        if parallel_config.enable_elastic_ep:
            local_only_eng = False

        front_publish_address = get_engine_client_zmq_addr(local_only, host=host)
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host=host)
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host=host)

        # [CN] 用 pipe 回传地址，子进程 daemon=True（随父进程退出）。
        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "enable_wave_coordination": enable_wave_coordination,
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        # [CN] weakref.finalize：对象被 GC 时也能确保子进程被收掉。
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown coordinator process with configurable timeout."""
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


# [CN] 单个引擎的负载快照：[waiting 数, running 数, KV cache 使用率]。
class EngineState:
    def __init__(self):
        # [waiting, running, kv_cache_usage]
        self.request_counts: list[int | float] = [0, 0, 0.0]


# [CN] 协调器**进程**的主体（跑在独立进程里）。
class DPCoordinatorProc:
    # [CN] engine_count 个引擎，每个一个 EngineState。
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination

    # [CN] 进程入口（staticmethod 是因为要作为 Process 的 target 被 pickle）。
    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    # [CN] ==================== 协调器主循环 ====================
    #      三个 socket：
    #        publish_front —— XPUB，向前端广播（负载表 / wave 状态），
    #                        同时**接收**前端的订阅与新请求通知；
    #        publish_back  —— XPUB，向引擎广播控制消息（READY / START_DP_WAVE）；
    #        output_back   —— PULL，接收引擎推来的统计与 wave 事件。
    #      用 XPUB 而不是 PUB 就是为了能收到订阅消息（订阅帧 b'\x01'+topic / 取消订阅帧 b'\x00'+topic）。
    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)

        # [CN] wave 状态：当前波号 + 引擎是否处于运行态。
        # For tracking request wave progression.
        current_wave = 0
        engines_running = False

        # [CN] 统计发布节流：stats_changed 时才高频发（100ms），
        #      否则 5 秒发一次心跳。
        # For tracking request counts for internal load-balancing.
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int | float]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            # [CN] 把三个 socket 实际绑定的地址回传给父进程。
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()
            # [CN] 等所有引擎都订阅上来（每个引擎发一个 \x01），
            #      然后广播 READY —— 引擎收到 READY 才开始干活。
            #      这是一个简单的启动屏障。
            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            # [CN] 三个 socket 一起 poll。
            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(publish_back, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            # [CN] ---------- 主循环 ----------
            while True:
                # [CN] 计算本轮 poll 的超时：
                #      有变更 → 按 stats_update_interval_ms；否则 5 秒。
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # [CN] lockstep（MoE）DP 下，各 rank 的 step 是同步的，
                #      所以要**至少等 50ms** 才能收齐本 step 所有引擎的统计；
                #      非 lockstep 的引擎没有统一 step 边界，不必等。
                # Wait at least 50ms to ensure we've received all stats for
                # the current step. Only applicable to lockstep (MoE) DP;
                # non-lockstep engines have no synchronized step boundaries.
                if self.enable_wave_coordination and last_step_counts is None:
                    min_timeout = 50
                else:
                    min_timeout = 0

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                # [CN] poll 超时 = 该发一次统计广播了。
                #      优先发「step 边界快照」（last_step_counts），
                #      没有就发现场值。
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    # [CN] 广播三元组：(各引擎负载表, 当前 wave, 是否运行中)。
                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                # [CN] 引擎侧 socket 有消息：只可能是「新引擎订阅」。
                if publish_back in events:
                    buffer = publish_back.recv()
                    # [CN] 弹性 EP 扩容的新引擎会在这里订阅。
                    #      直接回 READY，而不等 SCALE_ELASTIC_EP 通知 —— 
                    #      后者要等新引擎**初始化完成**才发，太晚；
                    #      订阅消息则是初始化期间就发的。
                    if buffer == b"\x01":
                        # NOTE(yongji): newly started engine subscribed
                        # We need to send READY message here instead of receiving
                        # SCALE_ELASTIC_EP notification from engine core client
                        # as SCALE_ELASTIC_EP is only sent when
                        # new engines finished initialization.
                        # Subscription message, on the other hand, is sent
                        # by each engine during initialization
                        publish_back.send(b"READY")
                    elif buffer != b"\x00":
                        logger.error(
                            "DP Coordinator received unexpected message from engines"
                        )

                # [CN] 前端侧 socket 有消息：订阅消息忽略，
                #      其余是「我要发新请求」的通知或扩容通知。
                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # Ignore subscription messages.
                        continue

                    # [CN] 扩容 / 缩容通知：调整 engines 列表长度。
                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        # Handle scale up notification
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            # [CN] 新引擎的 wave 从 0 开始，
                            #      可能与现有引擎的 wave 不一致，
                            #      这里只记日志，实际靠后面的 START_DP_WAVE 收敛。
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        continue  # Skip normal engine notification processing

                    # [CN] **核心竞态处理**：前端要在引擎暂停时发新请求。
                    #      此时必须把其他引擎一起唤醒，否则发到的那个 rank
                    #      会独自空转、其他 rank 一直等（死锁）。
                    # Wave coordination: handle new-request messages from front-end.
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
                        # We received a message on the front-end XPUB socket,
                        # from an API server sending a new request while the
                        # engines are paused, so that we can wake the other
                        # engines.
                        # [CN] decoded = (要排除的引擎 index, wave)。
                        #      那个引擎马上就会收到真实请求，不必再通知它。
                        engine_to_exclude, wave = decoded
                        if not engines_running:
                            # [CN] wave 号过期（stale）说明消息滞后，
                            #      此时**不能**排除任何引擎 —— 
                            #      必须保证所有引擎都能收到唤醒。
                            if wave < current_wave:
                                # If the wave number is stale, ensure the message
                                # is handled by all the engines.
                                engine_to_exclude = None

                            # [CN] 注意 engines_running 只由引擎自己的通知来置位；
                            #      暂停中的引擎会直接丢弃 START_DP_WAVE，
                            #      所以「发过 START_DP_WAVE」不等于「引擎已经跑起来了」。
                            # engines_running is only set from the engines'
                            # own notifications; a paused engine discards
                            # START_DP_WAVE, so sending it is not evidence
                            # that the engines are running.
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                # [CN] 引擎推来的消息：统计 + wave 事件。
                if output_back in events:
                    # We received a message from one of the engines.

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    # [CN] 协调器只收**控制类**消息，不转发实际输出，
                    #      所以 outputs 与 utility_output 必须为空。
                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats
                    # [CN] ① 更新该引擎的负载快照。
                    if scheduler_stats:
                        # Elastic EP stats may arrive while the engine list changes.
                        # [CN] 弹性 EP 下 engines 列表可能在统计到达前就变了，
                        #      越界就丢弃这条。
                        if eng_index >= len(self.engines):
                            continue
                        # 1. Updated request load stats - update our local
                        # state with these.
                        stats = self.engines[eng_index].request_counts
                        # [CN] lockstep DP：所有 rank 的 step 同步，
                        #      所以在 **step 边界**做一次快照，
                        #      这样前端看到的负载表才是「同一时刻」的。
                        if self.enable_wave_coordination:
                            # Steps are synchronized across lockstep (MoE) DP
                            # ranks; snapshot counts at step boundaries.
                            stats_step = scheduler_stats.step_counter
                            stats_wave = scheduler_stats.current_wave
                            # [CN] 只在 (wave, step) 单调前进时才推进。
                            if (
                                stats_wave > last_stats_wave
                                or stats_wave == last_stats_wave
                                and stats_step > last_stats_step
                            ):
                                # [CN] 已有变更就先把上一 step 的快照存下来，
                                #      避免刚跨过边界就把新数据混进去。
                                if stats_changed:
                                    last_step_counts = self._get_engine_counts(
                                        do_copy=True
                                    )
                                last_stats_step = stats_step
                                last_stats_wave = stats_wave
                            # [CN] 收到乱序的 step → 记警告（可能意味着
                            #      某个 rank 跑偏了），但不更新状态。
                            elif stats_wave != last_stats_wave or (
                                stats_step != last_stats_step
                            ):
                                logger.warning(
                                    "Received stats for out-of-order "
                                    "step (%d, %d) from engine %d (expected "
                                    "> (%d, %d))",
                                    stats_wave,
                                    stats_step,
                                    eng_index,
                                    last_stats_wave,
                                    last_stats_step,
                                )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats[2] = scheduler_stats.kv_cache_usage
                        stats_changed = True

                    # [CN] ② wave 事件处理。
                    # Wave coordination: handle wave completion and start notifications
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
                        # [CN] rank 0 报告「本波结束」→ 全体进入暂停态，
                        #      wave 号 +1。
                        if (wave := outputs.wave_complete) is not None:
                            # 2. Notification from rank 0 engine that we've
                            # moved into the global paused state
                            # (engines_running==False).
                            if current_wave <= wave:
                                new_wave = wave + 1
                                logger.debug(
                                    "Moving DP wave from %d to %d.",
                                    current_wave,
                                    new_wave,
                                )
                                current_wave = new_wave
                                engines_running = False
                                wave_state_changed = True
                        # [CN] 某引擎收到了一个「不属于当前 wave」的请求
                        #      （竞态：它还没收到本波的 START_DP_WAVE），
                        #      此时协调器必须把大家推进到这个新 wave。
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            # 3. The engine received request for a non-current wave
                            # so we must ensure that other engines progress to the
                            # next wave (race condition handling).
                            logger.debug(
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(publish_back, wave, eng_index)

                # [CN] wave 状态一变就**立即**广播（不等统计节流），
                #      因为前端要靠它决定请求该发到哪个 wave。
                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    # [CN] 广播 START_DP_WAVE。带上 exclude_engine_index：
    #      那个引擎已经要被真实请求唤醒了，不必重复通知。
    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None
    ):
        """Broadcast the START_DP_WAVE message to all the engines.
        It includes the current wave number and index of engine which
        has already received a request with this wave number and so doesn't
        require additional notification.
        """
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    # [CN] 取各引擎负载表。do_copy=True 用于做 step 边界快照，
    #      否则返回的是活引用（下一拍会被就地修改）。
    def _get_engine_counts(self, do_copy=False) -> list[list[int | float]]:
        """Return list of [waiting, running] count lists for each engine."""
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
