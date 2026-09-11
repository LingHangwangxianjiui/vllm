# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kv_events import KVEventsConfig
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


# [CN] 调度器的"暂停"三态（注意与引擎层的 PauseMode 区分）：
#      UNPAUSED  ：正常调度
#      PAUSED_NEW：不再接纳**新**请求，已经在 running 的继续跑（排空）
#      PAUSED_ALL：完全停止调度（连 running 的也不跑）
#      对应到 sleep / 权重更新 / pause_generation 的不同语义。
class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - UNPAUSED: Normal operation
    - PAUSE_NEW: No new requests are scheduled, requests already in
                 running state are scheduled.
    - PAUSE_ALL: No requests are scheduled
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


# [CN] 调度器抽象基类。**读懂 vLLM 调度只需要抓住这个契约**：
#        schedule()            -> 决定这一轮给每个请求算多少 token
#        update_from_output()  -> 拿模型输出回写状态、判定结束
#        add_request() / finish_requests() -> 请求的进与出
#      把它当成"接口文档"读：抽象方法与 @property 就是全部对外能力，
#      具体实现见 sched/scheduler.py（目前唯一实现）。
#
#      为什么要抽象出接口：为了支持自定义调度策略（插件化），
#      以及让测试可以替换实现。
class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    # [CN] 一次 schedule() = **模型的一次前向**。这是理解 vLLM 的关键：
    #      调度不是「给请求分配资源」这种一次性动作，而是每步都要重新问
    #      "这一步算哪些 token"。返回值本质上是 {req_id: num_tokens} 的映射。
    #        - 新请求：num_tokens 可能是整个 prompt（或 chunked prefill 的一部分）
    #        - 解码中：通常是 1（投机解码时可能是 1 + k 个草稿）
    @abstractmethod
    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.

        Args:
            throttle_prefills: DP prefill balancing. When True (set by the DP
                engine core on non-cadence-aligned steps), new prefill compute is
                deferred to a later step so prefills stay aligned across DP ranks;
                automatically overridden when the rank is saturated.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            requests.
        """
        raise NotImplementedError

    # [CN] 算结构化输出（JSON schema / grammar）的位掩码。
    #      之所以单独成一个方法：它在 GPU 跑前向的**同时**在 CPU 上算，
    #      属于典型的"CPU/GPU 并行"优化（见 EngineCore.step）。
    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    # [CN] 前向跑完后回写状态：追加新 token、判定是否结束（长度/stop/abort）、
    #      回收或保留 KV block，并产出要发回前端的输出。
    #      返回 dict[client_index, EngineCoreOutputs]：
    #      scale-out 场景下多个前端连着同一引擎，输出要按来源分组发回。
    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.
        """
        raise NotImplementedError

    # [CN] 投机解码用：把 draft model 生成的草稿 token 记到请求上，
    #      并顺带做 grammar 校验（草稿可能违反 schema，要提前修掉）。
    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """Update requests with newly generated draft token ids, applying
        structured output grammar validation if needed.

        Args:
            draft_token_ids: The input draft token ids for each request.
        """
        raise NotImplementedError

    # [CN] 与上一个方法的区别：这个是在 **async scheduling** 下用的 ——
    #      此时 SchedulerOutput 已经提前构造好了，只能就地补上草稿 token，
    #      而不是等下一轮 schedule()。
    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update scheduler output with newly generated draft token ids, applying
        structured output grammar validation if needed.

        Args:
            draft_token_ids: The input draft token ids for each request.
            scheduler_output: Update the given scheduler_output
                with the corresponding draft token ids.
        """
        raise NotImplementedError

    # [CN] 请求入队（进入 waiting 队列），不等于立刻会跑。
    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        raise NotImplementedError

    # [CN] 结束请求的统一入口。两种触发方式：
    #        1) 客户端主动 abort；
    #        2) 前端 detokenize 后发现了 stop string（引擎侧只看 token，
    #           不知道文本层面有没有命中停止串，所以要由前端回头通知）。
    #      request_ids=None 表示结束全部（关闭/pause 时用）。
    #      返回值只包含"这次真的被结束掉的"请求（已结束的不重复计入）。
    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "RequestStatus",
    ) -> "list[Request]":
        """Finish the requests in the scheduler's internal queue. If the request
        is not in the queue, this method will do nothing for that request.

        This method is called in two cases:
        1. When the request is aborted by the client.
        2. When the frontend process detects a stop string of the request after
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs, or None to finish all.
            finished_status: The finished status of the given requests.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        raise NotImplementedError

    # [CN] 未完成请求数（waiting + running）。
    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in the scheduler's internal queue."""
        raise NotImplementedError

    # [CN] 非抽象方法：直接由上一个抽象方法派生，子类不用重复实现。
    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        return self.get_num_unfinished_requests() > 0

    # [CN] 注意这个语义很微妙：**上一轮刚结束、但还没在 SchedulerOutput 里
    #      通知模型侧清理**的请求。它和 "not has_unfinished_requests()" 不是一回事。
    #      为什么需要这个状态：模型 runner 里有这些请求的缓存状态
    #      （比如 CUDA graph 里的 slot），必须等下一轮输出带出去才能清。
    #      DP attention 场景下这个标志尤其重要（各 rank 要一致地清理）。
    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Returns True if there are finished requests that need to be cleared.
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in the
        previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.
        """
        raise NotImplementedError

    # [CN] "还有事没做完" = 有未完成的请求，或者有已结束但没清理的。
    #      引擎忙循环的 has_work() 最终就落到这个方法上。
    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests() or self.has_finished_requests()

    # [CN] 当前暂停状态（property，因为它是调度器的内部状态快照）。
    @property
    @abstractmethod
    def pause_state(self) -> PauseState:
        """Current pause state of the scheduler."""
        raise NotImplementedError

    # [CN] 设置暂停状态，供引擎层 pause_scheduler 调用。
    @abstractmethod
    def set_pause_state(self, pause_state: PauseState) -> None:
        raise NotImplementedError

    # [CN] 清空前缀缓存。**权重热更新后必须调用** —— 否则新权重会复用
    #      旧权重算出来的 KV，直接产生错误输出。
    #      reset_running_requests=True 时会抢占所有在跑的请求重算；
    #      False 时只有在没有请求占用 KV cache 的情况下才真的清。
    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache for KV cache.

        This is particularly required when the model weights are live-updated.

        Args:
            reset_running_requests: If True, all the running requests will be
                preempted and moved to the waiting queue. Otherwise, this method
                will only reset the KV prefix cache when there is no running request
                taking KV cache.
        """
        raise NotImplementedError

    # [CN] 清多模态 encoder 缓存：同理，权重变了之后旧的视觉 embedding 不能复用。
    @abstractmethod
    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        raise NotImplementedError

    # [CN] (running 数, waiting 数)，用于统计上报和 DP 负载均衡。
    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        raise NotImplementedError

    # [CN] KV cache 使用率 0~1。基类默认 0（不知道），由实现覆盖。
    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        return 0.0

    # [CN] 每步生成一份统计快照（不是累计值），交给日志/监控系统。
    @abstractmethod
    def make_stats(self) -> "SchedulerStats | None":
        """Make a SchedulerStats object for logging.

        The SchedulerStats object is created for every scheduling step.
        """
        raise NotImplementedError

    # [CN] 关闭：释放 KV connector 等外部资源。
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        raise NotImplementedError

    # [CN] 下面三个是"可选能力"的默认空实现：KV/EC 传输连接器、KV 事件发布。
    #      基类返回 None 表示"本调度器不支持"，实现方按需覆盖。
    #      这种"返回 None 表示能力缺失"的写法在 vLLM 里很常见。
    def get_kv_connector(self) -> "KVConnectorBase_V1 | None":
        return None

    def get_ec_connector(self) -> "ECConnectorBase | None":
        return None

    def get_kv_event_publisher_config(self) -> "KVEventsConfig | None":
        return None
