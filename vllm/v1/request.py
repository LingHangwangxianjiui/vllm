# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ==============================================================================
# 本文件职责：定义 V1 引擎里「一次请求」的服务端表示 —— Request 类，以及它的状态枚举
#   RequestStatus。Request 是调度器、KV Cache 管理器、Worker 之间传递的唯一请求载体，
#   请求从进入到结束的全部可变状态（已算到第几个 token、占了多少 block、输出了哪些
#   token、被抢占过几次）都挂在这个对象上。它只存在于 EngineCore 进程内，不会跨进程
#   传输；跨进程传输用的是 vllm/v1/engine/__init__.py 里的 EngineCoreRequest。
#
# 在系统链路中的位置（EngineCore 进程内，调度面）：
#   前端进程：LLM.generate / AsyncLLM → EngineCoreRequest（可序列化）
#     -> ZMQ(或进程内直连) -> EngineCore.add_request()
#     -> 【本文件 Request】由 Request.from_engine_core_request() 构造
#       -> Scheduler.schedule()       决定本轮给它算多少 token
#       -> KVCacheManager             分配 / 复用 block
#     -> SchedulerOutput -> Worker -> ModelRunner 真正执行
#     -> ModelRunnerOutput 回来后 append_output_token_ids() 更新本对象
#     -> 状态落到 FINISHED_* 后由 Scheduler 释放 block 并回收
#
# 核心内容速查：
#   - Request                : 请求主体，本文件 95% 的内容
#   - RequestStatus          : 状态枚举，注意 PREEMPTED 是「未完成/已完成」的分界线
#   - StreamingUpdate        : 流式会话续写时增量更新用的轻量数据类
#   - _FINISHED_REASON_MAP   : 终态 -> 对外暴露的 FinishReason 映射
#
# 请求状态机（理解调度逻辑的关键）：
#   WAITING ──(调度器选中，分到 KV block)──────────> RUNNING
#   WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR ──(语法编译完成)──> WAITING
#   WAITING_FOR_REMOTE_KVS ──(P/D 分离，远端 KV 到达)────> WAITING
#   WAITING_FOR_STREAMING_REQ ──(流式会话下一轮输入到达)──> WAITING
#   RUNNING ──(显存不足被抢占，释放 block)────────> PREEMPTED ──(重新排队)──> WAITING
#   RUNNING ──(正常结束 / 超长 / 被取消 / 报错 / 重复)──> FINISHED_*
#   说明：RequestStatus.is_finished() 的判据是 status > PREEMPTED，
#         所以**枚举里凡是排在 PREEMPTED 之后的都算终态**，加新状态时顺序不能乱放。
#
# 阅读提示（几个容易踩的点）：
#   1. num_computed_tokens 是「乐观计数」：异步调度和 PP 提前跑时，已经派发但还没
#      回结果的步数也算进去了（配合 num_in_flight_tokens 使用），不要当成"已确认算完"。
#   2. _output_token_ids / _all_token_ids 对外只暴露只读视图 output_token_ids /
#      all_token_ids（ConstantList），因为这两个列表必须同步更新，禁止外部直接 append。
#   3. block_hashes 是增量计算的：每凑满一个 block 才算一次 hash 并 append，
#      用于 prefix cache 命中判定。block_hasher 存成 _block_hasher 而非闭包，
#      是为了避免 Request -> partial -> Request 的循环引用导致引用计数无法即时回收。
#   4. 混合输入（同时有 prompt_token_ids 和 prompt_embeds）时，被 embeds 覆盖的位置
#      在 _all_token_ids 里填 0 占位，真正的向量来自 prompt_embeds。
# ==============================================================================
import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.metrics.stats import PrefillStats, RequestSpecDecodeMetrics
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.

    中文说明：流式会话（session）续写用的增量包。流式场景下同一个 request_id 会被
    反复追加新的输入，但不必重建整个 Request —— 只需要把新的 mm_features /
    prompt_token_ids / max_tokens / sampling_params 塞进 Request.streaming_queue，
    由调度器在下一轮取出并合并。注意 arrival_time 会被更新，因此排队优先级会随之变化。

    from_request() 只在 request.resumable 为 True 时返回实例，否则返回 None —— 这是
    「普通请求不进流式队列」的开关。
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    """V1 引擎内一次请求的完整运行时状态。

    在链路中的职责：作为调度器（Scheduler）、KV Cache 管理器（KVCacheManager）与
    Worker 之间唯一共享的请求载体。它同时承担三种身份：
      1) 输入描述：prompt token ids / embeds、多模态特征、采样或池化参数；
      2) 调度账本：status、num_computed_tokens、num_preemptions、block_hashes；
      3) 输出容器：_output_token_ids、stop_reason、events（事件时间线）。

    关键设计：
      - 生命周期只存在于 EngineCore 进程；跨进程传输用 EngineCoreRequest（见
        vllm/v1/engine/__init__.py），两者通过 from_engine_core_request() 转换。
      - 状态迁移集中在 Scheduler 里完成，本类只提供状态字段与少量自维护方法
        （append_output_token_ids / update_block_hashes / take_events）。
      - 列表类字段对外只给只读视图，避免调用方绕过同步更新逻辑。
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        session_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        abort_immediately: bool = False,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
            self.structured_output_request.reasoning_parser_kwargs = (
                reasoning_parser_kwargs
            )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # 初始状态一律是 WAITING；若带结构化输出语法，下面会改成
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR，等语法编译完再回落到 WAITING。
        self.status = RequestStatus.WAITING
        # 事件时间线（QUEUED / SCHEDULED / PREEMPTED ...），由前端输出处理器消费，
        # 用于统计排队耗时等指标；take_events() 取走后会清空。
        self.events: list[EngineCoreEvent] = []
        # 停止原因：命中 stop token 时是 token id，命中 stop string 时是字符串。
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None
        # E/P/D: Connector-specific encoder-cache transfer parameters.
        self.ec_transfer_params: dict[str, Any] | None = None

        # sampling_params 与 pooling_params 互斥：前者是生成式（要吐 token），
        # 后者是池化式（embedding / 分类 / reward），池化模型固定只跑 1 步。
        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                self.ec_transfer_params = sampling_params.extra_args.get(
                    "ec_transfer_params"
                )
                self.kv_cache_report_mode = sampling_params.extra_args.get(
                    "kv_cache_report_mode", "incremental"
                )
            else:
                self.kv_cache_report_mode = "incremental"
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Per-position mask used in mixed-mode (chat completion with
        # prompt_embeds). `None` except when both `prompt_token_ids` and
        # `prompt_embeds` are set and their positions are interleaved.
        self.prompt_is_token_ids = prompt_is_token_ids
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        if self.prompt_token_ids is None:
            self._all_token_ids: list[int] = [0] * self.num_prompt_tokens
        elif self.prompt_is_token_ids is None:
            self._all_token_ids = self.prompt_token_ids.copy()
        else:
            # Mixed-mode prompt: positions covered by prompt_embeds hold a sentinel
            # special token id that may lie outside the embedding. Zero them, matching
            # the no-token-ids case above, so embedding gathers over these placeholder
            # ids stay in bounds; the actual inputs come from prompt_embeds.
            self._all_token_ids = [
                t if is_tok else 0
                for t, is_tok in zip(self.prompt_token_ids, self.prompt_is_token_ids)
            ]

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Tokens of output in flight when the request was preempted: delivered
        # on return, but must not mutate the reset counters.
        # 中文：被抢占那一刻「已经在路上」的输出 token 数。回来时要投递给用户，
        # 但不能再计入已重置的计数器，否则重算时会重复计数。
        self.num_stale_output_tokens = 0
        # Drop the stale output instead, for same-step preempt + resume
        # (reset_prefix_cache).
        # 中文：同一 step 内「抢占 + 恢复」时改为直接丢弃这批陈旧输出
        # （典型场景是 reset_prefix_cache），避免把已作废的 token 吐给用户。
        self.drop_stale_output = False

        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        self.num_in_flight_tokens = 0

        # V2+PP+async: Enforces `pp_size` cadence between same-request decode steps
        # so the worker's broadcast slot ring stays consistent.
        self.next_decode_eligible_step = 0

        # Seq of the most recent step this request was scheduled in; fences
        # deferred block freeing (see Scheduler._free_request_blocks).
        self.last_sched_seq = 0

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        self.session_id = session_id

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # Block-aligned token position of a proven shared prefix worth pinning
        # in the (sparse) prefix cache; 0 means none. Set at admission for
        # hybrid/Mamba models when a shared prefix is detected (Marconi-style).
        self.shared_prefix_boundary = 0

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        self.prefill_stats: PrefillStats | None = PrefillStats()

        # Per-request speculative-decoding acceptance accumulator. Populated by
        # the scheduler when --per-request-spec-decode-metrics is set (eagerly on
        # add_request, then observed each verify step); stays None otherwise.
        self.spec_decode_metrics: RequestSpecDecodeMetrics | None = None

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

        # If True, request should be aborted immediately after being added to
        # the scheduler so the connector's request_finished hook runs.
        self.abort_immediately = abort_immediately

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            session_id=request.session_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        """追加新产出的 token（单个或一批，投机解码一次会来多个）。

        Args:
            token_ids: 新生成的 token id，可以是单个 int，也可以是 list[int]。
        Returns:
            None
        Note:
            会同步写入 _output_token_ids 与 _all_token_ids 两个列表 —— 后者要参与
            prefix cache 的 hash 计算，所以每次追加后必须立刻 update_block_hashes()。
            本方法是「两个列表必须同步更新」这一约束的唯一入口。
        """
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them.

        中文：增量计算 block 哈希并追加到 self.block_hashes。只有凑满一个完整 block
        才会产生新哈希 —— 这是 prefix cache 能做「按块复用」的前提。由
        append_output_token_ids() 和构造末尾调用，调度器用它去查询前缀是否命中。
        若构造时没传 block_hasher（_block_hasher 为 None），prefix cache 功能整体关闭。
        """
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        """是否跳过「读」prefix cache（只写不读）。

        用于基准测试或需要强制走全量 prefill 的场景：命中缓存会掩盖真实延迟。
        取值优先看 sampling_params，其次 pooling_params，都没设则默认 False。
        注意只跳过「读」，写入侧仍然会填充缓存。
        """
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        """是否已进入终态（判据：status > PREEMPTED）。"""
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        """取对外的结束原因；未结束返回 None。"""
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        """取第 input_id 个多模态特征展开后的 embedding 个数（占位 token 数）。"""
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        """记录一个时间点事件（QUEUED / SCHEDULED / PREEMPTED 等）到事件时间线。

        timestamp 省略时取当前时间。事件不会自动上报，需要由输出处理器调用
        take_events() 取走后随输出一起返回给前端。
        """
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        """取走并清空已积累的事件列表；无事件时返回 None。

        这是「取走即清空」语义（take 而非 get），保证同一批事件不会被重复上报。
        """
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def take_prefill_stats(self) -> PrefillStats | None:
        if self.prefill_stats is None:
            return None
        prefill_stats = self.prefill_stats
        self.prefill_stats = None
        return prefill_stats

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
