# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ==============================================================================
# 本文件职责：定义 V1 引擎「前端进程 ↔ EngineCore 进程」之间的**通信协议**。
#   注意它不是引擎实现，而是引擎两侧共同依赖的一本"报文字典"：请求怎么打包、
#   输出怎么回传、控制指令有哪些、状态如何表示，全部在这里用 msgspec.Struct 定义。
#   几乎所有结构体都加了 array_like=True / gc=False / omit_defaults=True，
#   这是为了跨进程（ZMQ + msgpack）传输时**体积小、速度快**（详见下方阅读提示 1）。
#
# 在系统链路中的位置（跨进程边界的那一条线）：
#
#   前端进程（Frontend）                        │  EngineCore 进程（Core）
#   --------------------------------------------┼----------------------------------
#   LLM.generate() / AsyncLLM.generate()        │
#     -> InputProcessor 处理 prompt             │
#     -> EngineCoreRequest  【本文件定义】        │
#     -> core_client.add_request(request)       │
#        ├─ InprocClient：进程内直接函数调用      │  -> EngineCore.add_request()
#        └─ MPClient：ZMQ 发送                  │     -> Request（v1/request.py）
#           (EngineCoreRequestType 作为首帧)  ⇒  │        -> Scheduler.schedule()
#                                              │        -> Worker -> ModelRunner
#   输出处理 OutputProcessor  <-  EngineCoreOutputs【本文件定义】  ⇐  EngineCore.step()
#     -> RequestOutput（vllm/outputs.py）        │
#
# 核心内容速查：
#   - PauseMode / FINISH_REASON_STRINGS   : 对外暴露的字符串常量（属于公开 API）
#   - EEP_* / FT_STATUS_CALL_ID           : 特殊 call_id 约定（负数是"系统保留号"）
#   - FinishReason                        : 终态原因枚举，IntEnum 便于紧凑序列化
#   - EngineCoreReadyResponse             : 引擎启动完成后回给前端的"能力清单"
#   - EngineCoreRequest                   : 前端 -> Core 的请求报文（含多模态/embeds）
#   - EngineCoreEvent / EngineCoreEventType: 请求生命周期事件（排队/被调度/被抢占）
#   - EngineCoreOutput / EngineCoreOutputs : Core -> 前端的输出报文（一帧可能装多请求）
#   - UtilityOutput / UtilityResult        : 控制类 RPC 的返回（如 abort/sleep/profile）
#   - EngineCoreRequestType                : ZMQ 多帧消息的**首帧类型标记**（字节枚举）
#   - ReconfigureDistributedRequest        : 运行时改 DP 规模（elastic EP 用）
#   - EngineStatusType                     : 引擎健康状态，供 /health 之类的接口查询
#
# 阅读提示（几个容易踩的点）：
#   1. array_like=True 的消息结构：字段按**声明顺序**编码成数组（不带字段名），
#      因此**新增字段必须追加在末尾**，插到中间会让新旧版本收发错位（升级不兼容）。
#      文件里多处注释都写了 "Appended last so array_like positional serialization
#      stays backward compatible"，就是这个原因。
#   2. 时间戳用 time.monotonic()（单调时钟），**不能跨进程比较**——每个进程的
#      monotonic 起点可能不同。跨进程要做时间对齐得用别的机制。
#   3. call_id 为负数是"内部保留号"：EEP_NOTIFICATION_CALL_ID=-1 用于 EEP
#      （Elastic Expert Parallelism）通知，FT_STATUS_CALL_ID=-2 用于容错状态查询。
#      业务 utility 调用的 call_id 从 0 开始自增，因此不会撞车。
#   4. EngineCoreOutput 是"一条请求一帧"，EngineCoreOutputs 是"一次 step 一帧"，
#      后者把本轮所有请求的输出打包，还顺带捎上调度统计、DP wave 信号、finished
#      请求集合等**控制信息**——它是前端驱动的节拍器。
#   5. torch.Tensor（prompt_embeds / pooling_output）走的是单独通道：默认 ZMQ 只传
#      小消息，大张量用 tensor_ipc.py 的共享内存句柄，避免序列化拷贝开销。
# ==============================================================================
import enum
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import msgspec
import numpy as np
import torch

from vllm.config.kv_events import KVEventsConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import (
    PrefillStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplingMaskLists
from vllm.v1.serial_utils import UtilityResult

# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
#
# [CN] 暂停生成的三种模式，决定"已经在跑的请求"怎么处理：
#   - "abort"：立刻中止所有在途请求（默认）。请求会收到 finish_reason="abort"，
#              已分配的 KV block 立即释放，最省显存但会丢工作。
#   - "wait" ：等所有在途请求自然跑完再真正暂停。不丢工作，但暂停有延迟，
#              且暂停期间不能接收新请求（否则就永远等不到"全部跑完"）。
#   - "keep" ：把请求**冻结在队列里**（不释放 block、不丢弃已算的 token），
#              resume_generation() 之后原地继续。这是"无损暂停"，代价是暂停期间
#              显存一直被占着。sleep mode / RL 权重更新这类场景用 "keep"。
# 注意：这是**对外 API 的取值**（OpenAI 兼容服务的 pause 接口会透传），
#       改字符串就是破坏性变更。
PauseMode = Literal["abort", "wait", "keep"]

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
#
# [CN] 对外暴露的 finish_reason 字符串，与下面 FinishReason 枚举**按值一一对应**
#      （FinishReason.__str__ 就是用 self.value 去索引这个元组）。
#      顺序即 enum 的数值，不能重排；新增原因必须追加在末尾。
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")

# [CN] EEP（Elastic Expert Parallelism，弹性专家并行）通知专用的 call_id。
#      取负数是为了和"业务 utility 调用"的自增 call_id（从 0 开始）区分开：
#      收到 call_id < 0 的输出时，前端知道这不是某个 RPC 的应答，而是引擎主动
#      推来的通知（如 RECONFIGURE_FINISHED / SHUTDOWN_COMPLETE），
#      要走 EEPNotificationType 分支处理，不能塞进普通 result 队列。
EEP_NOTIFICATION_CALL_ID = -1

# [CN] 容错（Fault Tolerance）状态查询专用的 call_id，同样取负数避开业务号段。
FT_STATUS_CALL_ID = -2


class EEPNotificationType(enum.Enum):
    """[CN] EEP 通知的类型。配合 EEP_NOTIFICATION_CALL_ID（-1）使用。

    RECONFIGURE_FINISHED：弹性扩缩容（改 DP/EP 规模）的 reconfigure 流程已经完成，
        前端可以继续正常派发请求了。在此之前前端必须拦住新请求，否则会打到
        还没重建好通信域的 rank 上。
    SHUTDOWN_COMPLETE：引擎侧（被缩容掉的那部分 rank）已经安全退出，
        前端可以据此回收对应资源 / 更新路由表。
    """

    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    [CN] 中文补充：
      - 用 IntEnum 而不是 str：这是**跨进程高频传输**的字段，int 编码只有 1 字节，
        字符串要几十字节；同时 msgpack 对 int 也有紧凑编码。
      - STOP      : 命中了 stop_strings / stop_token_ids / eos（正常结束）。
      - LENGTH    : 用满了 max_tokens，或者 prompt+output 触到 max_model_len。
                    **这两种情况对外都是 "length"**，要看是哪种得结合输出长度判断。
      - ABORT     : 客户端主动取消（断连、abort 接口、引擎暂停时 mode="abort"）。
      - ERROR     : 请求级可重试错误（典型是 KV 加载失败）。约定：**一律转成
                    HTTP 500**，不要当成用户输入错误返回 4xx。
      - REPETITION: 检测到复读（幻觉）被主动掐断，依赖 repetition_penalty 类检测。
      - __str__ 直接查 FINISH_REASON_STRINGS 元组，省掉一个字典。
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


@dataclass
class EngineCoreReadyResponse:
    """Sent from EngineCore to each frontend at the end of engine startup.

    Contains post-initialization config that may differ from the original
    values (e.g. max_model_len after KV cache auto-fitting).

    [CN] 引擎启动"握手"阶段由 Core 回给每个前端的一帧**能力清单 / 最终配置**。
      为什么需要它：前端在启动 Core 之前只知道用户给的参数，而很多真实值必须等
      Core 里跑完 profile / 显存测量 / 权重加载才知道。典型例子：
        - max_model_len：KV cache 不够时会被自动下调（auto-fitting）；
        - num_gpu_blocks / kv_cache_size_tokens：只有算完显存才知道有多少 block；
        - dtype：config 里写 "auto" 时，实际 dtype 由权重和硬件决定；
        - world_size / 各种并行度：DP 外部启动模式下要等 rank 协商完。
      前端拿到它之后才能正确构造 OutputProcessor、初始化 tokenizer 的 max length、
      填写 /v1/models 之类的元信息接口，以及**校验用户请求是否超长**。
      它是 dataclass（不是 msgspec.Struct），因为只在启动时传一次，不追求编码性能。
    """

    max_model_len: int
    num_gpu_blocks: int
    block_size: int
    dp_stats_address: str | None
    dtype: str
    vllm_version: str
    world_size: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    decode_context_parallel_size: int
    data_parallel_rank: int
    max_num_seqs: int
    max_num_batched_tokens: int
    instance_id: str
    supports_lora: bool
    max_loras: int
    mamba_block_size: int | None = None
    # KV cache capacity (None for encoder-only/attention-free models).
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None
    kv_events_config: KVEventsConfig | None = None
    weight_transfer_backend: str | None = None
    enable_sleep_mode: bool = False
    supports_draft_weight_updates: bool = False


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """[CN] 前端 -> EngineCore 的**请求报文**，是唯一能穿过进程边界的请求形态。

    序列化三件套的含义（本文件所有跨进程结构体基本都这么配）：
      - array_like=True ：编码成**数组**而非字典，不带字段名，体积最小；
                          代价是字段顺序即协议，**新字段只能追加在末尾**。
      - omit_defaults=True：等于默认值的字段不编码，进一步省带宽；
                          对 list[int] 这类"必填"字段没有影响。
      - gc=False        ：关掉 msgspec 的循环引用检测（本结构是纯 DAG），
                          解码/编码能快一截。
    与 vllm/v1/request.py 的 Request 的区别：
      - 本结构是"线上格式"（可序列化、无方法、无运行时状态）；
      - Request 是 Core 进程内的"活对象"（持有 block、状态、事件时间线）。
      - 转换入口是 Request.from_engine_core_request()。
    """

    request_id: str
    # [CN] 已经 tokenize 完的 prompt。为 None 表示纯 prompt_embeds 请求
    #      （调用方直接给向量，跳过 embedding 查表）。
    prompt_token_ids: list[int] | None
    # [CN] 多模态特征（图像/音频/视频的预处理结果）。
    #      注意这里传的是 **MultiModalFeatureSpec（描述/占位）**，真正的像素张量
    #      走共享内存通道，不进 msgpack —— 否则一张图能把消息体撑到几百 MB。
    mm_features: list[MultiModalFeatureSpec] | None
    # [CN] 采样参数（生成任务）与池化参数（embedding/分类任务）**二选一**，
    #      由下面的 params 属性统一取出。两者都 None 是非法状态。
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    # [CN] 请求到达时间（time.monotonic()）。用途：
    #      1) 调度器排队优先级（配合 priority 字段）；
    #      2) 前端统计排队时延（TPOT/TTFT 指标）。
    arrival_time: float
    lora_request: LoRARequest | None
    # [CN] 前缀缓存的"命名空间"：相同 prompt + 不同 cache_salt 不会互相命中。
    #      典型用法是多租户隔离，或者做实验时强制绕过缓存（每次随机 salt）。
    cache_salt: str | None
    # [CN] 指定这个请求交给哪个 DP rank 处理。外部 DP 负载均衡模式下由前端
    #      （或路由器）决定；None 表示由 Core 自己按负载挑一个。
    data_parallel_rank: int | None
    # [CN] 预计算好的 prompt 向量（torch.Tensor）。走 tensor_ipc 的共享内存通道，
    #      不走 msgpack 序列化。与 prompt_token_ids 可混合出现（见下）。
    prompt_embeds: torch.Tensor | None = None

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    prompt_is_token_ids: list[bool] | None = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    # [CN] 多前端（scale-out）场景下标识"这个请求是哪个前端发来的"。
    #      Core 的输出必须**原路返回**给同一个前端进程，否则那个前端的
    #      OutputProcessor 里没有对应请求的状态，会直接把输出丢掉。
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    # [CN] DP "wave（波次）"机制：为了让所有 DP rank 每轮跑**同样的批次形状**
    #      （否则 NCCL 集合通信会死锁），引擎按波次推进 —— 一波请求必须全部完成
    #      才能开下一波。这里填的是"发送方认为当前是第几波"，用来消除竞态：
    #      请求在"上一波结束通知"到达之前就被发出了，Core 收到后发现 wave 号
    #      落后，就会用 start_wave 信号去触发下一波，而不是把请求挂死。
    current_wave: int = 0
    # [CN] 排队优先级：数值越小越优先（与常见约定相反，注意别搞混）。
    #      只在 waiting 队列排序时生效，已经在跑的请求不会被更高优先级抢占。
    priority: int = 0

    # [CN] 分布式链路追踪头（W3C traceparent 之类），透传给 Core 再带回输出，
    #      用于把一次推理的 trace 串起来（OpenTelemetry）。
    trace_headers: Mapping[str, str] | None = None
    # [CN] 是否为"可续写"的流式会话请求。为 True 时同一 request_id 会反复收到
    #      新的输入增量（StreamingUpdate），而不是一次性的 prompt。
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None

    # [CN] 推理模型（reasoning / thinking）相关：
    #   reasoning_ended            : 思考段是否已经结束（用于分离 reasoning_content
    #                                与 content 的流式输出）。
    #   reasoning_parser_kwargs    : 传给 reasoning parser 的额外参数
    #                                （如 ThinkingPlan 的分隔符配置）。
    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None

    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    abort_immediately: bool = False

    # [CN] 会话 ID：用于把同一会话的多轮请求关联起来（会话级 KV 复用 / 计费）。
    session_id: str | None = None

    @property
    def params(self) -> SamplingParams | PoolingParams:
        """Return the processed params (sampling or pooling)."""
        # [CN] 统一取出"真正的参数对象"：生成任务拿 sampling_params，
        #      池化任务拿 pooling_params。Core 侧绝大多数逻辑不关心是哪种任务，
        #      都通过这一个属性访问，避免到处写 if/else。
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""
    # [CN] 请求生命周期的三个**时间点**，用来给前端算时延指标：
    #   QUEUED     : 进入 waiting 队列（排队开始）
    #   SCHEDULED  : 第一次被调度器选中，真正开始算（排队结束 => 这就是 TTFT 的终点）
    #   PREEMPTED  : 被抢占（显存不足，已算的 token 作废重排）
    # 注意这里的"事件"不是每个 step 都发，只在状态**第一次**跃迁时记一笔时间戳，
    # 所以一条请求的 events 列表通常只有几个元素。

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamp and is used by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.

    [CN] 注意它是 **msgspec.Struct 但没有 array_like**，所以按**字段名**编码。
        事件量小（每请求个位数），可读性优先于体积。
        另外 timestamp 用 monotonic 时钟：同一进程内可做差（算间隔），
        但**跨进程无意义**（各进程起点不同），文档里特意强调了这点。
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """[CN] Core -> 前端的**单请求输出**，一次 step 里每个有进展的请求产生一个。

    关键语义：
      - 它是"增量"的：new_token_ids 只装**本轮新增**的 token，不是全量。
        前端（OutputProcessor）负责累积拼接成完整输出。这样跨进程传输量最小。
      - finish_reason 不为 None 即代表这条请求结束（见 finished 属性），
        这一帧也是该请求的**最后一帧**；之后 Core 会把它放进
        EngineCoreOutputs.finished_requests 通知前端清理状态。
      - 字段顺序即协议，新增字段必须追加到末尾（array_like 的位置编码）。
    """

    request_id: str
    # [CN] 本轮新产出的 token ids（增量）。prefill 阶段通常是空列表或 1 个，
    #      decode 阶段每轮 1 个（投机解码时可能多个）。
    new_token_ids: list[int]

    # [CN] 采样 logprobs：new_logprobs 是"对数概率"（给 API 用户的最终形态），
    #      new_prompt_logprobs_tensors 是 prompt logprobs 的张量形式（量更大，
    #      走专用通道）。两者都只在 SamplingParams 里显式要求时才非 None。
    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    # [CN] 池化任务（embedding / classify / reward）的输出向量，走共享内存传输。
    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    # [CN] 触发停止的具体原因值：可能是 int（stop token id）也可能是 str
    #      （匹配的 stop 字符串）。只有 finish_reason == STOP 时才有意义。
    stop_reason: int | str | None = None
    # [CN] 本轮新产生的事件（QUEUED / SCHEDULED / PREEMPTED），
    #      由 Core 侧累积后 take 出来，前端用来算排队/首 token 时延。
    events: list[EngineCoreEvent] | None = None
    # [CN] KV 传输参数：P/D 分离（prefill 与 decode 分离部署）场景下，
    #      prefill 节点把 KV 的位置信息通过这个字段回传给前端/调度层。
    kv_transfer_params: dict[str, Any] | None = None
    # [CN] EC（Encoder Cache）传输参数：多模态 encoder 输出的传输信息，同上。
    ec_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None

    prefill_stats: PrefillStats | None = None

    # [CN] MoE 模型里这条请求每层的路由专家编号（用于 EPLB 统计与负载均衡分析）。
    routed_experts: np.ndarray | None = None
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    # [CN] 本轮 logits 里 NaN 的个数。> 0 说明这条请求的输出已经坏了
    #      （典型原因：非法采样参数、数值溢出、坏权重）。前端据此把请求判为失败，
    #      而不是把一堆 NaN token 返回给用户。
    num_nans_in_logits: int = 0
    # Multi-modal hashes missing from the P1 receiver cache (P0/P1 drift; see
    # `MultiModalCacheMissError`). Non-empty => retryable: the frontend drops these
    # from its sender cache and the request is resent with the data. Appended last
    # so `array_like` positional serialization stays backward compatible.
    mm_cache_miss_hashes: list[str] | None = None

    # [CN] 采样掩码：记录哪些位置被 mask 掉（如结构化输出 grammar 约束、
    #      或者 logit processor 屏蔽了某些 token），用于回传"为什么不能采这个 token"。
    new_sampling_mask: SamplingMaskLists | None = None

    # Per-request spec-decode acceptance; attached only on the final output.
    # Appended last so `array_like` positional serialization stays compatible.
    spec_decode_metrics: RequestSpecDecodeMetrics | None = None

    @property
    def finished(self) -> bool:
        """[CN] 这条请求是否已结束：判据就是 finish_reason 非 None。

        注意这是"这一帧是最后输出"的判据，不等于前端可以立刻删状态 ——
        真正可以清理要等 EngineCoreOutputs.finished_requests。
        """
        return self.finish_reason is not None


class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """[CN] "控制类 RPC"的返回包。

    vLLM 把两类消息分开走：数据面（请求/输出）用上面的 EngineCoreOutputs，
    控制面（abort / sleep / wake_up / profiling / add_lora / collective_rpc ...）
    用 UtilityOutput —— 前端发一个带 call_id 的 utility 调用，然后**阻塞等待**
    相同 call_id 的应答回来（所以 call_id 是匹配请求与响应的关键）。
    failure_message 非 None 即代表调用失败（此时 result 必为 None），
    前端会把它转成异常抛给调用方。
    """

    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """[CN] Core -> 前端的**每步一帧**总输出，是整条链路的"节拍器"。

    一次 EngineCore.step() 产出一帧，里面可能装着多条请求的增量输出，
    外加本轮的控制/统计信息。前端（SyncMPClient / AsyncMPClient）拿到它之后：
      1) 把 outputs 分发给 OutputProcessor 拼装成 RequestOutput；
      2) 用 scheduler_stats 更新 /metrics 的调度指标；
      3) 用 finished_requests 清理已结束请求的状态（这是**权威的结束通知**，
         前端不能只靠 finish_reason 判断，因为还要回收 KV / detokenizer 状态）；
      4) DP 场景下用 wave_complete / start_wave 推进波次。
    timestamp 是本帧产生的时间（monotonic），前端用来算"引擎空闲/忙碌"间隔。
    """

    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    # [CN] 哪个引擎（DP rank）产生的这一帧；多引擎（DP>1 或外部 LB）时用于区分来源。
    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    # [CN] 控制类 RPC 的应答，捎带在同一帧里返回（见 UtilityOutput 的说明）。
    #      非 None 时前端会把它 routing 到等待该 call_id 的调用方。
    utility_output: UtilityOutput | None = None
    # [CN] 本轮**已经彻底结束**的请求 id 集合。注意与 output.finish_reason 的区别：
    #      finish_reason 只是"这一帧是这个请求的最后输出"，
    #      finished_requests 是 Core 侧"已经走完收尾（释放 block、记账）"的确认，
    #      前端必须等它才能安全删除本地状态，否则可能漏掉后续帧。
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    # [CN] 值为**波次号**：表示这一波已跑完、引擎已暂停，等待下一波开始。
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    # [CN] 值为需要开启的**下一波波次号**。当某个 rank 收到属于"更老波次"的请求时，
    #      说明有前端还没收到上一波的结束通知，于是广播这个信号让所有 rank 一起开新波，
    #      避免"部分 rank 进下一波、部分还在等"造成的集合通信死锁。
    start_wave: int | None = None

    def __post_init__(self):
        # [CN] 没显式给时间戳就填"现在"（monotonic）。之所以用 0.0 做哨兵值
        #      而不是 None，是为了保持字段类型是 float，序列化更省事。
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.

    [CN] ZMQ 多帧消息的**首帧（type frame）**：直接用一个字节表示"后面那帧是什么"，
        省掉一次结构化封装/解析。Core 的输入循环在 input_queue 上阻塞 get 到
        (type, data) 二元组后按这里的值分派。
        注意 EXECUTOR_FAILED / WAKEUP 不是网络消息，而是**进程内部的哨兵**：
        前者由 EngineCoreProc 在执行器崩溃时塞进队列触发优雅退出，
        后者用于在关闭时把阻塞在 queue.get() 的线程唤醒，避免卡死在 join。
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"


class ReconfigureDistributedRequest(msgspec.Struct):
    """[CN] 运行时**改变分布式拓扑**的请求（elastic EP / 在线扩缩容用）。

    场景：MoE 模型的专家并行规模需要在线调整（比如流量高峰扩容），
    此时要重建 DP 通信域、重新分配 rank、重连 master 地址与端口。
    这个结构体把"新的拓扑参数"一次打包发给所有 rank，各 rank 收到后
    按 new_data_parallel_rank 决定自己是 KEEP_CURRENT_RANK（继续干活）、
    SHUTDOWN_CURRENT_RANK（被裁掉，优雅退出）还是换成新 rank。
    完成后 Core 会给前端回 EEPNotificationType.RECONFIGURE_FINISHED。
    """

    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int
    # [CN] 每个 rank 各自要监听的端口列表（多节点时每个本地 rank 一个端口）。
    new_data_parallel_master_port_list: list[int]
    # [CN] 协调器（coordinator）的 TCP store 端口，用于各 rank 交换握手信息
    #      （torch.distributed 的 init_process_group 需要它）。
    coord_store_port: int


class ReconfigureRankType(enum.IntEnum):
    """
    Rank type for reconfiguring distributed request.

    [CN] reconfigure 时本 rank 的"新身份"。用负数是为了和实际 rank 号（>=0）区分：
      KEEP_CURRENT_RANK     (-1)：保持现在的 rank 不变，继续服务；
      SHUTDOWN_CURRENT_RANK (-2)：这个 rank 不再需要了，优雅退出释放资源；
      其他值（>=0）                ：切换到这个新的 DP rank 号继续服务。
    """

    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2


class EngineStatusType(enum.IntEnum):
    """[CN] 引擎健康状态，供 /health、/ready 之类的运维接口查询。

    HEALTHY   : 正常，可以接请求；
    DEAD      : 引擎进程已经没了（崩溃/被杀），不可恢复，只能重启；
    UNHEALTHY : 进程还在但状态异常（如执行器失败、通信域损坏），
                根据具体策略可能自愈也可能要重启。
    """

    HEALTHY = 0
    DEAD = 1
    UNHEALTHY = 2
