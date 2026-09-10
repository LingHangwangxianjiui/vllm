# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================
# [CN] 文件：vllm/engine/protocol.py
# 职责：定义「引擎客户端」的接口契约（EngineClient），只声明方法签名，不含实现
# 位置：服务层（vllm/entrypoints/**）→ EngineClient → vllm/v1/engine/async_llm.py
# 核心成员：StreamingInput（流式输入）、EngineClient（抽象基类）
# 上游：entrypoints/openai、entrypoints/pooling、entrypoints/serve/** 等所有在线服务入口
# 下游：vllm/v1/engine/async_llm.py 的 AsyncLLM（当前唯一继承 EngineClient 的类）
# 关键概念：ABC 契约、抽象方法 vs 可选能力、异步生成器输出
# 状态：☑ 通读  ☑ 注释完成  □ 已验证
# ============================================================
#
# 【阅读时先分清两条并行的引擎接入路径】本文件只覆盖其中一条，混淆会白读很多代码。
# 1. 在线服务路径（本文件描述的路径）：
#    服务层持有 EngineClient，调用它的 generate/encode 得到异步生成器，逐个产出输出。
#    当前唯一实现是 vllm/v1/engine/async_llm.py 的 AsyncLLM（class AsyncLLM(EngineClient)）。
# 2. 离线推理路径（不经过本接口）：
#    vllm/entrypoints/llm.py 的 LLM 直接持有 vllm/v1/engine/llm_engine.py 的 LLMEngine。
#    LLMEngine 并没有继承 EngineClient（它是 class LLMEngine:），方法名也不同：
#    离线用 add_request + step 的同步循环，在线用 generate 的异步生成器。
#    因此本文件中的方法在 LLMEngine 上不一定存在，反之亦然。
#
# 【为什么要有这一层】服务层（OpenAI 兼容接口、pooling、Anthropic、语音等十几个入口）
# 只依赖这个抽象，不关心底层是单进程还是多进程引擎。这样替换引擎实现时，
# 服务层代码不用改；新增引擎时，照着这份契约实现即可被所有服务入口复用。
#
# 【两类方法的区别，是本文件最重要的设计点】
# 1. @abstractmethod：子类必须实现，否则实例化时直接 TypeError。
#    这些是「引擎的基本能力」，任何引擎都得有（generate/encode/abort/健康检查等）。
# 2. 普通 async def + raise NotImplementedError：可选能力。
#    基类给了默认实现（直接抛错），不强制子类重写。
#    用于只有部分引擎支持的特性：弹性 EP 扩缩、容错、权重在线更新等。
#    服务层调用前应先判断能力，或接受 NotImplementedError。

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.inputs import EngineInput, PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult

if TYPE_CHECKING:
    from vllm.v1.engine import PauseMode


# [CN] 流式请求的「一次输入」。用 @dataclass 定义，自动生成 __init__ 等样板代码。
# 用途：当 generate() 的 prompt 传的是异步生成器时，生成器每次产出的就是本对象。
# 场景是多轮流式会话（如实时语音）：后续轮次的输入在会话进行中才产生，
# 因此不能一次性给全，只能由调用方一边产生、引擎一边消费。
# [CN] 字段说明：
# - prompt：已经渲染好的 EngineInput，不是原始文本。
#   渲染（分词、多模态处理）由调用方/renderer 完成，引擎不再做这一步。
# - sampling_params：本轮输入的采样参数；None 表示沿用请求级别的采样参数。
#   注意它与 generate() 必填的 sampling_params 不同：后者是整次请求的参数。
@dataclass
class StreamingInput:
    """Input data for a streaming generation request.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
    """

    prompt: EngineInput
    sampling_params: SamplingParams | None = None


# [CN] 引擎客户端的抽象基类（ABC = Abstract Base Class，抽象基类）。
# 它规定「一个引擎必须提供哪些能力」，但不实现任何逻辑：方法体只有 ... 或 pass。
# 当前唯一实现：vllm/v1/engine/async_llm.py 的 AsyncLLM。
# 服务层（entrypoints/ 下各个 serving.py）拿到的都是 EngineClient 类型，
# 通过它调用引擎，从而与具体引擎实现解耦。
class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    # [CN] 下面四行是「类级属性标注」，只声明名称和类型，不赋值，运行时访问前必须由子类填充。
    # 它们不是构造参数；子类通常在自己的 __init__ 里给这些名字赋值。
    # - vllm_config：全局配置聚合对象，含 model/parallel/scheduler 等所有子配置
    # - model_config：模型配置，服务层常用它查 max_model_len、runner_type 等
    # - renderer：输入渲染器，负责文本/聊天模板/多模态输入 → EngineInput
    # - input_processor：输入处理器，负责 EngineInput → EngineCoreRequest（交给核心调度）
    vllm_config: VllmConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    input_processor: InputProcessor

    # [CN] 以下四个是「生命周期状态」属性，且都是抽象方法，必须由子类实现。
    # 设计要点：它们是 @property，调用时不带括号，写 engine.is_running 而不是 is_running()。
    # 服务层的健康检查与看门狗循环（entrypoints/launchers/launcher.py）依赖这四个状态。

    # [CN] 引擎是否正在运行（已启动且未停止）。注意与 is_stopped 不是简单取反：
    # 中间还存在「已启动但正在停止」这类过渡状态，两个判断要分开看。
    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    # [CN] 引擎是否已完全停止（资源已释放）。用于优雅关闭流程的判断。
    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    # [CN] 引擎是否发生过错误。比逐个 try/except 更轻：服务层在读状态下发请求前先查它。
    @property
    @abstractmethod
    def errored(self) -> bool: ...

    # [CN] 导致引擎失效的那个异常对象本身，而不是布尔值。
    # 用途：服务层需要把真实原因写进 HTTP 响应或日志时，取这个异常而不是另造一个。
    # 返回类型是 BaseException（异常的基类），因此连 KeyboardInterrupt 之类的也能承载。
    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    # [CN] 准入控制：在真正开始生成之前，先判断这个请求是否会被队列限制拒绝。
    #
    # 为什么必须放在「响应开始之前」：HTTP 流式响应一旦开始（已发出 200 和首个 chunk），
    # 状态码就无法再改成 429/503。所以超载拒绝只能在这个时间点做，晚一步就只能断开连接。
    #
    # 为什么它不是 @abstractmethod：这是一个「可选优化」，不是基本能力。
    # 基类给了空实现（什么都不做 = 接受一切），没有准入控制的引擎直接继承即可。
    # 有准入控制的引擎（如 AsyncLLM）重写它，在超限时抛出 GracefulHTTPError。
    #
    # 参数：n 是该请求会占用的序列数（n>1 的采样会占多个）；request_id 仅用于日志。
    # 注意：它不是「预留资源」，只是判断 + 拒绝，不保证后续一定不会因为显存被抢占。
    def check_admission(  # noqa: B027
        self, n: int = 1, request_id: str | None = None
    ) -> None:
        """Reject the request up front if it would exceed queue limits.

        Called before a response is started so that overload rejections can
        carry an HTTP status, which is not possible once a streaming response
        has begun. Engines without admission control accept everything.

        Args:
            n: Number of sequences the request will occupy.
            request_id: Request id, used for logging only.

        Raises:
            GracefulHTTPError: If the request cannot be admitted.
        """

    # [CN] 生成接口：整条在线服务链路的主入口，返回异步生成器。
    #
    # 【返回类型是异步生成器，不是列表】签名是 -> AsyncGenerator[RequestOutput, None]，
    # 调用方要写 `async for output in engine.generate(...)` 逐个取结果。
    # 每取一次，底层可能推进一步引擎；输出是「增量 + 最终」混合的 RequestOutput 序列。
    # 这与离线 LLM.generate() 返回 list[RequestOutput] 完全不同，不要混淆两条路径。
    #
    # 【prompt 的四种形态】这个联合类型是本方法最需要理解的部分，代表四种接入方式：
    # 1. PromptType      —— 原始文本或 token 列表，由引擎内部的 renderer 负责渲染
    # 2. EngineInput     —— 调用方已经渲染好的输入，引擎不再渲染
    # 3. EngineCoreRequest —— 调用方直接构造好的核心请求，引擎只做提交（跳过渲染与校验）
    # 4. AsyncGenerator[StreamingInput, None] —— 多轮流式会话，输入边产生边消费
    #
    # 【* 之后的参数都是关键字参数】调用时必须写名字，不能按位置传。
    # - prompt_text：原始文本，与 prompt 分开传，用于日志/回显，不参与分词
    # - lora_request：本次请求使用的 LoRA 适配器；None 表示不用
    # - tokenization_kwargs：分词阶段的覆盖项，传给 renderer，不控制采样
    # - trace_headers：分布式追踪的请求头，用于把引擎内部 span 接到上游 trace 上
    # - priority：请求优先级，默认 0；只有启用优先级调度策略时才真正生效
    # - data_parallel_rank：指定交给 DP 的哪一个 rank；None 由引擎自行分配
    # - session_id：会话标识，用于多轮/长会话场景关联同一次会话
    # - reasoning_ended / reasoning_parser_kwargs：推理内容（thinking）解析相关，
    #   用于把模型输出中的思考段落与普通回答分开处理
    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    # [CN] 池化模型（embedding / classify / score 等）的对应入口。
    # 与 generate 的区别：
    # 1. 第二个参数换成 pooling_params（如何池化），而不是 sampling_params（如何采样）；
    #    因为池化任务不做自回归采样，没有温度、top_p 这些概念。
    # 2. 返回的是 PoolingRequestOutput，里面是向量或分数，不是文本候选。
    # 3. prompt 只接受 PromptType 或 EngineInput，不接受异步生成器：
    #    池化是一次性前向计算，没有「多轮流式」的场景。
    # 注意：generate 与 encode 是两个独立入口，不能靠传参让其中一个变成另一个。
    @abstractmethod
    def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    # [CN] 中止请求。参数是单个 ID 或 ID 的可迭代对象，因此支持批量中止。
    # 为什么是 async：中止要跨进程通知 EngineCore，并等待确认，不是本地一个标志位。
    # 语义边界：中止只保证「不再继续生成」，不保证已经产出的输出被撤回；
    # 客户端断开连接时，服务层通常在 finally 里调用它来释放 KV cache 等资源。
    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    # [CN] 通知引擎：这个 KV 传输请求在进入引擎之前就被拒绝了。
    #
    # 存在的理由（对应 PD 分离 / KV transfer 场景）：
    # 预填充节点（P 节点）可能已经为这次请求把 prefix 的 KV block 固定（pin）住了，
    # 但请求在准入检查阶段就被拒，永远不会走到解码节点（D 节点）。
    # 如果不显式通知，那些被 pin 的 block 没有正常的释放路径，会一直占着显存。
    # 所以这个方法的作用是「给连接器一个补做清理的机会」，而不是改变请求状态。
    #
    # kv_transfer_params 是连接器自己定义的参数字典，本层不理解其字段；
    # data_parallel_rank 指定目标 DP rank，None 由引擎决定。
    @abstractmethod
    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Notify the engine that a KV-transfer request was rejected before
        engine admission, so connector-side cleanup can run (e.g. free
        prefill blocks pinned on the P node).
        """
        ...

    # ============================================================
    # [CN] 以下一组是「运维与可观测性」接口，服务层把它们暴露成 HTTP 端点：
    # 健康检查、性能采集开关、各类缓存重置、休眠/唤醒、LoRA 热加载、暂停恢复。
    # 它们都不走采样参数流程，也不产生生成结果。
    # ============================================================

    # [CN] 是否启用了分布式追踪。服务层据此决定要不要为本次请求创建 span。
    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    # [CN] 立即输出一次统计日志（不等到下一个统计周期）。通常由定时任务或端点触发。
    @abstractmethod
    async def do_log_stats(self) -> None: ...

    # [CN] 健康检查：不返回值，靠「抛异常」表达不健康。
    # 这是 Python 里常见的约定——正常路径无需返回值，异常路径携带具体原因。
    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine"""
        ...

    # [CN] 三个 reset_* 是三种不同的缓存，不要当成同一个东西：
    # 1. reset_mm_cache：多模态预处理缓存（图像等处理后的中间结果），在渲染阶段。
    # 2. reset_encoder_cache：多模态编码器的输出缓存，存的是 encoder 算出的特征。
    # 3. reset_prefix_cache：前缀 KV cache，存的是已算过的 token 的 KV，用于前缀复用。
    # 它们位于链路的不同层，释放其中一个不影响另外两个。

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    @abstractmethod
    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache"""
        ...

    # [CN] 返回 bool：是否真的重置成功。
    # 为什么可能失败：前缀缓存的 block 被仍在运行的请求引用着，不能强行释放。
    # - reset_running_requests=True：先把运行中的请求抢占回等待队列，腾出 block 再重置；
    #   代价是这些请求已算的 KV 作废，需要重算。False 则不强制，仍有占用时返回 False。
    # - reset_connector=True：连 KV 连接器（PD 分离场景）的远端缓存一起清理。
    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache"""
        ...

    # [CN] 休眠：释放 GPU 资源但不销毁引擎，用于「同一个进程里换模型」或「空闲时省显存」。
    # level 决定释放到什么程度（见 vllm/entrypoints/llm.py 的 sleep 文档）：
    #   0 —— 只暂停调度，仍可接收请求，请求排队但不执行
    #   1 —— 权重卸载到 CPU 并丢弃 KV cache（换回同一模型时用，需要足够 CPU 内存）
    #   2 —— 权重和 KV 全部丢弃（换一个完全不同的模型时用，CPU 压力更小）
    # mode 决定在途请求怎么处理：abort 立即中止 / wait 等其做完 / keep 冻结等唤醒后继续。
    # 调用方责任：本方法不保证「调用瞬间没有请求在跑」，文档要求调用方自行保证。
    @abstractmethod
    async def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None:
        """Sleep the engine"""
        ...

    # [CN] 唤醒。tags 是「要恢复哪些东西」的标签列表，取值 weights / kv_cache / scheduling。
    # 关键坑：None 表示全部恢复，而空列表 [] 不是 None，语义不同，不要混用。
    # 另一个坑：只传 tags=["scheduling"] 适合 level=0 的暂停恢复；
    # level>=1 的休眠必须先恢复 weights/kv_cache，只恢复调度是不够的（调度会起不来）。
    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    # [CN] 热加载一个 LoRA 适配器，只影响「之后」的请求，不影响已在跑的请求。
    # 返回 bool 表示是否加载成功（例如超出 max_lora_rank 或插槽不足会失败）。
    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    # [CN] 暂停生成：与 sleep 的区别是它不动 GPU 内存，只是停止接收/处理生成请求。
    # 典型用途是「滚动更新权重」：先暂停，等在途请求排空，再更新权重，最后 resume。
    # mode 与 sleep 的 mode 同义，但这里作用于在途的生成请求；
    # keep 会把请求冻结在队列里，resume_generation 后继续执行，这是与 abort 的本质区别。
    @abstractmethod
    async def pause_generation(
        self,
        *,
        mode: "PauseMode" = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation/encoding requests.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  and return partial results with "abort" reason (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED. Use ``mode="wait"`` instead.
            clear_cache: DEPRECATED. Whether to clear KV and prefix caches
                after draining.
        """
        ...

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        ...

    @abstractmethod
    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        ...

    # [CN] 关闭引擎并释放资源。注意这是本接口里少见的「同步」方法（没有 async）。
    # 原因：关闭通常在事件循环收尾阶段调用，做成同步更容易在 finally 和 atexit 里使用。
    # timeout 是等待优雅退出的秒数；None 的语义由实现决定（通常是一直等）。
    # 调用后引擎不可再用，与 sleep/wake_up 这种可恢复操作性质不同。
    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine with optional timeout."""
        ...

    # ============================================================
    # [CN] 以下全部是「可选能力」：不是 @abstractmethod，基类默认实现直接
    # raise NotImplementedError。子类按需重写；不支持时调用会抛该异常。
    # 判断依据：这些都是特定场景才有的能力（弹性扩缩、容错、RL 权重在线更新），
    # 不要求每个引擎都实现。服务层调用前应做能力判断或捕获 NotImplementedError。
    # ============================================================

    # [CN] 弹性 EP（Expert Parallel）扩缩容：在线把引擎的 data parallel 规模改成新值。
    # drain_timeout 是排空现有请求的最大等待秒数，默认 300 秒。
    # 属于高级运维能力，多数单进程/固定规模部署不支持。
    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None:
        """Scale the engine"""
        raise NotImplementedError

    # [CN] 向所有 worker 广播一次 RPC 调用，返回每个 worker 的结果组成的列表。
    # method 是 worker 上的方法名（字符串）；args/kwargs 是传给它的参数。
    # 官方建议只用它传「控制消息」，大数据用数据面通道传——因为它走进程间通信，
    # 传大张量会明显变慢并占用额外显存。
    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    # [CN] 容错：向引擎下发故障处理指令（如让某个 rank 失败、模拟故障等），
    # 用于容错机制的测试与演练，返回处理结果。
    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """send fault tolerance instruction to the engine"""
        raise NotImplementedError

    # [CN] 查询所有引擎的容错状态。注意 docstring 里是 engines（复数），
    # 在 DP 等多引擎场景下返回的是整体状态，不是单个引擎的。
    async def get_status(self):
        """Get fault tolerance status of all engines."""
        raise NotImplementedError

    # [CN] 查询该引擎支持的任务元组（generate / embed / classify / score 等）。
    # 服务层用它决定要不要暴露某个端点，也用于拒绝模型不支持的任务。
    # 注意离线侧 LLMEngine 也有同名方法，但那不是对本接口的实现（LLMEngine 不继承本类）。
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks"""
        raise NotImplementedError

    # ============================================================
    # [CN] 以下是 RL（强化学习）在线权重更新的一组接口，按顺序配套使用：
    #   init → start → update（可多次）→ finish；版本可单独设置/查询。
    # 用途：训练侧算出新权重后，不重启推理引擎就把权重推送进来。
    # 它们是三个独立调用，本层不提供事务保证：中间失败不会自动回滚。
    # ============================================================

    async def init_weight_transfer_engine(
        self, init_request: WeightTransferInitRequest
    ) -> None:
        """Initialize weight transfer for RL training."""
        raise NotImplementedError

    async def start_weight_update(self) -> None:
        """Start a new weight update."""
        raise NotImplementedError

    # [CN] 专门针对投机解码「草稿模型」的权重更新入口，与主模型更新分开。
    # 原因：投机解码下存在两套权重（主模型 + 草稿模型），必须能分别更新，
    # 用同一个入口无法区分目标。
    async def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        raise NotImplementedError

    # [CN] 批量更新权重本体。具体的数据格式与写入方式由传输后端解释，
    # 本层只负责把 request 传到引擎；「批量」指一次调用可含多个张量。
    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """Batched weight update for RL training."""
        raise NotImplementedError

    # [CN] 结束本次更新，并可选地把这次权重标记为某个版本号。
    # weight_version 为 None 时只结束、不打版本；非 None 时同时设置版本。
    # 版本号的作用：训练侧据此确认推理侧已经用到第几版权重，避免读到旧权重的结果。
    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        raise NotImplementedError

    # [CN] 只改版本标记，不动权重。注意它不校验实际权重内容是否与版本匹配，
    # 用错会导致版本号与实际权重不一致。
    async def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        raise NotImplementedError

    async def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        raise NotImplementedError
