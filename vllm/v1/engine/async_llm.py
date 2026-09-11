# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# 本文件职责：V1 的**异步**引擎门面 AsyncLLM —— 在线服务（OpenAI 兼容 API）走的
#   就是它。它实现了 vllm/engine/protocol.py 里的 EngineClient 协议，是"引擎"
#   在异步世界的标准形态（区别于 llm_engine.py 的同步 LLMEngine）。
#
# 在系统链路中的位置（前端进程内，异步侧）：
#   API server（entrypoints/openai/api_server.py）
#     -> AsyncLLM.generate()【本文件】-> AsyncGenerator[RequestOutput]
#          ├─ InputProcessor        : prompt -> EngineCoreRequest（可能 await）
#          ├─ EngineCoreClient      : 发给 EngineCore 进程
#          └─ OutputProcessor       : 增量输出 -> RequestOutput
#   后台还有一个常驻任务 output_handler：不停从引擎拉输出，塞进每条请求自己的
#   RequestOutputCollector（asyncio 队列）；generate() 只负责从队列里往外 yield。
#
# 必须理解的两个"后台 vs 前台"分工：
#   1) **输出靠后台任务推**：generate() 自己不跟引擎通信。这样 N 个并发请求
#      只需要 1 个到引擎的连接（一次 get_output_async 拿回所有请求的增量），
#      而不是 N 个连接 —— 这是 V1 在线路径吞吐高的关键设计。
#   2) **请求靠前台推**：add_request 是 await 的，因为它要发消息给引擎。
#
# 核心内容速查：
#   - AsyncLLM.__init__        : 组装三大件 + 惰性启动 output_handler
#   - add_request / _add_request: 入队，返回 RequestOutputCollector（输出队列）
#   - generate / encode        : 对外主入口，async generator，含完整的错误分类处理
#   - _run_output_handler      : 后台循环（分块处理 + 让出事件循环）
#   - check_admission          : 准入控制（超限直接 503，让 LB 重试别的实例）
#   - pause/resume_generation  : 为在线更新权重而暂停
#   - scale_elastic_ep         : MoE 弹性扩缩容
#   - *_weight_update / init_weight_transfer_engine : RL 场景的在线权重更新
#
# 阅读提示（几个容易踩的点）：
#   1. **循环引用陷阱**：output_handler 这个 asyncio.Task 如果直接捕获 self，
#      就形成 self -> task -> closure -> self 的环，AsyncLLM 永远不会被 GC，
#      引擎进程也不会退出。所以 _run_output_handler 里把需要的属性**先取成局部
#      变量**再让闭包捕获；logger_manager 因为运行中会被替换（elastic EP 扩缩容），
#      不能取快照，于是用了一个"单元素列表" self._logger_ref 做间接层。
#      读代码时看到这些看似多余的局部变量，原因都在这里。
#   2. output_handler **惰性启动**：构造时如果不在事件循环里（get_running_loop
#      抛 RuntimeError）就跳过，等第一次 add_request 或 generate 再启动。
#      这样可以在启动事件循环之前就构造 AsyncLLM —— OpenAI 服务器需要这个能力，
#      才能在引擎启动失败时给出干净的报错而不是崩在一个孤儿 task 上。
#   3. 输出**分块**处理（VLLM_V1_OUTPUT_PROC_CHUNK_SIZE）：一次 step 可能带回
#      几百条请求的输出，全部处理完会长时间占住事件循环，让其它协程（新请求、
#      健康检查）饿死。所以切成小块，块间 await asyncio.sleep(0) 主动让出。
#   4. generate() 的异常分支顺序**不能调换**：CancelledError/GeneratorExit
#      （客户端断开）要 abort；EngineDeadError 不能 abort（引擎都没了）；
#      客户端错误（参数错/被限流）不 abort（请求压根没进引擎）。
# ==============================================================================
import asyncio
import os
import socket
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any

import vllm.envs as envs
from vllm import TokensPrompt
from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.serve.elastic_ep.middleware import set_scaling_elastic_ep
from vllm.exceptions import (
    GracefulHTTPError,
    MaxQueuedTokensError,
    QueueOverflowError,
    VLLMClientError,
    VLLMValidationError,
)
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import STREAM_FINISHED, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class InputStreamError(Exception):
    """Wrapper for errors from the input stream generator.

    This is used to propagate errors from the user's input generator
    without wrapping them in EngineGenerateError.
    """

    # [CN] 为什么需要这层包装：流式输入（AsyncGenerator）的异常是**用户代码**抛的，
    #      比如用户生成 prompt 时访问数据库失败。这种错误不能和"引擎内部错误"
    #      混为一谈（后者会被包装成 EngineGenerateError 并返回 500）。
    #      generate() 里单独 catch 它，然后用 `raise e.cause from e`
    #      把**原始异常**原样抛回去，用户看到的是自己的报错信息。

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(str(cause))


class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""

    # [CN] 补充说明（它与 LLMEngine 的关系）：
    #   - 两者都站在"前端进程"这一侧，都不跑模型；区别只有**并发模型**：
    #     LLMEngine.step() 阻塞推进，AsyncLLM 用 asyncio 事件循环驱动。
    #   - AsyncLLM 是 EngineClient 协议的**唯一实现**（vllm/engine/protocol.py），
    #     所以在线服务、RL 框架、测试都面向它编程。
    #   - 它只支持**多进程**模式（make_async_mp_client），EngineCore 一定在
    #     另一个进程 —— 因为异步服务里绝不能让模型推理卡住事件循环。
    #   - 与 LLMEngine 相比，它多了：准入控制、流式输入、暂停/恢复、弹性扩缩容、
    #     在线权重更新（RL）—— 这些都是**在线服务**才需要的运维能力。

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        profiler: TorchProfilerWrapper | None = None,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # [CN] 补充几个__init__ 参数里容易忽略的：
        #   start_engine_loop    : 是否启动后台 output_handler（测试里常传 False
        #                          以便手动驱动）。
        #   client_count/index   : **多前端**模式。多个 API server 进程共享同一个
        #                          EngineCore 时，每个前端有一个 index，引擎靠它把
        #                          输出原路送回（对应 EngineCoreRequest.client_index）。
        #   client_addresses     : 多前端模式下各前端的地址表（ZMQ 连接串）。
        #   aggregate_engine_logging : 多前端时是否把指标聚合后再打日志。

        # Ensure we can serialize custom transformer configs
        # [CN] 让 msgspec 用"按值序列化"处理自定义的 transformers config：
        #      vLLM 要在进程间传 EngineCoreRequest，而用户自定义的 config 类默认
        #      走"按引用"（pickle 引用模块+类名），在子进程里可能 import 不到，
        #      注册之后就变成连数据一起序列化，跨进程更安全。
        maybe_register_config_serialize_by_value()

        self.vllm_config = vllm_config
        # [CN] 弹性扩缩容的互斥锁：scale_elastic_ep 期间不能同时来第二次，
        #      否则两个 reconfigure 流程会互相踩（改 DP size 是有状态的多步操作）。
        self._elastic_ep_lock = asyncio.Lock()
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        # [CN] 一个容易困惑的语义：log_stats=False 只表示"不用**默认**日志器"，
        #      但如果用户通过插件/参数提供了自定义日志器，指标采集必须仍然开着，
        #      否则自定义日志器收不到任何数据。所以这里是 `or` 而不是直接用入参。
        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (starts the engine in background process).
        # Hand the renderer to the client so it can start the frontend MM
        # warmup only after engine-core fork (the why is in
        # BaseRenderer.start_mm_warmup_in_background). The warmup is joined
        # by reset_mm_cache / warmup / shutdown.
        # [CN] 与 LLMEngine 的关键差异：这里**只能**用 make_async_mp_client，
        #      即 EngineCore 一定在独立进程里。异步服务绝不能让模型执行
        #      （几十到几百毫秒的 CUDA 计算）卡住事件循环。
        #      这一步会拉起子进程：建分布式环境、加载权重、profile 显存、
        #      分配 KV cache、捕获 CUDA graph。
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            renderer=renderer,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                # [CN] engine_idxs 决定指标按哪些"引擎编号"分组。多前端模式下
                #      本前端只管一部分 rank，所以指标也只统计自己管的那些，
                #      否则 N 个前端会重复上报 N 份一样的指标。
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self._client_count = client_count

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            # [CN] 已经在事件循环里（典型：在 async 函数中构造）-> 立刻启动后台任务。
            self._run_output_handler()
        except RuntimeError:
            # [CN] 不在事件循环里（典型：在同步代码里构造 AsyncLLM，比如
            #      OpenAI server 的 startup 阶段）。这时不能建 task（没有 loop），
            #      先跳过；等第一次 add_request / generate 时再启动。
            #      这换来一个很重要的能力：**引擎启动失败可以在事件循环之外
            #      被正常捕获和处理**，而不是留一个永远没人 await 的孤儿 task。
            pass

        self.profiler = profiler
        # [CN] 前端进程自己也开一个 torch profiler（只抓 CPU 活动）。
        #      为什么值得：tokenize、detokenize、事件循环调度这些**前端开销**
        #      在只看引擎侧的 trace 时是看不见的，但它们同样影响端到端延迟。
        #      ignore_frontend=True 就是关掉它（只要引擎侧 trace 时用）。
        #      注意 worker_name 里带了 hostname + pid：多进程/多机时用来区分来源。
        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = TorchProfilerWrapper(
                vllm_config.profiler_config,
                worker_name=worker_name,
                local_rank=0,
                activities=["CPU"],
            )

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        """[CN] 从 VllmConfig 直接构造（跳过 EngineArgs 解析）。
        start_engine_loop=False 时**不启动** output_handler：
        测试里常用这个开关来手动驱动引擎（一步一步断言输出），
        避免后台任务的输出时机不确定。
        """
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown, cleaning up the background proc and IPC."""
        # [CN] 关闭顺序有讲究，且**每一步都用 getattr 兜底**：
        #   1) prometheus（HTTP 指标端口）
        #   2) renderer（多模态后台线程）
        #   3) engine_core（真正去终止 EngineCore 子进程 / 关 ZMQ）
        #   4) output_handler（取消后台 task）
        # 用 getattr 的原因和 LLMEngine.__del__ 一样：__init__ 可能半途失败，
        # 那时这些属性还不存在，而 __del__ 里再抛异常只会制造噪音。
        shutdown_prometheus()

        if renderer := getattr(self, "renderer", None):
            renderer.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown(timeout=timeout)

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    def get_num_unfinished_requests(self) -> int:
        """[CN] 本前端还有多少条没跑完的请求。
        注意这是**本前端**的视图：多前端模式下别的进程可能还有请求在跑。"""
        return self.output_processor.get_num_unfinished_requests()

    def get_num_queued_tokens(self) -> int:
        """[CN] 还在 prefill（尚未产出第一个 token）的请求的 prompt 总长度。
        用于下面的 max_num_queued_tokens 准入控制。"""
        return self.output_processor.get_num_queued_tokens()

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        """Reject the request if it would exceed queue limits.

        Both limits return HTTP 503 (Service Unavailable) so that load
        balancers and client SDKs retry on a different instance.

        - ``max_num_queued_reqs``: hard cap on the number of unfinished requests
          (waiting + running).  A request with ``n > 1`` counts as ``n`` slots.
        - ``max_num_queued_tokens``: TTFT QoS — cap on the total prompt
          tokens of requests still in prefill.

        Note: ``get_num_queued_tokens`` uses ``prompt_len`` for all prefilling requests.
        Chunked prefill progress and prefix-cache hits are not subtracted because the
        scheduler's ``num_computed_tokens`` and ``num_cached_tokens`` are only
        propagated to the API server after prefill completes. The overestimation is
        conservative — earlier rejection, preserving TTFT targets.

        Args:
            n: Number of sequences the request will occupy.
            request_id: Request id, used for logging only.

        Raises:
            QueueOverflowError: If ``max_num_queued_reqs`` would be exceeded.
            MaxQueuedTokensError: If ``max_num_queued_tokens`` would be exceeded.

        [CN] 两个 limit 的区别，务必分清：
          - max_num_queued_reqs   ：按**请求条数**限流（保护调度队列不无限增长，
                                     属于"保护引擎不被打爆"）。
          - max_num_queued_tokens ：按**还在 prefill 的 token 总数**限流，
                                     目的是保护 **TTFT**（首 token 时延）：
                                     prefill 是算力密集的，队列里堆的 prompt token
                                     越多，新请求等得越久。
        都返回 503 是**故意的**：503 带"稍后重试"语义，负载均衡器和 OpenAI SDK
        会自动换一个实例重试，比在队列里干等（最后超时）体验好得多。
        另一个易错点：这里用的是 `current_tokens >= max`（而不是 +n 之后比较），
        因为还没算出本请求的 prompt 长度就不该硬算；换句话说这条判断偏保守，
        宁可多拒也不让 TTFT 恶化。
        """
        max_num_reqs = self.scheduler_config.max_num_queued_reqs
        if max_num_reqs is not None:
            current = self.get_num_unfinished_requests()
            if current + n > max_num_reqs:
                logger.info(
                    "Request queue full - rejecting request %s "
                    "(current=%d, n=%d, max=%d).",
                    request_id,
                    current,
                    n,
                    max_num_reqs,
                )
                raise QueueOverflowError()

        max_queued_tokens = self.scheduler_config.max_num_queued_tokens
        if max_queued_tokens is not None:
            current_tokens = self.get_num_queued_tokens()
            if current_tokens >= max_queued_tokens:
                logger.info(
                    "Max queued tokens reached - rejecting request %s "
                    "(current_tokens=%d, max=%d).",
                    request_id,
                    current_tokens,
                    max_queued_tokens,
                )
                raise MaxQueuedTokensError()

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()

        return self._supported_tasks

    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""
        # [CN] 注意返回值不是 RequestOutput，而是 **RequestOutputCollector** ——
        #      一个"输出队列"句柄。调用方（generate）再去这个队列上迭代取输出。
        #      这样设计的好处：add_request 只负责"把请求送进去"，取输出是另一回事，
        #      于是"批量提交 N 个请求再一起收结果"这种用法成为可能（在线服务里很常见）。

        # [CN] 引擎已经死了（后台 task 挂了 / 子进程没了）就直接拒绝新请求，
        #      否则会永远等一个不存在的输出 —— 这里的快速失败非常重要。
        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and not is_pooling
            and params.prompt_logprobs
        ):
            raise VLLMValidationError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        if isinstance(params, SamplingParams) and params.n > 1:
            # TODO (NickLucche) Batch check admission check for all n requests
            # [CN] n>1 会扇出成 n 条序列，占 n 个"槽位"，所以要按 n 做准入检查。
            self.check_admission(params.n, request_id)

        # [CN] 流式输入：prompt 本身是一个 AsyncGenerator（边生成边喂）。
        #      典型场景是"会话流式续写"、"边下载边推理"这类用法。
        #      这条路完全不同于普通请求（见 _add_streaming_input_request）。
        if isinstance(prompt, AsyncGenerator):
            if reasoning_ended is not None or reasoning_parser_kwargs is not None:
                raise NotImplementedError

            # Streaming input case.
            return await self._add_streaming_input_request(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
                session_id,
            )

        # Convert Input --> Request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in the future. You should "
                "instead pass the outputs of Renderer.render_cmpl() or "
                "Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            if isinstance(prompt, dict) and "type" in prompt:
                # Rendered EngineInput; no blocking preprocessing needed.
                # [CN] 已经"渲染过"的输入（含 "type" 字段，由 Renderer 产出）：
                #      tokenize、多模态处理都已完成，没有阻塞操作，直接同步处理即可。
                #      这是**推荐路径** —— 见 generate() docstring 里的示例。
                request = self.input_processor.process_inputs(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                    session_id=session_id,
                )
            else:
                # Raw prompts require tokenization and possibly multimodal
                # processing, which must not block the event loop.
                # [CN] 原始 prompt（纯文本 / 带图片的 dict）需要 tokenize，
                #      多模态还要做图像处理 —— 这些都是**CPU 密集**操作，
                #      直接在协程里做会把整个事件循环卡住（所有并发请求一起卡）。
                #      所以走 _async 版本：丢到线程池里执行。
                request = await self.input_processor.process_inputs_async(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                    session_id=session_id,
                )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        if reasoning_ended is not None:
            request.reasoning_ended = reasoning_ended
        if reasoning_parser_kwargs is not None:
            request.reasoning_parser_kwargs = reasoning_parser_kwargs

        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        # [CN] 惰性启动后台 output_handler（如果 __init__ 时没启动的话）。
        #      这个方法内部有幂等保护（已启动就直接 return），所以可以放心多次调用。
        self._run_output_handler()

        # Create a new output collector for the request.
        # [CN] output_kind 决定这个队列的行为：
        #   FINAL_ONLY   : 只在请求结束时放一个结果（非流式）
        #   DELTA        : 每有增量就放一个（流式 SSE）
        #   CUMULATIVE   : 每有增量就放一个"从开头到现在的全量"（某些 SDK 需要）
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # [CN] 准入检查只在"父请求首次进入"时做：n>1 的子请求不应重复计数
        #      （它们本来就在 add_request 里按 n 一次性检查过了）。
        if parent_req is None and not self.output_processor.has_request(
            request.request_id
        ):
            self.check_admission(request_id=request.request_id)

        # Register locally before the first await so concurrent tasks see this request.
        # [CN] 这一行**必须在 await 之前**，而且不是可选项：
        #      如果在 await 之后才注册，那么"请求已提交"和"请求已登记"之间有一个
        #      时间窗；引擎完全可能在这窗口里就吐回了第一个 token，
        #      而此时本地还没有 RequestState -> 输出会被当成"未知请求"丢弃。
        #      并发提交时这个竞态会稳定复现（尤其 n>1 扇出）。
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    async def _add_streaming_input_request(
        self,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
    ) -> RequestOutputCollector:
        """[CN] 流式输入请求：prompt 是一个 AsyncGenerator[StreamingInput]。

        整体思路（读这段代码前先理解这个模型）：
          - 用户的 generator 每 yield 一次，就产生**一条独立的 EngineCoreRequest**；
          - 这些请求共用同一个内部 request_id（internal_req_id），且 resumable=True，
            引擎侧会把它们当作"同一个流式会话的连续输入"依次处理；
          - 用户 generator 结束后，再发一条**空的 final_req** 作为"输入结束"信号，
            引擎收到后就结束这个会话（这就是为什么 final_req 的 prompt 是 [0]）。
        """
        self._validate_streaming_input_sampling_params(sampling_params)

        inputs = dict(
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            session_id=session_id,
        )

        # [CN] 克隆一份采样参数并打上 skip_clone 标记：后续每个 chunk 都会复用它，
        #      如果每次都重新 clone，几万个 chunk 会产生几万个对象（GC 压力）；
        #      skip_clone=True 是"我已经是独占副本了，别再拷"的信号。
        if not sampling_params.skip_clone:
            sampling_params = sampling_params.clone()
            sampling_params.skip_clone = True

        # Create request for validation, also used as the finished signal
        # once the input stream is closed.
        # [CN] prompt 用 [0] 这个占位 token：这条请求的唯一目的是**校验参数合法**
        #      （长度、max_tokens 等）和作为"输入流结束"的信号，
        #      真正的输入在下面 handle_inputs 里逐块发。
        final_req = self.input_processor.process_inputs(
            request_id=request_id,
            prompt=TokensPrompt(prompt_token_ids=[0]),
            params=sampling_params,
            **inputs,  # type: ignore[arg-type]
        )
        self.input_processor.assign_request_id(final_req)
        internal_req_id = final_req.request_id

        queue = RequestOutputCollector(sampling_params.output_kind, internal_req_id)

        async def handle_inputs():
            """[CN] 后台任务：消费用户的输入 generator，逐块转成请求发给引擎。"""
            cancelled = False
            try:
                async for input_chunk in input_stream:
                    sp = input_chunk.sampling_params
                    if sp:
                        self._validate_streaming_input_sampling_params(sp)
                    else:
                        sp = sampling_params
                    # TODO(nick): Avoid re-validating reused sampling parameters
                    req = self.input_processor.process_inputs(
                        request_id=internal_req_id,
                        prompt=input_chunk.prompt,
                        params=sp,
                        resumable=True,
                        **inputs,  # type: ignore[arg-type]
                    )
                    # [CN] 每块都用**同一个** internal_req_id：引擎侧据此把它们
                    #      归并到同一个流式会话；但 external_req_id 保持用户给的
                    #      那个，所以对外看来始终是一个请求。
                    req.external_req_id = request_id
                    if req.prompt_embeds is not None:
                        raise VLLMValidationError(
                            "prompt_embeds not supported for streaming inputs"
                        )
                    prompt_text, _, _ = extract_prompt_components(
                        self.model_config, input_chunk.prompt
                    )
                    await self._add_request(req, prompt_text, None, 0, queue)
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                # Wrap in InputStreamError so generate() can propagate it
                # without wrapping in EngineGenerateError.
                queue.put(InputStreamError(error))
            finally:
                queue._input_stream_task = None
                if not cancelled:
                    # Send empty final request to indicate that inputs have
                    # finished. Don't send if cancelled (session was aborted).
                    # [CN] 被取消时不发 final：会话已经被 abort 了，
                    #      再发一条"输入结束"反而会重新激活一个已死的会话。
                    await self._add_request(final_req, None, None, 0, queue)

        # Ensure output handler is running.
        self._run_output_handler()

        queue._input_stream_task = asyncio.create_task(handle_inputs())
        return queue

    @staticmethod
    def _validate_streaming_input_sampling_params(
        params: SamplingParams | PoolingParams,
    ):
        """[CN] 流式输入目前有四项硬限制，原因分别是：
          - 池化模型（PoolingParams）：池化是"一次前向出结果"，没有"逐块续写"语义；
          - n > 1：多序列扇出与"边喂边生成"的状态管理冲突；
          - FINAL_ONLY：流式输入必须边生成边看，只给最终结果没意义；
          - stop strings：stop 检测需要跨 chunk 的上下文，实现复杂度高，暂不支持。
        这些限制未来可能放开，但都是"语义上难以自洽"而非"懒得实现"。
        """
        if (
            not isinstance(params, SamplingParams)
            or params.n > 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or params.stop
        ):
            raise VLLMValidationError(
                "Input streaming not currently supported "
                "for pooling models, n > 1, request_kind = FINAL_ONLY "
                "or with stop strings."
            )

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
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
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.

        Note:
            Passing a raw prompt string directly to this method is deprecated.
            Advanced power-users can manually bypass the raw-prompt fallback
            path using the Engine's underlying Renderer pipeline:

            >>> from vllm.inputs import parse_model_prompt
            >>> parsed = parse_model_prompt(self.model_config, "Prompt text")
            >>> params = self.renderer.default_cmpl_tok_params
            >>> (engine_input,) = self.renderer.render_cmpl([parsed], params)
            >>> gen = self.generate(engine_input, sampling_params, request_id)

        [CN] 中文要点补充：
          1) 这是一个 **async generator**：调用它只是得到一个生成器对象，
             请求是在第一次 `async for` 迭代时才真正发出的（惰性）。
          2) 为什么不推荐再传原始字符串：tokenize / 多模态处理是阻塞操作，
             放在异步路径里会卡事件循环。用 Renderer 提前渲染成 EngineInput，
             这一步就可以在需要时同步完成（见 _add_request 里的分支注释）。
          3) 返回的 RequestOutput 是增量的：非流式（FINAL_ONLY）只有一个，
             流式（DELTA）会有很多个，最后一个带 finished=True。
          4) **取消语义**：客户端断开时这个生成器会被 cancel 或被 GC，
             下面的 except 分支会捕获并 abort 请求 —— 这是在线服务里
             "用户中途关掉网页，GPU 算力要立刻还给别人"的关键实现。
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                session_id=session_id,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            # [CN] 分工：**后台 output_handler 生产，本协程消费**。
            #      整条在线链路上只有这一个后台任务跟 EngineCore 通信。
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                # [CN] 先试非阻塞的 get_nowait()：如果队列里已经有货，直接拿走，
                #      不 await 就不会让出事件循环（省掉一次协程切换）。
                #      高负载下这个优化很可观 —— 否则每个 token 都要切一次。
                #      注意 `or` 的写法利用了 get_nowait() 无货时返回 None。
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                # [CN] 这里不需要手动清理：OutputProcessor 和 EngineCore 各自
                #      会根据 finished 做收尾（前者删 RequestState，后者放 block）。
                assert isinstance(out, RequestOutput)
                # [CN] 注意 out 可能是 STREAM_FINISHED 哨兵而不是真实输出：
                #      它被定义成 RequestOutput(request_id="", outputs=[],
                #      finished=True)（见 vllm/outputs.py），所以上面的 isinstance
                #      断言成立。它只在**流式输入**场景出现：输入流结束时
                #      collector 塞它进来，用于终止这个生成器；
                #      下面 `if out is not STREAM_FINISHED` 保证它不会被 yield 出去
                #      （用户不该看到一个空壳输出）。
                finished = out.finished
                if out is not STREAM_FINISHED:
                    yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        # [CN] 客户端断开（关网页 / 超时 / SDK 主动取消）时走到这里。
        #      **必须 abort**：否则引擎还在给一个没人要的请求生成 token，
        #      白白占着 batch 槽位和 KV block。在线服务里这是最常见的分支之一。
        #      注意用 internal=True：q.request_id 是内部 id（可能带 n>1 后缀）。
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        # [CN] 引擎已经死了就**不要** abort：abort 要发消息给引擎，
        #      而它已经收不到了（只会白白等超时，甚至抛第二次异常掩盖真正的原因）。
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error or admission control rejection.
        # [CN] 客户端自己的问题：参数不合法（VLLMClientError）或被限流/优雅拒绝
        #      （GracefulHTTPError，如 503 队列满）。同样**不 abort**：
        #      请求根本没进引擎，没什么可撤的；原样抛出让 API 层转成 4xx/503。
        except (VLLMClientError, GracefulHTTPError) as e:
            if self.log_requests:
                logger.info("Request %s failed (bad request): %s.", request_id, e)
            raise

        # Error from input stream generator - propagate directly.
        # [CN] 用户自己的输入 generator 抛异常：abort（请求已进引擎）+ 把
        #      **原始异常**抛回去（raise e.cause from e），不包装成引擎错误，
        #      这样用户看到的是自己代码的堆栈。
        except InputStreamError as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed (input error): %s.", request_id, e)
            raise e.cause from e

        # Unexpected error in the generate() task (possibly recoverable).
        # [CN] 兜底分支：引擎**可能还活着**（错误只是本请求的问题），所以要 abort，
        #      并且抛出 EngineGenerateError（对外 500）而不是让原始异常泄漏出去
        #      —— 避免把内部实现细节暴露给 API 调用方。
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                try:
                    s = f"{e.__class__.__name__}: {e}"
                except Exception as e2:
                    # [CN] 连 __str__ 都会抛的异常（比如某些 CUDA / C++ 层的错误
                    #      对象状态已损坏）也要能安全打日志，不能因为打日志再崩一次。
                    s = (
                        f"{e.__class__.__name__}: "
                        "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                logger.info("Request %s failed due to %s.", request_id, s)
            raise EngineGenerateError() from e
        finally:
            # [CN] 无论走哪条分支都要 close 队列：它会唤醒可能还在 await q.get()
            #      的协程并释放资源；漏掉会让 collector 悬挂
            #      （进而泄漏 OutputProcessor 里的 RequestState）。
            if q is not None:
                q.close()

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        # [CN] 幂等：已经启动过就直接返回（add_request 每次都会调本方法）。
        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        # [CN] **本文件最重要的一处设计**，务必理解：
        #      下面的 output_handler() 是个闭包。如果它直接用 self.xxx，
        #      就会形成 self -> output_handler(Task) -> closure -> self 的环。
        #      这个环里有 asyncio.Task，而 Task 不参与引用计数回收的那套逻辑，
        #      于是 AsyncLLM 永远不会被 GC，引擎进程也退不掉（典型"僵尸进程"）。
        #      解法：把闭包需要的东西**先取成局部变量**，闭包只捕获这些局部变量，
        #      不再持有 self。
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        # We use a mutable list for logger_manager so that it can be updated
        # during elastic EP scaling (see scale_elastic_ep) without creating
        # a circular reference via self.
        # [CN] logger_manager 是**唯一不能取快照**的东西：弹性扩缩容时会替换它
        #      （见 scale_elastic_ep）。所以不能直接 `logger_manager = self.logger_manager`
        #      （那样闭包会一直用旧的），又不希望闭包捕获 self。
        #      于是用"单元素列表"做一层间接：闭包捕获的是这个 list 对象，
        #      替换时改 `self._logger_ref[0] = new_manager`，闭包那边自然看到新值。
        #      这是个经典的"用可变容器打破不可变绑定"的技巧。
        self._logger_ref = [self.logger_manager]
        logger_ref = self._logger_ref
        renderer = self.renderer
        # P0 multi-modal sender ("shadow") cache; None for text-only models.
        # [CN] P0 = 前端进程，P1 = EngineCore 进程。前端保留一份多模态缓存的
        #      "影子副本"，目的是**避免重复上传大图**：命中缓存时只发 hash。
        #      两份缓存可能"漂移"（P1 重启/淘汰了而 P0 还以为有），
        #      下面 2b) 就是漂移的修复逻辑。纯文本模型这里是 None。
        mm_processor_cache = renderer.mm_processor_cache
        # [CN] 一次 step 的输出分块处理的大小（默认很大，所以通常不分块）。
        #      目的是防止"一次处理几千条输出"长时间占住事件循环。
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        async def output_handler():
            """[CN] 后台主循环（每个 AsyncLLM 只有一个）：
            拉输出 -> 分块处理 -> 回撤 stop 请求 -> 更新统计 -> 记录指标。
            它与 generate() 之间靠 asyncio 队列解耦。
            """
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    # [CN] 这里 await 会挂起，直到引擎产出一帧 —— 所以本任务
                    #      平时是"睡眠"的，不消耗 CPU。
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        # [CN] 为什么这里断言 request_outputs 为空：
                        #      在异步路径里，OutputProcessor **不返回**输出对象，
                        #      而是直接把 RequestOutput 推进每条请求自己的
                        #      RequestOutputCollector 队列（构造时传进去的）。
                        #      这跟同步 LLMEngine 的"返回 list"是两种模式。
                        assert not processed_outputs.request_outputs

                        # 2b) Recover from P0/P1 cache drift: the engine flags hashes
                        # it couldn't find (mm_cache_miss_hashes); drop them from the
                        # P0 shadow so the client's retry resends the data and
                        # repopulates P1. Hot-path no-op (field is None otherwise).
                        if mm_processor_cache is not None:
                            for eco in outputs_slice:
                                if eco.mm_cache_miss_hashes:
                                    for mm_hash in eco.mm_cache_miss_hashes:
                                        mm_processor_cache.invalidate(mm_hash)

                        # Allow other asyncio tasks to run between chunks
                        # [CN] asyncio.sleep(0) 的语义是"让出一次事件循环调度权"，
                        #      不是真的睡一段时间。这样正在等待的新请求、健康检查、
                        #      abort 等协程都有机会插进来执行 —— 高并发下
                        #      没有它会出现明显的尾延迟毛刺。
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_ref[0]:
                        logger_ref[0].record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=renderer.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                # [CN] 后台任务挂了 = 整个引擎不可用了。这里把错误**广播**给所有
                #      还在等待输出的请求（propagate_error 会往每个队列里塞异常），
                #      否则它们会永远 await 下去 —— 那比直接报错难查得多。
                #      之后 self.errored 变 True，新请求会被 add_request 直接拒绝。
                output_processor.propagate_error(e)

        # [CN] 注意没有保存 task 的强引用之外的东西：self.output_handler 持有它，
        #      但 __del__/shutdown 里会 cancel，避免"孤儿 task"泄漏。
        self.output_handler = asyncio.create_task(output_handler())

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        # [CN] 与 LLMEngine.abort_request 同一套路：先本地摘状态（返回真正需要
        #      abort 的 id），再通知引擎。顺序不能反，否则引擎的"最后一批输出"
        #     可能回来时发现本地状态已删（虽然不致命，但会产生未知请求告警）。
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Submit a pre-aborted request so the connector's request_finished
        hook runs to free any pre-admission KV-transfer resources (e.g. NIXL
        prefill blocks pinned on the P node)."""
        # [CN] 这是一个很"绕"但必要的技巧，值得单独理解：
        #      P/D 分离场景下，P 节点（prefill）可能已经为这个请求**预留**了
        #      KV block，但 D 节点（decode）在准入检查时把它拒了。
        #      此时 D 节点本地没有任何这个请求的状态，连接器（connector）的
        #      request_finished 钩子就永远不会被调用 -> P 节点那批 block 泄漏。
        #      解决：造一个 abort_immediately=True 的"空请求"发给引擎，
        #      引擎会走完整的入队 -> request_finished 流程，从而触发 P 侧清理。
        #      （对应 EngineCoreRequest.abort_immediately 字段的注释。）
        request = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[0],
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": dict(kv_transfer_params)},
            ),
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=data_parallel_rank,
            abort_immediately=True,
        )
        await self.engine_core.add_request_async(request)

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool | None = None,
        clear_cache: bool = True,
    ) -> None:
        """
        Pause generation to allow model weight updates.

        All mode handling (abort / wait / keep) and cache clearing is done
        in the engine. New generation/encoding requests will not be scheduled
        until resume is called.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED: use mode argument.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
        """
        if wait_for_inflight_requests:
            warnings.warn(
                "The `wait_for_inflight_requests` parameter in "
                "`AsyncLLM.pause_generation()` is deprecated. "
                "Please use `mode` argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "wait"
        # [CN] 典型使用场景：**RL 在线更新权重** —— 先暂停推理（让显存里的
        #      KV cache 状态稳定下来），训练侧更新权重，再 resume。
        if clear_cache:
            await self.renderer.clear_mm_cache_async()
        await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
        # Small sleep to help ensure that final outputs from any in-flight requests are
        # returned prior to this method returning. These outputs come out of the engine
        # prior to the wait-for-idle completion event, but involve additional async
        # tasks in output processing.
        # Note that this is not required for correctness, just more intuitive ordering
        # of events from caller's pov.
        # [CN] 为什么要睡 20ms：引擎侧的"暂停完成"事件比"最后一批输出"更早到达
        #      （输出还要经过 output_handler 这个异步任务）。
        #      不睡的话，调用方可能在暂停返回后**才**收到最后几个 token，
        #      时序上看起来很怪。这里纯粹是为了**事件顺序更符合直觉**，
        #      正确性不依赖它（所以注释里明确写了 not required for correctness）。
        await asyncio.sleep(0.02)

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""
        await self.engine_core.resume_scheduler_async()

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        return await self.engine_core.is_scheduler_paused_async()

    async def encode(
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
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.

        [CN] encode() 是 generate() 的**池化版本**：给 embedding / classify /
             reward / score 这类任务用。实现几乎与 generate() 逐行对应，
             主要差别：
             1) 参数是 PoolingParams（没有 n / stop / logprobs 等采样概念）；
             2) 产出 PoolingRequestOutput（一次就结束，没有"流式"概念，
                所以循环里直接 yield out，不判断 STREAM_FINISHED）；
             3) 异常分支少一个 InputStreamError（池化不支持流式输入）。
             读 generate() 的注释即可，这里不再重复。
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                reasoning_ended=reasoning_ended,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error or admission control rejection.
        except (VLLMClientError, GracefulHTTPError):
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        return self.observability_config.otlp_traces_endpoint is not None

    async def do_log_stats(self) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        """[CN] 供 Kubernetes / 负载均衡的健康检查（/health）调用。
        判据只有一条：**出错了就抛 EngineDeadError**。
        注意它不检查"引擎忙不忙"——忙是正常的，那是 /ready 或队列长度的事。"""
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self, profile_prefix: str | None = None) -> None:
        # [CN] 同时开两个 profile：
        #   1) EngineCore 侧的 torch profiler（在引擎进程里，抓 GPU/模型执行）；
        #   2) 本进程（前端）的 profiler（抓 tokenize、detokenize、事件循环开销）。
        # 用 asyncio.to_thread 是因为 profiler.start() 是**阻塞**的（会做 CUDA 同步）。
        # 用 gather 并发执行，两边同时开始，时间轴才能对齐。
        coros = [self.engine_core.profile_async(True, profile_prefix)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        # Join the background MM warmup first: the mm_processor_cache is not
        # safe for concurrent access with its apply/clear.
        # [CN] 与 LLMEngine.reset_mm_cache 完全同构，只是 clear 换成了 async 版本
        #      （多模态清理可能涉及线程池/IO）。先 join 预热线程的原因见那边。
        self.renderer._join_mm_warmup()
        await self.renderer.clear_mm_cache_async()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def reset_encoder_cache(self) -> None:
        await self.engine_core.reset_encoder_cache_async()

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        if level >= 1:
            await self.renderer.clear_mm_cache_async()
        await self.engine_core.sleep_async(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def checkpoint_prepare(self) -> None:
        """[CN] 为"做检查点"做准备（让各 worker 进入可保存状态，如暂停 CUDA graph、
        同步流）。与 sleep/wake_up 配合，用于**保存/恢复引擎快照**，
        避免重复加载模型（快速拉起）。"""
        await self.collective_rpc("checkpoint_prepare")

    async def checkpoint_restore(self) -> None:
        await self.collective_rpc("checkpoint_restore")

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        # [CN] 同 LLMEngine：这是一次**跨进程同步 RPC**（要等所有 worker 做完），
        #      是慢调用（可能几百毫秒），不要在请求热路径上用。
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        # [CN] "排空"：等到所有 DP engine 都空转（没有在跑的请求）为止。
        #      弹性扩缩容之前必须排空 —— 因为扩缩容要重建通信域，
        #      在途请求会在半途失去通信伙伴。
        #      注意是**轮询**（每秒一次）而不是事件驱动：简单可靠，
        #      且排空本来就是低频操作（几秒的误差无所谓）。
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def _drain_requests_for_elastic_ep(self, drain_timeout: int) -> None:
        try:
            logger.info(
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "
                "waiting for requests to drain before scaling"
            )
            await self.wait_for_requests_to_drain(drain_timeout)
        except BaseException:
            # [CN] 排空失败（超时/被取消）时必须**复位**"正在扩缩容"标志：
            #      这个标志会影响 API server 的中间件行为（比如拒绝新请求、
            #      返回 503 让 LB 重试）。不复位的话服务会永远停在"扩缩容中"。
            #      用 BaseException 而不是 Exception：也要兜住 CancelledError。
            set_scaling_elastic_ep(False)
            raise

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """Scale the elastic EP data parallel size."""
        # [CN] **在线改变 DP 规模**（MoE 专家并行的弹性扩缩容）。
        #      这是两阶段提交式的流程（读代码时按这个顺序看）：
        #        1) prepare_elastic_ep：引擎侧准备好新拓扑所需的资源/配置，**还没切换**；
        #        2) （可选）排空在途请求；
        #        3) commit_elastic_ep：真正切换，重建通信域，多余的 rank 退出。
        #      加锁的原因：整个流程跨越多次 await，中途不能被第二次 scale 插进来。
        async with self._elastic_ep_lock:
            await self._scale_elastic_ep(new_data_parallel_size, drain_timeout)

    async def _scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int
    ) -> None:
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return

        await self.engine_core.prepare_elastic_ep(new_data_parallel_size)

        # recreate stat loggers
        # [CN] 只在**扩容**时重建指标日志器：因为指标要按 engine 序号（engine_idx）
        #      分组，扩容后序号变多了，旧日志器不知道新序号。
        #      缩容时不重建（多余的指标序列留着无害，重建反而会清空历史数据）。
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )
            # Update the mutable ref so output_handler picks up the
            # new logger without creating a circular reference via self.
            if hasattr(self, "_logger_ref"):
                self._logger_ref[0] = self.logger_manager
            self.logger_manager.log_engine_initialized()

        set_scaling_elastic_ep(True)
        if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:
            await self._drain_requests_for_elastic_ep(drain_timeout)

        await self.engine_core.commit_elastic_ep()
        # [CN] 引擎侧切换成功后，才更新本地配置里的 DP size ——
        #      顺序不能反：中途失败时本地配置仍然描述"旧的、仍然生效"的拓扑。
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        set_scaling_elastic_ep(False)

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """send fault tolerance instruction to the engine"""
        # [CN] 容错指令通道：查询/注入/清除故障（用于 FT 测试与"某张卡挂了
        #      之后集群如何降级"的演练）。对应 FT_STATUS_CALL_ID 那条通道。
        return await self.engine_core.handle_fault(fault_tolerance_request)

    async def get_status(self):
        """[CN] 取引擎状态（EngineStatusType：HEALTHY / DEAD / UNHEALTHY）。"""
        return await self.engine_core.get_status()

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        # [CN] 两个状态在这里被故意合并成一个布尔：
        #   - output_handler is None：还没启动（惰性启动，见 __init__）；
        #   - 已启动但 .done() 为 False：正常在跑。
        #   所以"没启动"也算 is_running=True —— 含义是"没有异常地停下来了"，
        #   而不是"正在跑"。读代码时别被名字误导。
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        """[CN] 引擎是否处于"不可用"状态。两个条件任一成立即算：
        1) engine_core.resources.engine_dead：EngineCore 进程真的没了
           （被 OOM kill、崩溃、主动退出）；
        2) not is_running：后台 output_handler 已经结束
           （正常结束 = 关闭流程中，异常结束 = 崩了）。
        """
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    async def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info

        [CN] 下面这一组方法是 **RL 在线训练**（如 RLHF/GRPO）的核心通道：
             训练进程算完新权重后，要在**不重启引擎**的前提下把权重灌进推理进程。
             标准流程是四步：
               init_weight_transfer_engine（建通道，如 NCCL / 共享内存）
                 -> start_weight_update（引擎进入"可更新"状态，暂停相关缓存）
                 -> update_weights（真正传权重，可分批多次）
                 -> finish_weight_update（提交并打版本号）
             每一步都是 collective_rpc，即对所有 worker 广播执行。
        """
        await self.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": request.init_info}
        )

    async def start_weight_update(self) -> None:
        """Start a new weight update."""
        await self.collective_rpc("start_weight_update")

    async def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        # [CN] 投机解码场景下，**draft model（草稿模型）也是要跟着更新的**，
        #      否则它一直在用旧策略提草稿，接受率会暴跌。
        #      所以权重更新有"主模型"和"草稿模型"两条独立通道。
        await self.collective_rpc("start_draft_weight_update")

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """
        Batched weight update for RL training.

        Args:
            request: Weight update request with backend-specific update info
        """
        await self.collective_rpc(
            "update_weights", kwargs={"update_info": request.update_info}
        )

    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        # [CN] 提交这次更新。weight_version 用于 RL 的**权重版本对齐**：
        #      推理侧和训练侧各自记录版本号，能检测出"某次更新只成功了一半"
        #      （部分 rank 更新了、部分没有）这类非常难查的问题。
        await self.collective_rpc("finish_weight_update")
        if weight_version is not None:
            await self.update_weight_version(weight_version)

    async def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        await self.engine_core.set_weight_version_async(new_version)

    async def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return await self.engine_core.get_weight_version_async()
