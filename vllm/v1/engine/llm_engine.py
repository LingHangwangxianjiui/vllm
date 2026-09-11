# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ==============================================================================
# 本文件职责：V1 的**同步**引擎门面 LLMEngine（类名沿用旧名以保持向后兼容）。
#   它自己不跑模型，也不做调度，只做三件事：
#     1) 组装：把 InputProcessor / OutputProcessor / EngineCoreClient 拼成一个"引擎"；
#     2) 转发：把 add_request / step / abort / sleep / profile 等调用转给对应的部件；
#     3) 记账：把每一步的调度统计喂给 StatLoggerManager，产出日志与 Prometheus 指标。
#   所谓"同步"，是指 step() 会**阻塞**直到 EngineCore 吐出一帧输出，
#   由调用方（LLM.generate 的循环 / 离线批处理）自己驱动节奏。
#
# 在系统链路中的位置（前端进程内，同步侧）：
#   LLM(entrypoints/llm.py) ──持有──> 【本文件 LLMEngine】
#        │
#        ├─ InputProcessor     : PromptType/EngineInput -> EngineCoreRequest
#        ├─ EngineCoreClient   : 把请求送进 EngineCore 进程（或进程内直连）
#        └─ OutputProcessor    : EngineCoreOutputs -> RequestOutput（含 detokenize）
#   step() 是整条链路的**心跳**：一次调用 = 引擎前进一步 = 拿到一批增量输出。
#
# 与同目录其他文件的关系：
#   - v1/engine/async_llm.py ：本文件的异步版本（在线服务路径），二者接口高度对称，
#                              区别只在 step 是 await 还是阻塞，以及是否支持流式回调。
#   - v1/engine/core.py      ：EngineCore，真正跑调度+模型的那一侧（可能在另一个进程）。
#   - v1/engine/core_client.py：本文件通过 EngineCoreClient.make_client() 拿到它。
#
# 核心内容速查：
#   - LLMEngine.__init__            : 组装三大件 + DP 组初始化 + 清理 finalizer
#   - from_vllm_config / from_engine_args : 两个构造工厂
#   - add_request                   : 入队（含 n>1 并行采样时的"扇出"成 n 个子请求）
#   - step                          : 四步循环：取输出 -> 处理 -> 回撤 abort -> 记指标
#   - has_unfinished_requests_dp    : DP 下的"全局是否还有活儿"（含 dummy batch 机制）
#   - sleep / wake_up / collective_rpc / add_lora : 透传给 EngineCore 的控制类 API
#
# 阅读提示（几个容易踩的点）：
#   1. "Legacy ... for backwards compatibility"：类名和构造签名是为了兼容 v0 的
#      调用方（如 LLM 类、部分测试），**内部实现已经是 V1**，不要去找 v0 的调度器。
#   2. DP（数据并行）时"我这边没活儿了"不等于"全局没活儿了"：所以
#      has_unfinished_requests 会做一次跨 rank 的 all-reduce；当别人还有活而我没有时，
#      要置 should_execute_dummy_batch=True，下一步跑一个空批次 ——
#      否则**那个 rank 不参与集合通信，其他 rank 会卡死在 NCCL 上**。这是分布式
#      推理里最经典的死锁坑，务必理解。
#   3. multiprocess_mode 的分歧点很多（dp 组谁来建、model_executor 能不能直接摸到、
#      多模态预热放哪），读的时候建议先在脑子里固定一种模式再看另一半分支。
#   4. __del__ 只销毁"自己建的" dp group：external_launcher 模式下 dp group 是
#      外部 launcher 建的，销毁它会影响别的进程，所以这里显式跳过。
# ==============================================================================
import time
import weakref
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    # [CN] 补充说明：名字里的 "Legacy" 指**类名与对外接口**兼容旧的 v0 LLMEngine，
    #      内部实现完全是 V1 架构（没有 v0 的 Scheduler/BlockSpaceManager）。
    #      它代表"前端进程"这一侧：不持有 GPU 上的模型（multiprocess 模式下），
    #      只持有 processor + 一个到 EngineCore 的客户端。
    #      典型持有者：vllm/entrypoints/llm.py 的 LLM 类。

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        multiprocess_mode: bool = False,
    ) -> None:
        # [CN] 构造流程总览（顺序即依赖关系，不能随意调换）：
        #   1. 读配置（model/observability/parallel）
        #   2. 初始化链路追踪（可选）
        #   3. 建立 DP 通信组（必须在 EngineCore 之前）
        #   4. 建 renderer -> input_processor -> output_processor
        #   5. 建 EngineCoreClient（这一步真正在另一个进程里把模型加载起来，最耗时）
        #   6. 建指标日志器
        #   7. 清理多模态 dummy 缓存
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        # [CN] OpenTelemetry 追踪：只在这里初始化一次全局 tracer。
        #      之后每个请求会带着 trace_headers 透传到 EngineCore 再带回来，
        #      从而把"前端排队"和"引擎执行"两段 span 串成一条 trace。
        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_stats = log_stats

        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        # [CN] "外部 launcher 模式的 DP"：DP 由外部（torchrun / ray / k8s）拉起多个
        #      进程，每个进程是一个 DP rank，进程间通信组由 launcher 预先建好。
        #      这种模式下本进程**不自己建** dp group，而是后面直接复用 get_dp_group()。
        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        # [CN] 顺序很关键：单进程（非 multiprocess）模式下，DP 组要在这里先建好，
        #      因为 EngineCore 初始化时会用到它做集合通信（如 DP 间的负载同步）。
        #      而 multiprocess 模式下 EngineCore 在子进程里，那边自己会建（见
        #      EngineCoreProc），这里再建一遍会重复占用端口/资源。
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        # [CN] 见 has_unfinished_requests_dp()：DP 场景下"本 rank 空转但别人还有活"
        #      时置 True，下一步强制跑一个空批次，避免 NCCL 集合通信死锁。
        self.should_execute_dummy_batch = False

        # [CN] renderer 是"prompt -> 模型输入"的统一渲染层：负责 chat 模板、
        #      tokenizer、多模态占位符展开、以及多模态缓存（mm_cache）。
        #      它同时被 input_processor 和 output_processor 使用（后者要用它的
        #      tokenizer 做增量 detokenize），所以在这里建好一处共享。
        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        # [CN] 入方向的转换器：把用户的 prompt（字符串 / tokens / dict / 多模态）
        #      变成可以跨进程发送的 EngineCoreRequest。
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        # [CN] 出方向的转换器：把引擎回传的增量 token ids 累积、detokenize、
        #      处理 stop string、算 logprobs，最终拼成对外 API 的 RequestOutput。
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        # Hand the renderer to the client. In multiprocess mode the client
        # starts the MM warmup only after engine-core fork (the why is in
        # BaseRenderer.start_mm_warmup_in_background); InprocClient takes no
        # renderer, so MM warmup stays inside renderer.warmup() there.
        # [CN] 这一步是**最重的一步**：multiprocess 模式下会 fork/spawn 出
        #      EngineCore 子进程，在子进程里初始化分布式环境、加载模型权重、
        #      profile 显存、建 KV cache、捕获 CUDA graph —— 耗时要几十秒级。
        #      为什么多模态预热（MM warmup）要等 fork 之后再做：
        #      fork 之前做的话，预热处理里开起来的线程/持有的 CUDA context 会被
        #      子进程继承，容易和 fork 冲突（CUDA 不能被 fork 后安全使用）；
        #      进程内模式（InprocClient）没有 fork，直接在 renderer.warmup() 里做。
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            renderer=renderer,
        )

        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # for v0 compatibility
            # [CN] 进程内模式下 EngineCore 就在本进程，所以能直接摸到 model_executor。
            #      暴露这个属性纯粹是为了**兼容 v0 时代的调用方**（有些代码和测试
            #      会直接访问 engine.model_executor）。多进程模式下拿不到，故用
            #      type: ignore 绕开类型检查。
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

            # Capture the model while reachable so the finalizer can drop the
            # bytecode hooks pinning it (frees GPU memory on engine deletion).
            # [CN] 这里解决一个真实的显存泄漏问题：torch.compile 生成的
            #      TorchCompileWithNoGuardsWrapper 会用"字节码钩子"把编译产物钉在
            #      模块上，而编译产物又引用着 GPU 上的东西；光让 engine 对象被 GC，
            #      这条引用链还在，显存不会释放。
            #      做法：现在（模型还 reachable 时）用 **weakref** 记住它，
            #      注册一个 finalizer；等 LLMEngine 被回收时，如果模型还活着就
            #      主动调 cleanup() 拆掉钩子。
            #      为什么必须用 weakref.finalize + weakref.ref(model)：
            #      如果直接把 model 强引用存进 finalizer，那 model 就永远不会被
            #      GC（finalizer 还活着 => 引用还在），清理逻辑永远不会触发。
            model = self._get_driver_model_for_cleanup()
            if model is not None:
                self._finalizer = weakref.finalize(
                    self, LLMEngine._cleanup_instance_caches, weakref.ref(model)
                )

        if self.external_launcher_dp:
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            # [CN] 复用外部 launcher 已经建好的 DP 组（进程级全局单例）。
            #      取 .cpu_group 是因为这里只需要做"是否还有未完成请求"这类
            #      **小数据量的 all-reduce**，用 CPU 通信组即可，不必占用 GPU 流。
            self.dp_group = get_dp_group().cpu_group

        # Don't keep the dummy data in memory
        # [CN] 初始化阶段为了 probe / 显存测量 / 多模态预热，会造一些假数据
        #      （dummy image、dummy audio）。正式服务前必须清掉，否则会：
        #        1) 白占内存；
        #        2) 混入 prefix cache 的哈希，污染真实请求的缓存命中。
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        """[CN] 从已构造好的 VllmConfig 建引擎。

        与 from_engine_args 的区别：本方法**跳过** EngineArgs -> VllmConfig 的转换，
        适用于调用方已经自己组装/改过配置的场景（如 RL 框架、单元测试、
        或者需要在配置里注入非 CLI 可表达的字段）。
        multiprocess_mode 直接由环境变量 VLLM_ENABLE_V1_MULTIPROCESSING 决定。
        """
        return cls(
            vllm_config=vllm_config,
            # [CN] 根据 parallel_config.distributed_executor_backend 选出 Executor 子类
            #      （uni / multiproc / ray / external_launcher ...）。
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        # [CN] EngineArgs -> VllmConfig：这一步会做大量默认值推导与交叉校验
        #      （见 vllm/engine/arg_utils.py），是"用户参数"变成"引擎配置"的地方。
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # [CN] 环境变量可以强制打开多进程模式，优先级高于调用方传入的参数。
        #      这是个"逃生舱"：有些部署问题（如进程内模式下插件冲突）只能靠它绕开。
        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def get_num_unfinished_requests(self) -> int:
        """[CN] 本前端还有多少条没跑完的请求（由 OutputProcessor 记账）。"""
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """[CN] 是否还需要继续调 step()。

        两种情形：
          - 非 DP：看自己 + 看引擎侧是否还有别的 DP engine 在跑（多引擎共享一个
            core client 时）；
          - DP：必须跨 rank 聚合，因为"某个 rank 还有活"就意味着大家还得继续
            一起跑（下一步所有 rank 都必须参与集合通信）。
        """
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        # [CN] 跨所有 DP rank 做一次 all-reduce（用 MAX 实现"逻辑或"：
        #      True=1 > False=0）：只要有一个 rank 还有活，
        #      聚合结果就是 True。这一步本身就是一次集合通信 —— 注意**所有 rank
        #      必须同时执行到这里**，这就是下面 dummy batch 存在的原因。
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        # [CN] 我空了但别人没空：下一步必须也要"跟着跑一圈"，所以置位
        #      should_execute_dummy_batch，让 step() 去执行一个空批次。
        #      如果这里偷懒直接返回 False 去睡大觉，其他 rank 的 NCCL 调用会
        #      一直等不到我 -> 整集群挂死。这是 DP 实现里**最容易踩的坑**。
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        return self._supported_tasks

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""
        # [CN] 两处状态都要清：
        #   1) 前端的 OutputProcessor（里面存着 RequestState、已累积的文本、
        #      detokenizer 增量状态）；
        #   2) EngineCore（那边要释放 KV block、从调度队列摘掉请求）。
        # 注意 output_processor.abort_requests 返回的是"**真正需要 abort 的 id 列表**"
        # ——它可能比入参少（比如请求已经结束了、或者根本不存在），
        # 所以要用返回值去调引擎，避免给不存在的请求发 abort。
        # internal=True 表示 abort 的是"内部请求 id"（n>1 扇出后的子请求 id 等），
        # False 表示用户传进来的外部 id。

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        session_id: str | None = None,
        prompt_text: str | None = None,
    ) -> str:
        """[CN] 把一条请求送进引擎。返回实际使用的 request_id。

        完整流程：
          1) 校验 request_id 类型（必须是 str —— 因为它是跨进程的字典 key，
             int 之类的类型容易在序列化后变成别的类型，导致前后端对不上）；
          2) prompt 分两种：
             - 已经是 EngineCoreRequest（**已废弃**的用法，仅兼容旧调用方）；
             - 原始的 PromptType / EngineInput -> 交给 InputProcessor 转换
               （tokenize、多模态处理、算 mm hash、填 arrival_time 等）；
          3) assign_request_id：真正定下 id（EngineCoreRequest 里的
             external_req_id 就是在这里被记下来的）；
          4) n > 1（一次请求采样 n 个结果）时**扇出**成 n 个子请求：
             每个子请求有独立的 request_id（parent_id + 序号）和独立的
             sampling_params（主要是 seed 不同，保证 n 个结果不一样）；
          5) 先在 OutputProcessor 建状态，再送进 EngineCore。
             **顺序不能反**：否则引擎的输出可能先于前端状态到达而被丢弃。
        参数里的 prompt_text 用于 stop string 检测（要在原文上匹配），
        只在非 EngineCoreRequest 路径下从 prompt 里提取。
        """
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
                "is deprecated and will be removed in the future. You should "
                "instead pass the outputs of Renderer.render_cmpl() or "
                "Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "LLMEngine.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                session_id=session_id,
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        # [CN] 把"内部 id"正式定下来，并把用户给的 id 记到 external_req_id 上。
        #      为什么需要两套 id：n>1 时会派生出 parent_0 / parent_1 ... 这些内部 id，
        #      但对外（API 响应、abort 接口）必须还是用户给的那个 id。
        self.input_processor.assign_request_id(request)

        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        # [CN] 必须用 request 上的 params，而不是入参 params：
        #      InputProcessor 可能**克隆并修改**过它（例如把 max_tokens 按
        #      max_model_len 裁剪、补默认值、处理 seed）。
        #      用入参会拿到过期的值，导致前端与引擎对同一个请求理解不一致。
        params = request.params

        # [CN] n 只有采样任务才有；池化任务恒为 1。
        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return req_id

        # Fan out child requests (for n>1).
        # [CN] 扇出：ParentRequest 负责生成子 id（如 "req-0"/"req-1"）和子参数
        #      （seed 各不相同，否则 n 路采样会得到一模一样的结果），
        #      并把 n 个子请求的输出在 OutputProcessor 里合并回一个父请求的输出。
        parent_req = ParentRequest(request)
        for idx in range(n):
            request_id, child_params = parent_req.get_child_info(idx)
            # [CN] 最后一个子请求**直接复用**原 request 对象（省一次拷贝），
            #      前面的用 copy() —— 注意 copy 是浅拷贝，所以像 prompt_token_ids
            #      这种列表是**共享**的（只读，安全），而 request_id /
            #      sampling_params 这类会被改写的字段是各自独立的。
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # Make a new RequestState and queue.
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

        return req_id

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        """[CN] 驱动引擎前进**一步**，返回本步产出的（增量）请求输出。

        这是整个同步路径的心跳函数。调用方（LLM.generate 的循环）会一直调它，
        直到 has_unfinished_requests() 为 False。

        一步里做的四件事（顺序有讲究）：
          1) get_output()      ：从 EngineCoreClient 取一帧 EngineCoreOutputs
                                 （**阻塞**，直到引擎真的产出了东西）；
          2) process_outputs() ：增量 token -> detokenize -> 拼 RequestOutput，
                                 顺带算出"哪些请求因为命中 stop string 需要被 abort"；
          3) abort_requests()  ：把 2) 算出来的请求回撤掉。
                                 为什么不在 2) 里直接做：OutputProcessor 不持有
                                 引擎连接，它只能"提出要求"，由引擎侧统一执行；
          4) record()          ：把调度统计喂给指标系统（即使本步没有输出也要记，
                                 否则引擎空闲期的 waiting/running 数字会不更新）。

        注意：本步返回的 RequestOutput 是**增量**的（自上次返回以来新增的文本），
        除非该请求刚结束（那时它会带完整的 finish_reason）。
        """
        # [CN] 见 has_unfinished_requests_dp：本 rank 空转但其他 DP rank 还有活，
        #      必须也走一步（跑空批次）参与集合通信，否则其他 rank 会死锁。
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        # [CN] record_function_or_nullcontext：开着 torch profiler 时给这一步打上
        #      label（方便在 profile 里看时间花在哪），没开时是零开销的 nullcontext。
        #      下面几处同理。
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            # [CN] iteration_stats 只在"开了指标 且 本步确实有输出"时才建对象，
            #      避免每步都分配一个空对象（高频路径上的小优化）。
            iteration_stats = (
                IterationStats() if self.log_stats and outputs.outputs else None
            )
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if self.logger_manager is not None and outputs.scheduler_stats is not None:
                # Record even when this step produced no request outputs.
                # [CN] 即使本步一条输出都没有也要记：scheduler_stats 反映的是
                #      "队列里还有多少 waiting/running、KV cache 用了多少"，
                #      这些是**累积状态**，只在有输出时更新会让指标出现"卡住"的假象。
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                if outputs.outputs:
                    self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self, profile_prefix: str | None = None):
        """[CN] 开启 torch profiler。注意它会**严重影响性能**（每步都同步 CUDA），
        只用于排查性能问题。profile 数据在 EngineCore 侧采集落盘。"""
        self.engine_core.profile(True, profile_prefix)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        # Join the background MM warmup first: the mm_processor_cache is not
        # safe for concurrent access with its apply/clear.
        # [CN] 必须先等后台预热线程结束：多模态处理器缓存（mm_processor_cache）
        #      内部就是一个普通 dict，读写没有加锁；预热线程在写、这里在清，
        #      并发访问会破坏它（可能清到一半又被填回去，或者遍历时字典被改动）。
        self.renderer._join_mm_warmup()
        self.renderer.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """[CN] 清空 prefix cache（前缀复用缓存）。

        reset_running_requests：连**正在跑**的请求已缓存的 block 也一并清掉。
            默认 False 只清空闲 block，因为清掉在跑的会导致它们重算（很贵）。
        reset_connector：连 KV connector（P/D 分离、外部 KV 存储）那侧的缓存也清。
        返回 bool 表示是否真的清成功（某些后端不支持运行时清空）。
        """
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.

        [CN] 典型调用时机：**在线更新权重之后**（RLHF / 热更新）。
        视觉 encoder 的输出会被缓存复用（同一张图不必重复编码），
        但权重换了以后旧缓存就是"用老权重算出来的"，必须失效。
        """
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """[CN] 让引擎"睡下"，释放资源给别的进程用（如 RL 训练进程共用 GPU）。

        level=1：把权重 offload 到 CPU、释放 KV cache 显存（保留 CUDA context）；
        level=2：更彻底（连 CUDA context 都释放，唤醒更慢但让出的显存更多）。
        level>=1 时要清多模态缓存：因为 offload 后缓存里的张量指向的显存已经失效。
        mode 决定在途请求怎么处理，见 PauseMode 的注释（"abort"/"wait"/"keep"）。
        """
        if level >= 1:
            self.renderer.clear_mm_cache()
        self.engine_core.sleep(level, mode)

        if self.logger_manager is not None:
            # [CN] 记录睡眠状态到指标里（1=睡, level=睡的等级），
            #      这样监控上能看到"这张卡现在被让出去了"。
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        """[CN] 唤醒。tags 指定只恢复部分组件（如只恢复 "weights" 不恢复 kv_cache），
        用于更细粒度的资源管理。"""
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        """[CN] 拉取当前所有 Prometheus 指标的快照。
        注意前提是 log_stats=True：指标采集本身就是有开销的，关掉时指标对象根本
        不会被更新，返回的值没有意义，所以这里直接断言。"""
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        """[CN] 可能为 None：某些渲染器/池化模型不需要 tokenizer。"""
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        """[CN] 与上面的属性不同，这个方法保证返回一个可用的 tokenizer，
        拿不到就抛异常（适合"必须有 tokenizer 才能干活"的调用方）。"""
        return self.renderer.get_tokenizer()

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        # [CN] 按固定时间间隔节流打日志（VLLM_LOG_STATS_INTERVAL，默认 10 秒）。
        #      为什么需要节流：step() 每秒可能跑几十上百次，每次都打一行日志会
        #      把控制台/日志文件淹没，也会拖慢推理。
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        # [CN] 这几个 LoRA 方法都是**同步 RPC**：请求要一路发到 EngineCore，
        #      再由执行器广播到所有 worker，等它们都做完才返回。
        #      所以它们是"慢调用"（可能几百毫秒），不要在请求热路径上调用。
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        # [CN] "钉住"：LoRA 槽位有限，新 adapter 进来时会按 LRU 淘汰旧的；
        #      pin 住之后这个 adapter 就不会被淘汰（用于常驻的高优 adapter）。
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        """[CN] 在所有 worker 上执行同一个方法，收集各 worker 的返回值（list）。

        method 可以是一个字符串（worker 上的方法名）或一个以 WorkerBase 为参数的
        可调用对象。这是 vLLM 里"绕过引擎做自定义操作"的官方逃生口
        （比如在线改权重、dump 中间张量、查 GPU 状态）。
        注意：返回值是**每个 worker 一份**，TP=8 时长度就是 8。
        """
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def set_weight_version(self, weight_version: str) -> None:
        """[CN] 打一个"权重版本号"，用于 RL 场景下确认各 worker 是否都已更新到
        同一版权重（避免部分 rank 用新权重、部分用旧权重）。"""
        self.engine_core.set_weight_version(weight_version)

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self.engine_core.get_weight_version()

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """[CN] 便捷方法：在每个 worker 的**模型对象**上执行 func(model)。
        典型用途：在线更新权重（func 里做 load_state_dict）、打印参数统计等。"""
        return self.collective_rpc("apply_model", args=(func,))

    def _get_driver_model_for_cleanup(self) -> nn.Module | None:
        """[CN] 顺着 driver_worker -> model_runner -> model 找到模型对象，
        任一层缺失就返回 None（用 getattr 而不是属性访问，是为了兼容不同
        executor/worker 实现里没有这些字段的情况）。"""
        driver_worker = getattr(self.model_executor, "driver_worker", None)
        model_runner = getattr(driver_worker, "model_runner", None)
        return getattr(model_runner, "model", None)

    @staticmethod
    def _cleanup_instance_caches(model_ref: "weakref.ref[nn.Module]") -> None:
        """Remove the bytecode hooks that pin the compiled model."""
        # [CN] 注意两点：
        #   1) 必须是 @staticmethod：finalizer 持有的是 **weakref.ref(model)**，
        #      如果这是实例方法，就会通过 self 把 LLMEngine 强引用回去，
        #      形成"finalizer -> engine -> finalizer"的环，两者都永远不释放。
        #   2) 这里 local import 而不是文件顶部 import：
        #      compilation.wrapper 会拉起 torch.compile 相关的一堆依赖，
        #      在引擎销毁阶段才需要它，放顶部会拖慢正常启动路径。
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        model = model_ref()
        if model is None:
            return
        for module in model.modules():
            if isinstance(module, TorchCompileWithNoGuardsWrapper):
                module.cleanup()

    def __del__(self):
        """[CN] 销毁时释放自己建的 DP 进程组。

        两个防御细节：
          1) 用 getattr 取 dp_group：__init__ 可能在建 dp_group 之前就抛异常了，
             那时属性不存在，__del__ 里直接访问会再抛一个 AttributeError
             （异常发生在 __del__ 里会被 Python 吞掉，但会污染日志）。
          2) external_launcher_dp 时**不销毁**：那个 group 是外部 launcher 建的，
             别的组件/进程还在用，销毁它会导致别人通信失败。
        """
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
