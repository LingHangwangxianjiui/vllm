# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================
# [CN] 文件：vllm/entrypoints/llm.py
# 职责：离线推理入口 LLM 类，负责参数装配、请求提交与引擎生命周期控制
# 位置：用户代码 → LLM.generate/chat → OfflineInferenceMixin → LLMEngine → EngineCore
# 核心成员：LLM（唯一对外类）
# 上游：用户脚本 / 离线批处理
# 下游：vllm/entrypoints/offline_utils.py（OfflineInferenceMixin）、
#       vllm/engine/arg_utils.py（EngineArgs）、vllm/v1/engine/llm_engine.py（LLMEngine）
# 关键概念：离线同步 step 循环、runner_type（generate/pooling）、enqueue + wait 分离式提交
# 状态：☑ 通读  ☑ 注释完成  □ 已验证
# ============================================================
#
# 【一句话定位】本文件是「离线批处理」的门面：把用户给的 prompts 变成引擎请求，
# 然后同步地一步步推进引擎直到全部完成。它自己不实现调度、KV cache、模型 forward。
#
# 【与在线服务路径的关系，务必先分清】
# - 离线（本文件）：LLM → LLMEngine（v1/engine/llm_engine.py），
#   用 add_request + step 的同步循环推进，generate() 直接返回 list[RequestOutput]。
# - 在线：AsyncLLM（v1/engine/async_llm.py）实现 EngineClient（vllm/engine/protocol.py），
#   generate() 返回异步生成器，每取一次推进一步。
# 两条路径的引擎对象不同、方法名不同、返回类型也不同，不要互相套用。
#
# 【多继承结构】LLM 同时继承三个 mixin，各自提供一部分能力：
# - BeamSearchOfflineMixin：beam search（在 entrypoints/generate/beam_search/offline.py）
# - PoolingOfflineMixin：embed / classify / score 等池化任务（entrypoints/pooling/offline.py）
# - OfflineInferenceMixin：通用的请求预处理与 step 循环（entrypoints/offline_utils.py），
#   是 generate/chat/encode 等所有入口真正落下去的地方
# 注意：本类重写了 __init__，因此要靠手动调用 PoolingOfflineMixin.__init__(self)
# 来完成 mixin 的初始化——Python 不会自动调用多个父类的 __init__。

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import overload

from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    PoolerConfig,
    ProfilerConfig,
    StructuredOutputsConfig,
    is_init_field,
)
from vllm.config.compilation import CompilationMode
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    load_chat_template,
)
from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.serve.utils.api_utils import log_non_default_args
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.v1.engine import PauseMode
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.sample.logits_processor import LogitsProcessor

from ..renderers import ChatParams
from .offline_utils import _O, _R, OfflineInferenceMixin

if TYPE_CHECKING:
    from vllm.v1.metrics.reader import Metric

logger = init_logger(__name__)


# [CN] 离线推理的主类。构造参数有一百多个，但绝大多数都只是「转发」给 EngineArgs：
# 本类只在 __init__ 里做少量预处理（配置对象化、参数校验），真正的配置合并与默认值
# 填充发生在 vllm/engine/arg_utils.py 的 EngineArgs 与各 Config 类中。
# 因此读参数含义时，重点是 EngineArgs 而不是这里的 docstring。
class LLM(BeamSearchOfflineMixin, PoolingOfflineMixin, OfflineInferenceMixin):
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images
            or videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        allowed_media_domains: If set, only media URLs that belong to this
            domain can be used for multi-modal inputs.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `dtype` attribute of the Transformers model's config. However,
            if the `dtype` in the config is `float32`, we will use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        chat_template: The chat template to apply.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        kv_cache_memory_bytes: Size of KV Cache per GPU in bytes. By default,
            this is set to None and vllm can automatically infer the kv cache
            size based on gpu_memory_utilization. However, users may want to
            manually specify the kv cache memory size. kv_cache_memory_bytes
            allows more fine-grain control of how much memory gets used when
            compared with using gpu_memory_utilization. Note that
            kv_cache_memory_bytes (when not-None) ignores
            gpu_memory_utilization
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        offload_group_size: Prefetch offloading: Group every N layers
            together. Offload last `offload_num_in_group` layers of each group.
            Default is 0 (disabled).
        offload_num_in_group: Prefetch offloading: Number of layers to
            offload per group. Default is 1.
        offload_prefetch_step: Prefetch offloading: Number of layers to
            prefetch ahead. Higher values hide more latency but use more GPU
            memory. Default is 1.
        offload_params: Prefetch offloading: Set of parameter name segments
            to selectively offload. Only parameters whose names contain one of
            these segments will be offloaded (e.g., {"gate_up_proj", "down_proj"}
            for MLP weights, or {"w13_weight", "w2_weight"} for MoE expert
            weights). If None or empty, all parameters are offloaded.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        enable_return_routed_experts: Whether to return routed experts.
        disable_custom_all_reduce: See
            [ParallelConfig][vllm.config.ParallelConfig].
        hf_token: The token to use as HTTP bearer authorization for remote files
            . If `True`, will use the token generated when running
            `hf auth login` (stored in `~/.cache/huggingface/token`).
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor. Overrides for the
            multi-modal processor obtained from `AutoProcessor.from_pretrained`.
            The available overrides depend on the model that is being run.
            For example, for Phi-3-Vision: `{"num_crops": 4}`.
        pooler_config: Initialize non-default pooling config for the pooling model,
            e.g., `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the mode of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        attention_config: Configuration for attention mechanisms. Can be a
            dictionary or an AttentionConfig instance. If a dictionary, it will
            be converted to an AttentionConfig. Allows specifying the attention
            backend and other attention-related settings.
        return_sampling_mask: Return each sampled token's post-processing
            support set. Requires Model Runner V2 and processed log probabilities.
        spec_method: Top-level alias for `speculative_config["method"]`.
        spec_model: Top-level alias for `speculative_config["model"]`.
        spec_tokens: Top-level alias for
            `speculative_config["num_speculative_tokens"]`.
        **kwargs: Arguments for [`EngineArgs`][vllm.EngineArgs].

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the [AsyncLLMEngine][vllm.AsyncLLMEngine] class instead.
    """

    # [CN] 构造参数分两类，理解这个划分能省很多时间：
    # 1. 显式列出的（下面这一长串）：本类要「加工」一下再传给 EngineArgs 的。
    #    加工包括：dict → 配置对象、None → 默认值、以及一些校验。
    # 2. **kwargs：原样转发给 EngineArgs，本类不解释、不校验。
    #    因此查参数含义时，显式参数看这里的 docstring，其余要看
    #    vllm/engine/arg_utils.py 的 EngineArgs。
    #
    # 【两个 docstring 里没写、但很重要的参数】
    # - runner：决定这个模型以什么「运行形态」加载，取值 auto / generate / pooling。
    #   它后面会变成 model_config.runner_type，直接决定 generate() 和 chat() 能不能用。
    #   用 embedding 模型跑 generate() 报 "only supported for generative models"，
    #   根因就在这里，需要显式传 runner="generate"。
    # - convert：Transformers 模型转换为 vLLM 实现的策略（auto / none / ...）。
    #   某些模型架构在 HF 实现与 vLLM 实现之间需要显式指定走哪条路。
    #
    # 【性能相关的几个参数，改动前要知道代价】
    # - gpu_memory_utilization（默认 0.92）：vLLM 能占用的 GPU 显存比例。
    #   调高 → KV cache 更大 → 并发更高；但过高会在权重加载阶段就 OOM。
    #   注意它统计的是「总显存」比例，不是「剩余显存」。
    # - kv_cache_memory_bytes：直接指定每卡 KV cache 字节数，优先级高于
    #   gpu_memory_utilization（非 None 时后者被忽略）。需要精确控制时用这个。
    # - enforce_eager：True 则禁用 CUDA graph，全部 eager 执行。
    #   启动更快、显存更省、便于调试，但 decode 阶段每步都有 kernel 启动开销，
    #   小 batch 下吞吐会明显下降。生产环境默认应为 False。
    # - cpu_offload_gb：把部分权重放到 CPU，用 CPU-GPU 传输换显存。
    #   能跑更大的模型，但每次 forward 都要搬运，延迟显著上升。
    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template: Path | str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.92,
        cpu_offload_gb: float = 0,
        offload_group_size: int = 0,
        offload_num_in_group: int = 1,
        offload_prefetch_step: int = 1,
        offload_params: set[str] | None = None,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        return_sampling_mask: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        structured_outputs_config: dict[str, Any]
        | StructuredOutputsConfig
        | None = None,
        profiler_config: dict[str, Any] | ProfilerConfig | None = None,
        attention_config: dict[str, Any] | AttentionConfig | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        quantization_config: dict[str, Any] | QuantizationConfigArgs | None = None,
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
        spec_method: str | None = None,
        spec_model: str | None = None,
        spec_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """LLM constructor."""

        # [CN] 离线模式默认关闭周期性统计日志。在线服务才需要定时把吞吐/延迟打到日志，
        # 离线批处理打出来只会干扰输出。用户仍可通过 kwargs 显式覆盖。
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        # [CN] worker_cls 允许传「类对象」。多进程模式下它要跨进程传到 worker，
        # 而普通 pickle 序列化不了动态定义的类，所以改用 cloudpickle 先序列化成字节。
        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
            kwargs["kv_transfer_config"], dict
        ):
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        # [CN] hf_overrides 允许传 dict 或 callable；这里把 None 统一成空 dict，
        # 让下游不必反复判空。注意它只影响 HF config 的字段，不改变 vLLM 自己的配置。
        if hf_overrides is None:
            hf_overrides = {}

        # [CN] compilation_config 支持「传整数」的简写形式：整数被当作编译等级。
        # 这是为了兼容 `--compilation-config 3` 这类 CLI/旧代码写法。
        # 走字典或实例时则不特殊处理，交给下面的 _make_config 统一收敛。
        # [CN] 局部小工具：把「None / 字典 / 已构造好的实例」统一收敛成一个配置实例。
        # 这样调用方传这三种形态都能工作，下游拿到的必然是对象，不用到处判空。
        # 字典分支用 is_init_field 过滤键：只保留该类 __init__ 真正接受的字段。
        # 这是防御性设计——CLI 或 YAML 里常混进无关键，不过滤会直接 TypeError。
        def _make_config(value: Any, cls: type[_R]) -> _R:
            """Convert dict/None/instance to a config instance."""
            if value is None:
                return cls()
            if isinstance(value, dict):
                return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})  # type: ignore[arg-type]
            return value

        if isinstance(compilation_config, int):
            compilation_config_instance = CompilationConfig(
                mode=CompilationMode(compilation_config)
            )
        else:
            compilation_config_instance = _make_config(
                compilation_config, CompilationConfig
            )

        structured_outputs_instance = _make_config(
            structured_outputs_config, StructuredOutputsConfig
        )
        profiler_config_instance = _make_config(profiler_config, ProfilerConfig)
        attention_config_instance = _make_config(attention_config, AttentionConfig)

        # [CN] 单进程下 data_parallel_size > 1 会直接禁止。
        # 原因：DP 需要多个引擎实例各自跑在自己的进程/执行器里，而 LLM 只有一个进程，
        # 多个实例会互相等待集合通信，表现为「卡住」而不是报错，因此提前拦掉。
        # 例外：external_launcher 由外部启动器负责拉起多进程；TPU 平台走自己的路径。
        # 正确用法见 examples/features/data_parallel/data_parallel_offline.py。
        # warn about single-process data parallel usage.
        _dp_size = int(kwargs.get("data_parallel_size", 1))
        _distributed_executor_backend = kwargs.get("distributed_executor_backend")
        if (
            _dp_size > 1
            and not _distributed_executor_backend == "external_launcher"
            and not current_platform.is_tpu()
        ):
            raise ValueError(
                f"LLM(data_parallel_size={_dp_size}) is not supported for single-"
                "process usage and may hang. Please use "
                "the explicit multi-process data-parallel example at "
                "'examples/features/data_parallel/data_parallel_offline.py'."
            )

        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            cpu_offload_gb=cpu_offload_gb,
            offload_group_size=offload_group_size,
            offload_num_in_group=offload_num_in_group,
            offload_prefetch_step=offload_prefetch_step,
            offload_params=offload_params or set(),
            enforce_eager=enforce_eager,
            enable_return_routed_experts=enable_return_routed_experts,
            return_sampling_mask=return_sampling_mask,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            structured_outputs_config=structured_outputs_instance,
            profiler_config=profiler_config_instance,
            attention_config=attention_config_instance,
            compilation_config=compilation_config_instance,
            quantization_config=quantization_config,
            logits_processors=logits_processors,
            spec_method=spec_method,
            spec_model=spec_model,
            spec_tokens=spec_tokens,
            **kwargs,
        )

        # [CN] 把非默认值的参数打进日志，便于复现实验与排查「参数没生效」类问题。
        log_non_default_args(engine_args)

        # [CN] 【整个构造过程最关键的一步】真正创建引擎。
        # 这一步会：加载模型权重 → 探测可用显存并确定 KV cache 大小 → 捕获 CUDA graph →
        # 初始化分布式进程组。因此 LLM(...) 通常耗时几十秒到数分钟，慢是正常的。
        # usage_context 用于区分调用来源（离线类 / 服务 / 自定义），影响使用统计上报。
        #
        # 注意类型：这里是 vllm/v1/engine/llm_engine.py 的 LLMEngine，
        # 不是 vllm/engine/llm_engine.py（那只是 7 行的 V0 兼容别名桩）。
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        # [CN] 下面几行把引擎上的常用对象「提升」为 LLM 的属性。
        # 目的是让 generate/chat 等方法可以直接 self.xxx，不必层层 self.llm_engine.xxx，
        # 也让这些关键对象在调试时能直接从 LLM 实例上看到。
        self.model_config = self.llm_engine.model_config
        self.engine_class = type(self.llm_engine)

        # [CN] 请求 ID 的自增计数器。注意它只保证本进程内唯一，
        # 不是全局唯一 ID——跨进程场景由各自进程独立计数。
        self.request_counter = Counter()
        # [CN] 默认采样参数缓存：从模型配置里提取的「与默认值不同的采样参数」，
        # 首次需要时才计算（懒加载），因为要读 model_config 的 generation_config。
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()
        self.supported_tasks = supported_tasks

        # [CN] runner_type 决定这个模型能干什么：generate（生成式）还是 pooling（池化）。
        # 后面 generate/chat 都会先检查它，用错任务类型会在这里被拦住而不是等到模型报错。
        self.runner_type = self.model_config.runner_type
        # [CN] renderer：把文本/聊天消息/多模态输入渲染成引擎能吃的 EngineInput。
        # input_processor：把 EngineInput 进一步处理成 EngineCoreRequest 交给核心调度。
        # 两者都在引擎里创建，这里只是持有引用。
        self.renderer = self.llm_engine.renderer
        self.chat_template = load_chat_template(chat_template)
        self.input_processor = self.llm_engine.input_processor

        # [CN] 预热渲染器：提前编译聊天模板、初始化多模态处理器等。
        # 目的：把这些一次性开销从「第一个请求」挪到构造阶段，避免首个请求延迟异常高，
        # 也避免它污染性能测量。
        self.renderer.warmup(ChatParams(chat_template=self.chat_template))

        # The renderer thread pool is only consumed by the async renderer
        # path; the synchronous `LLM` entrypoint runs multimodal
        # preprocessing serially. Warn so the setting is not a silent
        # no-op. See vllm-project/vllm#42901.
        if self.model_config.renderer_num_workers > 1 and self.runner_type != "pooling":
            logger.warning_once(
                "`renderer_num_workers=%d` was set, but the offline `LLM` "
                "entrypoint uses the synchronous renderer path and runs "
                "multimodal preprocessing serially across prompts. The "
                "renderer thread pool is only consumed by the async "
                "renderer path used by `vllm serve` / `AsyncLLM`, so this "
                "setting has no effect here.",
                self.model_config.renderer_num_workers,
            )

        # [CN] 手动初始化池化 mixin。为什么必须显式写这一行：
        # LLM 自己定义了 __init__，Python 就不会自动调用父类/mixin 的 __init__。
        # PoolingOfflineMixin 需要在实例化时准备自己的状态，漏掉会导致池化方法不可用。
        PoolingOfflineMixin.__init__(self)

        # Cache for __repr__ to avoid repeated collective_rpc calls
        self._cached_repr: str | None = None

    # [CN] 反向构造：已有 EngineArgs 时直接展开成关键字参数传给 __init__。
    # vars(engine_args) 取出 dataclass 的全部字段字典。
    # 注意这条路会绕过 __init__ 里对 kwargs 的预处理之外的所有校验，
    # 且 engine_args 的字段名必须与 __init__ 形参名完全一致才能对上。
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLM":
        """Create an LLM instance from EngineArgs."""
        return cls(**vars(engine_args))

    # [CN] 返回分词器。注意 skip_tokenizer_init=True 时引擎没有分词器，
    # 调用这里会报错——那种模式下必须自己传 prompt_token_ids。
    def get_tokenizer(self) -> TokenizerLike:
        return self.llm_engine.get_tokenizer()

    # [CN] world_size 即「一共用了多少个 GPU 副本」。
    # include_dp=True  → TP * PP * DP（所有副本，含数据并行的重复副本）
    # include_dp=False → TP * PP（只算一个模型实例被切成了几份）
    # 二者在开了 DP 时不同：估算总 GPU 数量用 True，
    # 判断张量并行切分方式（进而判断权重怎么切）用 False。
    def get_world_size(self, include_dp: bool = True) -> int:
        """Get the world size from the parallel config.

        Args:
            include_dp: If True (default), returns the world size including
                data parallelism (TP * PP * DP). If False, returns the world
                size without data parallelism (TP * PP).

        Returns:
            The world size (tensor_parallel_size * pipeline_parallel_size),
            optionally multiplied by data_parallel_size if include_dp is True.
        """
        parallel_config = self.llm_engine.vllm_config.parallel_config
        if include_dp:
            return parallel_config.world_size_across_dp
        return parallel_config.world_size

    # [CN] 清多模态缓存要清两处，因为它们是两个独立的缓存：
    # 1. renderer 侧：多模态输入预处理（如图像解码/变换）的中间结果
    # 2. 引擎侧：多模态编码器算出的特征缓存
    # 只清一个不生效；这也是为什么这里要显式调用两遍。
    def reset_mm_cache(self) -> None:
        self.renderer.clear_mm_cache()
        self.llm_engine.reset_mm_cache()

    # [CN] 取「模型自带的默认采样参数」：读 HF generation_config 中与 vLLM 默认值
    # 不同的部分（get_diff_sampling_param），拼成一个 SamplingParams。
    # 两个细节：
    # 1. 结果缓存在 self.default_sampling_params，避免每次请求都重新解析配置。
    # 2. 只在用户「没有显式传 sampling_params」时才使用它（见 generate/chat），
    #    一旦用户传了参数，模型的默认配置不会与它合并——这是常见的困惑点。
    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()

    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.
                If provided, must be a list of integers matching the length
                of `prompts`, where each priority value corresponds to the prompt
                at the same index.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.
        """
        # 此入口要求生成式 runner；校验发生在请求预处理和入队之前。
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        # 未显式传参时，从模型配置获取默认采样参数；没有差异配置则使用
        # SamplingParams()。显式传入的参数不会在此与模型默认配置合并。
        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        # 下游先将输入统一为请求序列、校验参数数量，再预处理并入队。
        # 随后同步循环调用引擎 step()，直到引擎中所有未完成请求结束；
        # 收集完成结果并按请求 ID 排序，因此返回顺序不取决于完成先后。
        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def enqueue(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Enqueue prompts for generation without waiting for completion.

        This method adds requests to the engine queue but does not start
        processing them. Use wait_for_completion() to process the queued
        requests and get results.

        Args:
            prompts: The prompts to the LLM. See generate() for details.
            sampling_params: The sampling parameters for text generation.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
            use_tqdm: If True, shows a tqdm progress bar while adding requests.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of request IDs for the enqueued requests.
        """
        # [CN] enqueue 与 generate 的关系：把「提交请求」和「推进引擎」拆成两步。
        # generate() = enqueue() + wait_for_completion()。
        # 拆开的价值：可以先把大批请求一次性入队，再统一推进，
        # 避免边入队边执行导致的批次碎片化，也让调用方能控制何时开始计算。
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError("LLM.enqueue() is only supported for generative models.")

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._add_completion_requests(
            prompts=prompts,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    # [CN] 下面两个 @overload 只是给类型检查器看的「签名声明」，函数体是 ...，
    # 运行时不会执行；真正的实现是第三个不带 @overload 的同名函数。
    # 作用：让静态类型检查能区分「不传 output_type」与「传 output_type」两种调用
    # 各自的返回类型，从而 IDE 能推断出正确的元素类型，而不是笼统的 Any。
    @overload
    def wait_for_completion(
        self,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[RequestOutput | PoolingRequestOutput]: ...

    @overload
    def wait_for_completion(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]: ...

    def wait_for_completion(
        self,
        output_type: type[Any] | tuple[type[Any], ...] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[Any]:
        """Wait for all enqueued requests to complete and return results.

        This method processes all requests currently in the engine queue
        and returns their outputs. Use after enqueue() to get results.

        Args:
            output_type: The expected output type(s). If not provided, accepts
                both RequestOutput and PoolingRequestOutput.
            use_tqdm: If True, shows a tqdm progress bar.

        Returns:
            A list of output objects for all completed requests.
        """
        # [CN] 不指定类型时两种输出都接受。注意这里给的是「元组」，
        # 传给 _run_engine 后用于过滤/校验：混入不期望的类型会被拦下而不是静默返回。
        if output_type is None:
            output_type = (RequestOutput, PoolingRequestOutput)

        # [CN] _run_engine（在 OfflineInferenceMixin 里）同步循环调用引擎 step()，
        # 直到队列里没有未完成的请求，然后按请求 ID 排序返回。
        # 排序的意义：返回顺序与提交顺序一致，而不是按完成先后——
        # 调用方可以放心按索引对应自己的输入。
        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """

        # [CN] 直接转发给引擎。三个使用要点：
        # 1. method 可以是字符串（worker 上已有的方法名），也可以是 callable。
        #    传 callable 时会被序列化后发到各 worker 执行，且它要多收一个 self 参数
        #    （即 worker 对象本身），这是容易忘记的地方。
        # 2. 返回值是「每个 worker 的结果组成的列表」，长度等于 world_size，
        #    各副本结果通常相同但不一定——比如按 rank 分片的操作要自己归并。
        # 3. 官方建议只用它传控制消息。传大张量走的是进程间通信，
        #    既慢又会在两端各占一份显存；大数据应另开数据面通道。
        return self.llm_engine.collective_rpc(method, timeout, args, kwargs)

    # [CN] 把函数直接作用在「每个 worker 里的模型对象」上，返回各 worker 的结果列表。
    # 与 collective_rpc 的区别：这里拿到的是 nn.Module 实例本身，可以直接改权重、
    # 读参数、做结构检查；collective_rpc 是调用 worker 上的方法。
    #
    # 警告（docstring 里的那条）的具体含义：返回值要经过进程间序列化传回主进程，
    # 如果返回 GPU 张量，会额外占用显存且传输很慢。要返回就先 .cpu()。
    # 另外 func 必须能被序列化（cloudpickle），闭包捕获了不可序列化对象时会失败。
    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.

        !!! warning
            To reduce the overhead of data transfer, avoid returning large
            arrays or tensors from this method. If you must return them,
            make sure you move them to CPU first to avoid taking up additional
            VRAM!
        """
        return self.llm_engine.apply_model(func)

    def chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][vllm.LLM.generate] method to generate
        the responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A sequence of conversations or a single conversation.

                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
                If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

                - "string" will render the content as a string.
                  Example: `"Who are you?"`
                - "openai" will render the content as a list of dictionaries,
                  similar to OpenAI schema.
                  Example: `[{"type": "text", "text": "Who are you?"}]`

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                `True` if `add_generation_prompt` is also `True`.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        """
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        # [CN] 两个互斥参数的含义（用错会直接报错，不是静默忽略）：
        # - add_generation_prompt=True（默认）：在末尾补上「助手回复开头」的提示，
        #   用于让模型接着生成回复。
        # - continue_final_message=True：不补提示，直接把最后一条消息当作待续写内容，
        #   用于「预填助手回复前缀」的场景。
        # 两者不能同时为 True。
        #
        # [CN] chat 与 generate 的关系：chat 只多了一步「聊天模板渲染」，
        # 渲染在 _run_chat 里由 renderer 完成（含 chat_template、tools、多模态内容格式），
        # 渲染结果再走与 generate 完全相同的入队 + step 循环。
        # 因此采样、调度、输出行为都与 generate 一致；
        # 排查问题时应先区分是「渲染阶段」还是「生成阶段」出的。
        return self._run_chat(
            messages=messages,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def enqueue_chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Enqueue chat conversations for generation without waiting.

        This method renders chat conversations and adds the resulting requests
        to the engine queue. Use wait_for_completion() to get results. To
        guarantee that all requests are queued before scheduling starts, pause
        scheduling with sleep(level=0) before calling this method and resume it
        with wake_up(tags=["scheduling"]) afterward.

        Args:
            messages: A sequence of conversations or a single conversation.
                Each conversation is represented as a list of messages.
            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar while rendering
                conversations.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
            chat_template: The template to use for structuring the chat.
            chat_template_content_format: The format to render message content.
            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one.
            tools: Tools to make available to the model, if any.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of request IDs for the enqueued requests.
        """
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.enqueue_chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        # [CN] 文档里那句「先 sleep(level=0) 暂停调度，入队完再 wake_up(tags=["scheduling"])」
        # 的原因：入队本身是逐个渲染 + 提交的，耗时可能不短；
        # 如果不暂停，先入队的请求会在后面的请求还没入队时就开始执行，
        # 于是拿不到「整批一起调度」的效果，吞吐会明显下降。
        # 需要严格控制批次组成时（例如做可比的性能测量）必须这么做。
        return self._add_chat_requests(
            messages=messages,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    # ============================================================
    # [CN] 以下是运维与调试接口，绝大多数都只是「转发给 self.llm_engine」。
    # 它们不改变请求语义，也不参与生成流程。
    # ============================================================

    def start_profile(self, profile_prefix: str | None = None) -> None:
        """Start profiling with optional custom trace prefix.

        Args:
            profile_prefix: Optional prefix for the trace file names. If provided,
                           trace files will be named as "<prefix>_dp<X>_pp<Y>_tp<Z>".
                           If not provided, default naming will be used.
        """
        self.llm_engine.start_profile(profile_prefix)

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    # [CN] 清空前缀缓存（prefix cache）——即那些被复用、按内容哈希索引的 KV block。
    # 返回 bool 表示「是否真的清空了」，不是「调用是否成功」：
    # - False 的常见原因：还有正在运行的请求占着 block，不能强行释放。
    # - 想强制清空就传 reset_running_requests=True，代价是把这些请求抢占回等待队列，
    #   它们已算好的 KV 作废、需要重算。
    # - reset_connector=True 会连 KV 连接器（PD 分离场景）那侧的缓存一起清。
    #
    # 什么时候需要调：做性能对比实验时，上一轮的 prefix cache 会让第二轮偏快，
    # 需要先清掉才能保证两次测量可比。
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.llm_engine.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level.
                - Level 0: Pause scheduling but continue accepting requests.
                           Requests are queued but not processed.
                - Level 1: Offload model weights to CPU, discard KV cache.
                           The content of kv cache is forgotten. Good for
                           sleeping and waking up the engine to run the same
                           model again. Please make sure there's enough CPU
                           memory to store the model weights.
                - Level 2: Discard all GPU memory (weights + KV cache).
                           Good for sleeping and waking up the engine to run
                           a different model or update the model, where
                           previous model weights are not needed. It reduces
                           CPU memory pressure.
            mode: How to handle any existing requests, can be "abort", "wait",
                or "keep".
        """
        # [CN] 三个 level 的区别只在于「释放到什么程度」，不涉及是否接受请求：
        #   0 —— 只暂停调度，权重和 KV 都还在，唤醒最快，适合「攒一批请求再跑」
        #   1 —— 权重搬到 CPU、丢弃 KV cache，省显存但换回同一模型时要拷回权重
        #   2 —— 权重和 KV 全丢，显存释放最彻底，适合换一个不同的模型
        # 调用方责任：本方法不负责确保「此刻没有请求在跑」，
        # 需要调用方自己保证（通常在 enqueue 之前或 wait_for_completion 之后调用）。
        self.llm_engine.sleep(level=level, mode=mode)

    def wake_up(self, tags: list[str] | None = None):
        """
        Wake up the engine from sleep mode. See the [sleep][vllm.LLM.sleep]
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache", "scheduling")`. If None, all memory
                is reallocated. wake_up should be called with all tags
                (or None) before the engine is used again.
                Use tags=["scheduling"] to resume from level 0 sleep.
        """
        # [CN] tags 控制恢复哪些资源，取值 weights / kv_cache / scheduling。
        # 两个易错点：
        # 1. None 表示「全部恢复」，而空列表 [] 不是 None，语义不同，不要混用。
        # 2. 从 level>=1 唤醒必须先恢复 weights/kv_cache，只传 ["scheduling"] 会起不来；
        #    只有 level=0 的暂停才适合用 ["scheduling"] 单独恢复调度。
        self.llm_engine.wake_up(tags)

    def get_metrics(self) -> list["Metric"]:
        """Return a snapshot of aggregated metrics from Prometheus.

        Returns:
            A `MetricSnapshot` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        """
        # [CN] 返回 Prometheus 指标的当前快照（不是累计值）。
        # 典型用法：在 generate 前后各取一次，做差得到这一批请求的吞吐、延迟等。
        # 注意它是「快照」语义，两次调用之间没有被引擎自动累积，需要调用方自己算差。
        return self.llm_engine.get_metrics()

    # ============================================================
    # [CN] 以下是 RL 在线权重更新接口，按 init → start → update（可多次）→ finish 顺序使用。
    # 实现方式：全部通过 collective_rpc 把方法名广播到各个 worker 上执行，
    # 本文件不做任何权重处理，只是把请求转成 RPC 参数。
    # 注意：这三个步骤是独立调用，没有事务保证，中间失败不会自动回滚。
    # ============================================================

    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        init_info_dict = (
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    def start_weight_update(self) -> None:
        """Start a new weight update."""
        self.llm_engine.collective_rpc("start_weight_update")

    def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        self.llm_engine.collective_rpc("start_draft_weight_update")

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """
        Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info
        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        # [CN] 注意这里发了两次调用：先广播 finish_weight_update 到各 worker，
        # 再单独设置版本号。版本号存在引擎侧，不通过 RPC 下发。
        # 因此「权重已更新」与「版本号已更新」不是原子发生的，中间存在短暂不一致窗口。
        self.llm_engine.collective_rpc("finish_weight_update")
        if weight_version is not None:
            self.llm_engine.set_weight_version(weight_version)

    def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        self.llm_engine.set_weight_version(new_version)

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self.llm_engine.get_weight_version()

    def __repr__(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        # Cache the result to avoid repeated collective_rpc calls
        # [CN] 为什么要缓存：collective_rpc 是一次跨进程广播，成本很高；
        # 而 __repr__ 可能在打印、日志、调试器里被反复触发，不缓存会造成明显的卡顿。
        # 缓存的副作用是模型结构变化后不会刷新——对离线场景可以接受。
        if self._cached_repr is None:
            results = self.llm_engine.collective_rpc("get_model_inspection")
            # In distributed settings, we get results from all workers
            # Just return the first one (they should all be the same)
            if results:
                self._cached_repr = results[0]
            else:
                self._cached_repr = f"LLM(model={self.model_config.model!r})"
        return self._cached_repr
