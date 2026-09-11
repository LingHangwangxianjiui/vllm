# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# [CN] 文件总览：**输入预处理器** —— 把用户给的 prompt 变成引擎认识的
#      **EngineCoreRequest**。它运行在 **前端进程**，是请求进入系统的第一道门。
#
#      ========================= 它负责什么 =========================
#      ① 参数校验（SamplingParams / PoolingParams、LoRA、DP rank、长度）；
#      ② prompt 渲染与分词（真正的活儿委托给 Renderer）；
#      ③ 多模态特征规整（把按模态分桶的 placeholders / hashes / kwargs
#         拍平成一个**按位置排序**的 mm_features 列表）；
#      ④ 生成内部 request id（外部 id + 随机后缀，保证唯一）；
#      ⑤ 组装 EngineCoreRequest。
#
#      ========================= 一个关键设计 =========================
#      **原始 prompt 已经不推荐直接传进来了**。新路径是调用方先用
#      Renderer.render_cmpl() / render_chat() 得到 EngineInput，再传进来。
#      这样做的好处：分词与多模态预处理是**阻塞**的，可以在渲染阶段
#      放到线程池里（见 process_inputs_async），不卡 asyncio 事件循环。
#
#      为什么输入处理要单独一个文件：它既要懂模型配置、又要懂多模态、
#      还要懂采样参数的各种互相约束，塞进 LLM 类里会失控。

import time
from collections.abc import Mapping
from typing import Any, Literal

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import (
    EngineInput,
    PromptType,
    SingletonInput,
    split_enc_dec_input,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import argsort_mm_positions
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.sampling_params import SamplingParams
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS, SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid
from vllm.utils.async_utils import make_async
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


# [CN] 输入预处理器。注意它**不是**纯粹的函数集合 —— 
#      它持有 renderer（渲染器）与 mm_registry（多模态注册表）两件重武器。
class InputProcessor:
    # [CN] 构造：把需要的子配置都挂上来，并按是否支持多模态决定
    #      encoder 缓存大小与「是否跳过 prompt 长度检查」。
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.structured_outputs_config = vllm_config.structured_outputs_config
        self.observability_config = vllm_config.observability_config

        # [CN] 从模型的 generation_config 里取默认采样参数（temperature 等）。
        self.generation_config_fields = model_config.try_get_generation_config()

        # [CN] renderer 是真正干渲染 + 分词活儿的对象，允许外部注入以便复用。
        self.renderer = renderer or renderer_from_config(vllm_config)

        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
        self.mm_encoder_cache_size = 0
        self.skip_prompt_length_check = False
        # [CN] 多模态：算出 encoder 缓存容量（后面校验单条 mm 是否放得下要用）。
        if self.supports_mm_inputs:
            mm_budget = MultiModalBudget(vllm_config, mm_registry)
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size
            self.skip_prompt_length_check = (
                mm_budget.processor.info.skip_prompt_length_check
            )
            mm_budget.reset_cache()  # Not used anymore

        # [CN] **异步包装**：分词与多模态处理是阻塞的，
        #      所以给 async 调用方提供一个跑在 renderer 线程池上的版本，
        #      避免把事件循环堵住。
        # Raw-prompt preprocessing (tokenization and multimodal processing)
        # is blocking, so async callers should run it on the renderer's
        # thread pool to keep their event loop responsive.
        self.process_inputs_async = make_async(
            self.process_inputs, executor=self.renderer._executor
        )

    # [CN] tokenizer 可能为 None（例如纯 embedding 或 pooling 场景）。
    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    # [CN] 参数校验总入口：按 SamplingParams / PoolingParams 分流。
    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
    ) -> None:
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid."""
        # [CN] 采样参数：先确认这个模型确实支持生成任务，
        #      再交给 params.verify() 做细粒度检查。
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            if not supported_generation_tasks:
                raise VLLMValidationError("This model does not support generation")

            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )

            # [CN] 采样分布回放需要真实的 logits 分布：
            #      temperature 必须 > 0（贪心没有分布可言），
            #      top_k 必须 > 0（否则掩码会覆盖整个词表，传输量爆炸）。
            if self.model_config.return_sampling_mask:
                if params.temperature <= 0:
                    raise ValueError(
                        "sampling distribution replay requires temperature > 0"
                    )
                if params.top_k <= 0:
                    raise ValueError(
                        "sampling distribution replay requires top_k > 0 to "
                        "bound sampling mask size, reduce transfer overhead, "
                        "and avoid potential OOMs"
                    )
            # [CN] thinking_token_budget 依赖 reasoning 配置，没开就报错。
            if params.thinking_token_budget is not None and (
                self.vllm_config.reasoning_config is None
                or not self.vllm_config.reasoning_config.enabled
            ):
                raise VLLMValidationError(
                    "thinking_token_budget is set but reasoning_config is "
                    "not configured. Please set --reasoning-parser "
                    "and/or --reasoning-config to use thinking_token_budget."
                )
            # [CN] trace 回放必须先在引擎侧用 --enable-trace-replay 打开。
            if (
                params.trace_decode_token_ids
                and not self.model_config.enable_trace_replay
            ):
                raise VLLMValidationError(
                    "trace_decode_token_ids is set but trace replay is not "
                    "enabled. Start the engine with --enable-trace-replay "
                    "to use it."
                )
        # [CN] pooling 参数：确认支持 pooling，
        #      并在 task 未指定时按优先级挑一个默认任务。
        elif isinstance(params, PoolingParams):
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            if not supported_pooling_tasks:
                raise VLLMValidationError("This model does not support pooling")

            if params.task is None:
                if "token_embed" in supported_pooling_tasks:
                    params.task = "token_embed"
                elif "token_classify" in supported_pooling_tasks:
                    params.task = "token_classify"
                elif "plugin" in supported_pooling_tasks:
                    params.task = "plugin"

            if params.task not in supported_pooling_tasks:
                raise VLLMValidationError(
                    f"Unsupported task: {params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

            params.verify(self.model_config)
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

    # [CN] trace 回放模式下，要把「回放语义」强行覆盖到请求参数上：
    #      max_tokens 收到 trace 长度、min_tokens 归零、
    #      **ignore_eos + 清空所有 stop 条件**。
    #      否则 generation_config 里的 EOS 会在 trace 跑完之前就把请求停掉。
    def _normalize_trace_replay_params(
        self, sampling_params: SamplingParams, prompt_len: int
    ) -> None:
        """Apply trace replay's generation semantics to request-local params."""
        trace_token_ids = sampling_params.trace_decode_token_ids
        assert trace_token_ids
        assert sampling_params.max_tokens is not None

        max_trace_len = max(self.model_config.max_model_len - prompt_len, 1)
        trace_token_ids = trace_token_ids[:max_trace_len]
        sampling_params.trace_decode_token_ids = trace_token_ids

        # Apply this after the generation config so its EOS token cannot stop
        # replay before the trace is exhausted.
        sampling_params.max_tokens = min(
            len(trace_token_ids), sampling_params.max_tokens
        )
        sampling_params.min_tokens = 0
        sampling_params.ignore_eos = True
        sampling_params._eos_token_id = None
        sampling_params.stop = []
        sampling_params.stop_token_ids = []
        sampling_params._all_stop_token_ids = set()

    # [CN] LoRA 校验：没开 LoRA 却传了 lora_request → 报错；
    #      另外提醒用户「不同 LoRA 用不同 tokenizer」的支持已废弃。
    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise VLLMValidationError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            logger.warning_once(
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    # [CN] 多模态哈希的**身份标识**。
    #      开启 tower_connector_lora 后，同一个图的 embedding 会随 LoRA 变化，
    #      所以哈希必须带上 LoRA 名字，否则会命中错误的缓存。
    def _get_mm_identifier(
        self,
        mm_hash: str,
        lora_request: LoRARequest | None,
    ) -> str:
        """
        When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore, the mm_hash must be
        generated based on the LoRA request to prevent incorrect cache hits.
        """
        if (
            lora_request is None
            or self.lora_config is None
            or not self.lora_config.enable_tower_connector_lora
        ):
            return mm_hash
        return f"{lora_request.lora_name}:{mm_hash}"

    # [CN] 把「外部已经处理好的」多模态 kwargs 注入 processor 缓存。
    #      场景：前端已经跑过 HF processor，直接把张量传过来。
    #      注入后缓存命中率指标才准，后续相同图片也不用再处理一遍。
    def inject_into_mm_cache(
        self,
        mm_hashes: dict[str, list[str]],
        mm_kwargs: dict[str, list],
    ) -> None:
        """Inject pre-processed mm_kwargs into the processor cache.

        Call this when mm_kwargs have already been through the HF processor
        externally (e.g. by a frontend that transfers pre-processed tensors
        to the backend).  This ensures MM cache hit rate metrics are reported
        accurately and avoids redundant processing on subsequent requests
        with the same images.

        Uses ``get_and_update_item()`` with an empty prompt_updates list,
        since token expansion has already been handled externally.
        """
        cache = self.renderer.mm_processor_cache
        if cache is None:
            return
        try:
            for modality, hashes in mm_hashes.items():
                items = mm_kwargs.get(modality, [])
                for i, mm_hash in enumerate(hashes):
                    if i < len(items) and items[i] is not None:
                        # Insert into cache via get_and_update_item.
                        # Use the returned item (may be an address for SHM
                        # cache or the original item for LRU cache).
                        # [CN] prompt_updates 传空表：token 展开已经在外面做过了。
                        items[i], _ = cache.get_and_update_item(
                            (items[i], []),
                            mm_hash,
                        )
            # Update cache stats to reflect the externally processed items
            self.renderer.update_mm_cache_stats()
        except Exception:
            logger.warning(
                "Failed to inject mm_kwargs into processor cache",
                exc_info=True,
            )

    # [CN] **内部 request id 生成**：保留用户给的 id 到 external_req_id，
    #      request_id 换成加了 8 位随机后缀的新串。
    #      为什么：同一个外部 id 可能被复用（重试 / n>1），
    #      引擎内部必须能唯一定位到每一次请求。
    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    # [CN] ============ 主入口 ============
    #      校验 →（必要时）渲染 → 拆分 encoder/decoder → 校验模型输入
    #      → 整理采样/pooling 参数 → 规整多模态特征 → 组装 EngineCoreRequest。
    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
        session_id: str | None = None,
    ) -> EngineCoreRequest:
        # [CN] ① 参数与 LoRA 校验。
        self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)

        # [CN] ② DP rank 合法性检查。注意用的是 dp_local_size 还是 dp_size，
        #      取决于是否只允许本地引擎。
        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise VLLMValidationError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        # [CN] ③ 新路径：调用方已经渲染好了，直接拿 EngineInput。
        if isinstance(prompt, dict) and "type" in prompt:
            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

            engine_input: EngineInput = prompt  # type: ignore[assignment]
        # [CN] 旧路径（已废弃）：传进来的还是原始 prompt，
        #      这里就地调用 renderer 渲染 + 分词。
        else:
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in the future. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                arrival_time = time.time()

            renderer = self.renderer
            model_config = self.model_config

            parsed_prompt = parse_model_prompt(model_config, prompt)
            tok_params = renderer.default_cmpl_tok_params.with_kwargs(
                **(tokenization_kwargs or {})
            )

            (engine_input,) = renderer.render_cmpl(
                [parsed_prompt],
                tok_params,
            )

        # [CN] ④ 交给平台层做额外校验（不同平台可能有不同限制）。
        current_platform.validate_request(engine_input, params)

        # [CN] ⑤ 拆成 encoder 输入与 decoder 输入（encoder-decoder 模型才有前者），
        #      然后分别校验。
        encoder_input, decoder_input = split_enc_dec_input(engine_input)
        self._validate_model_inputs(encoder_input, decoder_input)

        # [CN] ⑥ 取出 decoder 侧的三件套：embeds / token ids / 是否就是 token ids。
        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_input["type"] == "embeds":
            prompt_embeds = decoder_input["prompt_embeds"]
            prompt_token_ids = decoder_input.get("prompt_token_ids")
            prompt_is_token_ids = decoder_input.get("is_token_ids")
        else:
            prompt_token_ids = decoder_input["prompt_token_ids"]
            prompt_embeds = None
            prompt_is_token_ids = None

        # [CN] ⑦ 参数克隆 + 补全：
        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            # [CN] 为什么要 clone：请求级参数会被后续逻辑修改
            #      （补 max_tokens、generation_config 等），
            #      不能污染用户传进来的那个对象。
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            # [CN] 未指定 max_tokens → 默认「生成到 max_model_len 为止」。
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            # [CN] 用模型 generation_config 与 tokenizer 补全 EOS / stop 等。
            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
            # [CN] trace 回放：覆盖成回放语义（见 _normalize_trace_replay_params）。
            if sampling_params.trace_decode_token_ids:
                self._normalize_trace_replay_params(
                    sampling_params,
                    length_from_prompt_token_ids_or_embeds(
                        prompt_token_ids, prompt_embeds
                    ),
                )
        else:
            pooling_params = params.clone()

        # [CN] ⑧ 多模态：把「按模态分桶」的结构拍平成一个**按位置排序**的列表。
        #      为什么必须排序：后面调度器要按 token 位置顺序消费这些 item。
        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_input["type"] == "multimodal":
            decoder_mm_inputs = decoder_input["mm_kwargs"]
            decoder_mm_positions = decoder_input["mm_placeholders"]
            decoder_mm_hashes = decoder_input["mm_hashes"]

            # [CN] 哈希必须是字符串（否则自定义 processor 实现有问题）。
            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            # [CN] argsort_mm_positions：按每个 item 在序列中的位置排序。
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_input.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
            session_id=session_id,
        )

    # [CN] 长度校验。注意 encoder 侧的上限是 **encoder 缓存大小**，
    #      而不是 max_model_len —— 两者是不同的资源。
    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: Literal["encoder", "decoder"],
    ):
        # [CN] 有些多模态 processor 明确要求跳过 encoder 长度检查。
        if self.skip_prompt_length_check and prompt_type == "encoder":
            return

        # [CN] decoder 侧 prompt 不能为空。
        if prompt_len == 0 and prompt_type == "decoder":
            raise VLLMValidationError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            if self.supports_mm_inputs:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise VLLMValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )
        # [CN] 恰好等于上限也要报错：生成任务至少还要留 1 个位置给输出。
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise VLLMValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    # [CN] 单侧（encoder 或 decoder）输入的校验：
    #      长度 → 单条多模态是否放得下 encoder 缓存 → token id 是否越界。
    def _validate_model_input(
        self,
        prompt_input: SingletonInput,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_ids = (
            None
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        # [CN] 单条多模态 item 的 embedding 数不能超过预分配的 encoder 缓存，
        #      这是一个**启动期就定死**的容量，超了只能改 --limit-mm-per-prompt。
        if prompt_input["type"] == "multimodal":
            decoder_mm_positions = prompt_input["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    num_embeds = mm_position.get_num_embeds()
                    if num_embeds > self.mm_encoder_cache_size:
                        raise VLLMValidationError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        # [CN] 词表越界检查。为什么取 tokenizer 与 model 两者的**最大值**：
        #      Qwen3 的语言模型有 tokenizer 里没有的额外 token；
        #      反过来多模态占位 token 又只在 tokenizer 侧存在。
        #      取最大才不会误杀合法输入。
        if prompt_ids and tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)
            min_input_id = min(prompt_ids, default=0)

            # NOTE: tokenizer.max_token_id is the tokenizer’s vocab size while
            # self.model_config.get_vocab_size() is the model’s vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            model_vocab_size = model_config.get_vocab_size()
            # A negative id is out of vocabulary just like an over-large one,
            # but is not caught by the upper-bound check below. Reject it here
            # so it is not used as an embedding index downstream. This
            # validation path is shared by generate, embedding and pooling
            # requests, so the check covers all three.
            # [CN] 负数 id 也是越界（会被当成 embedding 下标），
            #      但它逃过了下面的「上界」检查，所以单独拦一道。
            if min_input_id < 0:
                raise VLLMValidationError(
                    f"Token id {min_input_id} is out of vocabulary"
                )
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                raise VLLMValidationError(
                    f"Token id {max_input_id} is out of vocabulary"
                )

    # [CN] encoder-decoder 模型两侧都要校验。
    def _validate_model_inputs(
        self,
        encoder_input: SingletonInput | None,
        decoder_input: SingletonInput,
    ):
        if encoder_input is not None:
            self._validate_model_input(encoder_input, prompt_type="encoder")

        self._validate_model_input(decoder_input, prompt_type="decoder")
