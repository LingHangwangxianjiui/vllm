# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================
# [CN] 文件：vllm/entrypoints/offline_utils.py
# 职责：离线推理的通用执行层——请求预处理、批量入队、同步 step 循环
# 位置：LLM.generate/chat → 【本文件】→ LLMEngine.add_request / step
# 核心成员：OfflineInferenceMixin（唯一类）
# 上游：vllm/entrypoints/llm.py 的 LLM、BeamSearchOfflineMixin、PoolingOfflineMixin
# 下游：vllm/v1/engine/llm_engine.py 的 LLMEngine
# 关键概念：mixin 模式、渲染即提交（生成器流水线）、FINAL_ONLY 输出、按 request_id 排序
# 状态：☑ 通读  ☑ 注释完成  □ 已验证
# ============================================================
#
# 【本文件在链路中的位置】llm.py 的 LLM 只是「门面」，真正干活的是这里：
#   LLM.generate()
#     → _run_completion()          （本文件）
#     → _add_completion_requests() （本文件：渲染 + 逐个入队）
#     → _render_and_add_requests() （本文件：循环调用 _add_request）
#     → _run_engine()              （本文件：while 循环 step）
#     → LLMEngine.add_request / step
# 也就是说，离线推理的「批处理语义」全部定义在这一个文件里。
#
# 【为什么用 mixin 而不是继承】OfflineInferenceMixin 只提供方法，不持有状态。
# 它需要的 request_counter / renderer / llm_engine / model_config 都在类体内
# 「只声明类型、不赋值」，由混入它的类（LLM）在 __init__ 里填充。
# 好处：PoolingOfflineMixin、BeamSearchOfflineMixin 也能复用同一套入队与 step 逻辑，
# 不需要各自复制一遍。代价是单独看本文件无法确定这些属性从哪来。
#
# 【本文件的三个核心设计，理解了就理解了离线推理】
# 1. 渲染与提交用生成器串起来：prompt 逐个渲染、逐个提交，
#    于是「第一个请求开始执行」不必等到「最后一个渲染完」（见 _render_and_run_requests）。
# 2. 输出只要最终结果：_add_request 里强制 output_kind = FINAL_ONLY，
#    离线不需要在线那种逐步吐字的增量输出。
# 3. 返回前按 request_id 排序：保证输出顺序 == 提交顺序，
#    而不是完成顺序（见 _run_engine 末尾）。

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from tqdm import tqdm
from typing_extensions import TypeVar

from vllm import (
    PoolingParams,
    PoolingRequestOutput,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.renderers import BaseRenderer, ChatParams, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    conversation_to_seq,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.sampling_params import RequestOutputKind
from vllm.utils.counter import Counter
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.tqdm_utils import maybe_tqdm
from vllm.v1.engine.llm_engine import LLMEngine

logger = init_logger(__name__)


# [CN] 三个 TypeVar 的作用（PEP 696 的 default 是较新语法）：
# - _P：参数类型，只能是 SamplingParams 或 PoolingParams 或 None。
#   bound 的意义：generate 用 SamplingParams、embed 用 PoolingParams，
#   用同一个 _P 就能让「传什么类型的参数，就返回什么类型的输出」这件事被类型检查器追踪。
# - _O：输出类型，default 表示调用时不显式指定就用默认联合类型。
# - _R：任意返回类型，供 collective_rpc 等通用方法使用。
_P = TypeVar("_P", bound=SamplingParams | PoolingParams | None)
_O = TypeVar(
    "_O",
    bound=RequestOutput | PoolingRequestOutput,
    default=RequestOutput | PoolingRequestOutput,
)
_R = TypeVar("_R", default=Any)


# [CN] 离线推理的公共实现，以 mixin 形式提供（类名里的 Mixin 就是这个意思）。
# 它不独立实例化，而是被 LLM 等多个类继承（见 llm.py 里 LLM 的基类列表）。
class OfflineInferenceMixin:
    """Offline inference utils"""

    # [CN] 下面四行是「只有类型标注、没有赋值」的类属性声明。
    # 它们不是默认值，运行时访问前必须由混入方的 __init__ 赋值，否则 AttributeError。
    # 这样写的用途有两个：
    # 1. 让类型检查器和 IDE 知道这些属性的类型，在本文件内使用时能补全和检查；
    # 2. 明确声明「本 mixin 依赖宿主提供什么」，相当于一份口头契约。
    # 真正赋值的位置是 vllm/entrypoints/llm.py 的 LLM.__init__。
    request_counter: Counter
    renderer: BaseRenderer
    llm_engine: "LLMEngine"
    model_config: ModelConfig

    # [CN] 解析「按模态自动挂载的 LoRA」。
    # 背景：vLLM 支持为不同模态（image / audio / video）各注册一个默认 LoRA，
    # 配置在 lora_config.default_mm_loras。这样多模态请求不必每次手动传 LoRA。
    #
    # 返回值规则（优先级从高到低）：
    # 1. 显式传入的 lora_request —— 永远优先，自动推断的不会覆盖它；
    # 2. 按 prompt 实际包含的模态推断出的默认 LoRA；
    # 3. 都没有则原样返回 None。
    #
    # 限制：一个请求只能挂一个 LoRA。因此「一个 prompt 同时命中多个模态的 LoRA」
    # 时只能放弃自动挂载并告警（下面的 len(intersection) > 1 分支）。
    def _resolve_mm_lora(
        self,
        prompt: EngineInput,
        lora_request: LoRARequest | None,
    ) -> LoRARequest | None:
        # [CN] 非多模态请求没有「模态」概念，直接短路返回。
        if prompt["type"] != "multimodal":
            return lora_request

        lora_config = self.llm_engine.vllm_config.lora_config
        default_mm_loras = None if lora_config is None else lora_config.default_mm_loras
        if not default_mm_loras:
            return lora_request

        prompt_modalities = prompt["mm_placeholders"].keys()
        intersection = set(prompt_modalities).intersection(default_mm_loras.keys())
        if not intersection:
            return lora_request

        if len(intersection) > 1:
            # TODO: Would be nice to be able to have multiple loras per prompt
            logger.warning(
                "Multiple modality specific loras were registered and would be "
                "used by a single prompt consuming several modalities; "
                "currently we only support one lora per request; as such, "
                "lora(s) registered with modalities: %s will be skipped",
                intersection,
            )
            return lora_request

        # Build the LoRA request; the ID of the default mm lora is the
        # index of the modality name sorted alphabetically + 1.
        # [CN] LoRA 的 int id 由「模态名按字典序排序后的下标 + 1」决定，
        # 而不是随请求分配。这样做是为了让同一个模态在不同请求里得到稳定相同的 id，
        # 引擎侧才能据此复用已加载的适配器。+1 是为了避开 0（0 通常保留给「无 LoRA」）。
        modality_name = intersection.pop()
        modality_lora_path = default_mm_loras[modality_name]
        modality_lora_id = sorted(default_mm_loras).index(modality_name) + 1

        # If we have a collision, warn if there is a collision,
        # but always send the explicitly provided request.
        if lora_request:
            if lora_request.lora_int_id != modality_lora_id:
                logger.warning(
                    "A modality with a registered lora and a lora_request "
                    "with a different ID were provided; falling back to the "
                    "lora_request as we only apply one LoRARequest per prompt"
                )
            return lora_request

        return LoRARequest(
            modality_name,
            modality_lora_id,
            modality_lora_path,
        )

    def _preprocess_cmpl(
        self,
        prompts: Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        """
        Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
        a format that can be passed to `_add_request`.

        Refer to [LLM.generate][] for a complete description of the arguments.

        Returns:
            A list of `EngineInput` objects ready to be passed into LLMEngine.
        """
        # [CN] 预处理分两步，顺序不能反：
        # 1. parse_model_prompt：先把用户给的 PromptType（字符串 / token 列表 /
        #    TextPrompt / TokensPrompt 等多种形态）解析成统一的内部结构，
        #    并处理「纯文本」与「带多模态内容」两种情形的区分。
        # 2. renderer.render_cmpl：再真正做分词与多模态处理，产出 EngineInput。
        # 拆开的意义：解析是纯结构操作、与模型无关；渲染要用到分词器和处理器。
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            parse_model_prompt(model_config, prompt) for prompt in prompts
        ]
        # [CN] with_kwargs 是「在默认参数基础上叠加覆盖项」的写法：
        # 默认分词参数来自 renderer，用户传的 tokenization_kwargs 覆盖同名字段。
        # 注意只影响分词阶段，不影响采样。
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )
        prompt_extras = (
            None
            if mm_processor_kwargs is None
            else {"mm_processor_kwargs": mm_processor_kwargs}
        )

        return renderer.render_cmpl(
            parsed_prompts,
            tok_params,
            prompt_extras=prompt_extras,
        )

    def _preprocess_cmpl_one(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> EngineInput:
        (engine_input,) = self._preprocess_cmpl(
            [prompt],
            tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return engine_input

    def _preprocess_chat(
        self,
        conversations: Sequence[list[ChatCompletionMessageParam]],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        """
        Convert a list of conversations into prompts so that they can then
        be used as input for other LLM APIs.

        Refer to [LLM.chat][] for a complete description of the arguments.

        Returns:
            A list of `EngineInput` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer

        chat_params = ChatParams(
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                chat_template_kwargs,
                dict(
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenize=(
                        is_mistral_tokenizer(renderer.tokenizer)
                        or self.model_config.enable_prompt_embeds
                    ),
                ),
            ),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # The chat template is responsible for emitting BOS/EOS, so do not let
        # the tokenizer add them again unless the caller asks for it. This
        # matches the online chat API (`ChatCompletionRequest.add_special_tokens`
        # defaults to `False`) and avoids a double BOS for multimodal models,
        # whose processor default is `add_special_tokens=True` (#55197).
        # [CN] 这段是离线 chat 最容易踩的坑之一，值得展开：
        # 聊天模板本身已经把 BOS/EOS 写进渲染结果里了，如果再让分词器加一次，
        # 就会出现「双 BOS」——模型输入前面多一个特殊 token，影响输出质量。
        # 所以这里默认 add_special_tokens=False。
        # 那为什么多模态模型尤其危险？因为多模态 processor 的默认值是 True，
        # 不显式关掉就会中招（对应 issue #55197）。
        # 字典展开顺序是 `{默认, **用户传入}`，因此用户显式指定的值可以覆盖这里的默认。
        tokenization_kwargs = {
            "add_special_tokens": False,
            **(tokenization_kwargs or {}),
        }
        tok_params = renderer.default_chat_tok_params.with_kwargs(**tokenization_kwargs)
        prompt_extras = (
            None
            if mm_processor_kwargs is None
            else {"mm_processor_kwargs": mm_processor_kwargs}
        )

        _, engine_inputs = renderer.render_chat(
            conversations,
            chat_params,
            tok_params,
            prompt_extras=prompt_extras,
        )

        return engine_inputs

    def _preprocess_chat_one(
        self,
        conversation: list[ChatCompletionMessageParam],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> EngineInput:
        (engine_input,) = self._preprocess_chat(
            [conversation],
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return engine_input

    # [CN] 三个 _xxx_to_seq 是同一套模式，合起来看：
    # 「单个值 → 复制成 N 份的序列；已经是序列 → 校验长度后原样返回」。
    # 这样下游就能统一按索引 pairs 处理，不必到处判断「是一个还是多个」。
    #
    # 两个要注意的细节：
    # 1. 用 `isinstance(x, Sequence)` 判断「是不是序列」，靠的是 SamplingParams /
    #    LoRARequest 本身不是 Sequence 类型。如果将来这些类实现了 __len__/__getitem__，
    #    这里会误判——属于依赖类型事实的隐式约定。
    # 2. 复制用的是 `[params] * n`，即**同一个对象的 N 个引用**，不是 N 份深拷贝。
    #    因此下游若原地修改其中某个 params（见 _add_request 里的 output_kind 赋值），
    #    所有请求都会受影响。这在离线场景是期望行为（参数本就一致），
    #    但复用同一个 SamplingParams 对象跨多次调用时要小心。
    def _params_to_seq(
        self,
        params: _P | Sequence[_P],
        num_requests: int,
    ) -> Sequence[_P]:
        if isinstance(params, Sequence):
            if len(params) != num_requests:
                raise VLLMValidationError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and params ({len(params)}) must be the same."
                )

            return params

        return [params] * num_requests

    def _lora_request_to_seq(
        self,
        lora_request: LoRARequest | None | Sequence[LoRARequest | None],
        num_requests: int,
    ) -> Sequence[LoRARequest | None]:
        if isinstance(lora_request, Sequence):
            if len(lora_request) != num_requests:
                raise VLLMValidationError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )

            return lora_request

        return [lora_request] * num_requests

    def _priority_to_seq(
        self,
        priority: list[int] | None,
        num_requests: int,
    ) -> Sequence[int]:
        if priority is not None:
            if len(priority) != num_requests:
                raise VLLMValidationError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )

            return priority

        return [0] * num_requests

    def _add_completion_requests(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        # 统一为请求序列：单个字符串、字典或非空 token ID 列表视为一个 prompt。
        seq_prompts = prompt_to_seq(prompts)
        # 单个 params/LoRA 按请求数重复引用，序列则校验长度；此处不复制对象。
        # 未指定 priority 时为每个请求填 0，保证后续可按同一索引配对。
        seq_params = self._params_to_seq(params, len(seq_prompts))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_prompts))
        seq_priority = self._priority_to_seq(priority, len(seq_prompts))

        # 生成器由下游逐项消费：每个 prompt 转为 EngineInput 后立即提交，
        # 无需先完成整批预处理；这里的进度条表示输入处理进度。
        # 返回请求 ID 列表，不在此调用引擎 step() 等待生成结果。
        # 若中途失败，下游会中止本次调用中已成功添加并记录 ID 的请求。
        return self._render_and_add_requests(
            prompts=(
                self._preprocess_cmpl_one(
                    prompt,
                    tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for prompt in maybe_tqdm(
                    seq_prompts,
                    use_tqdm=use_tqdm,
                    desc="Rendering prompts",
                )
            ),
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )

    def _run_completion(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        self._add_completion_requests(
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)

    def _run_chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
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
    ):
        self._add_chat_requests(
            messages=messages,
            params=params,
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
        return self._run_engine(output_type=output_type, use_tqdm=use_tqdm)

    def _add_chat_requests(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        *,
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
        seq_convs = conversation_to_seq(messages)
        seq_params = self._params_to_seq(params, len(seq_convs))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_convs))
        seq_priority = self._priority_to_seq(priority, len(seq_convs))

        # When thinking is enabled or tools are provided, and the model
        # uses special tokens for structured output (e.g. Gemma4's
        # <|channel>, <|tool_call>, <|"|>), automatically set
        # skip_special_tokens=False so these tokens are preserved in
        # output.text for downstream parsing.
        # [CN] 为什么开了 thinking 或传了 tools 就要特殊处理：
        # 部分模型（这里是 Gemma4）把「思考段落分隔符」「工具调用标记」注册成了
        # 特殊 token。而分词时默认 skip_special_tokens=True，会把它们从 output.text
        # 里剥掉——结果就是下游拿不到分隔符，没法切分思考内容和工具调用，解析直接坏掉。
        # 所以要在提交前把 skip_special_tokens 改回 False（见 _adjust_params_for_parsing）。
        needs_parsing = (
            chat_template_kwargs and chat_template_kwargs.get("enable_thinking")
        ) or tools
        if needs_parsing:
            self._adjust_params_for_parsing(seq_params)

        return self._render_and_add_requests(
            prompts=(
                self._preprocess_chat_one(
                    conversation,
                    chat_template=chat_template,
                    chat_template_content_format=chat_template_content_format,
                    chat_template_kwargs=chat_template_kwargs,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenization_kwargs=tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for conversation in maybe_tqdm(
                    seq_convs,
                    use_tqdm=use_tqdm,
                    desc="Rendering conversations",
                )
            ),
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )

    def _adjust_params_for_parsing(
        self, params: Sequence[SamplingParams | PoolingParams]
    ) -> None:
        """Set ``skip_special_tokens=False`` when the model encodes
        structured output syntax as special tokens.

        Models like Gemma4 register thinking delimiters
        (``<|channel>``/``<channel|>``) and tool call tokens
        (``<|tool_call>``/``<tool_call|>``/``<|"|>``) as special tokens.
        The default ``skip_special_tokens=True`` strips them from
        ``output.text``, breaking parsing of both reasoning blocks and
        tool calls.

        This is a no-op for models whose structured tokens are regular
        text tokens (e.g. DeepSeek's ``<think>``/``</think>``).
        """
        # The offline API currently lacks a unified rendering pipeline.
        # Until the planned Renderer refactor is complete, we hardcode
        # this token preservation logic specifically for Gemma4 models
        # to avoid regressions on other models.
        hf_config = getattr(self.model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", [])

        if any("Gemma4" in arch for arch in architectures):
            tokenizer = self.renderer.get_tokenizer()
            vocab = tokenizer.get_vocab()
            special_ids = set(getattr(tokenizer, "all_special_ids", []))

            # Tokens used for thinking delimiters and tool call syntax
            # that some models (Gemma4) register as special tokens.
            structured_tokens = (
                "<|channel>",
                "<channel|>",  # thinking delimiters
                "<|tool_call>",
                "<tool_call|>",  # tool call delimiters
                '<|"|>',  # string quoting in tool args
            )
            needs_special = any(
                vocab.get(tok) in special_ids
                for tok in structured_tokens
                if tok in vocab
            )
            if needs_special:
                for sp in params:
                    if isinstance(sp, SamplingParams) and sp.skip_special_tokens:
                        sp.skip_special_tokens = False

    def _render_and_run_requests(
        self,
        prompts: Iterable[EngineInput],
        params: Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,
        priorities: Sequence[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ):
        # [CN] 为什么传 list/tuple 要告警：这是本文件最重要的性能提示。
        # prompts 是「已经全部渲染好的列表」时，必须先渲染完所有请求才能开始提交，
        # 也就是引擎在整个渲染期间完全空闲。
        # 反之如果传生成器（_add_completion_requests 里就是这么做的），
        # 每渲染好一个就立刻提交，引擎可以在渲染第二个的时候就开始执行第一个——
        # 渲染与执行形成流水线。大批量、多模态（渲染很慢）时差距非常明显。
        if isinstance(prompts, (list, tuple)):
            logger.warning_once(
                "Rendering all prompts before adding them to the engine "
                "is less efficient than performing both on the same prompt "
                "before processing the next prompt. You should instead pass "
                "a generator that renders one prompt per iteration, as that allows "
                "engine execution to begin for the first prompt while processing "
                "the next prompt."
            )

        self._render_and_add_requests(
            prompts=prompts,
            params=params,
            lora_requests=lora_requests,
            priorities=priorities,
        )

        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def _render_and_add_requests(
        self,
        prompts: Iterable[EngineInput],
        params: Sequence[SamplingParams | PoolingParams],
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,
        priorities: Sequence[int] | None = None,
    ) -> list[str]:
        # [CN] 边遍历边提交：prompts 通常是生成器，因此这里是「渲染一个、提交一个」。
        # 注意 enumerate 的下标 i 同时用于取 params / lora / priority，
        # 三者必须与 prompts 等长——长度校验在各自的 _xxx_to_seq 里做过。
        added_request_ids: list[str] = []

        try:
            for i, prompt in enumerate(prompts):
                request_id = self._add_request(
                    prompt,
                    params[i],
                    lora_request=self._resolve_mm_lora(
                        prompt,
                        None if lora_requests is None else lora_requests[i],
                    ),
                    priority=0 if priorities is None else priorities[i],
                )
                added_request_ids.append(request_id)
        except Exception as e:
            # [CN] 失败回滚：中途抛异常时，本次已经成功入队的请求如果不撤销，
            # 会一直留在引擎里占着 KV cache，而调用方永远拿不到结果也不会再来取。
            # 所以这里显式 abort 掉本次调用已添加的全部请求，再向上抛原异常。
            # 用 internal=True 表示这是内部发起的中止，区别于用户主动取消。
            if added_request_ids:
                self.llm_engine.abort_request(added_request_ids, internal=True)
            raise e

        return added_request_ids

    def _add_request(
        self,
        prompt: EngineInput,
        params: SamplingParams | PoolingParams,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
    ) -> str:
        # [CN] 强制只输出最终结果。这是离线与在线最核心的一个差异：
        # 在线要逐 token 吐给客户端，输出 kind 是 DELTA（增量）；
        # 离线一次拿到完整结果即可，用 FINAL_ONLY 省掉中间增量输出的构造与传输开销。
        #
        # 注意这里是**原地修改**传入的 params 对象（不是拷贝）。
        # 结合 _params_to_seq 里 `[params] * n` 的共享引用，
        # 意味着调用方传入的同一个 SamplingParams 实例会被改写 output_kind。
        # 复用参数对象时（比如同一个 params 先传给 generate 再看它的字段）要留意。
        #
        # 只对 SamplingParams 生效：PoolingParams 没有采样过程，也就没有输出粒度概念。
        if isinstance(params, SamplingParams):
            # We only care about the final output
            params.output_kind = RequestOutputKind.FINAL_ONLY

        # [CN] 请求 ID 就是自增计数器的字符串形式。
        # 它只在本进程内唯一——跨进程/多实例场景各自独立计数，不保证全局唯一。
        # 末尾 _run_engine 依赖它是整数可解析的（int(x.request_id)）来排序。
        request_id = str(next(self.request_counter))

        return self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            priority=priority,
        )

    def _run_engine(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # Run the engine.
        # [CN] 【离线推理的主循环】不断调用引擎 step()，直到没有未完成请求。
        # 每一次 step() 就是引擎推进「一个批次的一步」：调度 → 模型 forward → 采样 → 更新状态。
        # 因此这个 while 循环的次数 ≈ 总 decode 步数，而不是请求数。
        #
        # 注意它是**同步阻塞**的：整个 generate() 期间主线程都在这里转，
        # 这也是离线 API 不能用于在线服务的根本原因。
        outputs: list[_O] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                # [CN] output_type 就是调用方声明的期望类型（RequestOutput 等）。
                # 这条 assert 兼作过滤：混入预期之外的输出类型时直接暴露，而不是静默返回。
                assert isinstance(output, output_type)
                # [CN] 只收集已完成的输出。因为 output_kind=FINAL_ONLY，
                # 每个请求在其最终结果产生时才会出现在这里一次。
                if output.finished:
                    outputs.append(output)  # type: ignore[arg-type]
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs
                            )
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        # [CN] 按 request_id 的数值升序排序后返回。
        # 为什么必须排：请求是并发执行的，短请求会先完成，
        # 直接按完成顺序收集会导致输出顺序与输入顺序不对应。
        # 排序后「第 i 个输出」严格对应「用户传入的第 i 个 prompt」，
        # 调用方可以放心用下标对齐，这也是 LLM.generate 文档承诺的行为。
        return sorted(outputs, key=lambda x: int(x.request_id))
