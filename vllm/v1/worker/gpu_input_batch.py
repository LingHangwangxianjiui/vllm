# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining a GPU input batch


# [CN] 文件总览：GPU 侧的「持久批次」（persistent batch）状态容器。
# [CN] 核心思想：batch 跨 step 常驻，不再每步重建。
# [CN]   每步只做三件事：新增请求、删除已完成请求、把空洞压实（condense）。
# [CN]   这样绝大多数 step 里「只是改几个数组元素」，而非重新分配张量。
# [CN] 两个核心类：
# [CN]   - CachedRequestState：单个请求的完整状态（由调度器侧传来并被缓存）；
# [CN]   - InputBatch：整个 batch 的列式存储（每个字段一个 (max_num_reqs,) 数组）。
# [CN] 关键性能设计：
# [CN]   1) CPU/GPU 双份数组：_*_cpu 是 numpy 视图（改起来快），GPU 张量按需 copy_slice 同步；
# [CN]   2) 一组 *_reqs 集合（greedy_reqs / top_p_reqs / ...）记录「哪些请求真的需要该参数」，
# [CN]      全batch 都不需要时整块跳过同步（见 _make_sampling_metadata）；
# [CN]   3) batch_update_builder 记录本步的增删移动，供 logits processors 增量更新内部状态。
# [CN] 最容易看错的点：
# [CN]   1) remove_request 之后必须紧跟 condense()，否则数组里会留空洞；
# [CN]   2) swap_states 只交换「活跃 token 前缀」，不是整行 —— 整行拷贝太贵；
# [CN]   3) numpy 的 fancy indexing 交换是「先取值再赋值」，直接 a[[i,j]] = a[[j,i]] 才安全，
# [CN]      而普通切片 a[i1],a[i2] = a[i2],a[i1] 对 ndarray 是错的，所以要先 copy 一份临时值。

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from vllm.config.reasoning import ReasoningConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.collection_utils import swap_dict_values
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import (
    BatchUpdateBuilder,
    LogitsProcessors,
    MoveDirectionality,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.thinking_budget_state import (
    maybe_create_thinking_budget_state_holder,
)
from vllm.v1.utils import copy_slice
from vllm.v1.worker.block_table import MultiGroupBlockTable, SlotMappingMode


# [CN] 单个请求的缓存状态：调度器侧 Request 的「快照」，供 worker 侧重建 batch 用。
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    mrope_positions: torch.Tensor | None = None
    mrope_position_delta: int | None = None

    xdrope_positions: torch.Tensor | None = None

    lora_request: LoRARequest | None = None
    prompt_embeds: torch.Tensor | None = None
    # [CN] prompt logprobs 是分块累加的（chunked prefill 下一次算不完），故需暂存。
    # To accumulate prompt logprobs tensor chunks across prefill steps.
    in_progress_prompt_logprobs_cpu: LogprobsTensors | None = None

    # [CN] 逐位置标记该位置是 token id 还是 embedding（chat completion 混合输入时用）。
    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). See `Request.prompt_is_token_ids`.
    prompt_is_token_ids: list[bool] | None = None

    # Used when both async_scheduling and spec_decode are enabled.
    prev_num_draft_len: int = 0

    # for pooling models
    pooling_params: PoolingParams | None = None
    pooling_states: PoolingStates | None = None

    # [CN] 预计算 prompt 长度，避免每次 num_tokens 都重新算。
    def __post_init__(self):
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )

        if self.pooling_params is not None:
            self.pooling_states = PoolingStates()

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    # [CN] 按全局位置取 token id。越界返回 -1（不是抛异常），
    # [CN] 因为 async scheduling 下会用 -1 作占位符。
    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            if self.prompt_token_ids is None:
                raise ValueError(
                    f"Tried to access token index {idx}, but that token was "
                    "provided via prompt_embeds, and its ID is unknown."
                )
            return self.prompt_token_ids[idx]
        if idx - self.num_prompt_tokens < len(self.output_token_ids):
            return self.output_token_ids[idx - self.num_prompt_tokens]
        return -1


# [CN] 持久 batch：所有状态按「请求下标」列式存放，下标即 req_index。
class InputBatch:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[int],
        max_num_blocks_per_req: list[int],
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        num_spec_tokens: int = 0,
        is_pooling_model: bool = False,
        cp_kv_cache_interleave_size: int = 1,
        reasoning_config: ReasoningConfig | None = None,
        use_replayssm: bool = False,
        slot_mapping_modes: list[SlotMappingMode] | None = None,
    ):
        self.thinking_budget_state_holder = maybe_create_thinking_budget_state_holder(
            reasoning_config,
            max_num_reqs,
            num_spec_tokens,
            device,
            PIN_MEMORY,
        )
        self.thinking_token_budget_reqs: set[str] = set()
        self.is_pooling_model = is_pooling_model
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.vocab_size = vocab_size

        # [CN] _req_ids 允许出现 None（删除后、condense 前的瞬态），req_ids property 会 cast 掉。
        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        # [CN] (max_num_reqs, max_model_len) 的大表。只在 CPU 上，不传 GPU，所以不需要 pinned。
        # [CN] TODO：max_model_len 很大时这张表会非常占内存。
        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        # [CN] is_token_ids：与 token_ids_cpu 同形的布尔表，标记该位置是否为真实 token id。
        self.is_token_ids_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=bool,
            pin_memory=False,
        )
        self.is_token_ids = self.is_token_ids_tensor.numpy()
        # [CN] prompt embeds 按需存（dict 而非大表），避免 max_model_len 大时的一次性巨额分配。
        # Store prompt embeddings per request to avoid OOM from large upfront
        # allocation if max_model_len is big.
        # Maps req_index -> tensor of shape (num_prompt_tokens, hidden_size)
        self.req_prompt_embeds: dict[int, torch.Tensor] = {}
        # [CN] num_tokens_no_spec：不含投机草稿 token 的真实长度。
        # [CN] 有它才能区分「真实 token」与「草稿占位 token」。
        self.num_tokens_no_spec_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        self.num_tokens_no_spec = self.num_tokens_no_spec_cpu_tensor.numpy()
        self.num_prompt_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        self.num_prompt_tokens = self.num_prompt_tokens_cpu_tensor.numpy()
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # [CN] Mamba2 ReplaySSM：记录每个请求最后一次写全量状态时的 num_computed，
        # [CN] 作为 decode 环形缓冲的起点。仅该特性开启时使用。
        # Mamba2 ReplaySSM decode ring origin (num_computed at each request's
        # last full-state write); populated only when the feature is on.
        self.use_replayssm = use_replayssm
        self.replayssm_decode_base_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        self.replayssm_decode_base = self.replayssm_decode_base_cpu_tensor.numpy()

        # [CN] 块表：按 KV cache group 分组，每组一张 (max_num_reqs, max_num_blocks) 表。
        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=PIN_MEMORY,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks=max_num_blocks_per_req,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            slot_mapping_modes=slot_mapping_modes,
        )

        # [CN] 采样参数的「三件套」模式：GPU 张量 + CPU pinned 张量 + numpy 视图。
        # [CN] 日常改动写在 numpy 视图上，需要时再 copy_slice 同步到 GPU。
        # Sampling-related.
        self.temperature = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device=device
        )
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=PIN_MEMORY
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        # [CN] 集合类成员记录「哪些请求真的用了非默认值」，用于整块跳过同步。
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.top_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=PIN_MEMORY
        )
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.top_k_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=PIN_MEMORY
        )
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=PIN_MEMORY
        )
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=PIN_MEMORY
        )
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # Speculative decoding
        self.num_accepted_tokens_cpu_tensor = torch.ones(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()

        # [CN] LoRA：请求 -> lora_id 的映射，以及反向索引（便于统计活跃 LoRA）。
        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # [CN] 只登记「自带 generator」的请求：没有的请求不能进 dict，否则会被误用。
        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}

        # req_id -> list of specific token IDs to compute logprobs for
        # More efficient than num_logprobs=-1 when only a few tokens are needed
        self.logprob_token_ids: dict[str, list[int]] = {}

        # [CN] 本步 batch 变更记录器：供 logits processors 做增量状态迁移，每步重置。
        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # TODO convert this to LogitsProcessor
        # [CN] allowed_token_ids 的掩码语义是反的：True 表示要屏蔽（会被填 -inf）。
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: torch.Tensor | None = None
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs, dtype=bool)

        # [CN] req_output_token_ids 是 list 的 list：外层按 req_index，内层是该请求已生成的 token。
        self.req_output_token_ids: list[list[int] | None] = []

        # Store provided logitsprocs. If none are provided, initialize empty
        # data structure
        self.logitsprocs = logitsprocs or LogitsProcessors()
        self.logitsprocs_need_output_token_ids = logitsprocs_need_output_token_ids

        # [CN] 每个请求的投机草稿 token，供拒绝采样与 penalty 计算使用。
        # Store last speculative tokens for sampler.
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]

        # This is updated each time the batch constituents change.
        # [CN] 首次构造 sampling_metadata；后续只在 batch 变化时重建（见 refresh_metadata）。
        self.sampling_metadata = self._make_sampling_metadata()

        # for pooling models
        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}

        # [CN] async scheduling 专用：保存上一步的采样结果与「拷贝完成」事件，
        # [CN] 以便在本步真正需要 output_token_ids 时再回填。
        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: torch.Tensor | None = None
        self.prev_req_id_to_index: dict[str, int] | None = None
        # These are used to update output_token_ids with real sampled
        # ids from prior step, if required by current sampling params
        # (e.g. penalties).
        self.sampled_token_ids_cpu: torch.Tensor | None = None
        self.async_copy_ready_event: torch.Event | None = None

    # [CN] None 只在状态更新的瞬态出现，正常读取时不应有 None。
    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    # [CN] 为新增请求分配下标，并登记到 batch_update_builder。
    def _register_add_request(self, request: "CachedRequestState") -> int:
        """Track add-request operations for logits processors.
        Not applicable to pooling models.
        """

        # Fill the next empty index if there is one.
        # [CN] 优先复用「本步刚被删除」腾出的空位，没有则追加到末尾。
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # Append to end otherwise.
            new_req_index = self.num_reqs

        assert new_req_index < self.max_num_reqs
        self.batch_update_builder.batch_changed = True
        if request.sampling_params:
            # Detailed added request metadata is only required for non-pooling
            # models, to support logitsprocs.
            self.batch_update_builder.added.append(
                (
                    new_req_index,
                    request.sampling_params,
                    request.prompt_token_ids,
                    request.output_token_ids,
                )
            )

        return new_req_index

    # [CN] 把 CachedRequestState 展开写入各列。这是热路径，所有写入都走预分配数组。
    def add_request(
        self,
        request: "CachedRequestState",
    ) -> int:
        req_index = self._register_add_request(request)

        req_id = request.req_id
        # [CN] 复用空位时直接覆盖，不 append。
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
            self.spec_token_ids.append([])
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids
            self.spec_token_ids[req_index].clear()

        self.req_id_to_index[req_id] = req_index

        # [CN] token_ids_cpu 布局：[prompt tokens][output tokens][（可选）草稿占位]。
        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        if request.prompt_token_ids is not None:
            self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
            if request.prompt_is_token_ids is not None:
                self.is_token_ids[req_index, :num_prompt_tokens] = (
                    request.prompt_is_token_ids
                )
            else:
                self.is_token_ids[req_index, :num_prompt_tokens] = True
        else:
            self.is_token_ids[req_index, :num_prompt_tokens] = False
        # [CN] prompt_embeds 单独存 dict，不占 token_ids_cpu 的行。
        if request.prompt_embeds is not None:
            self.req_prompt_embeds[req_index] = request.prompt_embeds
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        self.is_token_ids[req_index, start_idx:end_idx] = True
        # Number of tokens without spec decode tokens.
        self.num_tokens_no_spec[req_index] = request.num_tokens

        # [CN] 环起点 = 入队时的完整上下文长度，因此被抢占后恢复的请求会重新锚定到 prompt 之后。
        if self.use_replayssm:
            # Ring origin = full context at (re)admission (prompt + any resumed
            # output), so a resumed request re-anchors past the prompt.
            self.replayssm_decode_base[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        # [CN] 块表也要新增一行（各 group 各写一份）。
        self.block_table.add_row(request.block_ids, req_index)

        if sampling_params := request.sampling_params:
            # [CN] greedy 时温度写 0.0 而不是 -inf：后续 apply_temperature 会做除法，0 会导致除零。
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Should avoid division by zero later when apply_temperature.
                self.temperature_cpu[req_index] = 0.0
                self.greedy_reqs.add(req_id)
            else:
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)

            # [CN] top_p 只在 < 1 时登记进集合：全 batch 都是 1.0 时可整块跳过。
            self.top_p_cpu[req_index] = sampling_params.top_p
            if sampling_params.top_p < 1:
                self.top_p_reqs.add(req_id)
            # [CN] top_k 越界（<=0 或 >= vocab_size）时直接取 vocab_size，等价于不生效。
            top_k = sampling_params.top_k
            if 0 < top_k < self.vocab_size:
                self.top_k_reqs.add(req_id)
            else:
                top_k = self.vocab_size
            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
            if sampling_params.frequency_penalty != 0.0:
                self.frequency_penalties_reqs.add(req_id)
            self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
            if sampling_params.presence_penalty != 0.0:
                self.presence_penalties_reqs.add(req_id)
            self.repetition_penalties_cpu[req_index] = (
                sampling_params.repetition_penalty
            )
            if sampling_params.repetition_penalty != 1.0:
                self.repetition_penalties_reqs.add(req_id)

            # [CN] 必须只登记真正有 generator 的请求，否则采样器会误用别人的随机源。
            # NOTE(woosuk): self.generators should not include the requests that
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator

            # [CN] num_logprobs == -1 是「全词表 logprobs」的特殊约定，展开成 vocab_size。
            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = (
                    self.vocab_size
                    if sampling_params.logprobs == -1
                    else sampling_params.logprobs
                )

            # Store specific token IDs to compute logprobs for (more efficient)
            if sampling_params.logprob_token_ids is not None:
                self.logprob_token_ids[req_id] = sampling_params.logprob_token_ids

            # [CN] allowed_token_ids 掩码是 (max_num_reqs, vocab_size) 的大张量，按需惰性分配。
            if sampling_params.allowed_token_ids:
                self.has_allowed_token_ids.add(req_id)
                if self.allowed_token_ids_mask_cpu_tensor is None:
                    # Lazy allocation for this tensor, which can be large.
                    # False means we don't fill with -inf.
                    self.allowed_token_ids_mask = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device="cpu",
                    )
                # [CN] 默认全 True（全屏蔽），再把允许的 token 位置置 False —— 配合 masked_fill_(-inf)。
                self.allowed_token_ids_mask_cpu_tensor[req_index] = True
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask_cpu_tensor[req_index][
                    sampling_params.allowed_token_ids
                ] = False

            if sampling_params.bad_words_token_ids:
                self.bad_words_token_ids[req_index] = (
                    sampling_params.bad_words_token_ids
                )
        # [CN] pooling 模型走这条分支，没有采样参数。
        elif pooling_params := request.pooling_params:
            pooling_states = request.pooling_states
            assert pooling_states is not None

            self.pooling_params[req_id] = pooling_params
            self.pooling_states[req_id] = pooling_states
            self.logits_processing_needs_token_ids[req_index] = False
        else:
            raise NotImplementedError("Unrecognized request type")

        # [CN] 投机解码下每步「至少接受 1 个 token」（即便草稿全被拒，也有 target 模型自己的那个）。
        # Speculative decoding: by default 1 token is generated.
        self.num_accepted_tokens_cpu[req_index] = 1

        # Add request lora ID
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0

        return req_index

    # [CN] 把本步调度器给的草稿 token 写进 token_ids_cpu 与 spec_token_ids。
    def update_req_spec_token_ids(
        self, request: CachedRequestState, scheduled_spec_tokens: dict[str, list[int]]
    ) -> None:
        req_id = request.req_id
        req_index = self.req_id_to_index[req_id]
        cur_spec_token_ids = self.spec_token_ids[req_index]
        # [CN] 结构化输出会丢弃不合规的草稿 token，导致 scheduled_spec_decode_tokens 为空，
        # [CN] 这与「投机解码已开启」并不矛盾，代码必须能容忍。
        # When speculative decoding is used with structured output,
        # the scheduler can drop draft tokens that do not
        # conform to the schema. This can result in
        # scheduler_output.scheduled_spec_decode_tokens being empty,
        # even when speculative decoding is enabled.
        cur_spec_token_ids.clear()
        spec_token_ids = scheduled_spec_tokens.get(req_id, ())
        num_spec_tokens = len(spec_token_ids)
        request.prev_num_draft_len = num_spec_tokens
        if not spec_token_ids:
            return

        # [CN] async scheduling 下这里写的是占位值，真正的草稿 id 会在 _prepare_input_ids 覆盖。
        # For async scheduling, token_ids_cpu assigned from
        # spec_token_ids are placeholders and will be overwritten in
        # _prepare_input_ids.
        start_index = self.num_tokens_no_spec[req_index]
        end_token_index = start_index + num_spec_tokens
        self.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
        self.is_token_ids[req_index, start_index:end_token_index] = True
        cur_spec_token_ids.extend(spec_token_ids)

    # [CN] 删除请求：只做逻辑删除并把下标登记为「空洞」，压实留给 condense()。
    def remove_request(self, req_id: str) -> int | None:
        """This method must always be followed by a call to condense().

        Args:
          req_id: request to remove

        Returns:
          Removed request index, or `None` if `req_id` not recognized
        """

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

        # [CN] 登记到 removed 列表（有序），condense 时按降序处理。
        self.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None
        self.spec_token_ids[req_index].clear()
        self.block_table.clear_row(req_index)

        # [CN] LoRA 反向索引也要清理；某个 lora 不再被任何请求使用时一并删掉。
        # LoRA
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            lora_req_ids = self.lora_id_to_request_ids[lora_id]
            lora_req_ids.discard(req_id)
            if not lora_req_ids:
                del self.lora_id_to_request_ids[lora_id]
                del self.lora_id_to_lora_request[lora_id]
            self.request_lora_mapping[req_index] = 0

        # [CN] pooling 模型没有采样状态，提前返回。
        if self.is_pooling_model:
            self.pooling_params.pop(req_id, None)
            self.pooling_states.pop(req_id, None)
            return req_index

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.logprob_token_ids.pop(req_id, None)
        if self.prev_req_id_to_index is not None:
            self.prev_req_id_to_index.pop(req_id, None)

        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        self.thinking_token_budget_reqs.discard(req_id)
        return req_index

    # [CN] 交换两个请求下标处的全部状态。用于「排序」式重排（不同于 condense 的单向搬移）。
    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        # [CN] 性能关键：只交换活跃前缀，不交换整行（整行是 max_model_len 长度，太贵）。
        # Only swap the active token prefix for each request. Copying full
        # max_model_len rows is expensive and unnecessary during reordering.
        i1_active_token_count = self._get_active_token_count(i1)
        i2_active_token_count = self._get_active_token_count(i2)
        max_active_token_count = max(i1_active_token_count, i2_active_token_count)

        self._req_ids[i1], self._req_ids[i2] = self._req_ids[i2], self._req_ids[i1]  # noqa
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )
        self.spec_token_ids[i1], self.spec_token_ids[i2] = (
            self.spec_token_ids[i2],
            self.spec_token_ids[i1],
        )
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = (
            self.req_id_to_index[old_id_i2],
            self.req_id_to_index[old_id_i1],
        )
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] = (
            self.num_tokens_no_spec[i2],
            self.num_tokens_no_spec[i1],
        )
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] = (
            self.num_prompt_tokens[i2],
            self.num_prompt_tokens[i1],
        )
        if self.use_replayssm:
            self.replayssm_decode_base[i1], self.replayssm_decode_base[i2] = (
                self.replayssm_decode_base[i2],
                self.replayssm_decode_base[i1],
            )
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] = (
            self.num_computed_tokens_cpu[i2],
            self.num_computed_tokens_cpu[i1],
        )

        # [CN] 陷阱：numpy 的 a[i1], a[i2] = a[i2], a[i1] 对二维数组切片是错的
        # [CN] （右侧会先求值成视图，赋值时互相覆盖），必须先 copy 出一份临时值。
        # NOTE: the following is unsafe
        # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
        #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
        # instead, we need to temporarily copy the data for one of the indices
        tmp_token_ids = self.token_ids_cpu[i1, :max_active_token_count].copy()
        self.token_ids_cpu[i1, :max_active_token_count] = self.token_ids_cpu[
            i2, :max_active_token_count
        ]
        self.token_ids_cpu[i2, :max_active_token_count] = tmp_token_ids

        # [CN] 布尔表可以用 fancy indexing 整体交换（先取值再赋值，语义正确）。
        self.is_token_ids[[i1, i2], :max_active_token_count] = self.is_token_ids[
            [i2, i1], :max_active_token_count
        ]

        # Swap prompt embeddings if they exist
        embeds_i1 = self.req_prompt_embeds.get(i1)
        embeds_i2 = self.req_prompt_embeds.get(i2)
        if embeds_i1 is not None:
            self.req_prompt_embeds[i2] = embeds_i1
        else:
            self.req_prompt_embeds.pop(i2, None)
        if embeds_i2 is not None:
            self.req_prompt_embeds[i1] = embeds_i2
        else:
            self.req_prompt_embeds.pop(i1, None)

        # [CN] 块表逐 group 交换行。
        self.block_table.swap_row(i1, i2)

        self.request_lora_mapping[i1], self.request_lora_mapping[i2] = (
            self.request_lora_mapping[i2],
            self.request_lora_mapping[i1],
        )

        # [CN] pooling 模型不需要采样状态，直接返回。
        if self.is_pooling_model:
            # Sampling and logits parameters don't apply to pooling models.
            return

        # For autoregressive models, track detailed request reordering info
        # to support logitsprocs.
        # [CN] 登记 SWAP 类型的移动，让 logits processors 知道要交换内部状态。
        self.batch_update_builder.moved.append((i1, i2, MoveDirectionality.SWAP))

        self.temperature_cpu[i1], self.temperature_cpu[i2] = (
            self.temperature_cpu[i2],
            self.temperature_cpu[i1],
        )
        self.top_p_cpu[i1], self.top_p_cpu[i2] = self.top_p_cpu[i2], self.top_p_cpu[i1]
        self.top_k_cpu[i1], self.top_k_cpu[i2] = self.top_k_cpu[i2], self.top_k_cpu[i1]
        self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2] = (
            self.frequency_penalties_cpu[i2],
            self.frequency_penalties_cpu[i1],
        )
        self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2] = (
            self.presence_penalties_cpu[i2],
            self.presence_penalties_cpu[i1],
        )
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] = (
            self.repetition_penalties_cpu[i2],
            self.repetition_penalties_cpu[i1],
        )
        self.num_accepted_tokens_cpu[i1], self.num_accepted_tokens_cpu[i2] = (
            self.num_accepted_tokens_cpu[i2],
            self.num_accepted_tokens_cpu[i1],
        )

        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            (
                self.allowed_token_ids_mask_cpu_tensor[i1],
                self.allowed_token_ids_mask_cpu_tensor[i2],
            ) = (
                self.allowed_token_ids_mask_cpu_tensor[i2],
                self.allowed_token_ids_mask_cpu_tensor[i1],
            )

    # [CN] 活跃 token 数 = 真实 token 数 + 草稿 token 数。
    def _get_active_token_count(self, req_index: int) -> int:
        return int(self.num_tokens_no_spec[req_index]) + len(
            self.spec_token_ids[req_index]
        )

    # [CN] 压实：把尾部还活着的请求搬到前面的空洞里，使 [0, num_reqs) 连续无洞。
    def condense(self) -> None:
        """Slide non-empty requests down into lower, empty indices.

        Any consecutive empty indices at the very end of the list are not
        filled.

        Returns:
          swaps: list of (from,to) swap tuples for moved requests
          empty_req_indices: indices not filled by condensation
        """
        num_reqs = self.num_reqs

        # [CN] 没有空洞（或空洞已被新增请求填掉）就直接返回，省一次遍历。
        if not (empty_req_indices := self.batch_update_builder.removed):
            # All removed requests were replaced by added requests, or else no
            # requests were removed at all. No condense() needed
            return
        # [CN] 全空时直接清空三个 list。
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            self.spec_token_ids.clear()
            return

        # [CN] 前提：empty_req_indices 按降序排列，算法依赖这个顺序。
        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        # [CN] 双指针：last_req_index 从后往前找非空，empty_index 从前往后取空洞。
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = self.batch_update_builder.peek_removed()
            assert empty_index is not None
            # [CN] 空洞已在尾部之后，说明剩下的都是空洞，无需再搬。
            if empty_index >= last_req_index:
                break

            # Move active request down into empty request
            # index.
            self.batch_update_builder.pop_removed()
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = self._get_active_token_count(last_req_index)

            (self.spec_token_ids[last_req_index], self.spec_token_ids[empty_index]) = (
                self.spec_token_ids[empty_index],
                self.spec_token_ids[last_req_index],
            )
            self.spec_token_ids[last_req_index].clear()

            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            self.is_token_ids[empty_index, :num_tokens] = self.is_token_ids[
                last_req_index, :num_tokens
            ]
            if last_req_index in self.req_prompt_embeds:
                self.req_prompt_embeds[empty_index] = self.req_prompt_embeds.pop(
                    last_req_index
                )
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index
            ]
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            if self.use_replayssm:
                self.replayssm_decode_base[empty_index] = self.replayssm_decode_base[
                    last_req_index
                ]
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]
            self.block_table.move_row(last_req_index, empty_index)

            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index
            ]

            # [CN] pooling 模型不搬采样状态。
            if self.is_pooling_model:
                last_req_index -= 1
                # Sampling state not used by pooling models.
                continue

            # Autoregressive models require detailed tracking of condense
            # operations to support logitsprocs
            # [CN] UNIDIRECTIONAL：只把 last 的状态复制到 empty，不反向（因为是搬移不是交换）。
            self.batch_update_builder.moved.append(
                (last_req_index, empty_index, MoveDirectionality.UNIDIRECTIONAL)
            )

            self.temperature_cpu[empty_index] = self.temperature_cpu[last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[empty_index] = self.frequency_penalties_cpu[
                last_req_index
            ]
            self.presence_penalties_cpu[empty_index] = self.presence_penalties_cpu[
                last_req_index
            ]
            self.repetition_penalties_cpu[empty_index] = self.repetition_penalties_cpu[
                last_req_index
            ]
            self.num_accepted_tokens_cpu[empty_index] = self.num_accepted_tokens_cpu[
                last_req_index
            ]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # TODO convert these to LogitsProcessors
            if self.allowed_token_ids_mask_cpu_tensor is not None:
                self.allowed_token_ids_mask_cpu_tensor[empty_index] = (
                    self.allowed_token_ids_mask_cpu_tensor[last_req_index]
                )

            bad_words_token_ids = self.bad_words_token_ids.pop(last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # [CN] 尾部多余的元素直接删掉，保持 list 长度 == num_reqs。
        # Trim lists to the batch size.
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]
        del self.spec_token_ids[num_reqs:]

    # [CN] 每步调用：把本步的增删移动同步给 logits processors，必要时重建 sampling_metadata。
    def refresh_metadata(self):
        """Apply any batch updates to sampling metadata."""

        if self.is_pooling_model:
            batch_changed = self.batch_update_builder.reset()
            if batch_changed:
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        # [CN] get_and_reset 一次性取出并清空变更记录。
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        if self.thinking_budget_state_holder is not None and batch_update:
            self.thinking_budget_state_holder.sync_batch(batch_update)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()

    # [CN] 构造 SamplingMetadata。大量 `if not no_xxx` 判断都是为了「能跳过就跳过」。
    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        # [CN] 全 greedy 时 temperature 直接给 None，采样器走更快的代码路径。
        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            temperature = None
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        # [CN] penalty 张量的 H2D 同步很贵，只在真有请求需要时才做。
        if not self.no_penalties:
            # Since syncing these tensors is expensive only copy them
            # if necessary i.e. if there are requests which require
            # penalties to be applied during sampling.
            copy_slice(
                self.frequency_penalties_cpu_tensor, self.frequency_penalties, num_reqs
            )
            copy_slice(
                self.presence_penalties_cpu_tensor, self.presence_penalties, num_reqs
            )
            copy_slice(
                self.repetition_penalties_cpu_tensor,
                self.repetition_penalties,
                num_reqs,
            )

        # [CN] prompt_token_ids 只在需要 penalty 或某些 logits processor 时才搬到 GPU。
        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any()
        )
        # The device prompt tokens are used only for applying penalties or
        # pooling methods that explicitly request GPU token IDs.
        # Hence copy these tensors only when there are requests which
        # need penalties/step_pooler to be applied.
        prompt_token_ids_cpu = (
            self._make_prompt_token_ids_cpu_tensor() if needs_prompt_token_ids else None
        )
        prompt_token_ids = (
            prompt_token_ids_cpu.to(device=self.device, non_blocking=True)
            if prompt_token_ids_cpu is not None
            else None
        )

        # [CN] output_token_ids 是 list[list[int]]（变长），同步成本更高，只在必要时传。
        # Only set output_token_ids if required by the current requests'
        # sampling parameters.
        holder = self.thinking_budget_state_holder
        thinking_budget_tracks_reqs = (
            holder is not None and holder.has_tracked_requests()
        )
        needs_output_token_ids = (
            not self.no_penalties
            or bool(self.bad_words_token_ids)
            or self.logitsprocs_need_output_token_ids
            or thinking_budget_tracks_reqs
        )
        output_token_ids = (
            cast(list[list[int]], self.req_output_token_ids)
            if needs_output_token_ids
            else []
        )

        allowed_token_ids_mask: torch.Tensor | None = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            copy_slice(
                self.allowed_token_ids_mask_cpu_tensor,
                self.allowed_token_ids_mask,
                num_reqs,
            )
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        # Build per-request logprob_token_ids mapping: req_index -> token_ids
        logprob_token_ids_by_index: dict[int, list[int]] | None = None
        if self.logprob_token_ids:
            logprob_token_ids_by_index = {}
            for req_id, token_ids in self.logprob_token_ids.items():
                if req_id in self.req_id_to_index:
                    req_index = self.req_id_to_index[req_id]
                    logprob_token_ids_by_index[req_index] = token_ids

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            logprob_token_ids=logprob_token_ids_by_index,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=output_token_ids,
            spec_token_ids=self.spec_token_ids,
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=self.logitsprocs,
            thinking_budget_state_holder=self.thinking_budget_state_holder,
        )

    def get_pooling_params(self) -> list[PoolingParams]:
        assert len(self.req_ids) == len(self.pooling_params)
        return [self.pooling_params[req_id] for req_id in self.req_ids]

    def get_pooling_states(self) -> list[PoolingStates]:
        assert len(self.req_ids) == len(self.pooling_states)
        return [self.pooling_states[req_id] for req_id in self.req_ids]

    def get_pooling_metadata(self) -> PoolingMetadata:
        pooling_params = self.get_pooling_params()
        pooling_states = self.get_pooling_states()
        prompt_token_ids_cpu = None
        if any(p.requires_token_ids for p in pooling_params):
            prompt_token_ids_cpu = self._make_prompt_token_ids_cpu_tensor()

        return PoolingMetadata(
            prompt_lens=self.num_prompt_tokens_cpu_tensor[: self.num_reqs].clone(),
            prompt_token_ids=None,
            prompt_token_ids_cpu=prompt_token_ids_cpu,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )

    # [CN] 构造 (num_reqs, max_prompt_len) 的 prompt token 表（用于 penalty / step pooler）。
    def _make_prompt_token_ids_cpu_tensor(self) -> torch.Tensor:
        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=PIN_MEMORY,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        # [CN] 用 vocab_size 当 pad：它不是合法 token id，不会与真实 id 混淆。
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i] :] = self.vocab_size
        return prompt_token_ids_cpu_tensor

    # [CN] 生成 LoRA 的三种映射：按采样位置、按 token 位置、以及活跃 LoRA 集合。
    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        Given the num_scheduled_tokens for each request in the batch, return
        datastructures used to activate the current LoRAs.
        Returns:
            1. prompt_lora_mapping: A tuple of size np.sum(num_sampled_tokens)
               where, prompt_lora_mapping[i] is the LoRA id to use for the ith
               sampled token.
            2. token_lora_mapping: A tuple of size np.sum(num_scheduled_tokens)
               where, token_lora_mapping[i] is the LoRA id to use for ith token.
            3. lora_requests: Set of relevant LoRA requests.
        """

        # [CN] repeat(num_sampled_tokens)：把「每请求一个 lora_id」展开成「每 token 一个」。
        req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
        token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))

        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values()
        )

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    # [CN] async scheduling：暂存上一步采样结果与完成事件，等真正需要时再取。
    def set_async_sampled_token_ids(
        self,
        sampled_token_ids_cpu: torch.Tensor,
        async_copy_ready_event: torch.Event,
    ) -> None:
        """
        In async scheduling case, store ref to sampled_token_ids_cpu
        tensor and corresponding copy-ready event. Used to repair
        output_token_ids prior to sampling, if needed by logits processors.
        """
        if self.sampling_metadata.output_token_ids:
            self.sampled_token_ids_cpu = sampled_token_ids_cpu
            self.async_copy_ready_event = async_copy_ready_event
        else:
            self.sampled_token_ids_cpu = None
            self.async_copy_ready_event = None

    # [CN] 用真实的（上一步）采样结果替换 output_token_ids 里的 -1 占位符。
    def update_async_output_token_ids(self) -> None:
        """
        In async scheduling case, update output_token_ids in sampling metadata
        from prior steps sampled token ids once they've finished copying to CPU.
        This is called right before they are needed by the logits processors.
        """
        output_token_ids = self.sampling_metadata.output_token_ids
        if self.sampled_token_ids_cpu is None or not output_token_ids:
            # Output token ids not needed or not async scheduling.
            return

        assert self.prev_req_id_to_index is not None
        sampled_token_ids = None
        for index, req_id in enumerate(self.req_ids):
            prev_index = self.prev_req_id_to_index.get(req_id)
            if prev_index is None:
                continue
            req_output_token_ids = output_token_ids[index]
            # [CN] 末尾不是 -1 说明这一轮没有占位符，可能是 kv-load 失败后丢弃了 token。
            if not req_output_token_ids or req_output_token_ids[-1] != -1:
                # Final output id is not a placeholder, some tokens must have
                # been discarded after a kv-load failure.
                continue
            # [CN] 惰性同步：只有真的要用到时才 synchronize()，避免每步都等 D2H 拷贝。
            if sampled_token_ids is None:
                assert self.async_copy_ready_event is not None
                self.async_copy_ready_event.synchronize()
                sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            # Replace placeholder token id(s) with actual sampled id(s).
            new_ids: list[int] = sampled_token_ids[prev_index]
            if not new_ids:
                continue
            # [CN] -1 也可能是有效结束标记（未采样到的位置），取第一个 -1 之前的部分。
            num_sampled_ids = len(new_ids) if new_ids[-1] != -1 else new_ids.index(-1)
            # Also account for case where there may be a smaller number of
            # output placeholders (tokens can be discarded after kv-load
            # failure) or a larger number (async spec decode adds optimistic
            # placeholders that may exceed the actual acceptance count).
            # [CN] 占位符个数与实际采样个数可能不等（丢弃 / 乐观预估），取 min 后截断对齐。
            first_placeholder = len(req_output_token_ids)
            while (
                first_placeholder > 0
                and req_output_token_ids[first_placeholder - 1] == -1
            ):
                first_placeholder -= 1
            num_placeholders = len(req_output_token_ids) - first_placeholder
            num_to_replace = min(num_sampled_ids, num_placeholders)
            del new_ids[num_to_replace:]
            req_output_token_ids[first_placeholder:] = new_ids
            # ^ Implicitly resizes to (first_placeholder + num_to_replace)

    # [CN] 同理，用真实草稿 id 替换 spec_token_ids，供拒绝采样算 penalty / bad_words。
    def update_async_spec_token_ids(self, draft_token_ids: list[list[int]]) -> None:
        """
        In async scheduling case, update spec_token_ids in sampling metadata with
        real draft token ids from prior step. This is called right before they are
        needed by the rejection sampler for penalty/bad_words computation.
        """
        if not draft_token_ids or not self.prev_req_id_to_index:
            return

        if (spec_token_ids := self.sampling_metadata.spec_token_ids) is not None:
            for req_id, spec_ids in zip(self.req_ids, spec_token_ids):
                if spec_ids:
                    prev_index = self.prev_req_id_to_index.get(req_id)
                    if prev_index is not None:
                        draft_ids = draft_token_ids[prev_index]
                        if draft_ids:
                            del draft_ids[len(spec_ids) :]
                            spec_ids.clear()
                            spec_ids.extend(draft_ids)

    # [CN] num_reqs 用 dict 长度而非 list 长度：list 里可能有 None 空洞。
    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    # [CN] 以下 property 都是「能不能整块跳过某个采样步骤」的快速判断。
    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def no_thinking_budget(self) -> bool:
        return (
            self.thinking_budget_state_holder is None
            or len(self.thinking_token_budget_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0
