# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：V2 runner 的 attention 初始化与 metadata 构造。
# [CN] 把 gpumodel_runner.py 里散落的 initialize_attn_backend /
# [CN] _build_attention_metadata 抽成纯函数，便于复用与测试。
# [CN] 三条主线：
# [CN]   1) 发现 attention group（按 后端 + spec + Q 头数 聚类）；
# [CN]   2) 决定 kernel 块大小与 CUDA graph 支持度；
# [CN]   3) 为每层构造 metadata（同一 group 内共享一份）。
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.ubatch_utils import get_num_ubatches
from vllm.v1.worker.utils import (
    AttentionGroup,
    add_kv_sharing_layers_to_kv_cache_groups,
    allocate_kv_cache,
    bind_kv_cache,
    prepare_kernel_block_sizes,
)


@dataclass(frozen=True)
# [CN] 「本模型对 CUDA graph 的支持度」= 所有 backend 里最悲观的那个。
# [CN] narrow() 让「不经过 init_attn_backend 建立的 group」（如 encoder-only 层）
# [CN] 也能参与这个决策，避免漏掉某个不支持图的后端。
class AttentionCGSupportInfo:
    min_cg_support: AttentionCGSupport = AttentionCGSupport.ALWAYS
    min_cg_attn_backend: str | None = None

    def narrow(
        self, support: AttentionCGSupport, backend: str | None
    ) -> "AttentionCGSupportInfo":
        """Return an info tightened by ``support`` if it is more restrictive.

        Lets attention groups built outside ``init_attn_backend`` (e.g.
        encoder-only layers) contribute to the runner's cudagraph decision.
        """
        if support.value < self.min_cg_support.value:
            return AttentionCGSupportInfo(support, backend)
        return self


# [CN] 收集各层 KV cache 规格。声明了共享目标的层直接跳过 ——
# [CN] 对 KV cache 管理来说它「不存在」，从而省下显存。
def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            # This layer will use KV cache of the sharing target layer.
            continue
        # Skip modules that don't need KV cache (eg encoder-only attention)
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            if isinstance(spec, AttentionSpec):
                spec = attn_module.get_attn_backend().customize_spec(spec)
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec


def get_shared_kv_cache_layers(vllm_config: VllmConfig):
    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
    return {
        layer_name: kv_tgt_layer
        for layer_name, attn_module in attn_layers.items()
        if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name)
    }


# [CN] 三阶段：
# [CN]   1) 发现各 KV cache group 内的 attention group；
# [CN]   2) 为每个 group 挑一个「组内所有后端都支持」的 kernel 块大小；
# [CN]   3) 建 metadata builder 并汇总 CUDA graph 支持度。
def init_attn_backend(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    active_layer_names: set[str] | None = None,
) -> tuple[list[list[AttentionGroup]], AttentionCGSupportInfo, list[int]]:
    # Phase 1: discover attention groups for each kv cache group.
    attn_groups: list[list[AttentionGroup]] = []

    # Add KV-sharing layers to their target's kv cache group so they are
    # discovered alongside the target layer in Phase 1 below.
    add_kv_sharing_layers_to_kv_cache_groups(
        get_shared_kv_cache_layers(vllm_config), kv_cache_config.kv_cache_groups
    )

    # Phase 1: discover attention groups for each kv cache group.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
        kv_cache_config.kv_cache_groups
    ):
        layer_names = kv_cache_group_spec.layer_names
        if active_layer_names is not None:
            layer_names = list(active_layer_names.intersection(layer_names))

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)

        group_map: dict[tuple[tuple[str, str], KVCacheSpec, int], AttentionGroup] = {}
        group_order: list[tuple[tuple[str, str], KVCacheSpec, int]] = []

        for layer_name in layer_names:
            attn_backend = attn_layers[layer_name].get_attn_backend()

            layer_kv_cache_spec: KVCacheSpec = kv_cache_group_spec.kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]

            # [CN] 按「每 rank 的 Q 头数」再细分 group：
            # [CN] 草稿层的头数可能与目标层不同，而 builder 的 scratch 是按头数定尺寸的，
            # [CN] 混在一组会尺寸错配。
            # Split on per-rank num_heads_q so layers with different Q-head
            # counts (e.g. a spec-decode draft head and its target) get separate
            # metadata builders.
            num_heads_q = getattr(attn_layers[layer_name], "num_heads", 0)
            key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)
            if key not in group_map:
                group_map[key] = AttentionGroup(
                    attn_backend, [layer_name], layer_kv_cache_spec, kv_cache_group_id
                )
                group_order.append(key)
            else:
                group_map[key].layer_names.append(layer_name)

        attn_groups.append([group_map[key] for key in group_order])

    # Phase 2: pick a kernel block size per kv cache group that is supported
    # by all backends within that group.
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)

    # Phase 3: create metadata builders and determine cudagraph support.
    attn_backend_workspace: torch.Tensor | None = None
    for kv_cache_group_id, groups in enumerate(attn_groups):
        kernel_block_size = None
        if kv_cache_group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[kv_cache_group_id]
        for group in groups:
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=kernel_block_size,
                # Microbatches build attention metadata concurrently, and some
                # builders keep the prepared metadata on themselves (MLA stores
                # it on the prefill backend), so each ubatch needs its own.
                # [CN] 每个微批次一个 builder：微批次并发构造 metadata，
                # [CN] 而某些 builder（如 MLA）会把 prepared metadata 存在自己身上，必须隔离。
                num_metadata_builders=get_num_ubatches(vllm_config.parallel_config),
            )
            # [CN] 但 workspace 可以共享：所有微批次的 attention 都提交到同一个计算流，
            # [CN] 是串行写的，与跨 step 复用的情形一样。
            # The microbatches' builders share the workspace: they all issue
            # attention on the one compute stream the threads hand off, so the
            # buffer is written serially, as it already is across steps.
            for builder in group.metadata_builders:
                if attn_backend_workspace is None:
                    if hasattr(builder, "_get_workspace_buffer"):
                        attn_backend_workspace = builder._get_workspace_buffer()
                elif hasattr(builder, "set_workspace_buffer"):
                    builder.set_workspace_buffer(attn_backend_workspace)
    attn_cg_support_info = get_attn_cg_support(attn_groups, vllm_config)
    return attn_groups, attn_cg_support_info, kernel_block_sizes


# [CN] 取所有 backend 中「最不支持 CUDA graph」的那个作为整体结论：
# [CN] 只要有一层不能入图，整条前向就不能整图捕获。
def get_attn_cg_support(
    attn_groups: list[list[AttentionGroup]],
    vllm_config: VllmConfig,
    checked_layer_names: set[str] | None = None,
) -> AttentionCGSupportInfo:
    """Return the weakest CUDA graph support among the checked layers."""
    min_cg_support = AttentionCGSupport.ALWAYS
    min_cg_attn_backend = None
    for groups in attn_groups:
        for group in groups:
            if checked_layer_names is not None and checked_layer_names.isdisjoint(
                group.layer_names
            ):
                continue
            builder = group.get_metadata_builder(0)
            cg_support = builder.get_cudagraph_support(
                vllm_config,
                group.kv_cache_spec,
            )
            if cg_support.value < min_cg_support.value:
                min_cg_support = cg_support
                min_cg_attn_backend = group.backend.__name__
    return AttentionCGSupportInfo(
        min_cg_support=min_cg_support,
        min_cg_attn_backend=min_cg_attn_backend,
    )


# [CN] 哪些后端要求 CPU 侧 query_len 必须与设备侧完全一致。
# [CN] 自适应验证会把 CPU 侧偏移压缩掉，这类后端就不兼容。
# [CN] 注意：有些模型硬编码了后端，不走 attention selector，因此这里要再查一遍。
def get_query_lens_mismatch_unsupported_backend(
    attn_groups: list[list[AttentionGroup]],
    checked_layer_names: set[str] | None = None,
) -> str | None:
    """Name the first backend needing the CPU query lengths to be exact, if any.

    The attention selector already excludes these when adaptive verification is
    enabled, but models that hard-wire their backend never consult it. See
    AttentionBackend.supports_device_cpu_query_lens_mismatch().
    """
    for groups in attn_groups:
        for group in groups:
            if checked_layer_names is not None and checked_layer_names.isdisjoint(
                group.layer_names
            ):
                continue
            if not group.backend.supports_device_cpu_query_lens_mismatch():
                return group.backend.__name__
    return None


# [CN] 分配 KV cache 显存并绑定到各层。
# [CN] 双注意力模型（LongCat-Flash）每个 decoder layer 有两个 Attention 模块，
# [CN] 所以层名里带两个整数，bind 时要传 num_attn_module=2。
def init_kv_cache(
    runner_kv_caches: list[torch.Tensor | list[torch.Tensor]],
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
    kv_cache_allocation_context: AbstractContextManager | None = None,
) -> dict[str, Any]:
    allocation_context = kv_cache_allocation_context or nullcontext()
    with allocation_context:
        kv_caches = allocate_kv_cache(
            kv_cache_config,
            device,
            vllm_config.cache_config.get_resolved_kv_cache_layout(),
            kernel_block_sizes,
        )
    for layer_name, target in get_shared_kv_cache_layers(vllm_config).items():
        kv_caches[layer_name] = kv_caches[target]
    # Dual-attention models (e.g. LongCat-Flash) put two Attention modules per
    # decoder layer, so a layer name carries two integers (layer + module index).
    num_attn_module = (
        2
        if vllm_config.model_config.hf_config.model_type
        in ("longcat_flash", "longcat_flash_ngram")
        else 1
    )
    bind_kv_cache(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_groups=kv_cache_config.kv_cache_groups,
    )
    return kv_caches


def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping
    return slot_mappings_by_layer


# [CN] 为每个 KV cache group 构造 CommonAttentionMetadata，
# [CN] 再交给该 group 内各 attention group 的 builder 产出各自的 metadata，
# [CN] 最后按层名展开成 {layer_name: metadata}。
# [CN] 同一 attention group 内的所有层共享同一份 metadata（省内存与构造时间）。
def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    is_prefilling: torch.Tensor | None = None,
    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    causal: bool | torch.Tensor | Mapping[int, bool] = True,
    rswa_prefix_lens: torch.Tensor | None = None,
    ubatch_idx: int = 0,
) -> dict[str, Any]:
    seq_lens = seq_lens[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        # [CN] 混合草稿器里不同 group 的因果性可能不同（SWA 与 full 混用），
        # [CN] 因此 causal 允许是 dict，按 group id 取值。
        # Per-group causal for hybrid drafters (mixed SWA/full attention).
        group_causal = (
            causal if isinstance(causal, (bool, torch.Tensor)) else causal.get(i, True)
        )

        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        # Model-specific metadata (e.g. Mamba hybrid) may supply its own
        # padding-aware is_prefilling, which takes precedence over the default.
        group_is_prefilling = common_attn_metadata_extra_kwargs.pop(
            "is_prefilling", is_prefilling
        )
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=group_causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            is_prefilling=group_is_prefilling,
            mm_req_doc_ranges=mm_req_doc_ranges,
            rswa_prefix_lens=rswa_prefix_lens,
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(ubatch_idx)
            if for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata


# [CN] PrefixLM：多模态 token 段内部允许双向注意力。
# [CN] 超过 sliding_window 的段会被跳过，否则靠前的 token 会跨越整张图去 attend。
def compute_mm_prefix_ranges(
    req_ids: list[str],
    mm_features: dict[str, list[MultiModalFeatureSpec]],
    sliding_window: int | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """Compute PrefixLM bidirectional ranges for multimodal tokens.

    Ranges exceeding sliding_window are skipped to prevent early tokens
    from attending across the entire image span.
    """
    req_doc_ranges: dict[int, list[tuple[int, int]]] = {}
    for req_idx, req_id in enumerate(req_ids):
        image_doc_ranges = []
        for mm_feature in mm_features.get(req_id, ()):
            if mm_feature.modality not in ("image", "video"):
                continue
            for r in mm_feature.mm_position.extract_embeds_range():
                if sliding_window is not None and (r[1] - r[0] + 1) > sliding_window:
                    continue
                image_doc_ranges.append(r)
        req_doc_ranges[req_idx] = image_doc_ranges
    return req_doc_ranges
