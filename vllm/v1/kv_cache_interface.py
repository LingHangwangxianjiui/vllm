# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from fractions import Fraction
from functools import cached_property
from math import prod
from typing import TYPE_CHECKING, TypeVar

# [CN] 本模块定义 **KV cache 的"规格"（spec）体系** —— 只描述"长什么样、
#      占多少字节"，不做任何分配。分配逻辑在 v1/core/ 下的各 manager 里。
#
#      理解这个文件的三条主线：
#        1) KVCacheSpec 家族：一层（或一组同构层）的 KV cache 长什么样
#           —— 多少 head、head_dim 多大、什么 dtype、是否量化、
#           一个 block 多少字节（page_size_bytes）。
#        2) 形状/布局计算：把一维字节缓冲"看成" 5D 逻辑张量
#           [L,B,H,N,C]（见 compute_layout_strides / create_kv_cache_views）。
#        3) 配置聚合：KVCacheGroupSpec / KVCacheConfig —— 哪些层共用一个
#           block table、总共多少个 block、要不要清零等。
#
#      为什么用 frozen dataclass：spec 在启动期算好之后就到处传递、
#      还被用作 dict key 和缓存键，可变会带来极难排查的问题。
import torch
from typing_extensions import Self

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_layout import _DIM_B, _DIM_L, KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_SpecT = TypeVar("_SpecT", bound="KVCacheSpec")


# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


# [CN] KV cache 量化模式。用 IntEnum 而不是字符串，让 attention 后端
#      可以直接 switch 分发（避免到处做字符串比较，也避免拼错）。
class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    INT4_PER_TOKEN_HEAD = 4  # packed 2×int4/byte, RHT + asymmetric zp
    NVFP4 = 5  # packed fp4 data + fp8 block scales
    # Hadamard-rotated Lloyd-Max quant, packed K+V per slot.
    TURBOQUANT_K8V4 = 6
    TURBOQUANT_4BIT_NC = 7
    TURBOQUANT_K3V4_NC = 8
    TURBOQUANT_3BIT_NC = 9
    NVFP4_DS_MLA = 10  # opaque-bytes NVFP4 DS-MLA layouts (FlashMLA sparse)

    # [CN] 是否"每个 token、每个 head 一组 scale"。这类量化除了 KV 数据本身，
    #      还要额外存一份 scale 张量 —— 下游算 page_size 时必须算进去。
    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4

    @property
    def is_turboquant(self) -> bool:
        """True for any turboquant quantization mode."""
        return self in (
            KVQuantMode.TURBOQUANT_K8V4,
            KVQuantMode.TURBOQUANT_4BIT_NC,
            KVQuantMode.TURBOQUANT_K3V4_NC,
            KVQuantMode.TURBOQUANT_3BIT_NC,
        )


# [CN] 字符串 -> 枚举。注意两处**顺序敏感**：
#        nvfp4_ds_mla 必须在 nvfp4 前缀判断**之前**（否则被前缀吃掉）；
#        fp8 前缀判断放在最后兜底。
def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int4_per_token_head":
        return KVQuantMode.INT4_PER_TOKEN_HEAD
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    # [CN] 就是上面说的顺序陷阱：nvfp4_ds_mla 也以 nvfp4 开头，必须先判。
    # Must precede the ``nvfp4`` prefix test below, which would otherwise match.
    if kv_cache_dtype == "nvfp4_ds_mla":
        # Page size is keyed on cache_dtype_str in the MLA specs, not
        # nvfp4_kv_cache_full_dim.
        return KVQuantMode.NVFP4_DS_MLA
    if kv_cache_dtype.startswith("nvfp4"):
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_"):
        return KVQuantMode[kv_cache_dtype.upper()]
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


# [CN] 把一个 spec 重建为另一个 spec 类（例如 SlidingWindowSpec ->
#      FullAttentionSpec，用于关闭混合分配器时的降级）。
#      语义：共有字段照搬，drop 里的字段丢掉，目标类独有的字段用默认值。
def replace_as(
    spec: KVCacheSpec,
    target_cls: type[_SpecT],
    *,
    drop: Collection[str] = (),
    **changes,
) -> _SpecT:
    """``dataclasses.replace``, but rebuilding *spec* as *target_cls*
      e.g. ``SlidingWindowSpec`` -> ``FullAttentionSpec``

    Every field of *spec* must exist on *target_cls* unless named in *drop*;
    fields only *target_cls* has keep their default values.
    """
    kwargs = {
        f.name: getattr(spec, f.name)
        for f in fields(spec)
        if f.init and f.name not in drop
    }
    kwargs.update(changes)
    return target_cls(**kwargs)


def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


# [CN] spec 的"种类"标签（供日志、统计、外部系统使用）。
#      注意它和类继承体系是**两套**分类：kind 是扁平枚举，
#      便于序列化与跨版本兼容（见 get_kv_cache_spec_kind）。
class KVCacheSpecKind(str, Enum):
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    MAMBA = "mamba"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    SINK_FULL_ATTENTION = "sink_full_attention"
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    CROSS_ATTENTION = "cross_attention"
    UNKNOWN = "unknown"


# [CN] **所有 KV cache 规格的基类**。它对子类要求的四个核心量：
#        num_heads            ：H，多少个 KV head（或 slot）
#        tokens_per_state     ：一个 state 覆盖几个 token（压缩/膨胀）
#        state_content_size_bytes：C，单 head 单 state 的内容字节数
#        page_size_bytes      ：一个 block（block_size 个 token）多少字节
#      内存布局一律抽象成 [B, H, N, C]：
#        B = block 数，N = 每块 state 数，C = 每 state 字节
#      于是 **page_size_bytes = H * N * C**，几乎所有容量计算都从这里出发。
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    # [CN] 该 group 是否参与前缀缓存。Mamba 的循环状态、环形缓冲等
    #      "内容会被就地改写"的缓存必须返回 False（复用会读到脏状态）。
    @property
    def prefix_cacheable(self) -> bool:
        """Whether this spec's group participates in prefix caching."""
        return True

    # [CN] 逻辑 [B,H,N,C] 里的 H。通常由子类覆盖（可能被 num_head_slots 改写）。
    @property
    def num_heads(self) -> int:
        raise NotImplementedError

    # [CN] 一个 state 覆盖几个 token：
    #        >1 的整数 = 多个 token 压进一个 state（DSv4 稀疏 MLA）
    #        0~1 的分数 = 一个 token 要存多个 state（Whisper 的 block pooling）
    #      用 Fraction 而不是 float，是为了避免浮点误差导致的整除判断错误。
    @property
    def tokens_per_state(self) -> int | Fraction:
        raise NotImplementedError

    @property
    def state_content_size_bytes(self) -> int:
        raise NotImplementedError

    # [CN] 一个 block（page）的字节数 —— **容量计算的基石**。
    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    # [CN] 一个 block 里有多少个 state = block_size // tokens_per_state。
    @property
    def num_states(self) -> int:
        return self.get_num_kernel_states(self.block_size)

    # [CN] 允许用"内核块大小"（可能小于管理块）来算：
    #      内核看到的 N 与管理层看到的 N 可能不同（块细分）。
    def get_num_kernel_states(self, kernel_block_size: int) -> int:
        if self.tokens_per_state > 0:
            return kernel_block_size // self.tokens_per_state
        return 1

    # [CN] 这个 group **最多**可能吃多少字节 —— 启动时用它来算
    #      "显存够不够、能分多少个 block"（见 profile_run）。
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    # [CN] 每个请求需要多少个 **block table 表项**（= worker 侧 block table
    #      的行宽）。默认就是 ceil(max_len / block_size)。
    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        """
        The number of block table entries needed per request, i.e. the row
        length of the worker-side block table for this cache group.

        Args:
            vllm_config: The vllm config.
            max_len: The maximum sequence length to size for, including the
                encoder length for encoder-decoder models.
        """
        return cdiv(max_len, self.block_size)

    # [CN] 换 block_size 重建一个 spec（allocator 调 block size 时用）。
    def copy_with_new_block_size(self, block_size: int) -> Self:
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        return replace(self, block_size=block_size)

    # [CN] 把一层层 spec **合并成一个 group 的 spec**。
    #      基类要求所有层完全一致；子类会放宽（例如允许合并 window size）。
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        if not all(spec == specs[0] for spec in specs[1:]):
            raise AssertionError(
                "All layers in the same KV cache group must be the same."
            )
        return copy.deepcopy(specs[0])

    # [CN] 判断一批 spec 是否"同构"到可以放进一个 group。
    #      这里**走注册表**取 uniform_type_base_spec，所以自定义子类
    #      只要声明了基类，也能正确分组（插件化的关键）。
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        """
        Whether this KVCacheSpec is uniform with all specs of all layers.
        """
        uniform_type_base_spec = KVCacheSpecRegistry.get_uniform_type_base_spec(self)
        assert uniform_type_base_spec is not None, (
            f"Unsupported KV cache spec type: {type(self)}. "
            "Please register it using @register_kv_cache_spec decorator."
        )
        return all(
            isinstance(spec, uniform_type_base_spec) for spec in kv_cache_specs.values()
        )


# [CN] 把"内核块"粒度的张量重新 view 成"管理块"粒度：
#      纯 view 操作（unflatten），不拷贝数据。
def group_kernel_blocks(cache: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """View a kernel-block-granular layer cache with manager blocks as dim 0.

    Kernel block splitting subdivides each manager block into uniformly strided
    kernel blocks, so grouping is a pure view: ``(num_blocks * ratio, ...)``
    """
    if cache.shape[0] == num_blocks:
        return cache
    assert cache.shape[0] % num_blocks == 0
    return cache.unflatten(0, (num_blocks, -1))


# [CN] 算 4D 逻辑形状 (B, H, N, C)，其中 C 的单位是**字节**而不是元素数 ——
#      因为不同层 dtype 可能不同，用字节做统一尺度。
def compute_layer_kv_cache_shape_bytes(
    spec: KVCacheSpec,
    num_blocks: int,
    kernel_block_size: int | None = None,
) -> tuple[int, ...]:
    """Return the 4D logical shape ``(B, H, N, C)`` where C is in bytes."""
    bs = kernel_block_size if kernel_block_size is not None else spec.block_size
    assert spec.block_size % bs == 0, (
        f"Kernel block size {bs} must divide KV cache block size {spec.block_size}."
    )
    blocks_per_page = spec.block_size // bs
    return (
        num_blocks * blocks_per_page,
        spec.num_heads,
        spec.get_num_kernel_states(bs),
        spec.state_content_size_bytes,
    )


# [CN] 按给定 layout 算 5D [L,B,H,N,C] 的**字节 stride**。
#      做法：从物理最后一轴往前推，当前 stride 累乘形状。
#      fixed_strides 允许外部强制指定某些轴的 stride
#      （worker 侧已按某种布局分配好了，这里要对齐过去）。
def compute_layout_strides(
    spec: KVCacheSpec,
    num_blocks: int,
    num_layers: int,
    layout: KVCacheLayout,
    kernel_block_size: int | None = None,
    fixed_strides: tuple[int | None, ...] = (None,) * 5,
) -> tuple[int, ...]:
    """Byte strides in logical ``[L, B, H, N, C]`` axis order."""
    assert len(fixed_strides) == 5
    assert all(stride is None or stride > 0 for stride in fixed_strides)
    shape = (
        num_layers,
        *compute_layer_kv_cache_shape_bytes(spec, num_blocks, kernel_block_size),
    )
    order = layout.stride_order
    padded_page_size = getattr(spec, "page_size_padded", None)
    if padded_page_size is not None:
        assert kernel_block_size is None or kernel_block_size == spec.block_size, (
            "Padded KV pages do not support kernel block splitting."
        )
        page_grid_end = max(order.index(_DIM_L), order.index(_DIM_B)) + 1
        page_grid_shape = tuple(shape[dim] for dim in order[:page_grid_end])
        assert prod(page_grid_shape) == num_layers * num_blocks, (
            "Page padding requires dimensions outside the page tail to be L, B, "
            f"or singleton; got {layout.name} with shape {shape}."
        )

        # [CN] 倒序推导 stride：物理最后一轴 stride=1，往前依次乘上形状。
        #      padded_page_size 存在时，在 page 边界处把 stride 抬到对齐值。
    strides = [0] * 5
    current_stride = 1
    for physical_idx, dim in reversed(tuple(enumerate(order))):
        if padded_page_size is not None and physical_idx == page_grid_end - 1:
            current_stride = max(current_stride, padded_page_size)
            assert current_stride % padded_page_size == 0
        strides[dim] = fixed_strides[dim] or current_stride
        current_stride = strides[dim] * shape[dim]
    return tuple(strides)


# [CN] 把一块**扁平的 int8 缓冲** as_strided 成每层的 4D 视图。
#      这是"零拷贝建立 KV cache 视图"的核心：不搬数据，只改解释方式。
def create_kv_cache_views(
    raw: torch.Tensor,
    spec: KVCacheSpec,
    num_blocks: int,
    layout: KVCacheLayout,
    kv_cache_tensor: KVCacheTensor,
    kernel_block_size: int | None = None,
) -> list[torch.Tensor]:
    """View a flat int8 buffer as one 4D ``[B, H, N, C]`` view per layer.

    Block ``b`` of layer ``l`` starts at the tensor offset plus its layer and
    block stride contributions.
    """
    num_layers = len(kv_cache_tensor.layers)
    layer_stride = kv_cache_tensor.layer_stride
    block_stride = kv_cache_tensor.block_stride
    shape_bytes = compute_layer_kv_cache_shape_bytes(
        spec, num_blocks, kernel_block_size
    )
    ratio = shape_bytes[0] // num_blocks
        # [CN] 内核块细分（ratio>1）有个前提：一个管理块必须是**一页致密内存**，
        #      即末尾没有 padding、也没有别的层插在中间。
        #      不满足就只能报错让用户调小 block-size 或换 layer-compact 布局 ——
        #      这正是错误信息里那两句建议的由来。
    if ratio > 1:
        # Kernel blocks subdivide a manager block into `ratio` equal pieces, so
        # they sit a constant stride apart only if a block is one dense page: no
        # padding at its end, and no other layer's page before the next block.
        dense_page_size = prod(compute_layer_kv_cache_shape_bytes(spec, 1)[1:])
        if block_stride != dense_page_size:
            raise ValueError(
                f"The resolved KV cache layout ({layout.name}) does not store "
                "blocks as dense, unpadded pages (block stride "
                f"{block_stride} != page {dense_page_size}), so a manager "
                f"block cannot be split into {ratio} kernel blocks of "
                f"{kernel_block_size} tokens. Reduce --block-size to "
                f"{kernel_block_size} or set VLLM_KV_CACHE_LAYOUT to a "
                "layer-compact layout (e.g. LBNHC)."
            )
        assert block_stride % ratio == 0, (
            f"Block stride {block_stride} must divide into {ratio} equal kernel blocks."
        )
        block_stride //= ratio

    logical_shape = (num_layers, *shape_bytes)
    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size,
        fixed_strides=(layer_stride, block_stride, None, None, None),
    )
    dtype = getattr(spec, "dtype", None)

    view_5d = torch.as_strided(
        raw,
        size=logical_shape,
        stride=strides,
        storage_offset=raw.storage_offset() + kv_cache_tensor.offset,
    )

    views = []
    for layer_idx in range(num_layers):
        cache_logical = view_5d[layer_idx]
        if dtype is not None:
            cache_logical = cache_logical.view(dtype)
        views.append(cache_logical)
    return views


# [CN] **注意力类** KV cache 的公共字段。
#      与 MambaSpec（状态空间模型）并列，两者对"state"的理解完全不同：
#        注意力：state = 一个 token 的 K/V，N 个 token 就是 N 个 state；
#        Mamba  ：state = 整个序列的循环状态，位置固定、就地更新。
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    # [CN] V 的头维度。多数模型 K/V 同维，所以默认跟着 head_size；
    #      MLA 只有一个 latent 向量，把它设为 0。
    #      注意这里用 object.__setattr__ 是因为 dataclass 是 frozen 的。
    head_size_v: int = None  # type: ignore[assignment]
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    # [CN] **打包后**的 H。当内核把多个 head 打包成一个 slot（比如量化打包）
    #      时，逻辑 H 会不等于 num_kv_heads。None 表示一对一。
    num_head_slots: int | None = None
    """H of the logical ``[B, H, N, C]`` page when packing diverges from one
    slot per KV head. None means one slot per KV head. Published by the backend.
    """
    # [CN] 打包后每格内容的字节数；None 表示"K/V 连续存放、直接算"。
    state_content_bytes: int | None = None
    """C in bytes when packed; None means dense K/V content."""
    # [CN] 见基类同名属性。整数>1 表示多 token 压一 state，
    #      分数<1 表示一 token 存多 state。
    tokens_per_state: int | Fraction = 1
    """Tokens covered by one stored state. Ints > 1 compress multiple tokens
    into one state (DSv4 sparse MLA); fractions < 1 store multiple states per
    token (Whisper block pooling: ``Fraction(1, block_pool_size)``)."""

    # [CN] frozen dataclass 里改字段只能这样绕（object.__setattr__）。
    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    # [CN] 优先用打包后的 num_head_slots，否则就是 KV head 数。
    @property
    def num_heads(self) -> int:
        if self.num_head_slots is not None:
            return self.num_head_slots
        return self.num_kv_heads

    # [CN] C = (head_size + head_size_v) * dtype 字节数，
    #      即 K 和 V 拼起来一整格占多少字节。
    @property
    def state_content_size_bytes(self) -> int:
        """Bytes per (head slot, stored state) cell of the page."""
        if self.state_content_bytes is not None:
            return self.state_content_bytes
        return (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)

    # [CN] 未对齐的 page 大小 = H * N * C。
    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.num_heads * self.num_states * self.state_content_size_bytes

    # [CN] 实际使用的 page 大小：若指定了对齐后的 padded 值就用它
    #      （某些硬件要求 page 按 128B / 512B 对齐，尾部留空洞）。
    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes

    # [CN] 别名：真正存数据的字节数（不含对齐空洞）。TPU 后端在用。
    @property
    def real_page_size_bytes(self) -> int:
        """Alias of ``unpadded_page_size_bytes``
        TODO(lucas): follow up with TPU backend to see if we can remove this property.
        """
        return self.unpadded_page_size_bytes

    # [CN] **DCP（decode context parallel）** 会把序列沿上下文切分，
    #      每个 rank 只存自己那一段，所以表项数要除以 kv_shard_count。
    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)


# [CN] 全注意力 spec。这段 docstring 讲了一个重要背景：
#      **关闭混合分配器**时，滑窗层会被"当成"全注意力来分配
#      （所有 token 都分配块），只是在模型执行时仍按滑窗算。
#      代价是多占显存，好处是分配逻辑统一、不会踩混合管理的坑。
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    """

    # [CN] 记录滑窗大小（可能来自上面说的"降级"场景）。None = 无滑窗。
    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    attention_chunk_size: int | None = None

    # [CN] 是否**非因果**（Prefix LM 之类）。它不影响 KV 布局，
    #      但会影响调度策略（chunked prefill / 前缀缓存要相应调整）。
    #      之所以要放在 spec 上：引擎 core 在建调度器之前收集所有 worker 的
    #      spec，那时还没法按 TP 布局去问模型。
    non_causal: bool = False
    """
    Whether the layer attends non-causally (e.g. Prefix LM). Carried on the
    spec so the engine core, which collects specs from all workers before the
    scheduler is built, can adjust scheduling policy (chunked prefill / prefix
    caching) regardless of tensor-parallel layout. It does not affect the KV
    cache layout itself.
    """

    # [CN] 最坏情况 = ceil(max_model_len / block_size) 个 page。
    #      DCP 时每个 rank 只存 1/dcp 的序列。
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    # [CN] 合并窗口大小：全组必须一致，否则报明确的错（混窗不合法）。
    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    # [CN] 合并一组的 FullAttentionSpec。要点：
    #        - window / chunk size 各自去重合并；
    #        - 只要有一层 non_causal，整组都算 non_causal（保守策略）；
    #        - 不允许"滑窗 + chunked local"两种同时存在。
    #      末尾还有一次全字段一致性校验，防止漏合并某个字段。
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
            tokens_per_state=specs[0].tokens_per_state,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            # If any layer in the group is non-causal, treat the group as
            # non-causal so the engine core disables incompatible scheduling.
            non_causal=any(spec.non_causal for spec in specs),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec


# [CN] MLA 的 page 可能需要按 alignment 对齐（DeepSeek V4）。
#      做法：算出对齐后的大小写回 page_size_padded；
#      注意是**就地改** frozen dataclass（object.__setattr__）。
def _apply_alignment_padding(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


# [CN] **MLA**（Multi-head Latent Attention，DeepSeek 系）。
#      与标准注意力的根本差异：它不存 K/V 两个矩阵，只存一个
#      **低秩 latent 向量**，所以 head_size_v = 0。
#      带来的连锁影响：page_size 小很多、量化方式特殊（见 NVFP4_DS_MLA）。
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    model_version: str | None = None
    storage_block_size: int | None = None
    """Token width used to view storage when it differs from the kernel block."""
    # Group capability enabled when any member flattens a non-causal query block
    # into decode rows. Runtime metadata still selects causal vs. non-causal mode.
    non_causal_multi_token_decode: bool = False
    # MLA stores a single latent vector per state; there is no separate V.
    head_size_v: int = 0

    def __post_init__(self):
        super().__post_init__()
        _apply_alignment_padding(self)

    # [CN] MLA 的合并比 FullAttention 更严格：cache_dtype_str、
    #      tokens_per_state、model_version、storage_block_size 都必须一致 ——
    #      因为 MLA 的存储格式直接由这些字段决定，不一致就没法共用 block。
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        tokens_per_state_set = set(spec.tokens_per_state for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        storage_block_size_set = set(spec.storage_block_size for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(tokens_per_state_set) == 1
            and len(model_version_set) == 1
            and len(storage_block_size_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, tokens per state, model version, and storage "
            "block size."
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
            cache_dtype_str=cache_dtype_str_set.pop(),
            tokens_per_state=tokens_per_state_set.pop(),
            model_version=model_version_set.pop(),
            storage_block_size=storage_block_size_set.pop(),
            non_causal_multi_token_decode=any(
                spec.non_causal_multi_token_decode for spec in specs
            ),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        return merged_spec


# [CN] 标记类：给 extract_hidden_states 用的隐藏态缓存层，
#      存储格式与 MLA 相同（都是一个 latent），所以直接继承。
@dataclass(frozen=True, kw_only=True)
class HiddenStateCacheSpec(MLAAttentionSpec):
    """Marker for hidden-state cache layers used by extract_hidden_states."""

    pass


# [CN] **R-SWA**（Reference Sliding Window Attention）：
#      prefill（图像+文本）部分**永远可见**，只有最后 rswa_window 个
#      生成 token 留在 KV cache 里；中间那段"空隙"块在每步解码时回收。
#      于是显存从 O(seq) 降到 O(prefix + window)。
@dataclass(frozen=True, kw_only=True)
class RSWASpec(FullAttentionSpec):
    """KV cache spec for Reference Sliding Window Attention (R-SWA).

    Prefill (image + text prompt) tokens are always globally visible.
    Only the last ``rswa_window`` generated tokens are kept in the KV cache;
    gap blocks (between the prefill tail and the current decode window) are
    evicted during each decode step to bound memory at
    O(prefix_blocks + window_blocks).
    """

    rswa_window: int

    @classmethod
    def merge(cls, specs: list[RSWASpec]) -> RSWASpec:
        assert all(isinstance(spec, RSWASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be RSWASpec."
        )
        rswa_windows = {spec.rswa_window for spec in specs}
        assert len(rswa_windows) == 1, (
            f"All R-SWA layers must share the same rswa_window, got {rswa_windows}"
        )
        # Delegate common field merging to the parent, then reattach rswa_window.
        base = FullAttentionSpec.merge(specs)  # type: ignore[arg-type]
        return cls(
            block_size=base.block_size,
            num_kv_heads=base.num_kv_heads,
            head_size=base.head_size,
            head_size_v=base.head_size_v,
            dtype=base.dtype,
            kv_quant_mode=base.kv_quant_mode,
            page_size_padded=base.page_size_padded,
            num_head_slots=base.num_head_slots,
            state_content_bytes=base.state_content_bytes,
            tokens_per_state=base.tokens_per_state,
            sliding_window=base.sliding_window,
            attention_chunk_size=base.attention_chunk_size,
            non_causal=base.non_causal,
            rswa_window=rswa_windows.pop(),
        )


# [CN] **分块局部注意力**（chunked local attention）：
#      只在 attention_chunk_size 大小的块内做注意力，块之间不互相看。
@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    # [CN] 单请求**准入上限**（块数）。这里的重点是注释里那句
    #      "single source of truth"：启动期算池子大小和运行期做准入判断
    #      用的是**同一个函数** —— 否则会出现"启动时算够、运行期却拒绝"
    #      这种极难复现的 bug。
    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for at most one chunk window plus
        # the in-flight tokens, since frees happen on the processed-token basis.
        num_tokens = min(
            self.attention_chunk_size + max_in_flight_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, ChunkedLocalAttentionSpec)
            and spec.attention_chunk_size == self.attention_chunk_size
            for spec in kv_cache_specs.values()
        )


# [CN] **滑窗注意力** spec：只保留最近 sliding_window 个 token 的 KV。
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    # [CN] 窗口尾部**额外保留**的 token 数：这些块留着但不参与注意力。
    #      为什么需要：多模块投机解码可能会"重算"末尾若干个 token，
    #      如果那些块已被回收，重算时就得从头再来。
    # The trailing edge of the window is extended by ``extra_retained_tokens``
    # so that those extra trailing tokens' blocks are retained (but not
    # attended). This is needed for multi-module spec decoding which can
    # re-prefill the last num_spec_prefill_tokens - 1 tokens from the end
    # of the sequence, and thus needs to delay freeing/caching of blocks.
    extra_retained_tokens: int = 0

    # [CN] 滑窗的准入上限：窗口内 token + 额外保留 token + 在途 token，
    #      且不超过 max_model_len。
    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the in-flight tokens (frees happen on the
        # processed-token basis); never more than `max_model_len`. An additional
        # `extra_retained_tokens` trailing tokens are kept alive below the
        # window for multi-module spec decoding, and must be accounted here too.
        num_tokens = min(
            self.sliding_window - 1 + self.extra_retained_tokens + max_in_flight_tokens,
            max_model_len,
        )
        # [CN] 那个 +1 是**窗口不对齐块边界**的补偿。
        #      举例：block_size=4，窗口要存 4 个 token [CDEF]，
        #      但它可能横跨两个块 [XXCD][EF] —— 所以必须多留一块。
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


# [CN] **环形缓冲** spec：每请求固定一块，存放"正在被压缩"的原始 key。
#      prefix_cacheable = False：内容是滚动覆盖的，不能当前缀复用。
@dataclass(frozen=True, kw_only=True)
class CircularBufferSpec(AttentionSpec):
    """One block per request holding the raw keys of the token group that
    is still being compressed.

    ``block_size`` is the ring capacity. It must exceed the compression ratio
    by the speculative lookahead: a speculative step stores all of its rows,
    drafts included, before acceptance is known, while the next step still
    reads the open group's committed keys from the ring.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # The ring occupies one block per request for its whole lifetime.
        del vllm_config
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        del vllm_config, max_len
        return 1

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, CircularBufferSpec) for spec in kv_cache_specs.values()
        )

    @property
    def prefix_cacheable(self) -> bool:
        return False


# [CN] 滑窗 + MLA 的组合（DeepSeek 的滑窗变体）。
@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    model_version: str | None = None

    # MLA stores a single latent vector per state; there is no separate V.
    head_size_v: int = 0

    def __post_init__(self):
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        super().__post_init__()
        _apply_alignment_padding(self)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        tokens_per_state_set = set(spec.tokens_per_state for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        extra_retained_set = set(spec.extra_retained_tokens for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(tokens_per_state_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
            and len(extra_retained_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, tokens per state, model version, sliding "
            "window size, and retained token count."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
            sliding_window=sliding_window_set.pop(),
            extra_retained_tokens=extra_retained_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            tokens_per_state=tokens_per_state_set.pop(),
            model_version=model_version_set.pop(),
        )

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowMLASpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


# [CN] kpool indexer 的"尾部暂存"：固定一个块的循环草稿区。
#      同样不可前缀复用。
@dataclass(frozen=True, kw_only=True)
class KpoolTailSpec(SlidingWindowSpec):
    """One-block circular scratch cache for a kpool indexer's raw tail."""

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        return 1

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        return 1

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(isinstance(spec, KpoolTailSpec) for spec in kv_cache_specs.values())

    @property
    def prefix_cacheable(self) -> bool:
        return False


# [CN] **Mamba / 状态空间模型**的 KV cache spec。
#      和注意力类最大的不同：它的 state 是**定长循环状态**，
#      "一个请求占几个块"与序列长度几乎无关（见 max_memory_usage_bytes）。
#      shapes / dtypes 是元组：因为一个 Mamba 层有多个状态张量
#      （conv state、ssm state 等）。
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    page_size_padded: int | None = None
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0
    num_prefill_checkpoint_blocks: int = 0
    prefill_checkpoint_alignment: int | None = None
    num_heads: int = 1
    tokens_per_state: int = -1
    # [CN] False = 状态沿 TP 切分（如 GDN）；True = 每个 TP rank 各存一份完整状态。
    #      直接影响显存估算：后者要按 rank 数倍增。
    # False: the state is sharded across TP ranks (e.g. GDN). True: every TP
    # rank holds the full state (e.g. the replicated PLE conv state).
    tp_replicated: bool = False

    # [CN] 所有状态张量的字节数之和。
    @property
    def state_content_size_bytes(self) -> int:
        return sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )

    @property
    def real_page_size_bytes(self) -> int:
        return self.state_content_size_bytes

    @property
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    # [CN] 三种 mamba_cache_mode 决定了完全不同的容量模型：
    #        all  ：按整条序列存（显存 O(seq)，但支持任意位置复用）
    #        align：只存 2 + 投机 + 检查点 个状态块（省显存，靠重算补齐）
    #        其它 ：只存 1 + 投机 个块（最省）
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (
                2 + self.num_speculative_blocks + self.num_prefill_checkpoint_blocks
            )
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    # [CN] 注意 align 模式下的坑：虽然常驻只有少数几个块，
    #      但 **block table 的行宽仍要按 max_len 给** —— 因为表是按下标索引的，
    #      早期位置会被置空而不是缩表。注释里讲的就是这个。
    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        # Mamba state is replicated across DCP/PCP ranks, never sharded, so
        # no CP scaling applies.
        if vllm_config.cache_config.mamba_cache_mode == "align":
            # Block table rows are position-indexed over the full sequence
            # even though only 2 + num_speculative_blocks state blocks are
            # resident at a time (earlier states are nulled out by
            # remove_skipped_blocks), so the row length must cover max_len
            # rather than max_memory_usage_bytes.
            return cdiv(max_len, self.block_size) + self.num_speculative_blocks
        return cdiv(self.max_memory_usage_bytes(vllm_config), self.page_size_bytes)

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, MambaSpec)
            and spec.num_speculative_blocks == self.num_speculative_blocks
            and spec.num_prefill_checkpoint_blocks == self.num_prefill_checkpoint_blocks
            and spec.prefill_checkpoint_alignment == self.prefill_checkpoint_alignment
            and spec.page_size_bytes == self.page_size_bytes
            and spec.tp_replicated == self.tp_replicated
            for spec in kv_cache_specs.values()
        )


# [CN] 算 Mamba prefill 的**可复用检查点位置**（向下取到 hash block 边界）。
#      这是 Mamba 能做前缀复用的关键：只有落在块边界上的状态才可复用。
def get_mamba_prefill_checkpoint_position(
    num_tokens: int,
    hash_block_size: int,
    drop_eagle_block: bool,
) -> int:
    """Return the reusable Mamba checkpoint boundary for a prefill."""
    checkpoint_position = (num_tokens - 1) // hash_block_size * hash_block_size
    if drop_eagle_block:
        checkpoint_position -= hash_block_size
    return max(checkpoint_position, 0)


# [CN] 判断这个 query 区间里能不能导出 check point。
#      条件很严格：起点对齐、检查点落在区间内、还要满足对齐要求 ——
#      任一不满足就只能放弃复用（正确性优先）。
def is_mamba_prefill_checkpoint_valid(
    query_start: int,
    query_end: int,
    checkpoint_position: int,
    hash_block_size: int,
    mamba_block_size: int,
    checkpoint_alignment: int | None,
) -> bool:
    """Whether a backend can export the checkpoint in this query."""
    if checkpoint_alignment is None:
        return False
    assert checkpoint_alignment > 0

    initial_state_col = (query_start - 1) // mamba_block_size
    checkpoint_col = cdiv(query_end, mamba_block_size) - 2
    return (
        query_start % hash_block_size == 0
        and checkpoint_col > initial_state_col
        and query_start + hash_block_size <= checkpoint_position
        and query_start < checkpoint_position < query_end
        and (checkpoint_position - query_start) % checkpoint_alignment == 0
    )


# [CN] 仅编码器的注意力层：**不需要 KV cache**（返回 0）。
#      因为编码器是一次性前向，不做自回归。
@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Encoder-only layers do not need KV cache
        return 0


# [CN] encoder-decoder 里的**交叉注意力**：要缓存的是 **encoder 的输出**，
#      容量按最大 encoder 输入长度算（比如 Whisper 的 1500）。
@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # For cross-attention, we need to cache encoder states
        # Get encoder length (e.g., 1500 for Whisper).
        max_encoder_len = vllm_config.scheduler_config.max_num_encoder_input_tokens
        return cdiv(max_encoder_len, self.block_size) * self.page_size_bytes


# [CN] 带 **sink token**（注意力汇聚点）的全注意力：
#      前 sink_len 个 token 永远保留，不参与滑窗淘汰。
@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            sink_len=specs[0].sink_len,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            num_head_slots=specs[0].num_head_slots,
            state_content_bytes=specs[0].state_content_bytes,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            non_causal=any(spec.non_causal for spec in specs),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec


# [CN] **同构多层的聚合 spec**：把若干"每层需要相同 slot 数"的层打包，
#      在 KV cache manager 眼里当成**一层**处理。
#      意义：一组同构层共用一个 block table，管理开销从 O(层数) 降到 O(1)。
#      注意 docstring 里的限制：滑窗大小不同的层**不算**同构，不能合并。
@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs: dict[str, KVCacheSpec]

    # [CN] 只有**所有**层都可前缀缓存，整组才可前缀缓存（一票否决）。
    @property
    def prefix_cacheable(self) -> bool:
        return all(spec.prefix_cacheable for spec in self.kv_cache_specs.values())

    @property
    def first_spec(self) -> KVCacheSpec:
        """Return the first spec in the group."""
        return next(iter(self.kv_cache_specs.values()))

    # [CN] 整组 page = 各层 page 之和（多层打包在一页里）。
    @property
    def page_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    # [CN] 取各层所需 page 数的**最大值**再乘整组 page 大小：
    #      因为一页里要放下所有层，所以按"最能吃"的那层来定。
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    # [CN] 要求组内所有层算出的表项数**完全一致**，不一致直接报错。
    #      原因见注释：metadata builder 是按单层 spec 构造的，
    #      如果组宽了对不齐，运行期会越界。
    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        # Metadata builders are constructed from the per-layer spec, so the base
        # cdiv(max_len, block_size) would drop its DCP sharding and size the
        # block table wider than those builders expect.
        widths = {
            spec.max_num_blocks_per_req(vllm_config, max_len)
            for spec in self.kv_cache_specs.values()
        }
        assert len(widths) == 1, (
            "All layers in the same KV cache group must need the same number "
            f"of block table entries, got {sorted(widths)}."
        )
        return next(iter(widths))

    # [CN] 先查 block_size 是否统一，再用注册表的 uniform_type_base_spec 判定。
    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.

        Uses the registry to determine grouping base classes, so custom specs
        that inherit from FullAttentionSpec are treated as full attention.
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()))
        return first_spec.is_uniform_with_collection(kv_cache_specs)

    # [CN] 同构就构造、不同构返回 None（让调用方降级成"每层一组"）。
    #      这种"返回 Optional"的工厂比直接抛异常更好用。
    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None

    # [CN] 共享同一 page 大小的最多层数 —— 用于判断"层模式重复了几次"，
    #      在分桶（bucket）分配时用来算平衡。
    def get_max_layers_per_page_size(self) -> int:
        """Max number of layers sharing a page size. For a balanced bucket
        this equals the number of repetitions of the layer pattern."""
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


# [CN] 统一"取层 spec"的入口：无论传进来的是聚合 spec 还是单层 spec，
#      都返回逐层的列表，调用方不用写 isinstance 分支。
def iter_layer_specs(kv_cache_spec: KVCacheSpec) -> Collection[KVCacheSpec]:
    """The per-layer specs a KV cache group spec covers.

    ``UniformTypeKVCacheSpecs`` groups keep one spec per layer; every other
    spec describes its group on its own. Returns the layer specs either way so
    callers do not have to special-case the wrapper.
    """
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        return kv_cache_spec.kv_cache_specs.values()
    return (kv_cache_spec,)


# [CN] 判断一个 group spec 是否**全部是全注意力**。
#      为什么不能只做 isinstance：聚合 spec 本身不是 FullAttentionSpec，
#      必须拆开逐层看（注释里点名了 DeepSeek-V4 MLA 这个真实案例）。
def is_full_attention_spec(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether a KV cache group spec is (or wraps) full attention.

    ``UniformTypeKVCacheSpecs`` is not itself a ``FullAttentionSpec``, so a bare
    isinstance check misses groups that carry the wrapper -- DeepSeek-V4's MLA
    layers, or any model taking the ``UniformTypeKVCacheSpecs.from_specs`` path.

    Every layer must be full attention: a group holding a recycling
    (sliding-window) layer has no stable slot layout, so callers that key data
    by slot cannot use it.
    """
    layer_specs = iter_layer_specs(kv_cache_spec)
    return len(layer_specs) > 0 and all(
        isinstance(spec, FullAttentionSpec) for spec in layer_specs
    )


# [CN] spec -> KVCacheSpecKind。注意注释里那句提醒：
#      **子类判断必须排在父类之前**（SlidingWindowMLA 在 MLA 之前、
#      MLA 在 FullAttention 之前），否则子类会被父类的 isinstance 吃掉。
def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        # A group is only formed when all members share one registered
        # uniform_type_base_spec, so UNKNOWN would discard what the merge
        # already established.
        base_specs = {
            KVCacheSpecRegistry.get_uniform_type_base_spec(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(base_specs) == 1 and next(iter(base_specs)) is FullAttentionSpec:
            return KVCacheSpecKind.FULL_ATTENTION
        return KVCacheSpecKind.UNKNOWN
    # Keep subclass checks before base classes so specialized specs keep their
    # more precise kind.
    if isinstance(kv_cache_spec, SlidingWindowMLASpec):
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(kv_cache_spec, SinkFullAttentionSpec):
        return KVCacheSpecKind.SINK_FULL_ATTENTION
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        return KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return KVCacheSpecKind.CROSS_ATTENTION
    return KVCacheSpecKind.UNKNOWN


def get_kv_cache_spec_sliding_window(kv_cache_spec: KVCacheSpec) -> int | None:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_windows = {
            get_kv_cache_spec_sliding_window(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        return next(iter(inner_windows)) if len(inner_windows) == 1 else None
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return kv_cache_spec.sliding_window
    return None


# [CN] 一块 KV cache 物理内存的**放置描述**：
#      第 l 层第 b 块 = offset + l*layer_stride + b*block_stride。
#      这段 docstring 最重要的信息在最后一句：
#      **不同 group 的 tensor 地址范围可以重叠**（故意 aliasing），
#      之所以安全，是因为同一个 block id 在任一时刻只被一个 group 持有。
#      这是"分层共享同一块显存"的核心技巧。
@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.

    Placement of a set of same-shaped layers in the KV cache allocation.
    Layer ``layers[l]``'s page for block ``b`` starts at
    ``offset + l * layer_stride + b * block_stride`` bytes into the backing
    allocation of ``size`` bytes. Layer-outermost layouts give each layer a
    contiguous region (``layer_stride = page * num_blocks``,
    ``block_stride = page``); block-outermost layouts make each block a
    block of all layers' pages (``layer_stride = page``, ``block_stride`` =
    the packed block). Tensors whose address ranges overlap
    alias the same bytes: cache groups overlay each other, which is sound
    because a block ID is owned by one group at a time.
    """

    size: int  # total size of the backing allocation in bytes
    layers: list[str]  # layer names in L order
    layer_stride: int
    block_stride: int
    offset: int = 0  # byte offset of layers[0]'s block 0


# [CN] 一个 **KV cache group** = 共享同一张 block table 的一组层。
#      在 KV cache manager 眼里它就是"一层"。
@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    is_eagle_group: bool = False
    # Whether this group is part of the externally transferable KV state.
    enable_kv_transfer: bool = True


# [CN] 整个模型的 KV cache 配置（启动时算好，之后几乎只读）：
#        num_blocks        ：总共有多少个 block
#        kv_cache_tensors  ：每块物理内存怎么放（给 worker 初始化用）
#        kv_cache_groups   ：分了哪些组
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """
    prefix_cache_retention_interval: int | None = None
    """Resolved retention policy for local prefix-cache checkpoints."""
    kv_cache_layout: str | None = None
    """The KV cache layout resolved by the engine core, adopted by all workers."""

    # [CN] 参与**外部 KV 传输**（PD 分离 / KV connector）的组。
    #      不是所有组都需要传（比如 Mamba 状态、草稿模型的组）。
    @cached_property
    def transfer_group_ids(self) -> tuple[int, ...]:
        """IDs of cache groups that participate in external KV transfer."""
        return tuple(
            group_id
            for group_id, group in enumerate(self.kv_cache_groups)
            if group.enable_kv_transfer
        )

    @cached_property
    def transfer_groups(self) -> tuple[KVCacheGroupSpec, ...]:
        """Cache groups that participate in external KV transfer."""
        return tuple(
            self.kv_cache_groups[group_id] for group_id in self.transfer_group_ids
        )

    @cached_property
    def transfer_group_index_by_layer(self) -> dict[str, int]:
        """Transfer-group tuple index for each participating layer."""
        return {
            layer_name: group_index
            for group_index, group in enumerate(self.transfer_groups)
            for layer_name in group.layer_names
        }

    # [CN] 从"所有组的 block ids"里挑出需要传输的那几组。
    #      先校验长度再索引，避免静默错位。
    def select_transfer_block_ids(
        self, block_ids: Sequence[list[int]]
    ) -> tuple[list[int], ...]:
        """Select block IDs for externally transferable cache groups."""
        if len(block_ids) != len(self.kv_cache_groups):
            raise ValueError(
                f"Expected {len(self.kv_cache_groups)} KV cache groups, "
                f"got {len(block_ids)}."
            )
        return tuple(block_ids[group_id] for group_id in self.transfer_group_ids)

    # [CN] 是否含 Mamba 层（决定要不要走特殊的清零/分配路径）。
    @property
    def has_mamba_layers(self) -> bool:
        return any(
            isinstance(spec, MambaSpec)
            for group in self.kv_cache_groups
            for spec in iter_layer_specs(group.kv_cache_spec)
        )

    # [CN] 是否**混合精度**：不同组用了不同的 (dtype, quant_mode)。
    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """Whether attention groups store their KV cache at more than one precision."""
        kv_cache_precisions: set[tuple[torch.dtype, KVQuantMode]] = set()
        for group in self.kv_cache_groups:
            kv_cache_precisions.update(
                (spec.dtype, spec.kv_quant_mode)
                for spec in iter_layer_specs(group.kv_cache_spec)
                if isinstance(spec, AttentionSpec)
            )
        return len(kv_cache_precisions) > 1

    # [CN] 新分配的块是否必须**先清零**再用。两种情况必须清：
    #        1) Mamba：状态会先被读、后写完（读到未初始化数据）；
    #        2) 混合精度：一个块被另一个精度的组复用时，
    #           旧字节按新格式解释可能变成 NaN/Inf。
    #      统一精度时可以省掉这一步（纯属浪费带宽）。
    @property
    def needs_kv_cache_zeroing(self) -> bool:
        """Whether newly allocated KV cache blocks must be zeroed before use.

        Required for Mamba layers, whose state is read before it is fully written
        (#35219), and for mixed-precision caches, where a block reused across
        groups can be reinterpreted under a different precision and decode stale
        bytes to NaN/Inf. Uniform-precision caches skip zeroing.
        """
        return self.has_mamba_layers or self.has_mixed_precision_kv_cache
