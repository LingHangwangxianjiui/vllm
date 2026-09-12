# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：worker 侧的 **block table**（块表）与 slot mapping。
#
# 位置：调度层（v1/core）决定“请求用哪些块”，本文件负责把结论
#       **翻译成 attention kernel 能直接读的 GPU 张量**：
#         block_table : [max_num_reqs, max_num_blocks_per_req] 的 int32 表，
#                       第 i 行是第 i 个请求用到的块 id 序列；
#         slot_mapping: [num_tokens] 的 int64 表，
#                       第 j 个元素是第 j 个 token 应该写到 KV cache 的哪个槽位。
#
# 两个容易混淆的“块大小”：
#   block_size        —— **kernel** 的块大小（注意力算子要求的大小）；
#   kv_cache_block_size —— **分配/管理**的块大小（v1/core 那边的块）。
#   两者不同时（hybrid blocks），一个管理块要摊成多个 kernel 块。
#
# 末尾还有一个 Triton kernel：把 (block_table, positions) 算成 slot_mapping，
#   顺带处理 CP（context parallel）下的“本机只存一部分 KV”的错位映射。

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


# [CN] 计算块表**宽度**（每行多少个 kernel 块）。
#      两个处理：
#        1) 按 token_alignment（默认 128）向上取整 —— 
#           让每行长度对齐，便于 kernel 向量化与 CUDA graph 复用；
#        2) 从“管理块数”换算成“kernel 块数”（乘 block_size / kernel_block_size）。
def get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int | None = None,
    *,
    token_alignment: int | None = 128,
) -> int:
    """Return the width after optional alignment and virtual block splitting."""
    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"kernel_block_size {kernel_block_size} must divide "
            f"block_size {block_size} evenly"
        )
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment
    return max_num_blocks * block_size // kernel_block_size


# [CN] slot mapping 模式：
#   TOKEN_TO_KV_SLOT —— 普通注意力：每个 token 对应一个 KV 槽位，需要映射；
#   NONE             —— Mamba 类状态缓存：块表直接当“状态下标”用，
#                      不需要逐 token 映射（也就不需要 slot_mapping 缓冲）。
class SlotMappingMode(Enum):
    TOKEN_TO_KV_SLOT = "token_to_kv_slot"
    NONE = "none"


# [CN] 单个 KV cache group 的块表。核心是三块缓冲：
#   block_table  —— CPU/GPU 双份（CpuGpuBuffer），CPU 上填、一次性拷到 GPU；
#   slot_mapping —— 本步每个 token 的目标槽位；
#   num_blocks_per_row —— 每行当前有效长度。
class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        slot_mapping_mode: SlotMappingMode = SlotMappingMode.TOKEN_TO_KV_SLOT,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
            slot_mapping_mode: How this cache group maps scheduled tokens to
                cache slots. Mamba-like state caches do not use token slot
                mappings and should use SlotMappingMode.NONE.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.kv_cache_block_size = block_size

        # [CN] 两种情形：
        #   kernel 块 == 管理块 —— 一一对应，最简单；
        #   kernel 块 <  管理块 —— 需要做“块拆分”（hybrid）：
        #     一个管理块摊成 blocks_per_kv_block 个 kernel 块。
        #     为什么要拆：分配时希望块大一点（减少块表长度、便于复用），
        #     但 kernel 只认小一点的块（算子实现/性能要求）。
        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        # [CN] 取 PCP / DCP 的 world size 与 rank。
        #      CP 下“本机只保存一部分 KV”，所以 slot 映射要按 rank 过滤：
        #      不属于本 rank 的位置填 PAD_SLOT_ID（见下面的 kernel）。
        #      测试环境里通信组可能没初始化，所以用 try/except 兜底为 1/0。
        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size
        self.slot_mapping_mode = slot_mapping_mode
        if self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT:
            _COMPUTE_SLOT_MAPPING_KERNEL.register_warmup(
                kv_cache_block_size=self.kv_cache_block_size,
                blocks_per_kv_block=self.blocks_per_kv_block,
                total_cp_world_size=self.dcp_world_size,
                total_cp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                block_table_stride=self.block_table.gpu.stride(0),
                block_size=self.block_size,
            )

    # [CN] 在指定行**追加**块 id（请求继续生成、拿到新块时用）。
    #      注意只在 CPU 侧 numpy 缓冲上写，真正上传到 GPU 要等
    #      commit_block_table()。这样一次 step 只做一次 H2D 拷贝。
    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    # [CN] 重设某一行（请求重新调度 / 块表重建时用）：先把长度清零再 append。
    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    # [CN] 清空某一行：把用过的部分填 0（0 号块是保留的 null 块）。
    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    # [CN] 把 src 行搬到 tgt 行（请求在批次里换位置时用）。
    #      **同时清空 src 行** —— 这个细节很重要：
    #      dummy batch / mamba 状态槽可能还会引用旧行并**原地写状态**，
    #      而这些块可能已经被释放并重新分配给别人了。
    #      不清零就会踩到别人的块（非常难查的串数据 bug）。
    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        # Clear the vacated source row: dummy-run batches dereference stale
        # rows as mamba state slots and write state in place there, possibly
        # after the blocks have been freed and reallocated.
        block_table_np[src, :num_blocks] = 0
        self.num_blocks_per_row[src] = 0

    # [CN] 交换两行（调度器做请求排序时常用，比 move 更省一次拷贝）。
    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    # [CN] 计算本步的 slot_mapping：把 (block_table, positions) 交给 Triton kernel。
    #      Mamba 类（NONE 模式）直接返回 —— 它们不用逐 token 槽位。
    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        if self.slot_mapping_mode == SlotMappingMode.NONE:
            # Mamba/GDN groups consume the block table as recurrent state
            # indices and do not use per-token slot mappings.
            return
        assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT

        _COMPUTE_SLOT_MAPPING_KERNEL(
            num_reqs,
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            self.kv_cache_block_size,
            self.blocks_per_kv_block,
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )

    # [CN] 把 CPU 侧填好的块表前 num_reqs 行一次性拷到 GPU。
    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    # [CN] 管理块 id -> kernel 块 id 的换算：
    #      管理块 b 展开为 [b*N, b*N+1, ..., b*N+N-1]，N = blocks_per_kv_block。
    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


# [CN] 多 KV cache group 的块表集合（混合注意力模型每个 group 一张表）。
#      它本身只是“对每张表做同样的事”的转发层，
#      但构造时要为每个 group 单独算块表宽度（块大小可能不同）。
class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int],
        cp_kv_cache_interleave_size: int = 1,
        slot_mapping_modes: list[SlotMappingMode] | None = None,
    ) -> None:
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if slot_mapping_modes is None:
            slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT] * len(block_sizes)
        if len(slot_mapping_modes) != len(block_sizes):
            raise ValueError(
                f"slot_mapping_modes length ({len(slot_mapping_modes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # [CN] 按 group 分别计算块表宽度。
        #      NONE 模式（Mamba）不做 token_alignment 对齐：
        #      它把块表当状态下标用，对齐反而会打乱下标语义。
        max_num_blocks = [
            (
                get_block_table_width(n, block_size, token_alignment=None)
                if slot_mapping_mode == SlotMappingMode.NONE
                else get_block_table_width(n, block_size)
            )
            for n, block_size, slot_mapping_mode in zip(
                max_num_blocks, block_sizes, slot_mapping_modes
            )
        ]

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
                slot_mapping_mode=slot_mapping_mode,
            )
            for (
                block_size,
                kernel_block_size,
                max_num_blocks_per_req,
                slot_mapping_mode,
            ) in zip(
                block_sizes, kernel_block_sizes, max_num_blocks, slot_mapping_modes
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]


# [CN] 计算 slot_mapping 的 Triton kernel。
#      输入：query_start_loc（每个请求的 token 区间）、positions（每个 token 的位置）
#            block_table（请求的块序列）；
#      输出：slot_mapping（每个 token 写到哪个 KV 槽位）。
#
#      核心公式：slot_id = block_table[req][pos // block_size] * block_size
#                         + pos % block_size
#
#      CP 下的额外处理：一个“虚拟块”（virtual_block_size = 管理块 * CP world size）
#      的 KV 被切到多个 rank 上，按 interleave 交错存放。
#      所以要先判断这个位置**是否属于本 rank**（is_local），
#      不属于就填 PAD_SLOT_ID（kernel 会跳过它）。
#
#      另外注意第一个 program（req_idx == num_reqs）专门负责：
#      把 slot_mapping 尾部**补满 PAD_ID** —— CUDA graph 要求张量形状固定，
#      不能因为本步 token 少就留着脏数据。
class ComputeSlotMappingKernel(
    VllmTritonJitKernel["ComputeSlotMappingKernel.CompileKey"]
):
    triton_block_size = 1024

    @dataclass(frozen=True)
    class CompileKey:
        kv_cache_block_size: int
        blocks_per_kv_block: int
        total_cp_world_size: int
        total_cp_rank: int
        cp_kv_cache_interleave_size: int
        block_table_stride: int
        block_size: int

    @staticmethod
    @triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
    def kernel(
        num_tokens,
        max_num_tokens,
        query_start_loc_ptr,  # [num_reqs + 1], int32
        positions_ptr,  # [num_tokens], int64
        block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
        block_table_stride,  # max_num_blocks_per_req
        block_size,
        slot_mapping_ptr,  # [max_num_tokens], int64
        KV_CACHE_BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_KV_BLOCK: tl.constexpr,
        TOTAL_CP_WORLD_SIZE: tl.constexpr,
        TOTAL_CP_RANK: tl.constexpr,
        CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
        PAD_ID: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        req_idx = tl.program_id(0)

        if req_idx == tl.num_programs(0) - 1:
            # Pad remaining slots for CUDA graph compatibility.
            for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(
                    slot_mapping_ptr + offsets,
                    PAD_ID,
                    mask=offsets < max_num_tokens,
                )
            return

        start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

        # [CN] 虚拟块大小：CP 把一个管理块的 KV 切到 world_size 个 rank 上，
        #      所以从“位置”角度看，一个虚拟块覆盖 world_size 倍 token。
        virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
        row_offset = req_idx * block_table_stride
        for i in range(start_idx, end_idx, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < end_idx
            pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
            # [CN] 判断这个位置是否属于本 rank：
            #      按 interleave 粒度交错切分，(offset / interleave) % world_size
            #      等于本 rank 才是本地数据。
            is_local = (
                virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
            ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
            local_block_offsets = (
                virtual_block_offsets
                // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
                virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
            )

            block_indices = (
                virtual_block_indices * BLOCKS_PER_KV_BLOCK
                + local_block_offsets // block_size
            )
            block_numbers = tl.load(
                block_table_ptr + row_offset + block_indices,
                mask=mask & is_local,
                other=0,
            ).to(tl.int64)
            slot_offsets = local_block_offsets % block_size
            slot_ids = block_numbers * block_size + slot_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
            tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)

    # [CN] dispatch：把运行期参数转成编译期常量（constexpr），
    #      让 Triton 能为每种形状组合编译一份特化 kernel。
    #      block_table_stride / block_size 用 triton_scalar_specialization_rep
    #      归拢到少数几个档位，避免组合爆炸。
    def dispatch(  # type: ignore[override]
        self,
        *,
        block_table_stride: int,
        block_size: int,
        **compile_key_fields: int,
    ) -> CompileKey:
        return self.CompileKey(
            **compile_key_fields,
            block_table_stride=triton_scalar_specialization_rep(block_table_stride),
            block_size=triton_scalar_specialization_rep(block_size),
        )

    def get_warmup_keys(self, **dispatch_kwargs: int) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**dispatch_kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        int64_ptr = TritonWarmupTensor(torch.int64)
        return dict(
            num_reqs=1,
            num_tokens=2,  # arbitrary, in do_not_specialize
            max_num_tokens=2,  # arbitrary, in do_not_specialize
            query_start_loc=int32_ptr,
            positions=int64_ptr,
            block_table=TritonWarmupTensor(
                torch.int32,
                shape=(1, compile_key.block_table_stride),
            ),
            block_table_stride=compile_key.block_table_stride,
            block_size=compile_key.block_size,
            slot_mapping=int64_ptr,
            kv_cache_block_size=compile_key.kv_cache_block_size,
            blocks_per_kv_block=compile_key.blocks_per_kv_block,
            total_cp_world_size=compile_key.total_cp_world_size,
            total_cp_rank=compile_key.total_cp_rank,
            cp_kv_cache_interleave_size=compile_key.cp_kv_cache_interleave_size,
        )

    # [CN] 启动配置：grid = num_reqs + 1 —— 多出来的那一个 program
    #      专门负责给 slot_mapping 尾部填充 PAD_ID（CUDA graph 需要定长）。
    @kernel_launcher
    def __call__(
        self,
        num_reqs: int,
        num_tokens: int,
        max_num_tokens: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        block_table_stride: int,
        block_size: int,
        slot_mapping: torch.Tensor,
        kv_cache_block_size: int,
        blocks_per_kv_block: int,
        total_cp_world_size: int,
        total_cp_rank: int,
        cp_kv_cache_interleave_size: int,
    ) -> LaunchSpec:
        return (num_reqs + 1,), dict(
            KV_CACHE_BLOCK_SIZE=kv_cache_block_size,
            BLOCKS_PER_KV_BLOCK=blocks_per_kv_block,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=self.triton_block_size,
        )


_COMPUTE_SLOT_MAPPING_KERNEL = ComputeSlotMappingKernel()
