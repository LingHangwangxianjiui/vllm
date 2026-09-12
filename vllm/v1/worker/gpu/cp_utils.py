# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：Context Parallel（CP / DCP）下的「本地视角」换算。
# [CN] CP 把一条序列的 KV 按 interleave 粒度轮转切给各 rank，
# [CN] 因此每个 rank 看到的 seq_len 与 slot 都不是全局值，需要换算。
# [CN] 两个核心换算：
# [CN]   1) 全局 seq_len -> 本 rank 的 local_seq_len；
# [CN]   2) 全局 position -> 本 rank 的 slot（不属于本 rank 的给 PAD_ID）。
import torch

from vllm.triton_utils import tl, triton


# [CN] 用 Triton kernel 填充持久缓冲，而不是在 Python 里算：
# [CN] 这样 CUDA graph 捕获时可以整段重放（CUDA graph safe）。
def prepare_dcp_local_seq_lens(
    dcp_local_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
) -> None:
    """Populate the persistent DCP local seq_lens buffer (CUDA graph safe)."""
    if dcp_size == 1:
        return

    max_num_reqs = dcp_local_seq_lens.shape[0]
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv(max_num_reqs, BLOCK_SIZE)
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )


@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    # [CN] 轮转分配：每 dcp_size × cp_interleave 个 token 为一轮，
    # [CN] 每轮里第 rank 段（长 cp_interleave）归本 rank。
    # Distribute KV cache among different ranks, in a round-robin manner.
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)

    remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
    remainder = tl.minimum(remainder, cp_interleave)
    local_seq_lens = rounds * cp_interleave + remainder

    # For [num_reqs, max_num_reqs), pad with 0
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)


@triton.jit
# [CN] 返回本 rank 负责的 slot；不归本 rank 的位置返回 PAD_ID。
# [CN] 注意 CP_SIZE == 1 时直接短路，省掉无谓的整除运算。
def cp_local_slot(
    positions,
    block_numbers,
    block_size,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
):
    """Return rank-local KV slots, or PAD_ID for positions not owned by this rank."""
    block_offsets = positions % (block_size * CP_SIZE)
    if CP_SIZE == 1:
        return block_numbers * block_size + block_offsets
    is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
    rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
    remainder = block_offsets % CP_INTERLEAVE
    local_offsets = rounds * CP_INTERLEAVE + remainder
    return tl.where(is_local, block_numbers * block_size + local_offsets, PAD_ID)
