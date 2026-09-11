# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical KV cache layout descriptor.

A leaf module so ``vllm.config`` can import the enum without pulling in the
full KV cache interface.
"""

# [CN] 本模块只做一件事：描述 KV cache 的**物理内存布局**。
#      为什么要单独拆成一个"叶子模块"（leaf module）？
#      因为它被 vllm.config 依赖，而 vllm.config 几乎被所有东西依赖 ——
#      如果和 kv_cache_interface 放一起，必然造成循环导入。
#
#      核心概念：逻辑形状恒为 [L, B, H, N, C]（RFC #42082）
#        L = 层数 layer      B = 块数 block
#        H = 头数 head       N = 每块 token 数（= block_size）
#        C = 每头维度 head_dim（或者更广义的 content）
#      而**物理**排布可以是这 5 个轴的任意排列；每个枚举成员的值
#      就是一个"物理轴位置 -> 逻辑轴编号"的 stride 置换。
#      例：LBNHC = (0,1,3,2,4) 表示物理第 2 轴放的是逻辑 N，
#          物理第 3 轴放的是逻辑 H，即 [L, B, N, H, C]。
from enum import Enum

# [CN] 给 5 个逻辑轴起名字，下面所有下标运算都用这些常量而不是魔法数字。
# Logical dim indices in the 5D stride permutation [L, B, H, N, C] (see: RFC #42082).
_DIM_L, _DIM_B, _DIM_H, _DIM_N, _DIM_C = 0, 1, 2, 3, 4


class KVCacheLayout(Enum):
    """Physical layout descriptor for a KV cache group.

    The logical shape is always [L, B, H, N, <content>] (RFC #42082).
    Each member's value is a stride permutation that maps logical axes
    to physical (memory) order.
    """

    # [CN] 六种受支持的布局。挑几个典型的：
    #      LBHNC：恒等置换，纯逻辑序（调试/参考实现常用）
    #      LBNHC：同层内先按 token(N) 再按 head(H)，某些 attention 核喜欢
    #      BHLNC：block 在最外层，一个 block 的 [H,N,C] 连续 —— 便于整块拷贝
    LBHNC = (0, 1, 2, 3, 4)  # [L, B, H, N, C] (identity)
    LBNHC = (0, 1, 3, 2, 4)  # [L, B, N, H, C]
    LHBNC = (0, 2, 1, 3, 4)  # [L, H, B, N, C]
    BLHNC = (1, 0, 2, 3, 4)  # [B, L, H, N, C]
    BLNHC = (1, 0, 3, 2, 4)  # [B, L, N, H, C]
    BHLNC = (1, 2, 0, 3, 4)  # [B, H, L, N, C]

    @property
    def stride_order(self) -> tuple[int, ...]:
        return self.value

    # [CN] 去掉 L 轴、并把剩余下标减 1，得到"单层视角"的物理轴序。
    #      因为实际内核拿到的往往是某一层的 4D 视图。
    @property
    def layer_view_order(self) -> tuple[int, ...]:
        """Physical axis order of a logical 4D per-layer cache view."""
        return tuple(i - 1 for i in self.value if i != _DIM_L)

    # [CN] L 在最外层 => 同一层的所有数据在物理上连成一片（层紧凑）。
    #      按层做权重/缓存操作时（比如逐层传输）这种布局最省事。
    @property
    def is_layer_compact(self) -> bool:
        """True when the layer is compact; i.e. the L dimension is outermost."""
        return self.value[_DIM_L] == 0

    # [CN] 最后三个物理轴正好是 H、N、C => 一个 block 内部完全连续，
    #      可以当成一整块内存做 memcpy / DMA。
    @property
    def is_block_contiguous(self) -> bool:
        """True when [H, N, C] is contiguous within a block."""
        return self.value[-3:] == (_DIM_H, _DIM_N, _DIM_C)

    # [CN] L 和 B 都在最外两层 => 每个 page 的 [H,N,C] 是**一段连续内存**，
    #      这决定了能不能按页做零拷贝传输（PD 分离、KV transfer 很看重这点）。
    @property
    def is_block_compact(self) -> bool:
        """True when each page's [H, N, C] bytes form one contiguous run; i.e.
        the L and B dimensions are outermost."""
        return set(self.value[:2]) == {_DIM_L, _DIM_B}

    # [CN] B 在最外层：不同 block 之间完全分开，便于 block 粒度的换入换出。
    @property
    def is_block_outermost(self) -> bool:
        """True when B is the outermost physical dimension."""
        return self.value[0] == _DIM_B
