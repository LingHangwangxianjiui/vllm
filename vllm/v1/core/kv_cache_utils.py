# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import hashlib
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NamedTuple, NewType, TypeAlias, cast, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import xxhash, xxhash_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
    iter_layer_specs,
    replace_as,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# [CN] BlockHash：一个 KV cache 块的哈希值，用于**前缀缓存**。
#      用 NewType 而不是裸 bytes，是为了让类型检查器能抓到
#      "把普通字节串当块哈希传"这类误用。
# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# [CN] 带上 **group id** 的块哈希 = 实际的缓存键。
#      为什么必须带 group id：不同 KV cache group 的块内容不同
#      （比如 full attention 与 sliding window），同一个 token 前缀
#      在不同 group 里算出的 KV 也不同，必须分开索引。
#      实现上直接把 group id 以 4 字节大端**拼在哈希后面**，
#      而不是用 tuple —— 省掉一次对象分配（热路径上的常见优化）。
# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# [CN] 对外暴露（KV 事件）时用的哈希形式：bytes 或 int 的联合，
#      保留 int 是为了兼容早期版本的消费方。
# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


# [CN] 打包：hash + group_id 的 4 字节大端表示。
def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    return int.from_bytes(key[-4:], "big", signed=False)


# [CN] 按环境变量决定对外发 bytes 还是 int（后者取低 64 位）。
def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# [CN] NONE_HASH 是**前缀链起点**的哈希（第一个块的"父哈希"）。
#      这段长注释讲了一个很实际的安全/可复现权衡：
#        - 加密哈希（sha256）：用**固定种子**，因此不同进程、不同节点
#          算出的块哈希完全一致，可以跨实例共享前缀缓存；
#          碰撞安全性不依赖种子保密，所以这样做是安全的。
#        - 非加密哈希（xxhash）：必须用**每进程随机种子**，
#          否则攻击者可以离线预构造碰撞块（见 issue #12621）。
#          代价是块哈希不可复现，跨进程无法复用。
#      想跨实例复用就用 sha256，或设置 PYTHONHASHSEED。
# The hash seed for the first block of any prefix block sequence.
#
# For cryptographic hash algorithms it is derived deterministically from a fixed
# default seed, so independent vLLM processes compute identical block hashes for
# identical content and can share a prefix cache (e.g. KV cache reuse across
# nodes) without extra configuration. This does not weaken collision resistance,
# which for SHA-256 does not depend on keeping the seed secret; ``cache_salt``
# remains the mechanism for intentional cache isolation.
#
# Non-cryptographic algorithms keep a per-process random seed, because a
# predictable seed would let an attacker precompute colliding blocks offline
# (see #12621). Setting PYTHONHASHSEED overrides the seed in both cases.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash

# Fixed seed used when the PYTHONHASHSEED environment variable is not set and
# the hash algorithm is cryptographic.
DEFAULT_NONE_HASH_SEED = "vllm-none-hash"

# Algorithms that are not collision resistant, so the seed must stay secret.
_NON_CRYPTO_HASH_FUNCTIONS = frozenset({xxhash, xxhash_cbor})

# The seed NONE_HASH was derived from, set by init_none_hash.
_NONE_HASH_SEED: str | None = None


# [CN] 决定种子：PYTHONHASHSEED 优先；否则加密哈希用固定值，
#      非加密哈希用随机值。
def resolve_none_hash_seed(hash_fn: Callable[[Any], bytes]) -> str:
    """Resolve the seed to derive NONE_HASH from.

    PYTHONHASHSEED wins if set. Otherwise cryptographic algorithms get the
    fixed default (shareable across processes) and non-cryptographic ones get
    fresh random bytes, keeping the seed unpredictable where collision
    resistance depends on it.
    """
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is not None:
        return hash_seed
    if hash_fn in _NON_CRYPTO_HASH_FUNCTIONS:
        return os.urandom(32).hex()
    return DEFAULT_NONE_HASH_SEED


# [CN] 把**已解析出来的种子**暴露出去，而不是让别处重新推导 ——
#      否则随机种子那一路别的组件拿不到，P2P 握手就会对不上。
def get_none_hash_seed() -> str:
    """Return the seed NONE_HASH was derived from.

    Components that must agree on NONE_HASH across processes (the P2P tier
    advertises this during its connect handshake) read the resolved seed here
    instead of re-deriving it, so they observe the random seed too. Falls back
    to the deterministic seed before ``init_none_hash`` has run.
    """
    if _NONE_HASH_SEED is None:
        return DEFAULT_NONE_HASH_SEED
    return _NONE_HASH_SEED


# [CN] 全局初始化 NONE_HASH（进程启动时调用一次）。
def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH, _NONE_HASH_SEED

    _NONE_HASH_SEED = resolve_none_hash_seed(hash_fn)
    if hash_fn in _NON_CRYPTO_HASH_FUNCTIONS and os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "Using a random per-process NONE_HASH seed because %s is not "
            "collision resistant. Block hashes are therefore not reproducible "
            "across processes; set PYTHONHASHSEED to a shared value to reuse "
            "the prefix cache across instances, or use sha256.",
            hash_fn.__name__,
        )
    NONE_HASH = BlockHash(hash_fn(_NONE_HASH_SEED))


# [CN] **一个 KV cache 块的元数据**（不是数据本身，数据在 GPU 上）。
#      @dataclass(slots=True)：块的数量可能有几十万个，
#      用 slots 去掉 per-instance __dict__ 能省一大笔内存。
@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # [CN] 块 id，范围 [0, num_gpu_blocks)。它**就是** GPU 上那块内存的索引。
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # [CN] **引用计数**。0 表示在空闲队列（可被分配/淘汰），
    #      >0 表示被若干请求共享（前缀复用）。
    # Reference count.
    ref_cnt: int = 0
    # [CN] 本块的缓存键（含 group id）。只有"写满且已缓存"的块才有。
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None
    # [CN] _block_hash 覆盖的前缀 token 数。
    #      对完整块 = 块边界；对 partial 条目则可能落在块内部。
    # Number of prefix tokens covered by _block_hash. For full blocks this is
    # the full block boundary; partial entries can end inside a cache block.
    _block_hash_num_tokens: int | None = None

    # [CN] 空闲块**双向链表**的指针。约定：只能由 FreeKVCacheBlockQueue 改。
    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # [CN] null block 标记（占位块，永不缓存、不参与引用计数）。
    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        return self._block_hash_num_tokens

    # [CN] 设置哈希。assert 卡住重复设置 —— 保证"一个块只有一个主哈希"。
    def set_block_hash(
        self,
        block_hash: BlockHashWithGroupId,
        num_tokens: int | None = None,
    ) -> None:
        assert self.block_hash is None and self._block_hash_num_tokens is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    # [CN] 淘汰/复用时清空哈希。
    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None
        self._block_hash_num_tokens = None

    # [CN] repr 里只打印**相邻块的 id**而不是块对象，否则会递归打印整条链表。
    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"_block_hash_num_tokens={self._block_hash_num_tokens}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


# [CN] 一次 **CoW 拷贝** 的描述（源块 -> 目标块），随 SchedulerOutput 下发。
class KVCacheBlockCopy(NamedTuple):
    src_block_id: int
    dst_block_id: int


# [CN] **空闲块双向链表**。为什么不用 collections.deque？
#      因为需要 O(1) 删除**链表中间的任意块**（前缀命中时要把块
#      从淘汰候选里摘出来），而 deque 的 remove 是 O(n)。
#      为了逼近 C++ deque 的性能，这个类**不分配任何 Python 对象**：
#      链表指针直接存在块自身的 prev/next 字段里（侵入式链表）。
#
#      队列顺序 = 淘汰顺序：队头最久未用（LRU），先被淘汰。
#      注意最后一句："释放时反转顺序"的动作在 BlockPool 里做，
#      不在这个类里。
class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    # [CN] 把传入的所有块按顺序串成双向链表。O(n) 一次性建好。
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # [CN] **哨兵头/尾节点**：有了它们，插入删除就不用判断"是不是头/尾"，
        #      既少分支又快。约定：这两个哨兵永远不会被弹出。
        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    # [CN] 弹出队头（= 最该被淘汰的块）。
    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    # [CN] 一次弹出 n 个：只在**最后**接一次链表，比调 n 次 popleft 快。
    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    # [CN] O(1) 摘掉链表中间的块（deque 做不到，这正是本类存在的理由）。
    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    # [CN] 追加到队尾（= 最不容易被淘汰）。
    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    # [CN] 批量插到队头（无缓存块的 LIFO 复用，见 BlockPool.free_blocks）。
    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks at the front of the free list."""
        if len(blocks) == 0:
            return

        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, (
            "next_free_block of fake_free_list_head should always exist"
        )

        prev_block = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block

        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block

        self.num_free_blocks += len(blocks)

    # [CN] 批量追加到队尾（有缓存块的 FIFO，形成 LRU 顺序）。
    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret

    # [CN] 从某个游标之后按淘汰顺序迭代（供外部增量扫描空闲块用）。
    def iter_blocks_after(
        self,
        cursor: KVCacheBlock | None,
    ) -> Iterator[KVCacheBlock]:
        """Iterate free blocks in eviction order after the cursor."""
        if cursor is None:
            curr_block = self.fake_free_list_head.next_free_block
        else:
            curr_block = cursor.next_free_block

        while curr_block is not None and curr_block is not self.fake_free_list_tail:
            yield curr_block
            curr_block = curr_block.next_free_block


# [CN] 生成多模态相关的**额外哈希键**。
#      为什么需要：两个请求可能 prompt token 完全一样，但一张是猫、
#      一张是狗 —— 如果只哈希 token id 就会错误地命中缓存。
#      所以要把"这块里包含哪些多模态输入、以及在块内的偏移"也纳入哈希。
def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # [CN] 前提：mm_features 已按 offset 排序。
    #      这个早退很有效：解码阶段绝大多数块都在所有多模态输入之后，
    #      直接返回，不用遍历。
    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # [CN] start_mm_idx = -1 表示"最后一个多模态输入"，
    #      这是解码阶段（新块由生成 token 填满）的常见情形。
    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx >= offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # [CN] 关键点：把 **mm 输入相对块起点的偏移** 也放进哈希。
            #      否则同一个图片占位符出现在块内不同位置时，
            #      会产生相同的哈希 —— 但它们的 KV 其实不同。
            # The block contains the current mm input. Include its offset
            # relative to the start of the block so prefix-cache keys stay
            # distinct when the same MM item appears at different positions
            # within otherwise-identical placeholder blocks.
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


# [CN] LoRA 相关的额外键：**用 LoRA 名字**（不是 id），
#      因为同一个 id 在不同部署里可能指向不同权重，名字更稳。
def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


# [CN] prompt_embeds 的额外键：对张量做 sha256，
#      并且**按 block 区间缓存**在请求上（避免每步重复算）。
def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """Generate extra keys related to prompt embeds for block hash computation.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.

    Returns:
        Return a stable hash of the block prompt embeddings if prompt embeds
        are present. Return empty list otherwise.
    """
    if request.prompt_embeds is None:
        return []
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # Hash prompt embeds once per block and cache on request
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


# [CN] 汇总三类额外键：LoRA + 多模态 + cache_salt + prompt embeds。
#      没有额外键时返回 None（让 hash 输入保持最简，省算力）。
def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    # [CN] cache_salt 只加在**第一个块**上：
    #      它用于人为隔离缓存（比如多租户），只需在链起点生效一次，
    #      后续块因为链式哈希会自然继承差异。
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


# [CN] **计算一个块的哈希**。核心是：hash(父哈希, 本块 token, 额外键)。
#      这就是所谓的**链式哈希**：每个块的哈希都隐含了它之前的全部内容，
#      于是"两个块哈希相等"就等价于"它们的完整前缀相同" ——
#      这正是前缀缓存能只比一个哈希就判定命中的原因。
#      注意它被 lru_cache 包过（同内容不重复计算）。
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    # [CN] 第一个块没有父，用 NONE_HASH 作为链的起点。
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


# [CN] DCP（decode context parallel）下，注意力的 KV 是按上下文**切分**的，
#      所以一个"逻辑块"实际横跨 dcp_world_size 个物理块的 token 跨度。
def resolve_dcp_kv_block_size(spec: KVCacheSpec, dcp_world_size: int) -> int:
    """Return the token span of a cache block under DCP."""
    layer_specs = iter_layer_specs(spec)
    if len(layer_specs) > 0 and all(
        isinstance(layer_spec, AttentionSpec) for layer_spec in layer_specs
    ):
        return spec.block_size * dcp_world_size
    return spec.block_size


def resolve_dcp_kv_cache_spec(spec: KVCacheSpec, dcp_world_size: int) -> KVCacheSpec:
    """Return a KV cache spec with block sizes adjusted for DCP."""
    block_size = resolve_dcp_kv_block_size(spec, dcp_world_size)
    if block_size == spec.block_size:
        return spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return replace(
            spec,
            block_size=block_size,
            kv_cache_specs={
                name: resolve_dcp_kv_cache_spec(layer_spec, dcp_world_size)
                for name, layer_spec in spec.kv_cache_specs.items()
            },
        )
    return replace(spec, block_size=block_size)


# [CN] 哪些 spec 受 DCP 影响、哪些不受 —— 这段 docstring 讲得很清楚：
#      全注意力（含 MLA）会被分片，所以块几何要乘 DCP；
#      而 Mamba / 滑窗 / 分块局部注意力保存的是**每 rank 各自完整**的状态，
#      即使进程开了 DCP，它们也必须按 dcp=1 计算。
def dcp_world_size_for_kv_cache_spec(spec: KVCacheSpec, dcp_world_size: int) -> int:
    """Return the DCP size that owns this group's block geometry.

    Full-attention KV (including MLA) is sharded across DCP ranks, so prefix
    hashing and manager ``block_size`` use the process DCP size. Other specs
    keep replicated per-rank state (Mamba, sliding window, chunked-local) and
    must keep ``dcp_world_size=1`` even when the process runs with DCP > 1.

    Draft MLA groups on the sharded DSpark path are ``FullAttentionSpec`` /
    ``MLAAttentionSpec`` and therefore keep the process DCP size. A replicated
    draft group would need a different spec, not this helper.
    """
    if dcp_world_size <= 1:
        return 1
    inner = spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        inner = next(iter(spec.kv_cache_specs.values()))
    if isinstance(inner, FullAttentionSpec):
        return dcp_world_size
    return 1


# [CN] 解析出两个关键尺寸（**很容易混淆，务必分清**）：
#        scheduler_block_size：调度器用的 token 对齐粒度
#        hash_block_size     ：计算块哈希的粒度（前缀匹配的最小单位）
#      单 group 时两者相同；多 group 时前者取**最小公倍数**（LCM，
#      保证对所有 group 都对齐），后者取**最大公约数**（GCD，
#      让前缀匹配尽可能细）。
def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler (e.g. for ``num_computed_tokens`` rounding). Single group:
      ``cache_config.block_size * dcp``. Multiple groups: LCM of every
      group's effective block size. Attention groups are scaled by DCP;
      Mamba groups keep their full per-rank state and are not scaled.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.prefix_match_unit`` override if set, else the GCD of
      group block sizes; every group's block size must be divisible by it.
      Returns the scheduler block size (i.e. disables finer hashing) if block
      hashing is inactive or a mamba group is not using cache mode "align".
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    group_block_sizes = [
        resolve_dcp_kv_block_size(g.kv_cache_spec, dcp) for g in groups
    ]
    # [CN] 多 group：调度粒度取 LCM —— 因为 num_computed_tokens 之类的量
    #      必须对所有 group 同时对齐，取整到公倍数才安全。
    scheduler_block_size = math.lcm(*group_block_sizes)

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    # [CN] 只有前缀缓存或 KV connector 开启时才需要细粒度哈希；
    #      都没开就让 hash_block_size 等于 scheduler 块大小（省算力）。
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # [CN] 非 align 模式的 Mamba 组会破坏整除性，只能回退到粗粒度。
    # Mamba groups outside align mode break divisibility; back off to the
    # scheduler block size. Read the mode from the resolved group spec because
    # its block size may have been updated independently of cache_config.
    if any(
        isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align"
        for group in groups
        for spec in iter_layer_specs(group.kv_cache_spec)
    ):
        return scheduler_block_size, scheduler_block_size

    # [CN] 只对**可前缀缓存**的 group 求 GCD；用户可通过
    #      prefix_match_unit 手工指定（更可控但需自行保证整除）。
    hashing_sizes = [
        block_size
        for group, block_size in zip(groups, group_block_sizes)
        if group.kv_cache_spec.prefix_cacheable
    ] or group_block_sizes
    requested = cache_config.prefix_match_unit
    hash_block_size = requested if requested is not None else math.gcd(*hashing_sizes)
    if any(bs % hash_block_size != 0 for bs in hashing_sizes):
        raise ValueError(
            f"Invalid prefix_match_unit={hash_block_size}; prefix-cacheable "
            "KV cache group block sizes must be divisible by prefix_match_unit. "
            f"Got group block sizes={group_block_sizes}, "
            f"prefix-cacheable={hashing_sizes}."
        )
    prefix_alignments = {
        spec.tokens_per_state
        for group in groups
        for spec in iter_layer_specs(group.kv_cache_spec)
        if spec.prefix_cacheable
        and isinstance(spec.tokens_per_state, int)
        and spec.tokens_per_state > 1
    }
    # [CN] Mamba 的 align 模式允许"块内部分边界"复用，
    #      此时命中对齐粒度要用 hash_block_size 而不是整块。
    has_partial_mamba_group = any(
        isinstance(spec, MambaSpec)
        and spec.mamba_cache_mode == "align"
        and (
            (dcp == 1 and block_size > hash_block_size)
            or (dcp > 1 and block_size >= hash_block_size)
        )
        for group, block_size in zip(groups, group_block_sizes)
        for spec in iter_layer_specs(group.kv_cache_spec)
    )
    cache_hit_alignment = (
        hash_block_size if has_partial_mamba_group else scheduler_block_size
    )
    if any(cache_hit_alignment % alignment for alignment in prefix_alignments):
        raise ValueError(
            f"Invalid prefix_match_unit={hash_block_size}; prefix-cache boundaries "
            "must align with each spec's per-state compression. "
            f"Got alignments={sorted(prefix_alignments)}."
        )
    return scheduler_block_size, hash_block_size


# [CN] 返回一个"给请求计算新块哈希"的函数（闭包捕获 hash_block_size）。
#      设计成工厂是因为块大小在启动期才确定。
def get_request_block_hasher(
    hash_block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request.

    Hashes are computed at ``hash_block_size`` granularity and chained over the
    full prefix, so each hash uniquely fingerprints the prefix ending at its
    boundary. Coarser group block sizes and partial-cache boundaries reuse
    these hashes directly (see ``BlockHashListWithBlockSize``).
    """

    # [CN] **增量**计算：只算上次之后新填满的块，老的哈希复用。
    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * hash_block_size
        num_tokens = request.num_tokens

        # [CN] 没有新填满的块就直接返回（解码每一步只生成一个 token，
        #      绝大多数调用都会在这个早退返回 —— 这是热路径）。
        if start_token_idx + hash_block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        curr_mm_idx = 0
        # [CN] 非首块时用 -1（表示"只看最后一个多模态输入"）：
        #      因为能走到这里的块必然是由**生成 token** 填满的，
        #      不可能引入新的多模态输入。
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + hash_block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += hash_block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


# [CN] 显存不够时的报错：除了报错还会**估算**"这个显存大概能跑多长"，
#      比干巴巴一句 OOM 有用得多（用户可以直接照着调 max_model_len）。
def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


# [CN] 所有 spec 的最大内存占用之和。
def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


# [CN] **二分查找**估算"给定显存最多能跑多长的序列"。
#      注意它临时改 max_model_len、用完在 finally 里恢复 ——
#      借用了"修改配置 -> 复用现成计算"的偷懒做法，但保证了无副作用。
def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
        # [CN] 把 model_len 临时塞进 config，复用 max_memory_usage_bytes 来估算。
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    # [CN] 无论是否抛异常都要还原，否则会污染后续所有计算。
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len


# [CN] 启动期校验：至少要能装下**一条** max_model_len 的请求。
def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    if kv_cache_spec:
    # [CN] 减掉一个 block 的开销，是为常驻的 null block 预留；
    #      传的是 spec 的**拷贝**，因为分组过程可能就地修改 spec。
        # Reserve the null block BlockPool permanently holds back, so the check
        # plans against usable blocks, as in get_kv_cache_configs. Group a copy
        # of the specs since grouping may unify them in-place.
        groups = get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
        check_memory = (
            available_memory - _pool_bytes_per_block(groups)
            if groups
            else available_memory
        )
        _check_enough_kv_cache_memory(
            check_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


# [CN] 把"层名分组"变成 KVCacheGroupSpec 列表，
#      每组的最终 spec 由各层 spec 的 merge() 合成。
def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


# [CN] 所有层是否同构。实现很巧妙：**试着 merge 一次**，
#      成功就是同构（因为 merge 内部有全部一致性断言），
#      比罗列一堆 isinstance 判断更不容易漏。
def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


# [CN] 算 **最大并发数** = 总块数 / 单条满长请求要占的块数。
#      注意"单条请求占多少块"是各 group 之和：所有 group 都从
#      **同一个共享块池**里取块，所以要把各组的开销加起来。
def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.

    A request at max_model_len consumes whole blocks from each group's block
    table — cdiv(per-request bytes, page bytes) of the group's spec — and all
    groups draw those block ids from one shared pool, so the per-request
    total is the sum over groups. The memory/page ratio is identical whether
    a group carries an aggregated UniformTypeKVCacheSpecs (worker config) or
    a representative per-layer spec (scheduler config), so both capacity
    call sites agree.
    """
    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency


# [CN] 用户用 num_gpu_blocks_override 手工指定块数时，在这里覆盖掉实测值。
def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


# [CN] 一个块在 worker 共享池里占多少字节 = 后面 num_blocks 计算的除数。
def _pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """
    Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    return _get_kv_cache_bytes_per_block(kv_cache_groups)


# [CN] 所有层 page 大小必须一致，否则说明还没做过 unify。
def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


# [CN] 最简单的情况：所有层 spec 完全一样 -> **一个组**装下所有层。
#      绝大多数模型走这条路。
def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


# [CN] 次简单：层与层的 spec 不完全相等，但**同构**（需要的 slot 数相同），
#      比如都是全注意力、只是 hidden size 不同。仍然合成一个组。
def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


# [CN] 流水线并行（PP）下，Mamba 层要分几组才能让每个 stage 的
#      "Mamba 层数 / MLA 层数"比例都站得住 —— 因为 Mamba 状态要
#      借宿在 MLA 的 page 里（见 _get_kv_cache_groups_glm5_next）。
#      某个 stage 有 Mamba 却没有 MLA -> 返回 None（无法安排，报错）。
def _pp_balanced_mamba_group_count(
    vllm_config: VllmConfig,
    mamba_layer_names: list[str],
    mla_layer_names: list[str],
) -> int | None:
    """Return a Mamba group count whose PP projections fit the MLA slots."""
    num_groups = cdiv(len(mamba_layer_names), len(mla_layer_names))
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    if pp_size == 1:
        return num_groups

    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.models.utils import extract_layer_index

    total_layers = vllm_config.model_config.get_total_num_hidden_layers()
    mamba_indices = [extract_layer_index(name) for name in mamba_layer_names]
    mla_indices = [extract_layer_index(name) for name in mla_layer_names]
    for rank in range(pp_size):
        start, end = get_pp_indices(total_layers, rank, pp_size)
        num_mamba = sum(start <= index < end for index in mamba_indices)
        num_mla = sum(start <= index < end for index in mla_indices)
        if not num_mamba:
            continue
        if not num_mla:
            return None
        num_groups = max(num_groups, cdiv(num_mamba, num_mla))
    return num_groups


# [CN] **GLM-5.3-Flash 的专用分组**（Mamba + MLA 混合架构）。
#      核心技巧是 **aliasing（别名复用）**：
#        - Mamba 的状态页被 padding 到与 MLA 页同宽，于是可以和
#          MLA 层共享同一个 block id（两份数据叠在同一块显存上）；
#        - kpool 的 tail 暂存页同理复用 indexer 页。
#      之所以安全：KVCacheTensor 的地址范围允许重叠，
#      而同一时刻一个块只被一个 group 持有。
def _get_kv_cache_groups_glm5_next(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build GLM-5.3-Flash groups with Mamba/MLA and tail/indexer aliasing."""
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, MambaSpec)
    }
    tail_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, KpoolTailSpec)
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None

    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
    idx_pages = {
        spec.page_size_bytes for spec in mla_specs.values() if spec.tokens_per_state > 1
    }
    if not idx_pages:
        return None

    assert all(spec.page_size_padded is None for spec in mla_specs.values())
    assert len(idx_pages) == 1
    mla_names = [name for name, spec in mla_specs.items() if spec.tokens_per_state == 1]
    mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
    assert len(mla_pages) == 1
    mla_page = mla_pages.pop()
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(attn_specs)
    assert uniform_spec is not None

    tail_group: KVCacheGroupSpec | None = None
    if tail_specs:
        idx_page = next(iter(idx_pages))
        padded_tail_specs: dict[str, KVCacheSpec] = {
            name: replace(spec, page_size_padded=idx_page)
            for name, spec in tail_specs.items()
        }
        tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
        assert tail_uniform is not None
        tail_group = KVCacheGroupSpec(list(padded_tail_specs), tail_uniform)

    any_mamba = next(iter(mamba_specs.values()))
    assert all(spec == any_mamba for spec in mamba_specs.values())
    # [CN] Mamba 状态页必须能塞进 MLA 页，否则 aliasing 不成立 ——
    #      报错并给出可操作的建议（加大 TP 或用更宽的 dtype）。
    if any_mamba.real_page_size_bytes > mla_page:
        raise ValueError(
            f"the mamba state page ({any_mamba.real_page_size_bytes} bytes) "
            f"does not fit the MLA page ({mla_page} bytes); increase tensor "
            "parallelism or use a wider KV cache dtype"
        )
    padded_specs: dict[str, KVCacheSpec] = {
        name: replace(any_mamba, page_size_padded=mla_page) for name in mamba_specs
    }
    num_groups = _pp_balanced_mamba_group_count(
        vllm_config, list(mamba_specs), mla_names
    )
    if num_groups is None:
        raise ValueError(
            "a pipeline stage has mamba layers but no MLA layer to share "
            "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
        )
    # [CN] 按 index % num_groups 轮转分配，让每个 PP stage 都能拿到
    #      均衡的 Mamba/MLA 比例。
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for index, name in enumerate(mamba_specs):
        mamba_grouped_names[index % num_groups].append(name)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )


# [CN] **反查**函数：给定已经分好的组，识别出"这是不是 GLM5 布局"，
#      如果是就把关键几何参数（各页大小、层名分组）解出来。
#      之所以需要反查：分组结果会经过 PP 投影等变换，
#      后面的内存计算需要重新认出这个布局。
def _glm5_next_tensor_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> (
    tuple[
        KVCacheGroupSpec,
        list[KVCacheGroupSpec],
        list[str],
        list[str],
        int,
        int,
        list[str],
        int,
    ]
    | None
):
    """Recognize the GLM-5.3-Flash grouping after optional PP projection."""
    uniform_groups = [
        group
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    ]
    mamba_groups = [
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
    if attn_group is None or not mamba_groups:
        return None
    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):
        return None

    attn_uniform = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
    mla_inner = cast(dict[str, MLAAttentionSpec], attn_uniform.kv_cache_specs)
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in mla_inner.values()
    ):
        return None
    mla_names = [
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state == 1
    ]
    idx_names = [
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state > 1
    ]
    mla_pages = {mla_inner[name].page_size_bytes for name in mla_names}
    idx_pages = {mla_inner[name].page_size_bytes for name in idx_names}
    if len(mla_pages) != 1 or len(idx_pages) != 1:
        return None
    mla_page = mla_pages.pop()
    idx_page = idx_pages.pop()
    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None

    tail_names: list[str] = []
    tail_page = 0
    if tail_group is not None:
        tail_names = list(tail_group.layer_names)
        tail_inner = cast(
            UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
        ).kv_cache_specs
        tail_pages = {
            cast(KpoolTailSpec, spec).unpadded_page_size_bytes
            for spec in tail_inner.values()
        }
        if len(tail_pages) != 1 or len(tail_names) != len(idx_names):
            return None
        tail_page = tail_pages.pop()
        if tail_page > idx_page:
            return None

    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
    )


# [CN] **统一各层的 page 大小**（混合模型分组的前提）。
#      为什么必须统一：所有 group 从同一个块池取块，
#      如果块大小不一致，分配时会产生内存碎片、无法管理。
#      三种手段，按优先级：
#        1) page 能整除最大值 -> 调大 block_size 让它自然变大；
#        2) Mamba / 非 MLA 注意力 -> 直接把物理页 padding 到最大值
#           （读的时候用 strided view，尾部空洞浪费一点）；
#        3) 都不行 -> 抛 NotImplementedError，由上层走兜底路径。
#      MLA 被排除在 padding 之外：稀疏 MLA 按整 token 行索引缓存，
#      只能按它自己的 alignment 对齐，不能随便 pad。
def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Two
    cases cannot be unified by block size alone and pad their physical page to
    the maximum instead: Mamba layers, whose page size comes from state shapes
    and is independent of block size; and non-MLA attention layers whose page
    does not evenly divide the maximum (the padded page is read through a
    strided view). MLA is excluded because sparse MLA indexes the cache in
    whole token rows (see ``flat_kv_row_view``), so its block stride can only
    be padded by its own row-aligned ``alignment``, not to an arbitrary page
    size. Raise NotImplementedError if failed to unify the page size;
    ``get_kv_cache_groups`` catches it to try the full-allocation fallback
    (e.g. MLA next to an incompatible sliding-window draft).

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        # [CN] Mamba 的 page 大小由**状态形状**决定，和 block_size 无关，
        #      所以没法靠调 block_size 变大，只能 pad。
        elif isinstance(layer_spec, MambaSpec):
            # MambaSpec's page size is determined by its state shapes and does
            # not scale with block_size, so pad the page instead. This is the
            # same padding mechanism the platform uses to align Mamba pages
            # with the main model's attention page size; it is needed here
            # when another layer (e.g. from a draft model) has a larger page
            # than the already-aligned Mamba page.
            new_spec: KVCacheSpec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif isinstance(layer_spec, AttentionSpec) and not isinstance(
                layer_spec, MLAAttentionSpec
            ):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for non-MLA attention layers."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


# [CN] 无注意力模型（spec 为空 dict）不需要 KV cache。
def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


# [CN] **混合注意力模型的通用分组**，本文件最核心的算法之一。
#      思路：模型的层是按**模式重复**的（比如 1 层全注意力 + 2 层滑窗，
#      重复 10 次）。于是分成 3 个组，每组 10 层，
#      worker 侧只需为这 3 个组各建一张 block table 再重复套用。
def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    # [CN] 六条假设（读懂这段代码的关键）：
    #        1) 每个 block 的物理内存必须各组相同（否则碎片化无法管理）；
    #        2) 每块 token 数目前统一用 cache_config.block_size；
    #        3) 每 token 每层的字节数由模型配置决定，目前要求全都一样；
    #        4) 每组的层数目前假设相同（不足就补 padding 层）；
    #        5) 组内必须是同一种注意力类型（唯一的例外见第 6 条）；
    #        6) find_longest_cache_hit 目前只支持一种类型，
    #           或"全注意力 + 恰好一种其它类型"。
    #      这些假设都是**为了简化实现**，注释里也写明了哪里可以放松。
    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # [CN] 第一步：按 spec **相等**分桶（KVCacheSpec 是 frozen dataclass，
    #      可直接当 dict key）。
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # [CN] 第二步：尝试把"只有少数属性不同但可 reconcile"的桶再合并
    #      （比如只有滑窗大小不同的全注意力层），以**减少组数**。
    #      判定方式很巧妙：试着 merge 一下，不抛异常就说明能合。
    # Attempt to further merge same-type layers based on whether their KV
    # cache specs can be merged, to minimize the group count. This benefits
    # situations where specs share a block layout and differ only in a
    # property it can reconcile (e.g. full attention layers differing only in
    # sliding window / attention chunk size).
    layer_buckets: list[list[str]] = []
    spec_buckets: list[list[KVCacheSpec]] = []
    for layer_spec, layer_names in same_type_layers.items():
        for names, specs in zip(layer_buckets, spec_buckets):
            try:
                # A raise means that the specs are incompatible.
                type(specs[0]).merge([*specs, layer_spec])
            except (AssertionError, ValueError):
                continue
            names.extend(layer_names)
            specs.append(layer_spec)
            break
        else:
            layer_buckets.append(list(layer_names))
            spec_buckets.append([layer_spec])

    # [CN] 第三步：把每个桶切成若干小组，让**每组层数相同**，
    #      不够就在最后一组补 padding 层。
    #      例：(full.0, full.1) + (sw.0, sw.1, sw.2) -> 3 组各 2 层，
    #      其中一组是 (sw.1, padding)。
    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    # [CN] group_size 的**启发式**：默认取最少层的那个类型（n:1 里的 1）；
    #      但如果最多的也没比最少的多多少（<1.5 倍），就直接取最大值 ——
    #      理由是补 padding 层会浪费显存，而投机解码的 draft 模型往往会
    #      给某一种类型多加几层（注释里举了 gpt-oss-20b + eagle 的例子）。
    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    grouped_layers = []
    for layers in layer_buckets:
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
    # [CN] 这里用 layers[i::num_groups]（**跨步取**）而不是连续切片，
    #      是为了让流水线并行的每个 stage 都分到均衡的层数。
    #      注释里举了反例：连续切会让某个 stage 出现空组，被迫整组 padding。
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


# [CN] 取某层在组内的真实 spec（聚合 spec 要拆开取）。
def _get_per_layer_spec(
    group: KVCacheGroupSpec,
    layer_name: str,
) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


# [CN] 一个块要装下"最大的那个组"的所有层页之和。
def _get_kv_cache_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Return the largest cache group's bytes per block."""
    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout
        return len(mla_names) * mla_page + len(idx_names) * idx_page

    bytes_per_block = max(
        sum(
            _get_per_layer_spec(group, layer_name).page_size_bytes
            for layer_name in group.layer_names
        )
        for group in kv_cache_groups
    )
    assert bytes_per_block > 0
    return bytes_per_block


# [CN] 校验选定的 **layout** 能否表达这个模型的打包方式。
#      混合 page 大小 = 把多个页**并排**塞进一个块，
#      这要求"每页在块内是连续的一段"（block-compact）。
#      走到这里还不行就报错，并提示改 VLLM_KV_CACHE_LAYOUT。
def validate_kv_cache_layout(
    layout: KVCacheLayout,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """Validate that the resolved layout can express this model's packing.

    The layout was chosen once in the engine core from the backends' supported
    sets; a backend whose model packs pages side by side (e.g. the DeepSeek-V4
    indexer) declares block-outermost layouts there, so an inexpressible
    layout reaching this point is an error.
    """
    page_sizes = {
        _get_per_layer_spec(group, layer_name).page_size_bytes
        for group in kv_cache_groups
        for layer_name in group.layer_names
    }
    if len(page_sizes) == 1:
        # A rectangular layer dim exists; every layout can express it.
        return

    # Mixed page sizes pack pages side by side within a block, which needs each page
    # to be one contiguous chunk inside its block (a block-compact layout) and, with
    # multiple KV cache groups, the layer dim inside the block dim.
    if not layout.is_block_compact or (
        len(kv_cache_groups) > 1 and layout.is_layer_compact
    ):
        raise ValueError(
            f"KV cache layout {layout.name} cannot express this model's "
            f"mixed page sizes ({sorted(page_sizes)}); a backend should "
            "declare block-outermost supported layouts (e.g. BLHNC), or "
            "set VLLM_KV_CACHE_LAYOUT=BLHNC."
        )


# [CN] **从分组结果生成最终的 KVCacheConfig**（num_blocks + tensor 布局）。
#      核心一步：available_memory // bytes_per_block = num_blocks。
def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    # [CN] GLM5 特殊布局：手工安排每个层的 offset，让 Mamba / tail 页
    #      **叠在** MLA / indexer 页上（aliasing）。
    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        size = bytes_per_block * num_blocks
        attn_specs = cast(
            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
        ).kv_cache_specs

        kv_cache_tensors: list[KVCacheTensor] = []

        def add_tensor(layer_name: str, spec: KVCacheSpec, offset: int) -> None:
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[layer_name],
                    layer_stride=spec.page_size_bytes * num_blocks,
                    block_stride=spec.page_size_bytes,
                    offset=offset,
                )
            )

        for index, mla_name in enumerate(mla_names):
            offset = index * mla_page * num_blocks
            add_tensor(mla_name, attn_specs[mla_name], offset)
            for group in mamba_groups:
                if index < len(group.layer_names):
                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)

        idx_base = len(mla_names) * mla_page * num_blocks
        for index, idx_name in enumerate(idx_names):
            offset = idx_base + index * idx_page * num_blocks
            add_tensor(idx_name, attn_specs[idx_name], offset)
            if tail_names:
                tail_name = tail_names[index]
                tail_group = next(
                    group for group in kv_cache_groups if tail_name in group.layer_names
                )
                tail_specs = cast(
                    UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
                ).kv_cache_specs
                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    # [CN] 通用路径：先确定 layout 并校验，再算每块字节数。
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    validate_kv_cache_layout(layout, kv_cache_groups)
    bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups)
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None

    num_blocks = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    size = bytes_per_block * num_blocks

    # [CN] **关键设计：所有 group 都从字节 0 开始 aliasing**。
    #      上面那张 ASCII 图讲了两者的区别：
    #        block-outer：每个块里按 [A|B|pad] 排布，每个块重复同样的打包；
    #        layer-outer：每个层一大片连续区域，里面按 block 排开。
    #      能这么叠，是因为同一个 block id 任一时刻只属于一个 group。
    # Groups alias from byte 0. Spec regions are laid out differently:
    #
    # block-outer (the same packing repeats for every block):
    # group 0: | blk 0 [ A | B  | pad ] | blk 1 [ A | B  | pad ] | ...
    # group 1: | blk 0 [  C  |    D   ] | blk 1 [  C  |    D   ] | ...
    #          |<--- bytes_per_block -->|
    #
    # layer-outer (only supported for uniform page sizes or single-group models):
    # group 0: | A [ blk 0 | blk 1 | ... ] | B [ blk 0 | blk 1 | ... ] |
    # group 1: | C [ blk 0 | blk 1 | ... ] | D [ blk 0 | blk 1 | ... ] |

    kv_cache_tensors = []
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            for layer_name, spec in group_spec.kv_cache_specs.items():
                layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
            layers_by_spec[group_spec].extend(group.layer_names)

        byte_offset = 0
        for spec, layer_names in layers_by_spec.items():
            layer_stride, block_stride, _, _, _ = compute_layout_strides(
                spec,
                num_blocks,
                len(layer_names),
                layout,
                fixed_strides=(None, interleaved_block_stride, None, None, None),
            )
            offset = (
                byte_offset
                * max(layer_stride, spec.page_size_bytes)
                // spec.page_size_bytes
            )
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=layer_names,
                    layer_stride=layer_stride,
                    block_stride=block_stride,
                    offset=offset,
                )
            )
            byte_offset += len(layer_names) * spec.page_size_bytes

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )


# [CN] 把**局部注意力**（滑窗 / 分块局部）的 spec **提升**为全注意力 spec。
#      重要：这只影响 **KV cache 的分配**（按全量 token 分配块、不做
#      窗口外回收），注意力模块本身的计算行为**完全不变**。
#      用途：关闭混合 KV cache manager 时的统一化降级路径。
def _promote_local_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Use full-attention allocation for local-attention cache specs.

    The returned specs affect KV cache management only. Attention modules keep
    their original sliding-window or chunked-local compute behavior.
    """
    promoted_specs = kv_cache_spec.copy()

    if is_kv_cache_spec_uniform(
        promoted_specs
    ) or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs):
        return promoted_specs

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in promoted_specs.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in promoted_specs.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in promoted_specs.values()
    )
    full_block_sizes = {
        spec.block_size
        for spec in promoted_specs.values()
        if isinstance(spec, FullAttentionSpec)
    }
    full_attention_block_size = (
        next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    )

    def promoted_page_size_padded(spec: AttentionSpec, block_size: int) -> int | None:
        if spec.page_size_padded is None:
            return None
        unpadded_page_size = (
            spec.unpadded_page_size_bytes * block_size // spec.block_size
        )
        return max(spec.page_size_padded, unpadded_page_size)

    # [CN] 提升映射表：滑窗 MLA -> MLA，滑窗 -> 全注意力，
    #      分块局部 -> 全注意力。
    promotions: dict[type[AttentionSpec], type[AttentionSpec]] = {
        SlidingWindowMLASpec: MLAAttentionSpec,
        SlidingWindowSpec: FullAttentionSpec,
        ChunkedLocalAttentionSpec: FullAttentionSpec,
    }

    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            target_cls = next(
                (promotions[c] for c in type(spec).__mro__ if c in promotions), None
            )
            if target_cls is None:
                continue
            assert isinstance(spec, AttentionSpec)
            block_size = full_attention_block_size or spec.block_size
            promoted_specs[layer_name] = replace_as(
                spec,
                target_cls,
                # Promoted specs allocate blocks for all tokens and never free
                # below the window, so the trailing-edge extension is moot.
                drop=("extra_retained_tokens",),
                block_size=block_size,
                page_size_padded=promoted_page_size_padded(spec, block_size),
            )

    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")

    return promoted_specs


# [CN] 兜底路径：page 大小无法统一时，尝试"把滑窗当全注意力分配"。
#      只在 **MLA + 普通滑窗**这一特定组合下尝试，其它直接放弃。
def _try_get_full_allocation_fallback_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Try a supported full-allocation fallback for local-attention layers."""
    if any(isinstance(spec, HiddenStateCacheSpec) for spec in kv_cache_spec.values()):
        return None
    if any(
        isinstance(spec, (SlidingWindowMLASpec, ChunkedLocalAttentionSpec))
        for spec in kv_cache_spec.values()
    ):
        return None

    has_mla = any(isinstance(spec, MLAAttentionSpec) for spec in kv_cache_spec.values())
    has_regular_swa = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    if not (has_mla and has_regular_swa):
        return None

    try:
        promoted_specs = _promote_local_kv_cache_specs(kv_cache_spec)
    except ValueError:
        return None
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(promoted_specs)
    if uniform_spec is None:
        return None
    logger.warning(
        "KV cache page sizes cannot be unified; treating sliding-window "
        "layers as full attention for cache allocation. Sliding-window "
        "attention compute is unchanged."
    )
    return _get_kv_cache_groups_uniform_type(uniform_spec)


# [CN] 关闭混合 KV cache manager 时的入口：把所有局部注意力提升为
#      全注意力。会打一条 warning 告诉用户"省显存的优化没了，
#      但滑窗的计算节省还在"。
def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )
    kv_cache_spec.update(_promote_local_kv_cache_specs(kv_cache_spec))


# [CN] 挑一个"向上取整后总 padding 最少"的块大小（暴力枚举）。
#      平手时取**更大的** d（组数更少、管理开销更小）。
def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


# [CN] **混合 page 大小的打包分组**（block-outermost 布局专用）。
#      思路：先把层贪心地装进"同构桶"，再按**层模式重复次数**切组，
#      使所有组能塞进同样的每块布局。
#      Mamba 桶额外处理：限制它不要撑宽块（Mamba 状态页本来就被
#      padding 到和注意力页同宽）。
def _get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Group mixed-page-size layers for contiguous block-outermost packing.

    Greedily buckets layers into uniform-type specs. Buckets with equal layer
    counts per page size are treated as a repeating layer pattern (one layer
    per page size) and split into groups covering the same number of pattern
    repeats (picked by ``_approximate_gcd`` to minimize padding), so all
    groups pack into the same per-block layout. Mamba buckets are additionally
    split to fit the block the attention buckets already need.
    Returns None when the layout is not block-outermost or all layers already
    share one page size.
    """
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if not layout.is_block_outermost or len(page_sizes) <= 1:
        return None

    buckets: list[dict[str, KVCacheSpec]] = []
    for name, spec in kv_cache_spec.items():
        for bucket in buckets:
            candidate = {**bucket, name: spec}
            if UniformTypeKVCacheSpecs.is_uniform_type(candidate):
                bucket[name] = spec
                break
        else:
            buckets.append({name: spec})

    bucketed = []
    for bucket in buckets:
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(bucket)
        assert uniform_spec is not None
        page_size_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, layer_spec in bucket.items():
            page_size_layers[layer_spec.page_size_bytes].append(layer_name)
        # [CN] 只支持 1:1 的层模式（每种 page 大小各一层）；
        #      2:1 这类虽然理论上也能重复，但当前实现选择整桶输出。
        # Only 1:1 patterns (one layer of each page size per repeat) are
        # supported; counts sharing a gcd > 1 (e.g. 2:1) could in principle
        # repeat too, but such buckets are emitted whole instead.
        balanced = len(set(map(len, page_size_layers.values()))) == 1
        bucketed.append((uniform_spec, page_size_layers, balanced))

    # [CN] 混合 page 大小的桶必须整桶保留，所以它的"每组重复数"给其它
    #      桶定了一个**下界**；更大的同尺寸桶则被往下切。
    # Balanced buckets that mix page sizes must stay whole, so the largest one
    # sets a floor on the repeats per group; larger single-size buckets are
    # split down toward it. No such bucket means nothing needs packing.
    min_repeats_per_group = max(
        (
            spec.get_max_layers_per_page_size()
            for spec, page_size_layers, balanced in bucketed
            if balanced and len(page_size_layers) > 1
        ),
        default=0,
    )
    repeats_per_group = (
        _approximate_gcd(
            [
                spec.get_max_layers_per_page_size()
                for spec, _, balanced in bucketed
                if balanced
            ],
            lower_bound=min_repeats_per_group,
        )
        if min_repeats_per_group
        else None
    )

    def num_groups_for(spec: UniformTypeKVCacheSpecs, balanced: bool) -> int:
        if balanced and repeats_per_group is not None:
            return cdiv(spec.get_max_layers_per_page_size(), repeats_per_group)
        return 1

    def widest_group_bytes(page_size_layers: dict[int, list[str]], n: int) -> int:
        """Page bytes of the largest of the n groups a bucket splits into."""
        return sum(
            cdiv(len(names), n) * page for page, names in page_size_layers.items()
        )

    # [CN] anchor_bytes = 无论 Mamba 怎么切，一个块**至少**要装下的字节数。
    # Bytes a block must hold however the mamba buckets end up split: a mamba
    # bucket can go down to one state per group, every other bucket's split is
    # already fixed by the repeat pattern.
    anchor_bytes = max(
        (
            widest_group_bytes(
                page_size_layers,
                len(spec.kv_cache_specs)
                if isinstance(spec.first_spec, MambaSpec)
                else num_groups_for(spec, balanced),
            )
            for spec, page_size_layers, balanced in bucketed
        ),
        default=0,
    )

    groups = []
    for spec, page_size_layers, balanced in bucketed:
        num_groups = num_groups_for(spec, balanced)
            # [CN] Mamba 状态已被 padding 到一个注意力页，所以限制它的组数，
            #      让它"就着现有的块宽"放，而不是反过来把块撑大。
        # `_align_hybrid_block_size` pads a mamba state up to one attention
        # page, so cap a mamba group at the states a block already fits rather
        # than let it widen the block.
        if anchor_bytes and isinstance(spec.first_spec, MambaSpec):
            states_per_block = max(anchor_bytes // spec.first_spec.page_size_bytes, 1)
            num_groups = max(
                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
            )
        if num_groups == 1:
            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))
            continue

        pattern_repeats = list(zip(*page_size_layers.values()))
        for i in range(num_groups):
            group_layer_names = [
                name for repeat in pattern_repeats[i::num_groups] for name in repeat
            ]
            group_layer_specs = {
                name: spec.kv_cache_specs[name] for name in group_layer_names
            }
            group_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert group_spec is not None
            groups.append(KVCacheGroupSpec(group_layer_names, group_spec))

    _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        groups,
        use_deepseek_v4_fallback=_is_deepseek_v4_eagle(vllm_config),
    )
    _warn_if_unannotated_eagle_mamba(vllm_config, groups)
    return groups


# [CN] 是否是 DeepSeek-V4 + EAGLE 投机（需要走位置兜底规则）。
def _is_deepseek_v4_eagle(vllm_config: VllmConfig) -> bool:
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return False
    model_config = vllm_config.model_config
    return (
        model_config is not None and model_config.hf_config.model_type == "deepseek_v4"
    )


# [CN] 标记哪些组属于 **draft（草稿）模型**。
#      两条规则：
#        1) 看 spec 上的 non_causal_multi_token_decode 标志（可靠）；
#        2) DeepSeek-V4 的兜底：草稿层总是最后注册的那一层（hack）。
#      第 2 条只在"分组恰好按 kv_cache_spec 划分"时才成立，
#      所以由调用方按模型类型决定是否启用。
def _annotate_eagle_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
    use_deepseek_v4_fallback: bool = False,
) -> None:
    """Flag the KV cache groups that hold drafter attention layers.

    Two detection rules, in order of preference:

    1. Spec-driven. ``non_causal_multi_token_decode`` is declared on
       MLAAttentionSpec and set by drafter attention layers that run a
       non-causal multi-token decode (today only Kimi-K3 DSpark). It survives
       MLAAttentionSpec.merge, so it still identifies a group after per-group
       spec merging, wherever grouping happens to land. It is sufficient but
       not necessary: a drafter whose spec is indistinguishable from the
       target's cannot be found this way.
    2. Model-scoped positional fallback for DeepseekV4, whose MTP block reuses
       the target's own decoder layer and so carries no spec marker. Its draft
       attention layer is always the last registered layer, so flag whichever
       group holds it. This rule is only valid where the groups partition
       exactly the layers of ``kv_cache_spec``, which is true on the packed
       grouping path and not in general; other callers must leave
       ``use_deepseek_v4_fallback`` False. The caller gates this fallback on
       the configured model type.
       FIXME(yifan): avoid/generalize this hacky check.

    Args:
        vllm_config: Config supplying the speculative method, if any.
        kv_cache_spec: The kv cache spec of each attention layer, in layer
            registration order. Only read by rule 2.
        kv_cache_groups: Groups to annotate in place.
        use_deepseek_v4_fallback: Enable rule 2 for a DeepseekV4 packed group.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle_block_drop():
        return

    # [CN] 规则 1：spec 驱动。
    for group in kv_cache_groups:
        if any(
            getattr(spec, "non_causal_multi_token_decode", False)
            for spec in iter_layer_specs(group.kv_cache_spec)
        ):
            group.is_eagle_group = True

    # [CN] 规则 2：位置兜底（最后一层所在的组 = 草稿组）。
    if not use_deepseek_v4_fallback:
        return
    last_layer = next(reversed(kv_cache_spec))
    for group in kv_cache_groups:
        if last_layer in group.layer_names:
            group.is_eagle_group = True
            break


# [CN] 一个很有价值的**告警**：如果一个组都没被标记成草稿组，
#      下游会把**所有**组都当草稿组处理，这会让 Mamba 组的查找窗口
#      要求"连续两个 chunk" —— align 模式永远产生不了，
#      于是前缀复用**静默地**降到 0。
#      这种"既不报错也没有指标"的性能悬崖最难查，所以专门加个 warning。
def _warn_if_unannotated_eagle_mamba(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """Warn when the flag-all eagle fallback will silently disable reuse.

    With no group annotated, consumers flag every group as a draft group. That
    widens a Mamba group's required lookup window to two consecutive chunks,
    which align-mode checkpointing never produces, so reuse drops to zero with
    no error and no metric to show it.

    Args:
        vllm_config: Config supplying the speculative method, if any.
        kv_cache_groups: Groups as they will be handed to consumers.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    if any(group.is_eagle_group for group in kv_cache_groups):
        return
    mamba_groups = [
        idx
        for idx, group in enumerate(kv_cache_groups)
        if any(
            isinstance(spec, MambaSpec)
            for spec in iter_layer_specs(group.kv_cache_spec)
        )
    ]
    if not mamba_groups:
        return
    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's, so every group -- "
        "including Mamba groups %s -- will be treated as a draft group. A "
        "Mamba group cannot satisfy the widened lookup window that implies, "
        "so prefix-cache reuse across requests will be disabled and any "
        "external KV offload tier will store without ever serving a hit.",
        spec_config.method,
        mamba_groups,
    )


# [CN] 求 <= limit 的最大因数（用于给隐藏态层挑合适的 block_size）。
def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


# [CN] **分组主入口**。按优先级依次尝试：
#        1) 关闭混合管理器 -> 先做 spec 统一化；
#        2) 无注意力模型 -> 空列表；
#        3) 全部同构 -> 一个组；
#        4) 同构类型（UniformType）-> 一个聚合组；
#        5) GLM5 专用布局；
#        6) 打包分组（block-outermost）；
#        7) 统一 page 大小后走通用混合分组。
#      隐藏状态层（HiddenStateCacheSpec）全程**单独成组**，不参与合并。
def get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    # [CN] 用户显式关闭混合管理器时，先统一 spec（会就地修改入参）。
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)
    elif glm5_groups := _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec):
        return glm5_groups

    # [CN] 隐藏状态层要用**自己的 block table**，必须先摘出去，
    #      否则会被同构分桶吞掉。
    # Hidden-state layers use their own block table and must not be absorbed
    # into a compatible attention bucket.
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    filtered_spec = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, HiddenStateCacheSpec)
    }

    if packed_groups := _get_packed_kv_cache_groups(vllm_config, filtered_spec):
        # Block-outermost blocks are strided by the widest group, so hidden
        # groups need no page alignment.
        packed_groups += [
            KVCacheGroupSpec([name], spec) for name, spec in hidden_specs.items()
        ]
        return packed_groups

    # [CN] 优先保留每层原本的缓存语义；只有 page 实在统一不了，
    #      才退到"按全注意力分配"的兜底方案。
    # Prefer preserving each layer's cache semantics. If physical pages cannot
    # be unified, try a supported allocation-only fallback before failing.
    try:
        filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    except NotImplementedError:
        fallback_groups = _try_get_full_allocation_fallback_groups(kv_cache_spec)
        if fallback_groups is None:
            raise
        return fallback_groups
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # [CN] 把隐藏态层加回来，并把它的 page 对齐到公共 page 大小
    #      （挑一个不超公共页的最大 block_size，浪费的字节打日志告知）。
    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        group_block_size = math.gcd(*(g.kv_cache_spec.block_size for g in groups))
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            max_block_size = max(common_page // per_token, 1)
            new_bs = _largest_divisor_at_most(group_block_size, max_block_size)
            wasted_bytes = common_page - new_bs * per_token
            logger.info(
                "Using block size %d for hidden-state cache layer %s; "
                "page alignment wastes %d bytes (%.2f%%) per block",
                new_bs,
                name,
                wasted_bytes,
                wasted_bytes / common_page * 100,
            )
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))

    _annotate_eagle_groups(vllm_config, kv_cache_spec, groups)
    _warn_if_unannotated_eagle_mamba(vllm_config, groups)
    return groups


# [CN] 生成**调度器侧**的配置：各 worker 的配置除了层名都一样，
#      所以随便取一份深拷贝，再把聚合 spec 简化成"代表性单层 spec"。
#      （调度器只关心容量，不关心具体是哪些层。）
def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


# [CN] KV cache 总容量（token 数）与最大并发。
def get_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, float]:
    """
    Get the group-aware KV cache token capacity and max concurrency.
    """
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    return int(max_concurrency * max_model_len), max_concurrency


# [CN] 把容量写回 cache_config 并打日志（那句 "GPU KV cache size: N tokens"
#      就是这里输出的，是排查显存问题时最常见的日志之一）。
def update_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Store and log the resolved KV cache capacity."""
    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
    vllm_config.cache_config.kv_cache_size_tokens = num_tokens
    vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency
    max_model_len = vllm_config.model_config.max_model_len
    logger.info_once(
        "GPU KV cache size: %s tokens, "
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{num_tokens:,}",
        f"{max_model_len:,}",
        max_concurrency,
    )


# [CN] 从分组算最大内存占用。注意它**把 padding 也算进去**了：
#      混合模型补齐层数后，显存是按补齐后的数字算的。
#      另外每个组独立从共享池取块，所以总开销是各组之和。
def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.

    Each group independently claims blocks from the shared pool, so a request consumes
    the sum of the per-group block counts, i.e. ``bytes_per_block * total_blocks``.
    """
    if not kv_cache_groups:
        return 0

    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
        total_blocks = uniform_spec.max_memory_usage_pages(vllm_config)
        total_blocks += sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in mamba_groups
        )
        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)

    bytes_per_block = _pool_bytes_per_block(kv_cache_groups)
    total_blocks = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            total_blocks += spec.max_memory_usage_pages(vllm_config)
        else:
            total_blocks += cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )

    return bytes_per_block * total_blocks


# [CN] 二分查找"给定显存最多能跑多长"（组版本，比 spec 版本更准）。
def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


# [CN] max_model_len = -1 时的**自动适配**：二分找所有 worker 都能
#      支持的最大长度，取最小值（木桶效应），并打日志说明被谁限制。
def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return

    # Find the max_model_len that fits across all workers.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
        )


# [CN] 把全局分组**投影**到某个 worker 实际拥有的层上（PP 场景）。
#      聚合 spec 要按该 worker 的层名重建，空组也要保留（占位）。
def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


# [CN] **生成所有 worker 的 KV cache 配置** —— 整个模块的顶层入口。
#      docstring 里的五步流程就是全部要点：
#        1) 合并各 worker 的 spec（PP 各 stage 层不同，要并起来）；
#        2) 按整模型的层比例分组（顺带处理混合模型的 spec 统一）；
#        3) 用"投影到各 worker 的分组"做自动适配与显存校验；
#        4) 为每个 worker 生成配置；
#        5) 把所有 rank 的 num_blocks **拉齐到最小的那个** ——
#           因为调度器是中心化的，块数必须各 rank 一致，
#           否则某个 rank 会分配到别的 rank 没有的块 id。
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # [CN] 合并时要求**同名层的 spec 必须一致**，否则直接 assert 失败。
    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Check if the KV cache specs are registered correctly.
    # This is to prevent that some layers are initialized with unregistered specs.
    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)

    # [CN] 多层 MTP 投机时，给所有滑窗 spec 打上"额外保留 token 数" ——
    #      因为草稿模型可能会回过头重算末尾若干 token。
    # When speculating with more than 1 speculative module (e.g. multi-layered MTP)
    # tag every SlidingWindowSpec with how many extra tokens to retain in the window.
    extra_retained_tokens = (
        vllm_config.speculative_config.num_speculative_tokens - 1
        if vllm_config.speculative_config is not None
        and vllm_config.speculative_config.use_multi_module_mtp()
        else 0
    )
    for layer_name, layer_spec in merged_kv_cache_specs.items():
        if isinstance(layer_spec, SlidingWindowSpec):
            merged_kv_cache_specs[layer_name] = replace(
                layer_spec, extra_retained_tokens=extra_retained_tokens
            )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    # [CN] num_gpu_blocks_override 会让"实际分配的块数"与"实测显存"脱钩，
    #      所以这里同步调整 available_memory，让自动适配、准入检查、
    #      配置生成三者**按同一个容量**规划（否则会互相矛盾）。
    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block(groups)
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    # [CN] 预留一个块给常驻的 null block，让自动适配与容量检查都按
    #      "真正可用"的块数来规划。
    # Reserve the null block BlockPool permanently holds back, so auto-fit and
    # the capacity check both plan against usable blocks. Allocation below
    # still uses the full memory.
    check_memory = [
        avail_mem - _pool_bytes_per_block(groups) if groups else avail_mem
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(vllm_config, projected_groups_per_worker, check_memory)

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, check_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # [CN] 拉齐 num_blocks：用最小块的显存量**重新规划**一遍，
    #      而不是简单改个数字 —— 这样 stride 和 offset 才保持一致。
    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for i, kv_cache_config in enumerate(kv_cache_configs):
        if kv_cache_config.num_blocks == min_num_blocks:
            continue
        # Re-plan with exactly the memory the smallest rank can afford, so
        # strides and offsets stay consistent with the shrunken allocation.
        groups = kv_cache_config.kv_cache_groups
        kv_cache_configs[i] = get_kv_cache_config_from_groups(
            vllm_config, groups, min_num_blocks * _pool_bytes_per_block(groups)
        )

    return kv_cache_configs


# [CN] **块哈希粒度的适配器**：把按 hash_block_size 算出来的哈希，
#      "看成"按 target_block_size 粒度的哈希。
#      为什么能这么做：每个 hash_block_size 的哈希是**链式**的，
#      已经包含了它之前的全部内容 —— 所以目标块内**最后一个**
#      细粒度哈希，天然就是整个目标块的哈希。
#      docstring 里那张对照表把这个关系画得很清楚（16->32 取 B、D）。
#      好处：不同 group 用不同块大小时，哈希**只算一次**就能共用。
class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency. Each `hash_block_size` hash is
    already chained over its entire prefix, so the hash at the last
    `hash_block_size` boundary of a `target_block_size` block uniquely
    fingerprints that block's prefix; we use it directly.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    the second 16-size hash already covers tokens 0-31, so it is the 32-size
    hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | B    | D     |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """

    # [CN] 只支持**整数倍放大**（target 是 hash 的整数倍）。
    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    # [CN] 就是上面说的：取目标块内最后一个细粒度哈希。
    def _get_value_at(self, idx: int) -> BlockHash:
        # The last hash_block_size hash within the target block already chains
        # over the whole prefix, so it is the target block's hash.
        return self.block_hashes[(idx + 1) * self.scale_factor - 1]


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize


# [CN] 按目标块大小解析出合适的哈希视图。
#      三种情形：粒度相同直接用；已经是视图就复用；
#      支持细粒度查找时保留原始细哈希（用于块内部分命中）。
def resolve_block_hashes(
    block_hashes: BlockHashList,
    hash_block_size: int,
    block_size: int,
    *,
    supports_fine_grained_hash_lookup: bool = False,
    alignment_tokens: int | None = None,
) -> BlockHashList:
    """Resolve the block-hash view at ``block_size``.

    When ``block_size`` equals ``hash_block_size``, reuse the precomputed block
    hashes directly; otherwise view them at ``block_size`` granularity.
    Fine-grained lookup keeps the original hashes for partial cache hits.
    """
    if block_size == hash_block_size:
        return block_hashes
    if isinstance(block_hashes, BlockHashListWithBlockSize):
        # Already a block-size view
        assert block_hashes.scale_factor == block_size // hash_block_size
        return block_hashes
    # Fine-grained partial hits keep the raw hashes. The caller passes
    # alignment_tokens = hash_block_size to enable them, else >= block_size.
    if (
        supports_fine_grained_hash_lookup
        and alignment_tokens is not None
        and alignment_tokens < block_size
        and block_size % alignment_tokens == 0
    ):
        return block_hashes
    assert block_size % hash_block_size == 0
    return BlockHashListWithBlockSize(block_hashes, hash_block_size, block_size)
