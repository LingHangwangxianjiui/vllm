# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [CN] 文件总览：单类型 KV cache 管理器（SingleTypeKVCacheManager 家族）
#
# 它在栈里的位置：BlockPool（块） <- 本文件（一个 group） <- Coordinator（多 group）
#
# 一句话职责：**管“某一个 KV cache group”里，请求占了哪些块、能命中多长前缀**。
# 之所以叫 single type，是因为同一个 group 里所有层的 KV 结构完全相同
# （块大小、层数和注意力类型一致），可以统一用一套逻辑管理。
#
# 子类一览（每一个对应一种注意力语义）：
#   FullAttentionManager          全注意力：块全留、命中最长连续前缀
#     └ RSWAManager               参考滑动窗口：额外回收“中间空隙块”
#     └ SinkFullAttentionManager  带 sink token（永久保留开头几块）的全注意力
#     └ CircularBufferManager     环形缓冲（Mamba 之外的一类循环复用，1 块/请求）
#         └ KpoolTailManager      Kpool 尾巴：同样是 1 块临时缓冲
#   SlidingWindowManager          滑动窗口：只看最近 W 个 token，块可边跑边回收
#   ChunkedLocalAttentionManager  分块局部注意力（类似 SWA，但按 chunk 对齐）
#   MambaManager                  Mamba / 线性注意力：状态是**递推**的，不是追加的
#   CrossAttentionManager         编码器-解码器的 cross-attention（不共享、不缓存）
#
# 贯穿全文件的两个抽象概念（理解子类的钥匙）：
#   1) get_num_skipped_tokens(n)：已经算到第 n 个 token 时，
#      有哪些 token 的 KV **永远不会被再用到**，可以释放/用 null 占位。
#      全注意力恒为 0（都要用），SWA 返回滑出窗口的那段，Mamba 返回 n-1。
#   2) find_longest_cache_hit()：按块哈希查前缀缓存的最长命中。
#      各子类语义差别很大 —— 是本文件最需要逐个对照读的部分。

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    resolve_block_hashes,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KpoolTailSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request

logger = init_logger(__name__)


# [CN] 抽象基类：只关心“一个 group”的 KV cache 管理。
#
# 子类只需要按需重写几个钩子：
#   - find_longest_cache_hit    （必须）怎么查前缀命中
#   - get_num_skipped_tokens    （可选）哪些 token 的 KV 不再需要（默认 0）
#   - reachable_block_mask      （可选）稀疏保留时哪些块值得缓存（默认全缓存）
#   - get_num_common_prefix_blocks（必须）公共前缀块数（多数直接返回 0）
#
# supports_fine_grained_hash_lookup：
#   是否支持**细粒度**哈希查找（命中长度可以不是整块，而是 hash_block_size 的倍数）。
#   只有 full attention 与 mamba 支持；SWA / chunked-local 只能整块命中。
class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    # [CN] 构造。几个关键字段：
    #   block_size          : 本 manager 真正按多大的块分配
    #                         （注意 DCP 会把 block_size 放大 dcp_world_size 倍，
    #                           因为一个块的 KV 被切到多个 rank 上）
    #   scheduler_block_size: 全局调度粒度（所有 group 块大小的公倍数）
    #   cache_hit_alignment_tokens: 命中长度要对齐到多少 token，
    #                         默认等于 scheduler_block_size，
    #                         后面 Coordinator 可能调细（细粒度命中场景）
    #   _record_new_block_ids: 是否要记录“本步新分配的块 id” —— 
    #                         worker 侧需要把这些块**清零**再写入，
    #                         否则脏数据会被当成有效 KV 读出来
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            scheduler_block_size: The scheduling granularity (LCM of all group
                block sizes); a multiple of this manager's ``block_size``.
            needs_kv_cache_zeroing: Whether worker-side KV cache zeroing needs
                newly allocated block IDs from this manager.
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every
                block until the request finishes.
        """
        self.scheduler_block_size = scheduler_block_size
        # Hybrid fine-grained lookup may lower this after all participating
        # managers have been validated by the coordinator.
        self.cache_hit_alignment_tokens = scheduler_block_size
        # The block size for this manager; used for actual block allocation.
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        self._record_new_block_ids = (
            needs_kv_cache_zeroing
            and isinstance(kv_cache_spec, AttentionSpec)
            and not isinstance(kv_cache_spec, CircularBufferSpec)
        )
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

        # Whether this group's prefix-cache hits drop the EAGLE/MTP lookahead
        # block. Only consulted by managers whose hit logic is sparse within an
        # aligned segment (SWA). Initialized lazily by the coordinator after
        # determining the attention groups.
        self.use_eagle = False

        # Partial-hit copy-on-write bookkeeping. Populated only by fine-grained
        # managers (full attention, mamba "align"); harmlessly empty elsewhere.
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        # Boundary-state offload hand-off for external KV connectors. A mamba
        # "align" block table is not append-only (interior states are
        # nulled/freed and speculative blocks relocate in place), so a
        # connector cannot resolve its state blocks positionally. Record
        # (request, group, block, exact token boundary) for each committed
        # boundary state so a connector can offload the right block under the
        # right hash. Populated only by mamba "align".
        self._pending_boundary_state_offloads: list[
            tuple[str, int, KVCacheBlock, int]
        ] = []

    @classmethod
    # [CN] 统计这批块里有几个是“可被驱逐的”（ref_cnt == 0 且不是 null 块）。
    #      用途：命中到一个正待淘汰的块时，它马上会被本请求 touch 而救活，
    #      所以在算“还需要多少空闲块”时必须把它算进可用容量里。
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    # [CN] 是否发生了**部分命中**：本地命中的 token 数不是块大小的整数倍，
    #      说明最后一个块是“共享的”（别人也在用、且内容比我们多）。
    #      这种情况必须做 **CoW（写时复制）**：把共享块复制一份私有的，
    #      否则我们往里写 token 会污染别人的前缀缓存。
    def _has_partial_local_hit(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> bool:
        # The local prefix-cache hit ends inside one of this manager's
        # blocks: the shared tail block needs CoW.
        return (
            len(new_computed_blocks) > 0
            and num_local_computed_tokens % self.block_size != 0
        )

    # [CN] 预测“还需要分配几个块”——**不改状态**，只给调度器做准入判断。
    #
    # 主体公式：需要 = ceil(num_tokens / block_size)
    #            - max(可跳过的块数, 已持有块数 + 新命中块数)
    # 其中“可跳过的块”是指滑出注意力窗口、可以直接丢弃的那些块（SWA 场景）。
    #
    # 两个细节：
    #   1) 已在跑的请求（在 num_cached_block 里）不会有新命中，走快路径；
    #   2) 命中块里 ref_cnt == 0 的那些（待淘汰）要额外计入，
    #      因为它们虽然现在“占着位”，但马上会被本请求复用。
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, clamp by `num_required_blocks` by
                `_max_admission_blocks_per_request`for recycling-aware specs
                (SWA, chunked-local).

        Returns:
            The number of blocks to allocate.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Reserve the extra block that allocate_new_blocks pulls for the
            # partial-hit CoW redirect.
            num_new_blocks += 1
        return num_new_blocks + num_evictable_blocks

    # [CN] 把本地前缀命中的块挂到请求名下（Coordinator 两阶段分配的第一阶段）。
    #      步骤：跳过被滑出窗口的块 -> touch（抬高引用计数，防止被淘汰）
    #            -> 用 null 块补齐“被跳过的位置”（保持块表下标对齐）
    #            -> 记录 num_cached_block（这些块已经是缓存内容，不必再缓存）。
    #
    #      末尾的部分命中要额外记账：把“共享尾块”记进 _partial_hit_reqs，
    #      并把 num_cached_block 回退到整块边界，
    #      好让后续 cache_blocks 在复制完成后重新缓存那一块。
    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the locally cached (prefix-hit) blocks to the request:
        1. Touch the computed blocks (paired with adding them to `req_blocks`)
           so their ref_cnt exactly tracks the referencing requests.
        1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
        2. Add the remaining computed blocks.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Record the partial tail for the CoW redirect in
            # allocate_new_blocks; cap the cached count at the full blocks so
            # cache_blocks() re-caches the private copy once full.
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
            self.num_cached_block[request_id] = block_idx

    # [CN] 为“外部已算 token”（external，例如 KV connector 从别的实例搬来的 KV）
    #      分配新块。必须在**所有 group 的本地命中块都 touch 完之后**才调用，
    #      否则这里的 get_new_blocks 可能淘汰别的 group 刚命中的块（#33775）。
    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Allocate new blocks for external (KV-connector) computed tokens.

        Must run only after every group's local blocks have been touched via
        `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
        evict another group's cache-hit blocks (issue #33775).

        Args:
            request_id: The request ID.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = max(
            0, cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    # [CN] 真正的分配。先处理挂起的部分命中（CoW 重定向），再补足到 num_tokens。
    #      返回值是“本次新拿到的块”，上层拿去更新请求的块表。
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        cow_blocks: list[KVCacheBlock] = []
        if request_id in self._partial_hit_reqs:
            # Partial hit: redirect the shared tail to a private CoW block.
            # Replacing in place keeps the length-based allocation below
            # correct; the extra block was reserved by
            # get_num_blocks_to_allocate.
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return cow_blocks
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return cow_blocks + new_blocks

    @property
    def records_new_block_ids(self) -> bool:
        """Whether this manager's new blocks are zeroed by the worker."""
        return self._record_new_block_ids

    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def take_pending_cow_copies(
        self,
    ) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Drain pending CoW source and destination block pairs."""
        pending_copies = self._pending_cow_copies
        self._pending_cow_copies = []
        return pending_copies

    def take_pending_boundary_state_offloads(
        self,
    ) -> list[tuple[str, int, KVCacheBlock, int]]:
        """Drain producer boundary-state hand-offs.

        Entries are ``(req_id, group_id, block, boundary_tokens)``.

        Only mamba "align" populates this. The blocks are not kept alive by
        the request block table for the whole request, so a caller that reads
        them asynchronously must pin them first.
        """
        pending = self._pending_boundary_state_offloads
        self._pending_boundary_state_offloads = []
        return pending

    def finalize_partial_tail_offload(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_in_flight_tokens: int,
    ) -> tuple[int, KVCacheBlock, int] | None:
        """Finalize a producer partial tail when its request finishes."""
        return None

    # [CN] CoW 重定向：把请求块表里第 block_idx 个位置从共享的 source_block
    #      换成私有的 cow_block，并登记一对待复制 (source, cow) 交给 worker 执行。
    #
    #      两端都要保持引用：source_block 保留它原本的命中引用，
    #      cow_block 额外 +1 —— 这样即使同一 step 内有释放操作，
    #      也不会在拷贝完成前把任一端回收掉。
    def _apply_cow(
        self,
        request_id: str,
        block_idx: int,
        source_block: KVCacheBlock,
        cow_block: KVCacheBlock,
    ) -> None:
        """Redirect a partial prefix-cache hit to a private CoW block.

        Both copy endpoints stay retained until the copy has run on the worker,
        so a same-step free cannot recycle them: ``source_block`` keeps its
        hit-ref, ``cow_block`` takes an extra ref beyond the one handed to the
        request.
        """
        req_blocks = self.req_to_blocks[request_id]
        assert block_idx < len(req_blocks)
        assert req_blocks[block_idx] is source_block
        assert not source_block.is_null and source_block.ref_cnt > 0
        req_blocks[block_idx] = cow_block
        self._pending_cow_copies.append((source_block, cow_block))
        cow_block.ref_cnt += 1

    # [CN] 把已算好的 KV 注册进前缀缓存（写块哈希），供后续请求命中。
    #      只处理 [num_cached_block, num_full_blocks) 这段“新变成整块”的区间。
    #
    #      block_mask 是稀疏保留的关键：None 表示“全部缓存”，
    #      子类（SWA / Mamba）会返回一个布尔掩码，只缓存“将来可能被命中的块”，
    #      从而大幅降低缓存占用。
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
            retention_interval: Sparse local-checkpoint granularity. ``None``
                keeps dense checkpointing; ``0`` keeps only the latest replay
                boundary; a positive multiple of ``scheduler_block_size`` keeps
                a tail once per that-sized segment. Only SWA acts on it.
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        # Token boundaries whose reachable tail must be retained under sparse
        # retention: the replay boundary (``num_prompt - 1``, capped by
        # ``get_computed_blocks``) and any detected shared-prefix junction.
        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)

        block_mask = self.reachable_block_mask(
            start_block=num_cached_blocks,
            end_block=num_full_blocks,
            alignment_tokens=self.cache_hit_alignment_tokens,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
            dcp_world_size=self.dcp_world_size,
        )
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
            block_mask=block_mask,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    # [CN] 稀疏保留掩码：在 [start_block, end_block) 里，哪些块值得写哈希。
    #      返回 None = 全部缓存（全注意力的默认行为）。
    #      子类按自己的“命中语义”重写：
    #        SWA   —— 一次命中需要连续 need 个块，所以只缓存每个边界前的 need 块；
    #        Mamba —— 一次命中只需要 1 个状态块，所以每个边界只留 1 块。
    #      reachable_boundaries 是“必须保留”的边界：
    #        重放边界（num_prompt - 1）与跨请求的公共前缀接点。
    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
        final_segment_end_block: int | None = None,
    ) -> list[bool] | None:
        """Per-block mask for ``cache_full_blocks``. ``None`` means cache
        every (non-null) block — the default for full attention.

        Subclasses with sparse hit semantics (SWA / Mamba) override this to skip
        blocks that can never serve a hit at any alignment-aligned prefix length.
        ``reachable_boundaries`` are token positions whose reachable tail must be
        retained; the base (dense) policy ignores them.
        ``final_segment_end_block`` is the exclusive end of the request's final
        segment. It may be later than ``end_block`` while prefill is still in
        progress. Non-EAGLE sparse managers may keep its reachable tail. The
        base policy already keeps every block, so it ignores this value.
        """
        return None

    # [CN] 摘掉请求的全部记账（块表、缓存计数、部分命中记录）并返回它的块，
    #      **但不归还给块池** —— 由调用方决定归还时机（异步/批量释放）。
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        Pop the request's bookkeeping and return its blocks without yet
        returning them to the block pool. The caller is responsible for
        eventually passing the returned blocks to `block_pool.free_blocks`,
        freeing them in reverse order (so that tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        self._partial_hit_reqs.pop(request_id, None)
        return req_blocks

    # [CN] 释放：逆序归还（尾部先还），保证剩余块始终是一条连续前缀。
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Free blocks in reverse order so that the tail blocks are freed first.
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache.

        Args:
            running_request_id: The request ID.

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.
        """

        raise NotImplementedError

    # [CN] **本家族最核心的抽象方法**：按块哈希查最长前缀命中。
    #      各子类的实现差异很大，是理解每种注意力“复用语义”的入口。
    #
    #      返回值里的块列表用 **null 块占位**表示“这块被跳过了/窗口外”，
    #      这样块表长度仍然是 token 数 / block_size，下标不会错位。
    #
    #      drop_eagle_block：EAGLE/MTP 要丢掉最后匹配到的块，
    #      因为草稿头需要那一个 token 的隐藏状态，必须重算。
    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.
        If eagle is enabled, drop the last matched block to force recompute the
        last block to get the required hidden states for eagle drafting head.
        For multi-module MTP, this recompute also rewrites the dropped block's
        draft-layer KVs, which depend on up to num_speculative_tokens - 1
        tokens past the matched prefix (i.e. on the cache writer's
        continuation, which the block hash does not cover); the coordinator
        asserts the block size covers that window.
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
                Always False for non-EAGLE/MTP groups, but can be False for EAGLE/MTP
                groups too if the last block is already dropped (e.g., in a
                convergence loop in `find_longest_cache_hit`).
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens). By default, it should
                be set to the block_size.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.

        Returns:
            A tuple containing cached blocks and the exact cache-hit length in
            tokens. The cached block tuple has skipped blocks replaced by null
            blocks for each kv cache group in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """

        raise NotImplementedError

    # [CN] 释放 [first_block, last_block) 区间内的块并换成 null。
    #      **从后往前遍历**：前面的块可能在之前的调用里已被置 null，
    #      倒着走才能把新变得可驱逐的尾部块也一并处理到。
    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        """Free blocks in ``[first_block, last_block)`` and replace with null_block.

        Iterates backward so newly-evictable tail blocks are reached even after
        earlier blocks in the range were nulled in a prior call.
        """
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)

    # [CN] 回收注意力窗口外的块（SWA / chunked-local / Mamba 的核心回收路径）。
    #      “跳过多少 token”由子类 get_num_skipped_tokens 决定。
    #      注意对 num_skipped_blocks 做了上限裁剪：
    #      滑出的 token 可能还没分配块（例如窗口滑进了 external 区），
    #      不能越界。
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for attention types (e.g.
                R-SWA) that evict a middle gap rather than a head prefix. Ignored
                by the default implementation.
        """
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)

    # [CN] 已经算到第 n 个 token 时，前多少个 token 的 KV 再也用不到。
    #      基类（全注意力）返回 0 —— 所有 token 都要参与注意力，一个都不能扔。
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        return 0

    def new_step_starts(self) -> None:
        return None


# [CN] 全注意力管理器。两个特点：
#   1) 所有块的 KV 都要保留到请求结束（get_num_skipped_tokens 恒为 0）；
#   2) 命中是**从前往后**扫：块哈希是链式（每个块的哈希包含前缀），
#      所以一旦某块没命中，后面必然也不命中 —— 可以直接 break。
#      这也是它是唯一支持“细粒度哈希查找”的原因之一。
class FullAttentionManager(SingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    # [CN] 两阶段查找：
    #   阶段 1：从头连续匹配整块，遇到第一个 miss 就停（链式哈希保证后面全 miss）；
    #   阶段 2（仅细粒度模式）：在第一个未命中的整块内部，
    #          从高到低试探各个 hash 边界，取最长可命中的那一个。
    #  最后按 alignment_tokens 向下取整，并把块表截断到新长度。
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size > 1:
            # DCP shards each block's KV across ranks; hashes must be viewed at
            # the sharded block size.
            block_size *= dcp_world_size
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        # Fine-grained mode (alignment_tokens == hash_block_size <
        # block_size): resolve_block_hashes kept the raw hash-granularity
        # list so interior boundaries can be probed.
        fine_grained = (
            alignment_tokens < block_size and block_size % alignment_tokens == 0
        )
        if fine_grained:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            full_block_hashes: BlockHashList = BlockHashListWithBlockSize(
                block_hashes, alignment_tokens, block_size
            )
        else:
            full_block_hashes = block_hashes

        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        # Phase 1: longest run of cached full blocks from the start. A missing
        # block implies every later block misses too (chained hashes).
        # [CN] 阶段 1：连续整块匹配。max_length // block_size 限制最多看几块。
        for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
            cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached_block:
                break
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        hit_length = len(computed_blocks[0]) * block_size

        # Phase 2 (fine-grained only): extend into the first non-full block by
        # probing its interior hash boundaries high-to-low (longest hit first).
        if fine_grained:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            scale_factor = block_size // alignment_tokens
            first_partial_idx = len(computed_blocks[0]) * scale_factor
            max_partial_idx = min(
                first_partial_idx + scale_factor - 1,
                max_length // alignment_tokens,
                len(block_hashes),
            )
            for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
                cached_tail = block_pool.get_cached_block(
                    block_hashes[fine_idx], kv_cache_group_ids
                )
                if not cached_tail:
                    continue
                for computed, cached in zip(computed_blocks, cached_tail):
                    computed.append(cached)
                hit_length = (fine_idx + 1) * alignment_tokens
                break

        # Eagle needs the tokens right before the generation point recomputed:
        # drop one hash unit when fine-grained (the tail block's KV is
        # append-only, so it still covers the reduced length), else one cache
        # block.
        # [CN] EAGLE：丢掉一个单位（细粒度时是 1 个 hash 单位，否则 1 个整块），
        #      让最后那点 token 重算，草稿头才能拿到隐藏状态。
        if drop_eagle_block and hit_length > 0:
            hit_length -= min(alignment_tokens, block_size)
        # Round down to the alignment; a no-op when fine-grained (hits land on
        # hash boundaries by construction) and when alignment_tokens ==
        # block_size. Then trim blocks past the new tail.
        hit_length -= hit_length % alignment_tokens
        num_blocks = cdiv(hit_length, block_size)
        for computed in computed_blocks:
            del computed[num_blocks:]
        return computed_blocks, hit_length

    # [CN] 先走基类的整块缓存；若本 group 的块比 hash 块大，
    #      再单独把“prompt 尾部落在块中间”的那一段注册为部分哈希条目。
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
        hash_block_size = self.block_pool.hash_block_size
        if self.block_size == hash_block_size:
            return
        self._cache_partial_tail_block(request, num_tokens)

    # [CN] 缓存“部分尾块”：prompt 长度不是块大小的整数倍时，
    #      最后一个 hash 边界落在某个块**内部**。
    #      只注册**最后一个**边界 —— 同一块内的中间边界不注册，
    #      因为它们共享同一份物理 KV，注册多个会造成“命中到错误内容”。
    def _cache_partial_tail_block(
        self,
        request: Request,
        num_tokens: int,
    ) -> None:
        """Cache the prompt tail when it ends inside a cache block.

        Only the final prompt hash boundary is registered as a partial
        prefix-cache entry; intermediate hash boundaries inside the same cache
        block are intentionally skipped.
        """
        hash_block_size = self.block_pool.hash_block_size
        boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size
        if boundary_tokens == 0 or boundary_tokens > num_tokens:
            return
        if boundary_tokens % self.block_size == 0:
            return

        blocks = self.req_to_blocks[request.request_id]
        block_idx = boundary_tokens // self.block_size
        if block_idx >= len(blocks):
            return
        self.block_pool.cache_partial_block(
            request=request,
            block=blocks[block_idx],
            num_tokens=boundary_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )

    # [CN] 公共前缀块数：从头数，直到某个块的引用计数 != 在跑请求数为止
    #      （ref_cnt == 所有请求都在用 => 它是公共前缀的一部分）。
    #      cascade attention 用它决定“公共前缀”那段可以复用一次计算。
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


# [CN] R-SWA（Reference Sliding Window Attention，参考滑动窗口）。
#      与普通 SWA 的区别：普通 SWA 从头滑，R-SWA 保留开头的 prefix 段，
#      只回收“prefix 尾部”与“当前窗口”之间的**中间空隙块**。
#      效果：单请求 KV 占用从 O(解码长度) 降为 O(prefix + 窗口)，
#      长输出场景下省得非常可观。
class RSWAManager(FullAttentionManager):
    """KV cache manager for Reference Sliding Window Attention (R-SWA).

    When ``num_prompt_tokens`` is supplied to ``remove_skipped_blocks``, frees
    gap blocks between the prefill tail and the current decode window.  This
    bounds per-request KV memory at O(prefix_len + rswa_window) instead of
    growing linearly with decode length.
    """

    def __init__(self, kv_cache_spec: RSWASpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.rswa_window: int = kv_cache_spec.rswa_window

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Free gap blocks that are no longer needed for attention.

        Gap = blocks entirely within
            [ceil(prefix_len / block_size) * block_size,
             max(prefix_len, processed_computed_tokens - rswa_window))

        Freed blocks are replaced with null_block in req_to_blocks so the
        block_table passed to FA4 is valid (null_block KV is all-zero;
        rswa_mask_mod marks gap positions as non-visible so FA4 skips them).
        """
        if num_prompt_tokens is None:
            super().remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )
            return

        bs = self.block_size
        # First block fully after the prefill boundary.
        first_gap_block = cdiv(num_prompt_tokens, bs)
        # Decode window start position; blocks before this are evictable.
        window_start = max(
            num_prompt_tokens, processed_computed_tokens - self.rswa_window
        )
        last_gap_block = window_start // bs  # exclusive upper bound
        self._remove_blocks_in_range(request_id, first_gap_block, last_gap_block)


# [CN] 滑动窗口管理器。与全注意力的根本区别：
#   请求只需要最近 W 个 token 的 KV，更早的块**可以边跑边扔**。
#
# 这带来两个连锁反应：
#   1) 命中判定：不能像全注意力那样“从头连续匹配”，
#      而是需要**连续 C 个块都命中**才算命中（C = 覆盖窗口所需的块数），
#      因为只拿最后 1 块是凑不出一个完整窗口的；
#   2) 查找方向：从右往左扫，找到即停 —— 因为窗口只需要尾部那段，
#      越靠右的命中价值越高。
class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        # Extra trailing tokens to retain below the window (never attended) so a
        # multi-module MTP store-side lag can still reconstruct the window from
        # cached blocks.
        self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens

    # [CN] 一次命中需要多少个**连续**块：ceil((窗口 - 1) / 块大小)。
    #      EAGLE 时 +1（先多匹配一块再丢掉，理由同前）。
    @classmethod
    def _contiguous_blocks_for_hit(
        cls, window_size: int, block_size: int, use_eagle: bool
    ) -> int:
        blocks = cdiv(window_size - 1, block_size)
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            blocks += 1
        return blocks

    # [CN] 从右往左扫，维护“当前连续命中了几块”：
    #      命中就 +1，miss 就归零；累计到 C 块即成功，截断尾部并退出。
    #      若整轮都没凑够 C 块，也保留已连续命中的前缀部分（聊胜于无）。
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."
        # Sliding-window cache hits must stay at the group's physical block
        # granularity. resolve_block_hashes() converts finer-grained hashes to
        # that view when the hybrid-cache alignment is smaller than block_size.
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        # The number of contiguous blocks needed for a prefix cache hit.
        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block
        )

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                    post_pop_blocks = i if drop_eagle_block else i + 1
                    if (post_pop_blocks * block_size) % alignment_tokens != 0:
                        continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if drop_eagle_block and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
            # Re-align after eagle pop: the pop may break the alignment
            # when block_size != alignment_tokens (hybrid models with
            # different page sizes, e.g. Gemma4).
            while (
                block_size != alignment_tokens
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        hit_length = len(computed_blocks[0]) * block_size
        return computed_blocks, hit_length

    # [CN] SWA 的稀疏保留掩码：
    #      一次命中需要连续 need 块，所以每个对齐边界前 need 个块才值得缓存；
    #      其余块将来也不可能凑出命中，缓存它们纯属浪费。
    #      另外一定会保留 reachable_boundaries（重放边界 / 公共前缀接点）前的块，
    #      否则稀疏保留会把“确定能复用”的那次机会也一起省掉。
    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
        final_segment_end_block: int | None = None,
    ) -> list[bool] | None:
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        if alignment_tokens is None:
            # Fast path: when the coordinator imposes no alignment constraint.
            return None
        block_size = kv_cache_spec.block_size * dcp_world_size
        if alignment_tokens % block_size != 0:
            # The mask is block-granular, so a sub-block alignment cannot be
            # represented exactly. This happens for hybrid offloading, where
            # ``alignment_tokens`` is the full-attention chunk size and need not
            # be a multiple of this SWA group's (DCP-scaled) block size (e.g.
            # Gemma). Fall back to dense: every block is reachable, which never
            # drops a block that could serve a hit.
            return None

        # Contiguous blocks a hit needs at a boundary (incl. the EAGLE peek).
        need = cls._contiguous_blocks_for_hit(
            window_size=kv_cache_spec.sliding_window,
            block_size=block_size,
            use_eagle=use_eagle,
        )
        # The matched run's right edge sits on the aligned boundary block when
        # EAGLE peeks one block past it (shift=1), otherwise on the last block
        # before the boundary (shift=0).
        shift = 1 if use_eagle else 0

        mask = [False] * (end_block - start_block)

        # (1) Segment-boundary tails. ``retention_interval``:
        #   None -> dense (a tail at every ``alignment_tokens`` boundary);
        #   0    -> no dense tails (only the replay boundary below);
        #   >0   -> a tail once per ``retention_interval``-sized segment.
        segment_tokens = (
            alignment_tokens
            if retention_interval is None
            else (None if retention_interval == 0 else retention_interval)
        )
        if segment_tokens is not None:
            per_segment = segment_tokens // block_size
            if need >= per_segment:
                # Every block is reachable; cache them all.
                return None
            for i in range(start_block, end_block):
                if i >= shift and (i - shift) % per_segment >= per_segment - need:
                    mask[i - start_block] = True

            if final_segment_end_block is not None and not use_eagle:
                final_segment_size = final_segment_end_block % per_segment
                if final_segment_size:
                    final_segment_start = final_segment_end_block - final_segment_size
                    final_tail_start = max(
                        final_segment_start, final_segment_end_block - need
                    )
                    final_tail_end = min(end_block, final_segment_end_block)
                    for i in range(max(start_block, final_tail_start), final_tail_end):
                        mask[i - start_block] = True

        # (2) Reachable-boundary tails: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction. Both
        # land before segments would cover them under sparse retention, so keep
        # the ``need``-block tail ending on each boundary explicitly.
        if retention_interval is not None:
            for boundary_tokens in reachable_boundaries:
                aligned = boundary_tokens // alignment_tokens * alignment_tokens
                end = aligned // block_size + shift
                for j in range(max(start_block, end - need), min(end_block, end)):
                    mask[j - start_block] = True

        return mask

    # [CN] 滑出窗口的 token 数。注意末尾 extra_retained_tokens：
    #      多模块 MTP 可能回退重算最后几个 token，所以尾部要**多留**一段不回收，
    #      否则重算时发现 KV 已经被扔了。
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        The trailing edge of the window is extended by ``extra_retained_tokens``
        so that those extra trailing tokens' blocks are retained (but not
        attended). This is needed for multi-module spec decoding which can
        re-prefill the last num_spec_prefill_tokens - 1 tokens from the end
        of the sequence, and thus needs to delay freeing/caching of blocks.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        return max(
            0,
            num_computed_tokens - self.sliding_window + 1 - self.extra_retained_tokens,
        )

    # [CN] SWA 的前缀块是 null 占位（不是真块），不能用 ref_cnt 统计，
    #      所以直接返回 0：暂不支持 cascade attention + sliding window。
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        return 0


# [CN] 环形缓冲管理器：每个请求**只占 1 个块**，循环覆盖写入，
#      因此天然不支持前缀缓存（内容会被后来的 token 覆盖）。
#      适用于某些特定的局部注意力实现（如部分 sink / kpool 类结构）。
class CircularBufferManager(FullAttentionManager):
    """Claims the ring's single block per request; prefix caching disabled."""

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def _claim_ring_block(self, request_id: str) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        return 0 if self.req_to_blocks.get(request_id) else 1

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        return self._claim_ring_block(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        self._claim_ring_block(request_id)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        return

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        return

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        return

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return 0


# [CN] Kpool 尾巴：同样是“1 块/请求”的环形临时缓冲，
#      只是对应不同的 spec 类型（KpoolTailSpec）。
class KpoolTailManager(CircularBufferManager):
    """One-block circular scratch manager for ``KpoolTailSpec``."""


# [CN] 分块局部注意力：把序列切成固定大小的 chunk，
#      注意力只在“当前 chunk 及之前已完成的 chunk 边界”内做。
#      与 SWA 的区别：窗口是按 chunk **对齐**的，而不是滑动的，
#      所以“哪些块用不到”可以直接用除法算出来。
class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens).

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            "chunked local attention groups"
        )
        assert drop_eagle_block is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        hit_length = len(computed_blocks[0]) * kv_cache_spec.block_size
        return computed_blocks, hit_length

    # [CN] 跳过的是“当前 chunk 左边”的所有完整 chunk：
    #      (num_computed_tokens // chunk_size) * chunk_size。
    #      注意与 SWA 的区别：这里不减窗口、不减 1，因为 chunk 是硬对齐的。
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


# [CN] Mamba / 线性注意力管理器 —— **本文件最特殊的一个**。
#
# 与注意力的本质区别：Mamba 的“状态”是**递推**的，不是追加的。
#   注意力：第 i 个 token 的 KV 独立存在，可以任意复用前 k 个；
#   Mamba  ：状态是“读到第 i 个 token 时的压缩结果”，只能整块复用，
#            而且必须**恰好落在某个边界上**才有意义。
#
# 由此产生三个特殊机制：
#   1) get_num_skipped_tokens = n - 1：只需要最新那一个状态，历史全扔；
#   2) mamba_cache_mode == 'align' 时，块表**不是 append-only** —— 
#      中间状态会被清空释放、投机块会原地挪位；
#   3) 因此外部 KV connector 无法靠“位置”定位状态块，
#      必须靠 _pending_boundary_state_offloads 显式交接（请求,组,块,边界token）。
class MambaManager(SingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        # Mamba layers use TP instead of DCP, so each rank holds the full
        # recurrent state. Undo the DCP/PCP block_size scaling that the base
        # class applies for attention groups whose KV cache is partitioned.
        self.block_size = kv_cache_spec.block_size
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
        self.has_prefill_checkpoint_blocks = (
            self.mamba_cache_mode == "align"
            and kv_cache_spec.num_prefill_checkpoint_blocks > 0
        )
        # Mamba checkpoints follow Eagle's global replay boundary.
        self.drop_eagle_checkpoint_block = False
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()
        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
            # checkpoint position and reserved block index for the current
            # allocation.
            self._checkpoints: dict[str, tuple[int, int]] = {}
            # Requests that registered their own last-prompt-boundary partial
            # tail (producers). A later CoW hands its private copy to the
            # connector; a request that finishes first hands off this table
            # source directly.
            self._producer_partial_tail_reqs: dict[str, tuple[KVCacheBlock, int]] = {}

    # [CN] Mamba 的命中查找是**从右往左找单个块**：
    #      找到最后一个命中的块即可（状态块本身就是完整的“读到这里”的结果），
    #      然后在前面补 null 占位，使块表长度与 token 数对应。
    #      细粒度模式下则按 hash 单位从高到低试探，取最长可命中边界。
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        hit_length = 0

        block_size = kv_cache_spec.block_size
        if alignment_tokens < block_size and block_size % alignment_tokens == 0:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            hash_block_size = alignment_tokens
            scale_factor = block_size // hash_block_size
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
                num_tokens = (fine_idx + 1) * hash_block_size
                block_hash = block_hashes[fine_idx]
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    block_idx = fine_idx // scale_factor
                    for computed, cached in zip(computed_blocks, cached_block):
                        computed.extend([block_pool.null_block] * block_idx)
                        computed.append(cached)
                    hit_length = num_tokens
                    break
            return computed_blocks, hit_length

        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                hit_length = (i + 1) * block_size
                break  # we just need the last match - early stopping

        return computed_blocks, hit_length

    # [CN] Mamba 的稀疏保留：一次命中只需要**一个**状态块（不需要连续窗口），
    #      所以每个边界只保留 1 块，省得比 SWA 更激进。
    #      reachable_boundaries 同 SWA —— 重放边界与公共前缀接点必留。
    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
        final_segment_end_block: int | None = None,
    ) -> list[bool] | None:
        """Sparse Mamba state-snapshot retention.

        ``retention_interval``:

          ``None`` -> dense (cache every block; default, unchanged behavior)
          ``0``    -> keep only the ``reachable_boundaries`` states
          ``> 0``  -> keep one state per ``retention_interval``-sized segment

        ``reachable_boundaries`` are proven reuse points (the replay boundary and
        any cross-request shared-prefix junction, Marconi-style APC); their
        boundary state is always kept so sparse retention does not defeat reuse.
        """
        if retention_interval is None or alignment_tokens is None:
            # Dense caching (default) or no alignment constraint imposed.
            return None
        assert isinstance(kv_cache_spec, MambaSpec)
        block_size = kv_cache_spec.block_size
        mask = [False] * (end_block - start_block)

        # (1) Segment-boundary states. A Mamba hit needs exactly the single
        # state block ending on the boundary (no window, and draft models have
        # no mamba layers, so no eagle shift). Block ``i`` ends at token
        # ``(i + 1) * block_size``.
        segment_tokens = None if retention_interval == 0 else retention_interval
        if segment_tokens is not None:
            per_segment = segment_tokens // block_size
            if per_segment <= 1:
                # Interval at/below the block size: every block is a boundary.
                return None
            first_boundary = (
                start_block + per_segment
            ) // per_segment * per_segment - 1
            for i in range(first_boundary - start_block, len(mask), per_segment):
                mask[i] = True

        # (2) Reachable-boundary states: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction, both
        # of which segments would otherwise skip under sparse retention. A Mamba
        # hit needs exactly the single state block ending on the boundary.
        for boundary_tokens in reachable_boundaries:
            aligned = boundary_tokens // alignment_tokens * alignment_tokens
            boundary_block = aligned // block_size - 1
            if start_block <= boundary_block < end_block:
                mask[boundary_block - start_block] = True

        return mask

    # [CN] Mamba 的回收：除了基类的逻辑，align 模式还要额外释放
    #      “前前一步分配的状态块”（last_state_block_idx）。
    #      原因：align 模式下每步把状态从上一块拷到新块，
    #      拷完上一块就没用了；但 prefill 期间块可能不连续，
    #      所以必须靠记录的 idx 精确定位，不能按长度推算。
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        assert isinstance(self.kv_cache_spec, MambaSpec)

        super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
            # The block allocated in the previous step is used to copy Mamba states
            # into the block allocated in the current step; the earlier block is
            # no longer needed and should be freed here.
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # Blocks allocated during prefill may be non-contiguous. Use
            # `last_state_block_idx` to free the appropriate block and replace it
            # with a null block.
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(processed_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        return 0

    # [CN] 是否需要写一个“内部 checkpoint 块”：
    #      为了让长 prefill 也能部分复用，Mamba 会在特定边界额外存一份状态。
    #      判定条件包括 spec 是否开启、该位置是否符合对齐要求、
    #      以及目标块当前是否为空/是否会被投机块覆盖。
    def _needs_internal_checkpoint(
        self,
        request_id: str,
        query_start: int,
        query_end: int,
        checkpoint_position: int,
    ) -> bool:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        checkpoint_idx = cdiv(query_end, self.block_size) - 2
        blocks = self.req_to_blocks[request_id]
        return (
            self.has_prefill_checkpoint_blocks
            and is_mamba_prefill_checkpoint_valid(
                query_start=query_start,
                query_end=query_end,
                checkpoint_position=checkpoint_position,
                hash_block_size=self.block_pool.hash_block_size,
                mamba_block_size=self.block_size,
                checkpoint_alignment=(self.kv_cache_spec.prefill_checkpoint_alignment),
            )
            and checkpoint_idx >= 0
            and (
                checkpoint_idx >= len(blocks)
                or blocks[checkpoint_idx].is_null
                or (
                    request_id in self._allocated_block_reqs
                    and checkpoint_idx >= len(blocks) - self.num_speculative_blocks
                )
            )
        )

    # [CN] Mamba 的分配预测，头部有一处很“脏”但很实在的技巧：
    #      如果将要命中的块是**本 step 内别的请求刚缓存的**，
    #      直接返回 num_gpu_blocks + 1 —— 一个必然超过池容量的数，
    #      于是调度器会认为“块不够”，把这个请求推到下一步再调度。
    #      为什么：本 step 内新写的 Mamba 状态尚未稳定（还要被本步前向覆盖），
    #      此时命中它会读到错误内容。宁可延后一步，也不能读脏。
    #
    #      align 模式还额外处理：lookahead token 不分配块（会破坏对齐）、
    #      投机块要预留、checkpoint 块要预留。
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if (
            len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            # Mamba can't rely on blocks generated by other requests in the current step
            # To put it in the next step, we return num_gpu_blocks + 1 so
            # that kv_cache_manager will think there is no enough blocks to allocate now
            # and don't schedule it in the current step.
            return self.block_pool.num_gpu_blocks + 1
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += (
                    self.kv_cache_spec.block_size * self.num_speculative_blocks
                )
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model

            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            num_new_blocks = (
                num_required_blocks
                - len(new_computed_blocks)
                - len(self.req_to_blocks[request_id])
            )
            has_partial_hit = (
                self._has_partial_local_hit(
                    new_computed_blocks, num_local_computed_tokens
                )
                or request_id in self._partial_hit_reqs
            )
            if has_partial_hit:
                num_new_blocks = max(num_new_blocks, 0) + 1
            checkpoint_position = get_mamba_prefill_checkpoint_position(
                num_tokens,
                self.block_pool.hash_block_size,
                self.drop_eagle_checkpoint_block,
            )
            if not self._needs_internal_checkpoint(
                request_id,
                total_computed_tokens,
                num_tokens,
                checkpoint_position,
            ):
                checkpoint_position = 0
            checkpoint_block = int(checkpoint_position > 0)
            if not apply_admission_cap:
                if checkpoint_position > 0:
                    checkpoint_idx = cdiv(num_tokens, self.block_size) - 2
                    self._checkpoints[request_id] = (
                        checkpoint_position,
                        checkpoint_idx,
                    )
                else:
                    self._checkpoints.pop(request_id, None)
            if num_new_blocks > 0:
                blocks_allocated = request_id in self._allocated_block_reqs
                if not (checkpoint_block and blocks_allocated):
                    num_new_blocks = 1 + int(has_partial_hit) + checkpoint_block
                    if not blocks_allocated:
                        num_new_blocks += self.num_speculative_blocks

            num_evictable_computed_blocks = self._get_num_evictable_blocks(
                new_computed_blocks
            )
            return num_new_blocks + num_evictable_computed_blocks

    # [CN] Mamba 的实际分配（align 模式相当复杂，按这个顺序读）：
    #   1) 若块已够且无部分命中/checkpoint，直接返回 []（本步不新分配）；
    #   2) 记录 last_state_block_idx（本步状态所在的块，下一步用来拷贝）；
    #   3) 补 null 占位（滑过的中间状态位置），注意跳过 checkpoint 块；
    #   4) 已在跑的请求：把独占的投机暂存块**原地挪位**到新位置；
    #   5) 处理部分命中的 CoW —— 已在跑的请求不能换块（worker 块表是
    #      append-only），所以改为把缓存条目**迁移**到新块（move_block_hashes）；
    #      新请求则直接替换块表项（_apply_cow）。
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += self.block_size * self.num_speculative_blocks
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]
            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            checkpoint_block = int(request_id in self._checkpoints)
            partial_hit = self._partial_hit_reqs.get(request_id)
            has_partial_hit = partial_hit is not None
            # `num_required_blocks` might be less than `len(req_blocks)` if blocks are
            # over-allocated at last round.
            if (
                num_required_blocks <= len(req_blocks)
                and not has_partial_hit
                and not checkpoint_block
            ):
                self._allocated_block_reqs.add(request_id)
                return []
            else:
                prev_block_len = len(req_blocks)
                blocks_allocated = request_id in self._allocated_block_reqs
                # Record the last state block
                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    self.last_state_block_idx[request_id] = prev_block_len - 1

                num_skipped_blocks = (
                    num_required_blocks - self.num_speculative_blocks - 1
                )
                # null blocks
                if prev_block_len < num_skipped_blocks:
                    # minus the internal checkpoint block
                    # so we don't set null for that block
                    null_end = num_skipped_blocks - checkpoint_block
                    req_blocks.extend(
                        [self._null_block for _ in range(prev_block_len, null_end)]
                    )

                if blocks_allocated and not checkpoint_block:
                    # Relocate exclusively owned speculative scratch blocks.
                    for block_idx in range(
                        prev_block_len - self.num_speculative_blocks, prev_block_len
                    ):
                        if block_idx < num_skipped_blocks:
                            self._relocate_speculative_block(req_blocks, block_idx)
                        else:
                            break
                num_new_blocks = max(num_required_blocks - len(req_blocks), 0)
                if has_partial_hit:
                    num_new_blocks = max(num_new_blocks, 0) + 1
                max_new_blocks = 1 + int(has_partial_hit) + checkpoint_block
                if not blocks_allocated or checkpoint_block:
                    max_new_blocks += self.num_speculative_blocks
                assert num_new_blocks <= max_new_blocks
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                returned_blocks = req_blocks[prev_block_len:]
                if partial_hit is not None:
                    block_idx, source_block = partial_hit
                    cow_block = new_blocks[0]
                    new_blocks = new_blocks[1:]
                    if blocks_allocated:
                        # The worker block table of a running request is
                        # append-only, so the request must stay on
                        # source_block. Move the cache entry to cow_block
                        # instead; the queued copy fills it before forward
                        # overwrites source_block.
                        assert req_blocks[block_idx] is source_block
                        self.block_pool.move_block_hashes(source_block, cow_block)
                        self._pending_cow_copies.append((source_block, cow_block))
                        source_block.ref_cnt += 1
                        producer_tail = self._producer_partial_tail_reqs.pop(
                            request_id, None
                        )
                        if producer_tail is not None:
                            marker_block, boundary_tokens = producer_tail
                            assert marker_block is source_block
                            # This CoW preserved a producer's own boundary
                            # state in cow_block; hand it to the connector for
                            # partial-tail offload once the copy has run.
                            self._pending_boundary_state_offloads.append(
                                (
                                    request_id,
                                    self.kv_cache_group_id,
                                    cow_block,
                                    boundary_tokens,
                                )
                            )
                        if cow_block.block_hash is not None:
                            # The moved entry is only filled by this step's
                            # copy, so defer same-step hits on it.
                            self.cached_blocks_this_step.add(cow_block.block_hash)
                    else:
                        self._apply_cow(request_id, block_idx, source_block, cow_block)
                        returned_blocks = [cow_block] + returned_blocks
                req_blocks.extend(new_blocks)
                self._allocated_block_reqs.add(request_id)
                self._partial_hit_reqs.pop(request_id, None)
                returned_blocks.extend(new_blocks)
                return returned_blocks

    # [CN] 投机暂存块挪位：把它从旧下标摘下、追加到末尾，原位置填 null。
    #      前提是这块“独占且没有哈希”—— 否则挪动会破坏别人的缓存引用。
    def _relocate_speculative_block(
        self, req_blocks: list[KVCacheBlock], block_idx: int
    ) -> None:
        block = req_blocks[block_idx]
        assert self.block_pool.is_block_writable(block), (
            "Speculative Mamba blocks must be exclusively owned and unhashed "
            "before relocation"
        )
        req_blocks.append(block)
        req_blocks[block_idx] = self._null_block

    # [CN] 请求结束时结算“部分尾部”的卸载：
    #      只有当在途 token 为 0 且已算 token 正好等于边界时才成立，
    #      否则那块状态还会被继续覆盖，交出去没有意义。
    def finalize_partial_tail_offload(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_in_flight_tokens: int,
    ) -> tuple[int, KVCacheBlock, int] | None:
        if self.mamba_cache_mode != "align":
            return None
        producer_tail = self._producer_partial_tail_reqs.pop(request_id, None)
        if producer_tail is None:
            return None
        source_block, boundary_tokens = producer_tail
        if num_in_flight_tokens != 0 or num_computed_tokens != boundary_tokens:
            return None
        return self.kv_cache_group_id, source_block, boundary_tokens

    # [CN] Mamba 的记账清理：除了基类的几项，还要清掉本请求尚未被领取的
    #      边界状态交接记录 —— 块马上要回到池子里了，
    #      不能让 connector 之后拿走一块“可能已经被别人复用”的块。
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
            self._checkpoints.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
            # An offer is only guaranteed to hold committed bytes until the end
            # of the pass that made it. This request's blocks are going back to
            # the pool now, so drop its not-yet-offered hand-offs rather than
            # let a connector claim a block another request may already have
            # been handed.
            self._pending_boundary_state_offloads = [
                entry
                for entry in self._pending_boundary_state_offloads
                if entry[0] != request_id
            ]
        return super().pop_blocks_for_free(request_id)

    # [CN] Mamba 只需要**最新**那一个状态，所以前面 n-1 个 token 的状态全可扔。
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens whose mamba state are not needed anymore. Mamba only
        need to keep the state of the last computed token, so we return
        num_computed_tokens - 1.
        """
        return num_computed_tokens - 1

    # [CN] Mamba 的缓存：写完哈希后，把这些块记进 cached_blocks_this_step，
    #      用来实现前面说的“本 step 内新缓存的块不许立刻被别人命中”。
    #      align 模式还要为每个保留下来的边界登记一条交接记录，
    #      供外部 KV connector 精确卸载（因为块表不连续、不能靠位置推断）。
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
        if self.mamba_cache_mode == "align":
            partial_hash = self._cache_partial_tail_block(request, num_tokens)
            if partial_hash is not None:
                self.cached_blocks_this_step.add(partial_hash)
        if num_cached_blocks_after > num_cached_blocks_before:
            blocks = self.req_to_blocks[request.request_id]
            for idx in range(num_cached_blocks_before, num_cached_blocks_after):
                block = blocks[idx]
                # Skip null blocks (align-mode skipped states) and blocks that
                # were not cached this step — with sparse retention
                # (reachable_block_mask) the intermediate state snapshots carry
                # no hash and must not be recorded as cached-this-step.
                if block.is_null or block.block_hash is None:
                    continue
                self.cached_blocks_this_step.add(block.block_hash)
                if self.mamba_cache_mode == "align":
                    assert block.block_hash_num_tokens is not None
                    # Offer every retained boundary with its exact block.
                    # The connector filters against its save window, which may
                    # extend past the original prompt during resumed prefill.
                    self._pending_boundary_state_offloads.append(
                        (
                            request.request_id,
                            self.kv_cache_group_id,
                            block,
                            block.block_hash_num_tokens,
                        )
                    )

    # [CN] 每步开始清空“本 step 缓存的块”—— 这些块过了本步就允许被命中了。
    def new_step_starts(self) -> None:
        self.cached_blocks_this_step.clear()

    # [CN] 缓存部分尾块。两种情形：
    #      a) 有 checkpoint：把预留块按 checkpoint 边界**重新定键**
    #         （replace_existing_hashes=True，因为旧键已经失效）；
    #      b) 无 checkpoint：prompt 尾部落在块内时，注册一个部分哈希，
    #         并把该请求标为“生产者”（producer），
    #         CoW 完成后把那块交给 connector 卸载。
    def _cache_partial_tail_block(
        self,
        request: Request,
        num_tokens: int,
    ) -> BlockHashWithGroupId | None:
        hash_block_size = self.block_pool.hash_block_size
        # Re-key the reserved block at its exported checkpoint boundary.
        checkpoint = self._checkpoints.get(request.request_id)
        if checkpoint is not None:
            checkpoint_position, checkpoint_idx = checkpoint
            blocks = self.req_to_blocks[request.request_id]
            assert 0 <= checkpoint_idx < len(blocks)
            checkpoint_block = blocks[checkpoint_idx]
            if checkpoint_block.block_hash_num_tokens == checkpoint_position:
                return None
            return self.block_pool.cache_partial_block(
                request=request,
                block=checkpoint_block,
                num_tokens=checkpoint_position,
                kv_cache_group_id=self.kv_cache_group_id,
                block_size=self.block_size,
                replace_existing_hashes=True,
            )
        if self.block_size == hash_block_size:
            return None
        if num_tokens % self.block_size == 0:
            return None
        if num_tokens % hash_block_size != 0:
            return None
        latest_prompt_hash_boundary = (
            request.num_prompt_tokens // hash_block_size
        ) * hash_block_size
        if num_tokens != latest_prompt_hash_boundary:
            return None

        block_idx = num_tokens // self.block_size
        blocks = self.req_to_blocks[request.request_id]
        if block_idx >= len(blocks):
            return None
        source_block = blocks[block_idx]
        if source_block.is_null:
            return None

        partial_hash = self.block_pool.cache_partial_block(
            request=request,
            block=source_block,
            num_tokens=num_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )
        if partial_hash is not None:
            self._partial_hit_reqs[request.request_id] = (block_idx, source_block)
            self.num_cached_block[request.request_id] = block_idx
            # Producer of this partial tail: the boundary state currently lives
            # in ``source_block`` but the next step's forward overwrites it. The
            # upcoming CoW copies it into a durable cow_block; record the req so
            # allocate_new_blocks hands that block to the connector for offload.
            self._producer_partial_tail_reqs[request.request_id] = (
                source_block,
                num_tokens,
            )
        return partial_hash


# [CN] Cross-attention（编码器-解码器）管理器。
#      关键认知：cross-attention 的 K/V 来自 **encoder 输出**，
#      每个请求的输入（图片/音频）都不一样，所以：
#        - 不共享、不做前缀缓存；
#        - 大小只跟 encoder token 数有关，与解码长度无关（一次性分配）。
class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # Cross-attention does not use prefix caching / external KV loads.
        return

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        raise NotImplementedError("CrossAttentionManager does not support caching")


# [CN] 带 sink 的全注意力：永久保留最开头 sink_len 个 token 的 KV
#      （attention sink 现象：开头的 token 承载了大量注意力权重，扔掉会崩）。
#      实现上直接从空闲队列**预先取走**这几块，永不参与淘汰。
class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)


# [CN] 工厂：按 KVCacheSpec 的类型查注册表拿 manager 类。
#      用注册表而不是 if/elif，是为了让**外部（平台/插件）可以注册自己的 spec**。
#
#      这里还负责给 SWA / chunked-local 设置 max_admission_blocks_per_request：
#      这两个类型会**边跑边回收块**，所以“单请求最多占多少块”不是
#      max_model_len / block_size，而是一个更小的“回收感知”上限。
#      这个上限必须和启动时计算池大小用的是**同一个函数**，
#      否则两边漂移会重新引入 #39734 那类死锁 / 中途 OOM。
def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_in_flight_tokens: int,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """
    Get the appropriate manager for a given KVCacheSpec.

    Uses the KVCacheSpecRegistry to look up the manager class, supporting
    both built-in and custom specs registered via @register_kv_cache_spec
    and KVCacheSpecRegistry.register.

    Args:
        kv_cache_spec: The KVCacheSpec instance
        max_in_flight_tokens: The max tokens scheduled but not yet settled
            (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`
        max_model_len: The maximum context length the model could serve
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass
    """
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"
    )
    # SlidingWindow / ChunkedLocalAttention managers recycle blocks;
    # the runtime admission cap must match the recycling-aware bound the
    # startup pool sizer uses (single source of truth: the spec method).
    # R-SWA also recycles gap blocks but peak physical KV still fits the
    # full-attention bound (prefix + window <= max_model_len), so it inherits
    # FullAttentionSpec sizing without a separate admission cap.
    if isinstance(
        kv_cache_spec,
        (SlidingWindowSpec, ChunkedLocalAttentionSpec),
    ):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


# [CN] 内置 spec -> manager 的注册表初始化。
#
#      uniform_type_base_spec 的作用：**分组归并**。
#      同一 uniform 基类的 spec 会被归到同一 KV cache group，
#      这样 MLA / RSWA / Sink 等 FullAttention 的子类可以和普通 full attention
#      合并处理，减少 group 数量（group 越少，调度与块表越简单）。
#
#      最后一行：交给当前平台注册自己的自定义 spec（插件扩展点）。
def register_all_kvcache_specs(vllm_config):
    """Built-in spec registration"""
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    KVCacheSpecRegistry.register(
        SlidingWindowSpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )
    KVCacheSpecRegistry.register(
        CircularBufferSpec,
        CircularBufferManager,
        uniform_type_base_spec=CircularBufferSpec,
    )
    KVCacheSpecRegistry.register(
        SlidingWindowMLASpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )
    KVCacheSpecRegistry.register(
        KpoolTailSpec,
        KpoolTailManager,
        uniform_type_base_spec=KpoolTailSpec,
    )

    KVCacheSpecRegistry.register(
        MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec
    )
    KVCacheSpecRegistry.register(
        ChunkedLocalAttentionSpec,
        ChunkedLocalAttentionManager,
        uniform_type_base_spec=ChunkedLocalAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        CrossAttentionSpec,
        CrossAttentionManager,
        uniform_type_base_spec=CrossAttentionSpec,
    )

    # FullAttentionSpec subclasses — grouped with FullAttentionSpec
    KVCacheSpecRegistry.register(
        MLAAttentionSpec, FullAttentionManager, uniform_type_base_spec=FullAttentionSpec
    )
    KVCacheSpecRegistry.register(
        RSWASpec, RSWAManager, uniform_type_base_spec=FullAttentionSpec
    )
    # NOTE(Mengqing): HiddenStateCacheSpec won't take part in
    # grouping, thus the uniform_type_base_spec is just a
    # placeholder.
    KVCacheSpecRegistry.register(
        HiddenStateCacheSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        SinkFullAttentionSpec,
        SinkFullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    from vllm.platforms import current_platform

    current_platform.register_custom_kv_cache_specs(vllm_config)
