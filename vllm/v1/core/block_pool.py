# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_block_hashes,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


# [CN] 前缀缓存的 **hash -> block** 映射表。
#      注意它的 value 是 **联合类型**：单个 KVCacheBlock，或者
#      {block_id: KVCacheBlock} 的字典。
class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    # [CN] 为什么**不做去重**：同一个 hash 可能存在多个物理块。
    #      代价是可能浪费一点显存，换来的是"已分配的 block id 永不改变"，
    #      于是 block table 可以保持 **append-only** —— 这对 worker 侧
    #      持久 batch 的实现非常重要（改 id 会牵动一大片）。
    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    # [CN] 为什么搞联合类型而不是一律用 dict：绝大多数情况一个 hash 只对应
    #      一个块，如果每个 key 都建一个 dict，GC 压力会明显变大。
    #      这是热路径上典型的"用类型判断换内存/GC"的优化。
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    # [CN] 取**任意一个**该 hash 对应的块（重复时不保证是哪一个）。
    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    # [CN] 判断该 hash 是否映射到了指定 block_id。
    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        """
        Checks whether the key maps to the given block ID.
        """
        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        if isinstance(blocks, dict):
            return block_id in blocks
        self._unexpected_blocks_type(blocks)
        return False

    # [CN] 插入。三种状态迁移：空 -> 单块；单块 -> 双元素 dict；dict -> 追加。
    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    # [CN] 弹出指定 hash 下的指定 block。
    #      注意单块模式下若 id 不匹配会把块**放回去**再返回 None ——
    #      这是保守做法（宁可多留一个失效条目，也不能误删别人的块）。
    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


# [CN] **块池**：vLLM KV cache 内存管理的地基。
#      它只管"块"这一个概念，不关心这些块属于哪个请求、哪个层。
#
#      两块核心状态：
#        1) free_block_queue：**双向链表**维护的空闲块队列，
#           同时充当 LRU 淘汰顺序（开了前缀缓存时，free 队列里其实
#           装的是"可淘汰的候选"，并非真的空闲）。
#        2) cached_block_hash_to_block：前缀缓存的 hash 索引。
#
#      最关键的概念是 **ref_cnt（引用计数）**：
#        ref_cnt > 0 ：被若干请求持有（前缀复用的块会被多个请求共享）
#        ref_cnt = 0 ：在空闲队列里，随时可被重新分配（= 淘汰候选）
#      所有分配/释放/淘汰的正确性都建立在这个计数上。
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # [CN] 池子里所有的块，**一次性全建好**，之后只改状态、不再增删。
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # [CN] 空闲块队列（双向链表）。开启缓存后它同时是 **LRU 淘汰顺序**：
        #      从队头取 = 优先淘汰最久未用的。
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        # [CN] hash -> block（查命中用）。
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        # [CN] **反向索引**：block_id -> 它身上挂的所有 hash。
        #      为什么需要：一个块可能同时被多个 hash 指向（partial 条目 +
        #      主 hash），回收时必须把它们**全部**摘掉，否则会留下悬空索引。
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # [CN] **空块（null block）**：block_id=0 的占位块，代表"这块不需要存储"。
        #      典型用途：滑窗注意力滑出去的 token、Mamba align 模式下被清空的
        #      状态 —— 它们在 block table 里要有位置，但不占真实显存。
        #      注意注释里的警告：它**不参与引用计数**，各处都要特判，
        #      否则会把它当成空闲块分配出去或"释放"掉。
        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

    # [CN] 按 hash 查缓存块。要点：要**每个 group 都命中**才算命中，
    #      任一 group 未命中就整体返回 None —— 因为不同 group 的块必须
    #      一一对齐（同一个 token 位置在所有 group 都要有块）。
    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    # [CN] 把请求中**已经写满**的块登记进前缀缓存。
    #      触发时机：每步解码后，请求末尾可能刚好填满一个新块。
    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
    # [CN] 块掩码：为 False 的块**跳过**哈希登记。
    #      用途：某些 group（比如滑窗的尾部窗口）只查一部分块，
    #      永远不可能被命中的块就没必要进 hash 表（省内存、省查询）。
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        # [CN] 跳过 null 块和被 mask 掉的块（它们的内容不具可复用性）。
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            block_hash = new_block_hashes[i]
            num_hash_tokens = (num_cached_blocks + i + 1) * block_size

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            # [CN] "新满块"身上却已经有 hash，只有一种合法情况：
            #      同一个块从 **partial 条目升级为 full 条目**（块被续写满了）。
            #      所以这里用 assert 卡住其它可能性，然后先把旧 hash 摘掉。
            if blk.block_hash is not None:
                # The only valid case where a "new full block" already has a
                # hash is partial->full promotion of the same cache block.
                assert (
                    blk.block_hash_num_tokens is not None
                    and blk.block_hash_num_tokens < num_hash_tokens
                )
                removed_hashes = self._remove_cached_block_hashes(blk)
                self._emit_block_removed_events(removed_hashes)
            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        # [CN] 发 KV 事件（供外部 KV 感知路由 / 网关消费）。
        #      每个块的 extra_keys 单独算：不同块的多模态特征可能不同，
        #      而且只有第一个块带 cache_salt。
        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                self._build_block_stored_event(
                    request,
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    block_size=block_size,
                    kv_cache_group_id=kv_cache_group_id,
                    extra_keys_list=extra_keys_list,
                )
            )

    # [CN] 构造 BlockStored 事件。**两条路径共用**这个构造：
    #      新缓存的块、以及前缀复用命中的块 —— 保证下游看到同样的事件形状。
    def _build_block_stored_event(
        self,
        request: Request,
        block_hashes: list[ExternalBlockHash] | None,
        parent_block_hash: ExternalBlockHash | None,
        start_token_idx: int,
        end_token_idx: int,
        block_size: int,
        kv_cache_group_id: int,
        extra_keys_list: list[tuple[Any, ...] | None],
    ) -> BlockStored:
        """Build a ``BlockStored`` KV event for ``request``.

        Shared by ``cache_full_blocks`` (newly cached blocks) and
        ``emit_cached_block_events`` (prefix-cache-reused blocks) so both emit
        identical event shapes for downstream consumers.
        """
        return BlockStored(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=request.all_token_ids[start_token_idx:end_token_idx],
            block_size=block_size,
            lora_id=request.lora_request.adapter_id if request.lora_request else None,
            medium=MEDIUM_GPU,
            lora_name=request.lora_request.name if request.lora_request else None,
            extra_keys=extra_keys_list if extra_keys_list else None,
            group_idx=kv_cache_group_id,
            session_id=request.session_id,
        )

    # [CN] 为"前缀复用命中的块"生成事件。与 cache_full_blocks 的区别：
    #      这里**不修改**任何块状态（块本来就已经缓存好了），只发事件，
    #      让外部消费者（比如网关的 KV 感知路由）知道这些块被复用了。
    def emit_cached_block_events(
        self,
        request: Request,
        num_cached_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Generate BlockStored events for blocks reused from prefix cache.

        Unlike cache_full_blocks(), this does NOT modify block state —
        the blocks are already cached. It only generates events so that
        external consumers (e.g. gateway) can learn about reused blocks.

        Args:
            request: The request whose prefix cache blocks were reused.
            num_cached_blocks: Number of blocks that were cache hits.
            block_size: Number of tokens per block.
            kv_cache_group_id: The KV cache group ID.
        """
        if not self.enable_kv_cache_events or num_cached_blocks == 0:
            return

        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        # Collect external hashes and extra_keys for cached blocks.
        cached_hashes: list[ExternalBlockHash] = []
        extra_keys_list: list[tuple[Any, ...] | None] = []
        curr_mm_idx = 0
        for i in range(num_cached_blocks):
            block_start = i * block_size
            block_end = block_start + block_size
            cached_hashes.append(maybe_convert_block_hash(block_hashes[i]))
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            extra_keys_list.append(extra_keys)

        if not cached_hashes:
            return

        # [CN] 前缀命中的块一定是从 block 0 开始的**连续前缀**，
        #      所以整组的 parent hash 必然是 None。
        # Prefix-cache hits always form a contiguous prefix starting at block 0,
        # so the first (and thus the whole group's) parent block hash is None.
        parent_block_hash: ExternalBlockHash | None = None
        start_token_idx = 0
        end_token_idx = num_cached_blocks * block_size

        logger.debug(
            "EmitCachedBlock event: block_size=%d, "
            "num_cached_blocks=%d, parent_block_hash=%s, "
            "token_ids_len=%d, group_idx=%s",
            block_size,
            num_cached_blocks,
            parent_block_hash,
            len(request.all_token_ids[start_token_idx:end_token_idx]),
            kv_cache_group_id,
        )

        self.kv_event_queue.append(
            self._build_block_stored_event(
                request,
                block_hashes=cached_hashes,
                parent_block_hash=parent_block_hash,
                start_token_idx=start_token_idx,
                end_token_idx=end_token_idx,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
                extra_keys_list=extra_keys_list,
            )
        )

    # [CN] 登记 **partial（部分）**前缀缓存条目。
    #      背景：默认的缓存键以"整块"为粒度，但如果 hash_block_size 小于
    #      block_size（混合 block size 场景），块内部其实存在更细的
    #      可复用边界。这个方法让一个已存在的块能从"块内某个前缀边界"
    #      被查到，而**不需要分配/拷贝新块**。
    #      典型用例：Mamba align 模式、以及不同 group 块大小不一致时。
    def cache_partial_block(
        self,
        request: Request,
        block: KVCacheBlock,
        num_tokens: int,
        kv_cache_group_id: int,
        block_size: int,
        replace_existing_hashes: bool = False,
    ) -> BlockHashWithGroupId | None:
        """Register a partial prefix-cache entry for an existing block.

        Prefix-cache keys normally identify full cache blocks. A partial entry
        makes an existing cache block reachable from a fine-grained prefix
        boundary inside that block without allocating or copying a new
        ``KVCacheBlock``.

        The partial entry is lookup metadata owned by ``block``. If ``block``
        has no primary hash, the key becomes its primary hash. If the block
        already has a primary hash, the partial entry is tracked in
        ``cached_block_hashes_by_block`` so eviction, reset, and promotion can
        remove every hash key that points to the block.

        Args:
            request: Request whose token IDs and block hashes define the
                partial entry.
            block: Existing cache block to make reachable from the partial
                prefix boundary.
            num_tokens: Prefix length represented by the partial entry. It
                must be a positive multiple of ``self.hash_block_size`` and
                cannot exceed the request's computed block hashes.
            kv_cache_group_id: KV cache group that owns the partial entry.
            block_size: Cache block size for the owning group. The partial
                entry hash itself is always the prefix-chain hash at
                ``num_tokens``; ``block_size`` is used to assert that the
                entry is partial within the owning cache block.
            replace_existing_hashes: Whether the block contents were replaced
                and all existing cache entries must be removed before the new
                entry is registered.

        Returns:
            The hash key with group ID if a partial entry can be registered;
            otherwise ``None`` for null blocks.
        """
        if block.is_null:
            return None

        # [CN] 两个前提：block_size 必须是 hash_block_size 的整数倍；
        #      且要么是"替换已有 hash"，要么确实是块内的部分边界。
        assert block_size % self.hash_block_size == 0
        assert replace_existing_hashes or (
            block_size > self.hash_block_size and num_tokens % block_size != 0
        )
        block_hash = self._get_partial_block_hash(request, num_tokens)
        num_hash_blocks = num_tokens // self.hash_block_size
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        already_cached = block.block_hash == block_hash_with_group_id or (
            self.cached_block_hash_to_block.contain(
                block_hash_with_group_id, block.block_id
            )
        )
        # [CN] 块内容被整体替换了 -> 先把旧的所有 hash 摘掉再登记新的。
        if replace_existing_hashes:
            removed_hashes = self._remove_cached_block_hashes(block)
            self._emit_block_removed_events(removed_hashes)
            already_cached = False
            # [CN] 不是替换、但该块上已有更"短"的 hash -> 说明这是**升级**
            #      （更长的前缀），同样需要先摘旧的。
        elif (
            not already_cached
            and block.block_hash is not None
            and block.block_hash_num_tokens is not None
            and block.block_hash_num_tokens < num_hash_blocks * self.hash_block_size
        ):
            removed_hashes = self._remove_cached_block_hashes(block)
            self._emit_block_removed_events(removed_hashes)
        self._insert_block_hash(
            block_hash_with_group_id,
            block,
            num_tokens=num_hash_blocks * self.hash_block_size,
        )
        if self.enable_kv_cache_events and not already_cached:
            parent_hash, block_start = self._get_partial_block_parent_hash_and_start(
                request, num_tokens
            )
            parent_block_hash = (
                maybe_convert_block_hash(parent_hash)
                if parent_hash is not None
                else None
            )
            block_end = num_tokens
            curr_mm_idx = -1 if block_start > 0 else 0
            extra_keys, _ = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[maybe_convert_block_hash(block_hash)],
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[block_start:block_end],
                    block_size=block_end - block_start,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=[extra_keys],
                    group_idx=kv_cache_group_id,
                    session_id=request.session_id,
                )
            )
        return block_hash_with_group_id

    # [CN] 取"前缀边界处的那个 hash"。
    #      因为每个 hash_block_size 的哈希都是**链式**包含完整前缀的，
    #      所以任意边界直接取对应下标即可，不需要重新计算。
    def _get_partial_block_hash(
        self,
        request: Request,
        num_tokens: int,
    ) -> BlockHash:
        assert num_tokens % self.hash_block_size == 0
        num_hash_blocks = num_tokens // self.hash_block_size
        assert 0 < num_hash_blocks <= len(request.block_hashes)

        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        return request.block_hashes[num_hash_blocks - 1]

    # [CN] 父 hash 与起始位置：父就是上一个边界的 hash（第一个则无父）。
    def _get_partial_block_parent_hash_and_start(
        self,
        request: Request,
        num_tokens: int,
    ) -> tuple[BlockHash | None, int]:
        num_hash_blocks = num_tokens // self.hash_block_size
        parent_hash = (
            request.block_hashes[num_hash_blocks - 2] if num_hash_blocks > 1 else None
        )
        block_start = (num_hash_blocks - 1) * self.hash_block_size
        return parent_hash, block_start

    # [CN] 摘掉一个块身上的**全部** hash（主 hash + 反向索引里的 partial），
    #      返回真正被移除的那些。这是"回收一个块"的清理入口。
    def _remove_cached_block_hashes(
        self,
        block: KVCacheBlock,
    ) -> list[BlockHashWithGroupId]:
        block_hashes: list[BlockHashWithGroupId] = []
        if block.block_hash is not None:
            block_hashes.append(block.block_hash)
        block_hashes.extend(self.cached_block_hashes_by_block.pop(block.block_id, ()))
        if not block_hashes:
            return []

        removed_hashes: list[BlockHashWithGroupId] = []
        for block_hash in block_hashes:
            if (
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ):
                removed_hashes.append(block_hash)
        block.reset_hash()
        return removed_hashes

    # [CN] 为每个被移除的 hash 发一个 BlockRemoved 事件。
    def _emit_block_removed_events(
        self,
        block_hashes: list[BlockHashWithGroupId],
    ) -> None:
        if not self.enable_kv_cache_events:
            return
        for block_hash in block_hashes:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )

    # [CN] 登记一个 hash -> block。两处早退：
    #        - 已经是主 hash / 已经在表里的同一块，直接返回（幂等）；
    #        - 否则：块还没有主 hash 就设成主 hash，
    #          已有主 hash 就挂到反向索引的"附加 hash"集合里。
    #      "一个块只能有一个主 hash"是这里的核心不变式。
    def _insert_block_hash(
        self,
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int | None,
    ) -> None:
        if block.block_hash == block_hash_with_group_id:
            return

        if self.cached_block_hash_to_block.contain(
            block_hash_with_group_id, block.block_id
        ):
            return

        if block.block_hash is None:
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
                block_hash_with_group_id
            )
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)

    # [CN] 把 src 块的所有缓存条目**改指向** dst 块。
    #      场景：请求还要继续往 src 里写（内容会变），于是前缀缓存需要
    #      另存一份私有副本 dst，用同样的 hash 对外提供复用。
    #      注意不发事件 —— 条目依然是活的，只是换了宿主。
    def move_block_hashes(
        self,
        src_block: KVCacheBlock,
        dst_block: KVCacheBlock,
    ) -> None:
        """Re-point ``src_block``'s prefix-cache entries to ``dst_block``.

        Used when the request owning ``src_block`` keeps writing into it
        : the prefix cache holds a private copy (``dst_block``)
        under the same hashes instead. Entries stay live; no events emitted.
        """
        assert dst_block.block_hash is None
        assert dst_block.block_id not in self.cached_block_hashes_by_block
        num_tokens = src_block.block_hash_num_tokens
        for block_hash in self._remove_cached_block_hashes(src_block):
            # `num_tokens` only applies to the first (primary) insertion.
            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)

    # [CN] 从空闲队列取 n 个新块。两个要点：
    #        1) 开缓存时取出的块可能还挂着 hash（是可淘汰候选），
    #           所以要先 _maybe_evict_cached_block 摘干净；
    #        2) 分配后 ref_cnt 从 0 变 1。
    #      注意注释里的说明：这里**故意复制了循环代码**，
    #      为的是只遍历一次列表（热路径上省一次分支判断）。
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # [CN] 就是上面说的"故意复制代码换单次遍历"。
        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    # [CN] 淘汰一个块：摘掉它身上所有 hash 并发出移除事件。
    #      返回 False 表示它本来就没有 hash（无需淘汰）。
    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        evicted_hashes = self._remove_cached_block_hashes(block)
        if not evicted_hashes:
            # The block doesn't have hash, eviction is not needed
            return False

        self._emit_block_removed_events(evicted_hashes)
        return True

    # [CN] **touch（提升引用）**：另一个请求命中了同一个前缀块。
    #      关键一行：ref_cnt == 0 的块此时还在空闲队列里（属于可淘汰候选），
    #      被命中后必须**从队列里摘出来**，否则它可能被当作空闲块分配出去，
    #      造成两个请求共用却互不知情 -> 数据被覆盖。
    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    # [CN] 判断一个块能否被**独占写入**（CoW / 原地复用优化的前提）：
    #      非 null + 引用计数恰好为 1 + 没有挂 hash（不参与前缀复用）。
    def is_block_writable(self, block: KVCacheBlock) -> bool:
        """Return whether a block can be mutated by its sole owner."""
        return not block.is_null and block.ref_cnt == 1 and block.block_hash is None

    # [CN] 释放一批块（按调用方给出的**淘汰优先级**排序，越靠前越先被淘汰）。
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # [CN] 这里区分两种回收策略，值得记住：
        #        无 hash 的块 -> **LIFO**（prepend 到队头，下次优先复用）：
        #          刚释放的块还在 cache 里，复用它对 GPU 局部性最好；
        #        有 hash 的块 -> **FIFO**（append 到队尾）：
        #          这样它们在队列里自然形成 LRU 顺序，越久没被复用越先淘汰。
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_to_evict_last = []
        blocks_to_evict_first = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    # FIFO reuse of cached blocks for LRU eviction behavior.
                    blocks_to_evict_last.append(block)

        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
        # Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(blocks_to_evict_last)

    # [CN] 按 block_id 把块**从前缀缓存里摘掉**（但不一定从池子释放）。
    #      注意语义：ref_cnt > 0 的块只是失去缓存身份，仍被请求持有。
    #      常用于 KV connector 报告"这些块的数据已失效"。
    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    # [CN] 清空整个前缀缓存。**权重热更新后必须调用**（RLHF 场景）。
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        # [CN] 为什么要求"只剩 null block 在用"：还有请求在跑时，
        #      它们持有的块身上挂着 hash；贸然清空会让这些块变成幽灵条目，
        #      后续被误命中或误释放。所以这里直接**拒绝**并告警，
        #      让调用方先排空请求。
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block.clear()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    # [CN] 空闲块数。注意：开启缓存后这个数包含"可淘汰的候选块"，
    #      所以它不是严格意义上的"完全空闲"。
    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    # [CN] KV cache 使用率 = 1 - 空闲/总量。
    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # [CN] 减 1 是为了排除常驻的 null block（它从来不算可用容量）。
        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    # [CN] 原子取走全部事件并清空队列（避免事件被重复消费）。
    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
