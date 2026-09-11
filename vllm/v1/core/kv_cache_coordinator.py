# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [CN] 文件总览：KV Cache 协调器（coordinator）
#
# 它在 KV cache 栈里的位置（自下而上共三层）：
#   BlockPool                 （block_pool.py）
#       只认“块”：引用计数、LRU 空闲链表、块哈希 -> 块 的映射；
#   SingleTypeKVCacheManager  （single_type_kv_cache_manager.py）
#       一个 KV cache group 一个实例，管“这个 group 里请求占了哪些块”；
#   KVCacheCoordinator        （本文件）
#       一条请求会**横跨多个 group**（混合注意力模型：
#       full attention + sliding window + mamba 等多组并存），
#       需要一个角色把“分配多少块”“命中多长前缀”在所有 group 之间对齐。
#
# 为什么必须协调（这是理解本文件的关键）：
#   模型前向要求所有 group 的 num_cached_tokens 完全一致，
#   但各 group 的块大小、保留策略不同，能命中的长度也不同；
#   所以必须取“木桶短板”并对齐到统一边界。
#   本文件最核心的算法就是 HybridKVCacheCoordinator 里的**不动点迭代**。
#
# 三个实现，由末尾的 get_kv_cache_coordinator 选择：
#   KVCacheCoordinatorNoPrefixCache：不开前缀缓存，支持任意（含 0）个 group；
#   UnitaryKVCacheCoordinator      ：只有 1 个 group，直接转发；
#   HybridKVCacheCoordinator       ：多 group，需要对齐与收敛算法。

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NamedTuple

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    dcp_world_size_for_kv_cache_spec,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


# [CN] 校验 prefix_cache_retention_interval（前缀缓存的“保留间隔”）是否合法。
#
# 什么是 retention_interval：
#   对 sliding window / Mamba 这类“只保留最近一段”的 group，
#   缓存 checkpoint 是**稀疏**的：每隔 N 个 token 才存一份可复用检查点。
#   好处是少占大量块，代价是命中长度要向下对齐到 N 的倍数。
#     None = 稠密（每个边界都存）
#     0    = 只保留最新的一个 replay 边界
#     N>0  = 每隔 N 个 token 存一份
#
# 两条校验规则：
#   1) 模型里如果没有 SWA / Mamba group（清一色 full attention），
#      retention 根本不会生效，所以除了用户显式写 0 之外一律报错 ——
#      避免“配了参数却静默无效”这种最难查的问题；
#   2) N 必须是 scheduler_block_size 的倍数：命中只能落在调度块边界上，
#      不对齐的话这个间隔永远也命中不到。
def _validate_prefix_cache_retention_interval(
    retention_interval: int | None,
    scheduler_block_size: int,
    kv_cache_config: KVCacheConfig,
) -> None:
    if retention_interval is None:
        return

    # Retention sparsifies sliding-window and Mamba (linear-attention)
    # checkpoints; full-attention and chunked-local groups cache densely and
    # ignore it (their hit granularity must stay fine).
    if not any(
        isinstance(g.kv_cache_spec, (SlidingWindowSpec, MambaSpec))
        for g in kv_cache_config.kv_cache_groups
    ):
        if retention_interval == 0:
            return
        raise ValueError(
            "prefix_cache_retention_interval is set but this model has "
            "no sliding-window or Mamba KV cache group, so retention has no "
            "effect. Set it to 0 (it only applies to sliding-window and Mamba "
            "attention)."
        )

    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:
        raise ValueError(
            f"prefix_cache_retention_interval ({retention_interval}) "
            "must be non-negative and a multiple of scheduler_block_size "
            f"({scheduler_block_size})."
        )


# [CN] KV cache 协调器抽象基类。职责边界抓住这三条：
#   1) **聚合**：把请求在每个 group 上的操作（分配 / 缓存 / 释放）串起来；
#   2) **对齐**：决定所有 group 共同的“前缀命中长度”（find_longest_cache_hit）；
#   3) **转发**：其余操作原样转给对应的 SingleTypeKVCacheManager。
#
# 类属性 enable_partial_hash_hits：能否做**细粒度**哈希命中（比块更小的粒度）。
#   基类默认 False；只有含 Mamba align group 的混合模型才会在 Hybrid 版里置 True，
#   且一旦发现有 group 的 manager 不支持细粒度查找，就整体退回 False 并告警。
class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    enable_partial_hash_hits = False

    # [CN] 构造：按 kv_cache_config 里的每个 group 建一个 manager，全部共用同一个
    #      BlockPool（块是全局池化的，不同 group 只是“看待块的方式”不同）。
    #
    # 几个容易看漏的点：
    #   - scheduler_block_size 是所有 group 块大小的**公倍数**，也是调度最小粒度；
    #   - num_reprefillable_tokens = num_prefill_lookahead - 1：
    #     多模块 MTP（投机解码）下，最后这几个 token 可能被**重新 prefill**，
    #     因此它们不算“已确定的 KV”，不能被缓存（见 cache_blocks）；
    #   - 所有 manager 共享一个 block_pool，所以某个 group 分配块时可能淘汰
    #     另一个 group 的块 —— 这也是后面“两阶段分配”要防的坑。
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # The scheduling granularity (LCM of all group block sizes), must be a multiple
        # of the hash_block_size and the block size of each group.
        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size
        self.num_reprefillable_tokens = max(0, num_prefill_lookahead - 1)

        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # [CN] EAGLE / MTP：草稿模型会多算一个 lookahead token 并**写进 KV cache**，
        #      于是最后一个块被污染了，别人不能再命中它。
        #      这些 group 在查找前缀命中时必须“丢掉最后一个块”（drop_eagle_block）。
        #      若 use_eagle 为真却没有任何 group 被明确标记（模型没声明），
        #      就**保守地把所有 group 都标记上**：宁可少命中，也不能读到脏数据。
        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        # During chunked prefill with EAGLE, the single next prefill lookahead
        # token past the chunk boundary is combined with the final hidden state
        # and written to the KV cache. Therefore, the final chunk token must be
        # excluded from prefix cache hits to prevent requests from acquiring the
        # KV cache slot polluted with the next prefill token, which may or may not
        # be present after the matching prefix. The last-block drop handles this
        # edge case. During multi-module MTP, the issue generalizes to a prefill
        # lookahead of num_speculative_tokens, so the dropped tail must be large
        # enough to contain them. Hits land on scheduler-block boundaries (see
        # `_cache_hit_alignment_tokens`), so the excluded tail is
        # scheduler_block_size, not the group's own block size.
        if (
            enable_caching
            and self.eagle_group_ids
            and scheduler_block_size < num_prefill_lookahead
        ):
            raise ValueError(
                f"Multi-module MTP with prefix caching requires scheduler_block_size"
                f" (={scheduler_block_size}) >= num_speculative_tokens"
                f" (={num_prefill_lookahead})."
            )

        # [CN] 为每个 KV cache group 建一个 manager。
        #      注意 dcp/pcp 会改变 effective block size（沿序列切分/复制），
        #      所以 world size 要按 group 的 spec 分别算，不能一刀切。
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size_for_kv_cache_spec(
                    kv_cache_group.kv_cache_spec, dcp_world_size
                ),
                pcp_world_size=pcp_world_size,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )
        # Match Mamba checkpoints to Eagle's attention replay boundary.
        # [CN] Mamba 的 checkpoint 必须和 EAGLE 的 attention replay 边界对齐：
        #      目标模型“多看”了一个 token，Mamba 状态也要跟着多丢一个块，
        #      否则草稿模型和目标模型的起始位置会差一个 token。
        if use_eagle:
            for manager in self.single_type_managers:
                if isinstance(manager, MambaManager):
                    manager.drop_eagle_checkpoint_block = True

        # A positive retention interval must be a multiple of the base hit granularity
        # (``scheduler_block_size``) to land on real cache-hit boundaries.
        # 0 = keep only the latest replay boundary; None = dense;
        self.retention_interval = kv_cache_config.prefix_cache_retention_interval
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )

    # [CN] **预测**还需要多少块 —— 注意它不修改任何状态，纯粹给调度器做准入判断
    #      （“这个请求现在塞得进来吗？”）。
    #
    # CrossAttention 是特例：cross-attention 的 KV 大小只取决于 encoder 输入长度，
    # 与解码步数无关，所以是一锤子买卖 —— 按 num_encoder_tokens 静态算一次即可。
    #
    # apply_admission_cap 为什么要分两种取值：
    #   True  只在“整条序列准入检查”时开：允许把 SWA / chunked-local 的
    #        **边跑边回收**算进去（现在看着不够，跑起来会腾出来）；
    #   False 每步分配时必须关掉，否则预测值会和真正的 allocate_new_blocks
    #        对不上，导致调度器以为够、实际分配失败。
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
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
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, apply the recycling-aware
                per-request admission cap (SWA / chunked-local). Set only by
                the full-sequence admission gate; per-step allocation must
                leave it False so the predictor matches `allocate_new_blocks`.

        Returns:
            The number of blocks to allocate.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
        return num_blocks_to_allocate

    # [CN] 把“刚命中的前缀缓存块”挂到请求名下。**分两阶段**，这是真实 bug 修复（#33775）：
    #
    #   阶段 1：遍历所有 group，先 touch（抬高引用计数）本地命中的块；
    #   阶段 2：再为“外部已算 token”（external，例如 KV connector 从别的实例搬来的）
    #           分配新块。
    #
    # 为什么必须分两阶段：阶段 2 的 get_new_blocks 可能触发**淘汰**；
    # 如果此时阶段 1 还没把所有 group 的命中块都 touch 一遍，就可能出现
    # “group A 分配时，把 group B 刚命中但还没 touch 的块淘汰掉”，
    # 于是 B 拿到一个内容被换掉的块 —— 静默的错误输出。
    # 先全部 touch、再统一分配，就没有这个时间窗口。
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # A running request is already tracked in num_cached_block and won't
        # have new prefix-cache hits, so this is a no-op for it.
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # Two-phase allocation (issue #33775): first touch every group's local
        # cache-hit blocks, then allocate external blocks for every group. This
        # ensures an earlier group's external `get_new_blocks` cannot evict a
        # later group's not-yet-touched cache-hit blocks.
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens > 0:
            for manager in self.single_type_managers:
                manager.allocate_external_computed_blocks(
                    request_id,
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                )

    # [CN] 真正的分配：给请求在**每个 group** 上都补足到至少 num_tokens 个槽位。
    #      返回值是“每个 group 的新块列表”组成的元组，顺序与 kv_cache_groups 一致，
    #      上层（scheduler）会据此组装 block table 发给 worker。
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
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
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    # [CN] 把请求已算好的 KV 标记为“可复用”（注册进前缀缓存）。
    #      关键：只缓存 num_computed_tokens - num_reprefillable_tokens 个 token ——
    #      末尾那几个 token 在多模块 MTP 下可能被重算，
    #      把它们缓存下来等于缓存了会被覆盖的脏数据。
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            # Only cache tokens with finalized KV. The last num_reprefillable_tokens
            # tokens can be re-prefilled during multi-module MTP.
            num_tokens_to_cache = max(
                0, num_computed_tokens - self.num_reprefillable_tokens
            )
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )

    # [CN] 释放：把请求在所有 group 上的块归还给 block pool
    #      （引用计数减一，归零则回到空闲链表；带哈希的块会先进入 LRU 待淘汰区）。
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    # [CN] “摘账但不立即归还”：把请求的块从各 manager 的账本里移除并返回给调用方，
    #      由调用方决定什么时候真正归还。用于异步释放 / 批量释放。
    #
    #      **必须逆序归还**（文档里专门强调）：块是按顺序分配的，
    #      先淘汰尾部块才能保证剩余块仍是“一条连续前缀”，
    #      否则会出现中间被挖空的块序列，前缀缓存的语义就坏了。
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        Pop the request's bookkeeping from all single-type managers and
        return its blocks without returning them to the block pool. The
        caller must eventually pass the returned blocks to
        `block_pool.free_blocks`, freeing them in reverse order (so that
        tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.
        """
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks

    # [CN] 返回每个 group 上“所有在跑请求共享的前缀块数”。
    #      用途：某些 kernel / 优化（如统一预取、CUDA graph 的公共前缀处理）
    #      需要知道哪些块是所有请求都要读的。
    #      参数 running_request_id 只是“任取一个在跑的请求”作为参照。
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache for each kv cache group.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    # [CN] 对 sliding window 这类“只看最近 W 个 token”的 group，
    #      已经彻底滑出窗口的块就没用了 —— 这里释放它们，并把请求块表里
    #      对应位置换成 null_block（**保留占位**，避免块表长度/下标错位）。
    #      num_prompt_tokens 只有 R-SWA 用得上：prefill 尾部与 decode 窗口之间
    #      可能还夹着一段“空隙块”，需要单独判断回收。
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length. R-SWA managers use this to
                free gap blocks between the prefill tail and decode window; other
                manager types ignore it.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )

    # [CN] 取请求在每个 group 上的块表。这是发给 worker 组装 block table 的原料。
    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    # [CN] **本文件最重要的抽象方法**：给定 block_hashes，返回三元组
    #        (每个 group 命中的块, 命中长度, 未缓存公共前缀长度)。
    #      第三个返回值是稀疏保留（retention）场景下的补偿量：
    #      某 group 缓存得更长、另一 group 因为稀疏保留没存，
    #      差值就是“其实所有请求都算过、只是没存下来”的那段前缀。
    #      调度器可以据此少算一部分 prefill，同时又不会去读不存在的块。
    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """Returns the per-group hit blocks, the hit length, and the number of
        ``num_uncached_common_prefix_tokens`` (a shared prefix that a
        sparse-retention group has not cached yet; 0 unless hybrid)."""
        pass

    def new_step_starts(self) -> None:
        """Notify each manager that a new step is starting."""
        for manager in self.single_type_managers:
            manager.new_step_starts()


# [CN] 关闭前缀缓存时的实现：所有“查找命中”相关操作直接返回空。
#      它是唯一支持 **0 个 group** 的实现（纯 encoder / 无 KV 的模型走这条路）。
class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0, 0


# [CN] 只有一个 KV cache group 时的快路径：不需要任何对齐算法，
#      直接把请求转发给唯一的那个 manager，少一层循环与元组拆包。
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.dcp_world_size = self.single_type_managers[0].dcp_world_size
        self.pcp_world_size = pcp_world_size
        self.block_size = self.single_type_managers[0].block_size
        # For models using only Mamba, block_size is set to max_model_len when
        # prefix caching is disabled, and hash_block_size validation is skipped.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )
        # Single group; useless but just set ``use_eagle`` for consistency regardless.
        self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        # Single group: nothing "uncached common" -- no other group to lag it.
        return hit_blocks, hit_length, 0


# [CN] 把“使用同一个 KVCacheSpec 的多个 group”打包成一个查表单元。
#
# 为什么：相同 spec 的 group 块大小、哈希算法完全一致，命中结果必然相同，
# 因此只需查一次，再把结果复制给所有成员 —— 省掉重复的哈希扫描。
#
# use_eagle 放在**整组**而不是单个 group 上：
#   既然整组一起查、一起命中，那“要不要丢最后一块”也只能整组统一决定；
#   任一成员是 EAGLE group，整组就按 EAGLE 处理。
class SpecGroup(NamedTuple):
    """KV cache groups that share one spec, batched together for a single
    cache-hit lookup.

    ``use_eagle`` is True iff any member group is an EAGLE/MTP group. Members
    sharing a spec are cached and looked up jointly, so the EAGLE last-block drop
    is necessarily decided for the whole spec group.
    """

    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    use_eagle: bool


# [CN] **混合注意力模型**的协调器（full attention + sliding window + mamba …）。
#
# 核心难题：各 group 的“可命中长度”不同（块大小、保留策略不一样），
# 但模型前向要求所有 group 的 num_cached_tokens **必须相同**，
# 因此要把各 group 的命中长度收敛到一个公共值。
#
# 算法（find_longest_cache_hit）是**不动点迭代**：
#   先由第 1 个 group 得到上界 L1，第 2 个 group 在 <= L1 范围内查得 L2，
#   若 L2 < L1（有人投了反对票）就带着 L2 重头再查一轮……
#   L 单调递减且有下界 0，所以必然收敛。
#
# 优化：把 full attention 排在最前面查。它是“向下封闭”的
#   （命中 N 个 token 就一定命中 N-1 个），能给出最紧的初始上界，
#   显著减少后续迭代轮数；而且第二轮起只需截断、不必重扫哈希。
class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        self.dcp_world_size = dcp_world_size
        # Only groups that participate in prefix caching must satisfy the
        # divisibility constraint; groups that opt out (e.g. GLM-5.3-Flash kpool
        # tail, block_size=kpool) are scratch buffers and excluded.
        group_block_sizes = [
            manager.block_size
            for manager, group in zip(
                self.single_type_managers, kv_cache_config.kv_cache_groups
            )
            if group.kv_cache_spec.prefix_cacheable
        ]
        assert all(
            block_size % hash_block_size == 0 for block_size in group_block_sizes
        ), (
            "Each KV cache group's real block_size must be divisible by "
            f"hash_block_size. block_sizes={group_block_sizes}, "
            f"hash_block_size={hash_block_size}"
        )
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        if dcp_world_size > 1:
            # DCP shards full-attention KV across ranks and replicates Mamba
            # state; other spec types (e.g. sliding window) have no DCP-aware
            # handling yet, so reject them explicitly.
            for g in kv_cache_config.kv_cache_groups:
                assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (
                    "DCP with hybrid KV cache layouts only supports "
                    "full-attention and Mamba groups, got: "
                    f"{type(g.kv_cache_spec).__name__}."
                )
        # Fine-grained hash hits require Mamba "align" and compatible cache
        # managers in every group. TP needs hashing finer than the Mamba block;
        # DCP accepts equality because it scales the effective full-attention
        # block instead.
        has_partial_mamba_group = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            and g.kv_cache_spec.mamba_cache_mode == "align"
            and (
                (dcp_world_size == 1 and g.kv_cache_spec.block_size > hash_block_size)
                or (
                    dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size
                )
            )
            for g in kv_cache_config.kv_cache_groups
        )
        self.enable_partial_hash_hits = has_partial_mamba_group
        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager, group in zip(
                    self.single_type_managers, kv_cache_config.kv_cache_groups
                )
                if group.kv_cache_spec.prefix_cacheable
                and not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
            if unsupported_partial_hit_managers:
                self.enable_partial_hash_hits = False
                logger.warning_once(
                    "Disabling fine-grained prefix-cache hits because these KV "
                    "cache managers require block-aligned lookups: %s.",
                    ", ".join(sorted(unsupported_partial_hit_managers)),
                )
        cache_hit_alignment_tokens = self._cache_hit_alignment_tokens
        for manager in self.single_type_managers:
            manager.cache_hit_alignment_tokens = cache_hit_alignment_tokens
        self.verify_and_split_kv_cache_groups()

    # [CN] 命中长度必须向下对齐到哪个刻度：
    #        开启细粒度命中（Mamba align）-> hash_block_size（更细）；
    #        否则                        -> scheduler_block_size。
    #      对齐的意义：让“命中时读的边界”和“缓存时写的边界”是同一套刻度，
    #      否则会出现“明明算过 100 个 token，却只能命中 96 个”的浪费。
    @property
    def _cache_hit_alignment_tokens(self) -> int:
        # Fine-grained partial hits may return hash-block-aligned lengths;
        # otherwise it must stay scheduler-block-aligned.
        return (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.scheduler_block_size
        )

    # [CN] 按 spec 把 group 归并成 SpecGroup 列表（相同 spec 的合并为一个）。
    #      另外做三件事：
    #        1) 跳过 prefix_cacheable=False 的 group（例如某些模型的 kpool 尾巴，
    #           那是每请求私有的临时缓冲，不可共享，也不能参与命中查找）；
    #        2) 把 full attention 排到最前（理由见类注释）；
    #        3) 记录 full_attention_group_id —— 作为“稠密参照组”，
    #           若某 group 报告了比它还长的命中，说明各 group 的命中
    #           不在同一个边界上，是不一致的状态（issue #46453）。
    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        self.attention_groups: list[SpecGroup] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            # Skip groups that opt out of prefix caching (e.g. GLM-5.3-Flash
            # kpool tail): their blocks are per-request scratch, never
            # shareable, so they must not participate in hit lookup (their
            # manager-level hooks already no-op). Their slot in the per-group
            # hit tuple stays empty.
            if not g.kv_cache_spec.prefix_cacheable:
                continue
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            use_eagle = i in self.eagle_group_ids

            # Try to find an existing group with the same spec
            for idx, group in enumerate(self.attention_groups):
                if group.spec == spec:
                    assert manager_cls is group.manager_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group.group_ids.append(i)
                    if use_eagle and not group.use_eagle:
                        self.attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                self.attention_groups.append(
                    SpecGroup(spec, [i], manager_cls, use_eagle)
                )

        assert self.attention_groups, (
            "HybridKVCacheCoordinator requires at least one cacheable group."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups.sort(
            key=lambda g: not isinstance(g.spec, FullAttentionSpec)
        )

        # Dense reference group for per-group lookups (None when the model
        # has no full-attention layers): full attention is downward-closed,
        # so any group reporting a longer per-group hit implies the union of
        # per-group hits is not consistent at a single boundary (#46453).
        first = self.attention_groups[0]
        self.full_attention_group_id: int | None = (
            first.group_ids[0] if isinstance(first.spec, FullAttentionSpec) else None
        )

        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True

    # [CN] 向下取整到“未来可能被命中的边界”。
    #      注意开启细粒度命中时**完全不取整** —— 哪怕取整到 hash_block_size，
    #      也会把 Mamba 那段已被“私有化”的尾巴重新登记进缓存，
    #      从而被别的请求错误命中。
    def _align_cacheable(self, num_tokens: int) -> int:
        """Largest prefix of ``num_tokens`` a future cache hit could match.

        Hits are ``scheduler_block_size``-aligned (see
        ``find_longest_cache_hit``) unless fine-grained partial hash hits are
        enabled, in which case no rounding applies -- rounding even to
        ``hash_block_size`` would re-register a privatized Mamba tail.
        """
        if self.enable_partial_hash_hits:
            return num_tokens
        return round_down(num_tokens, self.scheduler_block_size)

    # [CN] 混合版重写：先按 _align_cacheable 对齐，再给 EAGLE group 多补一块。
    #      原因：EAGLE group 查找时会“多匹配一块、然后丢掉”，
    #      所以缓存时也要多写一块（+ manager.block_size），那块才有机会被命中。
    #      同时仍然要扣掉 num_reprefillable_tokens，避免缓存会被重算的 token。
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        cached_num_computed_tokens = self._align_cacheable(num_computed_tokens)
        for manager in self.single_type_managers:
            num_tokens_to_cache = cached_num_computed_tokens
            # EAGLE groups match one block past each aligned boundary and drop
            # it, so make that lookahead block eligible to be cached.
            if manager.use_eagle and cached_num_computed_tokens > 0:
                # Only cache tokens with finalized KV. The last
                # num_reprefillable_tokens tokens can be re-prefilled during
                # multi-module MTP.
                num_finalized_computed_tokens = max(
                    0, num_computed_tokens - self.num_reprefillable_tokens
                )
                cached_num_finalized_computed_tokens = self._align_cacheable(
                    num_finalized_computed_tokens
                )
                num_tokens_to_cache = min(
                    num_finalized_computed_tokens,
                    cached_num_finalized_computed_tokens + manager.block_size,
                )
            # The manager already knows the fine hit granularity
            # (``scheduler_block_size``); retention is passed separately so it
            # can keep both the coarse segment tails and the fine replay
            # boundary (which needs the fine value).
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )

    # [CN] 不动点迭代求“所有 group 共同的命中长度”。主体逻辑见下面行内注释。
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the reconciled (combined) cache hit.
                - ``num_uncached_common_prefix_tokens``: a shared prefix that a
                  sparse-retention group has not cached yet (0 unless hybrid).
        """

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        # [CN] 主循环：每轮按当前上界把所有 group 查一遍；
        #      只要本轮结果比上轮短（有 group 投了反对票），就再走一轮。
        #      is_simple_hybrid（恰好 1 个 full attn + 1 个其他）可以只跑一轮：
        #      full attn 先给出上界，另一个 group 收敛一次即可结束。
        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
                self.attention_groups
            ):
                first_group_id = group_ids[0]
                # DCP/PCP shard each block's KV across ranks, so the manager's
                # effective block size may exceed the spec's.
                group_block_size = self.single_type_managers[first_group_id].block_size
                cached_blocks = hit_blocks_by_group[first_group_id]
                # [CN] full attention 的“向下封闭”优化：已经查过一次之后，
                #      后续轮次**不需要重新扫哈希**，直接把已有结果截断到新长度。
                #      这是让迭代变便宜的关键一步（否则每轮都是全量哈希扫描）。
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                # [CN] EAGLE 的“丢最后一块”对**同一个候选长度只能做一次**（#32802）：
                #      若在同一长度上重复丢，会一截一截越丢越短。
                #      因此用 eagle_verified 记下已校验过的 group；
                #      一旦长度变短，之前的校验全部作废（见下面的 clear()）。
                drop_eagle_block = use_eagle and idx not in eagle_verified

                # [CN] 给 EAGLE group 多一个块（eagle_margin）的查找空间：
                #      它先匹配到“候选长度 + 1 块”，再丢掉那块，正好落回候选长度。
                #      Mamba 不给这个 margin：mamba 的查找器从不丢块，
                #      给了反而会让命中长度变长，破坏收敛。
                _max_length = curr_hit_length
                # Eagle matches one extra drop unit (one hash unit for
                # fine-grained managers, else one cache block) and then drops
                # it, landing back at the candidate length. No margin for
                # mamba: its finder never drops (draft models have no mamba
                # layers), so the hit would grow past the candidate.
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size
                        else group_block_size
                    )
                    _max_length = min(
                        curr_hit_length + eagle_margin, max_cache_hit_length
                    )
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=self.single_type_managers[
                        first_group_id
                    ].dcp_world_size,
                    pcp_world_size=self.single_type_managers[
                        first_group_id
                    ].pcp_world_size,
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate every full-attention group (target and draft) blocks
        # to final hit_length.
        # [CN] 收敛之后：把所有 full attention group（含 target 与 draft 两组）
        #      的块表统一截断到最终命中长度。
        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            group_block_size = self.single_type_managers[group.group_ids[0]].block_size
            num_blocks = cdiv(hit_length, group_block_size)
            for group_id in group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        # Uncached shared prefix detection: if any attn. group cached a longer
        # prefix than the reconciled hit, it is an uncached common prefix across
        # requests that a sparse-retention group hasn't cached yet.
        # [CN] “理应命中、但因稀疏保留没存下来”的公共前缀长度。
        #      调度器拿它把这部分 token 计入 cached 从而少做一次 prefill，
        #      同时又不会真的去读那些不存在的块。
        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens

    # [CN] 诊断 / 统计用：每个 group **各自独立**查命中，不做对齐收敛。
    #      返回各 group 自己的长度，用来观察“到底是谁拖了后腿”。
    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        """Like find_longest_cache_hit but evaluates each group independently.

        Returns:
            (blocks_per_group, hit_lengths_per_group)
        """

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups

        for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
            manager = self.single_type_managers[group_ids[0]]
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=use_eagle,
                alignment_tokens=self._cache_hit_alignment_tokens,
                dcp_world_size=manager.dcp_world_size,
                pcp_world_size=manager.pcp_world_size,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit

        return tuple(hit_blocks), tuple(hit_lengths)


# [CN] 工厂函数：按 (是否开前缀缓存, group 数量) 三选一。
#       关闭缓存 -> NoPrefixCache（支持 0 个或多个 group）
#       1 个 group -> Unitary（快路径）
#       多个 group -> Hybrid（需要对齐收敛算法）
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
    num_prefill_lookahead: int = 0,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
        num_prefill_lookahead=num_prefill_lookahead,
    )
