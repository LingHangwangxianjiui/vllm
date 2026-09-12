# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：多模态 **encoder 输出** 的缓存管理器。
#
# 它管理的不是 KV cache，而是“视觉/音频编码器的输出 embedding”。
# 为什么需要缓存：同一张图可能出现在多条请求里（尤其是多轮对话中，
# 图片被反复携带），每次都重跑一遍 vision encoder 非常贵。
#
# 三个与 KV cache 不同的设计点：
#   1) 缓存粒度是**多模态 item**（一张图），由 mm_hash 标识，不是 token；
#   2) 容量按 **encoder embedding 数** 计，与文本 token 无关；
#   3) 淘汰是**分配时触发**的（can_allocate 里需要空间才淘汰），
#      而不是后台 LRU 线程。
#
# 记账三件套：
#   num_free_slots     —— 完全空闲的容量；
#   num_freeable_slots —— “空闲 + 可回收”（引用计数为 0 的条目也算可用）；
#   freeable           —— 按插入顺序排的 OrderedDict，实现 FIFO 淘汰。

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.logger import init_logger
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Cache is enabled to share embeddings of same multimodal data
    item (identified by their hash value) between different requests,
    and eviction takes place at allocation time when there's no free
    space for new embeddings.
    Oldest cached embeddings with no request referenced will be first evicted.

    NOTE: The EncoderCacheManager operates on the level of multimodal embeddings
    instead of encoder tokens (i.e. all tokens that represent the multimodal data
    in the input sequence). This means all break/text tokens in-between multimodal
    embeddings are not considered with respect to the cache size and the number
    of free slots.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    encoder embeddings from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder embeddings.
        num_free_slots: Current available cache capacity in encoder embeddings.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder embeddings).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_encoder_embeds) representing entries
            whose no current running request is needed and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
    """

    # [CN] 工厂方法：给平台/子类一个替换实现的入口（见文件末尾的 enc-dec 子类）。
    @classmethod
    def create_manager(
        cls, *, cache_size: int, vllm_config: "VllmConfig"
    ) -> "EncoderCacheManager":
        return cls(cache_size=cache_size)

    # [CN] 构造。注意 num_freeable_slots 初始等于 cache_size —— 
    #      此时没有任何已缓存内容，所谓“可回收”就是全部容量。
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}
        # request_id => set of input_ids cached for that request
        self.request_cached_ids: dict[str, set[int]] = {}

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    # [CN] 全量清空。**权重更新后必须调用**：
    #      旧权重算出的 embedding 还在缓存里，不清的话请求会读到过期结果。
    #      这是 RL / 在线权重更新场景里最容易漏掉的一步。
    def reset(self) -> None:
        """Reset the encoder cache to its initial state.

        This clears all cached encoder outputs and resets capacity tracking.
        Called when model weights are updated to invalidate stale embeddings.
        """
        self.cached.clear()
        self.request_cached_ids.clear()
        self.freeable.clear()
        self.freed.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size

    # [CN] 查询“这个多模态 item 的 encoder 输出是否已缓存”，命中则登记引用。
    #
    #      关键分支：条目存在但引用集为空（说明它已被“逻辑释放”、
    #      只是还没被真正淘汰）—— 此时要把它从 freeable 里取回来，
    #      并相应减少 num_freeable_slots。这就是“复用待淘汰条目”的机制。
    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        If the encoder output is cached, update `cached` to add the request id
        to the set of request ids that reference the cached encoder output.
        If the encoder output was previously not referenced by any request,
        update `freeable` and `num_freeable_slots` accordingly.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        self.request_cached_ids.setdefault(request.request_id, set()).add(input_id)
        return True

    # [CN] 准入检查 + 就地淘汰。返回值 True 表示“空间够了（必要时已淘汰）”。
    #
    #      两级容量判断：
    #        num_free_slots    够 -> 直接返回 True（不动任何状态）；
    #        num_freeable_slots 也不够 -> 返回 False（真的塞不下）；
    #        中间情况 -> 从 freeable 头部（最老）开始淘汰，直到够。
    #
    #      两个容易混淆的点：
    #        1) 这里**只改记账**，不释放显存；
    #           真正的释放要等 scheduler 把 freed 列表交给 worker；
    #        2) encoder_compute_budget 是“本步最多能算多少 encoder token”的
    #           预算，和缓存容量是两个独立维度 —— 都可能成为瓶颈。
    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """Check if there's sufficient cache space for a multimodal input.
        If there is, return True and update EncoderCacheManager state.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_freeable_slots`, entries will be
        evicted from `freeable` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True.
        Older entries are evicted first.

        Returns False only if the requested number of tokens exceeds both
        the free and reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder embeddings allowed to be
                computed when this method is invoked.
            num_embeds_to_schedule: Number of encoder embeddings already scheduled to be
                allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder
        output but only the state of EncoderCacheManager.
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # Not enough compute budget
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # Enough free slots
        if num_embeds <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_embeds > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
        return True

    # [CN] 正式占位（在 can_allocate 返回 True 之后调用）。
    #      同时扣减 num_free_slots 与 num_freeable_slots：
    #      因为刚占用的这部分**暂时不可回收**（有请求在引用它）。
    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.request_cached_ids.setdefault(request_id, set()).add(input_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    # [CN] 该请求当前持有哪些多模态 item 的缓存引用。
    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request."""
        return self.request_cached_ids.get(request.request_id, set())

    # [CN] 释放**一个**引用。注意它只做“逻辑释放”：
    #      引用集空了就放进 freeable、增加 num_freeable_slots，
    #      但物理显存要等到 can_allocate 真的需要空间时才回收。
    #
    #      末尾那个 any(...) 检查很值得注意：
    #      cached 记录的是“引用它的**请求**”，不是“出现次数”。
    #      如果同一个请求里这张图出现了多次（多轮对话把图带上），
    #      只要还有一处没释放，就不能让条目变成可淘汰 —— 
    #      否则请求后面还会用到它，encoder 就白重算一次。
    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freeable` and `num_freeable_slots` is
        increased by the number of encoder embeddings for that input.

        The entry is NOT physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # Always clean up request_cached_ids, even if the mm_hash was
        # already evicted from cache (e.g. by can_allocate).
        if req_id in self.request_cached_ids:
            self.request_cached_ids[req_id].discard(input_id)
            if not self.request_cached_ids[req_id]:
                del self.request_cached_ids[req_id]
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        # `cached` counts referencing requests, not positions, so one request
        # that repeats an item (an image carried across conversation turns) has
        # a single reference covering every occurrence. Hold it until the last
        # occurrence is freed: dropping it at the first makes the entry
        # evictable while the request still needs it, and the encoder then
        # recomputes an item it already has.
        if any(
            request.mm_features[other_id].identifier == mm_hash
            for other_id in self.request_cached_ids.get(req_id, ())
        ):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    # [CN] 释放请求持有的**全部**引用（请求结束/取消/abort 时调用）。
    #      同样只是逻辑释放，数据留在内存里等下次分配时再淘汰。
    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.

        For each cached input ID, `free_encoder_input` is invoked.
        The data stays in memory until eviction is triggered by a future
        attempt allocation called by 'can_allocate'.

        Typically called when a request is finished, cancelled, or aborted.
        """
        for input_id in list(self.get_cached_input_ids(request)):
            self.free_encoder_input(request, input_id)

    # [CN] 取出并清空“本轮真正被淘汰的 mm_hash 列表”，交给 worker 释放显存。
    #
    #      末尾的过滤（mm_hash not in self.cached）很关键：
    #      同一次调度 pass 里，一个刚被淘汰的条目**可能又被重新分配**了，
    #      这种情况不能让 worker 去释放 —— 否则刚写进去的内容就没了。
    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call to be used by the scheduler to notify workers about which
            encoder outputs can be removed from their caches. The internal
            list is cleared after this call.
        """
        # An entry evicted early in the scheduling pass can be allocated again
        # later in the same pass. Keep its worker-side tensor in that case.
        freed = [mm_hash for mm_hash in self.freed if mm_hash not in self.cached]
        self.freed = []
        return freed

    def get_manager_metadata(self) -> EncoderCacheManagerMetadata | None:
        return None


# [CN] 计算 encoder 的两个预算（单位都是“输入序列里的 token 数”）：
#       compute budget —— 单步最多能算多少 encoder token；
#       cache size    —— encoder 缓存总容量。
#
#       两者都取 max(配置值, 单个 item 的最大 token 数)：
#       因为哪怕配置得很小，也至少要装得下**一个**最大的多模态 item，
#       否则这条请求永远无法调度（死锁）。
#
#       另外：若禁用了 chunked mm input，就必须保证一个 item 能一步算完，
#       否则直接报错而不是运行时卡死。
def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        mm_max_toks_per_item: The maximum number of tokens per item for each
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """

    if not mm_max_toks_per_item:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size


# NOTE (NickLucche): Temporary implementation for encoder-decoder models that only
# use the manager for scheduling purposes. Encoder-decoder models will eventually
# utilize the cache and this class will fold into EncoderCacheManager, as
# differences with MM models shrink.
# [CN] 编码器-解码器模型（如 Whisper）的临时实现：
#      这类模型的 encoder 输出**不跨请求复用**，所以只借用调度框架，
#      不做真正的缓存（check_and_update_cache 恒返回 False）。
#
#      它也因此需要一套“延迟一拍”的释放机制（见 get_freed_mm_hashes）：
#      本步分配的条目要等**下一步**才能释放，
#      因为真正的释放在 runner 里发生在“模型执行之前”，
#      立刻释放会把本步还要用的东西删掉。
class EncoderDecoderCacheManager(EncoderCacheManager):
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.allocated: list[str] = []
        self.to_free: list[str] = []

    def reset(self) -> None:
        """Reset the encoder cache to its initial state."""
        self.num_free_slots = self.cache_size
        self.allocated.clear()
        self.to_free.clear()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        return False

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        # Not enough compute budget
        if num_encoder_embeds > encoder_compute_budget:
            return False

        num_encoder_embeds += num_embeds_to_schedule
        # Enough free slots
        return num_encoder_embeds <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots -= num_encoder_embeds

        mm_hash = request.mm_features[input_id].identifier
        self.allocated.append(mm_hash)

    def free(self, request: Request) -> None:
        for input_id in range(len(request.mm_features)):
            self.free_encoder_input(request, input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return set(range(len(request.mm_features)))

    # [CN] 延迟一拍的释放：返回上一步的 allocated，把本步的存起来下次再还。
    #      这样 worker 侧总在“模型执行之前”释放，且释放的一定是
    #      已经用过一轮的条目。
    def get_freed_mm_hashes(self) -> list[str]:
        # As encoder cache is not used for enc-dec models, we can free the entries here
        # The actual free happens in the runner, *before* the model is executed.
        # Therefore, `freeable` acts as a buffer to free the entries only after the
        # model is executed, mimicking the state transition of `EncoderCacheManager`.
        to_free = self.to_free
        self.to_free = self.allocated
        self.allocated = []
        return to_free

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots += num_encoder_embeds
