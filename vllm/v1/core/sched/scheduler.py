# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：V1 引擎的**调度中枢** —— Scheduler。
#
#     它运行在 EngineCore 进程里，每一拍做两件事：
#       1) schedule()           —— 决定「这一拍算哪些请求、各算多少 token」，
#                                  产出 SchedulerOutput 交给模型执行器；
#       2) update_from_output() —— 取回模型执行器的输出，推进／结束／回滚请求，
#                                  产出 EngineCoreOutputs 回给前端。
#
#     ========================= 核心心智模型 =========================
#     vLLM V1 **没有**独立的 prefill 阶段和 decode 阶段。每个请求只有
#     两个计数：
#         num_tokens          —— 它现在「应该」算到第几个 token；
#         num_computed_tokens —— 它「实际」已经算到第几个 token。
#     调度器每一拍做的事，就是给请求分配 token，让 num_computed_tokens
#     去追上 num_tokens。chunked prefill、前缀缓存、投机解码，在这个
#     模型下都只是「追的方式不同」，因此不需要特判。
#
#     ========================= 三步预算 ============================
#       token_budget —— max_num_scheduled_tokens，一拍最多算多少 token；
#       input_budget —— max_num_batched_tokens，输入张量能装多少（还要
#                       预留 draft_slots 给投机解码的草稿位）；
#       块数         —— KV cache 空闲块，不够就抢占。
#
#     ========================= 调度顺序 ============================
#       先 RUNNING（保证已在跑的请求优先拿到显存，避免抖动），
#       后 WAITING（且只有本拍**没发生抢占**时才调度新请求）。
#
#     ========================= 本文件的难点 ========================
#     1) **异步调度 / 流水并行**：schedule() 与 update_from_output() 在
#        时间上重叠，本拍的 schedule 可能在上上拍的输出回来之前就跑完了。
#        于是所有「释放」都必须延后：deferred_frees 围栏队列、
#        _re_block_ids 块号快照、num_in_flight_tokens 与
#        num_stale_output_tokens 计数，都是为此而存在。
#     2) **KV Connector**：P/D 分离、offloading、远程前缀命中。请求可能
#        处于 WAITING_FOR_REMOTE_KVS，块由连接器异步填充，失败还要回滚重算。
#     3) **Mamba 对齐**：Mamba 递推状态只写在块边界上，chunk 不能随便切。
#
#     代码组织：__init__ → 三个 prefill 切分/查询小工具 → schedule() 主体
#     → 输出构造 → update_from_output() 主体 → 请求生命周期 → 统计与重置
#     → 文件末尾的 KV Connector 专区。

import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    KVConnectorBlockState,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


# [CN] Scheduler 是 SchedulerInterface 的默认实现。
#      AsyncScheduler（async_scheduler.py）继承它并覆写少量钩子，
#      用于「异步调度」：在 GPU 还在跑上一拍时就把下一拍算出来。
class Scheduler(SchedulerInterface):
    # [CN] 构造。参数虽然多，但可以分成六组来看：
    #      ① 配置（vllm_config 及其子配置）；② 调度预算；
    #      ③ 连接器（KV / EC）；④ 请求容器（waiting / running / skipped）；
    #      ⑤ 缓存管理器（KV cache、encoder cache）；
    #      ⑥ 各种「异步安全」的记账字段（in-flight、stale、围栏序号）。
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # [CN] 把会反复用到的子配置直接挂成属性，避免在热路径上做属性链查找。
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.model_uses_mrope = vllm_config.model_config.uses_mrope
        self.model_uses_xdrope = vllm_config.model_config.uses_xdrope
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.spec_decode_metrics_level = (
            self.observability_config.per_request_spec_decode_metrics
        )
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        self.is_mm_encoder_only = vllm_config.is_mm_encoder_only

        # [CN] 多 Engine 场景下，需要按「客户端 index」分组汇报已结束的请求 id，
        #      这样前端不用遍历全部请求就能追踪请求生命周期。
        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # [CN] 上一拍调度过的请求 id。仅 MRV1（v1 model runner）需要：
        #      增量同步持久 batch 时用它判断「这个请求是不是新进 batch 的」。
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # [CN] 调度预算三兄弟。max_num_scheduled_tokens 缺省回退到
        #      max_num_batched_tokens：老配置里两者是同一个东西。
        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        # [CN] 位置上限（不是 KV 容量上限）。注意它和 KV cache 能装多少块
        #      是两回事：前者是模型结构硬限制，后者是显存软限制。
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # [CN] 扩散模型一个去噪步不采样 token，所以要写成 0，
        #      否则下面「num_new_tokens 至少要留 1 个位置给采样」的算术会错。
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )

        # [CN] 创建 Scheduler 侧的 KV Connector。注意每个 Worker 上还有一个
        #      Role=WORKER 的同伴，两者通过 SchedulerOutput 里的 metadata 通信。
        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        # [CN] 释放块时是否走「延后释放」（见 _free_request_blocks）。
        self.defer_block_free = False
        # [CN] 该连接器是否要求「KV 必须送达」：抢占时若 KV 交接还没完成，
        #      就必须丢弃在途输出而不能交付，否则会拿到没有 KV 支撑的 token。
        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        self.requires_kv_delivery = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            # [CN] KV 加载失败的策略：recompute（回滚重算）还是 fail（直接报错）。
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # [CN] 为什么需要延后释放：异步调度 / PP 下，某个 step 可能还在往
            #      「已释放请求」的块里写 KV。consumer 侧连接器随后可能把这些块
            #      重新分配并通过 load 填充，而这次 load 与那次写没有顺序保证。
            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

            self.requires_kv_delivery = self.connector.requires_kv_delivery

        # [CN] KV cache 事件发布器（块存储/淘汰事件），供外部做前缀缓存观测。
        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        # [CN] EC Connector：encoder 输出的远程传输（与 KV 相对，传的是
        #      多模态 embedding 而非 KV）。
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        # [CN] block_size 是调度器视角的块大小；dcp / pcp 是上下文并行度，
        #      两者都会影响「一个块实际覆盖多少 token」。
        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # [CN] 全量请求表（req_id -> Request）。注意它**包含**已经 finished
        #      但块还没真正释放的请求（连接器延迟释放的情况）。
        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # [CN] 调度策略：FCFS（默认）或 PRIORITY。
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # [CN] 三个请求容器：
        #      waiting         —— 正常等待调度；
        #      skipped_waiting—— 因为异步依赖/约束本拍被跳过的（下一拍优先重试）；
        #      running        —— 已在跑（list 而非队列，因为要按索引抢占）。
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # [CN] 上一拍到本拍之间结束的请求 id，用于通知 worker 清理缓存状态。
        #      每拍结束会清空（见 _update_after_schedule）。
        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # [CN] 自上次 schedule() 以来被抢占的请求 id，随 SchedulerOutput
        #      发给 worker，让它们把这些请求从持久 batch 里摘掉。
        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # [CN] 处于 WAITING_FOR_STREAMING_REQ 的请求数：它们不在 running 里，
        #      但仍占着 model runner 的请求槽位，所以算 num_running 时要加回来。
        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # [CN] 异步 KV 接收中：finished = 收完了可以恢复调度；
        #      failed = 收失败（部分块无效），恢复时要回滚 num_computed_tokens。
        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # [CN] grammar 编译失败的请求，在 update_from_output 里统一按错误结束。
        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        self.grammar_compile_error_reqs: set[str] = set()

        # [CN] 多模态相关：只有模型支持多模态输入时才计算 encoder 预算。
        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # [CN] 纯文本的 encoder-decoder 模型（如 bart）被「伪装」成多模态模型
        #      实现，因此最多只允许一种模态。
        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        # [CN] encoder 计算预算：一拍最多跑多少 encoder embedding。
        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        # [CN] 允许通过 ec_manager_config 注入自定义 encoder cache 管理器；
        #      否则 encoder-decoder 用 EncoderDecoderCacheManager，其余用普通版。
        manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
        if manager_cls_obj is None:
            manager_cls_obj = (
                EncoderDecoderCacheManager
                if self.is_encoder_decoder
                else EncoderCacheManager
            )
        self.encoder_cache_manager = manager_cls_obj.create_manager(
            cache_size=encoder_cache_size, vllm_config=vllm_config
        )
        # [CN] 投机解码配置解析。这一组字段决定后面几处「要不要多留几个位置」。
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.use_eagle_block_drop = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
        # [CN] **num_prefill_lookahead**：prefill 进行中，drafter 会往前偷看几个
        #      已知的 prefill token 当草稿输入。EAGLE 系偷看 1 个，多模块 MTP
        #      偷看 num_spec_tokens 个。它同时决定：encoder 调度窗口的偏移、
        #      encoder 的延迟释放、KV manager 的可重 prefill 窗口（这个值减 1），
        #      以及 chunk 末尾要预留多少 token。
        # Positions past the computed tokens that the drafter reads mid-prefill.
        # Eagle-family drafters read 1 ahead, but multi-module MTP reads
        # num_spec_tokens ahead at chunked-prefill boundaries. Determines the
        # encoder scheduling shift, the deferred encoder free, the KV cache
        # manager's re-prefillable window (this minus 1), and how many tokens to
        # reserve between a chunk boundary and the prefill end.
        self.num_prefill_lookahead = 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            # [CN] 只有 EAGLE 系才有 lookahead 的概念；普通投机解码为 0。
            self.use_eagle = speculative_config.use_eagle()
            if self.use_eagle:
                self.num_prefill_lookahead = (
                    self.num_spec_tokens
                    if speculative_config.use_multi_module_mtp()
                    else 1
                )
            # [CN] EAGLE 的「丢尾块」：命中前缀缓存时故意丢掉最后一块，
            #      强制重算以获得隐藏状态给 drafter 用。
            self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()
            if self.use_eagle and not self.use_eagle_block_drop:
                logger.warning(
                    "EAGLE trailing prefix-cache block dropping is disabled. "
                    "This is experimental and may affect speculative-token "
                    "acceptance rates."
                )

        # [CN] 创建 KV cache 管理器（阶段 3 已经详细读过）。
        #      注意 pcp_world_size 传 1 —— 调度器侧按「整块」记账，
        #      PCP 的切分是 worker 侧块表的事。
        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.hash_block_size = hash_block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle_block_drop,
            num_prefill_lookahead=self.num_prefill_lookahead,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
        )
        # [CN] 把 GPU block pool 交给连接器：连接器要能直接看到块池才能做
        #      远程 load 的块分配。必须等 kv_cache_manager 建好后才能绑。
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        # [CN] 流水并行 / V2 runner 标记。V2 runner 走增量 batch 同步路径，
        #      很多「是否要携带全量 token ids」的分支都靠 use_v2_model_runner。
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # [CN] 调度迭代计数。V2 + PP + 异步下用它实现「同一请求两次 decode
        #      之间必须间隔 pp_size 拍」的节奏控制。
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # [CN] DP 之间的 prefill 均衡：记录上一个「节奏对齐」的 prefill 批次
        #      是否把 waiting 队列抽干了。抽干了说明我们不是被容量卡住，
        #      下一拍就不必再节流 prefill。
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        # [CN] 是否要求「整个 ISL 一次性装下」才准入（否则不分块）。
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        # [CN] 是否有 Mamba 层 / 是否需要把新块清零。
        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        # [CN] 本拍会被异步 KV load 覆写的块：清零动作会与这次带外写竞争，
        #      所以要从清零列表里剔除。
        # Blocks that async KV loads will overwrite this step, skipped from
        # zeroing since the zeroing could race the out-of-band write.
        self._skip_zero_block_ids: set[int] = set()
        # [CN] Mamba cache 的 align 模式：状态只在块边界落盘，
        #      因此 prefill chunk 必须在块边界收尾（见 _mamba_block_aligned_split）。
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # [CN] 目前只取第一个 MambaSpec 的对齐要求，多规格模型的支持还是 TODO。
        # TODO: Support models with multiple Mamba specs that require different
        # prefill checkpoint alignments instead of selecting the first one.
        self.mamba_prefill_checkpoint_alignment = next(
            (
                group.kv_cache_spec.prefill_checkpoint_alignment
                for group in kv_cache_config.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            ),
            None,
        )
        # [CN] 是否所有 Mamba group 都配置了 prefill checkpoint 块。
        self.mamba_has_prefill_checkpoint_blocks = self.has_mamba_layers and all(
            not isinstance(group.kv_cache_spec, MambaSpec)
            or group.kv_cache_spec.num_prefill_checkpoint_blocks > 0
            for group in kv_cache_config.kv_cache_groups
        )
        # [CN] hash_block_size < block_size 时可以做「细粒度」前缀命中，
        #      Mamba 的尾部部分命中条目只能由「恰好停在 prompt 最后一个 hash
        #      边界」的那一拍来登记，所以切分时要额外加这个停靠点。
        # A finer prefix_match_unit is configured: a mamba partial tail entry
        # can only be registered by a step ending exactly at the prompt's last
        # hash boundary, so the split adds that stop.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )

        # [CN] 非空 step 的序号：schedule 侧 +1，update_from_output 侧也 +1，
        #      因为后者是按 FIFO 顺序每个被调度的 step 调用一次，两者保持同步。
        #      空 step（0 token）不推进序号 —— 它没有 GPU 写，不构成围栏。
        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # [CN] 延后释放队列：(围栏序号, 块列表)。
        #      只有 processed_step_seq >= 围栏序号时，这些块才真正安全可还。
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        # [CN] MFU 等性能指标采集器（可选）。
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        # [CN] 是否需要把 MoE 的路由专家（routed experts）回传给前端。
        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )
        self.return_sampling_mask = vllm_config.model_config.return_sampling_mask

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            # [CN] 路由专家管理器：按 slot 维护一张 CPU 侧表，
            #      worker 每拍 D2H 拷回路由结果后写进这里。
            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # [CN] 调度时刻拍下的块号快照。为什么要快照：异步调度下，
            #      update_from_output 运行时可能已经发生过新的 schedule() 并抢占了
            #      请求、释放了它的块；快照保证读 slot 数据时块号仍然有效。
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        # [CN] 暂停状态：PAUSED_ALL 完全不调度，PAUSED_NEW 只跑存量。
        self._pause_state: PauseState = PauseState.UNPAUSED

        # [CN] 仍在 prefill 中的在途请求（分块 prefill 未完 + 异步 KV 加载中）。
        #      它们「未来还需要多少块」被用来给异步加载做准入控制，防死锁。
        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

    # [CN] 把本拍想算的 prefill token 数**裁剪**到「Mamba 状态可以落盘的位置」。
    #
    #      为什么必须裁剪：Mamba 是递推状态，不像 attention 那样可以把任意
    #      位置的 KV 追加进去。在 align 模式下，可复用的 SSM 状态只在
    #      **块边界**被物化。如果 chunk 停在一个非边界位置，那这个位置的
    #      状态就没地方存，下次想复用前缀时只能从头重算。
    #
    #      除了块边界，还有三个「必须停」的位置：
    #        · prompt 最后一个 hash 边界（细粒度前缀命中要登记尾部条目）；
    #        · 共享前缀的接合点（Marconi 优化，让兄弟请求能复用）；
    #        · 内部 checkpoint 位置。
    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" cache mode reusable SSM states are materialized at block
        boundaries, plus mandatory early stops (the prompt's partial-tail hash
        boundary, a detected shared-prefix junction). If a block is larger
        than the configured prefill chunk limit, intermediate chunks keep
        private running state until they reach the next cacheable position.
        """
        # [CN] 本 chunk 的起始绝对位置：已算 + 本拍本地命中 + 外部（远程）命中。
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # [CN] prefill 的终点。用 max(prompt_tokens, num_tokens - 1) 是为了兼容
        #      「恢复执行」的请求 —— 它们需要把已生成的 output token 也重放一遍。
        #      已经越过终点就无需切分。
        # Split only during prefill: `request.num_tokens - 1` extends this to
        # resumed requests replaying their output tokens.
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens

        # [CN] 最后一个「状态可被缓存」的块对齐位置。
        #      EAGLE 会砍掉最后一个匹配块去重算，所以这里也后退一块，
        #      否则 Mamba 侧会命中不到。
        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle_block_drop:
            last_cache_position = max(last_cache_position - block_size, 0)

        end = start + num_new_tokens
        # [CN] 内部 checkpoint：允许在 prompt 内部（而非只有末尾）额外落盘一份状态，
        #      缓解「只有一个块边界可存」带来的重算开销。
        checkpoint_position = get_mamba_prefill_checkpoint_position(
            prefill_end,
            self.hash_block_size,
            drop_eagle_block=self.use_eagle_block_drop,
        )
        # [CN] 内部 checkpoint 只在「本 chunk 就打到 prefill 终点」且位置合法时启用。
        use_internal_checkpoint = (
            self.mamba_has_prefill_checkpoint_blocks
            and end >= prefill_end
            and is_mamba_prefill_checkpoint_valid(
                query_start=start,
                query_end=end,
                checkpoint_position=checkpoint_position,
                hash_block_size=self.hash_block_size,
                mamba_block_size=block_size,
                checkpoint_alignment=self.mamba_prefill_checkpoint_alignment,
            )
        )
        # [CN] 一旦走内部 checkpoint，边界就不受「最后可缓存位置」约束了。
        if use_internal_checkpoint:
            last_cache_position = 0
        # [CN] 核心不变式：slot p 里存的是「恰好算完 (p+1)*block_size 个 token
        #      之后」的状态。状态只在 chunk 末尾写，所以 chunk 末尾必须块对齐。
        #      豁免：prompt 的最后一个 chunk，它的 slot 由 decode 推进到边界。
        # Invariant: slot p holds the state after exactly (p + 1) * block_size
        # tokens. State is written at chunk ends, so chunk ends must be block
        # aligned. Exempt: the prompt's last chunk, whose slot decode advances
        # to the boundary. A block too wide for one chunk advances sub-block
        # and re-aligns at the next boundary.
        # [CN] 中间 chunk（还没打到 prefill 终点）强制向下对齐到块边界。
        #      若一个块比单 chunk 上限还宽（block_size > max_prefill_tokens），
        #      则允许 chunk 在块内部前进，等下一个边界再对齐。
        if end < prefill_end:
            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        # [CN] 收集所有「必须停」的候选位置：
        #      ① 下一个块边界（chunk 起点在块中间时必须停，不能跨过去）；
        #      ② 最后一个可缓存块边界（不能越过它）；
        #      ③ prompt 的尾部 hash 边界（细粒度命中登记点）；
        #      ④ 共享前缀接合点（向下取块对齐，子块位置的状态无法单独缓存）。
        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit and not use_internal_checkpoint
            else 0
        )
        stops = (
            # Same invariant: a chunk starting mid-block stops at the boundary
            # rather than running past it.
            next_block_boundary
            if start % block_size != 0 and not use_internal_checkpoint
            else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # Fine-grained hits: the prompt's partial-tail entry can only be
            # registered by a chunk ending exactly at its last hash boundary.
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            # Marconi shared-prefix junction, block-floored (a sub-block
            # junction's state is not separately cacheable): cache its state
            # so sibling requests sharing the prefix can reuse it.
            # [CN] Marconi 共享前缀：把接合点的状态也存下来，
            #      这样共享同一前缀的兄弟请求可以直接复用。
            start + (request.shared_prefix_boundary - start) // block_size * block_size
            if start < request.shared_prefix_boundary < end
            else 0,
        )
        # [CN] 取落在 chunk 内部的**最早**一个强制停靠点作为新的 chunk 末尾。
        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)

    # [CN] 查询本地前缀缓存命中。返回 (块, 本地命中 token 数, 共享前缀边界,
    #      是否「发散命中」)。
    #
    #      发散命中（hit_diverged）：混合注意力模型下各 group 的命中长度不一致，
    #      coordinator 通过不动点迭代给出一个「不保证所有 group 都成立」的更优
    #      命中。只有支持 divergent 的连接器才能消费它。
    def _get_local_prefix_cache_hit(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        connector = self.connector
        # [CN] 连接器能处理各 group 不一致的命中，走专门的查询路径。
        if connector is not None and connector.supports_divergent_local_hybrid_hits:
            return self.kv_cache_manager.get_computed_blocks_for_connector(request)

        blocks, num_local, shared_prefix_boundary = (
            self.kv_cache_manager.get_computed_blocks(request)
        )
        return blocks, num_local, shared_prefix_boundary, False

    # [CN] 不允许 prefill chunk 停在「距 prefill 终点不足 lookahead」的地方。
    #
    #      为什么：在 chunk 边界上，多模块 MTP 的 drafter 会把后面
    #      num_prefill_lookahead 个已知 prefill token 当草稿输入。如果边界离终点
    #      太近，drafter 只能退化成用采样结果当草稿，会永久污染尾部模块的 KV。
    #      所以要么这拍把 prefill 做完，要么给下一拍至少留 lookahead 个 token。
    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        """Never end a prefill chunk within num_prefill_lookahead of the
        prefill end.

        At a chunked-prefill boundary, the multi-module MTP drafter consumes
        the next num_prefill_lookahead known prefill tokens as draft inputs. A
        boundary closer to the end than that would make it fall back to
        sampled drafts, permanently polluting the trailing modules' KV caches.
        Either finish the prefill or leave at least num_prefill_lookahead for
        the next chunk. No-op for eagle-family drafters (lookahead 1).
        """
        # [CN] remaining = 本拍算完后还剩多少 prefill token。
        #      落在 (0, lookahead) 区间说明卡在危险窗口里，回退本拍的 token 数。
        remaining = request.num_tokens - num_computed_tokens - num_new_tokens
        if 0 < remaining < self.num_prefill_lookahead:
            num_new_tokens -= self.num_prefill_lookahead - remaining
        return max(num_new_tokens, 0)

    # [CN] ===================== 调度主循环（本文件的心脏） =====================
    #
    #      流程：
    #        ┌─ 阶段 A：RUNNING 请求（while req_index < len(running)）
    #        │    · 跳过各种「本拍不该算」的情况；
    #        │    · num_new_tokens = num_tokens_with_spec + placeholders - computed；
    #        │    · 依次被 token_budget / input_budget / max_model_len /
    #        │      Mamba 对齐 / encoder 预算 / lookahead 窗口裁剪；
    #        │    · allocate_slots 拿块；拿不到就**抢占**最低优先级请求重试。
    #        └─ 阶段 B：WAITING 请求（仅当本拍没发生抢占且未暂停）
    #             · 查本地前缀缓存 + 连接器远程命中；
    #             · 处理异步 KV 加载、encoder 预算、spec decode padding；
    #             · allocate_slots 成功才真正入场。
    #      最后：构造 SchedulerOutput + 更新调度后状态。
    #
    #      注意：**抢占只发生在阶段 A**。一旦本拍抢了别人，
    #      就不会再接收新请求（否则新请求会把刚腾出来的块又抢走）。
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # [CN] 步计数 +1。V2+PP+异步用它对齐 worker 侧采样广播的节奏。
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        # [CN] 四类收集容器：新请求 / 恢复（被抢占后重来）/ 存量 running / 被抢占。
        #      它们最终会被打包进 SchedulerOutput 交给 worker。
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # [CN] req_to_new_blocks：本拍给每个请求新分配的块（增量）；
        #      num_scheduled_tokens：每个请求本拍算多少 token（核心产出）。
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # [CN] 投机解码下，每个被调度的请求还要额外占 draft_slots 个输入位，
        #      所以 input_budget 每次扣减都要连带扣掉它们。
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        # [CN] 全暂停：预算直接归零，下面两个 while 都进不去。
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # [CN] encoder 侧预算：本拍还能算多少 encoder embedding。
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # [CN] 本拍每个请求实际带上的草稿 token（会被写进 SchedulerOutput）。
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # [CN] prefill_scheduled：本批是否含 prefill —— spec decode 的 padding
        #      以及 DP 均衡都依赖它；
        #      has_sync_kv_loads：是否有「同步」远程 KV 加载（worker 需要知道）。
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False
        # Whether any scheduled request has a synchronous connector KV load.
        has_sync_kv_loads = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        # [CN] 通知 KV cache 管理器「新的一拍开始了」：清空本拍累计的
        #      CoW 副本、新块 id、边界状态交接等临时状态。
        self.kv_cache_manager.new_step_starts()

        # [CN] DP prefill 均衡：在被节流（非节奏对齐）的那一拍，
        #      把 prefill 计算全部推迟 —— 除非已经饱和（prefill_capacity_bound）。
        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # [CN] ---------------- 阶段 A：先调度 RUNNING ----------------
        #      为什么先 running：它们的 KV 已经在显存里，放着不管会白白浪费
        #      已投入的显存；而且新请求进来可能把 running 挤到抢占。
        # First, schedule the RUNNING requests.
        req_index = 0
        # [CN] 注意这里用 while + req_index 而不是 for：抢占会修改 self.running，
        #      需要手动维护游标（见下面的 victim_index 修正）。
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # [CN] 连一个草稿位都塞不下了，直接停。
            if input_budget <= draft_slots:
                break

            # [CN] 异步调度下的「少算一拍」优化：上一拍其实已经把 max_tokens 算满了，
            #      再排一拍纯属浪费（而且会破坏 uniform decode 的优化）。
            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            # [CN] V2 + PP + 异步：同一个请求的两次 decode 之间必须间隔 pp_size 拍，
            #      才能对上 worker 侧「采样 token 广播」的环形槽位节奏。
            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            # [CN] DP 均衡：把进行中的 prefill chunk 推到节奏对齐的那一拍，
            #      decode 照跑，用它们填满本拍。
            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            # [CN] EC Connector：这个请求依赖的 encoder 输出还没到位，本拍跳过。
            if (
                self.ec_connector is not None
                and request.mm_features
                and not self.ec_connector.ensure_cache_available(
                    request,
                    request.num_computed_tokens - request.num_output_placeholders,
                )
            ):
                req_index += 1
                continue

            # [CN] 本拍要算的 token 数 = 目标长度 - 已算长度。
            #      num_tokens_with_spec 含草稿位，placeholders 也要追上。
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            # [CN] 长 prefill 阈值：把超长 prompt 切成更小的块，
            #      避免单个请求长期霸占整拍预算（缓解尾延迟）。
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # [CN] 位置不能越过 max_model_len。投机解码下尤其必要：
            #      草稿位也占位置，而且本拍还要留 num_sampled_tokens_per_step 个位置采样。
            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # [CN] Mamba 对齐要**先于** encoder 裁剪做：先确定 chunk 边界，
            #      再按这个边界决定能调度哪些 encoder 输入。
            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            # [CN] 多模态：按当前 chunk 窗口挑出需要本拍计算的 encoder 输入，
            #      可能反过来把 num_new_tokens 裁短（详见 _try_schedule_encoder_inputs）。
            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )

            # [CN] 多模块 MTP：不要把 chunk 边界落在「距终点不足 lookahead」的危险窗口。
            # Multi-module MTP: avoid ending a prefill chunk within
            # num_prefill_lookahead of the prefill end.
            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )

            # [CN] num_new_tokens 被裁成 0 的五种原因（见下面英文注释）。
            #      这里用 continue 而不是 break：strict FCFS 会在第一个请求卡住时
            #      让整拍空转，所以允许低优先级请求插队，提高利用率。
            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # 5. Insufficient budget to keep a multi-module MTP prefill
                #    chunk out of the prefill-lookahead window.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # [CN] 向 KV cache 管理器要块。这是唯一可能触发抢占的地方。
            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                # [CN] 拿不到块就抢占一个最低优先级的请求，然后重试。
                #      循环出口：拿到块（break）或把自己也抢掉了（break）。
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # [CN] 抢占谁？PRIORITY 策略下取「优先级最低、到达最晚」的；
                    #      FCFS 下取 running 末尾那个（后进先出）。
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        # [CN] 记录被抢占者在 running 里的下标。
                        # Record the index of the preemption victim to
                        # maintain accurate loop state.
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        # Decrement the loop cursor if the removed request
                        # preceded the current iteration, preventing the
                        # silent omission of the subsequent request.
                        if victim_index < req_index:
                            req_index -= 1

                        # [CN] 如果受害者**本拍已经被调度过**了，要把它刚才占掉的预算、
                        #      encoder 预算、块、草稿 token 全部吐回来，否则本拍账目就错了。
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            token_budget += restored
                            input_budget += restored + draft_slots
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                    # [CN] FCFS：直接把最后一个（最近入场的）请求挤出去。
                    else:
                        preempted_req = self.running.pop()

                    # [CN] 真正执行抢占：释放块、回 waiting 队列、标记 stale 输出。
                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    preempted_reqs.append(preempted_req)
                    # [CN] 把自己都抢掉了 —— 说明实在没块了，放弃本请求。
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            # [CN] 注意这里是 break 而不是 continue：块不够时继续遍历后面的
            #      running 请求也没有意义（它们同样拿不到块），还会白白抢占。
            if new_blocks is None:
                # Cannot schedule this request.
                break

            # [CN] 正式把请求计入本拍：登记块、扣预算。
            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            req_index += 1

            # [CN] 投机解码：算出本拍真正带上了几个草稿 token（可能被 chunk 截断），
            #      记录后清空 —— 下一拍的草稿由 update_draft_token_ids 重新填。
            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # [CN] encoder 缓存分配：注意 external_load 的那些也要占位，
            #      但它们不需要本地计算预算。
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # [CN] LoRA：统计本拍用到的 adapter 数，不能超过 max_loras。
        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # [CN] ---------------- 阶段 B：再调度 WAITING ----------------
        # Next, schedule the WAITING requests.
        # [CN] 两个准入条件：本拍没有发生过抢占（块已经很紧了），且未暂停。
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # [CN] 本拍被跳过的请求先放在临时队列，循环结束后**插到队首**，
            #      保证它们下一拍优先重试（避免饿死）。
            step_skipped_waiting = create_request_queue(self.policy)

            # [CN] 循环条件同时看 waiting 和 skipped_waiting 两个队列。
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if input_budget <= draft_slots:
                    break
                # [CN] 流式输入暂停中的会话虽然不在 running 里，但仍占着
                #      model runner 的请求槽位，所以算 running 数时要加回来。
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break

                # [CN] 选一个队列的队首：FCFS 优先 skipped（重试优先）；
                #      PRIORITY 则比较两个队列的队首谁更该先。
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # [CN] 被阻塞的请求（等 grammar / 等远程 KV / 等流式输入）先尝试提升状态；
                #      提升不了就挪到 skipped 队列，本拍不再看它。
                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # [CN] 还有「在途的 stale 输出」没有被排空：现在恢复可能会重复采样
                #      某个位置（该位置的输出稍后才被送达）。它在流水线深度内会自然排空。
                if (
                    request.num_stale_output_tokens > 0
                    and not request.drop_stale_output
                ):
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # [CN] LoRA 上限检查：再加一个新 adapter 就会超限时跳过。
                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # [CN] 远程（外部）命中 token 数与「是否异步加载」标志。
                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                did_prefix_cache_lookup = False

                # [CN] 只有全新请求（num_computed_tokens == 0）才查前缀缓存；
                #      恢复执行的请求走下面的 else 分支。
                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    did_prefix_cache_lookup = True
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        request.shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request)

                    # [CN] 有连接器时再查一次远程命中。
                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        # [CN] 给连接器看的是**块对齐**的本地命中数：
                        #      这样「更长的远程命中」才能干净地盖掉本地那个不足一块的尾巴，
                        #      而不必对共享的尾块做写时复制（CoW）。
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = (
                            num_new_local_computed_tokens - partial_tail
                        )
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, block_aligned_local
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        # [CN] 远程命中严格超过本地完整命中：丢掉本地的不足一块尾巴，
                        #      让远程 load 从头覆盖，省掉一次 CoW。
                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            new_computed_blocks = (
                                self.kv_cache_manager.truncate_computed_blocks(
                                    new_computed_blocks, block_aligned_local
                                )
                            )
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        # [CN] 本地有不足一块的尾巴，但远程没超过它：保留本地尾巴，不加载远程。
                        #      同时清掉 load_kv_async —— 下面有 assert 要求异步加载必须有外部 token。
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            load_kv_async = False
                        # [CN] 本地命中恰好块对齐：外部 token 直接采用。
                        else:
                            num_external_computed_tokens = ext_tokens

                        # [CN] 发散命中（各 group 不一致）且没有外部 token 支撑：
                        #      更深的那个本地命中的恢复边界上没有合法的 Mamba 状态，
                        #      退回到「所有 group 都认同」的那个边界。
                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(request)

                        # [CN] 连接器前缀缓存的命中统计（仅在真的查过时才记）。
                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # [CN] 总命中 = 本地 + 远程。
                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # [CN] 需要的 encoder 输出还在远程传输中，本拍跳过。
                    # Skip request with pending mm encoding prefetches
                    if self._ec_transfer_pending(request, num_computed_tokens):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    # [CN] 只在「首次 prefill」时记录 prefill 统计，
                    #      抢占后重来的那次不算（会污染指标）。
                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats and request.num_preemptions <= 0:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                # [CN] 两种情况会走到这里：
                #      ① KVTransfer：异步接收完成后 num_computed_tokens 已经 > 0；
                #      ② 流式输入会话恢复，带着上一块新增的媒体。
                #      两者都还要过一遍 encoder 传输检查。
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed. A streaming-input
                    # session resumes here too, carrying whatever media its
                    # latest chunk added, so this branch needs the same gate.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                    if self._ec_transfer_pending(request, num_computed_tokens):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                # [CN] 本拍要调度的 encoder 输入及其预算快照。
                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False

                # [CN] 异步加载中：本拍不做前向，num_new_tokens = 0，
                #      但块要先占住（给远程写入用）。
                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # [CN] DP 均衡：把本地 prefill 推到节奏对齐的那一拍（远程加载不受影响）。
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    # [CN] 本请求能拿到的预算 = min(全局 token 预算, 输入预算 - 草稿位)。
                    request_token_budget = min(token_budget, input_budget - draft_slots)
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # [CN] **spec decode padding**：把「只算 1 个 token」的新 decode 请求
                    #      补齐成 1 + num_spec_tokens，这样本拍能走完整的 cudagraph。
                    #      扩散模型不能补（草稿位无法填充），本批已有 prefill 时也不补。
                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    # Not for diffusion where draft tokens can't be padded.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and self.num_sampled_tokens_per_step > 0
                        and num_new_tokens == 1
                        and not prefill_scheduled
                        and (scheduled_running_reqs or num_computed_tokens > 0)
                    ):
                        padded_num_tokens = 1 + self.num_spec_tokens
                        # Pad only when there is room for the sampled token(s).
                        if (
                            num_computed_tokens
                            + padded_num_tokens
                            + self.num_sampled_tokens_per_step
                            <= self.max_model_len
                        ):
                            if padded_num_tokens > request_token_budget:
                                # Prefer to not schedule than schedule un-padded.
                                break
                            num_new_tokens = padded_num_tokens
                            pad_spec_decode = True

                    # [CN] 长 prefill 阈值（与阶段 A 同款裁剪）。
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # [CN] 未开启 chunked prefill 且这个请求一次装不下：直接停。
                    #      （pooling 请求默认不允许分块，必须显式开启。）
                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > request_token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    # [CN] 最终裁剪到本请求预算内。
                    num_new_tokens = min(num_new_tokens, request_token_budget)
                    assert num_new_tokens > 0

                    # [CN] Mamba 对齐。这里要多传两个命中数，因为对齐是相对「绝对位置」算的。
                    # Apply Mamba alignment before encoder caps.
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        # [CN] 对齐后一个 token 都排不上：说明预算小到连一个对齐块都装不下，
                        #      后面的请求只会更小，直接 break。
                        if num_new_tokens == 0:
                            break
                        # [CN] 对齐把 padding 出来的占位行裁掉了 —— 但那些行是投机位置而非
                        #      prefill token。补齐的请求必须保留完整的 1 + num_spec 行，
                        #      否则 sampler 的行数会和 query 行数对不上；所以宁可去掉补齐。
                        if (
                            pad_spec_decode
                            and num_new_tokens != 1 + self.num_spec_tokens
                        ):
                            # Alignment clipped the placeholder rows. The split
                            # aligns prefill chunks, but the padded tail rows are
                            # speculative positions, not prefill tokens. A padded
                            # request must keep all 1 + num_spec rows or the
                            # sampler's row count stops matching its query rows,
                            # so drop the padding instead of shortening it.
                            num_new_tokens = 1
                            pad_spec_decode = False

                    # [CN] 与阶段 A 相同：按 chunk 窗口挑 encoder 输入。
                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=self.num_prefill_lookahead,
                        )

                    # [CN] 多模块 MTP 的 lookahead 窗口保护。
                    # Multi-module MTP: avoid ending a prefill chunk within
                    # num_prefill_lookahead of the prefill end.
                    num_new_tokens = self._reserve_prefill_lookahead(
                        request, num_computed_tokens, num_new_tokens
                    )

                    # [CN] 裁成 0 就退出新请求调度循环。
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

                # [CN] 异步加载时不分配 lookahead 槽：前向还没跑，
                #      本地和远程的块数此刻并不一致，提前分配会对不上。
                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # [CN] encoder-decoder 要额外为 cross-attention 分配块，
                #      数量取决于本拍要算的 encoder embedding 数。
                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                # [CN] 异步加载会**长时间**占着块而且不能被抢占，
                #      所以准入时必须把其他在途 prefill 的预留也算进去，
                #      否则可能互相等待形成死锁。
                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

                # [CN] 分配块。参数比阶段 A 多：
                #      new_computed_blocks    —— 前缀缓存命中的块（直接复用）；
                #      delay_cache_blocks     —— 异步加载时先不把块登记进缓存；
                #      full_sequence_must_fit —— 要求整条序列一次装下；
                #      reserved_blocks        —— 其他在途请求已预留的块数。
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                # [CN] 分配失败：注意要先把请求从 encoder cache manager 上「摘干净」
                #      （free），否则刚才试探性占用的 encoder 条目会一直挂着。
                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                # [CN] 告诉连接器分配结果，由它决定是否真的需要发起 load。
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                # [CN] 前缀缓存命中统计：只在「真的做过查询」的请求上记，
                #      否则那些被跳过、根本没查的请求会被算成 0 命中。
                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )

                # [CN] 出队 —— 到这里才真正确认本请求本拍会被处理。
                request = request_queue.pop_request()
                # [CN] 异步加载：状态改成 WAITING_FOR_REMOTE_KVS，放进 skipped 队列，
                #      等连接器把 KV 搬完再提升回 WAITING（见 _try_promote_blocked_waiting_request）。
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    # [CN] 先把 num_computed_tokens 设成「期望值」，虽然 KV 还没真正到位。
                    #      在传输完成前它不会被用到；若连接器报错，
                    #      _update_requests_with_invalid_blocks 会把它改回正确值。
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    # [CN] 异步加载要覆写的块从清零列表里剔除 —— 清零会和这次带外写竞争。
                    if self.needs_kv_cache_zeroing:
                        # Skip zeroing of the blocks the async load will
                        # overwrite; the zeroing could race the write.
                        self._skip_zero_block_ids.update(
                            self.kv_cache_manager.get_zeroing_block_ids_in_range(
                                request.request_id,
                                num_new_local_computed_tokens,
                                num_computed_tokens,
                            )
                        )
                    continue

                # [CN] 同步路径：正式进入 running。
                self.running.append(request)
                # [CN] 有外部 token 且是同步加载 —— worker 需要在本拍阻塞等待。
                if num_external_computed_tokens > 0:
                    # load_kv_async is False here
                    has_sync_kv_loads = True
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                # [CN] 按「进来时的状态」分类：全新 → new，被抢占后恢复 → resumed。
                #      两者的差别是 resumed 请求在 worker 侧的持久 batch 里已经有记录。
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                # [CN] 注意这里取的是**全量**块（get_blocks），而不是增量 new_blocks：
                #      cached request 需要完整的块表。
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # [CN] 补齐出来的草稿位是 -1（占位），真正的草稿由 drafter 稍后填。
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
                # [CN] 只把「本拍算完还在 prefill」的请求记在在途集合里，
                #      用于给后续异步加载算预留。
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # [CN] 本拍被跳过的请求插到 skipped 队首（比更早被跳过的优先）。
            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # [CN] 只有「放行 prefill」的那一拍才更新 prefill_capacity_bound：
            #      waiting 还有剩 → 说明是被容量卡住，下一拍继续节流。
            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)

        # [CN] 一批不变式断言。注意最后一个是 <=：running 里的请求
        #      本拍未必都会被调度到。
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert input_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # [CN] 最长公共前缀块数，供 cascade attention 用（可省掉重复计算）。
        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # [CN] ---------------- 构造 SchedulerOutput ----------------
        # Construct the scheduler output.
        # [CN] V2 runner 走增量同步，resumed 请求也要带上全量 token ids，
        #      所以这里把两者合并。
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                    uses_mrope=self.model_uses_mrope,
                    uses_xdrope=self.model_uses_xdrope,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    uses_mrope=self.model_uses_mrope,
                    uses_xdrope=self.model_uses_xdrope,
                )
                for req in scheduled_new_reqs
            ]

        # [CN] 打包 cached request 数据（增量块表、新 token、输出计数等）。
        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # [CN] 记录本拍调度过的请求 id，供下一拍做「增量 batch 同步」判断。
        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # [CN] Mamba align 模式下的边界状态必须带**精确块号**交给连接器 —— 
        #      它无法从连接器那张「只追加」的块表里重建。每拍都取走，防止堆积。
        # Mamba "align" boundary states must be handed off with exact block ids;
        # they cannot be reconstructed from a connector's append-only block
        # table. Drained every step so stale offers cannot accumulate.
        boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()

        # [CN] 给连接器一份块号快照：新请求 + 本拍有新块的 cached 请求 +
        #      有边界状态交接的请求。
        kv_connector_block_state = None
        if self.connector is not None:
            snapshot_req_ids = {req.req_id for req in new_reqs_data}
            snapshot_req_ids.update(
                req_id
                for req_id, block_ids in zip(
                    cached_reqs_data.req_ids,
                    cached_reqs_data.new_block_ids,
                    strict=True,
                )
                if block_ids
            )
            snapshot_req_ids.update(
                req_id for req_id in boundary_state_offloads if req_id in self.requests
            )
            kv_connector_block_state = KVConnectorBlockState(
                block_ids={
                    req_id: self.kv_cache_manager.get_block_ids(req_id)
                    for req_id in snapshot_req_ids
                },
                boundary_state_offloads=boundary_state_offloads,
            )

        # [CN] CoW（写时复制）产生的块拷贝：部分命中共享尾块时，
        #      命中方要把尾块复制到自己的私有块才能写。
        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )
        # [CN] 拷贝动作随本拍执行一起跑。它对应的「已处理序号」是
        #      sched_step_seq + 1（0 token 的空拍不推进序号），
        #      因此用这个序号做围栏来延后释放被保留的源块。
        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # [CN] 动态投机解码：按本拍 batch 大小查表决定本拍用几个草稿 token。
        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        # [CN] 仅在开启迭代级明细日志时才统计 encoder 输入开销。
        scheduled_encoder_input_stats = None
        if (
            self.log_stats
            and self.observability_config.enable_logging_iteration_details
        ):
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
                scheduled_encoder_inputs
            )

        # [CN] 组装 SchedulerOutput。几个容易忽略的字段：
        #      finished_req_ids —— 是「两拍之间结束的」存量状态，不是本拍新产出的；
        #      free_encoder_mm_hashes —— 通知 worker 可以释放的 encoder 缓存条目；
        #      new_block_ids_to_zero —— 新块需要先清零（脏显存复用）。
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            has_sync_kv_loads=has_sync_kv_loads,
            kv_cache_block_copies=pending_kv_cache_block_copies,
            kv_connector_block_state=kv_connector_block_state,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
        )

        # [CN] 让连接器做三件事：规划 KV 存储、把 load/save 打包成不透明对象、
        #      清理自己的内部状态。
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # [CN] EC Connector 的同款 metadata（传的是 encoder embedding）。
        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # [CN] 块状态只给连接器看，不能下发到 worker。
        # Connector-only block state must not be dispatched to workers.
        scheduler_output.kv_connector_block_state = None

        # [CN] 只给「非空拍」推进围栏序号：空拍没有 GPU 写，不构成围栏。
        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        # [CN] 调度后状态推进（详见 _update_after_schedule）。
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    # [CN] 单独抽出来是为了让子类（如 AsyncScheduler）能覆写 metadata 的构造方式。
    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    # [CN] 取出本拍新分配的块 id，交给 worker 清零。
    #      不需要清零的模型直接返回 None，省掉一次传输。
    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        if not self.needs_kv_cache_zeroing:
            return None

        # [CN] 剔除「异步 KV 加载即将覆写」的块：清零会和那次带外写竞争。
        if self._skip_zero_block_ids:
            skip = self._skip_zero_block_ids
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]
            skip.clear()

        return new_block_ids_to_zero or None

    # [CN] 抢占一个请求并放回 waiting 队列。
    #      注意：调用方负责把它从 running 里摘掉（不同策略摘的对象不同）。
    #
    #      drop_stale_output 的语义：在途输出是「丢弃」还是「照常交付」。
    #      reset_prefix_cache 必须用丢弃 —— 它会在同一步内恢复该请求，
    #      照常交付会导致 token 乱序。
    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        # [CN] 释放块（可能延后，见 _free_request_blocks）与 encoder 缓存。
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        # [CN] 状态归零：下次从 0 开始重算（抢占不做部分保留）。
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # [CN] 异步调度下，把全部在途输出标记为 stale。
        #      这些 token 仍然会被交付（丢掉会扰动投机解码的接受率统计），
        #      但不能再改动已经重置的计数器；每拍排空自己那一份。
        #      注意是**赋值**而不是累加：num_in_flight_tokens 本身已经包含
        #      尚未排空的 stale 份额。
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # [CN] 放回 waiting **队首**（被抢占的请求优先恢复），
        #      并登记到 reset_preempted_req_ids 通知 worker 摘掉它。
        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    # [CN] 为什么「推进 num_computed_tokens」要放在构造完 SchedulerOutput 之后：
    #      ① 输出里必须带的是**本拍原始的** num_scheduled_tokens，
    #         worker 要靠它算 input_ids；
    #      ② 先推进，下一个 schedule() 就能立刻继续 prefill 这个请求；
    #      ③ 若稍后草稿 token 被拒绝，update_from_output 会把它调回来。
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # [CN] 推进已算 token 与在途 token 计数。
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            # [CN] 更新「是否仍在分块 prefill」标记 —— 它决定很多后续分支
            #      （structured output、spec decode padding、DP 均衡都看它）。
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            # [CN] 告诉 worker 本批是否有需要 grammar bitmask 的请求。
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # [CN] 给路由专家拍块号快照：并发的 schedule() 可能抢占比请求并释放块，
        #      快照能活过那次抢占。
        #      用 update 而不是赋值：异步调度下可能在上一次 update_from_output
        #      还没消费时又调了一次这里，要保留上一拍尚未消费的条目。
        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # [CN] 注意不能 clear() —— scheduler_output 持有同一个对象引用。
        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    # [CN] 流式输入：把下一块输入接进同一个「会话请求」。
    #      注意会**丢掉**上一块最后采样出的那个 token（当前语义：
    #      只保留已经「算进 KV」的输出 token）。
    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # [CN] 截断 output token，只留已计算进 KV 的部分。
        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        # [CN] 新块带来的多模态特征，position offset 要整体后移。
        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        # [CN] 追加新 chunk 的 token，重算块哈希（前缀缓存依赖它）。
        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        # [CN] 从「等流式输入」状态回到可调度状态。
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    # [CN] 打包 cached（已在跑 / 被抢占后恢复）请求的数据。
    #      与 NewRequestData 的区别：只传**增量**（新块、新 token），
    #      全量 token ids 只在必要时才带。
    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        # [CN] running 与 resumed 拼在一起遍历，靠下标区分。
        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # [CN] PP + 非异步：首stage worker 与末stage worker 之间没有直连，
            #      所以调度器要把采样出的 token 回传。
            #      PP + 异步则走 GPU 直连广播，不用带这份 payload。
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            # [CN] 后半段就是 resumed 请求（需要 worker 特殊处理持久 batch）。
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            # [CN] 上一拍没调度过的请求，worker 侧没有它的 token 历史，
            #      需要补一份全量 all_token_ids。
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            # [CN] allow_none：没有新块时传 None（worker 端据此跳过块表更新）。
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    # [CN] 决定本拍调度哪些 encoder 输入，并**反过来裁剪** num_new_tokens。
    #
    #      裁剪的原因：多模态 encoder 通常用双向注意力，一个 item 必须
    #      **整体**算完。如果本拍的窗口只能盖住某个 mm item 的一部分，
    #      那就把 num_new_tokens 缩到「刚好停在这个 item 之前」，
    #      让它下一拍整体算。
    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        # [CN] 没有 token 可算或不含 encoder 输入 —— 直接返回空。
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # [CN] 调度器是「按请求」工作的，但记账要「按 encoder item」，
        #      所以这里建两个临时计数器。
        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        # [CN] 本拍窗口 = [computed, computed + new_tokens)，
        #      再加上 drafter 的 lookahead 偏移（drafter 会多读几个位置）。
        encoder_window_end = (
            num_computed_tokens + num_new_tokens + shift_computed_tokens
        )
        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=encoder_window_end,
        )
        # [CN] encoder-decoder 的所有输入都在 start_pos=0，所以窗口起点恒为 0。
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        # [CN] 逐个检查落在窗口内的多模态 item。
        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # [CN] encoder-decoder：只要已经算过 decoder token，
            #      就说明 cross-attention 的 KV 早已算好，跳过。
            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            # [CN] 非 encoder-decoder 走标准 encoder 缓存路径：
            #      同一步内同一 item 只算一次；已缓存则直接复用。
            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # [CN] 不允许切分 mm item：本拍窗口只能覆盖一部分时，
            #      回退到这个 item 之前（含 lookahead 偏移，避免 encoder 缓存 miss）。
            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            # [CN] encoder 缓存满了或预算耗尽：
            #      若还没走到该 item 的位置，就只算它之前的 decoder token；
            #      若已经越过（前缀缓存导致的），则本拍一个 token 都排不了。
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # [CN] 算一下本拍窗口真正覆盖到哪些 embedding 下标。
            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            # [CN] 本窗口内一个 embedding 都没覆盖到 —— 这个 item 本拍不用管。
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            # [CN] EC Connector 远端已有这个 item：走外部加载，不占本地计算预算。
            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            # [CN] 扣预算、去重、加入本拍调度列表。
            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    # [CN] 仅用于日志：统计本拍调度了多少 encoder 输入、产生多少 embedding。
    def _make_scheduled_encoder_input_stats(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> ScheduledEncoderInputStats | None:
        stats = ScheduledEncoderInputStats()

        for req_id, input_ids in scheduled_encoder_inputs.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            for input_id in input_ids:
                mm_feature = request.mm_features[input_id]
                stats.num_inputs += 1
                stats.output_tokens += mm_feature.mm_position.get_num_embeds()

        return stats if stats.num_inputs else None

    # [CN] 为用到 structured output 的请求生成 grammar bitmask。
    #      注意顺序必须与 num_scheduled_tokens 的顺序一致 —— 
    #      bitmask 的行是按这个顺序排列的。
    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        # [CN] 只收「非 prefill chunk」的请求：prefill 还没算完时不需要采样掩码。
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    # [CN] ================= 输出处理（调度器的另一半） =================
    #
    #      输入：本拍的 SchedulerOutput + 模型执行器的 ModelRunnerOutput。
    #      输出：按客户端 index 分组的 EngineCoreOutputs。
    #
    #      主循环对**本拍调度过的每个请求**做这些事：
    #        ① 扣减在途计数、排空 stale 份额；
    #        ② 投机解码：按被拒绝的草稿数回滚 num_computed_tokens；
    #        ③ 释放已用完的 encoder 缓存；
    #        ④ 追加新 token、检查停止条件、推进 grammar；
    #        ⑤ 取路由专家 / logprobs / NaN 统计；
    #        ⑥ 生成 EngineCoreOutput；
    #        ⑦ 结束的请求从队列里摘掉并释放资源。
    #
    #      循环之后还要处理：错误请求、KV/EC 连接器事件、事件发布、统计。
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # [CN] 先把 model runner 的输出拆成局部变量 —— 
        #      主循环可能是上千次迭代，避免反复做属性查找。
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        ec_connector_output = model_runner_output.ec_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        # [CN] 本拍及更早入队的所有 GPU 写都已完成，
        #      可以把「围栏已满足」的延后释放块还给块池了。
        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        # [CN] MFU 等性能统计（可选）。
        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        # [CN] 远程 KV 加载失败的块：先找出受影响的请求，
        #      把它们回滚到「最长的有效前缀」，稍后重算。
        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # [CN] 把本拍的路由专家结果从 D2H 缓冲拷进调度器侧的 slot 表。
        #      **必须**在下面按请求读取之前做：
        #      有些请求正好会因为本拍生成的 token 而结束，
        #      而那些 token 的路由刚刚才被拷回来。
        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # [CN] offset 映射要按 model runner 的 req_ids 顺序（input_batch 顺序）
            #      来建，不能用调度器字典的顺序 —— 两者可能不同。
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        # [CN] 主循环。这里的每一次额外操作都会被乘上 batch size，
        #      所以上面的「先批量落盘」都是在为这里减负。
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            # [CN] 扣减在途 token；若有 stale 份额则同步排空（见 _preempt_request）。
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            # [CN] KV 加载失败被重排的请求，本拍不处理。
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            # [CN] 请求可能在执行期间被 abort（PP 或异步调度下会发生）。
            #      注意 delay_free_blocks 时请求不会被置 None，
            #      所以要用 is_finished() 判断而不是查 None。
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            # [CN] drop 模式的 stale 输出（同一步内恢复的场景）整包丢弃。
            # Drop-mode stale output (same-step resume) is discarded entirely.
            if output_is_stale and request.drop_stale_output:
                continue

            # [CN] req_index：本请求在 model runner 输出张量里的行号。
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            # [CN] 投机解码的接受/拒绝结算。
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # [CN] 被拒绝的草稿要把 num_computed_tokens 回滚（异步调度下还要回滚
                #      placeholders）。但**过期的**拒绝计数发生在抢占回滚之前，不能应用。
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                # [CN] 汇总投机解码统计（仅在开启日志时）。
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )
                # [CN] 每请求维度的指标：要从草稿数里剔除「被 grammar 判非法」的，
                #      与 make_spec_decoding_stats 口径保持一致。
                if request.spec_decode_metrics is not None:
                    # Exclude grammar-invalidated drafts from the proposed
                    # count, mirroring make_spec_decoding_stats; the accepted
                    # bucket (j) is unaffected.
                    adj_draft_tokens = num_draft_tokens
                    if scheduler_output.num_invalid_spec_tokens:
                        adj_draft_tokens -= (
                            scheduler_output.num_invalid_spec_tokens.get(req_id, 0)
                        )
                    request.spec_decode_metrics.observe(
                        num_draft_tokens=adj_draft_tokens,
                        num_accepted=num_accepted,
                        detailed=self.spec_decode_metrics_level == "detailed",
                    )

            # [CN] 只有本拍真的执行过了，才释放已消费的 encoder 输入。
            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            # [CN] 停止判定与输出字段准备。
            stopped = False
            new_logprobs = None
            new_sampling_mask = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            ec_transfer_params = None
            prefill_stats = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            # [CN] 追加新 token 并逐 token 检查停止条件（见 _update_request_with_output）。
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            # [CN] pooling 模型：一旦有输出就结束。
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            # [CN] 纯 encoder 实例：跑完 encoder 就发布 embedding，不采样。
            #      prompt 被消费完即意味着所有 item 都编码完了 —— 
            #      encoder 输入绝不会被排到「缓存装不下的 item」之后。
            elif (
                self.is_mm_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            # [CN] structured output：把新 token 喂给 grammar。
            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert isinstance(grammar, StructuredOutputGrammar)
                # [CN] 新 token 可能是「推理内容 + 推理结束标记 + grammar 内容」的混合块，
                #      先裁掉推理部分，只让 grammar 看它该看的部分。
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            # [CN] 读取本请求本拍的路由专家。
            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                # [CN] prefill 刚完成：用调度时刻的块号快照，从 slot 表里读出整个
                #      prompt 的路由结果（免疫异步抢占导致的块释放）。
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                # [CN] 非 prefill：投机解码时被接受的 token 在调度区间的**开头**，
                #      普通 decode / 重 prefill 则在**末尾**。
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            # [CN] 不变式：不产出「部分 prefill」的输出。
            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
            # [CN] prefill 统计定稿：补上「估计的缓存命中 token 数」。
            if should_emit_output:
                prefill_stats = request.take_prefill_stats()
                if prefill_stats is not None:
                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)
                    )

            # [CN] 必须在 _handle_stopped_request **之前**取 finish_reason —— 
            #      流式请求可能被重置回 WAITING，那时就读不到终止原因了。
            finish_reason = None
            # [CN] 处理停止：真正结束的才释放资源；可恢复的（流式）只是回队列。
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)

                # [CN] 按停止时的状态分类，稍后从对应队列里摘除。
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # [CN] 采样 logprobs（仅当请求要且模型产出了）。
            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            # [CN] 采样掩码：标记哪些位置真的被采样了（投机解码下会混合）。
            if self.return_sampling_mask:
                sampling_masks = model_runner_output.sampling_masks
                if new_token_ids and sampling_masks is not None:
                    new_sampling_mask = sampling_masks.slice_request(
                        req_index, len(new_token_ids)
                    )

            # [CN] logits 里出现 NaN 的计数（用于诊断数值问题）。
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # [CN] prompt logprobs 是 per-request 的张量，从字典里取。
            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            # [CN] 组装 EngineCoreOutput，按 client_index 分组。
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_sampling_mask=new_sampling_mask,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=prefill_stats,
                        spec_decode_metrics=(
                            request.spec_decode_metrics
                            if finish_reason is not None
                            else None
                        ),
                        kv_transfer_params=kv_transfer_params,
                        ec_transfer_params=ec_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            # [CN] 不变式：既然没有输出，就不该有 prompt logprobs。
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # [CN] 把停止的请求从各自队列里摘掉（批量 remove_all 更高效）。
        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # [CN] 汇总所有需要按「错误」结束的请求 id。
        error_req_ids = set(self.grammar_compile_error_reqs)
        self.grammar_compile_error_reqs.clear()
        # [CN] KV 加载失败且策略是 fail（不是 recompute）→ 结束为错误。
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            error_req_ids.update(failed_kv_load_req_ids)
        # [CN] EC Connector 拿不到的 encoder 输入：失败是可重试的，
        #      重新发一次请求会重跑编码。
        if self.ec_connector is not None:
            # An encoder input the connector can no longer obtain. Failing is
            # retryable: re-issuing the request re-runs the encode.
            error_req_ids.update(self.ec_connector.take_unavailable_requests())

        # [CN] 统一按 FINISHED_ERROR 结束，并给每个客户端补一条输出。
        if error_req_ids:
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR
            )
            for request in error_reqs:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # [CN] KV 传输完成事件：接收完成的请求可以恢复调度，
        #      发送完成的请求可以释放块。
        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # [CN] EC Connector 的 worker 侧输出回写。
        # EC Connector: update state from worker-side EC connector output.
        if self.ec_connector is not None and ec_connector_output:
            self.ec_connector.update_connector_output(ec_connector_output)

        # [CN] 汇总 worker 侧与调度器侧的 KV 连接器统计。
        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # [CN] 收集 KV cache 事件（块存储 / 淘汰），供外部观测前缀缓存。
        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # [CN] 一次性发布本拍收集到的全部事件。
        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # [CN] 按客户端 index 打包成 EngineCoreOutputs。
        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        # [CN] 多 Engine 场景：把「自上一次输出以来结束的」请求 id 也带上。
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        # [CN] 统计只回给**一个**前端（避免重复计数）；
        #      即便本拍没有任何输出也必须回，所以必要时造一个空壳。
        if (
            stats := self.make_stats(
                spec_decoding_stats,
                kv_connector_stats,
                cudagraph_stats,
                perf_stats,
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    # [CN] 该请求依赖的某个 encoder 输入还在远程传输中 → 本拍不可调度。
    def _ec_transfer_pending(self, request: Request, num_computed_tokens: int) -> bool:
        """Whether an encoder input this request needs is still in transit."""
        return (
            self.ec_connector is not None
            and bool(request.mm_features)
            and not self.ec_connector.ensure_cache_available(
                request, num_computed_tokens
            )
        )

    # [CN] 三种「被阻塞的 waiting」状态：它们都躺在 skipped_waiting 队列里，
    #      各自等待一个外部条件才能提升回可调度。
    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    # [CN] 入队：被阻塞的进 skipped_waiting，其余进 waiting。
    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    # [CN] 选一个队列来调度。两个队列都要看：skipped 里是「被跳过待重试」的，
    #      waiting 里是「新来的」。
    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # [CN] PRIORITY 模式：两个队列都非空时比较队首，谁优先级高取谁。
        #      （FCFS 模式直接让 skipped 优先，因为它们的到达时间更早。）
        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    # [CN] 处理「停止」：返回 True 表示真的结束了；
    #      False 表示这是可恢复请求（流式输入），只是回到等待状态。
    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        # [CN] 流式请求：队列里还有下一块输入 → 就地更新会话内容；
        #      pop 出 None 表示整个流结束了。
        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        # [CN] 没有排队的更新 → 进入「等下一块流式输入」状态，
        #      仍占着 model runner 的一个请求槽位。
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    # [CN] 追加输出 token 并逐 token 检查停止条件；
    #      一旦停止就把多余的 token 裁掉（后面的 token 不算数）。
    #      is_stale 只有 AsyncScheduler 的覆写会用到。
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    # [CN] 释放该请求已经用不到的 encoder 缓存条目。
    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # [CN] 延迟释放：条目要一直保留到 drafter 的预读也越过它为止，
        #      与 encoder 调度路径用的偏移保持一致。
        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's read-ahead has also passed it,
        # mirroring the shift the encoder scheduling path applies.
        spec_lookahead = self.num_prefill_lookahead

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            # [CN] Whisper 这类 encoder-decoder：一旦生成了第一个 token，
            #      cross-attention 的 KV 就已经算好，encoder 输入可以立即释放。
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self._free_encoder_input(request, input_id)
            # [CN] 通用条件：已经越过 placeholder 区间 + lookahead，
            #      此后任何草稿回滚或 drafter 读取都不会再引用它。
            elif (
                start_pos + num_tokens + spec_lookahead
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self._free_encoder_input(request, input_id)

    # [CN] 释放单个 encoder 输入，并通知 EC Connector。
    def _free_encoder_input(self, request: Request, input_id: int) -> None:
        self.encoder_cache_manager.free_encoder_input(request, input_id)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_free(request, input_id)

    # [CN] 接收 drafter 产生的新草稿 token。
    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # [CN] 还在分块 prefill 的请求不要草稿（草稿只用于 decode 步）。
            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            # [CN] structured output：草稿也要过 grammar 校验。
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    # [CN] 在 SchedulerOutput 已经构造好之后，再回填真实的草稿 token。
    #      为什么分两步：草稿由 drafter 异步产生，可能晚于 schedule() 一步。
    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            # [CN] 先裁到「本拍实际调度了几个草稿位」（分块 prefill 时会更少）。
            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            # [CN] 用 grammar 过滤掉不合法的草稿。
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            # [CN] 补回 -1 占位，保持行数不变（cudagraph 需要定长）；
            #      同时记录「被 grammar 判非法的个数」，供统计口径修正。
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    # [CN] 注意 waiting 数要把 skipped_waiting 一起算上。
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        return self.kv_cache_manager.usage

    # [CN] 新请求入队。若 req_id 已存在，说明这是流式输入会话的后续块。
    def add_request(self, request: Request) -> None:
        # [CN] 已有同名请求 → 流式续传：
        #      · 还没到「等流式输入」状态：把新块排进队列；
        #      · 正在等：立刻开始这一块；
        #      · 收到 None（结束哨兵）：按 abort 结束。
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            # [CN] 全新请求：可恢复的请求先建流式队列。
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.spec_decode_metrics_level != "none":
                request.spec_decode_metrics = RequestSpecDecodeMetrics.new(
                    self.num_spec_tokens
                )
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    # [CN] 外部（如 API server 在客户端断连时）发来的结束信号。
    #      注意 request_ids=None 表示结束**全部**请求。
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # [CN] 第一遍：只做收集与分诊（running / waiting），不改状态。
        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # [CN] 第二遍之前先批量摘队列（remove_all 比逐个 remove 快得多）。
        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # [CN] 第二遍：设置终止状态并释放资源。
        # Second pass: set status and free requests
        for request in valid_requests:
            # [CN] 还在 WAITING_FOR_REMOTE_KVS 且**没收到过**接收完成信号：
            #      连接器的异步传输还在跑，块不能现在释放（否则会写到已回收的块）。
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return valid_requests

    # [CN] 释放一个已结束请求的所有资源，返回要随响应回传的传输参数。
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        # [CN] 从在途 prefill 集合里摘掉（它的预留不再计入）。
        self._inflight_prefills.discard(request)
        # [CN] 先给 KV 连接器一个「请求结束」的钩子。
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # [CN] EC 连接器同款钩子。契约要求它**必须在** encoder 缓存释放之前触发，
        #      这样连接器还能查到它记录的 mm_hash 等 per-request 状态。
        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        # [CN] 释放 encoder 缓存，登记 finished id。
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # [CN] 连接器要求延后释放时，这里**不**释放块 —— 
        #      等传输真正完成（_update_from_kv_xfer_finished）再释放。
        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    # [CN] 真正释放块，并把请求从 requests 表里删掉。
    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    # [CN] 暂停状态：PAUSED_ALL 完全停，PAUSED_NEW 只让存量跑完。
    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        logger.info("setting pause state to %s", pause_state.name)
        self._pause_state = pause_state

    # [CN] 释放请求的 KV 块。**关键**：如果还有在途的 GPU step 可能往这些块里写，
    #      就不能立刻还回块池，而是压进 deferred_frees 围栏队列。
    #      判断依据：请求最后一次被调度的序号 > 已处理序号 → 还有写在途。
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    # [CN] 释放 CoW 保留下来的源块，逻辑同上（用传入的围栏序号判断）。
    #      倒序入队是为了和 _drain_deferred_frees 里的释放顺序配合。
    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))

    # [CN] 把「围栏已满足」的延后释放块还给块池。
    #      围栏序号近似单调（CoW 的围栏可能比请求释放的围栏领先一拍），
    #      所以遇到第一个未满足的就停 —— 后面即使有已满足的，晚一拍再放也无害。
    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # [CN] 倒序释放：先淘汰尾块，更符合 LRU 的淘汰顺序。
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    # [CN] 未结束请求数：流式等待中的会话已经在 waiting 里，
    #      但它不在 running 也不算「待调度」，这里要减掉。
    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    # [CN] 是否还有「已结束但尚未真正清理」的请求：
    #      连接器延迟释放时，请求已从调度队列摘掉但仍在 self.requests 里。
    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    # [CN] 引擎是否还要继续活着。除了未结束的请求，
    #      连接器还有未完成的推送工作时也必须保持存活（否则会提前静默）。
    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (
                self.ec_connector is not None
                and self.ec_connector.has_pending_push_work()
            )
        )

    # [CN] 重置前缀缓存。默认只在「没有任何请求占用 KV」时才允许重置；
    #      reset_running_requests=True 会先把所有 running 请求抢占掉再重置。
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        # [CN] 抢占全部 running 请求：这样所有块的引用计数都会降到 0，
        #      重置才可能成功。倒序抢占是为了让它们按 FIFO 顺序回到 running。
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, drop_stale_output=True)

            # [CN] 强制抢占 + 同一步恢复，所以要清掉「上一步调度过」的缓存 —— 
            #      这些请求会被从持久 batch 里冲掉。
            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        # [CN] 真的重置 KV 前缀缓存。若抢占了全部请求仍然失败，
        #      通常是还有请求在等远程 KV 传输（暂不支持）。
        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    # [CN] 重置连接器侧缓存。没有连接器时按「成功」处理 —— 
    #      否则权重更新后的级联清理会莫名其妙地失败。
    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    # [CN] 权重更新后必须调用：旧的视觉 embedding 还在缓存里，
    #      不清理的话请求会读到过期结果。
    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    # [CN] 组装调度器统计（运行/等待请求数、KV 使用率、前缀缓存命中、
    #      投机解码、连接器统计、cudagraph、MFU 等）。仅在开启日志时有效。
    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.to_dict() if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    # [CN] 投机解码统计：注意要从草稿数里剔除被 grammar 判非法的那些，
    #      否则接受率会被算低。
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    # [CN] 关闭事件发布器与两个连接器。
    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    # [CN] ==================== KV Connector 专区 ====================
    #      以下方法只在配置了 kv_transfer_config 时才有实际作用，
    #      负责 P/D 分离、KV offloading、远程前缀命中等场景。
    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        return self.kv_event_publisher.get_publisher_config()

    # [CN] 请求结束时通知 KV 连接器，返回 (是否延后释放块, 随响应回传的参数)。
    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # [CN] producer 侧：把「部分尾块」的卸载最终定稿，
        #      交给连接器登记（不足一块的尾巴也能被远端复用）。
        finished_partial_tails: list[tuple[int, int, int]] = []
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None and kv_transfer_config.is_kv_producer:
            finished_partial_tails = (
                self.kv_cache_manager.finalize_partial_tail_offloads(request)
            )

        # [CN] 先把滑出窗口的前缀块释放掉，再交给连接器 —— 
        #      依据是「已处理」的 token 数（在途的不算，它们还可能被回滚）。
        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        # [CN] 取「已算 token 数」对应的块号，而不是全量块表：
        #      超出已算部分的块内容无效，不该被推送。
        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )
        partial_tail_delay = False
        if finished_partial_tails:
            partial_tail_delay = self.connector.register_finished_partial_tail(
                request,
                block_ids,
                finished_partial_tails,
            )

        # [CN] 兼容还不支持 HMA（混合内存分配器，多 group）的老连接器：
        #      它们只接受单 group 的块列表。
        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            delay_free, kv_xfer_params = self.connector.request_finished(
                request, block_ids[0]
            )
        else:
            delay_free, kv_xfer_params = self.connector.request_finished_all_groups(
                request, block_ids
            )
        return delay_free or partial_tail_delay, kv_xfer_params

    # [CN] 该请求要装下整条序列**还差**多少块（用于异步加载的准入预留）。
    #      apply_admission_cap=True：按准入上限而不是理论需求估算。
    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    # [CN] 所有在途 prefill 还需要的块数之和。
    #      异步加载准入时要先把这部分减掉，避免大家都拿到准入然后互相等死。
    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""

        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    # [CN] 异步 KV 接收完成后，把请求从 WAITING_FOR_REMOTE_KVS 恢复。
    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        # [CN] 加载失败：num_computed_tokens 已被回滚到「最长有效前缀」。
        #      还有有效 token → 把这部分真的缓存起来，并登记后续块需要清零；
        #      一个有效 token 都没有 → 释放所有块（重试时可能命中本地缓存）。
        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        # [CN] 成功：把收到的块登记进前缀缓存（仅在开启缓存时生效）。
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # [CN] 整个 prompt 都命中了 → 需要回退一个 token 重算，
            #      否则没有 logits 可供采样出下一个 token。
            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    # [CN] 尝试把被阻塞的等待请求提升为可调度状态。返回 True 表示可以调度了。
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        # [CN] 等远程 KV：只有连接器报告「接收完成」才能提升。
        #      提升后按是否曾被抢占决定回到 PREEMPTED 还是 WAITING。
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        # [CN] 等 grammar 编译：编译成功就恢复；编译抛异常则登记为错误请求。
        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not structured_output_req or structured_output_req.grammar is None:
                return False
            if isinstance(structured_output_req.grammar, Exception):
                self.grammar_compile_error_reqs.add(request.request_id)
                return False
            request.status = RequestStatus.WAITING
            return True

        # [CN] 等流式输入：只有当新块已经到达（队列非空 → 已被 _update_request_as_session
        #      消费完）时才可能提升；队列为空说明还在等。
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    # [CN] 消费 worker 侧连接器的输出：
    #      finished_recving  → 登记为「可恢复」；
    #      finished_sending  → 推送完成，可以释放块了。
    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # [CN] 接收完成：若请求仍在等远程 KV，登记到 finished_recving_kv_req_ids，
        #      等下一拍 schedule 时提升；若请求已经结束，则直接释放块。
        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        # [CN] 发送完成：本地 KV 已经推走，可以释放块了。
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    # [CN] 扫描受「无效块」影响的请求，把它们的 num_computed_tokens 回滚到
    #      最长的有效前缀，并收集需要重新计算的 token 数与待淘汰的块。
    #
    #      两个要点：
    #      ① 同一个无效块可能被多个请求共享 —— 只有第一个请求负责重算它，
    #         其余请求可以把它当作「已算」继续（用 marked_invalid_block_ids 记账）；
    #      ② evict_blocks 对异步加载传 False —— 那些块还没进缓存，无从淘汰。
    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        # [CN] 三个产出：受影响请求 id、需要重算的 token 总数、待淘汰块。
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # [CN] 「已被标记为待重算」的块集合，用于实现上面的要点①。
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        # [CN] 逐个请求扫描。
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # [CN] TODO：尚未支持混合内存分配器（多 group），这里解包成单 group。
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            # [CN] 只看「本拍之前」就已经算过的部分：本拍新算的块还没落 KV，
            #      不可能含外部加载的内容。
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            # [CN] 遍历可能含外部 token 的块（前 req_num_computed_blocks 个）。
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                # [CN] 本请求已经标记过一个待重算块并回滚过了，
                #      后续无效块不再重复回滚（只记入统计）。
                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                # [CN] 把 num_computed_tokens 截断到第一个失败块的起点 —— 
                #      这就是「最长有效前缀」。
                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # [CN] 该块及其之后的所有块都要从缓存里淘汰：
                #      后面的块内容与它相关，等于一起失效。
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            # [CN] 若本请求的所有无效块都已被前面的请求标记重算，
            #      则回退到「只把已缓存 token 视为已算」。
            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    # [CN] 处理 KV 加载失败：返回需要在 update_from_output 主循环里跳过的请求 id。
    #      两种策略：
    #        · fail      —— 直接按错误结束（并把无效块淘汰掉）；
    #        · recompute —— 回滚重算，异步请求登记到 failed_recving_kv_req_ids
    #                       等传输完成后走 _update_waiting_for_remote_kv 重试。
    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # [CN] 异步加载的请求：块还没进缓存，evict_blocks=False。
        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # [CN] 同步加载的请求：块可能已经进了缓存，要收集待淘汰块。
        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # [CN] 只有在 fail 策略下才淘汰块；recompute 策略下这些块会被重算，
        #      淘汰掉反而会让共享它们的其他请求失去复用机会。
        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        # [CN] fail 策略：记一条错误日志并返回全部失败请求。
        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # [CN] recompute 策略：异步请求登记重试；
        #      只返回**同步**受影响的 id 给主循环跳过。
        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
