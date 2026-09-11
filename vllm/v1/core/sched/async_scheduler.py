# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


# [CN] 异步调度（async scheduling）版本的调度器。
#      背景：默认调度是**同步**的 —— 调度器算完 -> 模型跑 -> 拿到结果 ->
#      再调度下一步。GPU 在"调度"这段时间里是空闲的。
#      异步调度让"调度下一步"与"GPU 跑当前步"**重叠**：
#      下一步的 SchedulerOutput 在 GPU 还在跑时就构造好了。
#
#      代价：构造 SchedulerOutput 时还**不知道**这一步会生成什么 token，
#      所以要用**占位符**（placeholder = -1）先占住位置，
#      等真实 token 回来后再回填。本文件几乎全是围绕占位符的簿记。
class AsyncScheduler(Scheduler):
    # [CN] 占位符列表是**复用**的（每步重建一次而不是每个请求建一次），
    #      因为所有请求本步的草稿数相同，共用一个 list 即可省内存。
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    # [CN] 基类算完之后，这里补三件只有异步调度才需要的事：
    #        1) 按本步的 num_spec_tokens_to_schedule 重建占位符列表；
    #        2) 给每个请求累加"输出占位符计数"（后面要用它判断
    #           有没有拿到足够的真实 token 来算 grammar）；
    #        3) 把 request.spec_token_ids 设成占位符 —— 真实草稿 token
    #           稍后在 worker 进程里更新（省一次跨进程通信）。
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            # [CN] v2 runner + PP 微批次时：设置这个请求下一次**有资格被调度解码**
            #      的步号。PP 下一个请求要等 pp_size 步才能在某 stage 继续，
            #      否则 microbatch 会撞车。
            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size

    # [CN] 真实 token 回来了：减掉相应数量的占位符。
    #      两个防御：
    #        - is_stale（过期投递，比如请求已被抢占）时**不减**，
    #          因为抢占时占位符已被清零，再减就下溢成负数；
    #        - 只在这一步之前状态仍是 RUNNING 的请求才 cache_blocks
    #          （被抢占的请求不该再缓存块）。
    #      末尾那个 assert >= 0 就是兜底下溢的。
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
