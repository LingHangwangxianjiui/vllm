# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# [CN] 文件总览：**n>1 并行采样的「父子请求」拆分与聚合**。
#
#     为什么不在引擎内部实现：V1 调度器一次 step 只认「一个请求 = 一条序列」。
#     而 OpenAI 接口的 n>1 语义是「一个 prompt 出 n 个候选」。最省事的做法是
#     在**前端进程**把 1 个父请求拆成 n 个独立子请求发给引擎，结果回来再拼装成
#     一份 RequestOutput。引擎完全不知道 n 的存在。
#
#     这带来的两个副作用（都是正面的）：
#       · n 路采样会被调度器**独立**抢占、独立计费，互不拖累；
#       · n 个子请求 prompt 完全相同 → 前缀缓存命中率极高，
#         实际上只付一份 prefill 的钱。
#
#     子请求 ID 规则：f"{index}_{父请求ID}"，index 取 0..n-1。
#
#     两种返回语义：
#       · FINAL_ONLY：n 个子请求**全部**结束后，一次性返回 n 条结果；
#       · 其它（流式）：每个子请求有增量就立刻返回，逐条吐给客户端。
#
#     唯一的类 ParentRequest 只存在于前端 OutputProcessor 侧，负责三件事：
#     生成子请求参数、聚合输出、统计「本轮最大生成 token 数」。
from copy import copy
from typing import cast

from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.metrics.stats import IterationStats


# [CN] 父请求句柄：一个 n>1 请求在前端的完整状态。
#      注意它**不是**引擎里的 Request —— 引擎里对应的是 n 个普通子请求。
class ParentRequest:
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """

    request_id: str
    external_req_id: str
    sampling_params: SamplingParams

    # [CN] 还没结束的子请求 ID 集合；空了就代表整个父请求结束。
    # To track the completion of child requests
    child_requests: set[str]

    # [CN] 非流式时暂存 n 条最终结果，按子请求 index 定位槽位（保证顺序稳定）。
    # To aggregate child completions when not streaming
    output_aggregator: list[CompletionOutput]

    # [CN] 所有子请求里生成 token 数的最大值，用于统计吞吐。
    # To find the max number of generated tokens across all children
    max_num_generation_tokens: int

    # [CN] 无 seed 时 n 个子请求共用同一份参数，缓存起来避免重复 copy。
    # To efficiently obtain child sampling params
    cached_child_sampling_params: SamplingParams | None

    # [CN] 初始化：非流式时预开 n 个 None 槽位，流式则不需要聚合器。
    def __init__(self, request: EngineCoreRequest) -> None:
        assert request.external_req_id is not None
        sampling_params = request.params
        self.request_id = request.request_id
        self.external_req_id = request.external_req_id
        self.sampling_params = sampling_params

        # [CN] 子请求集合初始为空，由 get_child_info 逐个登记。
        self.child_requests = set()
        # [CN] FINAL_ONLY 才预分配聚合数组（用 cast(None) 占位，稍后被真实结果覆盖）。
        self.output_aggregator = (
            [cast(CompletionOutput, None)] * sampling_params.n
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    # [CN] 生成第 index 个子请求的采样参数：
    #      有 seed → 每个子请求种子不同；无 seed → n 个子请求共用同一份。
    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.
        """
        # [CN] 无 seed 时 n 个子请求完全同参，可复用同一个对象。
        seed = self.sampling_params.seed
        # [CN] 已经缓存过就直接复用，省一次 copy。
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # [CN] 拷一份父参数出来再改，绝不原地修改父请求的 SamplingParams。
        # Build child sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        # [CN] 无 seed → 缓存起来供后续子请求复用（n-1 次拷贝被省掉）。
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            # [CN] 有 seed → 每个子请求一个独立种子，保证 n 条结果互不相同。
            child_sampling_params.seed = seed + index
        return child_sampling_params

    # [CN] 登记并返回第 index 个子请求的 (ID, 采样参数)。
    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.

        Args:
          index: index within `n` child requests.

        Returns:
          (request ID, sampling_params) tuple
        """
        # [CN] 子请求 ID = "index_父ID"；父 ID 唯一，故子 ID 也唯一。
        child_req_id = f"{index}_{self.request_id}"
        # [CN] 登记进未完成集合，作为「整体是否结束」的判据。
        self.child_requests.add(child_req_id)
        return child_req_id, self._get_child_sampling_params(index)

    # [CN] n 即子请求个数。
    @property
    def n(self) -> int:
        return self.sampling_params.n

    # [CN] 收到某个子请求的输出后，决定「现在返回什么、是否整体结束」。
    #      返回值 (本次要吐出的输出列表, 父请求是否已结束)。
    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        # [CN] 标记：该子请求其实早已结束并已返回过，本次不要再重复吐。
        already_finished_and_returned: bool = False
        # [CN] 子请求结束 → 从未完成集合中摘掉。
        if completion_output.finished():
            # [CN] 正常路径：第一次见到这个子请求结束。
            if child_request_id in self.child_requests:
                self.child_requests.remove(child_request_id)
            # [CN] 异常路径：集合里没有它，说明上一拍已处理过并返回给客户端了。
            else:
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                # [CN] 打上标记，本拍不再输出，避免客户端收到重复内容。
                already_finished_and_returned = True

        # [CN] 流式语义：有增量就直接返回，不做聚合。
        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # If streaming, just return the current output
            #
            # DO NOT output finished and already returned child request to client again
            # [CN] 已返回过的不再输出；否则原样返回这一条。
            outputs = [] if already_finished_and_returned else [completion_output]
        else:
            # [CN] 非流式：把结果写进 index 对应的槽位（顺序由槽位保证，与到达次序无关）。
            # If not streaming, aggregate the n final outputs.
            self.output_aggregator[completion_output.index] = completion_output
            # [CN] 还有子请求没结束 → 返回空列表；全齐了 → 一次性吐出 n 条。
            outputs = [] if self.child_requests else self.output_aggregator

        # [CN] 未完成集合为空 == 整个父请求结束。
        finished = not self.child_requests
        return outputs, finished

    # [CN] 记录「所有子请求中最大的生成 token 数」，作为本请求的代表值。
    def observe_num_generation_tokens(self, num_generation_tokens: int):
        self.max_num_generation_tokens = max(
            num_generation_tokens, self.max_num_generation_tokens
        )
        return self.max_num_generation_tokens

    # [CN] 静态方法：子请求结束时更新 IterationStats。
    #      关键点：**只在最后一个子请求结束时才记账**，否则 n 会被重复计 n 次，
    #      吞吐统计直接翻倍。
    @staticmethod
    def observe_finished_request(
        parent_req: "ParentRequest | None",
        iteration_stats: IterationStats,
        num_generation_tokens: int,
    ):
        # [CN] 无父请求（普通 n=1 请求）时 n 记为 1。
        n_param = parent_req.n if parent_req is not None else 1

        # [CN] 有父请求 → 取「所有子请求的最大值」作为本次请求的生成长度。
        if parent_req is not None:
            num_generation_tokens = parent_req.observe_num_generation_tokens(
                num_generation_tokens
            )

        # Child requests finished, we can now record to iteration stats
        # [CN] 仅当本就是普通请求，或父请求的子请求已全部结束时，才写入统计。
        if parent_req is None or not parent_req.child_requests:
            iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)
            iteration_stats.n_params_iter.append(n_param)
