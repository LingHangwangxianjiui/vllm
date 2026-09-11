# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# [CN] 文件总览：把 EngineCore 吐出的 **EngineCoreOutput** 加工成面向用户的
#      **RequestOutput**。它运行在 **前端进程**（API server / LLM 类所在的进程），
#      是 V1 架构里「进程边界」之后的第一站。
#
#      ========================= 为什么需要它 =========================
#      EngineCore 只认 token id，不知道什么是文本、什么是流式、什么是 n>1。
#      这些「用户语义」全部在这里补齐：
#         · 增量 detokenize（把 token id 变成文本，见 detokenizer.py）；
#         · logprobs 后处理（见 logprobs.py）；
#         · 流式语义（FINAL_ONLY / DELTA / CUMULATIVE）；
#         · n>1 并行采样的子请求聚合（ParentRequest）；
#         · 停止字符串检测（「引擎没停但文本里出现了 stop 串」要反向 abort）；
#         · 统计与 tracing。
#
#      ========================= 三条重要约定 =========================
#      1) **只允许在这一处遍历整个 batch**。下面 process_outputs() 的 docstring
#         明确写了：V1 极力减少 Python 层对全批次的循环，谁要碰每个元素，
#         就把逻辑塞进这个函数里。
#      2) **外部 req_id 与内部 req_id 是两个东西**。外部的是用户传入的，
#         内部的是创建 EngineCoreRequest 时随机生成的；一个外部 id 可能对应
#         多个内部 id（n>1 或重试）。external_req_ids 维护这个映射。
#      3) **输出有两条去路**：有 queue → AsyncLLM（放进队列给 generate() 任务）；
#         无 queue → LLMEngine（直接放进返回列表）。
#
#      代码组织：RequestOutputCollector（单请求的输出信箱）→ 两个 dataclass →
#      RequestState（单请求的全部前端状态 + 输出构造）→ OutputProcessor（主体）。

import asyncio
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.outputs import (
    STREAM_FINISHED,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
    SamplingMask,
)
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    RequestSpecDecodeMetrics,
    RequestStateStats,
    SchedulerStats,
)
from vllm.v1.outputs import SamplingMaskLists

# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")


# [CN] 单个请求的「输出信箱」：连接 **asyncio 消费端**（generate() 任务）与
#      **同步生产端**（OutputProcessor.process_outputs）的桥。
#      为什么需要：前端是 async 的、引擎输出是同步批量到达的，两者速度不匹配。
class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    # [CN] output_kind 决定 aggregate：DELTA 模式下生产快于消费时，
    #      多条输出会被**合并**成一条（见 put），避免队列无限增长。
    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
        self.ready = asyncio.Event()

        self._input_stream_task: asyncio.Task | None = None

    # [CN] 非阻塞写。三种情况：
    #      ① 槽位为空或来的是异常 → 直接写入并置 ready；
    #      ② 槽位已有 RequestOutput 且新来的也是 → **合并**（这才是关键）；
    #      ③ pooling 输出 → 直接覆盖（pooling 没有增量概念）。
    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        # [CN] n>1 时不同 request index 的输出不能互相覆盖，
        #      所以走 add() 合并而不是替换。
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output

    # [CN] 阻塞读：等 ready 事件。异常在这里被**重新抛出**，
    #      这样 generate() 任务里就能用 try/except 拿到引擎侧的错误。
    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # [CN] 非阻塞读（LLMEngine 的 step 路径用）。
    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # [CN] 关闭时取消「流式输入」任务（如果开了的话）。
    def close(self):
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None

    # [CN] 兜底：万一用户忘了 close()，析构时也要把流式输入任务取消掉，
    #      否则它会一直挂着。注意要用 call_soon_threadsafe 保证线程安全。
    def __del__(self):
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
            self._input_stream_task = None


@dataclass
# [CN] process_outputs 的返回值：
#      request_outputs —— 给 LLMEngine 直接返回的那部分；
#      reqs_to_abort   —— 需要**反向通知引擎** abort 的请求
#                        （典型场景：detokenizer 在文本里发现了 stop 字符串，
#                          但引擎那边还在按 token 数继续跑）。
class OutputProcessorOutput:
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    reqs_to_abort: list[str]


@dataclass
# [CN] 流式输入的「下一块」更新。final=True 表示这一块之后整个流就结束了。
class StreamingUpdate:
    """Streaming input update data for output processor.

    Contains the incremental prompt data to be applied to a request state
    when the current sub-request completes.
    """

    prompt: str | None
    prompt_token_ids: list[int] | None
    arrival_time: float
    final: bool = False


# [CN] 单个请求在**前端进程**里的全部状态。
#      它对应引擎侧 vllm/v1/request.py 的 Request，但两者关心的东西不同：
#      引擎关心 token / 块 / 位置；这里关心文本、logprobs、统计、输出形态。
class RequestState:
    # [CN] 构造。参数多，但可以按职责分：
    #      ① 身份（request_id / external_req_id / parent_req / request_index）；
    #      ② 输入（prompt / prompt_token_ids / prompt_embeds）；
    #      ③ 加工器（logprobs_processor / detokenizer）；
    #      ④ 输出形态（output_kind / stream_interval / queue）。
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
    ):
        # [CN] 身份信息。request_index 在 n>1 时用来区分同父的多个子序列。
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.lora_request = lora_request
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.max_tokens_param = max_tokens_param
        self.top_p = top_p
        self.n = n
        self.temperature = temperature
        # [CN] 是否还在 prefill。首次收到输出时置 False（见 process_outputs）。
        self.is_prefilling = True
        self.queue = queue
        self.num_cached_tokens = 0
        self.num_cache_creation_tokens = 0
        # [CN] 投机解码的**按序列**累计指标：引擎只在请求结束时回传一次，
        #      然后挂到这条序列的 CompletionOutput 上。
        # Per-sequence spec-decode accumulator; arrives once (on finish) via
        # EngineCoreOutput, then attached to this sequence's CompletionOutput.
        self.spec_decode_metrics: RequestSpecDecodeMetrics | None = None

        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None

        # [CN] 路由专家 / 采样掩码分块累积：engine 每拍回一小块，
        #      请求结束时统一拼接（见 _new_completion_output）。
        # Routed experts accumulation (prompt + sample chunks)
        self.routed_experts_chunks: list[np.ndarray] = []
        self.sampling_mask_chunks: list[SamplingMaskLists] = []

        # [CN] stream_interval > 1 时不是每 token 都发，
        #      sent_tokens_offset 记录「已经发到哪」，用于 DELTA 切片。
        # Stream Interval
        self.stream_interval = stream_interval
        self.sent_tokens_offset = 0  # Offset of sent tokens

        # [CN] 流式输入：input_chunk_queue 存「还没被应用的后续输入块」。
        #      注意它是 None 与空 deque **语义不同**：None 表示流已经整体结束。
        # Streaming input queue
        self.streaming_input = stream_input
        self.input_chunk_queue: deque[StreamingUpdate] | None = (
            deque() if stream_input else None
        )

    # [CN] 应用一块流式输入：把新 prompt 追加进来，并重新回到 prefill 状态。
    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        # Apply the update to the request state.
        self.streaming_input = not update.final
        # TODO also include relevant output tokens in new prompt here
        #     (match scheduler behavior).
        if update.prompt:
            self.prompt = (
                (self.prompt + update.prompt) if self.prompt else update.prompt
            )
        if self.prompt_token_ids:
            self.prompt_token_ids.extend(update.prompt_token_ids or ())
        else:
            self.prompt_token_ids = update.prompt_token_ids or []
        assert self.prompt_token_ids is not None
        self.prompt_len = len(self.prompt_token_ids)
        if self.stats is not None:
            self.stats.arrival_time = update.arrival_time
        self.is_prefilling = True

    # [CN] 从 EngineCoreRequest 建状态。注意这里会按 sampling / pooling 分流：
    #      pooling 请求没有 detokenizer、logprobs 和 max_tokens 这些概念。
    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> "RequestState":
        # [CN] 采样类请求：detokenize=False 时直接把 tokenizer 置 None，
        #      后续就只回 token id 不回文本（省一轮解码开销）。
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            if sampling_params.stream_interval is not None:
                # clamp to the engine-level stream interval.
                stream_interval = max(sampling_params.stream_interval, stream_interval)
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        # [CN] pooling 类请求：没有 detokenizer / logprobs，
        #      output_kind 从 pooling_params 上取。
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            stream_input=request.resumable,
        )

    # [CN] 构造一条 RequestOutput（或返回 None 表示「这次不发」）。
    #
    #      「这次不发」有三种情况：
    #        · FINAL_ONLY 模式且还没结束；
    #        · stream_interval 节流；
    #        · n>1 时父请求还没集齐（get_outputs 返回空）。
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        # [CN] FINAL_ONLY：只要最终结果，中途全部丢弃。
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        # [CN] stream_interval 节流：只在 ① 已结束 ② 第一个 token
        #      ③ 距上次发送已累计 stream_interval 个 token 时才发。
        #      这是「流式太碎」与「用户等太久」之间的权衡旋钮。
        if self.stream_interval > 1:
            assert self.detokenizer is not None

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                >= self.stream_interval
            ):
                return None

            # [CN] DELTA 模式：只发「上次发过之后」的新 token，并推进 offset。
            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[
                    self.sent_tokens_offset :
                ]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        # [CN] 对外一律用 external_req_id —— 用户只认自己传进来的那个 id。
        external_req_id = self.external_req_id

        # [CN] pooling 请求：输出就是那一个张量，直接返回。
        if pooling_output is not None:
            return self._new_request_output(
                external_req_id,
                [self._new_pooling_output(pooling_output)],
                finished,
            )

        # [CN] 普通生成：先造 CompletionOutput（单条序列的结果）。
        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        # [CN] n>1 的并行采样：交给 ParentRequest 聚合。
        #      它可能返回空（还没集齐 / 已被裁剪），此时本次不发。
        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(
            external_req_id,
            outputs,
            finished,
            kv_transfer_params,
            ec_transfer_params,
        )

    # [CN] 把 CompletionOutput 列表包成 RequestOutput。
    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput] | list[PoolingOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        # [CN] prompt 是 embedding 而非 token id 时，用等长的 0 占位，
        #      保持返回结构一致（用户拿不到真实 prompt token）。
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        first_output = outputs[0]
        # [CN] pooling 输出永远只有一条，直接返回 PoolingRequestOutput。
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=external_req_id,
                outputs=first_output,
                num_cached_tokens=self.num_cached_tokens,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        # [CN] DELTA 模式下 prompt logprobs 只在第一次发，发完就 pop 掉 —— 
        #      否则每个增量包都会重复带上整段 prompt 的 logprobs。
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            num_cache_creation_tokens=self.num_cache_creation_tokens,
            metrics=self.stats,
        )

    # [CN] 构造单条序列的 CompletionOutput：text / token_ids / logprobs /
    #      finish_reason / stop_reason 都在这里按 delta 与否分别处理。
    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # [CN] 文本：DELTA 只取新增部分，非 DELTA 取全量。
        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # [CN] logprobs：DELTA 只取最后 N 个。
        #      注意避开 [-0:] 这个陷阱 —— 它会在「本轮没有新 token」时
        #      返回整段历史；[:0] 才是真的返回空且保持类型不变。
        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            num_new_tokens = len(token_ids)
            # Avoid [-0:], which returns the full accumulated history when a
            # delta contains no new token IDs. [:0] preserves the concrete
            # list or FlatLogprobs representation while returning no entries.
            logprobs = logprobs[-num_new_tokens:] if num_new_tokens else logprobs[:0]

        # [CN] 采样掩码只在请求结束时才拼好发出去。
        sampling_mask = None
        if finished and self.sampling_mask_chunks:
            sampling_mask = SamplingMask(
                [chunk.token_ids.tolist() for chunk in self.sampling_mask_chunks]
            )

        # [CN] 路由专家同理：结束时才把累积的分块拼成一个大数组。
        # Concatenate routed experts on finish
        routed_experts = None
        if finished and self.routed_experts_chunks:
            routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            routed_experts=routed_experts,
            sampling_mask=sampling_mask,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
            spec_decode_metrics=self.spec_decode_metrics if finished else None,
        )

    # [CN] pooling 输出的包装（就一个 data 字段）。
    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        return PoolingOutput(data=pooling_output)


# [CN] 输出处理主体。持有所有在途请求的 RequestState，
#      并把 EngineCoreOutputs 批量加工成 RequestOutput。
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    # [CN] 注意 tokenizer 可能为 None（detokenize=False 或纯 pooling 场景）。
    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        # [CN] 状态表：
        #      request_states   —— 内部 req_id -> RequestState；
        #      parent_requests  —— 并行采样的父请求；
        #      external_req_ids —— 外部 req_id -> [内部 req_id...]（一对多）。
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.lora_states = LoRARequestStates(log_stats)
        self.tracing_enabled = tracing_enabled

    # [CN] 未完成请求数 = 还留在 request_states 里的数量。
    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_request(self, request_id: str) -> bool:
        return request_id in self.request_states

    # [CN] 仍在 prefill 的请求占用的 prompt token 总数（用于限流 / 水位）。
    #      用 prompt_len 而不是「剩余 prefill 工作量」：引擎侧的
    #      num_computed_tokens 在 prefill 完成前不会传回前端。
    def get_num_queued_tokens(self) -> int:
        """Total prompt tokens of requests currently in the prefill phase.

        Uses ``prompt_len`` rather than remaining prefill work because the
        scheduler's ``num_computed_tokens`` is not propagated to the API
        server until prefill completes.  See ``SchedulerConfig`` docs.
        """
        return sum(
            req.prompt_len for req in self.request_states.values() if req.is_prefilling
        )

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    # [CN] 引擎致命错误：把异常塞进每个请求的信箱，
    #      这样所有阻塞在 get() 上的 generate() 任务都会醒来并抛出。
    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    # [CN] abort 一批请求。要点：
    #      · 入参既可以是**外部** id 也可以是**内部** id（用 internal 区分）；
    #      · 一个外部 id 可能对应多个内部 id（n>1），要全部 abort；
    #      · 内部 id 也可能是**父请求** id，此时要先递归 abort 它的子请求。
    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        # [CN] 第一遍：把「用户给的 id」翻译成「内部 id 列表」。
        internal_req_ids = []
        for request_id in request_ids:
            # [CN] 内部 id：可能是父请求 id，原样保留；
            #      同时把它从 external -> internal 映射里摘掉。
            if internal:
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)

                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):
                    external_req_id = req_state.external_req_id
                    internal_ids = self.external_req_ids[external_req_id]
                    internal_ids.remove(request_id)
                    if not internal_ids:
                        del self.external_req_ids[external_req_id]
            # [CN] 外部 id：取出它名下的全部内部 id。
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)

        # [CN] 第二遍：真正摘状态、发终止输出。
        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                self.lora_states.request_finished(request_id, req_state.lora_name)
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        pooling_output=EMPTY_CPU_TENSOR
                        if req_state.detokenizer is None
                        else None,
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        kv_transfer_params=None,
                        ec_transfer_params=None,
                    )
                ):
                    req_state.queue.put(request_output)
            # [CN] 这是父请求 id：**先递归 abort 子请求**，再移除父请求。
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    # [CN] 登记一个新请求。若 req_id 已存在 → 说明是流式输入的后续块。
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        # [CN] 已存在：交给 _update_streaming_request_state 排队处理。
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._update_streaming_request_state(req_state, request, prompt)
            return

        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        # [CN] 并行采样：把父请求也记下来（后续靠它聚合子请求输出）。
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # [CN] 维护 external -> internal 的一对多映射，供 abort 时反查。
        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)

    # [CN] 流式输入的后续块：**先排队**而不是立刻应用。
    #      因为此刻引擎里可能还跑着上一块，立刻改状态会对不上。
    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """Queue a streaming update instead of immediately applying it."""
        # [CN] 不再 resumable = 这是最后一块（或流结束）。
        #      三种子情况：引擎已结束 / 队列里还有块（标记最后一块为 final）/
        #      队列空（直接关掉流式标志）。
        if not request.resumable:
            # Final request - just mark completion, don't add its dummy tokens.
            if req_state.input_chunk_queue is None:
                # Engine already finished - emit final output and clean up.
                self._finish_request(req_state)
                if req_state.queue is not None:
                    # Emit a final output with finished=True
                    # to unblock the generate() loop.
                    req_state.queue.put(STREAM_FINISHED)
            elif req_state.input_chunk_queue:
                req_state.input_chunk_queue[-1].final = True
            else:
                req_state.streaming_input = False
            return

        # [CN] 构造一个待应用的更新。
        update = StreamingUpdate(
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            arrival_time=request.arrival_time,
        )

        # [CN] 引擎侧已经没有在跑的输入了 → 可以立即应用；
        #      否则入队，等当前子请求结束时再应用（见 process_outputs）。
        # Apply request updates now if the last input already completed.
        if req_state.input_chunk_queue is None:
            req_state.apply_streaming_update(update)
            req_state.input_chunk_queue = deque()
        else:
            # Queue the streaming update otherwise.
            req_state.input_chunk_queue.append(update)

    # [CN] ============ 唯一允许遍历整个 batch 的地方 ============
    #      每步做四件事：① 统计 ② detokenize ③ logprobs ④ 生成输出。
    #      需要「碰每个元素」的新逻辑都应该加进这个循环里。
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """

        # [CN] 主循环。
        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            # [CN] 请求已经被 abort 了（或从未见过）→ 直接忽略这条输出。
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # [CN] ① 统计（在 is_prefilling 被改写之前传进去）。
            # 1) Compute stats for this iteration.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            # [CN] 拆字段：先取出来，避免后面反复访问属性。
            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            ec_transfer_params = engine_core_output.ec_transfer_params
            # [CN] 路由专家分块累积，结束时统一拼接。
            if engine_core_output.routed_experts is not None:
                req_state.routed_experts_chunks.append(
                    engine_core_output.routed_experts
                )

            # [CN] 第一拍输出到达 = prefill 完成。此时把 prefill 统计
            #      （缓存命中 / 新建缓存的 token 数）落到 RequestState 上。
            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                    req_state.num_cache_creation_tokens = (
                        engine_core_output.prefill_stats.num_cache_creation_tokens
                    )
                req_state.is_prefilling = False

            # [CN] 投机解码指标只在结束时回传，先存着。
            if engine_core_output.spec_decode_metrics is not None:
                req_state.spec_decode_metrics = engine_core_output.spec_decode_metrics

            # [CN] 非 pooling 请求才需要 detokenize / logprobs。
            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                # [CN] 采样掩码分块累积。
                if engine_core_output.new_sampling_mask is not None:
                    req_state.sampling_mask_chunks.append(
                        engine_core_output.new_sampling_mask
                    )
                # [CN] ② 增量解码 + **停止字符串检测**。
                #      注意这里可能「就地升级」finish_reason：
                #      引擎是数 token 停的，用户是看文本停的，两者不一致。
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # [CN] ③ 采样 / prompt logprobs 后处理。
                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            # [CN] ④ 生成 RequestOutput（可能返回 None 表示本次不发）。
            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                ec_transfer_params,
            ):
                # [CN] 流式输入请求即使引擎侧报结束，对外也不能标 finished —— 
                #      后面还有输入块要接着生成。
                if req_state.streaming_input:
                    request_output.finished = False

                # [CN] 两条去路：AsyncLLM 进队列，LLMEngine 进返回列表。
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # [CN] 请求结束后的清理分叉。
            # Free completed requests.
            # [CN] 流式输入：从队列里取下一块应用；队列空了就把
            #      input_chunk_queue 置 None（表示整条流结束）。
            if finish_reason is not None:
                if req_state.streaming_input:
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update)
                    else:
                        req_state.input_chunk_queue = None
                # [CN] 普通请求：清理状态、统计、tracing。
                else:
                    self._finish_request(req_state)
                    # [CN] **反向 abort**：引擎没停，但 detokenizer 在文本里
                    #      发现了 stop 字符串 —— 必须通知引擎把它停下来，
                    #      否则引擎会继续按 max_tokens 生成。
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)

                    # Track per-request stats
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    # [CN] tracing 只在结束时做一次（拿到完整时间线）。
                    if self.tracing_enabled:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    # [CN] 清理完成请求的全部登记：状态表、external->internal 映射、父请求。
    def _finish_request(self, req_state: RequestState) -> None:
        req_id = req_state.request_id
        self.request_states.pop(req_id)

        internal_ids = self.external_req_ids[req_state.external_req_id]
        internal_ids.remove(req_id)
        if not internal_ids:
            del self.external_req_ids[req_state.external_req_id]

        # [CN] 父请求在没有子请求之后才能移除。
        # Remove parent request if applicable.
        parent_req = req_state.parent_req
        if parent_req and not parent_req.child_requests:
            self.parent_requests.pop(parent_req.request_id, None)

    # [CN] 把调度器统计同步给 LoRA 状态跟踪器。
    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        self.lora_states.update_scheduler_stats(scheduler_stats)

    # [CN] 手工埋点：用请求级别的四个时间点（到达 / 排队 / 首 token / 末 token）
    #      拼出 OpenTelemetry 风格的 gen_ai 语义属性。
    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,
        req_state: RequestState,
        iteration_stats: IterationStats | None,
    ) -> None:
        assert req_state.stats is not None
        assert iteration_stats is not None

        metrics = req_state.stats
        arrival_time_ns = int(metrics.arrival_time * 1e9)
        trace_context = extract_trace_context(engine_core_output.trace_headers)
        prompt_length = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # [CN] 五个关键时延：
        #      e2e = 全程；queued = 排队等调度；prefill = 首 token 前的计算；
        #      decode = 首 token 之后的生成；inference = prefill + decode。
        # Calculate timing metrics
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
        queued_time = metrics.scheduled_ts - metrics.queued_ts
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts
        decode_time = metrics.last_token_ts - metrics.first_token_ts
        inference_time = metrics.last_token_ts - metrics.scheduled_ts

        # Build attributes dict
        # [CN] 组装 span 属性（TTFT、E2E、prompt / completion token 数等）。
        attributes: dict[str, Any] = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
                metrics.first_token_latency
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
                metrics.num_generation_tokens
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,
        }

        # Add optional request parameters
        if req_state.top_p:
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p
        if req_state.max_tokens_param:
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
                req_state.max_tokens_param
            )
        if req_state.temperature:
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = (
                req_state.temperature
            )
        if req_state.n:
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n

        # [CN] 真正上报 span。start_time 用请求到达时间，
        #      这样 span 在时间轴上覆盖完整请求生命周期。
        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.SERVER,
        )

    # [CN] 把本次输出并入迭代级统计（引擎侧时间戳为准）。
    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert engine_core_timestamp is not None
        assert req_state.stats is not None
        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            req_state.is_prefilling,
            req_state.stats,
            self.lora_states,
            req_state.lora_name,
        )

    # [CN] 请求结束时的收尾统计：token 数、缓存命中、LoRA 释放、
    #      以及并行采样父请求的聚合观察。
    def _update_stats_from_finished(
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert finish_reason is not None
        assert req_state.stats is not None
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            request_id=req_state.external_req_id,
            num_prompt_tokens=req_state.prompt_len,
            max_tokens_param=req_state.max_tokens_param,
            req_stats=req_state.stats,
            num_cached_tokens=req_state.num_cached_tokens,
        )
        self.lora_states.request_finished(req_state.request_id, req_state.lora_name)

        ParentRequest.observe_finished_request(
            req_state.parent_req, iteration_stats, req_state.stats.num_generation_tokens
        )
