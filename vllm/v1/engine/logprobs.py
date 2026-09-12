# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# [CN] 文件总览：**logprobs 的前端处理与 UTF-8 修正**。
#
#     为什么还要前端处理：引擎只吐张量（token id / logprob / rank），
#     而用户要看到的是「token 字符串 + 对数概率 + 排名」这种可读结构。
#     把张量转成 Logprob 容器、把 id 解码成文本，都放在前端进程做，
#     避免占用引擎的 step 时间。
#
#     ===================== 本文件最反直觉的一块 =====================
#     **UTF-8 修正（_correct_decoded_token）**。背景：byte-fallback 类
#     分词器（GPT-2 系、Llama 3 部分词表）把一个多字节 UTF-8 字符拆成
#     多个 token，比如「中」= E4 B8 AD 三个字节对应三个 token。
#     单独 decode 任一字节 token 都会得到 U+FFFD（替换字符，问号方块）。
#
#     修正思路：拿**前面几个已生成的 token** 当上下文，把它们和当前 token
#     拼起来一起 decode。字节序列补齐后 U+FFFD 就消失了，再把「属于上下文
#     的那段前缀」切掉，剩下的才是当前 token 真正贡献的文本。
#     最多回溯 4 个 token —— UTF-8 最长 4 字节，4 个足够覆盖任何字符。
#
#     ===================== 两个容易混淆的概念 =====================
#     · **顺序上下文**（context_token_ids）：序列中前几个位置的 token，
#       时间上是前后关系。用于 UTF-8 修正。
#     · **同位置候选**（tokens / decoded_tokens_list）：同一个位置上
#       的 [采样到的, top1, top2, ...]，彼此是**并列**关系。
#       UTF-8 修正**不能**拿它们当上下文（见 _verify_tokens 的说明）。
#
#     另一个要点：prompt logprobs 是**跨 chunk 累积**的。分块 prefill 时
#     每拍只回一部分，聚齐后由 pop_prompt_logprobs 一次性取走并返回，
#     以满足 DELTA 语义（要么不给，要给就给全）。
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.logprobs import (
    FlatLogprobs,
    PromptLogprobs,
    SampleLogprobs,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike,
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

# [CN] 无限长的 None 迭代器：关闭 detokenization 时用它填充「解码文本」字段，
#      避免为每个位置都构造一个 None 列表。
NONES = itertools.repeat(None)


# [CN] 单个请求的 logprobs 处理器。随请求创建、随请求销毁，
#      在 OutputProcessor 里按 request_id 挂着。
@dataclass
class LogprobsProcessor:
    # [CN] 该请求的分词器；关闭 detokenization 时为 None。
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: TokenizerLike | None

    # [CN] 生成阶段（采样）的 logprobs 累积结果。
    # Logprobs for this request
    logprobs: SampleLogprobs | None
    # [CN] prefill 阶段的 logprobs 累积结果（分块 prefill 时跨多拍累积）。
    prompt_logprobs: PromptLogprobs | None
    # [CN] 累计对数概率：一路把采样 token 的 logprob 加起来。
    #      它是「这段生成结果有多可能」的度量，常用于打分/排序候选。
    cumulative_logprob: float | None
    # [CN] 每个位置返回几个候选（用户请求的 logprobs 数）；None 表示不启用。
    num_logprobs: int | None
    # [CN] prompt 每个位置返回几个候选；None 表示不启用。
    num_prompt_logprobs: int | None

    # [CN] 构造入口：按 sampling_params 决定启用哪些字段。
    #      设计要点 —— 用 None 表示「没启用」，而不是空列表。
    #      这样后续所有 update 都可以用 `if x is None` 直接短路，
    #      不必为一个从不使用 logprobs 的请求付出任何开销。
    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        # [CN] 读用户请求的采样参数。
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.num_logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            # [CN] 启用时才把累计值初始化为 0.0；不启用保持 None。
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            # [CN] 按是否 flat_logprobs 选择底层容器实现（省内存 vs 好读）。
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=(
                None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
            ),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    # [CN] 处理**生成阶段**的 logprobs（引擎每拍回一次）。
    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        # [CN] 解包四个张量列表：token id / logprob / rank / （第四个暂不用）。
        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists

        # [CN] 外层长度为 1；只有推测解码等场景一拍出多个 token 时才 > 1，
        #      此时每个元素对应一个生成位置。
        for rank_np, logprobs_np, token_ids_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst
        ):
            # [CN] 把 numpy 张量转成 Python 原生类型，便于构造 Logprob 容器。
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()
            # [CN] 非增量解码：这里得到的就是最终文本，不需要像 detokenizer 那样
            #      维护跨拍的中间状态（logprobs 每个位置是独立的）。
            # Detokenize (non-incrementally).
            decoded_tokens: list[str] | Iterable[None]
            # [CN] 没分词器 → 用无限 None 流填充，文本字段全为 None。
            if self.tokenizer is None:
                decoded_tokens = NONES
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
                # [CN] 取**前面几个已生成 token** 作为顺序上下文（用于 UTF-8 修正）。
                context_token_ids = self._get_sampled_context_ids(self.logprobs)
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list,
                    tokens=token_ids,
                    context_token_ids=context_token_ids,
                )

            # [CN] 约定：sampler 把「实际采样到的那个」放在第 0 位。
            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            # [CN] 累计对数概率累加的是采样 token 的概率，不是 top1 的。
            self.cumulative_logprob += sampled_token_logprob

            # [CN] 把这个位置的候选列表追加进容器，长度截断到 num_logprobs。
            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.logprobs,
                token_ids,
                logprobs,
                decoded_tokens,
                rank,
                self.num_logprobs,
            )

    # [CN] 处理 **prompt 阶段**的 logprobs（prefill 时一次性回来一大块）。
    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        # [CN] 解包张量；prompt logprobs 是二维的 [num_prompt_tokens, num_logprobs]。
        token_ids, logprobs, ranks, *_ = prompt_logprobs_tensors

        # Recover shapes.
        # [CN] 从 shape 里恢复两个维度（张量本身不携带语义信息）。
        num_prompt_tokens, num_logprobs = logprobs.shape

        # [CN] 一次性把**所有** prompt token 解码完。
        #      注意是先 flatten 成一维再解码 —— 输出是扁平的
        #      [num_tok * num_lps] 长列表，后面按 offset 切片还原。
        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        all_decoded_tokens: list[str] | None = (
            None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(
                self.tokenizer, token_ids.flatten().tolist()
            )
        )

        # [CN] 张量 → Python 列表。后续的逐位置处理都在 Python 层做。
        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids_list = token_ids.tolist()

        # Make Logprob for each position.
        # [CN] 逐位置处理：每个 prompt token 都要产出自己的候选列表。
        for pos in range(num_prompt_tokens):
            # [CN] 扁平化后的切片边界：第 pos 行占 [pos*num_logprobs, +num_logprobs)。
            # Handle flattening and UTF-8 correction per position
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs

            decoded_tokens_for_pos: list[str] | Iterable[None]
            # [CN] 无分词器 → 文本字段全填 None。
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES
            else:
                # [CN] 从扁平列表里切出当前位置的候选文本。
                # Extract decoded tokens for this position
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]
                # [CN] 上下文 = 已累积进 prompt_logprobs 的**前面那些 prompt token**。
                #      注意是顺序上下文（前几个位置），不是本位置的 top-k 候选。
                # Context: preceding prompt tokens accumulated in
                # self.prompt_logprobs from previous loop iterations.
                context_token_ids = self._get_sampled_context_ids(self.prompt_logprobs)
                # [CN] UTF-8 修正只在**本位置内部**进行，不跨位置混淆。
                # Apply UTF-8 correction within this position's token boundaries
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice,
                    tokens=token_ids_list[pos],
                    context_token_ids=context_token_ids,
                )

            # [CN] 追加到 prompt logprobs 容器。
            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids_list[pos],
                prompt_logprobs[pos],
                decoded_tokens_for_pos,
                prompt_token_ranks[pos],
                self.num_prompt_logprobs,
            )

    # [CN] 取出并清空累积的 prompt logprobs。
    #      为什么要「取走」而不是「读取」：分块 prefill 会分多拍回来，
    #      必须等**最后一块**到齐才能一次性返回，否则用户会看到不完整的
    #      prompt logprobs（违反 DELTA 语义：要么不给，给就给全）。
    #      取走即清空，保证下一批（如果有）从头累积。
    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        # [CN] 非空才清空：空列表表示「本次没有累积」，无需重置。
        if plp:
            self.prompt_logprobs = []
        return plp

    # [CN] 从已有 logprobs 容器里提取**最近几个已采样 token 的 id**。
    #      关键前提：每个位置采样到的 token 总被放在第 0 位（见 append 的约定），
    #      所以取每个 dict 的第一个 key 就能拿到真实生成的 token。
    @staticmethod
    def _get_sampled_context_ids(
        logprobs_source: SampleLogprobs | PromptLogprobs | None,
        max_context: int = 4,
    ) -> list[int]:
        """Extract recent sampled token IDs from a logprobs source.

        The sampled (or prompt) token at each position is the first
        entry, since it is always inserted first by
        append_logprobs_for_next_position.

        Args:
            logprobs_source: The logprobs container to extract from.
            max_context: Maximum number of preceding tokens to return.
                4 is sufficient for any UTF-8 multi-byte sequence.

        Returns:
            List of sampled token IDs, oldest first, most recent last.
        """
        # [CN] 还没有任何累积 → 返回空上下文。
        if not logprobs_source:
            return []

        # [CN] 只看最后 max_context 个位置。
        n = len(logprobs_source)
        start = max(0, n - max_context)

        # [CN] FlatLogprobs 快路径：直接用 start/end 索引取 token_ids，
        #      不必把每个位置都物化成 dict。
        # Efficient path for FlatLogprobs: access token_ids directly.
        if isinstance(logprobs_source, FlatLogprobs):
            return [
                logprobs_source.token_ids[logprobs_source.start_indices[i]]
                for i in range(start, n)
                # [CN] start==end 表示该位置没有记录（例如被截断），跳过。
                if logprobs_source.start_indices[i] < logprobs_source.end_indices[i]
            ]

        # [CN] list[dict] 通用路径：取每个位置 dict 的第一个 key（即采样 token）。
        # list[dict] path
        result: list[int] = []
        for i in range(start, n):
            entry = logprobs_source[i]
            if entry is not None:
                result.append(next(iter(entry)))
        return result

    # [CN] 修正一个含 U+FFFD（替换字符）的解码结果。
    #      返回**当前 token 真正贡献的文本**；如果字节序列确实还没闭合，
    #      返回空串（等下一个字节 token 到来时再补）。
    def _correct_decoded_token(
        self, token_id: int, context_token_ids: list[int]
    ) -> str:
        """Correct a decoded token that contains the replacement character.

        When byte-fallback tokenization splits multi-byte UTF-8
        characters across tokens, individual token decoding produces
        the replacement character U+FFFD. This method uses preceding
        sampled tokens as context to reconstruct the correct text.

        Args:
            token_id: The single token ID to correct.
            context_token_ids: Preceding sampled token IDs in sequential
                order (oldest first). These are the actual tokens in
                the generated sequence, NOT top-k alternatives.

        Returns:
            The corrected decoded string, or empty string if the byte
            sequence is genuinely incomplete at this point.
        """
        assert self.tokenizer is not None

        # [CN] 最多回溯 4 个 token：UTF-8 最长 4 字节，够了。
        max_ctx = min(len(context_token_ids), 4)

        # [CN] 从少到多逐步加长上下文，第一个能拼出完整字符的长度就是答案。
        for num_ctx in range(1, max_ctx + 1):
            context = context_token_ids[-num_ctx:]
            full_decoded = self.tokenizer.decode(context + [token_id])

            # [CN] 拼上上下文后仍以替换字符结尾 → 字节序列还没闭合，换更长的上下文。
            if full_decoded.endswith("�"):
                continue

            # [CN] 定位「干净前缀」的边界：从后往前找**连续**的字节回退 token
            #      （它们单独解码时返回空串），这些 token 的文本应当全部归属给
            #      当前这个「补完」的 token，否则会被重复计算。
            # Find the boundary between "clean" context tokens and
            # byte-fallback tokens that are part of the same incomplete
            # sequence. Byte-fallback context tokens returned "" when
            # they were processed, so their text must be attributed to
            # this completing token.
            clean_end = len(context)
            # [CN] 从最后一个上下文 token 往前扫，遇到非替换字符就停。
            for j in range(len(context) - 1, -1, -1):
                if self.tokenizer.decode([context[j]]).endswith("�"):
                    clean_end = j
                else:
                    break

            # [CN] 只解码干净的那段前缀，用它做「应该切掉多少」的标尺。
            # Decode only the clean (non-byte-fallback) prefix.
            if clean_end > 0:
                clean_prefix = self.tokenizer.decode(context[:clean_end])
            else:
                clean_prefix = ""

            # [CN] 正常情况：拼出来的串确实以干净前缀开头，直接切掉即可。
            if full_decoded.startswith(clean_prefix):
                return full_decoded[len(clean_prefix) :]

            # [CN] 兜底：分词器的归一化行为可能让前缀对不齐，
            #      退而求其次取两者的最长公共前缀再切。
            # Tokenizer normalization may cause prefix mismatch.
            # Find the longest common prefix between them.
            common_len = 0
            for a, b in zip(clean_prefix, full_decoded):
                if a != b:
                    break
                common_len += 1
            return full_decoded[common_len:]

        # [CN] 所有上下文长度都试过了仍不闭合 → 真的是不完整序列，返回空串。
        return ""

    # [CN] 批量校验并修正一批解码文本。
    def _verify_tokens(
        self,
        decoded_tokens_list: list[str],
        tokens: list[int],
        context_token_ids: list[int] | None = None,
    ) -> list[str]:
        """Verify and correct decoded tokens with replacement characters.

        Args:
            decoded_tokens_list: Decoded token strings to verify.
            tokens: Token IDs corresponding to decoded_tokens_list.
                These are alternatives at the SAME position (e.g.
                [sampled, top1, top2]), NOT sequential tokens.
            context_token_ids: Preceding sampled token IDs providing
                sequential context. If None, extracted from
                self.logprobs.
        """
        # [CN] 没显式传上下文 → 自动从 self.logprobs 里取。
        if context_token_ids is None:
            context_token_ids = self._get_sampled_context_ids(self.logprobs)

        # [CN] 先收集要改的下标，最后统一写回。
        #      为什么不边遍历边改：修正逻辑依赖**修正前**的上下文，
        #      就地修改会污染后续位置的判断。
        corrected_decoded_token_map = dict()
        for idx, text in enumerate(decoded_tokens_list):
            # [CN] 只有以替换字符结尾的才需要修 —— 这是字节回退的特征。
            if text.endswith("�"):
                # [CN] 重要：每个候选**独立**修正，只用顺序上下文。
                #      绝不能拿同位置的其它候选当上下文（它们是并列关系，不是前后关系）。
                # Replacement char at the end means a potential
                # unfinished byte sequence from byte-fallback
                # tokenization. Correct each token independently
                # using only the sequential context.
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    tokens[idx], context_token_ids
                )

        # [CN] 统一写回修正后的文本。
        for idx, text in corrected_decoded_token_map.items():
            decoded_tokens_list[idx] = text

        return decoded_tokens_list

    # [CN] 主入口：引擎回一拍输出，按需更新采样/prompt 两类 logprobs。
    def update_from_output(self, output: EngineCoreOutput) -> None:
        # [CN] 用 `is not None` 短路：不启用 logprobs 的请求这里零开销。
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
