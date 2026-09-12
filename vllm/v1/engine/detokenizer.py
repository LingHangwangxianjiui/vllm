# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

# [CN] 文件总览：**增量解码器** —— 把 token id 流变成文本流。
#
#     为什么必须「增量」：流式输出要求每来一个 token 就吐一小段文本，
#     但不能每次都把整个序列重新 decode 一遍（O(n^2)）。而且很多 tokenizer
#     （BPE / SentencePiece）的解码结果**依赖前文**：一个 token 可能只是
#     一个字节片段，必须攒齐才能拼出正确的 UTF-8 字符。
#
#     ========================= 三种实现 =========================
#       IncrementalDetokenizer       —— 空实现（不需要文本时用它占位）；
#       FastIncrementalDetokenizer   —— 用 tokenizers 库的 DecodeStream
#                                      （原生增量，含 prompt 预填充）；
#       SlowIncrementalDetokenizer   —— 纯 Python 的 prefix/read offset 方案，
#                                      兼容任意 TokenizerLike。
#      选择逻辑见 IncrementalDetokenizer.from_new_request。
#
#     ========================= 两个隐藏难点 =========================
#     1) **停止字符串检测**：用户在文本层面定义 stop（「\n\n」之类），
#        而引擎是数 token 停的。若 stop 串要被**排除**在输出之外，
#        就必须把末尾若干字符**扣住不发**（stop_buffer_length），
#        否则一旦发出去就收不回来了。
#     2) **min_tokens**：在达到最少生成 token 数之前，
#        stop 字符串检测要被推迟（stop_check_offset 不断后移）。

from abc import ABC, abstractmethod

import tokenizers
import tokenizers.decoders
from packaging import version
from tokenizers import Tokenizer
from transformers import TokenizersBackend

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# [CN] 只有 tokenizers >= 0.22.0 才支持带原生 prefill（ids 参数）的
#      DecodeStream，而 FastIncrementalDetokenizer 依赖它。
# Only tokenizers >= 0.22.0 supports DecodeStream with native prefill
# (ids parameter) used for FastIncrementalDetokenizer.
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


# [CN] 兜底实现：不做任何解码，永远返回空串。
#      用于 detokenize=False 或没有 tokenizer 的场景 —— 让用户代码
#      不必到处写 if detokenizer is not None。
class IncrementalDetokenizer:
    def __init__(self):
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids)

    # [CN] 基类版本只记 token id，不产出文本。
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        self.token_ids.extend(new_token_ids)
        return None

    # [CN] 基类版本永远返回空文本。
    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    # [CN] 工厂：按条件三选一。
    #      ① 没 tokenizer → 空实现；
    #      ② tokenizers 够新且是 TokenizersBackend → 快的；
    #      ③ 其余 → 慢的（兼容性最好）。
    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":
        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)


# [CN] 两种真实实现的公共基类（ABC）：负责停止串相关状态与 update 的骨架，
#      把「怎么解出下一个 token 的文本」留给子类 decode_next()。
class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    # [CN] 构造：解析停止串配置，并算出「要扣住多少字符」。
    def __init__(self, request: EngineCoreRequest):
        super().__init__()

        # [CN] 停止串归一化成 list（用户可能传 str 或 list[str]）。
        # Stop strings
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # [CN] **扣留缓冲长度**：若 stop 串要被排除出输出，
        #      末尾最多可能藏着一个「尚未完整的 stop 串」，
        #      长度最多是 max(len(s)) - 1，这部分先不发。
        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        # [CN] DELTA 模式下「上次发到哪」的字符下标。
        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""

    # [CN] 喂入新 token，返回匹配到的停止串（没有则 None）。
    #      两步：① 增量解码 ② 停止串判定。
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        # [CN] 没有新 token（例如被抢占回滚）→ 直接返回，不做解码。
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        # [CN] 引擎判定为「停止」且用户不要 stop 串 → 
        #      把最后一个 token 从**解码**里排除，但仍然记进 token_ids，
        #      这样返回的 token id 列表是完整的，只有文本不含它。
        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # [CN] ① 逐个 token 增量解码，拼到 output_text。
        # 1) Detokenize the new token ids incrementally.
        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # [CN] min_tokens 未满足时不断后移检测起点 —— 
            #      等价于「这段时间内不检查停止串」。
            # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                stop_check_offset = len(self.output_text)

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            self.token_ids.append(skipped_stop_token_id)

        # [CN] ② 停止串判定。只在超过 min_tokens 之后才检查。
        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            # [CN] 只检查**新产生的字符**区间（new_char_count），
            #      避免重复扫描整段历史。
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    # [CN] 子类必须实现：给定下一个 token id，返回它的增量文本。
    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        raise NotImplementedError

    # [CN] 取本次要发给用户的文本。
    #
    #      两个维度：
    #        delta    —— 只返回「上次之后的新增部分」还是全量；
    #        finished —— 是否结束。结束时**不再扣留**缓冲，
    #                    因为已经不会有新的字符来「补全」stop 串了。
    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        # [CN] 结束了就不再扣留末尾字符（buffer_length = 0）。
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            if not buffer_length:
                return self.output_text
            return self.output_text[:-buffer_length]

        # [CN] DELTA：返回 [上次 offset, 本次长度) 这一段。
        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


# [CN] 快路径：直接用 tokenizers 库的 DecodeStream。
#      它内部维护 UTF-8 解码状态，天然支持「按 token 喂、按字符吐」。
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: TokenizersBackend, request: EngineCoreRequest):
        super().__init__(request)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        # [CN] 注意这里拿的是 backend 内部的 _tokenizer（真正的 Tokenizer 对象）。
        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # [CN] **原生 prefill**：直接用 prompt token ids 给 DecodeStream 打底，
        #      比一个个 step 快得多。
        #      从 module 上查 DecodeStream 而不是直接 import 名字，
        #      这样第三方补丁（如 fastokens shim）无论导入顺序如何都生效。
        # Use native prefill to prime the decode stream with prompt tokens.
        # Look up DecodeStream on the module so backend patches (e.g. the
        # fastokens shim that replaces ``tokenizers.decoders.DecodeStream``)
        # are honored regardless of import order.
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        # [CN] 是否需要在特殊 token 之间插入空格。
        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )

        # [CN] 不需要空格时，要额外记录「上一个是不是特殊 token」，
        #      以便手动抑制相邻特殊 token 之间的空格。
        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            if added_token_ids is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                self.spaces_between_special_tokens = True

    # [CN] 解码一个 token，并处理「相邻特殊 token 之间不要空格」的细节。
    def decode_next(self, next_token_id: int) -> str:
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                token = special_token
            self.last_special = is_special

        return token or ""

    # [CN] **带兜底的 step**：tokenizers 的 DecodeStream 在某些边界下会抛异常，
    #      这里分两类处理：
    #        · Overflow / TypeError —— 极少数非法 token id，记日志后跳过；
    #        · Invalid prefix       —— tokenizer 吐出了非法 UTF-8，
    #          把 DecodeStream 的**内部状态搞坏了**，只能重建一个。
    #          注意重建后没有 prompt prefill（状态已丢失），属于可接受的降级。
    def _protected_step(self, next_token_id: int) -> str | None:
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except (OverflowError, TypeError):
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            logger.exception("Encountered invalid token id: %r", next_token_id)
            token = None
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            self.stream = tokenizers.decoders.DecodeStream(
                skip_special_tokens=self.skip_special_tokens
            )
            token = self.stream.step(self.tokenizer, next_token_id)
        return token


# [CN] 慢路径（纯 Python）：自己维护 prefix_offset / read_offset 两个游标，
#      兼容任何 TokenizerLike（包括 transformers 的慢 tokenizer）。
class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):
        super().__init__(request)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        # [CN] prompt 长度：慢路径把 prompt token 也存进 token_ids，
        #      所以要记住边界才能区分「prompt」和「输出」。
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )

        # [CN] 增量解码的三件套状态：prefix_offset / read_offset 是
        #      detokenize_incrementally 用来避免重复解码的游标。
        # Metadata for incremental detokenization.
        if request.prompt_token_ids is not None:
            self.tokens, self.prefix_offset, self.read_offset = (
                convert_prompt_ids_to_tokens(
                    tokenizer=tokenizer,
                    prompt_ids=request.prompt_token_ids,
                    skip_special_tokens=params.skip_special_tokens,
                )
            )
        # [CN] prompt embedding 请求没有 token id，
        #      无法 detokenize —— 用空串占位，输出文本为空。
        else:
            # Prompt embedding requests cannot be detokenized, in general.
            self.tokens = [""] * self.prompt_len
            self.prefix_offset = 0
            self.read_offset = 0

        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = params.spaces_between_special_tokens

    # [CN] 慢路径把 prompt token 也放进 token_ids，
    #      所以对外要**切掉 prompt 部分**才是真正的输出 token。
    @property
    def output_token_ids(self) -> list[int]:
        if self.prompt_len:
            return self.token_ids[self.prompt_len :]
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    # [CN] 调 detokenize_incrementally 拿增量文本，并推进两个游标。
    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,
            all_input_ids=self.token_ids,
            prev_tokens=self.tokens,
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text


# [CN] 停止串检测。返回 (匹配到的 stop 串, 截断位置)；
#      截断位置 -1 表示不需要截断。
#
#      为什么需要「选最早完成的那一个」：投机解码下一拍可能一次追加多个 token，
#      文本里可能同时出现多个 stop 串。取**最早结束**的那个，才能与
#      「一个一个 token 追加」的结果保持一致（同分则按 stop 列表顺序）。
def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.

    When several stop strings match within the newly generated text (for
    example when speculative decoding appends multiple tokens in a single
    step), the stop string that completes earliest in the text is selected,
    so the result matches appending one token at a time. Ties are broken by
    stop-list order.
    """
    # [CN] 没有新字符或没有 stop 串 → 无需检测。
    if not new_char_count or not stop:
        return None

    # [CN] 遍历所有 stop 串，找「完成位置最靠前」的那个。
    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        # [CN] 只在新字符区间里搜索。起点是
        #      1 - new_char_count - len(stop_str)：
        #      stop 串可能**跨越**本次新增与历史的边界，
        #      所以要从「能完整容纳该 stop 串的最早位置」开始找。
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        # [CN] 用「结束位置」而不是「起始位置」比较 —— 
        #      这才是「最早完成」的正确定义。
        # Prefer the stop string that completes earliest in the text.
        end = stop_index + stop_string_len
        if end < best_end:
            best_stop_str = stop_str
            best_stop_index = stop_index
            best_end = end

    if best_stop_str is None:
        return None

    # [CN] 要把 stop 串**保留**在输出里 → 截断到它的结束位置。
    if include_in_output:
        # Truncate to end of stop string.
        if best_end >= len(output_text):
            # No truncation required.
            return best_stop_str, -1
        return best_stop_str, best_end

    # [CN] 要把 stop 串**排除** → 截断到它的起始位置。
    # Truncate the output text to the beginning of the stop string.
    return best_stop_str, best_stop_index
