# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus


# [CN] 判断序列**尾部**是否出现了长度为 pattern_len 的重复模式。
#      做法：拿最后 pattern_len 个 token，跟前面 (min_count-1) 段比对，
#      每一段的对应位置都必须相同。
def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


# [CN] 重复（幻觉）检测：从 min_pattern_size 到 max_pattern_size 逐个试，
#      只要有一种周期命中就判定为重复 -> 结束请求。
#      注意 pattern_len * min_count > len(token_ids) 时直接返回 False：
#      序列还不够长，不可能出现这么长的重复。
def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


# [CN] 从 list 里删元素。**删单个时用原地 remove（快路径）**，
#      删多个时才用列表推导重建。
#      注意返回值语义不一致：单个时返回**原 list（已原地修改）**，
#      多个时返回**新 list** —— 所以调用方必须用返回值，
#      不能假设是原地修改。这是 Python 里很容易踩的坑。
def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


# [CN] 判断请求是否该停。判定顺序**不能随便改**（注释里列了三个 PR）：
#        1) eos / stop_token_ids —— 模型自己说要停，优先级最高；
#        2) 长度上限（max_model_len / max_tokens）；
#        3) **min_tokens 检查放在长度之后、重复检测之前** ——
#           还没到 min_tokens 就直接返回 False（不检查重复），
#           否则模型刚开头就被判重复而终止，min_tokens 形同虚设。
def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # Note(arpera):
    # Order of checks is important for min_tokens
    # If you decide to change the order, first look at these PRs please:
    # 47489, 49521, and 51299
    # These PRs show some reasoning about the order of checks
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
