# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：LoRA 与 CUDA graph 的耦合处理。
# [CN] 难点：LoRA 会让 CUDA graph 的输入形状与 kernel 选择发生变化，
# [CN] 因此「激活了几个 LoRA」本身就是 graph 捕获的一个维度。
"""LoRA utilities for the Model Runner V2 and cudagraph."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_captured_lora_counts

if TYPE_CHECKING:
    from vllm.config.compilation import CompilationConfig
    from vllm.config.lora import LoRAConfig

# [CN] 0 保留为「无 LoRA」的哨兵 ID（与 LoRARequest.lora_int_id 从 1 开始配套）。
NO_LORA_ID = 0


# [CN] 枚举需要为哪些「激活 LoRA 数」各捕一套图：
# [CN]   开启 specialize —— 2 的幂 + max_loras+1（精细但图多）；
# [CN]   否则            —— 只要 [0, max_loras+1] 两套（粗但省显存）；
# [CN]   未启用 LoRA     —— 只要 [0]。
def get_lora_capture_cases(
    lora_config: "LoRAConfig | None",
    compilation_config: "CompilationConfig",
) -> list[int]:
    """
    Return num_active_loras values for cudagraph capture.

    When cudagraph_specialize_lora=True: powers of 2 up to max_loras, plus
    max_loras+1. When False: [0, max_loras+1]. When LoRA disabled: [0].
    """
    if lora_config is None:
        return [0]
    if compilation_config.cudagraph_specialize_lora:
        specialize = getattr(lora_config, "specialize_active_lora", False)
        captured = get_captured_lora_counts(lora_config.max_loras, specialize)
        return [0] + [c for c in captured if c > 0]
    return [0, lora_config.max_loras + 1]


# [CN] 运行期决定本步命中哪套图。
# [CN] dummy run 时一律按 max_loras+1（最坏情况）选，保证捕获与运行一致。
def get_num_active_loras_for_dispatch(
    lora_config: "LoRAConfig | None",
    lora_state: "LoraState",
    req_ids: list[str],
    dummy_run: bool,
) -> int:
    """Compute num_active_loras for cudagraph dispatch."""
    if lora_config and not dummy_run:
        return len(lora_state.get_activate_loras(req_ids))
    if dummy_run and lora_config:
        return lora_config.max_loras + 1
    return 0


# [CN] 捕获图之前的钩子：按目标「激活数」挑一组 dummy LoRA 装上，
# [CN] 使捕获到的图里 LoRA 权重的形状与运行期一致。
def create_lora_capture_hook(
    lora_config: "LoRAConfig | None",
    runner: Any,
) -> Callable[[int, int, int], None] | None:
    """Create a hook to set up LoRA state before each cudagraph capture."""
    if lora_config is None:
        return None

    def hook(num_active_loras: int, num_reqs: int, num_tokens: int) -> None:
        # Match InputBatch.make_dummy: distribute the remainder evenly so no
        # dummy request exceeds ceil(num_tokens / num_reqs) tokens.
        num_scheduled = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_extra = num_tokens % num_reqs
        if num_extra > 0:
            num_scheduled[-num_extra:] += 1
        with runner.maybe_select_dummy_loras(
            lora_config, num_scheduled, num_active_loras=num_active_loras
        ):
            pass

    return hook


# [CN] 每请求的 LoRA 状态：slot -> lora_int_id，以及 req_id -> LoRARequest。
class LoraState:
    def __init__(self, max_num_reqs: int):
        self.lora_ids = np.zeros(max_num_reqs, dtype=np.int32)
        self.lora_ids.fill(NO_LORA_ID)
        # req_id -> lora_request
        self.lora_requests: dict[str, LoRARequest] = {}

    def add_request(
        self, req_id: str, req_index: int, lora_request: LoRARequest | None
    ) -> None:
        if lora_request is not None:
            self.lora_requests[req_id] = lora_request
            self.lora_ids[req_index] = lora_request.lora_int_id
        else:
            self.lora_ids[req_index] = NO_LORA_ID

    def remove_request(self, req_id: str) -> None:
        self.lora_requests.pop(req_id, None)

    # [CN] 生成两种映射：
    # [CN]   prompt_lora_mapping —— 每请求一个 id（给 punica 的 prompt 维）；
    # [CN]   token_lora_mapping  —— 每 token 一个 id（按 num_scheduled_tokens 展开）。
    def make_lora_inputs(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        lora_ids = self.lora_ids[idx_mapping]
        prompt_lora_mapping = tuple(lora_ids)
        token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))
        active_lora_requests: set[LoRARequest] = self.get_activate_loras(req_ids)
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    def get_activate_loras(self, req_ids: list[str]) -> set[LoRARequest]:
        active_lora_requests: set[LoRARequest] = set()
        for req_id in req_ids:
            lora_request = self.lora_requests.get(req_id)
            if lora_request is not None:
                active_lora_requests.add(lora_request)
        return active_lora_requests
