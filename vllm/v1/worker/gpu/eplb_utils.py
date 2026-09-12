# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# [CN] 文件总览：EPLB = Expert Parallelism Load Balancing。
# [CN] MoE 模型中各专家的负载会随输入漂移，EPLB 周期性重排
# [CN] 「逻辑专家 -> 物理专家」的映射，让各 EP rank 负载均衡。
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    get_mixture_of_experts_model,
)

logger = init_logger(__name__)


# [CN] 装饰器：在某个 runner 方法「成功返回后」推进一次 EPLB。
# [CN] 放在方法之后而不是之前，是为了让重排基于本步真实的专家统计。
# [CN] dummy run 也会走（is_dummy=True），否则各 DP rank 节奏不一致会挂死。
def step_eplb_after(*, is_dummy: bool = False) -> Callable:
    """Step EPLB after a model runner method completes successfully."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self: Any, *args, **kwargs) -> Any:
            result = fn(self, *args, **kwargs)
            if kwargs.get("skip_eplb", False):
                return result

            is_profile = kwargs.get("is_profile", False) if is_dummy else False
            self.eplb.step(is_dummy=is_dummy, is_profile=is_profile)
            return result

        return wrapper

    return decorator


# [CN] 把「是否启用、是否已注册模型、是否被抑制」这些散落判断收拢到一处，
# [CN] 让 runner 主链路只调 self.eplb.step() 即可。
class EPLBController:
    def __init__(self, parallel_config: Any, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.state: EplbState | None = None
        self.suppressed = False
        self._has_registered_models = False

    def prepare_load(self) -> None:
        self.state = None
        self._has_registered_models = False
        if self.parallel_config.enable_eplb:
            self.state = EplbState(self.parallel_config, self.device)

    # [CN] 草稿模型若也是 MoE，也要纳入 EPLB；
    # [CN] 但 elastic EP 与草稿模型不兼容，这里直接断言拒绝。
    def maybe_register_speculator(
        self,
        speculator: Any | None,
        speculative_config: Any | None,
        load_dummy_weights: bool,
    ) -> bool:
        # if speculator is a moe model, add it to eplb
        if (
            speculator is None
            or not hasattr(speculator, "model")
            or not self.parallel_config.enable_eplb
            or load_dummy_weights
        ):
            return False

        draft_model = speculator.model
        draft_moe_model = get_mixture_of_experts_model(draft_model)
        if draft_moe_model is None:
            return False

        assert not self.parallel_config.enable_elastic_ep, (
            "Elastic EP is not supported with draft model."
        )
        assert speculative_config is not None
        assert speculative_config.draft_model_config is not None
        assert self.state is not None
        self.state.add_model(
            draft_moe_model,
            speculative_config.draft_model_config,
        )
        speculator.set_eplb_state(self.state)
        self._has_registered_models = True
        return True

    def maybe_register_model(
        self,
        model: nn.Module,
        model_config: Any,
        load_dummy_weights: bool,
    ) -> bool:
        if not self.parallel_config.enable_eplb or load_dummy_weights:
            return False

        moe_model = get_mixture_of_experts_model(model)
        if moe_model is None:
            return False

        logger.info_once(
            "EPLB is enabled for MoE part of model %s.", model_config.model
        )
        assert self.state is not None
        self.state.add_model(moe_model, model_config)
        self._has_registered_models = True
        return True

    # [CN] 异步模式下另起线程做重排决策，不阻塞前向。
    # [CN] 只有真的注册了模型才需要启动。
    def maybe_start_async_loop(self, eplb_models_added: bool) -> None:
        if eplb_models_added and self.state is not None and self.state.is_async:
            self.state.start_async_loop()

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
    ) -> None:
        if (
            not self.parallel_config.enable_eplb
            or self.suppressed
            or self.state is None
            or not self._has_registered_models
        ):
            return

        self.state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def prepare_forward(
        self,
        model_config: ModelConfig,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
    ) -> None:
        if self.state is None or not self.parallel_config.enable_eplb:
            return
        self.state.prepare_forward(model_config, num_unpadded_tokens, ubatch_slices)

    # [CN] 从外部给定的「物理->逻辑」映射直接重建（用于弹性 EP / 扩缩容）。
    def setup_from_mapping(
        self,
        model: nn.Module,
        model_config: Any,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        moe_model = get_mixture_of_experts_model(model)
        assert moe_model is not None

        self.state = EplbState.from_mapping(
            model=moe_model,
            model_config=model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
        )
        self._has_registered_models = True
