# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ==============================================================================
# 本文件职责：定义 vLLM 的顶层配置容器 VllmConfig，以及配套的"当前配置"全局访问工具。
#
# 在系统链路中的位置（控制面，进程启动时构建一次，之后只读）：
#   EngineArgs(CLI / Python API)
#     -> 【本文件 VllmConfig】聚合所有子配置
#     -> ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig /
#        DeviceConfig / LoadConfig / LoRAConfig / SpeculativeConfig /
#        CompilationConfig / KVTransferConfig / ObservabilityConfig ...
#     -> LLMEngine / V1 EngineCore -> Executor -> Worker -> ModelRunner
#
# 核心内容速查：
#   - OptimizationLevel / OPTIMIZATION_LEVEL_* / OPTIMIZATION_LEVEL_TO_CONFIG
#       : 优化等级（-O0..-O3）及其对应的一整套默认值（多为 callable 判定）
#   - enable_*_fusion(cfg)      : 各融合 pass 是否启用的运行时判定函数
#   - VllmConfig                : 总配置类，聚合全部子配置并做跨配置联合校验
#   - VllmConfig.compute_hash() : 影响计算图的配置指纹（编译/cudagraph 缓存 key）
#   - VllmConfig.with_hf_config(): 基于已有配置派生一份替换了 hf_config 的新配置
#   - VllmConfig.__post_init__(): 跨配置联合校验 + 默认值推导的总入口
#   - try_verify_and_update_config() : 让 model 侧钩子回写本配置
#   - _set_cudagraph_sizes() / _set_compile_ranges() : 形状相关的尺寸推导
#   - set_current_vllm_config() / get_current_vllm_config() : 全局"当前配置"上下文
#
# 阅读提示：
#   1. 字段分组：model_config 是唯一没有默认值的必需字段；多数子配置用
#      Field(default_factory=...) 给出默认实例；LoRA / Speculative / Diffusion /
#      KVTransfer / KVEvents / ECTransfer / Reasoning / quant_config /
#      weight_transfer_config 等默认为 None，表示该特性未启用。
#   2. 跨配置的联合校验集中在 __post_init__ 及其调用的 _verify_* / _validate_* /
#      _resolve_* 私有方法中；新增约束应就近加在这里，而不是散落到各子配置。
#   3. __post_init__ 会**就地修改**子配置（推导默认值、降级不兼容开关），所以
#      "用户显式传入的值"与"最终生效的值"可能不同，调试请以最终对象为准。
#   4. 序列化与哈希约定：本配置是 pydantic dataclass（见 .utils.config 装饰器），
#      可直接由 pydantic/msgspec 序列化；参与指纹计算的子配置都要实现
#      compute_hash()，additional_config 若为 dict 则用
#      json.dumps(sort_keys=True) 归一化后再哈希，自定义类型需实现 SupportsHash。
# ==============================================================================

import copy
import getpass
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import is_dataclass
from datetime import datetime
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

import torch
from pydantic import ConfigDict, Field, model_validator

import vllm.envs as envs
from vllm.logger import enable_trace_function_call, init_logger
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.triton_utils import HAS_TRITON
from vllm.utils import random_uuid
from vllm.utils.hashing import safe_hash

from .attention import AttentionConfig
from .cache import CacheConfig
from .compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from .device import DeviceConfig
from .diffusion import DiffusionConfig
from .ec_manager_config import EncoderCacheManagerConfig
from .ec_transfer import ECTransferConfig
from .kernel import KernelConfig
from .kv_events import KVEventsConfig
from .kv_transfer import KVTransferConfig
from .load import LoadConfig
from .lora import LoRAConfig
from .mamba import MambaBackendEnum, MambaConfig
from .model import ModelConfig
from .observability import ObservabilityConfig
from .offload import OffloadConfig
from .parallel import ParallelConfig
from .profiler import ProfilerConfig
from .reasoning import ReasoningConfig
from .scheduler import SchedulerConfig
from .speculative import EagleModelTypes, NgramGPUTypes, SpeculativeConfig
from .structured_outputs import StructuredOutputsConfig
from .utils import SupportsHash, config, replace
from .weight_transfer import WeightTransferConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
else:
    PretrainedConfig = Any

    QuantizationConfig = Any

    KVCacheConfig = Any

logger = init_logger(__name__)

# TODO(rocm): These models are either unsupported by MRV2 or slower with
# MRV2 on AMD GPUs.
# 这些架构在 ROCm 上要么不被 MRV2 支持、要么更慢，命中后 use_v2_model_runner
# 会自动退回 V1 model runner（而不是报错）。
ROCM_DEFAULT_MRV1_ARCHITECTURES = frozenset(
    {"DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM", "GlmMoeDsaForCausalLM"}
)

# 默认开启 breakable CUDA graph 的架构白名单：基本都是大模型 / MTP(草稿)架构，
# 它们用完整图捕获代价过高，切分后收益明显。平台差异见下面的函数。
DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = frozenset(
    {
        "DeepseekV32MTPModel",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "DeepseekV4ForConditionalGeneration",
        "DeepSeekV4MTPModel",
        "Dots3NoteForCausalLM",
        "Dots3NoteMTPModel",
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
        "Glm5NextMTPModel",
        "GlmMoeDsaForCausalLM",
        "HYV4ForCausalLM",
        "HYV4MTPModel",
        "InklingForCausalLM",
        "InklingForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        "KimiK3MTPModel",
        "KimiLinearForCausalLM",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    }
)


# 按平台返回"默认启用 breakable CUDA graph"的架构集合；lru_cache 保证
# current_platform 的探测只做一次。返回空集表示本平台默认不启用（用户仍可
# 用 VLLM_USE_BREAKABLE_CUDAGRAPH=1 强制打开）。
@lru_cache
def default_breakable_cudagraph_architectures() -> frozenset[str]:
    """Architectures defaulting to breakable CUDA graphs on this platform."""
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        # Breakable CUDA graphs currently regress performance on ROCm, so no
        # architecture opts in by default here. Users can still force it with
        # VLLM_USE_BREAKABLE_CUDAGRAPH=1.
        return frozenset()
    return DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES


# 优化等级：用"启动耗时"换"运行性能"。O0 启动最快、O3 性能最好，默认 O2。
# 每个等级对应 OPTIMIZATION_LEVEL_TO_CONFIG 中的一组默认值，只覆盖用户没显式
# 设置的字段（见 _apply_optimization_level_defaults）。
class OptimizationLevel(IntEnum):
    """Optimization level enum."""

    O0 = 0
    """O0 : No optimization. no compilation, no cudagraphs, no other
    optimization, just starting up immediately"""
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor compilation and Piecewise
    cudagraphs"""
    O2 = 2
    """O2: Full optimizations. -O1 as well as Full and Piecewise cudagraphs."""
    O3 = 3
    """O3: Currently the same as -O2s."""


PerformanceMode = Literal["balanced", "interactivity", "throughput"]

# 【两个全局"是否启用"常量】本意是让优化开关随模型属性动态判定，目前恒为 False。
# 注释里的 lambda 才是原始设计：按模型是否量化 / 是否为 MoE 决定启用与否。
# 因为相关优化在这两项上还不稳定（见 issue 25689），统一关闭。
# 它们是 bool 而非函数，所以 OPTIMIZATION_LEVEL_02/03 里 fuse_attn_quant、enable_sp、
# fuse_gemm_comms 三项实际恒为 False——读配置时不要误以为它们会按模型自动打开。
IS_QUANTIZED = False
IS_DENSE = False
# The optimizations that depend on these properties currently set to False
# in all cases.
# if model_config is not None:
#     IS_QUANTIZED = lambda c: c.model_config.is_quantized()
#     IS_DENSE = lambda c: not c.model_config.is_model_moe()
# See https://github.com/vllm-project/vllm/issues/25689.


# 【融合开关族】下面这组 enable_*_fusion(cfg) 是"运行期判定函数"，不是常量。
# 它们被塞进 OPTIMIZATION_LEVEL_0x 字典的值位置：解析配置时若发现值是 callable，
# 就把当前 VllmConfig 传进去求值，得到该模型/平台下此融合是否真的可用。
# 这样同一份 -O2 默认表在 CUDA、ROCm、不同模型上会得出不同结果。
# 核心思路：只有当相应的 custom op（手写的融合算子）真的被启用时才开这个 pass，
# 否则交给 Inductor 在编译期自己融合，避免重复优化。


def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable if either RMS norm or quant FP8 custom op is active;
    otherwise Inductor handles fusion."""

    return (
        cfg.compilation_config.is_custom_op_enabled("rms_norm")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or cfg.kernel_config.ir_op_priority.rms_norm[0] != "native"
    )


def enable_act_fusion(cfg: "VllmConfig") -> bool:
    """
    Enable if either SiLU+Mul or quant FP8 custom op is active;
    otherwise Inductor handles fusion.
    Also enable for FP4 models as FP4 quant is always custom so Inductor cannot fuse it.
    """
    return (
        cfg.compilation_config.is_custom_op_enabled("silu_and_mul")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or (cfg.model_config is not None and cfg.model_config.is_nvfp4_quantized())
    )


# 把"张量并行 all-reduce"与后面的 RMSNorm 合成一个 kernel，省一次全局同步与显存往返。
# 前提很苛刻：必须真的有多卡（TP>1），且是 CUDA 上的 Hopper(90)/Blackwell(100) 家族，
# 并装了 flashinfer；ROCm 走 AITER 的等价路径。
# 另外它破坏了 batch-invariance（结果随 batch 组成变化），因此该模式下一律关闭。
def enable_allreduce_rms_fusion(cfg: "VllmConfig") -> bool:
    """Enable if TP > 1 and Hopper/Blackwell and flashinfer installed."""
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import has_flashinfer

    # The fused all-reduce + RMSNorm path is not batch-invariant
    if envs.VLLM_BATCH_INVARIANT:
        return False

    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        return (
            rocm_aiter_ops.is_enabled() and cfg.parallel_config.tensor_parallel_size > 1
        )

    return (
        cfg.parallel_config.tensor_parallel_size > 1
        and current_platform.is_cuda()
        and has_flashinfer()
        and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability(90)
        )
    )


# 下面四个是 ROCm/AITER 专属的融合开关，目标都是同一件事：
# 把「RoPE 位置编码 → 写 KV Cache」这一串小算子合成一个 kernel，减少访存往返。
# 共同前提是 AITER 算子库已启用；差别在于各自还要满足的额外条件：
#   - enable_rope_kvcache_fusion      : 还要 rotary_embedding custom op 生效
#   - enable_rope_kvcache_mla_fusion  : 只要求图切分方式允许（MLA 路径）
#   - enable_mla_dual_rms_norm_fusion : 仅 AITER 即可（MLA 的双 RMSNorm）
#   - enable_qk_norm_rope_kvcache     : 再把 QK-Norm 也并进来
# "use_inductor_graph_partition or not splitting_ops_contain_kv_cache_update()"
# 这个判据反复出现，含义是：只要没有被拆分算子把 KV 更新切到图外，就可以安全融合。
def enable_rope_kvcache_fusion(cfg: "VllmConfig") -> bool:
    """Enable if rotary embedding custom op is active and
    use_inductor_graph_partition is enabled.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    return (
        rocm_aiter_ops.is_enabled()
        and cfg.compilation_config.is_custom_op_enabled("rotary_embedding")
        and (
            cfg.compilation_config.use_inductor_graph_partition
            or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
        )
    )


def enable_rope_kvcache_mla_fusion(cfg: "VllmConfig") -> bool:
    """Enable if use_inductor_graph_partition is enabled."""

    return (
        cfg.compilation_config.use_inductor_graph_partition
        or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
    )


def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
    """Enable if using AITER RMSNorm and hidden size is 2880 i.e. gpt-oss."""

    return (
        cfg.kernel_config.ir_op_priority.fused_add_rms_norm[0] == "aiter"
        and cfg.model_config is not None
        and cfg.model_config.get_hidden_size() == 2880
    )


def enable_mla_dual_rms_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable MLA dual RMS norm fusion on ROCm with AITER."""
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.is_enabled()


def enable_qk_norm_rope_kvcache(cfg: "VllmConfig") -> bool:
    """Enable fused QK-norm + RoPE + KV cache update on ROCm with AITER."""
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.is_enabled():
        return False
    return cfg.compilation_config.is_custom_op_enabled("rotary_embedding")


# 【优化等级默认表】四个字典分别对应 -O0..-O3，结构是"嵌套的配置字段路径 → 默认值"。
# 键是子配置名（如 "compilation_config"），值是它的字段字典，可继续嵌套到 pass_config。
# 值有两种形态：
#   1) 直接量（True/False/CUDAGraphMode.XXX）——写死，所有平台一致；
#   2) callable（enable_*_fusion 等）——延后到配置装配时传入 VllmConfig 求值。
# 重要：这些只是"默认值"，只作用于用户没有显式指定的字段（见 _apply_optimization_level_defaults）。
# 所以 -O2 里 fuse_allreduce_rms 写的是 enable_allreduce_rms_fusion，在单卡上求值为 False。
# 各等级递进关系：O0 全关（启动最快）→ O1 开编译+piecewise cudagraph →
# O2 再加 full cudagraph 与更多融合（默认）→ O3 目前与 O2 相同。
OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": False,
            "fuse_mla_dual_rms_norm": False,
            "fuse_rope_kvcache": False,
            "fuse_qk_norm_rope_kvcache": False,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": False,
    },
}
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": False,
            "fuse_qk_norm_rope_kvcache": False,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": False,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
            "fuse_qk_norm_rope_kvcache": enable_qk_norm_rope_kvcache,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": enable_rope_kvcache_mla_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
            "fuse_qk_norm_rope_kvcache": enable_qk_norm_rope_kvcache,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": enable_rope_kvcache_mla_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}

OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


# 【VllmConfig：全项目唯一的"配置根对象"】
# 它是一个 pydantic dataclass（@config 装饰器见 .utils.config），把十几个子配置聚合在一起，
# 目的只有一个：让代码各处只需传递这一个对象，而不用传一堆互相耦合的配置。
#
# 字段分三类，判断依据是"默认值形态"：
#   1) 必填无默认：只有 model_config（构造空 ModelConfig 会触发下载，所以不能给 default_factory）
#   2) 默认实例：cache/parallel/scheduler/device/load/compilation/kernel/attention/mamba/
#      structured_outputs/observability/ec_manager/offload 等，用 Field(default_factory=...)
#      —— 每次构造都新建实例，避免多个 VllmConfig 共享同一份可变子配置。
#      注意 scheduler_config 用的是 SchedulerConfig.default_factory（自定义工厂），
#      因为它需要根据其他信息推导默认值，不是简单的无参构造。
#   3) 默认 None：lora/speculative/diffusion/quant/kv_transfer/kv_events/ec_transfer/
#      reasoning/weight_transfer —— None 语义是"该特性未启用"，不是"用默认值"。
#
# 生命周期：进程启动时装配一次 → __post_init__ 做跨配置校验与默认值推导（会就地改写子配置）
#          → 之后全程只读。调试时请以"最终对象"为准，而不是你传入的参数。
@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # ==========================================================================
    # [CN] VllmConfig 总览：整个 vLLM 配置体系的**根节点**。
    #
    # 它做三件事：
    #   1) **聚合**：把十几个子配置（ModelConfig / CacheConfig / ParallelConfig /
    #      SchedulerConfig / ...）收成一个对象，代码里只传 VllmConfig 一个参数；
    #   2) **联合校验与推导**：子配置各自只管自己，跨配置的约束（显存预算、并行度、
    #      投机解码与模型的一致性、KV 传输与调度策略的匹配）全在 __post_init__ 里做；
    #   3) **指纹**：compute_hash() 产出"计算图指纹"，供 torch.compile 缓存与
    #      CUDA Graph 缓存复用（见下方 compute_hash 的说明）。
    #
    # 字段分三组，理解分组是读这个类的前提：
    #   A. 子配置字段（model_config ... reasoning_config）
    #      —— 绝大多数用 `Field(default_factory=XxxConfig)` 延迟构造，
    #         因为很多子配置类在 __post_init__ 里会做探测（读环境变量、查硬件），
    #         写成 `= XxxConfig()` 会在**类定义时**就执行一次，既慢又有副作用。
    #   B. 顶层标量（instance_id / optimization_level / performance_mode /
    #      shutdown_timeout ...）—— 不属于任何子配置，由 VllmConfig 自己持有。
    #   C. 可选子配置（lora_config / speculative_config / diffusion_config /
    #      quant_config / kv_transfer_config / kv_events_config / ec_transfer_config /
    #      reasoning_config / weight_transfer_config）
    #      —— 默认 None，语义是"该功能未启用"，下游代码统一用 `is not None` 判断。
    #
    # 生命周期提示：
    #   - 由 EngineArgs.create_engine_config() 构造（见 engine/arg_utils.py）；
    #   - 构造后通常通过 set_current_vllm_config() 放进上下文变量，
    #     模型层用 get_current_vllm_config() 取（这样不用层层传参）；
    #   - 运行期**不应修改**，改了也不会重新触发校验。
    # ==========================================================================

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    # [CN] 唯一的例外：model_config 直接赋 None（带 type: ignore），
    #      不能用 default_factory=ModelConfig —— 因为构造一个默认 ModelConfig
    #      会触发模型下载/配置读取。它是**必填字段**，由调用方显式传入。
    model_config: ModelConfig = None  # type: ignore[assignment]
    """Model configuration."""
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    """Cache configuration."""
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)
    """Parallel configuration."""
    # [CN] 注意：这里不是 default_factory=SchedulerConfig，而是
    #      SchedulerConfig.default_factory —— 一个**类方法**，它会读取环境变量
    #      VLLM_MAX_NUM_BATCHED_TOKENS / VLLM_MAX_NUM_SEQS 来生成默认值。
    #      目的是让"scheduler 默认值"能在不构造完整配置的情况下被取到。
    scheduler_config: SchedulerConfig = Field(
        default_factory=SchedulerConfig.default_factory,
    )
    """Scheduler configuration."""
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)
    """Device configuration."""
    load_config: LoadConfig = Field(default_factory=LoadConfig)
    """Load configuration."""
    offload_config: OffloadConfig = Field(default_factory=OffloadConfig)
    """Model weight offloading configuration."""
    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    """Attention configuration."""
    mamba_config: MambaConfig = Field(default_factory=MambaConfig)
    """Mamba configuration."""
    kernel_config: KernelConfig = Field(default_factory=KernelConfig)
    """Kernel configuration."""
    lora_config: LoRAConfig | None = None
    """LoRA configuration."""
    speculative_config: SpeculativeConfig | None = None
    """Speculative decoding configuration."""
    diffusion_config: DiffusionConfig | None = None
    """Diffusion LLM (dLLM) configuration."""

    structured_outputs_config: StructuredOutputsConfig = Field(
        default_factory=StructuredOutputsConfig
    )
    """Structured outputs configuration."""
    observability_config: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig
    )
    """Observability configuration."""
    # [CN] quant_config 是**运行时解析结果**而非用户输入：
    #      用户输入是 model_config.quantization（一个名字），真正的
    #      QuantizationConfig 对象要等模型配置就绪后，由
    #      VllmConfig._get_quantization_config() 加载量化方法插件来生成，
    #      并在这里缓存。因此它是 None 直到 __post_init__ 跑完。
    quant_config: QuantizationConfig | None = None
    """Quantization configuration."""
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    """`torch.compile` and cudagraph capture configuration for the model.

    As a shorthand, one can append compilation arguments via
    -cc.parameter=argument such as `-cc.mode=3` (same as `-cc='{"mode":3}'`).

    You can specify the full compilation config like so:
    `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    profiler_config: ProfilerConfig = Field(default_factory=ProfilerConfig)
    """Profiling configuration."""
    kv_transfer_config: KVTransferConfig | None = None
    """The configurations for distributed KV cache transfer."""
    kv_events_config: KVEventsConfig | None = None
    """The configurations for event publishing."""
    ec_transfer_config: ECTransferConfig | None = None
    """The configurations for distributed EC cache transfer."""
    ec_manager_config: EncoderCacheManagerConfig = Field(
        default_factory=EncoderCacheManagerConfig
    )
    """The configurations for custom encoder cache manager."""
    reasoning_config: ReasoningConfig | None = None
    """The configurations for reasoning model."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    # [CN] additional_config 是"逃生舱"：给平台插件、实验特性、树外注册用的杂项口袋。
    #      它参与 compute_hash()，所以放进去的东西会影响编译缓存是否命中——
    #      只放真正影响行为的项，别拿它当运行时传参通道。
    additional_config: dict | SupportsHash = Field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    # [CN] instance_id：多实例（尤其是 DP 多 rank、或同一进程内多个引擎）时用于区分。
    #      空串表示单实例，很多日志/指标代码都以它是否为空来决定要不要加前缀。
    instance_id: str = ""
    """The ID of the vLLM instance."""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    """The optimization level. These levels trade startup time cost for
    performance, with -O0 having the best startup time and -O3 having the best
    performance. -O2 is used by default. See OptimizationLevel for full
    description."""

    performance_mode: PerformanceMode = "balanced"
    """Performance mode for runtime behavior, 'balanced' is the default.
    'interactivity' favors low end-to-end per-request latency at small batch
    sizes (fine-grained CUDA graphs, latency-oriented kernels).
    'throughput' favors aggregate tokens/sec at high concurrency (larger CUDA
    graphs, more aggressive batching, throughput-oriented kernels)."""

    weight_transfer_config: WeightTransferConfig | None = None
    """The configurations for weight transfer during RL training."""

    shutdown_timeout: int = Field(default=0, ge=0)
    """Shutdown grace period for in-flight requests. Shutdown will be delayed for
    up to this amount of time to allow already-running requests to complete. Any
    remaining requests are aborted once the timeout is reached.
    """

    # 【配置指纹】把"会影响计算图结构"的所有配置收集成列表，序列化后取哈希前 10 位。
    # 用途：torch.compile 的缓存 key、CUDA Graph 缓存目录名、prefix caching 的编译复用判断。
    # 关键约束（也是最容易踩的坑）：
    #   - 只覆盖"从 input_ids/embeddings 到最终 hidden states"这一段的计算图。
    #     采样、detokenize、日志等图外行为不进哈希，改它们不应导致重编译。
    #   - 新增字段时，如果它影响计算图，必须加进 factors，否则会命中错误的旧缓存。
    #   - 每个子配置各自实现 compute_hash()，这里只负责聚合（见 .utils.SupportsHash）。
    #   - quant_config 故意不加：它已经被 model_config.quantization 覆盖，避免重复。
    #   - additional_config 是 dict 时用 json.dumps(sort_keys=True) 归一化，保证键顺序不影响结果。
    def compute_hash(self, include_version: bool = True) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        Args:
            include_version: Include the vLLM version in the hash.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        if include_version:
            from vllm import __version__

            vllm_factors.append(__version__)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
            if (
                self.compilation_config
                and getattr(self.compilation_config, "compile_mm_encoder", False)
                and self.model_config.multimodal_config
            ):
                vllm_factors.append(self.model_config.multimodal_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.device_config:
            vllm_factors.append(self.device_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.load_config:
            vllm_factors.append(self.load_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.offload_config:
            vllm_factors.append(self.offload_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.attention_config:
            vllm_factors.append(self.attention_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.structured_outputs_config:
            vllm_factors.append(self.structured_outputs_config.compute_hash())
        if self.profiler_config:
            vllm_factors.append(self.profiler_config.compute_hash())
        else:
            vllm_factors.append("None")
        vllm_factors.append(self.observability_config.compute_hash())
        if self.quant_config:
            pass  # should be captured by model_config.quantization
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.kernel_config:
            vllm_factors.append(self.kernel_config.compute_hash())
        else:
            vllm_factors.append(None)
        if self.kv_transfer_config:
            vllm_factors.append(self.kv_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.ec_transfer_config:
            vllm_factors.append(self.ec_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = safe_hash(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append("None")
        factors.append(vllm_factors)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[
            :10
        ]
        return hash_str

    # ---------------- 派生属性区（只读、无副作用，供调度器与 worker 统一取值） ------
    # [CN] 这一组 property 的共同目的：把"某个数值该怎么算"收敛到唯一一处，
    #      避免调度器、worker warmup、KV cache 预留各自推导导致口径漂移。

    @property
    def is_mm_encoder_only(self) -> bool:
        """只跑多模态编码器的模式（不跑 LLM）。

        此时引擎的输入是图像/音频，输出是 embedding 而非文本。
        """
        mm_config = (
            self.model_config.multimodal_config
            if self.model_config is not None
            else None
        )
        return bool(mm_config and mm_config.mm_encoder_only)

    @property
    def max_concurrent_batches(self) -> int:
        """同时在飞的 batch 数量上限。

        中文：为什么会有多个 batch 同时"在飞"：
          - 流水线并行（PP）需要 pp_size 个 batch 才能填满流水线各级；
          - 异步调度（async scheduling）需要 2 个 batch：一个在 GPU 上跑，
            另一个在 CPU 上做调度准备，从而把 CPU 开销藏起来。
        """
        # PP requires PP-size concurrent batches to fill the pipeline.
        # Async scheduling requires 2 concurrent batches to overlap.
        pp_size = self.parallel_config.pipeline_parallel_size
        if self.scheduler_config.async_scheduling:
            if self.use_v2_model_runner:
                return pp_size + 1
            # V1 Model Runner does not fully support async scheduling with PP.
            if pp_size <= 1:
                return 2
        return pp_size

    @property
    def max_in_flight_tokens(self) -> int:
        """已调度但尚未结算（block 未释放）的 token 数上限。

        中文：= 并发 batch 数 × 每批 token 上限。它决定 KV cache 需要额外预留多少
        余量。滑动窗口 / chunked-local 这类"可回收"的 KV 规格尤其需要它：
        超出窗口的 block 是按"已处理 token"为基准释放的，
        因此同时在飞的几步会**暂时**多占一些 block，不预留就会踩空。
        """
        # Upper bound on tokens that are scheduled but not yet settled (freed):
        # every concurrent batch may hold up to a full `max_num_batched_tokens`.
        # Recycling-aware KV cache specs (sliding-window, chunked-local) reserve
        # for this because out-of-window blocks are freed on the processed-token
        # basis, so in-flight steps transiently keep their blocks.
        return (
            self.max_concurrent_batches * self.scheduler_config.max_num_batched_tokens
        )

    @property
    def num_speculative_tokens(self) -> int:
        """每步由 drafter 提议的 token 数；未启用投机解码时为 0。

        中文：两个来源互斥地提供这个值——
          - 投机解码（speculative_config.num_speculative_tokens）；
          - 扩散 LLM（diffusion_config.canvas_length，即一次生成的画布长度）。
        下游（KV 预留、batch 尺寸计算、uniform_decode_query_len）都读这一个属性。
        """
        if (
            self.speculative_config is not None
            and self.speculative_config.num_speculative_tokens is not None
        ):
            return self.speculative_config.num_speculative_tokens
        if (
            self.diffusion_config is not None
            and self.diffusion_config.canvas_length is not None
        ):
            return self.diffusion_config.canvas_length
        return 0

    @property
    def num_lookahead_tokens(self) -> int:
        """KV slots to reserve past the tokens the target model is scheduled for.

        The drafter writes KV for positions beyond the target model's query
        range, so every component that reserves blocks must add this margin:
        the scheduler through `allocate_slots`, and the worker warmup, which
        builds its own `SchedulerOutput`s. Consumers must read this property
        rather than re-deriving their own per-method lookahead, so the
        scheduler and warmup cannot drift apart.

        中文：需要在目标模型"本步 query 范围之外"额外预留多少个 KV slot。
        因为 drafter 会为超出目标 query 范围的位置写入 KV，所以**所有**预留 block
        的地方（调度器的 allocate_slots、以及 worker warmup 自己构造 SchedulerOutput 时）
        都必须加上这个余量。
        各方法的差异：
          - DFlash：in-fill 式解码，除了各 draft token 的 query，还多一个"最后采样 token"
            的 query，所以要 num_speculative_tokens + 1；
          - EAGLE / DSpark / draft model：draft block 里 anchor 本身就是第一个预测位置，
            不需要额外的 bonus query，恰好是 num_speculative_tokens；
          - 其他：0。
        """
        speculative_config = self.speculative_config
        if speculative_config is None:
            return 0
        if speculative_config.use_dflash():
            # DFlash requires an extra lookahead slot since it uses in-fill-style
            # decoding instead of standard next-token sampling, so it has a query
            # for the last sampled token plus queries for each draft token.
            return self.num_speculative_tokens + 1
        if speculative_config.use_eagle() or speculative_config.uses_draft_model():
            # DSpark (covered by use_eagle) drafts a block of num_speculative_tokens
            # query tokens in which the anchor itself is the first prediction
            # position (no separate bonus query), so it needs exactly
            # num_speculative_tokens lookahead slots.
            return self.num_speculative_tokens
        return 0

    @property
    def uniform_decode_query_len(self) -> int:
        """Query length of every request in a uniform decode batch.

        A decode step submits one query for the newly sampled token plus one
        for each draft token, so the widest uniform decode batch the scheduler
        can build is `max_num_seqs * uniform_decode_query_len` tokens. Anything
        that has to cover a decode batch reads this, so the sizing rule cannot
        drift between the places that apply it.

        This deliberately does not derive from the KV slots a drafter reserves
        past the target's query range, which is a *reservation* contract rather
        than a query-length one. The two do not differ by a constant: DFlash
        reserves `num_speculative_tokens + 1` slots yet still verifies `1 +
        num_speculative_tokens` queries, while EAGLE reserves
        `num_speculative_tokens` and verifies the same `1 + n`. Deriving one
        from the other would under-size EAGLE by a full request width, which is
        the failure this property exists to prevent.

        中文：= 1 + 投机 token 数。"1" 是本步新采样出的那个 token 的 query，
        其余是每个 draft token 各一个 query。因此调度器能构造的**最宽**的
        均匀 decode batch = max_num_seqs × uniform_decode_query_len 个 token。

        ⚠️ 它和上面的 num_lookahead_tokens 是**两个不同的契约**，不要互相推导：
        前者是"query 长度"（请求要算多少个位置），后者是"KV 预留量"（要占多少个 slot）。
        二者相差不是常数：DFlash 预留 n+1 个 slot 但只验证 1+n 个 query，
        EAGLE 预留 n 个也验证 1+n 个 query。若用预留量反推 query 长度，
        会把 EAGLE 少算整整一个请求宽度——这正是本属性存在的意义。
        """
        return 1 + self.num_speculative_tokens

    @property
    def use_v2_model_runner(self) -> bool:
        """是否启用 Model Runner V2（走 `vllm/v1/worker/gpu/` 那套拆分实现）。

        中文：判定优先级依次是——
          1) 环境变量 VLLM_USE_V2_MODEL_RUNNER 显式设置时，直接以它为准；
          2) ROCm 平台 + 特定架构（ROCM_DEFAULT_MRV1_ARCHITECTURES）→ 强制回退 V1；
          3) 没装 Triton → 回退 V1（V2 依赖 Triton）；
          4) 命中 V2 尚不支持的特性（_get_v2_model_runner_unsupported_features）→ 回退 V1；
          5) 其余默认走 V2。
        注意每步回退都会 warning_once，排查"为什么没走 V2"时看启动日志即可。
        """
        use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        if use_v2_model_runner is not None:
            return use_v2_model_runner

        from vllm.platforms import current_platform

        model_config = self.model_config
        if model_config is not None and current_platform.is_rocm():
            architectures = getattr(model_config, "architectures", ())
            if any(arch in ROCM_DEFAULT_MRV1_ARCHITECTURES for arch in architectures):
                logger.warning_once(
                    "Defaulting to V1 model runner on ROCm for model architectures: %s",
                    ", ".join(architectures),
                )
                return False

        if not HAS_TRITON:
            logger.warning_once(
                "Model Runner V2 requires Triton; using the V1 model runner instead."
            )
            return False

        unsupported = self._get_v2_model_runner_unsupported_features()
        if unsupported:
            logger.warning_once(
                "Model Runner V2 does not yet support %s; using the V1 model "
                "runner instead.",
                ", ".join(unsupported),
            )
            return False

        return True

    def _is_dflash2_draft(self) -> bool:
        """Whether the DFlash draft is a DFlash2 one, by the architecture the
        speculator selects on (v1/worker/gpu/spec_decode/__init__.py)."""
        spec = self.speculative_config
        if spec is None or spec.method != "dflash":
            return False
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is None:
            return False
        return "DFlash2DraftModel" in (draft_config.architectures or [])

    def _dflash_needs_multi_kv_group(self) -> bool:
        """Whether a DFlash draft mixes sliding-window and full attention."""
        spec = self.speculative_config
        if spec is None or spec.method != "dflash":
            return False
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is None:
            return False
        layer_types = getattr(draft_config.hf_config, "layer_types", None) or []
        num_sliding = sum(lt == "sliding_attention" for lt in layer_types)
        return 0 < num_sliding < len(layer_types)

    def _uses_breakable_cudagraph_by_default(self) -> bool:
        model_config = self.model_config
        if model_config is None:
            return False

        architectures = set(model_config.architectures)
        return bool(architectures & default_breakable_cudagraph_architectures())

    def _maybe_enable_breakable_cudagraph(self) -> bool:
        """按需开启 breakable cudagraph，并据此关闭 torch.compile。

        中文：breakable cudagraph 与 torch.compile 是**互斥**的两条加速路线
        （前者把 cudagraph 拆成可打断的小段以兼容动态形状），
        所以一旦启用就必须把 compilation_config.mode 置为 NONE。
        副作用：会**直接改写 os.environ**，但只在用户没显式设置
        VLLM_USE_BREAKABLE_CUDAGRAPH 时才写（尊重显式配置）。
        """
        if (
            "VLLM_USE_BREAKABLE_CUDAGRAPH" not in os.environ
            and self._uses_breakable_cudagraph_by_default()
        ):
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
            logger.info_once(
                "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1. "
                "Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out."
            )

        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )

        enabled = is_breakable_cudagraph_enabled()
        if enabled:
            self.compilation_config.mode = CompilationMode.NONE
        return enabled

    @property
    def needs_dp_coordinator(self) -> bool:
        """
        Determine if the DPCoordinator process is needed.

        The DPCoordinator is needed in two cases:
        1. For MoE models with DP > 1: to handle wave coordination
           (even in external LB mode, since wave coordination runs in the coordinator)
        2. For non-MoE models in internal/hybrid LB mode: to collect and publish
           queue stats for load balancing across DP ranks

        Returns:
            True if DPCoordinator process is needed, False otherwise.

        中文：DPCoordinator 是 DP 部署下的**独立协调进程**，两种场景需要它：
          1) MoE 模型 + DP>1：做 wave 协同（即使 external LB 也需要，
             因为 wave 协同逻辑本身就跑在 coordinator 里）；
          2) 非 MoE 模型 + internal/hybrid LB：收集并发布各 rank 的队列统计，
             供跨 rank 负载均衡决策。
        表达式的逻辑：DP>1 且（model_config 为空 或 是 MoE 或 非 external LB）。
        """

        # For non-MoE models, only need coordinator in internal/hybrid LB mode
        # (for stats collection).
        return self.parallel_config.data_parallel_size > 1 and (
            self.model_config is None
            or self.model_config.is_moe
            or not self.parallel_config.data_parallel_external_lb
        )

    def enable_trace_function_call_for_thread(self) -> None:
        """
        Set up function tracing for the current thread,
        if enabled via the `VLLM_TRACE_FUNCTION` environment variable.
        """
        if envs.VLLM_TRACE_FUNCTION:
            tmp_dir = tempfile.gettempdir()
            # add username to tmp_dir to avoid permission issues
            tmp_dir = os.path.join(tmp_dir, getpass.getuser())
            filename = (
                f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
                f"_thread_{threading.get_ident()}_at_{datetime.now()}.log"
            ).replace(" ", "_")
            log_path = os.path.join(
                tmp_dir,
                "vllm",
                f"vllm-instance-{self.instance_id}",
                filename,
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            enable_trace_function_call(log_path)

    # [CN] 量化配置的构造分三步：读 HF 的 quantization_config → 校验硬件能力 → 校验数据类型。
    #      三步都会抛 ValueError，且都在启动早期，属于"快速失败"。
    #
    #      注意这里有个隐藏约束：量化方法是在**读权重之前**根据 config 决定的，
    #      所以 model_config.quantization 可以是 None（自动推断）、也可以是显式方法名。
    #      自动推断发生在 ModelConfig 里，这里只处理"已经有结论"的情况。
    @staticmethod
    def _get_quantization_config(
        model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        """Get the quantization config."""
        # [CN] 延迟导入：platforms 会拉起 torch，避免在纯配置场景付出导入代价
        from vllm.platforms import current_platform

        if model_config.quantization is not None:
            # [CN] get_quant_config 会去模型仓库读 config.json 里的 quantization_config 字段；
            #      因此这一步可能有网络/磁盘 IO，是启动耗时的组成部分之一
            from vllm.model_executor.model_loader.weight_utils import get_quant_config

            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()

            # [CN] 硬件能力校验：例如 FP8 需要 SM89+(Ada)/SM90+(Hopper)，
            #      get_min_capability() 由各量化方法自己声明。CPU 等平台返回 None，跳过检查。
            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
            # [CN] 激活值 dtype 校验：有些量化只支持特定激活精度（如 FP8 要求 bf16/fp16）；
            #      与 capability 检查是两回事——一个是硬件，一个是算法语义
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}"
                )
            # [CN] 最后一环：让量化方法按实际模型做二次调整
            #      （如 FP8 需要根据 config 里的 activation_scheme 决定是否启用动态缩放，
            #       或 compressed-tensors 需要按 layer 名匹配忽略列表）
            quant_config.maybe_update_config(
                model_config.model,
                hf_config=model_config.hf_config,
                revision=model_config.revision,
            )
            return quant_config
        return None

    # [CN] 对外的安全版本。
    #      为什么需要深拷贝：下面的 _ 版本内部会调用 maybe_update_config，
    #      而它有可能**就地修改**传入的 model_config（这是一处已知的实现缺陷，
    #      见上面原注释 "For some reason..."）。用 deepcopy 隔离，
    #      保证调用方的 model_config 不被污染——代价是多一次拷贝开销，
    #      但这个方法只在启动时调用少数几次，可以接受。
    @staticmethod
    def get_quantization_config(
        model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        import copy

        # For some reason, the _ version of this modifies the model_config
        # object, so using deepcopy to avoid this problem.
        return VllmConfig._get_quantization_config(
            copy.deepcopy(model_config), load_config
        )

    # [CN] 「换一个 HF config，派生出一份新的 VllmConfig」——不可变式更新。
    #
    #      典型用途：
    #        - 模型注册/能力探测阶段，需要按不同 architectures 试算；
    #        - 多模态模型里拿 text_config 单独推一份语言侧配置；
    #        - speculative decoding 里为 draft 模型构造独立配置。
    #
    #      为什么用 dataclasses.replace 而不是改 self：
    #      保证原配置对象不被改动，可安全地并发/重复派生。
    def with_hf_config(
        self,
        hf_config: PretrainedConfig,
        architectures: list[str] | None = None,
    ) -> "VllmConfig":
        # [CN] 补齐 architectures：HF 的 config.json 未必写这个字段，
        #      但 vLLM 的模型注册是按 architectures 名字匹配的，缺了会找不到实现。
        #      优先用调用方显式给的；否则查 transformers 的 MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        #      按 model_type 反查一个默认架构名。
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures
        elif hf_config.architectures is None:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            )

            if hf_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                hf_config = copy.deepcopy(hf_config)
                hf_config.architectures = [
                    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[hf_config.model_type]
                ]

        model_config = copy.deepcopy(self.model_config)

        # In Transformers v5, tie_word_embeddings belongs to the config of the class
        # that can see both layers to be tied. For example:
        #
        # SomeVLModel:
        #   self.language_model = SomeLanguageModel(SomeVLTextConfig)
        #   self.vision_model = SomeVisionModel(SomeVLVisionConfig)
        #
        # SomeVLModelForMultimodalLM:
        #   self.model = SomeVLModel(SomeVLConfig)
        #   self.lm_head = nn.Linear()
        #
        # Therefore, tie_word_embeddings is defined in SomeVLConfig and is not present
        # in SomeVLTextConfig*. In vLLM, the lm_head belongs to the language_model, so
        # we must ensure that tie_word_embeddings is set in the language_model's config.
        #
        # *For some models, SomeVLTextConfig may also have a tie_word_embeddings field.
        # This is only the case if SomeVLTextConfig is also used for a text only version
        # of the same model. For example:
        #
        # SomeVLModelForCausalLM:
        #   self.model = SomeLanguageModel(SomeVLTextConfig)
        #   self.lm_head = nn.Linear()
        #
        # Therefore, the presence of tie_word_embeddings in SomeVLTextConfig cannot
        # be used as a signal for whether tie_word_embeddings should be copied from
        # hf_config to the language_model config.
        #
        #
        # [CN] 上面那段英文的中文概括（多模态权重绑定的补偿逻辑）：
        #      一句话概括：Transformers v5 把 tie_word_embeddings 放在**能看到被绑定两个层的那一级**
        #      配置上（多模态模型里是最外层的 VLConfig），而 vLLM 的 lm_head 挂在 language_model 下，
        #      所以必须手动把外层的开关同步到 text_config 上，否则权重绑定会静默失效 ——
        #      表现为输出乱码但不报错，极难排查。
        #
        #      注意：不能用「text_config 是否已有该字段」来判断是否需要拷贝，
        #      因为有些模型（存在纯文本版本时）本身就带这个字段，语义不同。
        if model_config.is_multimodal_model and hasattr(
            model_config.hf_config, "tie_word_embeddings"
        ):
            tie_word_embeddings = model_config.hf_config.tie_word_embeddings
            hf_config.get_text_config().tie_word_embeddings = tie_word_embeddings

        model_config.hf_config = hf_config
        # [CN] hf_config 换了，架构相关的派生信息必须重算（层数、hidden size、注意力类型等缓存）
        model_config.model_arch_config = model_config.get_model_arch_config()

        return replace(self, model_config=model_config)

    # [CN] 「仅在用户没显式设置时才填默认值」的辅助函数，是本文件默认体系的基石。
    #
    #      ⚠️ 判据是 `is None` 而不是「是否等于默认值」：
    #      所以用户显式写 `False` / `0` / `[]` 都会被尊重，不会被默认值覆盖。
    #      这就是为什么很多字段在 arg_utils 里被特意改成 `default=None` 当哨兵 ——
    #      只有 None 才能区分「没设」和「设成假值」。
    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:
        """Set config attribute to default if not already set by user.

        Args:
            config_obj: Configuration object to update.
            key: Attribute name.
            value: Default value (static or callable).
        """
        if getattr(config_obj, key) is None:
            # Some config values are known before initialization and are
            # hard coded.
            # Other values depend on the user given configuration, so they are
            # implemented with lambda functions and decided at run time.
            # [CN] value 可能是 lambda cfg: ... —— 需要依赖其他配置才能决定的值用可调用对象延迟求值，
            #      这样默认值可以引用最终配置，而不是构造时的中间状态
            setattr(config_obj, key, value(self) if callable(value) else value)

    # [CN] 按 optimization_level（O0/O1/O2/O3）批量套用默认值，见文件头的 OPTIMIZATION_LEVEL_* 表。
    #
    #      三个要点：
    #      1) 只覆盖 defaults 里列出的字段，其余不动；
    #      2) 递归下钻到嵌套 dataclass（如 compilation_config.cudagraph_mode）；
    #      3) 由于走 _set_config_default，用户显式设置的值一律优先于 level 默认值。
    #         即 level 只是"给没设的字段兜底"，不是强制覆盖。
    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        """Apply optimization level defaults using self as root.

        Recursively applies values from defaults into nested config objects.
        Only fields present in defaults are overwritten.

        If the user configuration does not specify a value for a default field
        and if the default field is still None after all user selections are
        applied, then default values will be applied to the field. User specified
        fields will not be overridden by the default.

        Args:
            defaults: Dictionary of default values to apply.
        """

        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            """Recursively apply defaults to config_obj, using self as root."""
            for key, value in config_defaults.items():
                # [CN] 静默跳过不存在的字段：defaults 表是手写的，字段改名后不会报错，
                #      只会悄悄失效——改配置字段名时记得同步 OPTIMIZATION_LEVEL_* 表
                if not hasattr(config_obj, key):
                    continue

                current = getattr(config_obj, key)
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                else:
                    self._set_config_default(config_obj, key, value)

        apply_recursive(self, defaults)

    # [CN] 「动态投机解码」与「full CUDA graph」互斥，这里是自动降级。
    #
    #      原因：full cudagraph 要求每步的 shape 完全固定，而动态投机解码会在运行时
    #      改变验证长度（num_speculative_tokens 随 batch 变化），shape 不再固定。
    #      与其在运行时崩，不如在配置期降级为 PIECEWISE（分段捕获，只包住 attention 之外的部分）。
    #
    #      注意降级是**静默改配置 + 打一条 warning**，不会报错。排查性能问题时
    #      如果看到 cudagraph_mode 和自己设的不一样，多半是这里动的。
    def _maybe_override_dynamic_sd_cudagraph_mode(self) -> None:
        speculative_config = self.speculative_config
        if (
            speculative_config is None
            or not speculative_config.uses_dynamic_speculative_decoding()
            or not self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            or self.use_v2_model_runner
        ):
            return

        logger.warning_once(
            "Dynamic speculative decoding changes the target verification "
            "length at runtime. Overriding cudagraph_mode from %s to "
            "PIECEWISE for reliability. Use VLLM_USE_V2_MODEL_RUNNER=1 "
            "if you want to use full CUDA graphs.",
            self.compilation_config.cudagraph_mode.name,
        )
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    # [CN] 动态投机解码与数据并行互斥，同样是自动降级（清掉 per-batch 表，退回静态 token 数）。
    #
    #      为什么必须禁：DP 下各个 rank 会**独立**决定本次用几个投机 token，
    #      一旦不同步，各 rank 每步推进的 token 数就不一致 —— 轻则结果发散，
    #      重则在需要集合通信的地方互相等待，直接死锁。这属于"必须保证同步"的硬约束。
    def _maybe_disable_dynamic_sd_for_data_parallel(self) -> None:
        speculative_config = self.speculative_config
        if (
            speculative_config is None
            or not speculative_config.uses_dynamic_speculative_decoding()
            or self.parallel_config.data_parallel_size <= 1
        ):
            return

        logger.warning_once(
            "Dynamic speculative decoding is not supported with data "
            "parallelism because data-parallel ranks can select different "
            "speculative-token counts, causing DP divergence and deadlocks. "
            "Disabling num_speculative_tokens_per_batch_size and falling back "
            "to static num_speculative_tokens=%d.",
            speculative_config.num_speculative_tokens,
        )
        speculative_config.num_speculative_tokens_per_batch_size = None

    def _post_init_kv_transfer_config(self) -> None:
        """Update KVTransferConfig based on top-level configs in VllmConfig.

        Right now, this function reads the offloading settings from
        CacheConfig and configures the KVTransferConfig accordingly.
        """
        # KV offloading is only activated when kv_offloading_size is set.
        if (kv_offloading_size := self.cache_config.kv_offloading_size) is None:
            return

        kv_offloading_backend = self.cache_config.kv_offloading_backend

        # [CN] 反向依赖：用户在 CacheConfig 上只填了 kv_offloading_size，
        #      但真正干活的是 KVTransferConfig。这里把它补全 ——
        #      也就是说「开启 KV offload」这个开关在 CacheConfig，而实现载体在 KVTransferConfig。
        # If no KVTransferConfig is provided, create a default one.
        if self.kv_transfer_config is None:
            self.kv_transfer_config = KVTransferConfig()

        # [CN] native 走 vLLM 自带的 CPU offload connector，cpu_bytes_to_use 单位是字节，
        #      所以要把用户给的 GiB 乘 1<<30。两种实现：Simple（简化版）/ Offloading（完整版）。
        if kv_offloading_backend == "native":
            if envs.VLLM_USE_SIMPLE_KV_OFFLOAD:
                config_connector = "SimpleCPUOffloadConnector"
            else:
                config_connector = "OffloadingConnector"
            self.kv_transfer_config.kv_connector = config_connector
            self.kv_transfer_config.kv_connector_extra_config.update(
                {"cpu_bytes_to_use": kv_offloading_size * (1 << 30)}
            )
        elif kv_offloading_backend == "lmcache":
            # Default to LMCache multi-process (MP) mode. The actual KV
            # storage capacity is managed by the standalone LMCache server
            # process, so ``kv_offloading_size`` is not propagated here.
            # ``LMCacheMPConnector`` falls back to ``tcp://localhost:5555``
            # when host/port are not provided via extra_config.
            self.kv_transfer_config.kv_connector = "LMCacheMPConnector"

        # This is the same for all backends
        # [CN] kv_both = 本进程既能当 sender 又能当 receiver。
        #      offload 场景（把 KV 卸到本地 CPU）本来就是单机自收自发，所以固定为 both。
        self.kv_transfer_config.kv_role = "kv_both"

    def _verify_kv_transfer_compat(self) -> None:
        """Reject configurations that silently corrupt KV transfers."""
        if (
            self.kv_transfer_config is None
            or self.kv_transfer_config.kv_connector is None
        ):
            return

        # PyTorch's expandable_segments allocator uses CUDA VMM, which can
        # remap a virtual address range to different physical pages over the
        # engine's lifetime. KV connectors that pin KV cache memory (e.g.
        # NixlConnector via ibv_reg_mr, MooncakeConnector) end up with their
        # registrations pointing at stale physical pages after any remap,
        # producing RDMA failures like IBV_WC_REM_ACCESS_ERR /
        # NIXL_ERR_REMOTE_DISCONNECT at the first inter-node KV transfer.
        # We can't enumerate every in-tree and out-of-tree connector that
        # pins memory, so we conservatively reject the combination whenever
        # any KV connector is configured.
        #
        # CuMem allocator is exempt: CuMemAllocator.use_memory_pool toggles
        # expandable_segments off around its pool (see #40812), so the KV
        # cache allocated within that context lands on stable physical pages
        # even when the env var is set.
        if "expandable_segments:True" not in os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        ):
            return
        # [CN] 唯一的例外：开了 cumem allocator。
        #      它会在自己的内存池作用域内临时关掉 expandable_segments，
        #      于是 KV cache 落在稳定的物理页上，不会重映射 —— 所以放行。
        #      这也是为什么开 sleep mode 能顺带解决这个冲突（sleep mode 会启用 cumem）。
        if self.model_config is not None and (self.model_config.enable_cumem_allocator):
            return

        raise ValueError(
            f"KV connector {self.kv_transfer_config.kv_connector} is "
            "incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "unless enable_cumem_allocator is also enabled. PyTorch's CUDA VMM "
            "allocator can remap KV cache virtual addresses to different "
            "physical pages, invalidating any pinned/registered KV memory "
            "(e.g. IB memory regions registered by NIXL or Mooncake). Either "
            "unset expandable_segments:True or enable the cumem allocator "
            "(sleep mode does this automatically and also "
            "routes KV allocations through CuMemAllocator's pool, where "
            "expandable_segments is automatically disabled)."
        )

    def _verify_sampling_replay_config(self) -> None:
        model_config = self.model_config
        if model_config is None or not model_config.return_sampling_mask:
            return
        if not self.use_v2_model_runner:
            raise ValueError("sampling distribution replay requires Model Runner V2")
        if self.speculative_config is not None:
            raise ValueError(
                "sampling distribution replay does not support speculative decoding"
            )
        if model_config.is_diffusion:
            raise ValueError(
                "sampling distribution replay does not support diffusion models"
            )
        if model_config.logits_processors:
            raise ValueError(
                "sampling distribution replay does not support custom logits processors"
            )
        if model_config.logprobs_mode != "processed_logprobs":
            raise ValueError(
                "sampling distribution replay requires "
                "logprobs_mode='processed_logprobs' so that returned logprobs "
                "are normalized over the same nucleus as the sampling mask"
            )

    # [CN] trace replay = 回放真实线上 trace（用于确定性复现/性能压测）。
    #      只支持 V2 runner，也是"能力未就绪就直接报错"而非静默降级的例子。
    #      与上面的 sampling replay 区别在于：它校验的是功能开关组合，而非采样语义一致性。
    def _verify_trace_replay_config(self) -> None:
        model_config = self.model_config
        if model_config is None or not model_config.enable_trace_replay:
            return
        if not self.use_v2_model_runner:
            raise ValueError("trace replay requires Model Runner V2")

    # [CN] ======================= 全文件最重要的一个方法 =======================
    #      __post_init__ 是「配置从『用户意图』变成『可执行事实』」的那一步。
    #      它做三件事，且**会就地修改**子配置对象：
    #
    #      (1) 推导默认值    —— 把 arg_utils 留下的 None 哨兵填上真实值
    #                          （max_num_batched_tokens、cudagraph sizes、compile ranges ...）
    #      (2) 跨配置联合校验 —— 单个子配置各自合法、组合起来非法的情形，只能在这里拦
    #                          （投机解码 vs DP、KV connector vs expandable_segments ...）
    #      (3) 自动降级      —— 不兼容的开关组合直接改写为安全值 + warning，而不是报错
    #
    #      ⚠️ 阅读提示：
    #      - 顺序即依赖。后面的校验常常假设前面的推导已完成，不要随意调整语句顺序。
    #      - 大量 `if X is not None` 是防御性的：VllmConfig 允许被部分构造
    #        （如只为算 hash 而造的临时实例），此时子配置可能是 None。
    #      - 这个方法只在启动时跑一次，里面出现的 O(1) 之外的循环/IO 都是启动耗时来源。
    def __post_init__(self):
        """Verify configs are valid & consistent with each other."""

        # To give each torch profile run a unique instance name.
        # [CN] 用纳秒时间戳当实例 ID：目的是让同一次进程内的多次 torch profile
        #      输出到不同目录，互不覆盖。不是稳定的业务 ID，别拿它做持久化标识。
        self.instance_id = f"{time.time_ns()}"

        self._resolve_mm_encoder_only()

        if self.performance_mode != "balanced":
            logger.info_once("Performance mode set to '%s'.", self.performance_mode)

        self.try_verify_and_update_config()

        # Models may have supplied their own DCP defaults above; anything still
        # unset falls back to the stock ones.
        self.parallel_config.set_dcp_defaults()

        if self.model_config is not None:
            self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(self.load_config)

            self.parallel_config.is_moe_model = self.model_config.is_moe

        if (
            self.model_config is not None
            and self.model_config.enable_return_routed_experts
        ):
            if self.parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "--enable-return-routed-experts is incompatible with "
                    "pipeline parallelism (PP > 1)."
                )
            if (
                self.parallel_config.decode_context_parallel_size > 1
                or self.parallel_config.prefill_context_parallel_size > 1
            ):
                raise ValueError(
                    "--enable-return-routed-experts is incompatible with context "
                    "parallelism (DCP > 1 or PCP > 1)."
                )

            # Incompatible with any KV connector — covers both PD disaggregation
            # (kv_producer/kv_consumer: routing captured on P can't reach D) and
            # single-instance KV offload/sharing (kv_both: slot_mapping semantics
            # change when KV blocks live outside local GPU memory, breaking the
            # slot-indexed routed_experts buffer).
            if (
                self.kv_transfer_config is not None
                and self.kv_transfer_config.is_kv_transfer_instance
            ):
                raise ValueError(
                    "--enable-return-routed-experts is incompatible with KV "
                    "connectors (PD disaggregation, KV cache offload)."
                )

        if (
            self.model_config is not None
            and self.model_config.multimodal_config is not None
            and self.model_config.multimodal_config.language_model_only
            and self.compilation_config.cudagraph_mm_encoder
        ):
            raise ValueError(
                "--language-model-only is incompatible with "
                "cudagraph_mm_encoder=True, since it disables all multimodal "
                "inputs and the multimodal encoder is never run. Please "
                "disable one of them."
            )

        self._verify_sampling_replay_config()
        self._verify_trace_replay_config()

        # [CN] NIXL（PD 分离的 RDMA 传输后端）对并行方式有三条硬约束，全部用 assert 而非 ValueError：
        #      因为这些属于"内部不可能出现"的不变量（上游 executor 会保证），
        #      真触发说明是代码 bug 而非用户配错，所以不给友好提示。
        #      约束含义：NIXL 侧要么完整复制、要么按 TP 粒度分片，不能是任意 DCP 值；
        #      且 >1 的分片只对 MLA 模型成立（Mamba/混合架构不支持）。
        # A NIXL side is either fully replicated or fully DCP-sharded; MLA only.
        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.has_connector("NixlConnector")
        ):
            assert self.parallel_config.prefill_context_parallel_size == 1, (
                "NIXL does not support prefill context parallelism."
            )
            dcp_size = self.parallel_config.decode_context_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            assert dcp_size in (1, tp_size), (
                f"decode_context_parallel_size={dcp_size} must be 1 or equal "
                f"to tensor_parallel_size={tp_size} when using NixlConnector."
            )
            if self.model_config is not None:
                assert self.model_config.use_mla or dcp_size == 1, (
                    "PD with decode_context_parallel_size > 1 is only "
                    "supported for MLA models."
                )
                assert not (self.model_config.is_hybrid and dcp_size > 1), (
                    "PD with decode_context_parallel_size > 1 is not "
                    "supported for hybrid Mamba/SSM models."
                )

        if self.lora_config is not None:
            self.lora_config.verify_with_model_config(self.model_config)

        if (
            self.mamba_config.enable_stochastic_rounding
            and self.cache_config.mamba_ssm_cache_dtype != "float16"
        ):
            raise ValueError(
                "Stochastic rounding for Mamba cache requires "
                "the SSM cache to be float16. Please set it explicitly, "
                "by specifying `--mamba-ssm-cache-dtype float16`, or disable "
                "stochastic rounding by not specifying "
                "`--enable-mamba-cache-stochastic-rounding`."
            )

        # [CN] 这里用的是**不带深拷贝**的 _ 版本，所以有可能就地修改 model_config。
        #      在 __post_init__ 里这是可接受甚至必要的（需要把量化参数写回 model_config）；
        #      而在 get_quantization_config（对外版）里就必须拷贝，见那里的说明。
        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config
            )

        # "dummy" reads no weights at all, and the sharded formats read a vLLM
        # state dict, which stores tied word embeddings under the lm_head only.
        # Neither can tell us what the original checkpoint contained.
        if self.model_config is not None and self.load_config.load_format not in (
            "dummy",
            "sharded_state",
            "runai_streamer_sharded",
        ):
            self.model_config.maybe_untie_word_embeddings()

        if (
            self.quant_config is not None
            and self.model_config is not None
            and hasattr(self.quant_config, "use_deep_gemm")
            and self.quant_config.use_deep_gemm is None
        ):
            from vllm.utils.deep_gemm import should_auto_disable_deep_gemm

            model_type = getattr(self.model_config.hf_text_config, "model_type", None)
            if should_auto_disable_deep_gemm(model_type):
                self.quant_config.use_deep_gemm = False
                logger.warning_once(
                    "Auto-disabled DeepGemm for model_type=%s on Blackwell. "
                    "DeepGemm E8M0 scale format causes accuracy degradation "
                    "for this architecture. Falling back to CUTLASS. "
                    "To disable DeepGemm globally, set VLLM_USE_DEEP_GEMM=0.",
                    model_type,
                )

        from vllm.platforms import current_platform
        from vllm.v1.executor.abstract import Executor

        executor_backend = self.parallel_config.distributed_executor_backend
        # [CN] 注意 Executor.get_class(self) 需要完整的 VllmConfig —— 说明 executor 的选择
        #      依赖模型/并行/调度等多维信息，不只是 distributed_executor_backend 一个字符串。
        #      这也是为什么 executor 相关校验只能放在 __post_init__ 里。
        executor_class = Executor.get_class(self)
        executor_supports_async_sched = executor_class.supports_async_scheduling()
        uses_rocm_deepep_ht_dbo = (
            current_platform.is_rocm()
            and self.parallel_config.enable_dbo
            and self.parallel_config.all2all_backend == "deepep_high_throughput"
        )

        # [CN] async_scheduling 是**三态**布尔（None / True / False），语义完全不同：
        #        None  = 用户没表态 → 自动决定（下面 elif 分支）：能用就开，有冲突就静默关 + warning
        #        True  = 用户显式要求 → 硬失败：有任何不兼容直接 ValueError，不降级
        #        False = 用户显式关闭 → 下面两个分支都不进
        #      这种"显式则报错、未指定则降级"的模式在 vLLM 配置里反复出现，
        #      是它区别于普通参数解析框架的一个设计取向：把决定权交给用户，但默认给可用解。
        #
        #      async scheduling 本身：让调度与模型执行重叠（CPU 调度下一步时 GPU 还在跑当前步），
        #      属于吞吐优化；代价是对投机解码、executor、DBO 都有约束。
        if self.scheduler_config.async_scheduling:
            # Async scheduling explicitly enabled, hard fail any incompatibilities.
            # Currently, async scheduling only support eagle speculative
            # decoding.
            if uses_rocm_deepep_ht_dbo:
                raise ValueError(
                    "Async scheduling is not compatible with ROCm DeepEP "
                    "high-throughput DBO. Please use --no-async-scheduling or "
                    "select a different all2all backend."
                )
            if self.speculative_config is not None:
                if (
                    self.speculative_config.method not in get_args(EagleModelTypes)
                    and self.speculative_config.method not in get_args(NgramGPUTypes)
                    and self.speculative_config.method != "draft_model"
                    and self.speculative_config.method != "dspark"
                ):
                    raise ValueError(
                        "Currently, async scheduling is only supported "
                        "with EAGLE/MTP/Draft Model/NGram GPU/DSpark kind of "
                        "speculative decoding"
                    )
                if self.speculative_config.disable_padded_drafter_batch:
                    raise ValueError(
                        "Async scheduling is not compatible with "
                        "disable_padded_drafter_batch=True."
                    )
            if not executor_supports_async_sched:
                raise ValueError(
                    f"`{executor_backend}` does not support async scheduling yet."
                )
        elif self.scheduler_config.async_scheduling is None:
            # Enable async scheduling unless there is an incompatible option.
            if (
                self.model_config is not None
                and self.model_config.runner_type == "pooling"
            ):
                # The current implementation of asynchronous scheduling negatively
                # impacts performance of pooling models, so we disable by default.
                logger.debug(
                    "Disabling asynchronous scheduling by default for pooling model."
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method not in get_args(EagleModelTypes)
                and self.speculative_config.method not in get_args(NgramGPUTypes)
                and self.speculative_config.method != "draft_model"
                and self.speculative_config.method != "dspark"
            ):
                logger.warning_once(
                    "Async scheduling not supported with %s-based "
                    "speculative decoding and will be disabled.",
                    self.speculative_config.method,
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.disable_padded_drafter_batch
            ):
                logger.warning_once(
                    "Async scheduling is not compatible with "
                    "disable_padded_drafter_batch=True and will be disabled.",
                )
                self.scheduler_config.async_scheduling = False
            elif not executor_supports_async_sched:
                logger.warning_once(
                    "Async scheduling will be disabled because it is not supported "
                    "with the `%s` distributed executor backend. ",
                    executor_backend,
                )
                self.scheduler_config.async_scheduling = False
            elif uses_rocm_deepep_ht_dbo:
                logger.warning_once(
                    "Async scheduling is disabled for ROCm DeepEP "
                    "high-throughput DBO because that combination can corrupt "
                    "DP+EP generation accuracy."
                )
                self.scheduler_config.async_scheduling = False
            else:
                self.scheduler_config.async_scheduling = True

        # [CN] DP 同步是否走 NCCL：开启 async scheduling 时改成不用 NCCL。
        #      原因：NCCL 集合通信会同步阻塞，与 async scheduling 想达成的"CPU/GPU 重叠"相冲突；
        #      改用基于 ZMQ 的轻量同步。同样只在未显式指定时才自动决定。
        if self.parallel_config.disable_nccl_for_dp_synchronization is None:
            if self.scheduler_config.async_scheduling:
                if self.parallel_config.data_parallel_size > 1 and (
                    self.model_config is None or self.model_config.is_moe
                ):
                    logger.info_once(
                        "Disabling NCCL for DP synchronization "
                        "when using async scheduling.",
                    )
                self.parallel_config.disable_nccl_for_dp_synchronization = True
            else:
                self.parallel_config.disable_nccl_for_dp_synchronization = False

        if (
            self.speculative_config is not None
            and self.scheduler_config.async_scheduling
            and self.model_config is not None
            and not self.model_config.disable_cascade_attn
        ):
            logger.warning_once(
                "Disabling cascade attention (not yet compatible with "
                "async speculative decoding).",
            )
            self.model_config.disable_cascade_attn = True

        if (
            self.observability_config.per_request_spec_decode_metrics != "none"
            and self.speculative_config is None
        ):
            raise ValueError(
                "--per-request-spec-decode-metrics requires speculative decoding "
                "to be enabled (via --speculative-config)."
            )

        if (
            self.model_config is not None
            and self.model_config.multimodal_config is not None
            and self.model_config.multimodal_config.mm_tensor_ipc == "torch_shm"
            and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"
        ):
            raise ValueError(
                "torch_shm is known to fail without "
                "VLLM_WORKER_MULTIPROC_METHOD set to spawn"
            )

        if (
            self.model_config is not None
            and self.scheduler_config.enable_chunked_prefill
            and self.model_config.dtype == torch.float32
            and current_platform.get_device_capability() == (7, 5)
        ):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels."
            )

        # [CN] enforce_eager 是"一键关掉所有编译优化"的总闸。
        #      注意它连带关掉两项：torch.compile(mode) 与 CUDA graph(cudagraph_mode)。
        #      调试时很好用（报错栈可读、启动快），但性能差距巨大，别在生产误开。
        if self.model_config is not None and self.model_config.enforce_eager:
            logger.warning_once(
                "Enforce eager set, disabling torch.compile and CUDAGraphs. "
                "This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none"
            )
            self.compilation_config.mode = CompilationMode.NONE
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # [CN] Proton profiler 要求关掉 CUDA graph —— 因为 graph 会把 kernel 序列
        #      "固化"成一个整体 launch，导致 profiler 拿不到逐 kernel 的时间线。
        #      这条是硬失败（用户显式开了 profiler 就必须能采到数据，降级没意义）。
        if self.profiler_config.profiler == "proton":
            if not current_platform.is_cuda():
                raise ValueError(
                    "The Proton profiler currently supports NVIDIA CUDA only"
                )
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                raise ValueError(
                    "The Proton profiler requires CUDA graphs to be disabled. "
                    "Use --enforce-eager or set "
                    "--compilation-config.cudagraph_mode=none."
                )

        if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
            logger.warning_once(
                "TORCH_COMPILE_DISABLE is set, disabling torch.compile. "
                "This is equivalent to setting -cc.mode=none"
            )
            self.compilation_config.mode = CompilationMode.NONE

        breakable_cudagraph_enabled = self._maybe_enable_breakable_cudagraph()

        if not breakable_cudagraph_enabled and (
            self.compilation_config.backend == "eager"
            or (
                self.compilation_config.mode is not None
                and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
            )
        ):
            logger.warning_once(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        # [CN] 局部辅助函数：判断当前量化方法是否使用「分块权重」（block-wise 量化）。
        #      两种量化实现暴露的查询接口不同（一个属性、一个方法），这里做兼容适配。
        def has_blocked_weights():
            if self.quant_config is not None:
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094
        # [CN] 分块 FP8 权重量化 → 强制开启 quant_fp8 自定义 CUDA op。
        #      custom_ops 用 "+/- 前缀"表达增删，且**后面的覆盖前面的**（类似有序列表），
        #      所以这里先检查是否已被显式 "-quant_fp8" 关掉，再决定要不要追加 "+quant_fp8"。
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        # [CN] 平台相关的兜底默认值（ROCm/TPU/CPU 各自覆盖一批字段）。
        #      放在所有通用逻辑之后，保证平台默认值只填"仍然为 None"的坑。
        current_platform.apply_config_platform_defaults(self)

        # [CN] 编译模式默认值：O0 不编译，O1+ 走 vLLM 自己的编译流水线
        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        # By default, enable torch wrapping only when using custom Inductor lowering
        if self.compilation_config.ir_enable_torch_wrap is None:
            self.compilation_config.ir_enable_torch_wrap = (
                self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                and self.compilation_config.backend == "inductor"
            )

        # [CN] custom_ops 的默认值分两种取向，取决于是否走 Inductor：
        #        - inductor 后端：默认 "none"（关闭自定义 op，让 Inductor 自己做融合）
        #        - 其他后端：默认 "all"（没有 Inductor 兜底，需要自定义 op 保证性能）
        #      已显式写了 all/none 的则不覆盖。
        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if (
                self.compilation_config.backend == "inductor"
                and self.compilation_config.mode != CompilationMode.NONE
            ):
                self.compilation_config.custom_ops.append("none")
            else:
                self.compilation_config.custom_ops.append("all")

        # This populates IR op priorities,
        # must happen after compilation mode and backend are decided,
        # but before fusion defaults are applied as those may depend on op priority.
        self.kernel_config.set_platform_defaults(self)

        # [CN] 按 optimization_level（O0~O3）套用那张预设默认值表。
        #      必须放在 mode / backend 决定之后 —— 因为表里可能有依赖它们的 lambda。
        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)
        # [CN] 这里是个自检：表必须覆盖这个字段，否则说明 OPTIMIZATION_LEVEL_* 表写漏了。
        #      用 ValueError 而不是 assert，因为 assert 可能被 -O 去掉。
        if self.kernel_config.enable_flashinfer_autotune is None:
            raise ValueError(
                "KernelConfig.enable_flashinfer_autotune must be set after applying "
                "optimization level defaults."
            )

        self._maybe_disable_dynamic_sd_for_data_parallel()
        self._maybe_override_dynamic_sd_cudagraph_mode()

        if (
            self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
            and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
        ):
            logger.info_once(
                "Cudagraph mode %s is not compatible with compilation mode %s."
                "Overriding to NONE.",
                self.compilation_config.cudagraph_mode,
                self.compilation_config.mode,
            )
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # [CN] 序列并行（SP）的依赖关系：fuse_gemm_comms（async TP）建立在 SP 之上，
        #      所以开了前者必须强制开后者。
        # async tp is built on top of sequence parallelism and requires it.
        pass_config = self.compilation_config.pass_config
        if pass_config.fuse_gemm_comms:
            pass_config.enable_sp = True
        if pass_config.enable_sp:
            if self.parallel_config.tensor_parallel_size == 1:
                logger.warning_once("Sequence Parallelism requires TP>1, disabling")
                pass_config.enable_sp = False
                pass_config.fuse_gemm_comms = False
            else:
                if pass_config.sp_min_token_num is None:
                    from vllm.compilation.passes.fusion.sequence_parallelism import (
                        get_sequence_parallelism_threshold,
                    )

                    tp_size = self.parallel_config.tensor_parallel_size
                    hidden_size = self.model_config.get_hidden_size()
                    assert isinstance(self.model_config.dtype, torch.dtype)
                    element_size = self.model_config.dtype.itemsize
                    pass_config.sp_min_token_num = get_sequence_parallelism_threshold(
                        hidden_size, tp_size, element_size
                    )

                if pass_config.sp_min_token_num is None:
                    logger.warning_once(
                        "Model hidden_size too small for the SP "
                        "threshold heuristic, disabling. To force SP, "
                        "set pass_config.sp_min_token_num manually."
                    )
                    pass_config.enable_sp = False
                    pass_config.fuse_gemm_comms = False

        from vllm.utils.torch_utils import HAS_OPAQUE_TYPE

        # [CN] fast_moe_cold_start 是 MoE 冷启动加速：首轮跳过部分编译/初始化以尽快出 token。
        #      风险点：如果投机解码的 draft 模型也带 MoE，冷启动路径可能对不上 → 默认关。
        #      另外 torch>=2.11 有了更好的实现，直接废弃开关。
        if HAS_OPAQUE_TYPE:
            # On torch >= 2.11 the hoisted OpaqueObject approach supersedes
            # fast_moe_cold_start, so force it off.
            self.compilation_config.fast_moe_cold_start = False
        elif self.compilation_config.fast_moe_cold_start is None:
            # resolve default behavior: try to be as safe as possible
            # this config is unsafe if any spec decoding draft model has a MOE.
            # We'll conservatively turn it off if we see spec decoding.
            self.compilation_config.fast_moe_cold_start = (
                self.speculative_config is None
            )

        # [CN] 关键推导之一：根据 max_num_batched_tokens / chunked prefill / 投机解码 等，
        #      算出每步最多调度多少 token。放在这里是因为它依赖前面已定稿的多个开关。
        self._set_max_num_scheduled_tokens()

        # [CN] ---- CUDA graph 的三轮降级 ----
        #      顺序很重要：先按模型类型降级（pooling / encoder-decoder），
        #      再按 KV connector 要求降级，最后按 enforce_eager 直接关掉。
        #      所以你最终看到的 cudagraph_mode 可能和你设的差好几级。
        if current_platform.support_static_graph_mode():
            # if cudagraph_mode has full cudagraphs, we need to check support
            if model_config := self.model_config:
                if (
                    self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                    and model_config.pooler_config is not None
                ):
                    logger.warning_once(
                        "Pooling models do not support full cudagraphs. "
                        "Overriding cudagraph_mode to PIECEWISE."
                    )
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                elif (
                    model_config.is_encoder_decoder
                    and self.compilation_config.cudagraph_mode
                    not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
                ):
                    logger.info_once(
                        "Encoder-decoder models do not support %s. "
                        "Overriding cudagraph_mode to FULL_DECODE_ONLY.",
                        self.compilation_config.cudagraph_mode.name,
                    )
                    self.compilation_config.cudagraph_mode = (
                        CUDAGraphMode.FULL_DECODE_ONLY
                    )

            # Check if KV connector requires PIECEWISE mode for CUDA graphs
            if (
                self.kv_transfer_config is not None
                and self.kv_transfer_config.is_kv_transfer_instance
                and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            ):
                # Lazy import to avoid circular dependencies
                from vllm.distributed.kv_transfer.kv_connector.factory import (
                    KVConnectorFactory,
                )

                connector_cls = KVConnectorFactory.get_connector_class(
                    self.kv_transfer_config
                )
                if connector_cls.requires_piecewise_for_cudagraph(
                    self.kv_transfer_config.kv_connector_extra_config
                ):
                    logger.warning_once(
                        "KV connector %s requires PIECEWISE CUDA graph mode "
                        "due to layerwise async operations that cannot be "
                        "captured in CUDA graphs. "
                        "Overriding cudagraph_mode from %s to PIECEWISE.",
                        connector_cls.__name__,
                        self.compilation_config.cudagraph_mode.name,
                    )
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

            # disable cudagraph when enforce eager execution
            if self.model_config is not None and self.model_config.enforce_eager:
                logger.info_once("Cudagraph is disabled under eager mode")
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                # override related settings when enforce eager
                self.compilation_config.max_cudagraph_capture_size = 0
                self.compilation_config.cudagraph_capture_sizes = []
            else:
                self.compilation_config.cudagraph_num_of_warmups = 1

            # [CN] 决定要捕获哪些 batch size 的 CUDA graph（显存与覆盖率的权衡）
            self._set_cudagraph_sizes()

        else:
            # [CN] 平台不支持静态图（如某些 CPU / 未适配的加速器）→ 整体关掉
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # [CN] kv_sharing_fast_prefill 与 EAGLE 互斥（硬失败）：
        #      fast prefill 会跳过部分 prompt token 的 logits 计算，
        #      而 EAGLE 需要每个位置的准确 logits 来做 draft/verify。
        if self.cache_config.kv_sharing_fast_prefill:
            if (
                self.speculative_config is not None
                and self.speculative_config.use_eagle()
            ):
                raise ValueError(
                    "Fast prefill optimization for KV sharing is not "
                    "compatible with EAGLE as EAGLE requires correct logits "
                    "for all tokens while fast prefill gives incorrect logits "
                    "for prompt tokens."
                )

            logger.warning_once(
                "--kv-sharing-fast-prefill requires changes on model side for "
                "correctness and to realize prefill savings."
            )

        if (
            self.model_config
            and self.model_config.architecture == "WhisperForConditionalGeneration"
            and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"
        ):
            logger.warning_once(
                "Whisper is known to have issues with "
                "forked workers. If startup is hanging, "
                "try setting 'VLLM_WORKER_MULTIPROC_METHOD' "
                "to 'spawn'."
            )

        if (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
            and not self.cache_config.enable_prefix_caching
        ):
            logger.warning_once(
                "KV cache events are on, but prefix caching is not enabled. "
                "Use --enable-prefix-caching to enable."
            )
        if (
            self.kv_events_config is not None
            and self.kv_events_config.publisher != "null"
            and not self.kv_events_config.enable_kv_cache_events
        ):
            logger.warning_once(
                "KV cache events are disabled, "
                "but the scheduler is configured to publish them. "
                "Modify KVEventsConfig.enable_kv_cache_events "
                "to True to enable."
            )
        # [CN] 平台最后一道"检查并改写"（与前面的 apply_config_platform_defaults 呼应：
        #      那个填默认值，这个做平台专属的合法性检查与修正）
        current_platform.check_and_update_config(self)

        self._resolve_allow_missing_mm_embeddings()
        self._resolve_mm_processor_device()
        self._validate_mm_processor_device()

        # [CN] V1 / V2 两套 model runner 各自有一份"不支持特性清单"，
        #      在这里集中比对并报错。新增特性时必须同步这两份清单，否则会漏检。
        if self.use_v2_model_runner:
            self._validate_v2_model_runner()
        else:
            self._validate_v1_model_runner()

        self._validate_batch_sharded_sampling()
        self._validate_adaptive_verification()

        # [CN] 编译范围要**重算**：平台层刚才可能改了 max_num_batched_tokens 等上游量，
        #      所以编译范围必须在平台更新之后、而不是之前确定。这是一处典型的顺序依赖。
        # Re-compute compile ranges after platform-specific config updates
        # (e.g., XPU may lower max_num_batched_tokens when MLA is enabled)
        self._set_compile_ranges()

        # Do this after all the updates to compilation_config.mode
        effective_dp_size = (
            self.parallel_config.data_parallel_size
            if self.model_config is None or self.model_config.is_moe
            else 1
        )
        self.compilation_config.set_splitting_ops_for_v1(
            all2all_backend=self.parallel_config.all2all_backend,
            data_parallel_size=effective_dp_size,
        )

        if self.compilation_config.pass_config.enable_sp:
            # With pipeline parallelism, native rms norm tracing errors due to
            # incorrect residual shape.
            # Use custom rms norm to unblock. In the future,
            # the pass will operate on higher-level IR to avoid the issue.
            # TODO: https://github.com/vllm-project/vllm/issues/27894
            if self.compilation_config.mode != CompilationMode.VLLM_COMPILE:
                logger.warning_once(
                    "Sequence parallelism is enabled, but running in wrong "
                    "vllm compile mode: %s.",
                    self.compilation_config.mode,
                )

            if self.parallel_config.pipeline_parallel_size > 1:
                if "-rms_norm" not in self.compilation_config.custom_ops:
                    self.compilation_config.custom_ops.append("+rms_norm")
                else:
                    logger.warning_once(
                        "Sequence parallelism not supported with "
                        "native rms_norm when using %s, "
                        "this will likely lead to an error.",
                        "pipeline parallelism",
                    )

        # [CN] cudagraph 的**最终一致性检查**（所有降级都跑完之后）。
        #      这里的 assert 表达的是不变量：PIECEWISE 模式必然要求走 vLLM 编译流水线，
        #      否则前面的降级逻辑有 bug。
        # final check of cudagraph mode after all possible updates
        if current_platform.is_cuda_alike():
            if (
                self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                and self.model_config is not None
                and not self.model_config.disable_cascade_attn
                and not self.compilation_config.cudagraph_mode.has_piecewise_cudagraphs()  # noqa: E501
            ):
                logger.warning_once(
                    "No piecewise cudagraph for executing cascade attention. "
                    "Will fall back to eager execution if a batch runs into "
                    "cascade attentions."
                )

            if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
                assert (
                    self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                    or envs.VLLM_USE_BREAKABLE_CUDAGRAPH
                ), (
                    "Compilation mode should be CompilationMode.VLLM_COMPILE "
                    "when cudagraph_mode piecewise cudagraphs is used, "
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"
                )
        if (
            self.model_config
            and envs.VLLM_BATCH_INVARIANT
            and not self.model_config.disable_cascade_attn
        ):
            self.model_config.disable_cascade_attn = True
            logger.warning_once(
                "Disabling cascade attention when VLLM_BATCH_INVARIANT is enabled.",
            )

        if self.parallel_config.use_ubatching:
            a2a_backend = self.parallel_config.all2all_backend
            assert a2a_backend in [
                "deepep_low_latency",
                "deepep_high_throughput",
                "nixl_ep",
            ], (
                "Microbatching currently only supports the deepep_low_latency, "
                "deepep_high_throughput, and nixl_ep all2all backends. "
                f"{a2a_backend} is not supported. To fix use "
                "--all2all-backend=deepep_low_latency, "
                "--all2all-backend=deepep_high_throughput, or "
                "--all2all-backend=nixl_ep and install the matching kernels."
            )

            if not self.model_config.disable_cascade_attn:
                self.model_config.disable_cascade_attn = True
                logger.warning_once("Disabling cascade attention when DBO is enabled.")

        # [CN] 兜底：如果时间戳为空（理论上不会），用随机 UUID 前 5 位。
        #      说明 instance_id 只要求"唯一"，不要求"可读/稳定"。
        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        if self.reasoning_config is not None and self.model_config is not None:
            self.reasoning_config.initialize_token_ids(self.model_config)
            if not self.reasoning_config.enabled:
                logger.warning_once(
                    "Auto-initialization of reasoning token IDs failed. "
                    "Please check whether your reasoning parser has implemented "
                    "the `reasoning_start_str` and `reasoning_end_str`."
                )

        # Resolve kv_offloading-derived connector name into kv_transfer_config
        # before the HMA check below, which inspects the connector class.
        self._post_init_kv_transfer_config()

        if self.is_mm_encoder_only and self.cache_config.enable_prefix_caching:
            # Such an instance publishes encoder embeddings and runs no language
            # model, so it holds no KV cache for prefix caching to reuse and its
            # coordinator would have no group to manage.
            logger.info(
                "Disabling prefix caching: this instance runs the "
                "multi-modal encoder only."
            )
            self.cache_config.enable_prefix_caching = False

        # [CN] 混合 KV cache 管理器（HMA）：统一管理 attention 的 KV cache 与 Mamba/SSM 的 state cache。
        #      是否启用是**三态**（None 自动 / False 显式开 / True 显式关），规则如下：
        #        None  → 平台不支持、或命中已知不兼容组合（chunked local attn、不支持 HMA 的
        #                KV connector）时自动关；否则默认开
        #        False → 用户显式要求开，但运行时发现不兼容 → **报错**（尊重用户意图，不静默降级）
        #        True  → 用户显式关，永远尊重
        #      影响面：混合 SSM 模型（Jamba/Bamba）**必须**有 HMA 否则起不来；
        #      滑动窗口模型没有 HMA 只是性能下降。
        #
        #      注意下面收集 need_disable 的过程是"累加或"：多个条件任一命中就关，
        #      且为了避免误报，warning 延后打印（此时还不知道模型是否真的是 hybrid）。
        # Hybrid KV cache manager (HMA) runtime rules:
        # - Explicit enable (--no-disable-kv-cache-manager): error if runtime
        #   disables it
        # - No preference: auto-disable for unsupported features or connector configs
        # - Explicit disable (--disable-kv-cache-manager): always respect it
        need_disable_hybrid_kv_cache_manager = False
        # logger should only print warning message for hybrid models. As we
        # can't know whether the model is hybrid or not now, so we don't log
        # warning message here and will log it later.
        if not current_platform.support_hybrid_kv_cache():
            # Hybrid KV cache manager is not supported on non-GPU platforms.
            need_disable_hybrid_kv_cache_manager = True
        if (
            self.model_config is not None
            and self.model_config.attention_chunk_size is not None
        ):
            if (
                self.speculative_config is not None
                and self.speculative_config.use_eagle()
            ):
                # Hybrid KV cache manager is not yet supported with chunked
                # local attention + eagle.
                need_disable_hybrid_kv_cache_manager = True
            elif not envs.VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE:
                logger.warning(
                    "There is a latency regression when using chunked local"
                    " attention with the hybrid KV cache manager. Disabling"
                    " it, by default. To enable it, set the environment "
                    "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1."
                )
                # Hybrid KV cache manager is not yet supported with chunked
                # local attention.
                need_disable_hybrid_kv_cache_manager = True

        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # Auto-disable HMA only when the connector config does not support it.
            if self.kv_transfer_config is not None:
                from vllm.distributed.kv_transfer.kv_connector.factory import (
                    KVConnectorFactory,
                )

                if not KVConnectorFactory.supports_hma_config(self.kv_transfer_config):
                    need_disable_hybrid_kv_cache_manager = True
                    logger.warning(
                        "Turning off hybrid kv cache manager because "
                        "`--kv-transfer-config` selects a KV connector that "
                        "does not support it. Impact: hybrid SSM models "
                        "(e.g. Jamba, Bamba) require HMA and will fail at "
                        "startup without it; models with sliding window "
                        "attention will run with reduced performance. "
                        "To add HMA support to a KV connector, subclass "
                        "`SupportsHMA` defined in kv_connector/v1/base.py "
                        "(for MultiConnector, all child connectors must "
                        "support HMA)."
                    )
            self.scheduler_config.disable_hybrid_kv_cache_manager = (
                need_disable_hybrid_kv_cache_manager
            )
        elif (
            self.scheduler_config.disable_hybrid_kv_cache_manager is False
            and need_disable_hybrid_kv_cache_manager
        ):
            raise ValueError(
                "Hybrid KV cache manager was explicitly enabled but is not "
                "supported in this configuration. Consider omitting the "
                "--no-disable-hybrid-kv-cache-manager flag to let vLLM decide"
                " automatically."
            )

        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # Default to enable HMA if not explicitly disabled by user or logic above.
            self.scheduler_config.disable_hybrid_kv_cache_manager = False

        # [CN] debug dump 路径：两者都设时**环境变量优先**并覆盖配置值。
        #      这与多数配置"显式参数 > 环境变量"的优先级相反，是一个例外，值得留意。
        if self.compilation_config.debug_dump_path:
            self.compilation_config.debug_dump_path = (
                self.compilation_config.debug_dump_path.absolute().expanduser()
            )
        if envs.VLLM_DEBUG_DUMP_PATH is not None:
            env_path = Path(envs.VLLM_DEBUG_DUMP_PATH).absolute().expanduser()
            if self.compilation_config.debug_dump_path:
                logger.warning(
                    "Config-specified debug dump path is overridden"
                    " by VLLM_DEBUG_DUMP_PATH to %s",
                    env_path,
                )
            self.compilation_config.debug_dump_path = env_path

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        self._verify_kv_transfer_compat()
        # Log the custom passes that are enabled
        self.compilation_config.pass_config.log_enabled_passes()

    # [CN] 开启序列并行后，batch size 必须能被 tp_size 整除（否则序列无法均分到各 TP rank）。
    #      这里把候选 size 里不整除的剔除，并打印被剔除的列表——方便用户理解
    #      "为什么某些 batch size 拿不到 CUDA graph"。
    def update_sizes_for_sequence_parallelism(self, possible_sizes: list) -> list:
        # remove the sizes that not multiple of tp_size when
        # enable sequence parallelism
        removed_sizes = [
            size
            for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size != 0
        ]
        if removed_sizes:
            logger.warning(
                "Batch sizes %s are removed because they are not "
                "multiple of tp_size %d when "
                "sequence parallelism is enabled",
                removed_sizes,
                self.parallel_config.tensor_parallel_size,
            )

        return [
            size
            for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size == 0
        ]

    # [CN] 只在启用投机解码时才需要调整。
    #
    #      背景：投机解码时一次 forward 要同时容纳「被验证的 token」+「新草稿 token」，
    #      所以每步实际调度的 token 上限会比 max_num_batched_tokens 少一部分 ——
    #      少掉的这部分就是 scheduled_token_delta（草稿额外占用的槽位）。
    #
    #      两个报错点：
    #        - max_num_scheduled_tokens <= 0：max_num_batched_tokens 太小，连一个 token 都排不下
    #        - max_num_batched_tokens <= delta：预算被草稿槽位吃光
    #      还有一条 8192 的性能提醒（低于此值吞吐会明显下降）。
    #
    #      ⚠️ 注意这里改的是 scheduler_config.max_num_scheduled_tokens，
    #      而 max_num_batched_tokens 保持不变 —— 二者是不同的量。
    def _set_max_num_scheduled_tokens(self):
        """
        In most cases, the scheduler may schedule a batch with as many tokens as the
        worker is configured to handle.
        """
        if self.speculative_config is not None:
            scheduled_token_delta = (
                self.speculative_config.max_num_new_slots_for_drafting
            )
            max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
            if self.scheduler_config.max_num_scheduled_tokens is None:
                self.scheduler_config.max_num_scheduled_tokens = max_num_batched_tokens

            if self.scheduler_config.max_num_scheduled_tokens <= 0:
                raise ValueError(
                    "max_num_scheduled_tokens is set to"
                    f" {self.scheduler_config.max_num_scheduled_tokens} based on"
                    " the speculative decoding settings, which does not allow"
                    " any tokens to be scheduled. Increase max_num_batched_tokens"
                    " to accommodate the additional draft token slots, or decrease"
                    " num_speculative_tokens."
                )
            if self.scheduler_config.max_num_scheduled_tokens < 8192:
                logger.warning_once(
                    "max_num_scheduled_tokens is set to"
                    f" {self.scheduler_config.max_num_scheduled_tokens} based on"
                    " the speculative decoding settings. This may lead to suboptimal"
                    " performance. Consider increasing max_num_batched_tokens to"
                    " accommodate the additional draft token slots, or decrease"
                    " num_speculative_tokens.",
                )

            if max_num_batched_tokens <= scheduled_token_delta:
                raise ValueError(
                    "VllmConfig does not have enough slots to schedule a token and"
                    " support the speculative decoding settings."
                    f" Got {max_num_batched_tokens=} and {scheduled_token_delta=}."
                )

    # [CN] 【CUDA graph 捕获尺寸的决策逻辑 —— 显存与覆盖率的权衡核心】
    #
    #      默认候选列表的形状（见下方英文 docstring）：
    #        [1, 2, 4] + 8 的倍数到 256 + 16 的倍数到 max_graph_size
    #      即**小 batch 密、大 batch 疏**：小 batch 出现频率高、且绝对填充浪费小，
    #      值得逐个捕获；大 batch 用 16 的步长，靠"向上取整到最近的已捕获尺寸"来复用。
    #
    #      运行时的匹配规则（务必记住）：
    #        - batch <= 某个已捕获尺寸 → 向上补齐(pad)到最近的尺寸，用对应 graph
    #        - batch > 最大已捕获尺寸 → **完全不用 CUDA graph**，退回 eager
    #      所以 max_cudagraph_capture_size 设太小会导致大 batch 直接失去图优化。
    #
    #      显存代价：每个尺寸都要一份独立的 graph 副本，尺寸数量直接线性影响显存。
    #
    #      投机解码的特殊处理：一次 decode 每请求是 decode_query_len 个 token（>1），
    #      这时不能按 token 数建网格（会产生几百个尺寸且多数不可用，
    #      因为 dispatch 要求恰好是 query_len 的倍数），改为按**请求数**建网格。
    def _set_cudagraph_sizes(self):
        """
        vLLM defines the default candidate list of batch sizes for CUDA graph
        capture as:

        ```python
        default_max_graph_size = 1024 if is_data_center_blackwell else 512
        decode_query_len = self.uniform_decode_query_len
        max_graph_size = min(
            max_num_seqs * decode_query_len * 2, default_max_graph_size
        )
        # 1, 2, 4, then multiples of 8 up to 256 and then multiples of 16
        # up to max_graph_size
        cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(
            range(256, max_graph_size + 1, 16))

        `max_num_batched_tokens` is also appended to the list if it fits
        within `max_cudagraph_capture_size`, so the max batch size is captured
        even when off-stride. Uniform decode sizes are appended when they fit
        within the platform's default capture ceiling, since they need not land
        on an 8- or 16-token stride.

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in ascending order).

        These sizes are used to capture and reuse CUDA graphs for
        performance-critical paths (e.g., decoding). Capturing enables
        significantly faster kernel dispatch by avoiding Python overhead. The
        list is then filtered based on `max_num_batched_tokens` (e.g., 8192 on
        most GPUs), which controls the total allowed number of tokens in a
        batch. Since each sequence may have a variable number of tokens, the
        maximum usable batch size will depend on actual sequence lengths.

        Example:
            With `max_num_batched_tokens = 8192`, and typical sequences
            averaging ~32 tokens, most practical batch sizes fall below 256.
            However, the system will still allow capture sizes up to the
            platform default if shape and memory permit.

        Note:
            If users explicitly specify cudagraph capture sizes in the
            compilation config, those will override this default logic.
            At runtime:

            - If batch size <= one of the `cudagraph_capture_sizes`, the closest
            padded CUDA graph will be used.
            - If batch size > largest `cudagraph_capture_sizes`, cudagraph will
            not be used.
        """

        if (
            self.model_config is not None
            and not self.model_config.enforce_eager
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # determine the initial max_cudagraph_capture_size
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            # Decode sizes to cover, in tokens. Populated only when a request
            # is more than one token wide and only when the default is computed
            # here, so an explicit capture range is left exactly as configured.
            uniform_decode_sizes: list[int] = []
            if max_cudagraph_capture_size is None:
                from vllm.platforms import current_platform

                default_max_graph_size = (
                    1024 if current_platform.is_device_capability_family(100) else 512
                )
                decode_query_len = self.uniform_decode_query_len
                max_num_seqs = self.scheduler_config.max_num_seqs
                max_cudagraph_capture_size = min(
                    max_num_seqs * decode_query_len * 2, default_max_graph_size
                )
                if decode_query_len > 1:
                    # A uniform decode batch is decode_query_len tokens per
                    # request, so the widest one is far outside this ceiling.
                    # Coverage comes from appending the decode sizes rather than
                    # extending the token-strided grid. Extending that grid to
                    # the widest decode size produces 581 sizes at
                    # max_num_seqs=512 and 16 draft tokens, versus 100 with the
                    # request-count grid.
                    #
                    # The grid would not buy decode coverage anyway. Dispatch
                    # requires an exact multiple of decode_query_len, so a
                    # token-strided entry is only usable when it happens to be
                    # one; at query length 17 a captured 560 rounds to 561 and
                    # is rejected. Scaling a request-count grid keeps every
                    # entry usable and the count comparable to the non-
                    # speculative case.
                    def request_counts(max_reqs: int) -> list[int]:
                        # At most the platform default number of requests,
                        # mirroring the one-token-per-request decode ceiling.
                        max_reqs = min(max_reqs, default_max_graph_size)
                        counts = [n for n in (1, 2, 4) if n <= max_reqs]
                        counts += list(range(8, min(max_reqs + 1, 256), 8))
                        counts += list(range(256, max_reqs + 1, 16))
                        return sorted(set(counts + [max_reqs]))

                    # Dynamic speculative decoding picks the draft width from
                    # the batch size, so a decode step is only uniform within a
                    # tier and each tier needs its own sizes. Scaling by the
                    # widest one alone leaves the narrower tiers short: the
                    # manager rounds a capture size up to a multiple of the
                    # tier's query length and drops it once the implied request
                    # count exceeds max_num_seqs, so at query length 3 sizes
                    # built from 17 stop covering at 227 of 256 requests.
                    decode_tiers = [(decode_query_len, max_num_seqs)]
                    speculative_config = self.speculative_config
                    if (
                        speculative_config is not None
                        and speculative_config.uses_dynamic_speculative_decoding()
                    ):
                        from vllm.v1.spec_decode.dynamic.utils import (
                            build_dynamic_sd_schedule_lookup,
                        )

                        schedule = (
                            speculative_config.num_speculative_tokens_per_batch_size
                        )
                        assert schedule is not None
                        # Read the tiers off the dense lookup the scheduler
                        # runs on, so the clamp against num_speculative_tokens
                        # and the carry-forward through gaps and the tail
                        # cannot drift from it. Validation lives elsewhere; an
                        # invalid schedule keeps the single-tier default.
                        try:
                            dense_schedule = build_dynamic_sd_schedule_lookup(
                                schedule,
                                vllm_max_batch_size=max_num_seqs,
                                vllm_num_speculative_tokens=self.num_speculative_tokens,
                            )
                        except ValueError:
                            pass
                        else:
                            # Ascending batch size, so the last write per
                            # query length is the widest batch running at it.
                            widest_batch: dict[int, int] = {}
                            for batch_size, num_spec in enumerate(
                                dense_schedule[1:], start=1
                            ):
                                widest_batch[num_spec + 1] = batch_size
                            decode_tiers = list(widest_batch.items())

                    uniform_decode_sizes = sorted(
                        {
                            n * query_len
                            for query_len, tier_max_reqs in decode_tiers
                            for n in request_counts(tier_max_reqs)
                            if n * query_len <= max_cudagraph_capture_size
                        }
                    )
            # [CN] 用 max_num_batched_tokens 再夹一次上界：
            #      超过它的尺寸运行时根本不会出现，捕获了纯属浪费显存。
            max_num_tokens = self.scheduler_config.max_num_batched_tokens
            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

            assert max_cudagraph_capture_size >= 1, (
                "Maximum cudagraph size should be greater than or equal to 1 "
                "when using cuda graph."
            )

            # determine the cudagraph_capture_sizes
            if self.compilation_config.cudagraph_capture_sizes is not None:
                assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                    "cudagraph_capture_sizes should contain at least one element "
                    "when using cuda graph."
                )
                # de-duplicate the sizes provided by the config
                dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
                cudagraph_capture_sizes = [
                    i for i in dedup_sizes if i <= max_num_tokens
                ]
                # sort to make sure the sizes are in ascending order
                cudagraph_capture_sizes.sort()
            else:
                # [CN] performance_mode 的两种取向：
                #        interactivity（低延迟优先）：1..32 **逐个**捕获，padding 浪费最小
                #        balanced / throughput：走下面的 [1,2,4]+步长8+步长16 稀疏网格，省显存
                #      这是"延迟 vs 显存"的显式取舍点。
                if self.performance_mode == "interactivity":
                    # Fine-grained CUDA graphs at small batch sizes
                    # for minimal padding overhead
                    interactivity_max = min(max_cudagraph_capture_size, 32)
                    cudagraph_capture_sizes = list(range(1, interactivity_max + 1))
                else:
                    cudagraph_capture_sizes = [
                        i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                    ]
                if max_cudagraph_capture_size >= 8:
                    # Step size 8 for small batch sizes, up to 256(not included)
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if max_cudagraph_capture_size >= 256:
                    # Step size 16 for larger batch sizes
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )
                # ensure max_num_tokens is captured if within max capture size
                if (
                    max_num_tokens <= max_cudagraph_capture_size
                    and max_num_tokens not in cudagraph_capture_sizes
                ):
                    cudagraph_capture_sizes.append(max_num_tokens)
                # Preserve the platform's default capture ceiling. Larger
                # uniform decode batches fall back to eager execution unless
                # users explicitly configure wider capture sizes.
                cudagraph_capture_sizes += [
                    size for size in uniform_decode_sizes if size <= max_num_tokens
                ]
                # de-duplicate and sort the sizes
                cudagraph_capture_sizes = sorted(set(cudagraph_capture_sizes))

            if (
                self.parallel_config.tensor_parallel_size > 1
                and self.compilation_config.pass_config.enable_sp
            ):
                # Sequence parallelism only captures TP-divisible sizes, so a
                # wider non-divisible decode batch cannot be captured under SP.
                cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(
                    cudagraph_capture_sizes
                )

            # user-specific compilation_config.max_cudagraph_capture_size get
            # truncated to valid_max_size when they are inconsistent.
            valid_max_size = (
                cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
            )
            if (
                self.compilation_config.max_cudagraph_capture_size is not None
                and self.compilation_config.max_cudagraph_capture_size != valid_max_size
            ):
                # raise error only when both two flags are user-specified
                # and they are inconsistent with each other
                if self.compilation_config.cudagraph_capture_sizes is not None:
                    raise ValueError(
                        "customized max_cudagraph_capture_size"
                        f"(={self.compilation_config.max_cudagraph_capture_size}) "
                        "should be consistent with the max value of "
                        f"cudagraph_capture_sizes(={valid_max_size})"
                    )

                logger.warning(
                    "Truncating max_cudagraph_capture_size to %d",
                    valid_max_size,
                )
            # always set the final max_cudagraph_capture_size
            self.compilation_config.max_cudagraph_capture_size = valid_max_size

            if self.compilation_config.cudagraph_capture_sizes is not None and len(
                cudagraph_capture_sizes
            ) < len(self.compilation_config.cudagraph_capture_sizes):
                # If users have specified capture sizes, we only need to
                # compare the lens before and after modification since the modified
                # list is only the subset of the original list.
                logger.warning(
                    (
                        "cudagraph_capture_sizes specified in compilation_config"
                        " %s is overridden by config %s"
                    ),
                    self.compilation_config.cudagraph_capture_sizes,
                    cudagraph_capture_sizes,
                )
            # always write back the final sizes
            self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes

        else:
            # no cudagraph in use
            self.compilation_config.max_cudagraph_capture_size = 0
            self.compilation_config.cudagraph_capture_sizes = []

        # complete the remaining process.
        self.compilation_config.post_init_cudagraph_sizes()

    # [CN] 编译范围（compile ranges）= 需要为哪些 token 数量区间各编译一份特化代码。
    #
    #      为什么需要：某些融合算子（如 allreduce+rms_norm 融合、序列并行的切分）
    #      只在 token 数处于特定范围内才成立/才划算。于是把 [0, max_num_batched_tokens]
    #      切成若干段，每段编译一份。
    #
    #      这里的"端点"来自三个来源，逐个 append：
    #        1. max_num_batched_tokens（总上界）
    #        2. allreduce-rms 融合的可用上限（受通信 buffer 大小限制，按 hidden_size×dtype 换算成 token 数）
    #        3. 序列并行的 min/max token 阈值
    #      最后交给 compilation_config 排序去重成实际区间。
    def _set_compile_ranges(self):
        """
        Set the compile ranges for the compilation config.
        """
        compilation_config = self.compilation_config
        computed_compile_ranges_endpoints = []

        # The upper bound of the compile ranges is the max_num_batched_tokens.
        compile_range_end = self.scheduler_config.max_num_batched_tokens
        if compile_range_end is not None:
            computed_compile_ranges_endpoints.append(compile_range_end)

        # Add the compile ranges for flashinfer/aiter.
        if compilation_config.pass_config.fuse_allreduce_rms:
            tp_size = self.parallel_config.tensor_parallel_size
            from vllm._aiter_ops import rocm_aiter_ops

            max_size: int | None = None
            if rocm_aiter_ops.is_custom_all_reduce_enabled():
                from vllm.distributed.device_communicators.aiter_custom_all_reduce import (  # noqa: E501
                    AiterCustomAllreduce,
                )

                max_size = AiterCustomAllreduce.effective_max_size()
            else:
                max_size = compilation_config.pass_config.flashinfer_max_size(tp_size)
            if max_size is not None and self.model_config is not None:
                assert isinstance(self.model_config.dtype, torch.dtype)
                max_token_num = max_size // (
                    self.model_config.get_hidden_size()
                    * self.model_config.dtype.itemsize
                )
                if compile_range_end is not None and max_token_num < compile_range_end:
                    computed_compile_ranges_endpoints.append(max_token_num)
                else:
                    logger.debug(
                        "Max num batched tokens below allreduce-rms fusion threshold, "
                        "allreduce-rms fusion will be enabled for all num_tokens."
                    )

        # Add the compile ranges for sequence parallelism
        if compilation_config.pass_config.enable_sp:
            pass_config = compilation_config.pass_config

            # Calculate min_token_num if not explicitly provided
            # User override works regardless of hidden_size
            if pass_config.sp_min_token_num is None:
                from vllm.compilation.passes.fusion.sequence_parallelism import (
                    get_sequence_parallelism_threshold,
                )

                tp_size = self.parallel_config.tensor_parallel_size
                hidden_size = self.model_config.get_hidden_size()
                assert isinstance(self.model_config.dtype, torch.dtype)
                element_size = self.model_config.dtype.itemsize
                pass_config.sp_min_token_num = get_sequence_parallelism_threshold(
                    hidden_size, tp_size, element_size
                )

            min_token_num = pass_config.sp_min_token_num
            max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
            if min_token_num is not None and (
                max_num_batched_tokens is not None
                and min_token_num < max_num_batched_tokens
                and min_token_num > 1
            ):
                # Add endpoint at min_token_num - 1 to ensure SP applies
                # starting from min_token_num
                # This creates ranges: [1, min-1] (no SP), [min, max] (SP applies)
                computed_compile_ranges_endpoints.append(min_token_num - 1)

        if compilation_config.pass_config.fuse_rope_kvcache:
            max_token_num = (
                compilation_config.pass_config.rope_kvcache_fusion_max_token_num
            )
            if max_token_num is not None:
                if compile_range_end is not None and max_token_num < compile_range_end:
                    computed_compile_ranges_endpoints.append(max_token_num)
                else:
                    logger.debug(
                        "Max num batched tokens below rope+kvcache fusion threshold, "
                        "rope+kvcache fusion enabled for num_tokens <= %d.",
                        compile_range_end,
                    )

        if compilation_config.pass_config.fuse_qk_norm_rope_kvcache:
            max_token_num = (
                compilation_config.pass_config.rope_kvcache_fusion_max_token_num
            )
            if max_token_num is not None:
                if compile_range_end is not None and max_token_num < compile_range_end:
                    computed_compile_ranges_endpoints.append(max_token_num)
                else:
                    logger.debug(
                        "Max num batched tokens below qk_norm+rope+kvcache "
                        "fusion threshold, fusion enabled for "
                        "num_tokens <= %d.",
                        compile_range_end,
                    )

        if compilation_config.compile_ranges_endpoints is not None:
            for x in compilation_config.compile_ranges_endpoints:
                assert isinstance(x, int)
                assert x > 0, f"Invalid compile range endpoint: {x}"
                if compile_range_end is not None and x < compile_range_end and x > 1:
                    computed_compile_ranges_endpoints.append(x)
        compilation_config.compile_ranges_endpoints = sorted(
            computed_compile_ranges_endpoints
        )

    # [CN] 「按架构定制配置」的钩子入口。
    #
    #      设计动机：某些模型的特殊需求（如 Jamba 的 Mamba 块尺寸、DeepSeek 的 MLA 参数）
    #      不适合写死在通用配置里，于是允许每个架构注册一个 MODELS_CONFIG_MAP 条目，
    #      在这里回调它的 verify_and_update_config(self) —— 可以**就地修改整个 VllmConfig**。
    #
    #      这是 vLLM 配置体系里少见的"模型反向修改全局配置"的路径，
    #      排查"我的配置怎么被改了"时要想到这里。
    #
    #      config_updated 标志：防止重复执行（VllmConfig 可能被多次构造/派生）。
    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        # Avoid running try_verify_and_update_config multiple times
        if getattr(self.model_config, "config_updated", False):
            return
        self.model_config.config_updated = True

        architecture = self.model_config.architecture
        if architecture is None:
            return

        from vllm.model_executor.models import ModelRegistry
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            HybridAttentionMambaModelConfig,
        )

        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is None:
            # `architecture` may be an HF base-model name (e.g. "Mamba2Model"
            # when `architectures` is omitted); normalize to the resolved arch
            # so per-arch config hooks are not skipped.
            architecture = ModelRegistry._normalize_arch(
                architecture, self.model_config
            )
            cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)

        if self.model_config.convert_type == "classify":
            # Maybe convert ForCausalLM into ForSequenceClassification model.
            from vllm.model_executor.models.adapters import SequenceClassificationConfig

            SequenceClassificationConfig.verify_and_update_config(self)

        if hasattr(self.model_config, "model_weights") and is_runai_obj_uri(
            self.model_config.model_weights
        ):
            if self.load_config.load_format == "auto":
                logger.info(
                    "Detected Run:ai model config. "
                    "Overriding `load_format` to 'runai_streamer'"
                )
                self.load_config.load_format = "runai_streamer"
            elif self.load_config.load_format not in (
                "modelexpress",
                "runai_streamer",
                "runai_streamer_sharded",
            ):
                raise ValueError(
                    f"To load a model from object storage (S3/GCS/Azure), "
                    f"'load_format' must be 'modelexpress', 'runai_streamer' or "
                    f"'runai_streamer_sharded', "
                    f"but got '{self.load_config.load_format}'. "
                    f"Model: {self.model_config.model}"
                )

    # [CN] 编译调试产物按 rank 分目录：多进程编译会各自 dump 一大堆文件，
    #      混在一个目录里无法分辨。命名包含 TP rank 与 DP index 两层。
    def compile_debug_dump_path(self) -> Path | None:
        """Returns a rank-aware path for dumping
        torch.compile debug information.
        """
        if self.compilation_config.debug_dump_path is None:
            return None
        tp_rank = self.parallel_config.rank
        dp_rank = self.parallel_config.data_parallel_index
        append_path = f"rank_{tp_rank}_dp_{dp_rank}"
        path = self.compilation_config.debug_dump_path / append_path
        return path

    # [CN] 注意 __str__ 只是**信息性**的，且只挑选了最常用的字段。
    #      它不等于序列化（序列化走 msgspec/json），也不保证覆盖所有配置。
    #      日志里看到的配置摘要来自这里，改动字段时记得同步。
    def __str__(self):
        return (
            f"model={self.model_config.model!r}, "
            f"speculative_config={self.speculative_config!r}, "
            f"tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init}, "
            f"tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len}, "
            f"download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size}, "  # noqa
            f"pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"data_parallel_size={self.parallel_config.data_parallel_size}, "  # noqa
            f"decode_context_parallel_size={self.parallel_config.decode_context_parallel_size}, "  # noqa
            f"dcp_comm_backend={self.parallel_config.dcp_comm_backend}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"quantization_config={self.model_config.quantization_config}, "  # noqa
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"enable_return_routed_experts={self.model_config.enable_return_routed_experts}, "  # noqa
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f"device_config={self.device_config.device}, "
            f"structured_outputs_config={self.structured_outputs_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"enable_chunked_prefill={self.scheduler_config.enable_chunked_prefill}, "  # noqa
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}, "
            f"kernel_config={self.kernel_config!r}"
        )

    def _resolve_allow_missing_mm_embeddings(self) -> None:
        """Allow `*_embeds` tensors to be omitted on disaggregated consumers.

        An EC consumer loads embeddings from its connector. A KV consumer
        receives the prompt KV produced from those embeddings, so it does not
        need the tensors either. On every other deployment a missing tensor is
        a client error and must keep failing fast in the frontend.
        """
        model_config = self.model_config
        if model_config is None:
            return
        mm_config = model_config.multimodal_config
        if mm_config is None:
            return

        ec_config = self.ec_transfer_config
        kv_config = self.kv_transfer_config
        # [CN] ⚠️ 这里是**无条件覆盖**，不尊重用户手设的值。
        #      因为它是从 EC/KV 的 consumer 角色推导出来的事实，不是偏好；
        #      手设成 True 但本实例不是 consumer 的话，会导致真正缺 tensor 时漏报错。
        # Derived, so overwrite unconditionally rather than honouring a value
        # that was set by hand.
        mm_config.allow_missing_mm_embeddings = (
            ec_config is not None and ec_config.is_ec_consumer
        ) or (kv_config is not None and kv_config.is_kv_consumer)
        if mm_config.allow_missing_mm_embeddings:
            logger.info_once(
                "EC/KV consumer: pre-computed-embedding inputs may "
                "omit the embedding tensor."
            )

    def _resolve_mm_encoder_only(self) -> None:
        """Enable encoder-only mode for a dedicated EC producer."""
        ec_config = self.ec_transfer_config
        if ec_config is None or not ec_config.is_encode_only:
            return

        model_config = self.model_config
        mm_config = model_config.multimodal_config if model_config is not None else None
        if mm_config is None:
            raise ValueError(
                "An EC producer-only instance requires a multimodal model."
            )
        mm_config.mm_encoder_only = True

    # [CN] `--mm-processor-device=auto` 的最终判定，判定条件比字面意思严格得多：
    #      "auto" 不等于"有加速器就用加速器"，而是同时满足：
    #        ① 本实例是 EC encode-only（只跑编码器，不跑 forward、不占 KV cache）
    #           → 加速器空闲，前端预处理可以独占
    #        ② 张量传输方式是 torch_shm
    #           → 否则输出要先拷回 host 再序列化，拷贝开销反而大于在设备上跑的收益
    #      两个条件任一不满足就留在 CPU。这也是"为什么我开了 auto 却还在 CPU 上"的答案。
    #
    #      注意：用户显式指定的设备在这里**不动**，留到 _validate_mm_processor_device 校验。
    def _resolve_mm_processor_device(self) -> None:
        """Settle `--mm-processor-device=auto` now that the EC role is known.

        "auto" means "the accelerator, but only where the processor has it to
        itself and its output can be handed over without a copy back to host":
        an encode-only instance whose tensor transport carries device tensors.
        Every other deployment keeps the processor on CPU.

        An explicit device -- from `--mm-processor-device` or straight from
        `mm_processor_kwargs` -- is already folded in by `MultiModalConfig`, so
        it is left alone here and validated by `_validate_mm_processor_device`.
        """
        model_config = self.model_config
        if model_config is None:
            return
        mm_config = model_config.multimodal_config
        if mm_config is None:
            return
        if mm_config.get_mm_processor_device_type() is not None:
            return

        from vllm.platforms import current_platform

        device_type = current_platform.device_type
        if device_type in ("", "cpu"):
            return

        ec_config = self.ec_transfer_config
        # An EC producer that is not also a consumer runs no forward pass and
        # allocates no KV cache, so frontend accelerator work has the device to
        # itself.
        if ec_config is None or not ec_config.is_encode_only:
            return

        if mm_config.mm_tensor_ipc != "torch_shm":
            # Any other transport serializes host bytes, so the output would be
            # copied back, and that copy costs more than running the transform
            # on device saves.
            logger.info_once(
                "EPD encoder instance: keeping the multi-modal processor on CPU "
                "because mm_tensor_ipc=%s cannot carry device tensors. Add "
                "--mm-tensor-ipc=torch_shm to run it on the accelerator.",
                mm_config.mm_tensor_ipc,
            )
            return

        mm_config.mm_processor_kwargs = {
            **(mm_config.mm_processor_kwargs or {}),
            "device": device_type,
        }
        logger.info_once(
            "EPD encoder instance: running the multi-modal processor on %s. "
            "Override with --mm-processor-device=cpu.",
            device_type,
        )

    def _validate_mm_processor_device(self) -> None:
        """Hand the EC config to `MultiModalConfig`, which owns the rule."""
        model_config = self.model_config
        if model_config is None:
            return
        mm_config = model_config.multimodal_config
        if mm_config is None:
            return

        mm_config.validate_mm_processor_device(self.ec_transfer_config)

    # [CN] V1 / V2 两套 model runner 各自维护一份"不支持特性清单"，
    #      以字符串列表的形式收集后统一报错。好处是所有不兼容项**一次性**列全，
    #      用户不必逐个试错。代价是新增特性时要记得往这两份清单里加。
    def _get_v2_model_runner_unsupported_features(self) -> list[str]:
        """Collect features not yet supported by the V2 model runner."""
        unsupported: list[str] = []
        model_config = self.model_config
        speculative_config = self.speculative_config

        if self.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            unsupported.append("stock torch.compile")

        if (
            self.compilation_config.pass_config.enable_sp
            and self.parallel_config.tensor_parallel_size > 1
        ):
            unsupported.append("sequence parallelism")

        # V2 does not implement the external_launcher (torchrun) PP-output
        # broadcast that V1 uses to keep all ranks in sync (broadcast_pp_output).
        if (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and self.parallel_config.pipeline_parallel_size > 1
        ):
            unsupported.append("pipeline parallelism with external_launcher")

        if speculative_config is not None:
            # TODO: ngram / ngram_gpu are not supported by the v2 model runner yet
            if speculative_config.method in ("ngram", "ngram_gpu"):
                unsupported.append("ngram/ngram_gpu speculative decoding")
            elif speculative_config.method not in (
                "eagle",
                "eagle3",
                "mtp",
                "dflash",
                "dspark",
                "extract_hidden_states",
            ):
                unsupported.append(f"speculative method '{speculative_config.method}'")

            # V2 EagleSpeculator does not support parallel_drafting (for P-Eagle).
            # DFlash and DSpark use parallel drafting natively in V2 via their
            # own speculators.
            if (
                speculative_config.parallel_drafting
                and speculative_config.method not in ("dflash", "dspark")
            ):
                unsupported.append("parallel drafting for EAGLE speculative decoding")

        if self.parallel_config.use_ubatching:
            unsupported.extend(self._get_dbo_unsupported_features())

        if self.parallel_config.enable_elastic_ep:
            unsupported.append("elastic expert parallelism")

        # [CN] 自定义 logits processors 有两种来源，都要拦：
        #        ① 配置里显式给的 model_config.logits_processors
        #        ② 通过 setuptools entry_points 注册的插件（"vllm.logits_processors" 组）
        #      只查①会漏掉以插件形式安装的第三方 logits processor。
        has_logitsproc_plugins = False
        if model_config is not None:
            from importlib.metadata import entry_points

            has_logitsproc_plugins = bool(entry_points(group="vllm.logits_processors"))

        if model_config is not None and (
            model_config.logits_processors or has_logitsproc_plugins
        ):
            unsupported.append("custom logits processors")

        if self.cache_config.kv_sharing_fast_prefill:
            # Will be added by https://github.com/vllm-project/vllm/pull/35045
            unsupported.append("KV sharing fast prefill")

        if self.cache_config.mamba_cache_mode == "all":
            unsupported.append("mamba cache mode 'all'")

        return unsupported

    def _get_v1_model_runner_unsupported_features(self) -> list[str]:
        unsupported: list[str] = []

        # PCP runtime support is implemented only by the V2 model runner.
        if self.parallel_config.prefill_context_parallel_size > 1:
            unsupported.append("prefill context parallel")

        # DSpark is implemented only by the V2 GPU model runner.
        if self.speculative_config:
            if self.speculative_config.method == "dspark":
                unsupported.append("dspark speculative decoding")
            if self.speculative_config.enable_adaptive_verification:
                unsupported.append("adaptive draft verification")

        # Mixed sliding/full DFlash drafts need multiple KV groups (V2 only).
        if self._dflash_needs_multi_kv_group():
            unsupported.append("mixed sliding/full dflash drafts")

        # The DFlash2 candidate selector exists only in the V2 speculator. On
        # V1 the same checkpoint drafts through DFlashProposer, which never
        # calls it, so the draft would degrade to DFlash1 silently.
        if self._is_dflash2_draft():
            unsupported.append("dflash2 drafts")

        if self.model_config is not None and self.model_config.is_diffusion:
            unsupported.append("diffusion models")

        if self.parallel_config.enable_batch_sharded_sampling:
            unsupported.append("batch-sharded sampling")

        return unsupported

    def _validate_adaptive_verification(self) -> None:
        spec_config = self.speculative_config
        if not spec_config or not spec_config.enable_adaptive_verification:
            return

        if self.lora_config is not None:
            # The per-token LoRA mapping is built from CPU placeholder boundaries,
            # while the trimmed batch's true boundaries are decided on the GPU.
            raise ValueError(
                "Adaptive verification is not currently compatible with LoRA"
            )

        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            # The draft budget divides by step costs profiled from captured
            # cudagraphs; eager execution captures none.
            raise ValueError(
                "Adaptive verification is not currently compatible with "
                "enforce_eager/cudagraph_mode=none"
            )

        if self.parallel_config.pipeline_parallel_size > 1:
            # Cost curves and confidences currently only exist on the last PP rank;
            # earlier ranks would diverge on the trimmed batch shape.
            # TODO: we should be able to support adaptive verification with PP by
            # broadcasting the cost curves and confidences to all ranks.
            raise ValueError(
                "Adaptive verification is not currently compatible "
                "with pipeline parallelism"
            )

    # [CN] batch-sharded sampling：把采样计算也按 TP 分片（每 rank 只算自己那段 logits 的采样）。
    #      优化目标是省掉采样前的 logits all-gather。
    #
    #      这里收集 blockers（而非遇到第一个就报错）的好处：一次性列出所有原因，
    #      避免用户改一个再报下一个。注意只有**显式开启**才校验——
    #      未指定时直接置 False（这里把 None 收敛成布尔，供下游无判空使用）。
    def _validate_batch_sharded_sampling(self) -> None:
        """Validate `enable_batch_sharded_sampling` against the rest of the config."""
        if not self.parallel_config.enable_batch_sharded_sampling:
            # Default to False if not set.
            self.parallel_config.enable_batch_sharded_sampling = False
            return

        blockers: list[str] = []
        tp_size = self.parallel_config.tensor_parallel_size

        if tp_size <= 1:
            blockers.append("tensor_parallel_size is 1, so there is nothing to shard")
        elif self.scheduler_config.max_num_seqs < tp_size:
            # Requests are assigned to ranks whole, so fewer slots than ranks
            # leaves some ranks without work in every step.
            blockers.append(
                f"max_num_seqs ({self.scheduler_config.max_num_seqs}) is below "
                f"tensor_parallel_size ({tp_size})"
            )

        if self.model_config is not None and self.model_config.max_logprobs < 0:
            # max_logprobs == -1 allows vocab-size logprob requests, which the
            # fixed-width logprobs gather cannot reasonably size for.
            blockers.append("max_logprobs is -1, allowing vocab-size logprob requests")

        if self.model_config is not None and self.model_config.return_sampling_mask:
            # gather_sampler_output() drops SamplingMaskTensors: masks come back None.
            blockers.append(
                "return_sampling_mask is set and the batch-sharded gather does "
                "not forward sampling masks"
            )

        if (
            self.speculative_config is not None
            and self.speculative_config.enable_adaptive_verification
        ):
            # Adaptive verification picks the per-request draft split on the GPU,
            # so cu_num_logits_np is only an upper bound, while the shard plan is
            # built from that CPU array. The two disagree once the budget binds.
            # TODO(TheEpicDolphin): Support adaptive verification with batch-sharded
            # sampling.
            blockers.append(
                "it does not yet work with adaptive verification, which decides "
                "the per-request logits counts on the GPU, where the CPU-side "
                "shard plan cannot see them"
            )

        if blockers:
            raise ValueError(
                "Batch-sharded sampling was explicitly enabled via "
                "the --enable-batch-sharded-sampling flag, but is not supported "
                "in this configuration for the following reason(s): "
                f"{'; '.join(blockers)}."
            )

    def _get_dbo_unsupported_features(self) -> list[str]:
        """Collect what the V2 model runner cannot combine with DBO.

        The V2 runner microbatches a plain decoder forward pass. Anything that
        slices or replays the batch differently (drafting, adapters, pipeline
        stages, context parallelism, encoders) is not handled yet.
        """
        # TODO: DBO with model runner V2 is under development.
        # It should be enabled with explicit VLLM_USE_V2_MODEL_RUNNER environ.
        # Remove it when stable.
        # [CN] 注意这个早退：DBO+V2 还在开发中，**必须显式设置** VLLM_USE_V2_MODEL_RUNNER
        #      才继续往下检查；未设置时无条件判为不支持（返回固定的一项）。
        #      即"默认不开放，需要用户主动声明我要用实验特性"。
        if envs.VLLM_USE_V2_MODEL_RUNNER is None:
            return ["dual batch overlap"]

        unsupported: list[str] = []
        model_config = self.model_config
        parallel_config = self.parallel_config

        if self.lora_config is not None:
            unsupported.append("dual batch overlap with LoRA")
        if self.speculative_config is not None:
            unsupported.append("dual batch overlap with speculative decoding")
        if parallel_config.pipeline_parallel_size > 1:
            unsupported.append("dual batch overlap with pipeline parallelism")
        if (
            parallel_config.decode_context_parallel_size > 1
            or parallel_config.prefill_context_parallel_size > 1
        ):
            unsupported.append("dual batch overlap with context parallelism")
        if model_config is not None and (
            model_config.is_multimodal_model or model_config.is_encoder_decoder
        ):
            unsupported.append("dual batch overlap with multimodal models")
        if model_config is not None and model_config.is_hybrid:
            unsupported.append("dual batch overlap with hybrid models")
        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            unsupported.append("dual batch overlap with CUDA graphs")
        if self.is_mm_encoder_only:
            unsupported.append("dual batch overlap with encoder only models")

        return unsupported

    def _validate_v2_model_runner(self) -> None:
        """Check for features not yet supported by the V2 model runner."""
        if not HAS_TRITON:
            raise ValueError("Model Runner V2 requires Triton.")

        unsupported = self._get_v2_model_runner_unsupported_features()
        if unsupported:
            raise ValueError(
                f"Model Runner V2 does not yet support: {', '.join(unsupported)}"
            )

    def _validate_v1_model_runner(self) -> None:
        unsupported = self._get_v1_model_runner_unsupported_features()
        if unsupported:
            raise ValueError(
                f"Model Runner V1 does not support: {', '.join(unsupported)}"
            )

    def adjust_dcp_kv_cache_interleave_size(
        self, kv_cache_config: "KVCacheConfig"
    ) -> None:
        """Normalize DCP interleave size against block_size for NIXL P/D.

        Called by each worker (via ensure_kv_transfer_initialized), once it knows its
        own final block_size via kv_cache_config.
        """
        dcp_size = self.parallel_config.decode_context_parallel_size
        if dcp_size <= 1:
            return
        if self.parallel_config.dcp_kv_cache_interleave_size > 1 and (
            self.parallel_config.cp_kv_cache_interleave_size
            != self.parallel_config.dcp_kv_cache_interleave_size
        ):
            self.parallel_config.cp_kv_cache_interleave_size = (
                self.parallel_config.dcp_kv_cache_interleave_size
            )
            logger.warning_once(
                "cp_kv_cache_interleave_size is overridden by dcp_kv_cache"
                "_interleave_size. And dcp-kv-cache-interleave-size will be "
                "deprecated when PCP is fully supported."
            )

        if self.kv_transfer_config is None or not self.kv_transfer_config.has_connector(
            "NixlConnector"
        ):
            return

        # Get the kernel block_size, but don't use resolve_kv_cache_block_size to avoid
        # scaling by dcp_size (we need the local block_size here).
        local_block_size = min(
            g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
        )
        if self.parallel_config.cp_kv_cache_interleave_size != local_block_size:
            interleave = self.parallel_config.cp_kv_cache_interleave_size
            self.parallel_config.cp_kv_cache_interleave_size = local_block_size
            logger.info_once(
                "When using PD disaggregation with DCP "
                "(decode_context_parallel_size=%d), "
                "cp_kv_cache_interleave_size is automatically adjusted "
                "from %d to block_size %d for block-level alignment.",
                dcp_size,
                interleave,
                local_block_size,
            )

    def validate_block_size(self) -> None:
        """Validate block_size against DCP and mamba constraints.

        Called after Platform.update_block_size_for_backend() has
        finalised block_size.
        """
        block_size = self.cache_config.block_size

        # Skip DCP interleave-size compatibility for NIXL P/D: the interleave
        # size is pinned to block_size by each worker.
        nixl_pd_active = (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.has_connector("NixlConnector")
        )
        if self.parallel_config.decode_context_parallel_size > 1 and not nixl_pd_active:
            assert (
                self.parallel_config.cp_kv_cache_interleave_size <= block_size
                and block_size % self.parallel_config.cp_kv_cache_interleave_size == 0
            ), (
                f"Block_size({block_size}) should be greater "
                "than or equal to and divisible by cp_kv_cache_interleave_size "
                f"({self.parallel_config.cp_kv_cache_interleave_size})."
            )
        # Mamba cache align-mode constraints
        if self.cache_config.mamba_cache_mode == "align":
            assert not self.scheduler_config.disable_chunked_mm_input, (
                "Chunked MM input is required because we need the flexibility "
                "to schedule a multiple of block_size tokens even if they are "
                "in the middle of a mm input"
            )

    # [CN] 下面几个是 pydantic 的 model_validator(mode="after")，与 __post_init__ 的区别：
    #      - __post_init__ 是 dataclass 钩子，只跑一次、可以任意改写字段
    #      - model_validator(mode="after") 是 pydantic 钩子，必须**返回 self**，
    #        且会在每次 pydantic 校验时跑（含反序列化场景）
    #      两者并存是因为 VllmConfig 用了 pydantic dataclass：既有 dataclass 的
    #      字段与默认值机制，又想借用 pydantic 的校验/序列化能力。
    #
    #      nvfp4 与 MLA 不兼容：MLA 的 latent 维度是 head_size 的特殊布局，
    #      普通 nvfp4 布局（head_size//2 + head_size//16）套不上；
    #      要用得选专门的 nvfp4_ds_mla 布局。
    @model_validator(mode="after")
    def validate_nvfp4_kv_cache_with_mla(self) -> "VllmConfig":
        if self.model_config is None:
            return self
        # The ds_mla layouts are MLA-only by construction; the plain nvfp4
        # layout (head_size//2 + head_size//16) does not apply to MLA.
        if (
            self.cache_config.cache_dtype.startswith("nvfp4")
            and not self.cache_config.cache_dtype.endswith("_ds_mla")
            and self.model_config.use_mla
        ):
            raise ValueError(
                "nvfp4 KV cache is not supported with MLA (Multi-head Latent "
                "Attention) backends. Please use a different --kv-cache-dtype "
                "(e.g., 'fp8', 'auto', or 'nvfp4_ds_mla' with a sparse MLA "
                "backend) for MLA models such as DeepSeek."
            )
        return self

    @model_validator(mode="after")
    def validate_mamba_block_size(self) -> "VllmConfig":
        if self.model_config is None:
            return self
        mamba_block_size_is_set = (
            self.cache_config.mamba_block_size is not None
            and self.cache_config.mamba_block_size != self.model_config.max_model_len
        )
        # [CN] mamba_block_size 只有在开启 prefix caching 时才有意义：
        #      Mamba 是状态递推模型，要能复用中间状态就必须按块对齐缓存，
        #      而这正是 prefix caching 提供的机制。没开的话设置它没有任何效果，直接报错。
        if mamba_block_size_is_set and not self.cache_config.enable_prefix_caching:
            raise ValueError(
                "--mamba-block-size can only be set with --enable-prefix-caching"
            )
        return self

    @model_validator(mode="after")
    def validate_mamba_cached_kernel(self) -> "VllmConfig":
        if not self.cache_config.use_replayssm:
            self.cache_config.use_kda_recoverssm = False
            return self
        self.cache_config.use_kda_recoverssm = self.num_speculative_tokens > 0

        if self.model_config is not None and not self.model_config.supports_replayssm:
            raise ValueError(
                "--use-replayssm is not supported for architecture "
                f"{self.model_config.architecture!r}"
            )
        if self.cache_config.use_kda_recoverssm:
            if self.model_config is not None and self.model_config.architecture not in (
                "KimiLinearForCausalLM",
                "KimiK3ForConditionalGeneration",
            ):
                raise ValueError("RecoverSSM is only supported for Kimi-K3 KDA")
            if self.mamba_config.enable_stochastic_rounding:
                raise ValueError(
                    "RecoverSSM supports bfloat16/float32 "
                    "SSM state caches, not --enable-mamba-cache-stochastic-"
                    "rounding, which requires an explicit float16 cache"
                )
            if self.cache_config.mamba_cache_mode not in ("none", "align"):
                raise ValueError(
                    "RecoverSSM supports only none and align Mamba cache modes"
                )
            if (
                self.cache_config.mamba_cache_mode == "align"
                and not self.use_v2_model_runner
            ):
                raise ValueError(
                    "RecoverSSM with align mode requires VLLM_USE_V2_MODEL_RUNNER=1"
                )
            if self.parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "RecoverSSM currently requires pipeline_parallel_size=1"
                )
            if self.mamba_config.backend != MambaBackendEnum.TRITON:
                raise ValueError("RecoverSSM requires --mamba-backend triton")
        elif self.cache_config.mamba_cache_mode == "all":
            raise ValueError(
                "--use-replayssm supports prefix caching only in align mode; "
                "pass --mamba-cache-mode align"
            )
        elif self.mamba_config.backend == MambaBackendEnum.FLASHINFER:
            if self.cache_config.mamba_cache_mode == "align":
                raise ValueError(
                    "FlashInfer ReplaySSM does not support "
                    "--mamba-cache-mode align yet; use none"
                )
        elif self.mamba_config.backend != MambaBackendEnum.TRITON:
            raise ValueError(
                "--use-replayssm requires --mamba-backend triton or flashinfer"
            )
        elif self.use_v2_model_runner:
            raise ValueError(
                "Triton ReplaySSM requires Model Runner V1; use "
                "--mamba-backend flashinfer or Model Runner V1"
            )
        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.is_kv_transfer_instance
        ):
            raise ValueError(
                "--use-replayssm is incompatible with KV connectors "
                "(P/D disaggregation, KV cache offload)"
            )
        return self


# [CN] ===== 模块级「当前配置」全局变量 =====
#      这是一个典型的"环境式上下文"（ambient context）：
#      模型初始化期间把 VllmConfig 挂到全局变量上，让**深层自定义算子**
#      不必层层透传配置就能读到（如 CustomOp 需要按 dtype/平台决定走哪条 kernel）。
#
#      代价与风险：
#      - 隐式依赖：读配置的代码看不出配置从哪来，测试/离线场景容易漏设置
#      - 非线程安全：多线程同时初始化不同模型会互相覆盖（所以它是栈式保存/恢复的）
#      因此 vLLM 只在「模型初始化」这个明确窗口内使用它，不扩散到推理主循环。
#
#      _current_prefix 用于多模型场景（如投机解码的 draft 模型）区分命名空间。
_current_vllm_config: VllmConfig | None = None
_current_prefix: str | None = None


# [CN] 上下文管理器：进入时设置、退出时**恢复上一个**配置（栈式）。
#      两个隐蔽但重要的细节：
#      1) 进出都要 get_cached_compilation_config.cache_clear() ——
#         否则上一份配置下的编译配置会被 lru_cache 缓存住，换模型后读到旧的
#      2) check_compile 会对比 compilation_counter.num_models_seen 前后变化，
#         没增加说明该模型**没有** @support_torch_compile 装饰器、
#         即"开了编译但模型不支持"，只能给 warning（不能报错，否则不支持编译的模型全挂）
@contextmanager
def set_current_vllm_config(
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    from vllm.compilation.counter import compilation_counter

    num_models_seen = compilation_counter.num_models_seen
    try:
        # Clear the compilation config cache when context changes.
        # This is needed since the old config may have been accessed
        # and cached before the new config is set.
        get_cached_compilation_config.cache_clear()

        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if check_compile:
            vllm_config.compilation_config.custom_op_log_check()

        if (
            check_compile
            and vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and compilation_counter.num_models_seen == num_models_seen
        ):
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                vllm_config.model_config.model,
            )
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_vllm_config()"""
    return get_current_vllm_config().compilation_config


# [CN] 与 get_current_vllm_config_or_none() 的区别：这里**明确报错**而不是返回 None。
#      因为绝大多数调用方拿到 None 也无法处理，早失败好过后面出诡异空指针。
#      错误信息里特别提示了两种常见触发场景（在上下文外调用 / 在 import 期实例化 CustomOp），
#      这是排障时的关键线索。
def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
            "For tests that directly test custom ops/modules, use the "
            "'default_vllm_config' pytest fixture from tests/conftest.py."
        )
    return _current_vllm_config


def get_current_vllm_config_or_none() -> VllmConfig | None:
    return _current_vllm_config


T = TypeVar("T")


def get_layers_from_vllm_config(
    vllm_config: VllmConfig,
    layer_type: type[T],
    layer_names: Iterable[str] | None = None,
) -> dict[str, T]:
    """
    Get layers from the vLLM config.

    Args:
        vllm_config: The vLLM config.
        layer_type: The type of the layer to get.
        layer_names: The names of the layers to get. If None, return all layers.
    """

    forward_context = vllm_config.compilation_config.static_forward_context
    if layer_names is None:
        layer_names = forward_context.keys()

    return {
        layer_name: layer
        for layer_name in layer_names
        if isinstance(layer := forward_context.get(layer_name), layer_type)
    }
