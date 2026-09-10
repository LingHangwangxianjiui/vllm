# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ============================================================
# [CN] 文件：vllm/__init__.py
# 职责：包的对外导出面。只暴露顶层 API，不含任何实现逻辑
# 位置：用户 `import vllm` 时第一个执行的文件
# 核心成员：MODULE_ATTRS（导出名 → 模块路径映射）、__getattr__（懒加载入口）
# 上游：用户代码
# 下游：engine / entrypoints / inputs / outputs / model_executor 等各子模块
# 关键概念：PEP 562 模块级 __getattr__ 懒加载、导入顺序约束
# 状态：☑ 通读  ☑ 注释完成  □ 已验证
# ============================================================
#
# 【本文件唯一的真问题：为什么 import vllm 不会立刻把 torch/CUDA 拉起来】
# 如果这里直接 `from .entrypoints.llm import LLM`，那么 `import vllm` 会连带加载
# torch、CUDA 驱动、模型代码等重量级依赖，实测要数秒甚至更久。
# 很多场景（比如只想读个版本号、只想用 SamplingParams 构造参数）并不需要这些。
# 解决办法见下方 MODULE_ATTRS + __getattr__ 的懒加载机制。
#
# 【两个必须遵守的导入顺序约束，改动本文件时极易踩坑】
# 1. version 必须最先导入：见下面原注释，某些定制化构建依赖这个前提。
# 2. env_override 必须在其它任何 vllm 子模块之前导入：
#    它负责把环境变量覆盖写进配置；如果晚于其它模块导入，
#    那些模块在导入时读到的就是未被覆盖的旧值，覆盖会静默失效。
#    这就是为什么它单独一行、且带 noqa: F401（看起来没用到，实际是为副作用而导入）。
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401

# [CN] 导出名 → "相对模块路径:属性名" 的映射表。
# 值是字符串而不是直接 import，就是为了把「导入动作」推迟到真正访问该属性时。
# 格式说明：".entrypoints.llm:LLM" 中的点开头表示相对于本包（vllm）解析。
#
# 注意这里指向的 LLMEngine / AsyncLLMEngine 是 vllm/engine/ 下的 7 行别名桩：
# V0 引擎已彻底移除，这两个名字现在只是转发到 vllm/v1/engine/ 下的真实实现。
# 读源码时不要停在 vllm/engine/llm_engine.py，那里面没有东西。
MODULE_ATTRS = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (
        ClassificationOutput,
        ClassificationRequestOutput,
        CompletionOutput,
        EmbeddingOutput,
        EmbeddingRequestOutput,
        PoolingOutput,
        PoolingRequestOutput,
        RequestOutput,
        ScoringOutput,
        ScoringRequestOutput,
    )
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.ray_utils import initialize_ray_cluster
else:

    # [CN] 模块级 __getattr__（PEP 562）：当访问本模块上不存在的属性时，Python 会调用它。
    # 于是 `vllm.LLM` 在第一次被访问时才去 import 对应模块，实现懒加载。
    #
    # 为什么要有 if TYPE_CHECKING / else 两个分支：
    # - TYPE_CHECKING 分支（上方）在类型检查时生效：直接 import，让 IDE 和 mypy
    #   能看到真实类型，从而获得补全和类型推断。这些 import 运行时不会执行。
    # - else 分支（这里）才是运行时真正走的路径：懒加载。
    # 两者导出名单必须保持一致，否则会出现「IDE 能补全但运行报 AttributeError」。
    #
    # 副作用：本模块上访问不存在的名字时，抛的是 AttributeError 而不是 ImportError，
    # 排查导入问题时需要注意这一点。
    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
