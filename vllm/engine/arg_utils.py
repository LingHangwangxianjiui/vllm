# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# 本文件职责：vLLM 的"参数中枢"。把三路来源（CLI 字符串 / 环境变量 / Python API
# kwargs）统一收敛进 EngineArgs，再校验、推导、装配出最终的 VllmConfig。
#
# 在系统链路中的位置（控制面，进程启动期执行一次）：
#   CLI  :   vllm serve / vllm bench -> FlexibleArgumentParser
#              -> EngineArgs.add_cli_args()   （注册参数）
#              -> EngineArgs.from_cli_args()  （Namespace -> EngineArgs）
#   API  :   LLM(...) / AsyncLLM(...) -> EngineArgs(**kwargs)
#   ENV  :   vllm.envs.* / 平台环境变量 -> 作为各 *Config 字段的默认值参与合并
#   三者汇合于 EngineArgs -> create_engine_config() -> VllmConfig
#            -> LLMEngine / EngineCore -> Worker -> ModelRunner（数据面）
#
# 核心内容速查：
#   - EngineArgs              : 数百个字段的扁平 dataclass，所有参数的总入口
#   - EngineArgs.add_cli_args : 由子配置类的类型注解反射生成 argparse 参数
#   - create_engine_config    : 装配 VllmConfig 的主流程（本文件最重要的函数）
#   - create_*_config         : 各子配置的工厂方法（Model/Load/Cache/...）
#   - _set_default_*_args     : 依赖模型与硬件的默认值推导，在装配中途调用
#   - _compute_kwargs / get_kwargs : 类型注解 -> argparse kwargs 的反射层
#   - AsyncEngineArgs         : 在 EngineArgs 上追加异步引擎专属参数
#
# 三套配置来源的优先级（由高到低）：
#   1) 显式传入：CLI 上真正写出的参数、Python API 显式传入的 kwargs
#   2) 环境变量：vllm.envs.* 以及各平台相关环境变量
#   3) 代码默认值：各 *Config dataclass 字段上的字面量默认值
#   合并机制：环境变量被"折叠"进 vllm/config/ 下各 *Config 的字段默认值里，
#   因此当 CLI 没有显式给出某参数时，argparse 拿到的 default 本身就是环境变量
#   的值 —— 环境变量天然比显式参数低一级。环境变量整体的合法性由
#   envs.validate_environ() 在 create_engine_config() 中统一校验。
#
# 阅读提示（本文件最容易踩的坑）：
#   - add_cli_args 中大量 {"default": None} 覆盖并不是"把默认值改成 None"，而是
#     拿 None 当"用户未指定"的哨兵，好让 _set_default_*_args 在拿到 ModelConfig /
#     ParallelConfig 之后按模型和硬件推导真实默认值（enable_prefix_caching、
#     enable_chunked_prefill、max_num_batched_tokens、max_num_seqs 都是如此）。
#     这也解释了为什么装配后期有多处 assert xxx is not None。
#   - create_engine_config() 里各子配置的装配顺序是依赖关系，不能随意调换：
#     ModelConfig 最先产出（后续所有推导都要用它）-> CacheConfig（依赖模型能力）
#     -> ParallelConfig（DP/EP/节点拓扑推导）-> SpeculativeConfig / SchedulerConfig
#     （依赖前三者）-> LoRA 等与投机解码做交叉校验 -> 各 override 覆盖 -> VllmConfig。
#   - "顶层扁平参数"与"嵌套 config"重复指定通常互斥（如 --attention-backend 与
#     --attention-config.backend），本文件统一用 ValueError 直接拒绝。
#   - 已废弃（deprecated）字段的处理不在这个文件里做：旧参数要么在 vllm/config/
#     下各 dataclass 的 __post_init__ 中被迁移到新字段，要么直接不再注册为 CLI
#     参数（例如多步调度的相关参数已被移除）。本文件只保留一个显式开关
#     allow_deprecated_quantization，用于放行已废弃的量化方法。
# =============================================================================

import argparse
import copy
import dataclasses
import functools
import json
import os
import sys
from collections.abc import Callable
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from itertools import permutations
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import huggingface_hub
import regex as re
import torch
from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import TypeIs

import vllm.envs as envs
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    ConfigType,
    DeviceConfig,
    DiffusionConfig,
    ECTransferConfig,
    EncoderCacheManagerConfig,
    EPLBConfig,
    FaultToleranceConfig,
    KernelConfig,
    KVEventsConfig,
    KVTransferConfig,
    LoadConfig,
    LoRAConfig,
    MambaConfig,
    ModelConfig,
    MultiModalConfig,
    ObservabilityConfig,
    OffloadConfig,
    ParallelConfig,
    PoolerConfig,
    PrefetchOffloadConfig,
    ProfilerConfig,
    ReasoningConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
    UVAOffloadConfig,
    VllmConfig,
    WeightTransferConfig,
    get_attr_docs,
)
from vllm.config.cache import (
    CacheDType,
    KVOffloadingBackend,
    MambaCacheMode,
    MambaDType,
    PrefixCachingHashAlgo,
)
from vllm.config.device import Device
from vllm.config.kernel import IrOpPriorityConfig, LinearBackend, MoEBackend
from vllm.config.load import SafetensorsLoadStrategy
from vllm.config.lora import MaxLoRARanks
from vllm.config.mamba import MambaBackendEnum, MambaSSUAlgorithm
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    LogprobsMode,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from vllm.config.multimodal import (
    MMCacheType,
    MMEncoderTPMode,
    MMHasherAlgorithm,
    MMProcessorDevice,
    MMTensorIPC,
)
from vllm.config.observability import DetailedTraceModules
from vllm.config.parallel import (
    All2AllBackend,
    DataParallelBackend,
    DCPCommBackend,
    DistributedExecutorBackend,
    ExpertPlacementStrategy,
)
from vllm.config.scheduler import SchedulerPolicy
from vllm.config.utils import get_field
from vllm.config.vllm import OptimizationLevel, PerformanceMode
from vllm.logger import init_logger, suppress_logging
from vllm.platforms import CpuArchEnum, current_platform
from vllm.plugins import load_general_plugins
from vllm.ray.lazy_utils import is_in_ray_actor, is_ray_initialized
from vllm.transformers_utils.config import maybe_override_with_speculators
from vllm.transformers_utils.repo_utils import get_model_path
from vllm.transformers_utils.utils import is_cloud_storage
from vllm.utils.argparse_utils import (
    FlexibleArgumentParser,
    human_readable_int,
    human_readable_int_or_auto,
)
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.version import __version__ as VLLM_VERSION

if TYPE_CHECKING:
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.model_loader import LoadFormats
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.executor import Executor
else:
    Executor = Any
    QuantizationMethods = str
    LoadFormats = str
    UsageContext = Any


logger = init_logger(__name__)

# object is used to allow for special typing forms
T = TypeVar("T")
# 类型注解既可能是普通 class，也可能是 Literal / Union / Annotated 这类"特殊
# 类型形式"（typing 对象，不是 type 的实例），所以放宽成 object 来兼容两者
TypeHint: TypeAlias = type[Any] | object
TypeHintT: TypeAlias = type[T] | object


def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:
    """把普通的字符串转换函数包装成 argparse 可用的 type= 回调。

    argparse 要求 type 回调失败时抛 ArgumentTypeError；直接抛 ValueError 会被
    当成未捕获异常并吐出一长串 traceback，所以这里做一次异常类型转换。
    """
    def _parse_type(val: str) -> T:
        try:
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to {return_type}."
            ) from e

    return _parse_type


def optional_type(return_type: Callable[[str], T]) -> Callable[[str], T | None]:
    """让 argparse 参数支持"显式置空"。

    空串与字面量 "None" 都解析为 Python None，这样用户才能在命令行上覆盖掉一个
    默认非 None 的值（例如把 kv_cache_memory_bytes 重新置回"不限制"）。
    """
    def _optional_type(val: str) -> T | None:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)

    return _optional_type


def union_dict_and_str(val: str) -> str | dict[str, str] | None:
    """解析 `str | dict` 联合类型：整体形如 JSON 对象就当 dict，否则当普通字符串。

    这样同一个 CLI 参数既能接受"名字 / 路径"这类字符串，也能接受内联 JSON 配置。
    """
    if not re.match(r"(?s)^\s*{.*}\s*$", val):
        return str(val)
    return optional_type(json.loads)(val)


# 下面三个是类型注解集合上的基础谓词/取值工具。之所以要用集合而不是单个类型，
# 是因为一个字段的注解可能是 Optional[Literal[...]] 这种嵌套形式，展开后是多个
# 候选类型的集合。is_type 同时兼容 `int` 和 `list[int]`（后者靠 get_origin）。


def is_type(type_hint: TypeHint, type: TypeHintT) -> TypeIs[TypeHintT]:
    """Check if the type hint is a specific type."""
    return type_hint is type or get_origin(type_hint) is type


def contains_type(type_hints: set[TypeHint], type: TypeHintT) -> bool:
    """Check if the type hints contain a specific type."""
    return any(is_type(type_hint, type) for type_hint in type_hints)


def get_type(type_hints: set[TypeHint], type: TypeHintT) -> TypeHintT:
    """Get the specific type from the type hints."""
    return next((th for th in type_hints if is_type(th, type)), None)


# Literal 注解在 CLI 上表现为 choices（枚举值）。但如果注解里同时含 str，说明
# 还允许传入任意字符串（典型如后端名/插件名），此时只能用 metavar 提示而不能限制。
def literal_to_kwargs(type_hints: set[TypeHint]) -> dict[str, Any]:
    """Get the `type` and `choices` from a `Literal` type hint in `type_hints`.

    If `type_hints` also contains `str`, we use `metavar` instead of `choices`.
    """
    type_hint = get_type(type_hints, Literal)
    options = get_args(type_hint)
    option_type = type(options[0])
    if not all(isinstance(option, option_type) for option in options):
        raise ValueError(
            "All options must be of the same type. "
            f"Got {options} with types {[type(c) for c in options]}"
        )
    kwarg = "metavar" if contains_type(type_hints, str) else "choices"
    return {"type": option_type, kwarg: sorted(options)}


def collection_to_kwargs(type_hints: set[TypeHint], type: TypeHint) -> dict[str, Any]:
    """把 list/set/tuple 注解转成 argparse 的 type + nargs。

    tuple[int, int] 这种定长写法会解析成固定 nargs=2；tuple[int, ...] 与
    list/set 则用 nargs="+" 表示"一个或多个"。
    """
    type_hint = get_type(type_hints, type)
    types = get_args(type_hint)
    elem_type = types[0]

    # Handle Ellipsis
    assert all(t is elem_type for t in types if t is not Ellipsis), (
        f"All non-Ellipsis elements must be of the same type. Got {types}."
    )

    # Handle Union types
    if get_origin(elem_type) in {Union, UnionType}:
        # Union for Union[X, Y] and UnionType for X | Y
        assert str in get_args(elem_type), (
            "If element can have multiple types, one must be 'str' "
            f"(i.e. 'list[int | str]'). Got {elem_type}."
        )
        elem_type = str

    return {
        "type": elem_type,
        "nargs": "+" if type is not tuple or Ellipsis in types else len(types),
    }


# 非 builtins 的类型（例如 vllm 自己的枚举、平台类）在 CLI 上一律按字符串接收，
# 真正的解析交给下游的 pydantic / 枚举构造函数。
def is_not_builtin(type_hint: TypeHint) -> bool:
    """Check if the class is not a built-in type."""
    return type_hint.__module__ != "builtins"


def get_type_hints(type_hint: TypeHint) -> set[TypeHint]:
    """Extract type hints from Annotated or Union type hints."""
    # 把 Optional[X] / X | Y / Annotated[X, ...] 递归"摊平"成候选类型的集合，
    # 供上面的 contains_type / get_type 做判定。
    type_hints: set[TypeHint] = set()
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Annotated:
        type_hints.update(get_type_hints(args[0]))
    elif origin in {Union, UnionType}:
        # Union for Union[X, Y] and UnionType for X | Y
        for arg in args:
            type_hints.update(get_type_hints(arg))
    else:
        type_hints.add(type_hint)

    return type_hints


# 只有在真正需要生成帮助文本时才去解析字段注释（get_attr_docs 要读源码，很慢），
# 因此用一个模块级开关提前判断：命令行里带 --help，或者正在跑 mkdocs 生成文档。
# 注意这是 import 期就固定的常量，运行期再改 sys.argv 不会影响它。
NEEDS_HELP = (
    any("--help" in arg for arg in sys.argv)  # vllm SUBCOMMAND --help
    or "mkdocs" in sys.modules  # mkdocs SUBCOMMAND
)


def _maybe_add_docs_url(cls: Any) -> str:
    """Generate API docs URL for a vllm config class."""
    # 只为 vllm.config 顶层导出的配置类加链接，插件自带的类会被跳过（返回空串）
    import vllm.config

    name = cls.__name__
    if getattr(vllm.config, name, None) is not cls:
        return ""
    version = f"v{VLLM_VERSION}" if "dev" not in VLLM_VERSION else "latest"
    return f"\n\nAPI docs: https://docs.vllm.ai/en/{version}/api/vllm/config/#vllm.config.{name}"


def _expand_json_human_readable_numbers(val: str) -> str:
    """Expand human-readable number suffixes in a JSON string.

    Based on :func:`human_readable_int` so that the ``k/m/g/t`` (decimal) and
    ``K/M/G/T`` (binary) conventions work out the box.
    Also works inside JSON config arguments such
    as ``--kv-transfer-config '{"cpu_bytes_to_use": 80m}'``.

    Only bare (unquoted) tokens are replaced so that JSON string values
    like ``"model_name"`` are never modified.

    中文补充：这是为了让 `--kv-transfer-config '{"cpu_bytes_to_use": 80m}'` 这类
    内联 JSON 也能用人类可读后缀。按引号切分后只处理偶数段（引号外的区域），
    避免把字符串值里的 "80m" 之类的内容误改掉。
    """
    # Split on quoted strings so we only touch non-string regions.
    parts = re.split(r'("(?:[^"\\]|\\.)*")', val)
    for i in range(0, len(parts), 2):  # even indices = outside strings
        parts[i] = re.sub(
            r"\b\d+(?:\.\d+)?[kKmMgGtT]\b",
            lambda m: str(human_readable_int(m.group())),
            parts[i],
        )
    return "".join(parts)


@functools.lru_cache(maxsize=30)
def _compute_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    """反射一个配置 dataclass，生成 {字段名: argparse 关键字参数} 映射。

    这是"一处定义驱动 CLI"的关键：CLI 参数不手写，而是从 *Config 字段的**类型注解**
    推导 type/choices/nargs/action，从字段注释推导 help 文本。新增一个配置字段，
    CLI 参数就自动生成。

    Args:
        cls: vllm 的配置 dataclass，如 ModelConfig / CacheConfig / SchedulerConfig

    Returns:
        {field_name: {"default": ..., "help": ..., "type"/"action"/...: ...}}

    Note:
        结果被 lru_cache 缓存；调用方应通过 get_kwargs() 拿深拷贝后再修改，
        否则对返回值的改动会污染缓存、影响后续所有调用者。
    """
    # Save time only getting attr docs if we're generating help text
    cls_docs = get_attr_docs(cls) if NEEDS_HELP else {}
    kwargs = {}
    for field in fields(cls):
        # Get the set of possible types for the field
        type_hints: set[TypeHint] = get_type_hints(field.type)

        # If the field is a dataclass, we can use the model_validate_json
        generator = (th for th in type_hints if is_dataclass(th))
        dataclass_cls = next(generator, None)

        # Get the default value of the field
        if field.default is not MISSING:
            default = field.default
            # Handle pydantic.Field defaults
            if isinstance(default, FieldInfo):
                if default.default_factory is None:
                    default = default.default
                else:
                    # VllmConfig's Fields have default_factory set to config classes.
                    # These could emit logs on init, which would be confusing.
                    with suppress_logging():
                        default = default.default_factory()  # type: ignore[call-arg]
        elif field.default_factory is not MISSING:
            default = field.default_factory()

        # Get the help text for the field
        name = field.name
        help = cls_docs.get(name, "").strip()
        # Escape % for argparse
        help = help.replace("%", "%%")

        # Initialise the kwargs dictionary for the field
        kwargs[name] = {"default": default, "help": help}

        # Set other kwargs based on the type hints
        json_tip = (
            "Should either be a valid JSON string or JSON keys passed individually."
        )
        # 下面是一条"按类型注解分派"的判定链，顺序有讲究：越特殊的越靠前。
        # dataclass(整块 JSON) > Optional[str|bool] > bool > Literal > tuple/list/set
        # > int > float > dict > str，兜底则报错。放错顺序会导致注解被误判。
        if dataclass_cls is not None:

            def parse_dataclass(val: str, cls=dataclass_cls) -> Any:
                try:
                    val = _expand_json_human_readable_numbers(val)
                    return TypeAdapter(cls).validate_json(val)
                except ValidationError as e:
                    raise argparse.ArgumentTypeError(repr(e)) from e

            kwargs[name]["type"] = parse_dataclass
            kwargs[name]["help"] += _maybe_add_docs_url(dataclass_cls)
            kwargs[name]["help"] += f"\n\n{json_tip}"
        elif type_hints == {bool, str, type(None)}:
            # Optional-valued flag: bare flag -> True, value -> str.
            kwargs[name]["type"] = str
            kwargs[name]["nargs"] = "?"
            kwargs[name]["const"] = True
        elif contains_type(type_hints, bool):
            # Creates --no-<name> and --<name> flags
            kwargs[name]["action"] = argparse.BooleanOptionalAction
        elif contains_type(type_hints, Literal):
            kwargs[name].update(literal_to_kwargs(type_hints))
        elif contains_type(type_hints, tuple):
            kwargs[name].update(collection_to_kwargs(type_hints, tuple))
        elif contains_type(type_hints, list):
            kwargs[name].update(collection_to_kwargs(type_hints, list))
        elif contains_type(type_hints, set):
            kwargs[name].update(collection_to_kwargs(type_hints, set))
        elif contains_type(type_hints, int):
            # Arguments that accept human-readable integer strings (e.g., 1K, 2M, 1G)
            # 这几个字段的量级天然很大（token 数、显存字节），允许写成 8K / 2G
            human_readable_int_args = {
                "max_num_batched_tokens",
                "max_num_scheduled_tokens",
                "kv_cache_memory_bytes",
                "safetensors_prefetch_block_size",
                "max_num_queued_tokens",
            }
            if name == "max_model_len":
                kwargs[name]["type"] = human_readable_int_or_auto
                kwargs[name]["help"] += f"\n\n{human_readable_int_or_auto.__doc__}"
            elif name in human_readable_int_args:
                kwargs[name]["type"] = human_readable_int
                kwargs[name]["help"] += f"\n\n{human_readable_int.__doc__}"
            else:
                kwargs[name]["type"] = int
        elif contains_type(type_hints, float):
            kwargs[name]["type"] = float
        elif contains_type(type_hints, dict) and (
            contains_type(type_hints, str)
            or any(is_not_builtin(th) for th in type_hints)
        ):
            kwargs[name]["type"] = union_dict_and_str
        elif contains_type(type_hints, dict):
            kwargs[name]["type"] = parse_type(json.loads)
            kwargs[name]["help"] += f"\n\n{json_tip}"
        elif contains_type(type_hints, str) or any(
            is_not_builtin(th) for th in type_hints
        ):
            kwargs[name]["type"] = str
        else:
            raise ValueError(f"Unsupported type {type_hints} for argument {name}.")

        # If the type hint was a sequence of literals, use the helper function
        # to update the type and choices
        if get_origin(kwargs[name].get("type")) is Literal:
            kwargs[name].update(literal_to_kwargs({kwargs[name]["type"]}))

        # If None is in type_hints, make the argument optional.
        # But not if it's a bool, argparse will handle this better.
        # 注解里含 None（即 Optional[...]）时允许显式传 None，但 bool 除外——
        # bool 由 BooleanOptionalAction 生成的 --x / --no-x 已经足够表达三态。
        if type(None) in type_hints and not contains_type(type_hints, bool):
            kwargs[name]["type"] = optional_type(kwargs[name]["type"])
            if kwargs[name].get("choices"):
                kwargs[name]["choices"].append("None")
    return kwargs


# get_kwargs 是 _compute_kwargs 对外的唯一入口：深拷贝一份缓存结果再返回，
# 因为调用方（add_cli_args 等）经常要就地改 default / choices。
def get_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    """Return argparse kwargs for the given Config dataclass.

    If `--help` or `mkdocs` are not present in the command line command, the
    attribute documentation will not be included in the help output.

    The heavy computation is cached via functools.lru_cache, and a deep copy
    is returned so callers can mutate the dictionary without affecting the
    cached version.
    """
    return copy.deepcopy(_compute_kwargs(cls))


# -----------------------------------------------------------------------------
# EngineArgs：所有配置来源汇合后的"扁平总入口"。
#
# 为什么是扁平的？因为 CLI 只认一层 --xxx 参数。这里把十几个子配置（ModelConfig、
# CacheConfig、ParallelConfig、SchedulerConfig ...）的字段全部提升为同级字段，
# 再由 create_engine_config() 按依赖顺序重新分组装配回去。
#
# 字段默认值的来源（见文件头的优先级说明）：绝大多数写成形如
#   `max_model_len: int = ModelConfig.max_model_len`
# 即"以对应子配置类的字段默认值为默认值"，而子配置的默认值本身可能来自环境变量，
# 因此环境变量被自动折叠进了这里。少数以 `= None` 出现的字段（block_size、
# enable_prefix_caching、max_num_batched_tokens、max_num_seqs 等）是"用户未指定"
# 的哨兵，真实值要等拿到 ModelConfig / 硬件信息后在 _set_default_*_args 里推导。
# -----------------------------------------------------------------------------
@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""

    # ---------------- 模型与 tokenizer（对应 ModelConfig / LoadConfig） ----------------
    # [CN] model：HF Hub 上的 repo id，或本地模型目录路径。整个 vLLM 的入口参数。
    model: str = ModelConfig.model
    # [CN] MoE 专用：把每层实际路由到的专家 id 一并返回（用于分析路由分布/负载均衡）。
    enable_return_routed_experts: bool = ModelConfig.enable_return_routed_experts
    # [CN] 返回采样时实际生效的 mask（如结构化输出的 logits mask），调试用。
    return_sampling_mask: bool = ModelConfig.return_sampling_mask
    # [CN] 权重路径可与 model（配置路径）分离，用于"配置取自 A、权重来自 B"的场景。
    model_weights: str = ModelConfig.model_weights
    # [CN] 对外暴露的模型名（OpenAI API 请求里的 `model` 字段），可给多个别名。
    #      它与 model 无关——请求中填的名字要匹配这里，而不是 HF repo id。
    served_model_name: str | list[str] | None = ModelConfig.served_model_name
    # [CN] None 表示与 model 同路径；显式指定用于"权重与 tokenizer 不同源"。
    tokenizer: str | None = ModelConfig.tokenizer
    # [CN] None 同 model；用于本地权重目录缺少 config.json 时另行指定配置来源。
    hf_config_path: str | None = ModelConfig.hf_config_path
    # [CN] runner 决定引擎"干什么"：generate(生成) / pooling(嵌入、分类) / diffusion。
    #      auto 时由模型的 HF config 推断；选错会导致后续所有接口报类型不匹配。
    runner: RunnerOption = ModelConfig.runner
    # [CN] 加载时对权重做在线格式转换（如把某些 HF 量化格式转成 vLLM 内部格式）。
    convert: ConvertOption = ModelConfig.convert
    # [CN] 完全不加载 tokenizer：输入必须是已 tokenize 的 id 列表，可省启动时间与内存。
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    # [CN] 允许直接用 embedding 向量当 prompt（跳过 tokenize + embed），用于 RLHF 等场景。
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds
    # [CN] tokenizer_mode 可为 "auto"/"slow"/"mistral" 等内置值，也允许填自定义插件名。
    tokenizer_mode: TokenizerMode | str = ModelConfig.tokenizer_mode
    # [CN] 执行 HF 仓库中的自定义 modeling 代码——存在远程代码执行风险，仅在信任来源时开启。
    trust_remote_code: bool = ModelConfig.trust_remote_code
    # [CN] 多模态安全边界：允许读取本地媒体文件的目录白名单。空串表示不限制。
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path
    # [CN] 允许拉取远程媒体的域名白名单；None 表示不限制。
    #      线上服务务必收紧这两项，否则可能被诱导读取/请求任意本地文件与内网地址（SSRF）。
    allowed_media_domains: list[str] | None = ModelConfig.allowed_media_domains
    # [CN] 权重下载缓存目录；None 用 HF 默认缓存。离线环境常配合 HF_HUB_OFFLINE 使用。
    download_dir: str | None = LoadConfig.download_dir
    # [CN] safetensors 的加载/预取策略；None 表示自动选择。
    safetensors_load_strategy: SafetensorsLoadStrategy | None = (
        LoadConfig.safetensors_load_strategy
    )
    # [CN] 预取线程数与块大小，只在使用 safetensors 预取（本地盘/网络盘）时生效。
    #      调大能加快大模型加载，代价是额外的内存与 IO 占用。
    safetensors_prefetch_num_threads: int = LoadConfig.safetensors_prefetch_num_threads
    safetensors_prefetch_block_size: int = LoadConfig.safetensors_prefetch_block_size
    # [CN] 权重加载格式：auto/safetensors/pt/binfile/...，也可传自定义插件名。
    load_format: str | LoadFormats = LoadConfig.load_format
    # [CN] config 的读取格式（auto/hf/mistral 等）。与 load_format 是两件事，别混淆。
    config_format: str = ModelConfig.config_format
    # [CN] 模型计算 dtype：auto 时按 HF config 的 torch_dtype 推断；
    #      也可直接指定（含 fp8/fp4 等量化 dtype）。注意它与 quantization 是不同维度。
    dtype: ModelDType = ModelConfig.dtype
    # [CN] 字段名不对齐：EngineArgs 侧叫 kv_cache_dtype，CacheConfig 侧叫 cache_dtype。
    #      它只决定 KV cache 的存储精度（auto/fp8/fp8_e5m2/...），不影响计算 dtype。
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    # [CN] 全局随机种子，影响采样与部分随机行为；做可复现压测时需要固定它。
    seed: int = ModelConfig.seed
    # [CN] 单条请求的最大上下文长度（prompt + output）。None 时取模型 config 的
    #      max_position_embeddings，随后仍可能被显存上限裁剪（见 _set_default_args）。
    max_model_len: int = ModelConfig.max_model_len
    # [CN] cudagraph 捕获的 batch size 档位。两者的协作方式：
    #      显式给出 cudagraph_capture_sizes 时按它来；否则由 max_cudagraph_capture_size
    #      配合编译配置自动生成一组档位。档位越多显存占用越大，但 padding 浪费越少。
    cudagraph_capture_sizes: list[int] | None = (
        CompilationConfig.cudagraph_capture_sizes
    )
    max_cudagraph_capture_size: int | None = get_field(
        CompilationConfig, "max_cudagraph_capture_size"
    )
    # [CN] IR 层算子优先级配置（Inductor 选择算子实现时的偏好），属于 KernelConfig。
    ir_op_priority: IrOpPriorityConfig = get_field(KernelConfig, "ir_op_priority")
    # ---------------- 并行与分布式（ParallelConfig / KernelConfig） ----------------
    #
    # [CN] vLLM 的并行有五个正交维度，弄清它们的区别是读懂 executor / worker 的前提：
    #   TP  tensor_parallel_size             —— 层内切分（注意力头、MLP 列切），通信量最大
    #   PP  pipeline_parallel_size           —— 层间切分，按 stage 流水
    #   CP  prefill/decode_context_parallel  —— 沿序列长度切分，服务长上下文
    #   DP  data_parallel_size               —— 整份模型复制多份，各自处理不同请求
    #   EP  enable_expert_parallel           —— MoE 专家分散到各卡（替代"每卡都放全部专家"）
    #   world_size = TP * PP * CP * DP，决定了要拉起多少个 worker 进程。
    #
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # [CN] 执行器后端：uni(单进程) / mp(多进程) / ray / external_launcher，也可传 Executor 子类。
    #      None 时按 world_size 与环境自动选：单卡用 uni，多卡优先 mp。
    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = ParallelConfig.distributed_executor_backend
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    # [CN] master_addr / master_port / nnodes / node_rank 只有在使用 external_launcher
    #      （slurm、torchrun 等由外部拉起进程的场景）时才需要显式设置；
    #      uni/mp/ray 后端由 vLLM 自己拉起进程，这四项实际不起作用。
    master_addr: str = ParallelConfig.master_addr
    master_port: int = ParallelConfig.master_port
    nnodes: int = ParallelConfig.nnodes
    node_rank: int = ParallelConfig.node_rank
    distributed_timeout_seconds: int | None = ParallelConfig.distributed_timeout_seconds
    cpu_distributed_timeout_seconds: int | None = (
        ParallelConfig.cpu_distributed_timeout_seconds
    )
    # [CN] NUMA 绑定：多路 CPU 服务器上把 worker 绑到指定 NUMA 节点/CPU，避免跨片访存。
    numa_bind: bool = ParallelConfig.numa_bind
    numa_bind_nodes: list[int] | None = ParallelConfig.numa_bind_nodes
    numa_bind_cpus: list[str] | None = ParallelConfig.numa_bind_cpus
    # [CN] 限制可见设备。注意这里是硬编码 None（并非取自 ParallelConfig）：
    #      为 None 表示不过滤，实际可见设备由 CUDA_VISIBLE_DEVICES 等环境变量决定。
    device_ids: list[int | str] | None = None
    # [CN] TP：把单层权重切开。要求能整除 num_attention_heads、num_key_value_heads
    #      与 intermediate_size，否则会在模型加载阶段直接报错。
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    # [CN] CP（上下文并行）拆成 prefill / decode 两档，因为两阶段的瓶颈不同，
    #      允许只对其一开启上下文并行。
    prefill_context_parallel_size: int = ParallelConfig.prefill_context_parallel_size
    decode_context_parallel_size: int = ParallelConfig.decode_context_parallel_size
    # [CN] DCP（decode context parallel）相关：通信后端、Q 是否各卡复制一份、
    #      以及 KV cache 的分片交织粒度——交织粒度直接影响负载均衡与访存连续性。
    dcp_comm_backend: DCPCommBackend | None = ParallelConfig.dcp_comm_backend
    dcp_q_replicate: bool | None = ParallelConfig.dcp_q_replicate
    dcp_kv_cache_interleave_size: int = ParallelConfig.dcp_kv_cache_interleave_size
    cp_kv_cache_interleave_size: int = ParallelConfig.cp_kv_cache_interleave_size
    # [CN] DP：复制多份完整模型，各自独立处理请求。它是提升吞吐最直接的手段，
    #      并且不会拉低单请求时延（与 TP 相反，TP 降时延但也引入通信开销）。
    data_parallel_size: int = ParallelConfig.data_parallel_size
    # [CN] 下面几个 None 是"由启动器或运行期推导"的哨兵：
    #      rank / start_rank / size_local / address / rpc_port 在单节点部署时由 vLLM 自动分配，
    #      只有跨节点部署、或需要外部负载均衡器接入时才需要手动指定。
    data_parallel_rank: int | None = None
    data_parallel_start_rank: int | None = None
    data_parallel_size_local: int | None = None
    data_parallel_address: str | None = None
    data_parallel_rpc_port: int | None = None
    # [CN] 三种负载均衡模式，语义各不相同：
    #      data_parallel_hybrid_lb             —— 内部混合调度（含 P/D 分离式负载分发）
    #      data_parallel_external_lb           —— 由外部网关分发，各 DP rank 各自接请求
    #      data_parallel_multi_port_external_lb —— external_lb 的变体，每个 rank 独占一个端口
    data_parallel_hybrid_lb: bool = False
    data_parallel_external_lb: bool = False
    data_parallel_multi_port_external_lb: bool = False
    data_parallel_backend: DataParallelBackend = ParallelConfig.data_parallel_backend
    # [CN] EP（专家并行）：MoE 的专家分散到各张卡，attention 部分仍按 TP/DP 切。
    #      开启后专家计算依赖 all-to-all 通信，需要 all2all_backend 支持。
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    # [CN] 让采样在 batch 维度分片并与 TP 的分片对齐，避免每张卡都重复算全量 logits。
    enable_batch_sharded_sampling: bool | None = (
        ParallelConfig.enable_batch_sharded_sampling
    )
    # [CN] 加载权重时按 EP 切分过滤掉本卡用不到的专家，显著省显存与加载时间。
    enable_ep_weight_filter: bool = ParallelConfig.enable_ep_weight_filter
    # [CN] 算子后端选择（MoE / linear），属于 KernelConfig，可按硬件挑最优实现。
    moe_backend: MoEBackend = KernelConfig.moe_backend
    linear_backend: LinearBackend = KernelConfig.linear_backend
    all2all_backend: All2AllBackend = ParallelConfig.all2all_backend
    # [CN] 弹性 EP：允许运行期动态伸缩专家规模（配合 scale_elastic_ep 接口使用）。
    enable_elastic_ep: bool = ParallelConfig.enable_elastic_ep
    # [CN] DBO（Dual Batch Overlap）：把一个 batch 拆成两个 micro-batch，
    #      让一个的计算与另一个的通信重叠，从而把通信"藏"起来。
    #      ubatch_size 是 micro-batch 大小；两个 threshold 分别是 prefill / decode
    #      阶段启用 DBO 的最小 token 数——低于阈值时拆批的开销大于收益，反而更慢。
    enable_dbo: bool = ParallelConfig.enable_dbo
    ubatch_size: int = ParallelConfig.ubatch_size
    dbo_decode_token_threshold: int = ParallelConfig.dbo_decode_token_threshold
    # [CN] DP 各 rank 之间同步权重/状态的间隔（按 step 计）。
    dp_sync_interval: int = ParallelConfig.dp_sync_interval
    dbo_prefill_token_threshold: int = ParallelConfig.dbo_prefill_token_threshold
    # [CN] 不用 NCCL 而改用更轻的方式做 DP 同步（某些环境 NCCL 初始化极慢或不可用）。
    disable_nccl_for_dp_synchronization: bool | None = (
        ParallelConfig.disable_nccl_for_dp_synchronization
    )
    # [CN] EPLB（Expert Parallel Load Balancing）：MoE 各专家负载不均衡时做重排/冗余复制。
    eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")
    enable_eplb: bool = ParallelConfig.enable_eplb
    # [CN] 专家放置策略：如何在各卡之间分配专家，直接影响 all-to-all 流量与均衡度。
    expert_placement_strategy: ExpertPlacementStrategy = (
        ParallelConfig.expert_placement_strategy
    )
    # [CN] 带下划线前缀 = 内部字段，由 API server 进程启动 DP 时自动填入，
    #      用户不应手动设置，也不保证跨版本稳定。
    _api_process_count: int = ParallelConfig._api_process_count
    _api_process_rank: int = ParallelConfig._api_process_rank
    # [CN] 并行加载权重的 worker 数；None 时按模型规模与可用显存自动决定。
    max_parallel_loading_workers: int | None = (
        ParallelConfig.max_parallel_loading_workers
    )
    # ---------------- KV cache（CacheConfig） ----------------
    # 这几个 None 是"用户未指定"哨兵：block_size 由平台/attention backend 决定，
    # enable_prefix_caching 由模型是否支持决定，二者都在装配期推导。
    block_size: int | None = None
    enable_prefix_caching: bool | None = None
    prefix_caching_hash_algo: PrefixCachingHashAlgo = (
        CacheConfig.prefix_caching_hash_algo
    )
    prefix_cache_retention_interval: int | None = get_field(
        CacheConfig, "prefix_cache_retention_interval"
    )
    disable_sliding_window: bool = ModelConfig.disable_sliding_window
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn
    offload_backend: str = OffloadConfig.offload_backend
    cpu_offload_gb: float = UVAOffloadConfig.cpu_offload_gb
    cpu_offload_params: set[str] = get_field(UVAOffloadConfig, "cpu_offload_params")
    offload_group_size: int = PrefetchOffloadConfig.offload_group_size
    offload_num_in_group: int = PrefetchOffloadConfig.offload_num_in_group
    offload_prefetch_step: int = PrefetchOffloadConfig.offload_prefetch_step
    offload_params: set[str] = get_field(PrefetchOffloadConfig, "offload_params")
    # [CN] 显存预算有【两种互斥的指定方式】，理解区别很重要：
    #   gpu_memory_utilization —— 按比例（占整卡显存的比例），传统方式；
    #   kv_cache_memory_bytes  —— 按绝对字节数，直接指定 KV cache 可用容量。
    #   二者只能设其一，装配期校验。后者更精确：比例方式在"多进程共享同一张卡"
    #   或"卡上已有其他进程占用显存"时容易算出错误结果。
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    kv_cache_memory_bytes: int | None = CacheConfig.kv_cache_memory_bytes
    # [CN] 这组"并发上限"是全项目最容易混淆的一组参数，务必分清：
    #   max_num_batched_tokens   —— 单个 step 内参与 forward 的 token 总数上限。
    #                               chunked prefill 下它决定 prefill 被切成多大的块。
    #   max_num_scheduled_tokens —— 单条请求在单个 step 内最多被调度多少 token，
    #                               防止一条超长请求独占整个 step。
    #   max_num_seqs             —— 单个 step 内并发的序列条数上限（batch 的"行数"）。
    #   三者共同决定 scheduler 每步能塞多少活：调大提升吞吐，但也抬高显存与时延。
    max_num_batched_tokens: int | None = None
    max_num_scheduled_tokens: int | None = None
    # [CN] chunked prefill 下，"prompt 长度超过该值"的请求被视作长 prefill，
    #      会额外施加调度上限（0 表示关闭该判定）。
    long_prefill_token_threshold: int = SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: int | None = None
    # [CN] 准入控制（背压）：在途（waiting 或 running）请求数 / 提示 token 总数的上限。
    #      超限后新请求直接被拒（HTTP 503 让客户端重试别处），而不是无限排队——
    #      这是给 vLLM 原本无界请求队列加的一道粗粒度容量阀门。
    #      注意与 max_num_seqs 的区别：后者是"每 DP rank 每 step 的并发数"，
    #      而 max_num_queued_reqs 在 API server 进程里统计、跨所有 DP rank 生效，
    #      建议按 data_parallel_size * max_num_seqs + 期望排队深度 来设置。
    max_num_queued_reqs: int | None = None
    max_num_queued_tokens: int | None = None
    # [CN] 单次请求可返回的 logprobs 数量上限。这是安全阈值：
    #      不加限制时 n × vocab 的返回量会轻易打爆内存与带宽。
    max_logprobs: int = ModelConfig.max_logprobs
    # [CN] logprobs 的口径（是否包含温度、penalty 等变换前的值）。
    logprobs_mode: LogprobsMode = ModelConfig.logprobs_mode
    # [CN] 用 float64 计算 gumbel（无放回采样等场景需要），精度更高但更慢。
    use_fp64_gumbel: bool = ModelConfig.use_fp64_gumbel
    # [CN] trace 回放模式：按记录的请求时间线重放，用于可复现的压测。
    enable_trace_replay: bool = ModelConfig.enable_trace_replay
    disable_log_stats: bool = False
    # [CN] DP 多 rank 时，把各 rank 的统计日志聚合后统一输出（否则同一条日志刷 N 遍）。
    aggregate_engine_logging: bool = False
    # [CN] revision / code_revision / tokenizer_revision：HF 仓库的分支、tag 或 commit。
    #      三者可分别指定权重、自定义建模代码、tokenizer 的来源；生产环境建议钉死版本。
    revision: str | None = ModelConfig.revision
    code_revision: str | None = ModelConfig.code_revision
    # [CN] HF 访问令牌。bool 形式是语义开关：True=使用环境变量中的 token，False=不使用。
    hf_token: bool | str | None = ModelConfig.hf_token
    # [CN] 直接覆盖 HF config 中的字段（如 max_position_embeddings、num_hidden_layers），
    #      用于不改权重文件就调整结构。注意：覆盖后若与真实权重不匹配会直接加载失败。
    hf_overrides: HfOverrides = get_field(ModelConfig, "hf_overrides")
    # [CN] {模型架构名: 自定义实现类路径}——不改动注册表就能替换某个架构的实现。
    model_class_overrides: dict[str, str] = get_field(
        ModelConfig, "model_class_overrides"
    )
    tokenizer_revision: str | None = ModelConfig.tokenizer_revision
    # [CN] quantization 是简写名（"fp8"/"awq"/"gptq" ...）或自定义量化插件名；
    #      若需要更细粒度的控制，则改用下面的 quantization_config。
    quantization: QuantizationMethods | str | None = ModelConfig.quantization
    quantization_config: "dict[str, Any] | QuantizationConfigArgs | None" = None
    """User-facing quantization configuration. Carries per-layer-kind
    QuantSpecs (linear, moe) and ignore patterns; see
    :class:`QuantizationConfigArgs`. Auto-populated from the matching online
    shorthand when `quantization` is one of the values in
    `ONLINE_QUANT_SHORTHAND_NAMES`."""
    allow_deprecated_quantization: bool = ModelConfig.allow_deprecated_quantization
    enforce_eager: bool = ModelConfig.enforce_eager
    disable_custom_all_reduce: bool = ParallelConfig.disable_custom_all_reduce
    # ---------------- 多模态（MultiModalConfig） ----------------
    # [CN] 只跑语言模型部分（跳过视觉塔等多模态编码器），用于调试或复用已算好的 embedding。
    language_model_only: bool = MultiModalConfig.language_model_only
    # [CN] 注意改名：EngineArgs 侧叫 limit_mm_per_prompt，MultiModalConfig 侧叫 limit_per_prompt。
    #      形如 {"image": 2, "video": {"num_frames": 32}}——值可以是数量上限，也可以带子维度限制。
    limit_mm_per_prompt: dict[str, int | dict[str, int]] = get_field(
        MultiModalConfig, "limit_per_prompt"
    )
    enable_mm_embeds: bool = MultiModalConfig.enable_mm_embeds
    interleave_mm_strings: bool = MultiModalConfig.interleave_mm_strings
    media_io_kwargs: dict[str, dict[str, Any]] = get_field(
        MultiModalConfig, "media_io_kwargs"
    )
    # [CN] 透传给 HF processor 的额外 kwargs（如图像分辨率策略）。
    #      注意它参与多模态 hash 计算：改动它会使已缓存的预处理结果全部失效。
    mm_processor_kwargs: dict[str, Any] | None = MultiModalConfig.mm_processor_kwargs
    # [CN] 多模态预处理结果（图像/视频张量）的缓存，按 hash 复用以避免重复编解码。
    #      type 可选 "shm"（共享内存，可跨进程复用）或 "lru"（仅进程内）。
    mm_processor_cache_gb: float = MultiModalConfig.mm_processor_cache_gb
    mm_processor_cache_type: MMCacheType | None = (
        MultiModalConfig.mm_processor_cache_type
    )
    mm_hasher_algorithm: MMHasherAlgorithm = get_field(
        MultiModalConfig, "mm_hasher_algorithm"
    )
    mm_shm_cache_max_object_size_mb: int = (
        MultiModalConfig.mm_shm_cache_max_object_size_mb
    )
    # [CN] 只跑多模态编码器（输出 embedding 供外部复用），与 language_model_only 正好相对。
    mm_encoder_only: bool = MultiModalConfig.mm_encoder_only
    # [CN] 编码器是否参与 TP（"replicated" / "sharded"）。视觉塔通常远小于 LLM，
    #      复制一份比切分更划算（省掉 all-gather 通信），这就是默认策略的由来。
    mm_encoder_tp_mode: MMEncoderTPMode = MultiModalConfig.mm_encoder_tp_mode
    # [CN] 编码器（如 ViT）可用独立的 attention backend 与 dtype，
    #      因为它的算子特征与 LLM decode 完全不同（短 batch 长序列 vs 长 batch 短序列）。
    mm_encoder_attn_backend: AttentionBackendEnum | str | None = (
        MultiModalConfig.mm_encoder_attn_backend
    )
    mm_encoder_attn_dtype: str | None = MultiModalConfig.mm_encoder_attn_dtype
    mm_encoder_fp8_scale_path: str | None = MultiModalConfig.mm_encoder_fp8_scale_path
    mm_encoder_fp8_scale_save_path: str | None = (
        MultiModalConfig.mm_encoder_fp8_scale_save_path
    )
    mm_encoder_fp8_scale_save_margin: float = (
        MultiModalConfig.mm_encoder_fp8_scale_save_margin
    )
    # [CN] 自定义 IO 处理器插件（接管输入预处理 / 输出后处理），None 表示用内置实现。
    io_processor_plugin: str | None = None
    # [CN] 渲染（chat template 展开）使用的 worker 数。硬编码默认 1：
    #      多 worker 需要额外的进程池开销，只有在渲染成为瓶颈时才值得开大。
    renderer_num_workers: int = 1
    skip_mm_profiling: bool = MultiModalConfig.skip_mm_profiling
    video_pruning_rate: float | None = MultiModalConfig.video_pruning_rate
    video_pruning_method: str = MultiModalConfig.video_pruning_method
    mm_tensor_ipc: MMTensorIPC = MultiModalConfig.mm_tensor_ipc
    mm_processor_device: MMProcessorDevice = "auto"
    mm_ipc_gpu_memory_gb: float = MultiModalConfig.mm_ipc_gpu_memory_gb
    mm_device_do_normalize: bool | None = MultiModalConfig.mm_device_do_normalize
    # LoRA fields
    # LoRAConfig 是"可选子配置"：enable_lora=False 时 create_engine_config 会直接
    # 返回 None，下面这些字段即使被设置也不会生效。
    enable_lora: bool = False
    # [CN] 同时活跃的 LoRA 数量上限；超出的请求需排队等待，它直接决定 LoRA 的显存占用。
    max_loras: int = LoRAConfig.max_loras
    # [CN] 支持的 rank 档位——只能是枚举里的那几个值，不能任意填写，
    #      因为 punica 等 kernel 需要按固定 rank 预先分配 buffer / 选择实现。
    max_lora_rank: MaxLoRARanks = LoRAConfig.max_lora_rank
    default_mm_loras: dict[str, str] | None = LoRAConfig.default_mm_loras
    # [CN] LoRA 权重也按 TP 分片（省显存），代价是每次用之前要多一次 all-gather。
    fully_sharded_loras: bool = LoRAConfig.fully_sharded_loras
    # [CN] CPU 侧常驻的 LoRA 数量，为 None 时等于 max_loras——
    #      即"热 LoRA 全留在 CPU 内存、按需换入显存"，用内存换显存。
    max_cpu_loras: int | None = LoRAConfig.max_cpu_loras
    lora_dtype: str | torch.dtype | None = LoRAConfig.lora_dtype
    lora_target_modules: list[str] | None = LoRAConfig.target_modules
    enable_tower_connector_lora: bool = LoRAConfig.enable_tower_connector_lora
    specialize_active_lora: bool = LoRAConfig.specialize_active_lora
    enable_mixed_moe_lora_format: bool = LoRAConfig.enable_mixed_moe_lora_format
    enable_moe_shared_loras: bool = LoRAConfig.enable_moe_shared_loras

    ray_workers_use_nsight: bool = ParallelConfig.ray_workers_use_nsight
    # [CN] 手动指定 KV cache 块数，跳过启动时的显存 profiling 自动探测。
    #      用于多实例共享同一张卡、或 profiling 结果不稳定的场景；
    #      设错会直接 OOM（设大）或浪费显存（设小）。
    num_gpu_blocks_override: int | None = CacheConfig.num_gpu_blocks_override
    # [CN] 透传给具体 model loader 的额外配置（如 runai 流式加载的参数）。
    model_loader_extra_config: dict = get_field(LoadConfig, "model_loader_extra_config")
    # [CN] 加载权重时跳过的文件名模式（如 "*.pt"、"original/*"），能显著加快加载速度。
    ignore_patterns: str | list[str] = get_field(LoadConfig, "ignore_patterns")

    # ---------------- 调度（SchedulerConfig） ----------------
    # 同样是"未指定"哨兵，依赖模型能力与硬件在 _set_default_*_args 中推导
    # [CN] chunked prefill：把长 prompt 切成小块，与 decode 混在同一个 step 里跑，
    #      避免一条长 prefill 独占 GPU 导致其余请求卡顿——这是 vLLM 高吞吐的关键机制之一。
    #      None 表示"用户未指定"，装配期由模型能力与硬件自动决定（通常是开）。
    enable_chunked_prefill: bool | None = None
    # [CN] 关闭多模态输入的切分：多模态 prefill 切分需要额外处理视觉 token 的边界。
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input

    # [CN] 准入新请求时，检查的是"整条输入序列能否装进 KV cache"，而不是"第一个 chunk 能否装下"。
    #      这样能防止 chunked prefill 下的过度准入（over-admission）与 KV cache 反复颠簸。
    scheduler_reserve_full_isl: bool = SchedulerConfig.scheduler_reserve_full_isl
    # [CN] 仅在 data-parallel 部署下有意义：每 N 个 engine step 才准入一批新的 prefill
    #      请求，且各 DP rank 步调对齐，从而让每步 forward 的耗时更均衡（默认 1=每步都准入）。
    prefill_schedule_interval: int = SchedulerConfig.prefill_schedule_interval

    # [CN] 水位线：预留"总块数 × watermark"的空闲块作为余量。它只在把
    #      waiting / preempted 状态的请求准入 running 队列时生效（且已有请求被调度的前提下），
    #      用来减少显存吃紧时的频繁驱逐与反复抢占。取值 [0.0, 1.0)，0.0 表示关闭。
    watermark: float = SchedulerConfig.watermark

    disable_hybrid_kv_cache_manager: bool | None = (
        SchedulerConfig.disable_hybrid_kv_cache_manager
    )

    structured_outputs_config: StructuredOutputsConfig = get_field(
        VllmConfig, "structured_outputs_config"
    )
    reasoning_parser: str = StructuredOutputsConfig.reasoning_parser
    reasoning_parser_plugin: str | None = None

    # [CN] 投机解码有两种写法：
    #   老写法：speculative_config 直接给一整块 JSON（如 {"method":"eagle","model":...}）
    #   新写法：spec_method / spec_model / spec_tokens 三个扁平参数（CLI 上更好敲）
    #   二者最终都会被归一化成 SpeculativeConfig；同时给出时以扁平参数为准并会告警。
    speculative_config: dict[str, Any] | None = None
    spec_method: str | None = None
    spec_model: str | None = None
    spec_tokens: int | None = None
    # [CN] 扩散模型（runner="diffusion"）的配置，与 LLM 生成链路相互独立。
    diffusion_config: dict[str, Any] | None = None

    # ---------------- 可观测性（ObservabilityConfig） ----------------
    # [CN] 这些开关默认大多关闭，因为它们会带来可测量的运行时开销。
    # [CN] 兼容开关：让指标输出保持某个旧版本的行为（升级后监控不炸）。
    show_hidden_metrics_for_version: str | None = (
        ObservabilityConfig.show_hidden_metrics_for_version
    )
    # [CN] OTLP trace 上报地址（如 http://jaeger:4318/v1/traces）。设为 None 则不上报。
    otlp_traces_endpoint: str | None = ObservabilityConfig.otlp_traces_endpoint
    # [CN] 指定要为哪些模块采集细粒度 trace（如 model、worker、sampler）。
    #      粒度越细开销越大，只应在定位问题时临时开启。
    collect_detailed_traces: list[DetailedTraceModules] | None = (
        ObservabilityConfig.collect_detailed_traces
    )
    per_request_spec_decode_metrics: Literal["none", "summary", "detailed"] = (
        ObservabilityConfig.per_request_spec_decode_metrics
    )
    kv_cache_metrics: bool = ObservabilityConfig.kv_cache_metrics
    kv_cache_metrics_sample: float = get_field(
        ObservabilityConfig, "kv_cache_metrics_sample"
    )
    cudagraph_metrics: bool = ObservabilityConfig.cudagraph_metrics
    enable_layerwise_nvtx_tracing: bool = (
        ObservabilityConfig.enable_layerwise_nvtx_tracing
    )
    enable_mfu_metrics: bool = ObservabilityConfig.enable_mfu_metrics
    enable_logging_iteration_details: bool = (
        ObservabilityConfig.enable_logging_iteration_details
    )
    jit_monitor_mode: Literal["warn", "error"] = ObservabilityConfig.jit_monitor_mode
    jit_monitor_verbose: bool = ObservabilityConfig.jit_monitor_verbose
    enable_mm_processor_stats: bool = ObservabilityConfig.enable_mm_processor_stats
    # [CN] 调度策略（fcfs / priority 等）。注意字段名改名：
    #      EngineArgs 侧叫 scheduling_policy，SchedulerConfig 侧叫 policy。
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
    # [CN] 自定义 scheduler 类（可传类本身或 "mod.cls" 字符串路径），
    #      替换默认的 vllm.v1.core.sched.scheduler.Scheduler。
    scheduler_cls: str | type[object] | None = SchedulerConfig.scheduler_cls

    # [CN] pooling 模型（embedding / 分类）的输出汇聚方式配置。
    pooler_config: PoolerConfig | None = ModelConfig.pooler_config
    # ---------------- 编译 / 内核 / 注意力（整块子配置直接提升） ----------------
    # [CN] 下面几个是"整块子配置"而非扁平字段：用 get_field(VllmConfig, ...) 取到
    #      VllmConfig 里该子配置的默认实例。因为它们是 dataclass，
    #      _compute_kwargs 会用 TypeAdapter.validate_json 解析，CLI 上直接传一整块 JSON。
    compilation_config: CompilationConfig = get_field(VllmConfig, "compilation_config")
    attention_config: AttentionConfig = get_field(VllmConfig, "attention_config")
    mamba_config: MambaConfig = get_field(VllmConfig, "mamba_config")
    kernel_config: KernelConfig = get_field(VllmConfig, "kernel_config")
    # [CN] 让 flashinfer 在启动时做 autotune（挑最优 kernel 配置），会拉长启动时间。
    enable_flashinfer_autotune: bool = get_field(
        KernelConfig, "enable_flashinfer_autotune"
    )
    # [CN] None = 未指定（由 kernel 后端按硬件决定），True/False = 强制开关。
    enable_bf16x3_router_gemm: bool | None = None
    # [CN] 替换 worker 实现类 / 给 worker 挂扩展类，是平台适配（非 GPU 后端）的挂载点。
    worker_cls: str = ParallelConfig.worker_cls
    worker_extension_cls: str = ParallelConfig.worker_extension_cls

    profiler_config: ProfilerConfig = get_field(VllmConfig, "profiler_config")

    # ---------------- KV / 编码器缓存的跨实例传输（P/D 分离、prefix 共享） ----------------
    # [CN] kv_transfer_config 为 None 表示不做 KV 传输（单机常规部署就是这个）。
    #      非空时启用 connector（如 NixlConnector / LMCache），支持 prefill-decode 分离。
    kv_transfer_config: KVTransferConfig | None = None
    # [CN] KV 事件发布（block 被换入/换出等）的配置，供外部缓存系统订阅。
    kv_events_config: KVEventsConfig | None = None

    # [CN] EC = Encoder Cache（多模态编码器输出缓存），与 KV cache 是两套独立缓存。
    ec_transfer_config: ECTransferConfig | None = None
    ec_manager_config: EncoderCacheManagerConfig = get_field(
        VllmConfig, "ec_manager_config"
    )
    reasoning_config: ReasoningConfig = get_field(VllmConfig, "reasoning_config")

    # [CN] 从模型的 generation_config.json 读取默认采样参数（temperature 等）。
    #      设为 "vllm" 时读 vLLM 自己的默认；设为 "auto" 时读 HF 的。
    generation_config: str = ModelConfig.generation_config
    # [CN] 睡眠模式：让引擎释放显存（权重/KV）后"休眠"，被唤醒时再恢复。
    #      用于 RLHF 等需要"训练与推理交替占用同一张卡"的场景，避免反复启停进程。
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode
    # [CN] 用 CUDA VMM（虚拟内存管理）分配器做显存的按需映射/释放，配合 sleep 模式。
    enable_cumem_allocator: bool = ModelConfig.enable_cumem_allocator
    # [CN] 休眠时挂起 NCCL 通信域而不是销毁它，唤醒时可复用（省去重建通信域的开销）。
    enable_nccl_comm_suspend: bool = ModelConfig.enable_nccl_comm_suspend
    # [CN] 覆盖模型自带的 generation_config 中的字段（优先级最高）。
    override_generation_config: dict[str, Any] = get_field(
        ModelConfig, "override_generation_config"
    )
    # [CN] 选用哪种模型实现（如 "vllm" / "transformers" / "auto"）。
    model_impl: str = ModelConfig.model_impl
    # [CN] 指定 attention 后端（FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENSION ...）。
    #      None 时会由 AttentionSelector 按硬件、dtype、head size 等自动挑选。
    attention_backend: AttentionBackendEnum | None = AttentionConfig.backend

    # ---------------- Mamba / 混合模型（状态空间模型）的缓存 ----------------
    # [CN] Mamba 类模型没有 KV cache，而是 SSM state cache，因此有一整套独立参数。
    # [CN] 按层名指定"这些层不参与 kv_cache_dtype 量化"，用于敏感层保精度。
    kv_cache_dtype_skip_layers: list[str] = get_field(
        CacheConfig, "kv_cache_dtype_skip_layers"
    )
    mamba_cache_dtype: MambaDType = CacheConfig.mamba_cache_dtype
    mamba_ssm_cache_dtype: MambaDType = CacheConfig.mamba_ssm_cache_dtype
    mamba_block_size: int | None = get_field(CacheConfig, "mamba_block_size")
    # [CN] 前缀匹配的粒度单位：Mamba 的 state 不能逐 token 复用，需要按块对齐。
    prefix_match_unit: int | None = get_field(CacheConfig, "prefix_match_unit")
    mamba_cache_mode: MambaCacheMode = CacheConfig.mamba_cache_mode
    replayssm_buffer_len: int = CacheConfig.replayssm_buffer_len
    use_replayssm: bool = CacheConfig.use_replayssm

    mamba_backend: MambaBackendEnum = MambaBackendEnum.TRITON
    mamba_ssu_algorithm: MambaSSUAlgorithm | None = None
    enable_mamba_cache_stochastic_rounding: bool = (
        MambaConfig.enable_stochastic_rounding
    )
    mamba_cache_philox_rounds: int = MambaConfig.stochastic_rounding_philox_rounds

    additional_config: dict[str, Any] = get_field(VllmConfig, "additional_config")

    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load
    pt_load_map_location: str | dict[str, str] = LoadConfig.pt_load_map_location

    logits_processors: list[str | type[LogitsProcessor]] | None = (
        ModelConfig.logits_processors
    )
    """Custom logitproc types"""
    # [CN] 自定义 logits processor 的插件入口：可在采样前对 logits 做任意变换
    #      （如禁用某些 token、加自定义 bias）。传类名或 "mod.cls" 字符串。

    # [CN] 异步调度：让 CPU 侧的调度准备与 GPU 上的 forward 重叠，降低 CPU 成为瓶颈的概率。
    #      None 时按配置自动决定；某些模型/后端不支持，会被强制关掉。
    async_scheduling: bool | None = SchedulerConfig.async_scheduling

    # [CN] 流式输出的间隔（每生成多少个 token 推一次）。调大降开销、调小更"实时"。
    stream_interval: int = SchedulerConfig.stream_interval

    # [CN] 开启"KV 共享快速 prefill"：多个请求共享同一段 KV 时走加速路径。
    kv_sharing_fast_prefill: bool = CacheConfig.kv_sharing_fast_prefill
    # [CN] 优化等级与性能模式是 VllmConfig 级别的"总开关"，
    #      会批量改写 CompilationConfig 中的融合/编译策略（见 config/vllm.py 的 OPTIMIZATION_LEVEL_*）。
    optimization_level: OptimizationLevel = VllmConfig.optimization_level
    performance_mode: PerformanceMode = VllmConfig.performance_mode

    # [CN] 容错：某个 worker 挂掉时尝试恢复/重启，而不是整个引擎退出。
    fault_tolerance_config: FaultToleranceConfig = get_field(
        ParallelConfig, "fault_tolerance_config"
    )
    enable_fault_tolerance: bool = ParallelConfig.enable_fault_tolerance

    # [CN] 把一部分 KV cache 卸载到 CPU/其他设备的容量（GB 或比例，视实现而定）。
    kv_offloading_size: float | None = CacheConfig.kv_offloading_size
    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend
    tokens_only: bool = False

    shutdown_timeout: int = 0

    weight_transfer_config: WeightTransferConfig | None = get_field(
        VllmConfig,
        "weight_transfer_config",
    )

    fail_on_environ_validation: bool = False
    gdn_prefill_backend: Literal["flashinfer", "triton", "cutedsl"] | None = None
    kda_prefill_backend: Literal["auto", "triton", "flashkda"] | None = None

    def __post_init__(self):
        """构造后的自动规范化。三件事：

        1. 把仍是 dict 的嵌套配置字段升级为真正的配置对象，使得 Python API 可以
           直接写 EngineArgs(compilation_config={...}) 而不必手工构造对象；
        2. 解析 quantization / quantization_config（前者是快捷名，后者是完整
           QuantizationConfigArgs），统一成规范化后的 quantization_config；
        3. 加载插件；当 HuggingFace 处于离线模式时，把 model / tokenizer 的
           repo id 改写成本地缓存路径。

        Note:
            会执行 load_general_plugins()，这是有副作用的操作（插件可注册新的
            后端、量化方法、平台等，进而影响后续所有默认值与校验）。
        """
        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, dict):
            self.compilation_config = CompilationConfig(**self.compilation_config)
        if isinstance(self.attention_config, dict):
            self.attention_config = AttentionConfig(**self.attention_config)
        if isinstance(self.mamba_config, dict):
            self.mamba_config = MambaConfig(**self.mamba_config)
        if isinstance(self.kernel_config, dict):
            self.kernel_config = KernelConfig(**self.kernel_config)
        if isinstance(self.ec_manager_config, dict):
            self.ec_manager_config = EncoderCacheManagerConfig(**self.ec_manager_config)
        if isinstance(self.eplb_config, dict):
            self.eplb_config = EPLBConfig(**self.eplb_config)
        if isinstance(self.weight_transfer_config, dict):
            self.weight_transfer_config = WeightTransferConfig(
                **self.weight_transfer_config
            )
        if isinstance(self.fault_tolerance_config, dict):
            # 隐式约束：只要用户显式传了容错配置，就认为他想开启容错，
            # 不必再单独写 --enable-fault-tolerance。
            if not self.enable_fault_tolerance:
                logger.warning(
                    "--fault-tolerance-config was passed. Fault tolerance is being "
                    "automatically enabled."
                )
                self.enable_fault_tolerance = True
            self.fault_tolerance_config = FaultToleranceConfig(
                **self.fault_tolerance_config
            )
        if isinstance(self.ir_op_priority, dict):
            self.ir_op_priority = IrOpPriorityConfig(**self.ir_op_priority)

        from vllm.config.quantization import resolve_quantization_config

        self.quantization_config = resolve_quantization_config(
            self.quantization, self.quantization_config
        )

        # Setup plugins
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        # when use hf offline,replace model and tokenizer id to local model path
        if huggingface_hub.constants.HF_HUB_OFFLINE:
            # Skip cloud storage URIs (s3://, gs://, az://) — they are not
            # HF repo IDs and will be resolved later by
            # ModelConfig.maybe_pull_model_tokenizer_for_runai().
            if not is_cloud_storage(self.model):
                model_id = self.model
                self.model = get_model_path(self.model, self.revision)
                if model_id is not self.model:
                    logger.info(
                        "HF_HUB_OFFLINE is True, replace model_id "
                        "[%s] to model_path [%s]",
                        model_id,
                        self.model,
                    )
            if self.tokenizer is not None and not is_cloud_storage(self.tokenizer):
                tokenizer_id = self.tokenizer
                self.tokenizer = get_model_path(self.tokenizer, self.tokenizer_revision)
                if tokenizer_id is not self.tokenizer:
                    logger.info(
                        "HF_HUB_OFFLINE is True, replace tokenizer_id [%s] "
                        "to tokenizer_path [%s]",
                        tokenizer_id,
                        self.tokenizer,
                    )

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # 中文补充：把 EngineArgs 的字段注册到 argparse 上。每个子配置对应一个
        # argument_group，参数名统一为 --连字符形式，而 kwargs（默认值、help、
        # type/choices）由 get_kwargs(子配置类) 反射得到。
        #
        # [CN] 【本函数的整体形态】下面 900 多行几乎是同一套模板的重复：
        #         <group>_kwargs = get_kwargs(SomeConfig)      # 反射出 {字段: argparse kwargs}
        #         <group>_group  = parser.add_argument_group(title="SomeConfig", ...)
        #         <group>_group.add_argument("--some-field", **<group>_kwargs["some_field"])
        #       所以读它时不必逐行看，只需理解三件事：
        #       1) 参数名 = 字段名把下划线换成连字符；
        #       2) 每个参数的 type/choices/nargs/default 全部来自字段的类型注解与默认值，
        #          即"配置类改一处，CLI 自动跟着变"；
        #       3) 少数地方会**手工覆盖** kwargs（改 default、追加 help、删掉 choices），
        #          这些才是需要特别留意的例外，都会在下面就近标注。
        #
        # [CN] 为什么 CLI 参数没有覆盖 EngineArgs 的全部字段？
        #       因为有些字段不适合/不需要在命令行上暴露（内部结构、内部推导值、
        #       或只在 Python API 里才有意义的字段），它们只能在 EngineArgs(...) 里传。
        #       反过来，有些 CLI 参数是本函数额外加的（不属于任何子配置字段），
        #       会在 from_cli_args 之后由调用方单独处理。

        # Model arguments
        # 例外：`vllm serve --help` 时不注册 --model，避免与 serve 子命令自己的
        # model 参数冲突/重复。
        model_kwargs = get_kwargs(ModelConfig)
        model_group = parser.add_argument_group(
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        if not ("serve" in sys.argv[1:] and "--help" in sys.argv[1:]):
            model_group.add_argument("--model", **model_kwargs["model"])
        model_group.add_argument("--runner", **model_kwargs["runner"])
        model_group.add_argument("--convert", **model_kwargs["convert"])
        model_group.add_argument("--tokenizer", **model_kwargs["tokenizer"])
        model_group.add_argument("--tokenizer-mode", **model_kwargs["tokenizer_mode"])
        model_group.add_argument(
            "--trust-remote-code", **model_kwargs["trust_remote_code"]
        )
        model_group.add_argument("--dtype", **model_kwargs["dtype"])
        model_group.add_argument("--seed", **model_kwargs["seed"])
        model_group.add_argument("--hf-config-path", **model_kwargs["hf_config_path"])
        model_group.add_argument(
            "--allowed-local-media-path", **model_kwargs["allowed_local_media_path"]
        )
        model_group.add_argument(
            "--allowed-media-domains", **model_kwargs["allowed_media_domains"]
        )
        model_group.add_argument("--revision", **model_kwargs["revision"])
        model_group.add_argument("--code-revision", **model_kwargs["code_revision"])
        model_group.add_argument(
            "--tokenizer-revision", **model_kwargs["tokenizer_revision"]
        )
        model_group.add_argument("--max-model-len", **model_kwargs["max_model_len"])
        model_group.add_argument("--quantization", "-q", **model_kwargs["quantization"])
        model_group.add_argument(
            "--quantization-config", **model_kwargs["quantization_config"]
        )
        model_group.add_argument(
            "--allow-deprecated-quantization",
            **model_kwargs["allow_deprecated_quantization"],
        )
        model_group.add_argument("--enforce-eager", **model_kwargs["enforce_eager"])
        model_group.add_argument(
            "--enable-return-routed-experts",
            **model_kwargs["enable_return_routed_experts"],
        )
        model_group.add_argument(
            "--return-sampling-mask",
            **model_kwargs["return_sampling_mask"],
        )
        model_group.add_argument("--max-logprobs", **model_kwargs["max_logprobs"])
        model_group.add_argument("--logprobs-mode", **model_kwargs["logprobs_mode"])
        model_group.add_argument("--use-fp64-gumbel", **model_kwargs["use_fp64_gumbel"])
        model_group.add_argument(
            "--enable-trace-replay", **model_kwargs["enable_trace_replay"]
        )
        model_group.add_argument(
            "--disable-sliding-window", **model_kwargs["disable_sliding_window"]
        )
        model_group.add_argument(
            "--disable-cascade-attn", **model_kwargs["disable_cascade_attn"]
        )
        model_group.add_argument(
            "--skip-tokenizer-init", **model_kwargs["skip_tokenizer_init"]
        )
        model_group.add_argument(
            "--enable-prompt-embeds", **model_kwargs["enable_prompt_embeds"]
        )
        model_group.add_argument(
            "--served-model-name", **model_kwargs["served_model_name"]
        )
        model_group.add_argument("--config-format", **model_kwargs["config_format"])
        model_group.add_argument("--hf-token", **model_kwargs["hf_token"])
        model_group.add_argument("--hf-overrides", **model_kwargs["hf_overrides"])
        model_group.add_argument(
            "--model-class-overrides", **model_kwargs["model_class_overrides"]
        )
        model_group.add_argument("--pooler-config", **model_kwargs["pooler_config"])
        model_group.add_argument(
            "--generation-config", **model_kwargs["generation_config"]
        )
        model_group.add_argument(
            "--override-generation-config", **model_kwargs["override_generation_config"]
        )
        model_group.add_argument(
            "--enable-sleep-mode", **model_kwargs["enable_sleep_mode"]
        )
        model_group.add_argument(
            "--enable-cumem-allocator", **model_kwargs["enable_cumem_allocator"]
        )
        model_group.add_argument(
            "--enable-nccl-comm-suspend",
            **model_kwargs["enable_nccl_comm_suspend"],
        )
        model_group.add_argument("--model-impl", **model_kwargs["model_impl"])
        model_group.add_argument(
            "--logits-processors", **model_kwargs["logits_processors"]
        )
        model_group.add_argument(
            "--io-processor-plugin", **model_kwargs["io_processor_plugin"]
        )
        model_group.add_argument(
            "--renderer-num-workers",
            **model_kwargs["renderer_num_workers"],
        )

        # Model loading arguments
        # 权重加载相关：load_format 决定走哪种 loader，ignore_patterns 用于跳过权重
        load_kwargs = get_kwargs(LoadConfig)
        load_group = parser.add_argument_group(
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument("--load-format", **load_kwargs["load_format"])
        load_group.add_argument("--download-dir", **load_kwargs["download_dir"])
        load_group.add_argument(
            "--safetensors-load-strategy", **load_kwargs["safetensors_load_strategy"]
        )
        load_group.add_argument(
            "--safetensors-prefetch-num-threads",
            **load_kwargs["safetensors_prefetch_num_threads"],
        )
        load_group.add_argument(
            "--safetensors-prefetch-block-size",
            **load_kwargs["safetensors_prefetch_block_size"],
        )
        load_group.add_argument(
            "--model-loader-extra-config", **load_kwargs["model_loader_extra_config"]
        )
        load_group.add_argument("--ignore-patterns", **load_kwargs["ignore_patterns"])
        load_group.add_argument("--use-tqdm-on-load", **load_kwargs["use_tqdm_on_load"])
        load_group.add_argument(
            "--pt-load-map-location", **load_kwargs["pt_load_map_location"]
        )

        # Attention arguments
        # [CN] AttentionConfig 只暴露一个 backend：其余注意力相关参数（如 KV cache 布局）
        # 要么自动推导、要么属于其它子配置。
        attention_kwargs = get_kwargs(AttentionConfig)
        attention_group = parser.add_argument_group(
            title="AttentionConfig",
            description=AttentionConfig.__doc__,
        )
        attention_group.add_argument(
            "--attention-backend", **attention_kwargs["backend"]
        )

        # Mamba arguments
        # [CN] Mamba（状态空间模型）专属配置。注意字段改名：
        #      MambaConfig 里是 ssu_algorithm / enable_stochastic_rounding /
        #      stochastic_rounding_philox_rounds，CLI 上都加了 mamba 前缀。
        #      随机舍入（stochastic rounding）用于低精度 state 累积时保持无偏。
        mamba_kwargs = get_kwargs(MambaConfig)
        mamba_group = parser.add_argument_group(
            title="MambaConfig",
            description=MambaConfig.__doc__,
        )
        mamba_group.add_argument("--mamba-backend", **mamba_kwargs["backend"])
        mamba_group.add_argument(
            "--mamba-ssu-algorithm", **mamba_kwargs["ssu_algorithm"]
        )
        mamba_group.add_argument(
            "--enable-mamba-cache-stochastic-rounding",
            **mamba_kwargs["enable_stochastic_rounding"],
        )
        mamba_group.add_argument(
            "--mamba-cache-philox-rounds",
            **mamba_kwargs["stochastic_rounding_philox_rounds"],
        )

        # Structured outputs arguments
        # [CN] 结构化输出（JSON schema / grammar）与 reasoning parser。
        #      注意 --reasoning-parser 的 choices 会被留到解析之后再校验，
        #      因为插件可能额外注册新的 parser，反射阶段拿不到完整列表。
        structured_outputs_kwargs = get_kwargs(StructuredOutputsConfig)
        structured_outputs_group = parser.add_argument_group(
            title="StructuredOutputsConfig",
            description=StructuredOutputsConfig.__doc__,
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser",
            # Choices need to be validated after parsing to include plugins
            **structured_outputs_kwargs["reasoning_parser"],
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser-plugin",
            **structured_outputs_kwargs["reasoning_parser_plugin"],
        )

        # Parallel arguments
        # TP/PP/EP/DP/CP 各种并行度与分布式后端都在这里注册
        parallel_kwargs = get_kwargs(ParallelConfig)
        parallel_group = parser.add_argument_group(
            title="ParallelConfig",
            description=ParallelConfig.__doc__,
        )
        parallel_group.add_argument(
            "--distributed-executor-backend",
            **parallel_kwargs["distributed_executor_backend"],
        )
        parallel_group.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            **parallel_kwargs["pipeline_parallel_size"],
        )
        parallel_group.add_argument("--master-addr", **parallel_kwargs["master_addr"])
        parallel_group.add_argument("--master-port", **parallel_kwargs["master_port"])
        parallel_group.add_argument("--nnodes", "-n", **parallel_kwargs["nnodes"])
        parallel_group.add_argument("--node-rank", "-r", **parallel_kwargs["node_rank"])
        parallel_group.add_argument(
            "--distributed-timeout-seconds",
            **parallel_kwargs["distributed_timeout_seconds"],
        )
        parallel_group.add_argument(
            "--cpu-distributed-timeout-seconds",
            **parallel_kwargs["cpu_distributed_timeout_seconds"],
        )
        parallel_group.add_argument("--numa-bind", **parallel_kwargs["numa_bind"])
        parallel_group.add_argument(
            "--numa-bind-nodes", **parallel_kwargs["numa_bind_nodes"]
        )
        parallel_group.add_argument(
            "--numa-bind-cpus", **parallel_kwargs["numa_bind_cpus"]
        )
        parallel_group.add_argument(
            "--device-ids",
            type=lambda s: [
                int(device_id) if device_id.isdigit() else device_id
                for device_id in (part.strip() for part in s.split(","))
            ],
            default=None,
            help="Comma-separated physical GPU device IDs or UUIDs to use "
            '(e.g. --device-ids "2,3,5,7"). Avoids setting '
            "CUDA_VISIBLE_DEVICES, preserving full GPU topology "
            "visibility for GPU-NIC affinity and DeepGEMM. "
            "Note: has no effect with Ray executors; use Ray "
            "placement groups for GPU selection instead.",
        )
        parallel_group.add_argument(
            "--tensor-parallel-size", "-tp", **parallel_kwargs["tensor_parallel_size"]
        )
        parallel_group.add_argument(
            "--decode-context-parallel-size",
            "-dcp",
            **parallel_kwargs["decode_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--dcp-comm-backend",
            **parallel_kwargs["dcp_comm_backend"],
        )
        parallel_group.add_argument(
            "--dcp-q-replicate",
            **parallel_kwargs["dcp_q_replicate"],
        )
        parallel_group.add_argument(
            "--dcp-kv-cache-interleave-size",
            **parallel_kwargs["dcp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--cp-kv-cache-interleave-size",
            **parallel_kwargs["cp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--prefill-context-parallel-size",
            "-pcp",
            **parallel_kwargs["prefill_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--data-parallel-size", "-dp", **parallel_kwargs["data_parallel_size"]
        )
        # [CN] 下面这几个 DP 参数是【手工注册】的（手写 type + help，不用反射），
        #      因为它们在 ParallelConfig 上的默认语义与 CLI 侧不同（CLI 侧默认 None
        #      = "未指定，稍后推导"，而配置侧需要一个确定值）。
        #      另一个共同点：它们都带一个短选项（-dpn/-dpr/-dpl/-dpa/-dpp/-dpb/-dph/-dpe/-dpm），
        #      便于在多节点脚本里书写。注意 -dp 系列短选项极易记混，建议脚本里写全称。
        parallel_group.add_argument(
            "--data-parallel-rank",
            "-dpn",
            type=int,
            help="Data parallel rank of this instance. "
            "When set, enables external load balancer mode for MoE "
            "data-parallel deployments. Unsupported for non-MoE models; "
            "launch independent vLLM instances instead.",
        )
        parallel_group.add_argument(
            "--data-parallel-start-rank",
            "-dpr",
            type=int,
            help="Starting data parallel rank for secondary nodes.",
        )
        parallel_group.add_argument(
            "--data-parallel-size-local",
            "-dpl",
            type=int,
            help="Number of data parallel replicas to run on this node.",
        )
        parallel_group.add_argument(
            "--data-parallel-address",
            "-dpa",
            type=str,
            help="Address of data parallel cluster head-node.",
        )
        parallel_group.add_argument(
            "--data-parallel-rpc-port",
            "-dpp",
            type=int,
            help="Fixed port for data parallel RPC communication. All nodes "
            "must use the same port.",
        )
        parallel_group.add_argument(
            "--data-parallel-backend",
            "-dpb",
            type=str,
            default="mp",
            help='Backend for data parallel, either "mp" or "ray".',
        )
        parallel_group.add_argument(
            "--data-parallel-hybrid-lb",
            "-dph",
            **parallel_kwargs["data_parallel_hybrid_lb"],
        )
        parallel_group.add_argument(
            "--data-parallel-external-lb",
            "-dpe",
            **parallel_kwargs["data_parallel_external_lb"],
        )
        parallel_group.add_argument(
            "--data-parallel-multi-port-external-lb",
            "-dpm",
            action="store_true",
            default=False,
            help="Run a node-local supervisor that launches one external-LB API "
            "server per local data parallel rank and exposes aggregated health on "
            "a supervisor port.",
        )
        parallel_group.add_argument(
            "--enable-expert-parallel",
            "-ep",
            **parallel_kwargs["enable_expert_parallel"],
        )
        parallel_group.add_argument(
            "--enable-batch-sharded-sampling",
            **parallel_kwargs["enable_batch_sharded_sampling"],
        )
        parallel_group.add_argument(
            "--enable-ep-weight-filter",
            **parallel_kwargs["enable_ep_weight_filter"],
        )
        parallel_group.add_argument(
            "--all2all-backend", **parallel_kwargs["all2all_backend"]
        )
        parallel_group.add_argument("--enable-dbo", **parallel_kwargs["enable_dbo"])
        parallel_group.add_argument(
            "--ubatch-size",
            **parallel_kwargs["ubatch_size"],
        )
        parallel_group.add_argument(
            "--enable-elastic-ep", **parallel_kwargs["enable_elastic_ep"]
        )
        parallel_group.add_argument(
            "--dbo-decode-token-threshold",
            **parallel_kwargs["dbo_decode_token_threshold"],
        )
        parallel_group.add_argument(
            "--dp-sync-interval",
            **parallel_kwargs["dp_sync_interval"],
        )
        parallel_group.add_argument(
            "--dbo-prefill-token-threshold",
            **parallel_kwargs["dbo_prefill_token_threshold"],
        )
        parallel_group.add_argument(
            "--disable-nccl-for-dp-synchronization",
            **parallel_kwargs["disable_nccl_for_dp_synchronization"],
        )
        parallel_group.add_argument("--enable-eplb", **parallel_kwargs["enable_eplb"])
        parallel_group.add_argument("--eplb-config", **parallel_kwargs["eplb_config"])
        parallel_group.add_argument(
            "--expert-placement-strategy",
            **parallel_kwargs["expert_placement_strategy"],
        )

        parallel_group.add_argument(
            "--max-parallel-loading-workers",
            **parallel_kwargs["max_parallel_loading_workers"],
        )
        parallel_group.add_argument(
            "--ray-workers-use-nsight", **parallel_kwargs["ray_workers_use_nsight"]
        )
        parallel_group.add_argument(
            "--disable-custom-all-reduce",
            **parallel_kwargs["disable_custom_all_reduce"],
        )
        parallel_group.add_argument("--worker-cls", **parallel_kwargs["worker_cls"])
        parallel_group.add_argument(
            "--worker-extension-cls", **parallel_kwargs["worker_extension_cls"]
        )
        parallel_group.add_argument(
            "--enable-fault-tolerance", **parallel_kwargs["enable_fault_tolerance"]
        )
        parallel_group.add_argument(
            "--fault-tolerance-config", **parallel_kwargs["fault_tolerance_config"]
        )

        # KV cache arguments
        # 显存与 KV cache 相关。注意 --enable-prefix-caching 的 default 被强制改成
        # None（哨兵），以便后续按模型能力推导；直接沿用 CacheConfig 的默认值会让
        # "用户未指定" 和 "用户显式指定为默认值" 无法区分。
        cache_kwargs = get_kwargs(CacheConfig)
        cache_group = parser.add_argument_group(
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
        cache_group.add_argument(
            "--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"]
        )
        cache_group.add_argument(
            "--kv-cache-memory-bytes", **cache_kwargs["kv_cache_memory_bytes"]
        )
        cache_group.add_argument("--kv-cache-dtype", **cache_kwargs["cache_dtype"])
        cache_group.add_argument(
            "--num-gpu-blocks-override", **cache_kwargs["num_gpu_blocks_override"]
        )
        cache_group.add_argument(
            "--enable-prefix-caching",
            **{
                **cache_kwargs["enable_prefix_caching"],
                "default": None,
            },
        )
        cache_group.add_argument(
            "--prefix-caching-hash-algo", **cache_kwargs["prefix_caching_hash_algo"]
        )
        cache_group.add_argument(
            "--prefix-cache-retention-interval",
            **cache_kwargs["prefix_cache_retention_interval"],
        )
        cache_group.add_argument(
            "--kv-cache-dtype-skip-layers", **cache_kwargs["kv_cache_dtype_skip_layers"]
        )
        cache_group.add_argument(
            "--kv-sharing-fast-prefill", **cache_kwargs["kv_sharing_fast_prefill"]
        )
        cache_group.add_argument(
            "--mamba-cache-dtype", **cache_kwargs["mamba_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-ssm-cache-dtype", **cache_kwargs["mamba_ssm_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-block-size", **cache_kwargs["mamba_block_size"]
        )
        cache_group.add_argument(
            "--prefix-match-unit", **cache_kwargs["prefix_match_unit"]
        )
        cache_group.add_argument(
            "--mamba-cache-mode", **cache_kwargs["mamba_cache_mode"]
        )
        cache_group.add_argument(
            "--replayssm-buffer-len", **cache_kwargs["replayssm_buffer_len"]
        )
        cache_group.add_argument("--use-replayssm", **cache_kwargs["use_replayssm"])
        cache_group.add_argument(
            "--kv-offloading-size", **cache_kwargs["kv_offloading_size"]
        )
        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )

        # Model weight offload related configs
        # 三种权重卸载策略共用一个 CLI group：基础 OffloadConfig + UVA(统一虚拟寻址)
        # + Prefetch(预取)，最终在 create_engine_config 里组装成一个 OffloadConfig。
        offload_kwargs = get_kwargs(OffloadConfig)
        uva_kwargs = get_kwargs(UVAOffloadConfig)
        prefetch_kwargs = get_kwargs(PrefetchOffloadConfig)
        offload_group = parser.add_argument_group(
            title="OffloadConfig",
            description=OffloadConfig.__doc__,
        )
        offload_group.add_argument(
            "--offload-backend", **offload_kwargs["offload_backend"]
        )
        offload_group.add_argument("--cpu-offload-gb", **uva_kwargs["cpu_offload_gb"])
        offload_group.add_argument(
            "--cpu-offload-params", **uva_kwargs["cpu_offload_params"]
        )
        offload_group.add_argument(
            "--offload-group-size",
            **prefetch_kwargs["offload_group_size"],
        )
        offload_group.add_argument(
            "--offload-num-in-group",
            **prefetch_kwargs["offload_num_in_group"],
        )
        offload_group.add_argument(
            "--offload-prefetch-step",
            **prefetch_kwargs["offload_prefetch_step"],
        )
        offload_group.add_argument(
            "--offload-params", **prefetch_kwargs["offload_params"]
        )

        # Multimodal related configs
        multimodal_kwargs = get_kwargs(MultiModalConfig)
        multimodal_group = parser.add_argument_group(
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument(
            "--language-model-only", **multimodal_kwargs["language_model_only"]
        )
        multimodal_group.add_argument(
            "--limit-mm-per-prompt", **multimodal_kwargs["limit_per_prompt"]
        )
        multimodal_group.add_argument(
            "--enable-mm-embeds", **multimodal_kwargs["enable_mm_embeds"]
        )
        multimodal_group.add_argument(
            "--media-io-kwargs", **multimodal_kwargs["media_io_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-kwargs", **multimodal_kwargs["mm_processor_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-gb", **multimodal_kwargs["mm_processor_cache_gb"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-type", **multimodal_kwargs["mm_processor_cache_type"]
        )
        multimodal_group.add_argument(
            "--mm-hasher-algorithm", **multimodal_kwargs["mm_hasher_algorithm"]
        )
        multimodal_group.add_argument(
            "--mm-shm-cache-max-object-size-mb",
            **multimodal_kwargs["mm_shm_cache_max_object_size_mb"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-only", **multimodal_kwargs["mm_encoder_only"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-tp-mode", **multimodal_kwargs["mm_encoder_tp_mode"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-attn-backend",
            **multimodal_kwargs["mm_encoder_attn_backend"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-attn-dtype",
            **multimodal_kwargs["mm_encoder_attn_dtype"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-fp8-scale-path",
            **multimodal_kwargs["mm_encoder_fp8_scale_path"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-fp8-scale-save-path",
            **multimodal_kwargs["mm_encoder_fp8_scale_save_path"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-fp8-scale-save-margin",
            **multimodal_kwargs["mm_encoder_fp8_scale_save_margin"],
        )
        multimodal_group.add_argument(
            "--interleave-mm-strings", **multimodal_kwargs["interleave_mm_strings"]
        )
        multimodal_group.add_argument(
            "--skip-mm-profiling", **multimodal_kwargs["skip_mm_profiling"]
        )

        multimodal_group.add_argument(
            "--video-pruning-rate", **multimodal_kwargs["video_pruning_rate"]
        )
        multimodal_group.add_argument(
            "--video-pruning-method",
            **multimodal_kwargs["video_pruning_method"],
        )
        multimodal_group.add_argument(
            "--mm-tensor-ipc", **multimodal_kwargs["mm_tensor_ipc"]
        )
        multimodal_group.add_argument(
            "--mm-processor-device",
            choices=["auto", "cpu"]
            + (
                [current_platform.device_type]
                if current_platform.device_type not in ("", "cpu")
                else []
            ),
            default="auto",
            help="Device the HF multi-modal processor runs the image/video "
            "transform on. Convenience for `--mm-processor-kwargs "
            "'{\"device\": ...}'`: the value is resolved here and stored there, "
            "it is not kept as separate state. Only takes effect for HF "
            '"fast" (torchvision-backed) processors, which accept a `device` '
            "argument; the others ignore it and stay on CPU.\n\n"
            '"auto" uses the accelerator on encoder instances of an '
            "encode/prefill/decode deployment -- an EC producer that is not "
            "also a consumer allocates no KV cache, so its accelerator is not "
            "contended by the language model -- and then only when "
            "`--mm-tensor-ipc=torch_shm` can carry device tensors, since every "
            "other transport would copy the result back to the host and that "
            'copy costs more than it saves. "auto" resolves to "cpu" '
            "everywhere else.",
        )
        multimodal_group.add_argument(
            "--mm-ipc-gpu-memory-gb",
            **multimodal_kwargs["mm_ipc_gpu_memory_gb"],
        )
        multimodal_group.add_argument(
            "--mm-device-do-normalize",
            **{
                **multimodal_kwargs["mm_device_do_normalize"],
                "default": None,
            },
        )

        # LoRA related configs
        # [CN] 注意 --enable-lora 是【手工注册】的，不走反射：因为 EngineArgs 上的
        #      enable_lora 是硬编码 False，而 LoRAConfig 里并没有这个字段。
        #      它是 LoRA 的总开关：为 False 时下面这些 LoRA 参数全部不生效。
        lora_kwargs = get_kwargs(LoRAConfig)
        lora_group = parser.add_argument_group(
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            "--enable-lora",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of LoRA adapters.",
        )
        lora_group.add_argument("--max-loras", **lora_kwargs["max_loras"])
        lora_group.add_argument("--max-lora-rank", **lora_kwargs["max_lora_rank"])
        lora_group.add_argument(
            "--lora-dtype",
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument(
            "--enable-tower-connector-lora",
            **lora_kwargs["enable_tower_connector_lora"],
        )
        lora_group.add_argument("--max-cpu-loras", **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument(
            "--fully-sharded-loras", **lora_kwargs["fully_sharded_loras"]
        )
        lora_group.add_argument(
            "--lora-target-modules", **lora_kwargs["target_modules"]
        )
        lora_group.add_argument("--default-mm-loras", **lora_kwargs["default_mm_loras"])
        lora_group.add_argument(
            "--specialize-active-lora", **lora_kwargs["specialize_active_lora"]
        )
        lora_group.add_argument(
            "--enable-mixed-moe-lora-format",
            **lora_kwargs["enable_mixed_moe_lora_format"],
        )
        lora_group.add_argument(
            "--enable-moe-shared-loras",
            **lora_kwargs["enable_moe_shared_loras"],
        )

        # Observability arguments
        # [CN] 这一组几乎全部默认关闭——trace / 细粒度指标都会带来可观测的运行时开销，
        #      只应在定位问题时临时打开。
        observability_kwargs = get_kwargs(ObservabilityConfig)
        observability_group = parser.add_argument_group(
            title="ObservabilityConfig",
            description=ObservabilityConfig.__doc__,
        )
        observability_group.add_argument(
            "--show-hidden-metrics-for-version",
            **observability_kwargs["show_hidden_metrics_for_version"],
        )
        observability_group.add_argument(
            "--otlp-traces-endpoint", **observability_kwargs["otlp_traces_endpoint"]
        )
        # TODO: generalise this special case
        # 特例：--collect-detailed-traces 允许一次传多个模块（用逗号连接），而
        # choices 只有单项。这里额外把所有两项排列组合也加进 choices，并改用
        # metavar 展示，否则 argparse 会把 "model,worker" 判为非法值。
        choices = observability_kwargs["collect_detailed_traces"]["choices"]
        metavar = f"{{{','.join(choices)}}}"
        observability_kwargs["collect_detailed_traces"]["metavar"] = metavar
        observability_kwargs["collect_detailed_traces"]["choices"] += [
            ",".join(p) for p in permutations(get_args(DetailedTraceModules), r=2)
        ]
        observability_group.add_argument(
            "--collect-detailed-traces",
            **observability_kwargs["collect_detailed_traces"],
        )
        observability_group.add_argument(
            "--per-request-spec-decode-metrics",
            **observability_kwargs["per_request_spec_decode_metrics"],
        )
        observability_group.add_argument(
            "--kv-cache-metrics", **observability_kwargs["kv_cache_metrics"]
        )
        observability_group.add_argument(
            "--kv-cache-metrics-sample",
            **observability_kwargs["kv_cache_metrics_sample"],
        )
        observability_group.add_argument(
            "--cudagraph-metrics",
            **observability_kwargs["cudagraph_metrics"],
        )
        observability_group.add_argument(
            "--enable-layerwise-nvtx-tracing",
            **observability_kwargs["enable_layerwise_nvtx_tracing"],
        )
        observability_group.add_argument(
            "--enable-mfu-metrics",
            **observability_kwargs["enable_mfu_metrics"],
        )
        observability_group.add_argument(
            "--enable-logging-iteration-details",
            **observability_kwargs["enable_logging_iteration_details"],
        )
        observability_group.add_argument(
            "--jit-monitor-mode",
            **observability_kwargs["jit_monitor_mode"],
        )
        observability_group.add_argument(
            "--jit-monitor-verbose",
            **observability_kwargs["jit_monitor_verbose"],
        )

        # Scheduler arguments
        # 注意 --max-num-batched-tokens / --max-num-seqs / --enable-chunked-prefill
        # 的 default 都被改成 None：它们的合理值依赖模型长度与硬件，必须延后到
        # _set_default_max_num_seqs_and_batched_tokens_args 里推导。
        scheduler_kwargs = get_kwargs(SchedulerConfig)
        scheduler_group = parser.add_argument_group(
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            "--max-num-batched-tokens",
            **{
                **scheduler_kwargs["max_num_batched_tokens"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-scheduled-tokens",
            **{
                **scheduler_kwargs["max_num_scheduled_tokens"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-seqs",
            **{
                **scheduler_kwargs["max_num_seqs"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-queued-reqs", **scheduler_kwargs["max_num_queued_reqs"]
        )
        scheduler_group.add_argument(
            "--max-num-queued-tokens",
            **scheduler_kwargs["max_num_queued_tokens"],
        )
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"],
        )
        # multi-step scheduling has been removed; corresponding arguments
        # are no longer supported.
        # （已废弃能力）V1 移除了多步调度，相关参数不再注册；这里保留注释以说明
        # 为什么下面直接跳到了 --scheduling-policy。
        scheduler_group.add_argument(
            "--scheduling-policy", **scheduler_kwargs["policy"]
        )
        scheduler_group.add_argument(
            "--enable-chunked-prefill",
            **{
                **scheduler_kwargs["enable_chunked_prefill"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--disable-chunked-mm-input", **scheduler_kwargs["disable_chunked_mm_input"]
        )
        scheduler_group.add_argument(
            "--scheduler-cls", **scheduler_kwargs["scheduler_cls"]
        )
        scheduler_group.add_argument(
            "--scheduler-reserve-full-isl",
            **scheduler_kwargs["scheduler_reserve_full_isl"],
        )
        scheduler_group.add_argument("--watermark", **scheduler_kwargs["watermark"])
        scheduler_group.add_argument(
            "--prefill-schedule-interval",
            **scheduler_kwargs["prefill_schedule_interval"],
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
            **scheduler_kwargs["disable_hybrid_kv_cache_manager"],
        )
        scheduler_group.add_argument(
            "--async-scheduling", **scheduler_kwargs["async_scheduling"]
        )
        scheduler_group.add_argument(
            "--stream-interval", **scheduler_kwargs["stream_interval"]
        )

        # Compilation arguments
        compilation_kwargs = get_kwargs(CompilationConfig)
        compilation_group = parser.add_argument_group(
            title="CompilationConfig",
            description=CompilationConfig.__doc__,
        )
        compilation_group.add_argument(
            "--cudagraph-capture-sizes", **compilation_kwargs["cudagraph_capture_sizes"]
        )
        compilation_group.add_argument(
            "--max-cudagraph-capture-size",
            **compilation_kwargs["max_cudagraph_capture_size"],
        )

        # Kernel arguments
        kernel_kwargs = get_kwargs(KernelConfig)
        kernel_group = parser.add_argument_group(
            title="KernelConfig",
            description=KernelConfig.__doc__,
        )
        kernel_group.add_argument("--ir-op-priority", **kernel_kwargs["ir_op_priority"])
        kernel_group.add_argument(
            "--enable-flashinfer-autotune",
            **kernel_kwargs["enable_flashinfer_autotune"],
        )
        kernel_group.add_argument(
            "--enable-bf16x3-router-gemm",
            **kernel_kwargs["enable_bf16x3_router_gemm"],
        )
        # 后端名在 CLI 上可能写成 FlashInfer / flash-infer 等风格，先统一成小写下划线
        moe_backend_kwargs = kernel_kwargs["moe_backend"]
        moe_backend_kwargs["type"] = lambda s: s.lower().replace("-", "_")
        kernel_group.add_argument("--moe-backend", **moe_backend_kwargs)
        linear_backend_kwargs = kernel_kwargs["linear_backend"]
        linear_backend_kwargs["type"] = lambda s: s.lower().replace("-", "_")
        kernel_group.add_argument("--linear-backend", **linear_backend_kwargs)

        # vLLM arguments
        # 这一组是"挂在 VllmConfig 上、但不属于任何单个子配置"的顶层参数。
        vllm_kwargs = get_kwargs(VllmConfig)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        # We construct SpeculativeConfig using fields from other configs in
        # create_engine_config. So we set the type to a JSON string here to
        # delay the Pydantic validation that comes with SpeculativeConfig.
        # 中文补充：投机解码配置依赖 ModelConfig / ParallelConfig（如目标模型信息），
        # 因此 CLI 阶段只解析成裸 JSON dict，真正的 pydantic 校验推迟到
        # create_speculative_config() 拿到上下游配置之后再做。
        vllm_kwargs["speculative_config"]["type"] = optional_type(json.loads)
        vllm_group.add_argument(
            "--speculative-config", "-sc", **vllm_kwargs["speculative_config"]
        )
        # --spec-method / --spec-model / --spec-tokens 是 --speculative-config 的
        # 便捷写法，两者互斥（见 create_speculative_config 里的互斥校验）。
        speculative_kwargs = get_kwargs(SpeculativeConfig)
        vllm_group.add_argument("--spec-method", **speculative_kwargs["method"])
        vllm_group.add_argument("--spec-model", **speculative_kwargs["model"])
        vllm_group.add_argument(
            "--spec-tokens", **speculative_kwargs["num_speculative_tokens"]
        )
        vllm_kwargs["diffusion_config"]["type"] = optional_type(json.loads)
        vllm_group.add_argument(
            "--diffusion-config", "-dc", **vllm_kwargs["diffusion_config"]
        )
        vllm_group.add_argument(
            "--kv-transfer-config", **vllm_kwargs["kv_transfer_config"]
        )
        vllm_group.add_argument("--kv-events-config", **vllm_kwargs["kv_events_config"])
        vllm_group.add_argument(
            "--ec-transfer-config", **vllm_kwargs["ec_transfer_config"]
        )
        vllm_group.add_argument(
            "--ec-manager-config", **vllm_kwargs["ec_manager_config"]
        )
        vllm_group.add_argument(
            "--compilation-config", "-cc", **vllm_kwargs["compilation_config"]
        )
        vllm_group.add_argument(
            "--attention-config", "-ac", **vllm_kwargs["attention_config"]
        )
        vllm_group.add_argument("--reasoning-config", **vllm_kwargs["reasoning_config"])
        vllm_group.add_argument("--kernel-config", **vllm_kwargs["kernel_config"])
        vllm_group.add_argument(
            "--additional-config", **vllm_kwargs["additional_config"]
        )
        vllm_group.add_argument(
            "--structured-outputs-config", **vllm_kwargs["structured_outputs_config"]
        )
        vllm_group.add_argument("--profiler-config", **vllm_kwargs["profiler_config"])
        vllm_group.add_argument(
            "--optimization-level", **vllm_kwargs["optimization_level"]
        )
        vllm_group.add_argument("--performance-mode", **vllm_kwargs["performance_mode"])
        vllm_group.add_argument(
            "--weight-transfer-config", **vllm_kwargs["weight_transfer_config"]
        )

        # Other arguments
        # 不属于任何子配置的零散开关，直接挂在 parser 根上（不进任何 group）
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="Disable logging statistics.",
        )

        parser.add_argument(
            "--aggregate-engine-logging",
            action="store_true",
            help="Log aggregate rather than per-engine statistics "
            "when using data parallelism.",
        )

        parser.add_argument(
            "--fail-on-environ-validation",
            help="If set, the engine will raise an error if "
            "environment validation fails.",
            default=False,
            action=argparse.BooleanOptionalAction,
        )

        parser.add_argument(
            "--shutdown-timeout",
            type=int,
            default=0,
            help="Shutdown timeout in seconds. 0 = abort, >0 = wait.",
        )

        parser.add_argument(
            "--gdn-prefill-backend",
            dest="gdn_prefill_backend",
            choices=["flashinfer", "triton", "cutedsl"],
            default=None,
            help="Select GDN prefill backend.",
        )
        parser.add_argument(
            "--kda-prefill-backend",
            dest="kda_prefill_backend",
            choices=["auto", "triton", "flashkda"],
            default=None,
            help="Select KDA prefill backend.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """把 argparse 的 Namespace 转成 EngineArgs。

        只挑 EngineArgs 里真实存在且 Namespace 上也有的字段，这样同一个
        Namespace 混杂了其它子命令的参数时也不会炸（比如 --host/--port 这类
        只属于 API server 的参数会被自然忽略）。
        """
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Set the attributes from the parsed arguments.
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args

    def create_model_config(self) -> ModelConfig:
        """构造 ModelConfig——所有子配置中第一个、也是最基础的一个。

        后续 CacheConfig / ParallelConfig / SchedulerConfig 的默认值推导都要读它
        （模型是否多模态、是否 MoE、是否支持 chunked prefill、max_model_len 等），
        所以它在 create_engine_config() 里必须最先产出。

        Note:
            max_model_len、tokenizer 等字段的真正解析与自动推导发生在 ModelConfig
            自己的 __post_init__ 里（例如从 HF config 读 max_position_embeddings，
            并把显式设置与模型上限做校验），这里只负责搬运参数。
        """
        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        return ModelConfig(
            model=self.model,
            model_weights=self.model_weights,
            hf_config_path=self.hf_config_path,
            runner=self.runner,
            convert=self.convert,
            tokenizer=self.tokenizer,  # type: ignore[arg-type]
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            allowed_media_domains=self.allowed_media_domains,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            model_class_overrides=self.model_class_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_config=self.quantization_config,
            allow_deprecated_quantization=self.allow_deprecated_quantization,
            enforce_eager=self.enforce_eager,
            enable_return_routed_experts=self.enable_return_routed_experts,
            return_sampling_mask=self.return_sampling_mask,
            max_logprobs=self.max_logprobs,
            logprobs_mode=self.logprobs_mode,
            use_fp64_gumbel=self.use_fp64_gumbel,
            enable_trace_replay=self.enable_trace_replay,
            disable_sliding_window=self.disable_sliding_window,
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
            served_model_name=self.served_model_name,
            language_model_only=self.language_model_only,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            enable_mm_embeds=self.enable_mm_embeds,
            interleave_mm_strings=self.interleave_mm_strings,
            media_io_kwargs=self.media_io_kwargs,
            skip_mm_profiling=self.skip_mm_profiling,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            mm_processor_cache_gb=self.mm_processor_cache_gb,
            mm_processor_cache_type=self.mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=self.mm_shm_cache_max_object_size_mb,
            mm_hasher_algorithm=self.mm_hasher_algorithm,
            mm_encoder_only=self.mm_encoder_only,
            mm_encoder_tp_mode=self.mm_encoder_tp_mode,
            mm_encoder_attn_backend=self.mm_encoder_attn_backend,
            mm_encoder_attn_dtype=self.mm_encoder_attn_dtype,
            mm_encoder_fp8_scale_path=self.mm_encoder_fp8_scale_path,
            mm_encoder_fp8_scale_save_path=self.mm_encoder_fp8_scale_save_path,
            mm_encoder_fp8_scale_save_margin=self.mm_encoder_fp8_scale_save_margin,
            pooler_config=self.pooler_config,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            enable_cumem_allocator=self.enable_cumem_allocator,
            enable_nccl_comm_suspend=self.enable_nccl_comm_suspend,
            model_impl=self.model_impl,
            logits_processors=self.logits_processors,
            video_pruning_rate=self.video_pruning_rate,
            video_pruning_method=self.video_pruning_method,
            mm_tensor_ipc=self.mm_tensor_ipc,
            mm_ipc_gpu_memory_gb=self.mm_ipc_gpu_memory_gb,
            mm_device_do_normalize=self.mm_device_do_normalize,
            mm_processor_device=self.mm_processor_device,
            io_processor_plugin=self.io_processor_plugin,
            renderer_num_workers=self.renderer_num_workers,
        )

    def validate_tensorizer_args(self):
        """把散落在 model_loader_extra_config 顶层的 tensorizer 字段收敛进
        model_loader_extra_config["tensorizer_config"]。"""
        from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

        for key in self.model_loader_extra_config:
            if key in TensorizerConfig._fields:
                self.model_loader_extra_config["tensorizer_config"][key] = (
                    self.model_loader_extra_config[key]
                )

    def create_load_config(self) -> LoadConfig:
        """构造 LoadConfig；tensorizer 是唯一需要额外搬运转参的 loader。"""
        if self.load_format == "tensorizer":
            # tensorizer 的配置历史上是直接平铺在 model_loader_extra_config 里的，
            # 这里把它规整到 "tensorizer_config" 子键下再传给 LoadConfig。
            if hasattr(self.model_loader_extra_config, "to_serializable"):
                self.model_loader_extra_config = (
                    self.model_loader_extra_config.to_serializable()
                )
            self.model_loader_extra_config["tensorizer_config"] = {}
            self.model_loader_extra_config["tensorizer_config"]["tensorizer_dir"] = (
                self.model
            )
            self.validate_tensorizer_args()

        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            safetensors_load_strategy=self.safetensors_load_strategy,
            safetensors_prefetch_num_threads=self.safetensors_prefetch_num_threads,
            safetensors_prefetch_block_size=self.safetensors_prefetch_block_size,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
            use_tqdm_on_load=self.use_tqdm_on_load,
            pt_load_map_location=self.pt_load_map_location,
        )

    def create_speculative_config(
        self,
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
    ) -> SpeculativeConfig | None:
        """Initializes and returns a SpeculativeConfig object based on
        `speculative_config`.

        中文补充：这是"延迟校验"的落地处。CLI 只把 --speculative-config 解析成裸
        dict，这里先把 --spec-method/--spec-model/--spec-tokens 三个便捷参数合入
        （与 dict 内同名字段互斥），再注入必须来自其它配置的 target_model_config /
        target_parallel_config，最后才交给 SpeculativeConfig 做 pydantic 校验。

        Returns:
            未配置投机解码时返回 None。
        """
        for flag, key, value in (
            ("--spec-method", "method", self.spec_method),
            ("--spec-model", "model", self.spec_model),
            ("--spec-tokens", "num_speculative_tokens", self.spec_tokens),
        ):
            if value is None:
                continue
            if self.speculative_config is None:
                self.speculative_config = {}
            if key in self.speculative_config:
                raise ValueError(
                    f"{flag} and --speculative-config['{key}'] are mutually exclusive"
                )
            self.speculative_config[key] = value

        if self.speculative_config is None:
            return None

        # CLI 上习惯写 "num-speculative-tokens" 这类连字符，统一成下划线字段名
        self.speculative_config = {
            k.replace("-", "_"): v for k, v in self.speculative_config.items()
        }

        # Note(Shangming): These parameters are not obtained from the cli arg
        # '--speculative-config' and must be passed in when creating the engine
        # config.
        self.speculative_config.update(
            {
                "target_model_config": target_model_config,
                "target_parallel_config": target_parallel_config,
            }
        )
        return SpeculativeConfig(**self.speculative_config)

    def _resolve_device_ids(self) -> list[int] | None:
        """把 --device-ids 解析成物理设备 id 列表。

        支持两种写法：整数（CUDA 序号）或 UUID 字符串，但不能混用。整数写法还会
        与 CUDA_VISIBLE_DEVICES 复合——此时 --device-ids 的值被当作"在 CVD 可见
        设备集合中的下标"，而不是物理 id。

        Returns:
            物理设备 id 列表；未指定 --device-ids 时返回 None。
        """
        if not self.device_ids:
            return None
        if self.distributed_executor_backend == "ray":
            logger.warning(
                "--device-ids has no effect when using the Ray executor. "
                "Use Ray placement groups for GPU selection instead."
            )
        ids = self.device_ids
        if len(set(ids)) != len(ids):
            raise ValueError(f"--device-ids must not contain duplicates: {ids}")
        if all(isinstance(i, str) for i in ids):
            return [
                current_platform.device_control_id_to_physical_device_id(i)
                for i in cast(list[str], ids)
            ]
        if any(isinstance(i, str) for i in ids):
            raise ValueError("--device-ids must not mix integer IDs and UUIDs")
        int_ids = cast(list[int], ids)
        # Compose with CUDA_VISIBLE_DEVICES: if CVD is set, treat
        # --device-ids values as indices into the CVD-visible set.
        cvd = getattr(
            envs,
            current_platform.device_control_env_var,
            os.environ.get(current_platform.device_control_env_var),
        )
        if cvd:
            cvd_ids = [
                current_platform.device_control_id_to_physical_device_id(x)
                for x in cvd.split(",")
            ]
            for i in int_ids:
                if i >= len(cvd_ids):
                    raise ValueError(
                        f"--device-ids index {i} is out of range for "
                        f"{current_platform.device_control_env_var}"
                        f"={cvd} ({len(cvd_ids)} devices visible)"
                    )
            return [cvd_ids[i] for i in int_ids]
        return int_ids

    def create_diffusion_config(self) -> DiffusionConfig | None:
        """构造 DiffusionConfig；未启用扩散模型时返回 None。

        cfg 允许是 JSON 字符串（CLI 传入）或 dict（Python API 传入）。
        """
        if self.diffusion_config is None:
            return None
        cfg = self.diffusion_config
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        return DiffusionConfig(**cfg)

    def create_observability_config(self) -> ObservabilityConfig:
        """把分散在 EngineArgs 上的可观测性字段收集成 ObservabilityConfig。"""
        return ObservabilityConfig(
            show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_detailed_traces=self.collect_detailed_traces,
            per_request_spec_decode_metrics=self.per_request_spec_decode_metrics,
            kv_cache_metrics=self.kv_cache_metrics,
            kv_cache_metrics_sample=self.kv_cache_metrics_sample,
            cudagraph_metrics=self.cudagraph_metrics,
            enable_layerwise_nvtx_tracing=self.enable_layerwise_nvtx_tracing,
            enable_mfu_metrics=self.enable_mfu_metrics,
            enable_mm_processor_stats=self.enable_mm_processor_stats,
            enable_logging_iteration_details=self.enable_logging_iteration_details,
            jit_monitor_mode=self.jit_monitor_mode,
            jit_monitor_verbose=self.jit_monitor_verbose,
        )

    def create_engine_config(
        self,
        usage_context: UsageContext | None = None,
        headless: bool = False,
    ) -> VllmConfig:
        """
        Create the VllmConfig.

        NOTE: If VllmConfig is incompatible, we raise an error.

        中文补充：装配顺序（每一步都依赖前一步的结果，不能随意调换）：
          1. DeviceConfig（平台探测）+ 环境变量整体校验 envs.validate_environ()
          2. speculators 探测并改写 model/tokenizer/speculative_config
          3. ModelConfig   <- 后续所有推导的基础
          4. _check_feature_supported / _set_default_chunked_prefill_and_prefix_caching_args
             _set_default_reasoning_config_args（依赖模型能力的默认值推导）
          5. CacheConfig   <- 依赖 ModelConfig（is_attention_free、sliding_window、kv dtype）
          6. DP/EP/多节点拓扑推导 -> ParallelConfig
          7. SpeculativeConfig / DiffusionConfig（依赖 ModelConfig + ParallelConfig）
          8. _set_default_max_num_seqs_and_batched_tokens_args -> SchedulerConfig
          9. LoRAConfig（并与投机解码做交叉校验）
         10. attention / mamba / kernel / compilation 等顶层参数对嵌套 config 的覆盖
         11. LoadConfig、ObservabilityConfig、OffloadConfig、additional_config
         12. 汇总成 VllmConfig 返回。

        Note:
            本方法会就地修改 self（例如把推导出的默认值写回 enable_prefix_caching、
            max_num_seqs 等字段），因此重复调用是幂等的但并非无副作用。
        """
        current_platform.pre_register_and_update()

        device_config = DeviceConfig(device=cast(Device, current_platform.device_type))

        # 环境变量优先于代码默认值、低于显式参数：这里是唯一一处整体校验点，
        # 默认只告警，fail_on_environ_validation=True 时才抛错。
        envs.validate_environ(self.fail_on_environ_validation)

        # Check if the model is a speculator and override model/tokenizer/config
        # BEFORE creating ModelConfig, so the config is created with the target model
        # Skip speculator detection for cloud storage models (eg: S3, GCS) since
        # HuggingFace cannot load configs directly from S3 URLs. S3 models can still
        # use speculators with explicit --speculative-config.
        if not is_cloud_storage(self.model):
            (self.model, self.tokenizer, self.speculative_config) = (
                maybe_override_with_speculators(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    vllm_speculative_config=self.speculative_config,
                    hf_token=self.hf_token,
                )
            )

        # ModelConfig 里可能修正 model / model_weights / tokenizer（离线路径改写、
        # speculators 重写等），回写 self 以保证 EngineArgs 与配置保持一致。
        model_config = self.create_model_config()
        self.model = model_config.model
        self.model_weights = model_config.model_weights
        self.tokenizer = model_config.tokenizer

        self._check_feature_supported()
        self._set_default_chunked_prefill_and_prefix_caching_args(model_config)
        self._set_default_reasoning_config_args()
        # 隐含约束：只有"全滑动窗口"模型才把 sliding_window 交给 CacheConfig 统一管理。
        # 交错式（interleaved）滑动窗口模型若在此设置，会让 CacheConfig 的值覆盖掉
        # 全局层的行为，所以这种情况保持 None。
        sliding_window: int | None = None
        layer_types = getattr(model_config.hf_text_config, "layer_types", None)
        if layer_types is None or all(lt == "sliding_attention" for lt in layer_types):
            # Only set CacheConfig.sliding_window if the model is all sliding
            # window. Otherwise CacheConfig.sliding_window will override the
            # global layers in interleaved sliding window models.
            sliding_window = model_config.get_sliding_window()

        # Resolve "auto" kv_cache_dtype to actual value from model config
        # "auto" 只是一种意图声明，真正落到什么精度要按模型配置决定，这里解析成实际值。
        resolved_cache_dtype = resolve_kv_cache_dtype_string(
            self.kv_cache_dtype, model_config
        )

        # 哨兵必须已被 _set_default_chunked_prefill_and_prefix_caching_args 消除
        assert self.enable_prefix_caching is not None, (
            "enable_prefix_caching must be set by this point"
        )

        cache_config = CacheConfig(
            block_size=self.block_size,  # type: ignore[arg-type]
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            cache_dtype=resolved_cache_dtype,  # type: ignore[arg-type]
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=sliding_window,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            prefix_cache_retention_interval=self.prefix_cache_retention_interval,
            kv_cache_dtype_skip_layers=self.kv_cache_dtype_skip_layers,
            kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
            mamba_cache_dtype=self.mamba_cache_dtype,
            mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
            mamba_block_size=self.mamba_block_size,
            prefix_match_unit=self.prefix_match_unit,
            mamba_cache_mode=self.mamba_cache_mode,
            replayssm_buffer_len=self.replayssm_buffer_len,
            use_replayssm=self.use_replayssm,
            kv_offloading_size=self.kv_offloading_size,
            kv_offloading_backend=self.kv_offloading_backend,
        )

        # TurboQuant 的边界层不能量化，把模型算出的 boundary 层并入 skip_layers
        # （按 int 排序，因为层名在此处是数字字符串）。
        if resolved_cache_dtype.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            boundary = TurboQuantConfig.get_boundary_skip_layers(model_config)
            existing = set(cache_config.kv_cache_dtype_skip_layers)
            cache_config.kv_cache_dtype_skip_layers = sorted(
                existing | set(boundary), key=int
            )

        ray_runtime_env = None
        if is_ray_initialized():
            # Ray Serve LLM calls `create_engine_config` in the context
            # of a Ray task, therefore we check is_ray_initialized()
            # as opposed to is_in_ray_actor().
            import ray

            ray_runtime_env = ray.get_runtime_context().runtime_env
            # Avoid logging sensitive environment variables
            sanitized_env = ray_runtime_env.to_dict() if ray_runtime_env else {}
            if "env_vars" in sanitized_env:
                sanitized_env["env_vars"] = {
                    k: "***" for k in sanitized_env["env_vars"]
                }
            logger.info("Using ray runtime env (env vars redacted): %s", sanitized_env)

        # Get the current placement group if Ray is initialized and
        # we are in a Ray actor. If so, then the placement group will be
        # passed to spawned processes.
        placement_group = None
        if is_in_ray_actor():
            import ray

            # This call initializes Ray automatically if it is not initialized,
            # but we should not do this here.
            placement_group = ray.util.get_current_placement_group()

        # ------------------------------------------------------------------
        # 数据并行（DP）部署形态推导。三种互斥形态：
        #   internal LB（默认，vLLM 内部负载均衡）
        #   external LB（外部负载均衡器，每个 rank 一个实例，需要显式 rank）
        #   hybrid LB  （多节点混合：节点内 internal、节点间 external）
        # 下面这段大量校验的意义在于：这些组合错误在运行时极难排查，必须在启动期拒绝。
        # ------------------------------------------------------------------
        assert not headless or not self.data_parallel_hybrid_lb, (
            "data_parallel_hybrid_lb is not applicable in headless mode"
        )
        if self.data_parallel_hybrid_lb and self.data_parallel_external_lb:
            raise ValueError(
                "Invalid data-parallel launch options: "
                "`--data-parallel-hybrid-lb` and "
                "`--data-parallel-external-lb` cannot be enabled together. "
                "Enable only one load-balancing mode."
            )
        if self.nnodes > 1 and self.data_parallel_backend != "mp":
            raise ValueError(
                "Invalid data-parallel launch options: "
                f"`--nnodes {self.nnodes}` requires "
                "`--data-parallel-backend mp`; got "
                f"`--data-parallel-backend {self.data_parallel_backend}`. "
                "Use the MP backend or set `--nnodes 1`."
            )
        # 多节点时：总 world size 必须能被节点数整除，才能均分出每节点的本地规模；
        # 并据此由 node_rank 反推出本节点的 data_parallel_rank。
        inferred_data_parallel_rank = 0
        if self.nnodes > 1:
            world_size_within_dp = (
                self.pipeline_parallel_size
                * self.tensor_parallel_size
                * self.prefill_context_parallel_size
            )
            world_size = self.data_parallel_size * world_size_within_dp
            if world_size % self.nnodes != 0:
                raise ValueError(
                    "Invalid data-parallel launch options: "
                    f"`--nnodes {self.nnodes}` must evenly divide the total "
                    f"world size ({world_size}). Adjust `--nnodes`, "
                    "`--data-parallel-size`, `--pipeline-parallel-size`, "
                    "`--tensor-parallel-size`, or `--prefill-context-parallel-size`."
                )
            if not 0 <= self.node_rank < self.nnodes:
                raise ValueError(
                    "Invalid data-parallel launch options: `--node-rank` must "
                    f"be between 0 and {self.nnodes - 1}; got "
                    f"`--node-rank {self.node_rank}`. Set it to this node's "
                    "zero-based index."
                )
            # [CN] 每个节点分到的进程数；本节点的第一个 DP rank =
            #      (本节点起始进程号) / (单个 DP rank 占多少进程)，
            #      即"节点内第 node_rank 段、每段 local_world_size 个进程"对应的 DP 编号。
            local_world_size = world_size // self.nnodes
            inferred_data_parallel_rank = (
                self.node_rank * local_world_size
            ) // world_size_within_dp
            if self.data_parallel_size > 1 and self.data_parallel_external_lb:
                # [CN] external LB 下每个 rank 必须有确定身份，因此直接采用推断值。
                self.data_parallel_rank = inferred_data_parallel_rank
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d for external lb",
                    self.data_parallel_rank,
                    self.node_rank,
                )
            elif self.data_parallel_size_local is None:
                # Infer data parallel size local for internal dplb:
                # [CN] internal LB 下只需知道"本节点有几个 DP rank"，
                #      = 本节点进程数 / 每个 DP rank 的进程数，至少为 1。
                self.data_parallel_size_local = max(
                    local_world_size // world_size_within_dp, 1
                )
        # [CN] 关键推导：只要用户显式给了 --data-parallel-rank，就等价于开启 external LB
        #      （因为指定了 rank 意味着外部已经知道要往哪个实例发请求）。
        data_parallel_external_lb = (
            self.data_parallel_external_lb or self.data_parallel_rank is not None
        )
        # 启用容错必须配合外部 LB：内部 LB 下没有独立的 rank 身份，无法做故障接管
        if self.enable_fault_tolerance and not data_parallel_external_lb:
            raise ValueError(
                "Fault tolerance requires external load balancer mode "
                "(--data-parallel-external-lb or --data-parallel-rank). "
                "Internal LB mode is not supported."
            )
        if (
            self.data_parallel_size > 1
            and data_parallel_external_lb
            and not model_config.is_moe
        ):
            raise ValueError(
                "Non-MoE models do not support external data parallel mode. "
                "For external load balancing, launch independent vLLM "
                "instances without --data-parallel-* arguments."
            )
        # Local DP rank = 1, use pure-external LB.
        # [CN] external LB 分支：每个实例独占一个 rank（local size 强制为 1），
        #      且必须能确定 rank；同时它天然不是 hybrid（hybrid 要求节点内有多个 rank）。
        if data_parallel_external_lb:
            if self.data_parallel_rank is None:
                raise ValueError(
                    "Invalid data-parallel launch options: "
                    "`--data-parallel-external-lb` requires a data-parallel "
                    "rank. Set `--data-parallel-rank`, or set "
                    "`--data-parallel-size` greater than 1 and use `--nnodes` "
                    "with `--node-rank` so the rank can be inferred."
                )
            if self.data_parallel_size_local not in (1, None):
                raise ValueError(
                    "Invalid data-parallel launch options: an external "
                    "data-parallel rank requires `--data-parallel-size-local "
                    f"1`; got {self.data_parallel_size_local}. Set it to 1 or "
                    "omit it."
                )
            data_parallel_size_local = 1
            # Use full external lb if we have local_size of 1.
            self.data_parallel_hybrid_lb = False
        elif self.data_parallel_size_local is not None:
            # [CN] 用户显式给了 --data-parallel-size-local（本节点的 rank 数）。
            data_parallel_size_local = self.data_parallel_size_local

            if self.data_parallel_start_rank is not None and not headless:
                # Infer hybrid LB mode.
                # [CN] 显式给了起始 rank 说明用户自己在编排 rank 分布 —— 判定为 hybrid LB。
                #      headless 模式（只跑 engine 不跑 API server）下不做此推断。
                self.data_parallel_hybrid_lb = True

            if self.data_parallel_hybrid_lb and data_parallel_size_local == 1:
                # Use full external lb if we have local_size of 1.
                # [CN] hybrid 的前提是"节点内有多个 rank 需要内部再分发"；
                #      若节点内只有 1 个 rank，hybrid 就退化成了 external，自动降级。
                logger.warning(
                    "data_parallel_hybrid_lb is not eligible when "
                    "data_parallel_size_local = 1, autoswitch to "
                    "data_parallel_external_lb."
                )
                data_parallel_external_lb = True
                self.data_parallel_hybrid_lb = False

            if data_parallel_size_local == self.data_parallel_size:
                # Disable hybrid LB mode if set for a single node
                # [CN] 本地规模 == 全局规模，说明实际只有一个节点，hybrid 无意义。
                self.data_parallel_hybrid_lb = False

            # [CN] 优先用用户给的 start_rank，否则用前面按 node_rank 推断出的值。
            self.data_parallel_rank = (
                self.data_parallel_start_rank
                if self.data_parallel_start_rank is not None
                else inferred_data_parallel_rank
            )
            if self.nnodes > 1:
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d",
                    self.data_parallel_rank,
                    self.node_rank,
                )
        else:
            # [CN] 既非 external、又没指定 local size 的兜底分支（最常见的单节点场景）。
            if self.data_parallel_hybrid_lb:
                raise ValueError(
                    "Invalid data-parallel launch options: "
                    "`--data-parallel-hybrid-lb` requires "
                    "`--data-parallel-size-local`. Set it to the number of "
                    "data-parallel ranks on this node."
                )

            if self.data_parallel_backend == "ray" and (
                envs.VLLM_RAY_DP_PACK_STRATEGY == "span"
            ):
                # Data parallel size defaults to 1 if DP ranks are spanning
                # multiple nodes
                # [CN] "span" 策略下 DP rank 会被打散到多个节点，
                #      本节点内可能不足一个完整 DP rank，故本地规模记为 1。
                data_parallel_size_local = 1
            else:
                # Otherwise local DP size defaults to global DP size if not set
                # [CN] 单节点默认：本地规模 = 全局 DP 规模。
                data_parallel_size_local = self.data_parallel_size

        # DP address, used in multi-node case for torch distributed group
        # and ZMQ sockets.
        # [CN] DP 主地址用于建立 torch 分布式通信组与 ZMQ socket。
        #      ray 后端用本机 IP（ray 自己管理节点发现）；mp 后端退回 master_addr 或默认值。
        if self.data_parallel_address is None:
            if self.data_parallel_backend == "ray":
                host_ip = get_ip()
                logger.info(
                    "Using host IP %s as ray-based data parallel address", host_ip
                )
                data_parallel_address = host_ip
            else:
                assert self.data_parallel_backend == "mp", (
                    "data_parallel_backend can only be ray or mp, got %s",
                    self.data_parallel_backend,
                )
                data_parallel_address = (
                    self.master_addr or ParallelConfig.data_parallel_master_ip
                )
        else:
            data_parallel_address = self.data_parallel_address

        # This port is only used when there are remote data parallel engines,
        # otherwise the local IPC transport is used.
        # [CN] 只有存在"远端" DP engine（跨节点/跨进程）时才需要 TCP 端口；
        #      同机同进程组直接用 IPC，不占端口。
        data_parallel_rpc_port = (
            self.data_parallel_rpc_port
            if (self.data_parallel_rpc_port is not None)
            else ParallelConfig.data_parallel_rpc_port
        )

        # [CN] tokens_only 模式：输入已是 token id，强制跳过 tokenizer 初始化。
        #      注意这里是【就地修改已构造好的 model_config】——因为 skip_tokenizer_init
        #      要影响后续 layout，来不及回到 create_model_config 里改。
        if self.tokens_only and not model_config.skip_tokenizer_init:
            model_config.skip_tokenizer_init = True
            logger.info("Skipping tokenizer initialization for tokens-only mode.")

        # [CN] ParallelConfig 依赖 ModelConfig（is_moe、以及下面要用到 model_config 的
        #      若干字段）与上面推导好的 DP 拓扑，因此必须排在两者之后构造。
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            prefill_context_parallel_size=self.prefill_context_parallel_size,
            data_parallel_size=self.data_parallel_size,
            data_parallel_rank=self.data_parallel_rank or 0,
            data_parallel_external_lb=data_parallel_external_lb,
            data_parallel_size_local=data_parallel_size_local,
            master_addr=self.master_addr,
            master_port=self.master_port,
            nnodes=self.nnodes,
            node_rank=self.node_rank,
            distributed_timeout_seconds=self.distributed_timeout_seconds,
            cpu_distributed_timeout_seconds=self.cpu_distributed_timeout_seconds,
            data_parallel_master_ip=data_parallel_address,
            data_parallel_rpc_port=data_parallel_rpc_port,
            data_parallel_backend=self.data_parallel_backend,
            data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,
            is_moe_model=model_config.is_moe,
            enable_expert_parallel=self.enable_expert_parallel,
            enable_batch_sharded_sampling=self.enable_batch_sharded_sampling,
            enable_ep_weight_filter=self.enable_ep_weight_filter,
            all2all_backend=self.all2all_backend,
            enable_elastic_ep=self.enable_elastic_ep,
            enable_dbo=self.enable_dbo,
            ubatch_size=self.ubatch_size,
            dbo_decode_token_threshold=self.dbo_decode_token_threshold,
            dp_sync_interval=self.dp_sync_interval,
            dbo_prefill_token_threshold=self.dbo_prefill_token_threshold,
            disable_nccl_for_dp_synchronization=self.disable_nccl_for_dp_synchronization,
            enable_eplb=self.enable_eplb,
            eplb_config=self.eplb_config,
            expert_placement_strategy=self.expert_placement_strategy,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            ray_runtime_env=ray_runtime_env,
            placement_group=placement_group,
            distributed_executor_backend=self.distributed_executor_backend,
            worker_cls=self.worker_cls,
            worker_extension_cls=self.worker_extension_cls,
            decode_context_parallel_size=self.decode_context_parallel_size,
            dcp_comm_backend=self.dcp_comm_backend,
            dcp_q_replicate=self.dcp_q_replicate,
            dcp_kv_cache_interleave_size=self.dcp_kv_cache_interleave_size,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            _api_process_count=self._api_process_count,
            _api_process_rank=self._api_process_rank,
            assigned_physical_gpu_ids=self._resolve_device_ids(),
            enable_fault_tolerance=self.enable_fault_tolerance,
            fault_tolerance_config=self.fault_tolerance_config,
            numa_bind=self.numa_bind,
            numa_bind_nodes=self.numa_bind_nodes,
            numa_bind_cpus=self.numa_bind_cpus,
        )

        speculative_config = self.create_speculative_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )
        diffusion_config = self.create_diffusion_config()

        self._set_default_max_num_seqs_and_batched_tokens_args(
            usage_context,
            model_config,
            parallel_config,
        )

        # [CN] 这四个断言是"哨兵已消除"的契约检查：前面三个字段初始为 None，
        #      必须由 _set_default_* 系列方法填入真实值；max_model_len 则应由
        #      ModelConfig 解析完成。它们不是给用户的入参校验，而是防止
        #      后续新增代码路径时漏掉默认值推导步骤的内部一致性保护。
        assert self.max_num_batched_tokens is not None, (
            "max_num_batched_tokens must be set by this point"
        )
        assert self.max_num_seqs is not None, "max_num_seqs must be set by this point"
        assert self.enable_chunked_prefill is not None, (
            "enable_chunked_prefill must be set by this point"
        )
        assert model_config.max_model_len is not None, (
            "max_model_len must be set by this point"
        )
        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            max_num_seqs=self.max_num_seqs,
            max_num_queued_reqs=self.max_num_queued_reqs,
            max_num_queued_tokens=self.max_num_queued_tokens,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
            is_multimodal_model=model_config.is_multimodal_model,
            is_encoder_decoder=model_config.is_encoder_decoder,
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            scheduler_reserve_full_isl=self.scheduler_reserve_full_isl,
            watermark=self.watermark,
            prefill_schedule_interval=self.prefill_schedule_interval,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
            async_scheduling=self.async_scheduling,
            stream_interval=self.stream_interval,
        )

        if not model_config.is_multimodal_model and self.default_mm_loras:
            raise ValueError(
                "Default modality-specific LoRA(s) were provided for a "
                "non multimodal model"
            )

        lora_config = (
            LoRAConfig(
                max_lora_rank=self.max_lora_rank,
                max_loras=self.max_loras,
                default_mm_loras=self.default_mm_loras,
                fully_sharded_loras=self.fully_sharded_loras,
                lora_dtype=self.lora_dtype,
                target_modules=self.lora_target_modules,
                enable_tower_connector_lora=self.enable_tower_connector_lora,
                specialize_active_lora=self.specialize_active_lora,
                enable_mixed_moe_lora_format=self.enable_mixed_moe_lora_format,
                enable_moe_shared_loras=self.enable_moe_shared_loras,
                max_cpu_loras=self.max_cpu_loras
                if self.max_cpu_loras and self.max_cpu_loras > 0
                else None,
            )
            if self.enable_lora
            else None
        )

        # LoRA + 投机解码的交叉约束：一个 step 内必须能放下所有序列的
        # (1 + num_speculative_tokens) 个 token，否则调度会永远凑不满一批。
        if (
            lora_config is not None
            and speculative_config is not None
            and scheduler_config.max_num_batched_tokens
            < (
                scheduler_config.max_num_seqs
                * (speculative_config.num_speculative_tokens + 1)
            )
        ):
            raise ValueError(
                "Consider increasing max_num_batched_tokens or "
                "decreasing num_speculative_tokens"
            )

        # ------------------------------------------------------------------
        # 以下若干段是同一模式的重复："顶层扁平参数" 覆盖 "嵌套 config 字段"。
        # 二者互斥，重复指定抛 ValueError；copy.deepcopy 保证不改用户传入的对象。
        # ------------------------------------------------------------------
        # Attention config overrides
        attention_config = copy.deepcopy(self.attention_config)
        if self.attention_backend is not None:
            if attention_config.backend is not None:
                raise ValueError(
                    "attention_backend and attention_config.backend "
                    "are mutually exclusive"
                )
            # Reuse the validator to handle "auto" and string-to-enum conversion
            attention_config.backend = AttentionConfig.validate_backend_before(
                self.attention_backend
            )

        # Batch-invariant mode requires deterministic attention behavior.
        # If no backend is explicitly requested, prefer Triton Attention.
        if (
            envs.VLLM_BATCH_INVARIANT
            and attention_config.backend is None
            and current_platform.is_xpu()
        ):
            attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info(
                "VLLM_BATCH_INVARIANT is enabled and no attention backend was "
                "specified; defaulting to TRITON_ATTN."
            )

        # TurboQuant requires FlashAttention 2 — FA3 boundary layers assert
        # FlashAttentionImpl which fails with TurboQuantAttentionImpl.
        if resolved_cache_dtype.startswith("turboquant_") and (
            attention_config.flash_attn_version is None
            or attention_config.flash_attn_version >= 3
        ):
            logger.warning(
                "TurboQuant is not yet compatible with FlashAttention >= 3. "
                "Overriding flash_attn_version to 2. To silence this "
                "warning, pass --attention-config.flash_attn_version=2"
            )
            attention_config.flash_attn_version = 2

        # Mamba config overrides
        # [CN] 下面三组（mamba / kernel / compilation）用的是同一个模式：
        #      先 deepcopy 出一份顶层子配置的副本，再把 EngineArgs 上那些"被提升过的
        #      扁平字段"逐个覆盖回去。之所以要 deepcopy：self.mamba_config 可能来自
        #      默认值（各调用方共享的同一实例），直接改会污染其他实例。
        #      覆盖时统一遵守"只有非 None / 非 auto 才覆盖"，以便区分"显式设置"与"未指定"。
        mamba_config = copy.deepcopy(self.mamba_config)
        # Convert string to enum if needed (CLI parsing returns a string)
        if isinstance(self.mamba_backend, str):
            mamba_config.backend = MambaBackendEnum[self.mamba_backend.upper()]
        else:
            mamba_config.backend = self.mamba_backend
        if self.mamba_ssu_algorithm is not None:
            mamba_config.ssu_algorithm = self.mamba_ssu_algorithm
        if self.enable_mamba_cache_stochastic_rounding:
            mamba_config.enable_stochastic_rounding = (
                self.enable_mamba_cache_stochastic_rounding
            )
        if self.mamba_cache_philox_rounds:
            mamba_config.stochastic_rounding_philox_rounds = (
                self.mamba_cache_philox_rounds
            )
        mamba_config.validate_ssu_algorithm()

        # Kernel config overrides
        # [CN] 互斥性校验的意义：enable_flashinfer_autotune 同时存在于
        #      EngineArgs（扁平字段）和 KernelConfig（嵌套字段）两侧，
        #      两边都设了就无法判断以谁为准，因此直接报错而不是"后者覆盖前者"。
        kernel_config = copy.deepcopy(self.kernel_config)
        if self.enable_flashinfer_autotune is not None:
            if kernel_config.enable_flashinfer_autotune is not None:
                raise ValueError(
                    "enable_flashinfer_autotune and "
                    "kernel_config.enable_flashinfer_autotune "
                    "are mutually exclusive"
                )
            kernel_config.enable_flashinfer_autotune = self.enable_flashinfer_autotune
        if self.enable_bf16x3_router_gemm is not None:
            kernel_config.enable_bf16x3_router_gemm = self.enable_bf16x3_router_gemm
        if self.moe_backend != "auto":
            kernel_config.moe_backend = self.moe_backend
        if self.linear_backend != "auto":
            kernel_config.linear_backend = self.linear_backend

        # Transfer top-level ir_op_priority into KernelConfig.ir_op_priority
        for op_name, op_priority in asdict(self.ir_op_priority).items():
            # Empty means unset
            if not op_priority:
                continue

            # Priority cannot be set 2x for the same op
            if getattr(kernel_config.ir_op_priority, op_name):
                raise ValueError(
                    f"Op priority for {op_name} specified via both ir_op_priority "
                    f"and KernelConfig.ir_op_priority, only one allowed at a time."
                )

            # Set the attribute
            setattr(kernel_config.ir_op_priority, op_name, op_priority)

        load_config = self.create_load_config()

        # Pass reasoning_parser into StructuredOutputsConfig
        # [CN] 同样是"扁平字段回填嵌套配置"：reasoning_parser / reasoning_parser_plugin
        #      在 EngineArgs 上是顶层字段，但语义上属于 StructuredOutputsConfig。
        #      注意这里改的是 self.structured_outputs_config 本身（就地修改）。
        if self.reasoning_parser:
            self.structured_outputs_config.reasoning_parser = self.reasoning_parser

        if self.reasoning_parser_plugin:
            self.structured_outputs_config.reasoning_parser_plugin = (
                self.reasoning_parser_plugin
            )

        observability_config = self.create_observability_config()

        # Compilation config overrides
        compilation_config = copy.deepcopy(self.compilation_config)
        if self.cudagraph_capture_sizes is not None:
            if compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "cudagraph_capture_sizes and compilation_config."
                    "cudagraph_capture_sizes are mutually exclusive"
                )
            compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
        if self.max_cudagraph_capture_size is not None:
            if compilation_config.max_cudagraph_capture_size is not None:
                raise ValueError(
                    "max_cudagraph_capture_size and compilation_config."
                    "max_cudagraph_capture_size are mutually exclusive"
                )
            compilation_config.max_cudagraph_capture_size = (
                self.max_cudagraph_capture_size
            )

        # [CN] 三种卸载策略在此合体：OffloadConfig 是外壳，内部挂 UVA（统一虚拟寻址，
        #      把权重映射到 CPU 内存按需换页）与 Prefetch（按组分批预取）两套具体策略。
        offload_config = OffloadConfig(
            offload_backend=self.offload_backend,
            uva=UVAOffloadConfig(
                cpu_offload_gb=self.cpu_offload_gb,
                cpu_offload_params=self.cpu_offload_params,
            ),
            prefetch=PrefetchOffloadConfig(
                offload_group_size=self.offload_group_size,
                offload_num_in_group=self.offload_num_in_group,
                offload_prefetch_step=self.offload_prefetch_step,
                offload_params=self.offload_params,
            ),
        )

        # 把 gdn/kda 的 prefill 后端选择塞进 additional_config：它们属于实验性
        # 后端开关，没有进入任何正式子配置，只能走这个"杂项口袋"。
        if self.gdn_prefill_backend is not None:
            self.additional_config["gdn_prefill_backend"] = self.gdn_prefill_backend
        if self.kda_prefill_backend is not None:
            self.additional_config["kda_prefill_backend"] = self.kda_prefill_backend

        # 最终汇总：所有子配置在此合体。注意传入的都是"已完成推导与覆盖"的对象，
        # VllmConfig 自身的 __post_init__ 还会再做一轮跨配置一致性校验。
        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_config,
            offload_config=offload_config,
            attention_config=attention_config,
            mamba_config=mamba_config,
            kernel_config=kernel_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            diffusion_config=diffusion_config,
            structured_outputs_config=self.structured_outputs_config,
            observability_config=observability_config,
            compilation_config=compilation_config,
            kv_transfer_config=self.kv_transfer_config,
            kv_events_config=self.kv_events_config,
            ec_transfer_config=self.ec_transfer_config,
            ec_manager_config=self.ec_manager_config,
            reasoning_config=self.reasoning_config,
            profiler_config=self.profiler_config,
            additional_config=self.additional_config,
            optimization_level=self.optimization_level,
            performance_mode=self.performance_mode,
            weight_transfer_config=self.weight_transfer_config,
            shutdown_timeout=self.shutdown_timeout,
        )

        return config

    def _check_feature_supported(self):
        """Raise an error if the feature is not supported."""
        # 中文补充：目前只校验流水线并行（PP）——自定义 executor backend 必须显式
        # 声明 supports_pp，或落在 ray / mp / external_launcher 这几个已知后端内。
        if self.pipeline_parallel_size > 1:
            supports_pp = getattr(
                self.distributed_executor_backend, "supports_pp", False
            )
            if not supports_pp and self.distributed_executor_backend not in (
                ParallelConfig.distributed_executor_backend,
                "ray",
                "mp",
                "external_launcher",
            ):
                name = (
                    "Pipeline Parallelism without Ray distributed "
                    "executor or multiprocessing executor or external "
                    "launcher"
                )
                _raise_unsupported_error(feature_name=name)

    @classmethod
    def get_batch_defaults(
        cls,
        world_size: int,
    ) -> tuple[dict[UsageContext | None, int], dict[UsageContext | None, int]]:
        from vllm.usage.usage_lib import UsageContext

        # 两个返回值都是 "usage context -> 默认值" 的映射：离线 LLM 类与在线 OpenAI
        # API server 的负载特征不同（后者请求更碎、并发更高），所以默认值不同。
        default_max_num_batched_tokens: dict[UsageContext | None, int]
        default_max_num_seqs: dict[UsageContext | None, int]

        # When no user override, set the default values based on the usage
        # context.
        # Use different default values for different hardware.

        # Try to query the device name on the current platform. If it fails,
        # it may be because the platform that imports vLLM is not the same
        # as the platform that vLLM is running on (e.g. the case of scaling
        # vLLM with Ray) and has no GPUs. In this case we use the default
        # values for non-H100/H200 GPUs.
        try:
            device_memory = current_platform.get_device_total_memory()
            device_name = current_platform.get_device_name().lower()
        except Exception:
            # This is only used to set default_max_num_batched_tokens
            device_memory = 0
            device_name = ""

        # NOTE(Kuntai): Setting large `max_num_batched_tokens` for A100 reduces
        # throughput, see PR #17885 for more details.
        # So here we do an extra device name check to prevent such regression.
        # 按显存容量分档选默认值，而不是按型号硬编码：>=160GB（B200/B300 档）取最大，
        # >=70GB 且非 A100（H100/H200 档）次之，其余（含 A100）取保守值。
        if device_memory >= 160 * GiB_bytes:
            # for GPUs like B200/B300 with >= 160GB memory, use the largest defaults
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 16384,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 1024,
                UsageContext.OPENAI_API_SERVER: 1024,
            }
        elif device_memory >= 70 * GiB_bytes and "a100" not in device_name:
            # For GPUs like H100 and H200, use larger offline defaults.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 1024,
                UsageContext.OPENAI_API_SERVER: 1024,
            }
        else:
            # TODO(woosuk): Tune the default values for other hardware.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256,
                UsageContext.OPENAI_API_SERVER: 256,
            }

        # TPU / CPU 平台没有上面的分档逻辑，直接按芯片型号覆盖；且它们的默认值
        # 与 world_size 成正比（每个 rank 一份预算）。
        # tpu specific default values.
        if current_platform.is_tpu():
            chip_name = current_platform.get_device_name()

            if chip_name == "V6E":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 2048,
                    UsageContext.OPENAI_API_SERVER: 1024,
                }
            elif chip_name == "V5E":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 1024,
                    UsageContext.OPENAI_API_SERVER: 512,
                }
            elif chip_name == "V5P":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 512,
                    UsageContext.OPENAI_API_SERVER: 256,
                }

        # cpu specific default values.
        if current_platform.is_cpu():
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 4096 * world_size,
                UsageContext.OPENAI_API_SERVER: 2048 * world_size,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256 * world_size,
                UsageContext.OPENAI_API_SERVER: 128 * world_size,
            }

        return default_max_num_batched_tokens, default_max_num_seqs

    def _set_default_chunked_prefill_and_prefix_caching_args(
        self, model_config: ModelConfig
    ) -> None:
        """消除 enable_chunked_prefill / enable_prefix_caching 的哨兵（None）。

        规则：
          - 未指定时采用"模型是否支持"（ModelConfig 上的能力探测结果）；
          - 显式指定但与模型能力冲突时只 warning 不报错（尊重用户，风险自负）；
          - RISC-V CPU 上两者被无条件关闭（V1 后端不支持）。
        """
        default_chunked_prefill = model_config.is_chunked_prefill_supported
        default_prefix_caching = model_config.is_prefix_caching_supported

        if self.enable_chunked_prefill is None:
            self.enable_chunked_prefill = default_chunked_prefill

            logger.debug(
                "%s chunked prefill by default",
                "Enabling" if default_chunked_prefill else "Disabling",
            )
        elif (
            model_config.runner_type == "generate"
            and not self.enable_chunked_prefill
            and default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support disabling chunked prefill. "
                "Disabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_chunked_prefill
            and not default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support chunked prefill. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
            )

        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = default_prefix_caching

            logger.debug(
                "%s prefix caching by default",
                "Enabling" if default_prefix_caching else "Disabling",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_prefix_caching
            and not default_prefix_caching
        ):
            logger.warning_once(
                "This model does not officially support prefix caching. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
            )

        # Disable chunked prefill and prefix caching for:
        # RISCV CPUs in V1
        if current_platform.is_cpu() and current_platform.get_cpu_architecture() in (
            CpuArchEnum.RISCV,
        ):
            logger.info(
                "Chunked prefill is not supported for"
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_chunked_prefill = False
            logger.info(
                "Prefix caching is not supported for "
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_prefix_caching = False

    def _set_default_reasoning_config_args(self):
        # reasoning_parser 是 StructuredOutputsConfig 字段的顶层快捷别名；
        # 传了它就必须保证 reasoning_config 存在（否则新建）。
        if not self.reasoning_parser:
            return
        if self.reasoning_config is None:
            self.reasoning_config = ReasoningConfig()
        self.reasoning_config.reasoning_parser = self.reasoning_parser

    @staticmethod
    def _get_min_mm_batched_tokens(
        model_config: ModelConfig,
    ) -> tuple[int, str] | None:
        """Get the minimum max_num_batched_tokens needed for a multimodal
        prefix-LM model to process at least one item of any supported modality.

        Returns (token_count, modality_name) for the most expensive modality,
        or None if the value cannot be determined at this stage.

        中文补充：这里刻意做了"降级"——解析失败或信息不足时返回 None 由调用方
        忽略，绝不能因为算不出一个默认值就让引擎启动失败。
        """
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            # get_processing_info returns the model's multimodal processing
            # metadata (supported modalities, token limits) without loading
            # model weights or generating dummy data.
            info = MULTIMODAL_REGISTRY.get_processing_info(model_config)
            mm_counts = {modality: 1 for modality in info.supported_mm_limits}
            # get_mm_max_tokens_per_item returns pre-computed per-item token
            # ceilings for models that override it (e.g., Gemma4), or None
            # for models that rely on dummy-input profiling. When None is
            # returned we bail out — no dummy generation is triggered here.
            max_tokens = info.get_mm_max_tokens_per_item(
                seq_len=model_config.max_model_len,
                mm_counts=mm_counts,
            )
            if max_tokens is not None:
                modality = max(max_tokens, key=max_tokens.__getitem__)
                return (max_tokens[modality], modality)
        except Exception as e:
            logger.warning("Failed to determine min multimodal batched tokens: %s", e)
        return None

    def _set_default_max_num_seqs_and_batched_tokens_args(
        self,
        usage_context: UsageContext | None,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ):
        """推导 max_num_batched_tokens 与 max_num_seqs 的默认值。

        推导次序（后面的约束会覆盖前面的）：
          1. 按 usage context + 硬件档位取基线（get_batch_defaults）；
          2. batched DP MoE 走专属常量；
          3. performance_mode == "throughput" 时把默认值翻倍（仅对"未显式指定"的生效）；
          4. 关闭 chunked prefill 时，一批至少要装下 max_model_len 个 token；
          5. 多模态 prefix-LM 需抬到"单个最大多模态 item"的 token 数；
          6. 上限不超过 max_num_seqs * max_model_len。

        Note:
            orig_* 两个局部变量用来记住"用户是否显式指定过"，是判断能否被自动
            调整（翻倍、取 max/min）的依据。
        """
        world_size = self.pipeline_parallel_size * self.tensor_parallel_size
        (
            default_max_num_batched_tokens,
            default_max_num_seqs,
        ) = self.get_batch_defaults(world_size)

        orig_max_num_batched_tokens = self.max_num_batched_tokens
        orig_max_num_seqs = self.max_num_seqs

        if self.max_num_batched_tokens is None:
            if parallel_config.use_batched_dp_moe:
                self.max_num_batched_tokens = (
                    SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP
                )
            else:
                self.max_num_batched_tokens = default_max_num_batched_tokens.get(
                    usage_context,
                    SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
                )

        if self.max_num_seqs is None:
            self.max_num_seqs = default_max_num_seqs.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_SEQS,
            )

        # If throughput mode is set, double max_num_batched_tokens and max_num_seqs.
        if self.performance_mode == "throughput":
            if orig_max_num_batched_tokens is None:
                self.max_num_batched_tokens *= 2
            if orig_max_num_seqs is None:
                self.max_num_seqs *= 2

        if orig_max_num_batched_tokens is None:
            assert model_config.max_model_len is not None, (
                "max_model_len must be set by this point"
            )
            if not self.enable_chunked_prefill:
                # If max_model_len is too short, use the default for higher throughput.
                self.max_num_batched_tokens = max(
                    model_config.max_model_len,
                    self.max_num_batched_tokens,
                )

            # For multimodal prefix-LM models (e.g., Gemma 4) that disable
            # chunked MM input, a single multimodal item must fit in one batch.
            # Raise the floor to accommodate the largest per-item token count.
            if model_config.is_multimodal_model and model_config.is_mm_prefix_lm:
                result = self._get_min_mm_batched_tokens(model_config)
                if result is not None and result[0] > self.max_num_batched_tokens:
                    mm_min, modality = result
                    logger.info(
                        "Raising max_num_batched_tokens from %d to %d to "
                        "accommodate '%s' input for prefix-LM model %s.",
                        self.max_num_batched_tokens,
                        mm_min,
                        modality,
                        model_config.model,
                    )
                    self.max_num_batched_tokens = mm_min

            # When using default settings,
            # Ensure max_num_batched_tokens does not exceed model limit.
            # Some models (e.g., Whisper) have embeddings tied to max length.
            self.max_num_batched_tokens = min(
                self.max_num_seqs * model_config.max_model_len,
                self.max_num_batched_tokens,
            )

            logger.debug(
                "Defaulting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens,
                usage_context.value if usage_context else None,
            )

        if orig_max_num_seqs is None:
            # 只用默认值时保证不会出现"序列数 > 每批 token 数"这种无意义组合：
            # 每条序列至少要分到 1 个 token。
            assert self.max_num_batched_tokens is not None  # For type checking
            self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

            logger.debug(
                "Defaulting max_num_seqs to %d for %s usage context.",
                self.max_num_seqs,
                usage_context.value if usage_context else None,
            )


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    # 中文补充：仅比 EngineArgs 多一个 enable_log_requests；其余全部继承，
    # 包括 add_cli_args（默认先调用父类注册全部参数，再追加自己的）。

    enable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
        parser: FlexibleArgumentParser, async_args_only: bool = False
    ) -> FlexibleArgumentParser:
        # Initialize plugin to update the parser, for example, The plugin may
        # add a new kind of quantization method to --quantization argument or
        # a new device to --device argument.
        # 中文补充：插件必须先于参数注册加载，因为它可能往 choices 里追加选项。
        load_general_plugins()
        if not async_args_only:
            parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--enable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=AsyncEngineArgs.enable_log_requests,
            help="Enable logging request information, dependent on log level:\n"
            "- INFO: Request ID, parameters and LoRA request.\n"
            "- DEBUG: Prompt inputs (e.g: text, token IDs).\n"
            "You can set the minimum log level via `VLLM_LOGGING_LEVEL`.",
        )
        current_platform.pre_register_and_update(parser)
        return parser


def _raise_unsupported_error(feature_name: str):
    """统一的"能力不支持"报错入口：所有平台/后端不支持的组合都走这里。"""
    msg = (
        f"{feature_name} is not supported. We recommend to "
        f"remove {feature_name} from your config."
    )
    raise NotImplementedError(msg)
