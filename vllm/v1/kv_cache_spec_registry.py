# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Registry for KVCacheSpec types and their associated managers.

This module provides a pluggable architecture for registering custom KVCacheSpec
subclasses without modifying vLLM core code. Out-of-tree platforms can define
custom specs and managers by using the @register_kv_cache_spec decorator.
"""

# [CN] KVCacheSpec 类型与其管理器的**注册表**。
#      存在的意义：让**树外平台**（out-of-tree，比如某款国产加速卡）
#      能够注册自己的 KV cache 规格与分配器，而不用改 vLLM 核心代码。
#      这是典型的"插件化注册表"模式。
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheSpec


# [CN] 一条注册记录：spec 类 + 它的管理器类 + 用于分组的基类。
#      frozen dataclass：注册后不可变，避免运行期被意外改写。
@dataclass(frozen=True)
class KVCacheSpecMetadata:
    """Metadata for a registered KVCacheSpec."""

    kvcache_spec_cls: type["KVCacheSpec"]
    manager_class: type["SingleTypeKVCacheManager"]
    # [CN] **分组基类**：两个 spec 的 uniform_type_base_spec 相同，
    #      就可以被归到同一个 KV cache group（共享同一套 block）。
    #      例：几种不同的 FullAttentionSpec 子类都指向 FullAttentionSpec，
    #      于是它们共用一个 group；而 MambaSpec 必须单独一组。
    # The base spec class for grouping compatibility checks.
    # KVCacheSpecs with the same uniform_type_base_spec will be
    # grouped into one kvcache group
    uniform_type_base_spec: type["KVCacheSpec"]


# [CN] 全局注册表本体。注意它是模块级 dict，而类方法都是 @classmethod ——
#      也就是说"注册表"是全局单例，类只是操作它的命名空间。
_REGISTRY_KVCACHESPEC_LIST: dict[type["KVCacheSpec"], KVCacheSpecMetadata] = {}


class KVCacheSpecRegistry:
    """Global registry for KVCacheSpec types and their associated managers."""

    # [CN] **懒注册** ：第一次用到时才真正执行所有 @register_kv_cache_spec。
    #      为什么必须懒：注册时要 import single_type_kv_cache_manager，
    #      而那个模块又要 import 这里 —— 直接 import 会循环依赖。
    #      判空条件用 `_REGISTRY_KVCACHESPEC_LIST` 是否为空，
    #      这是一个轻量但足够用的"已初始化"标志。
    @classmethod
    def _ensure_registered(cls, vllm_config=None) -> None:
        """
        Run full KVCacheSpec registration if the registration is not done.
        """
        if _REGISTRY_KVCACHESPEC_LIST:
            return

        if vllm_config is None:
            from vllm.config import get_current_vllm_config_or_none

            vllm_config = get_current_vllm_config_or_none()

        # lazy import to avoid circular dependency
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )

        register_all_kvcache_specs(vllm_config)

    # [CN] 注册入口。两个 assert 值得注意：
    #        - manager_class 必填（没有管理器就没人会分配这种 KV）；
    #        - spec 必须真的继承自它声明的 base spec（防止乱配对）。
    #      重复注册时**允许**，但要求两次注册完全一致，否则 assert 失败 ——
    #      这样模块被重复 import 不会炸，但真冲突能立刻暴露。
    @classmethod
    def register(
        cls,
        kvcache_spec_cls: type["KVCacheSpec"],
        manager_class: type["SingleTypeKVCacheManager"] | None = None,
        uniform_type_base_spec: type["KVCacheSpec"] | None = None,
    ) -> None:
        """
        Register a KVCacheSpec class with its manager and base spec.

        Args:
            kvcache_spec_cls: The KVCacheSpec subclass to register
            manager_class: The SingleTypeKVCacheManager to use for this spec
            uniform_type_base_spec: The base spec class for grouping compatibility.
                instead of being grouped to different kvcache group, `kvcache_spec_cls`
                and `uniform_type_base_spec` will be trated as uniform type.
                If None, defaults to kvcache_spec_cls itself (for built-in base specs).
        """
        assert manager_class is not None, "manager_class is required"
        if uniform_type_base_spec is None:
            uniform_type_base_spec = kvcache_spec_cls
        assert issubclass(kvcache_spec_cls, uniform_type_base_spec), (
            f"{kvcache_spec_cls.__name__} must inherit from its declared "
            f"uniform_type_base_spec {uniform_type_base_spec.__name__}."
        )

        if kvcache_spec_cls in _REGISTRY_KVCACHESPEC_LIST:
            registered_spec = _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls]
            is_same_registration = (
                manager_class == registered_spec.manager_class
                and uniform_type_base_spec == registered_spec.uniform_type_base_spec
            )
            assert is_same_registration, (
                f"Conflicting registration for KVCacheSpec "
                f": {kvcache_spec_cls.__name__}"
            )

        _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls] = KVCacheSpecMetadata(
            kvcache_spec_cls=kvcache_spec_cls,
            manager_class=manager_class,
            uniform_type_base_spec=uniform_type_base_spec,
        )

    # [CN] 查管理器。
    @classmethod
    def get_manager_class(
        cls, kvcache_spec: "KVCacheSpec"
    ) -> type["SingleTypeKVCacheManager"] | None:
        """
        Get the single type kvcache manager class for a given kvcache spec instance.

        Args:
            kvcache_spec: A KVCacheSpec instance

        Returns:
            The SingleTypeKVCacheManager class to use for this kvcache_spec
        """
        cls._ensure_registered()
        kvcache_spec_cls = type(kvcache_spec)

        # [CN] **沿 MRO 向上找**：如果某个自定义 spec 自己没注册，
        #      但它的父类注册过，就用父类的管理器。
        #      这让"只覆盖少量行为的子类"无需重复注册。
        # Walk up the MRO to find a registered base class
        for base in kvcache_spec_cls.__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].manager_class

        return None

    # [CN] 查分组基类（用途见上面的 uniform_type_base_spec 注释）。
    @classmethod
    def get_uniform_type_base_spec(
        cls, kvcache_spec: "KVCacheSpec"
    ) -> type["KVCacheSpec"] | None:
        """
        Get the base kvcache spec class for grouping compatibility checks.
        KVCacheSpecs with uniform_type_base_spec will be trated as one group.

        Args:
            kvcache_spec: A KVCacheSpec instance

        Returns:
            The base KVCacheSpec class for checking uniform type kvcache specs
        """
        cls._ensure_registered()
        kvcache_spec_cls = type(kvcache_spec)

        # Walk up the MRO to find a registered base spec
        for base in kvcache_spec_cls.__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].uniform_type_base_spec

        return None

    # [CN] 启动期自检：确保每一层的 spec 都注册了管理器和分组基类。
    @classmethod
    def check_kv_cache_spec_registry(
        cls, kv_cache_spec: dict[str, "KVCacheSpec"]
    ) -> None:
        """
        Check if the KVCacheSpecs of each layer are registered as expected.
        """
        cls._ensure_registered()
        for layer_name, spec in kv_cache_spec.items():
        # [CN] 这里用 raise 而不是 assert —— 注释里写得很清楚：
        #      python -O 模式下 assert 会被整体去掉，
        #      生产环境里就检查不到未注册的类型了。
            # use raise instead of assert to make it effective in production environment
            if cls.get_uniform_type_base_spec(spec) is None:
                raise ValueError(
                    f"Unsupported KV cache spec type for layer {layer_name}: "
                    f"{type(spec)}. Please register it using "
                    f"@register_kv_cache_spec decorator."
                )
            if cls.get_manager_class(spec) is None:
                raise ValueError(
                    f"No manager found for KV cache spec type for layer "
                    f"{layer_name}: {type(spec)}. Please register it using "
                    f"@register_kv_cache_spec decorator."
                )


# [CN] 供树外代码使用的装饰器，本质就是 register() 的语法糖。
def register_kv_cache_spec(
    manager_class: type["SingleTypeKVCacheManager"] | None = None,
    uniform_type_base_spec: type["KVCacheSpec"] | None = None,
):
    """
    Decorator to register a custom KVCacheSpec class.

    Args:
        manager_class: The SingleTypeKVCacheManager to use for this spec.
            Required for all registered specs.
        uniform_type_base_spec: The base spec class for uniform type kv cache specs
            compatibility. If None, the spec is treated as a new base
            type.

    Examples:
    - Register a new specs:
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec
        )
        @dataclass(frozen=True, kw_only=True)
        class CustomFullAttentionSpec(FullAttentionSpec):
            pass
    """

    def decorator(kvcache_spec_cls: type["KVCacheSpec"]) -> type["KVCacheSpec"]:
        KVCacheSpecRegistry.register(
            kvcache_spec_cls=kvcache_spec_cls,
            manager_class=manager_class,
            uniform_type_base_spec=uniform_type_base_spec,
        )
        return kvcache_spec_cls

    return decorator
