# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# [CN] 文件总览：**V1 的跨进程序列化层（基于 msgpack）**。
#
#     V1 里前端进程与 EngineCore 进程之间全靠消息通信，每条消息都要过一遍
#     序列化。请求量大时，这一层的开销直接决定吞吐上限，所以 vLLM 没有用
#     JSON 或原生 pickle，而是在 msgpack 之上做了一套**张量友好**的定制。
#
#     ===================== 三个核心设计 =====================
#     ① **张量零拷贝**：大张量不进 msgpack 主缓冲区，而是把它的底层
#        buffer 单独挂在 aux_buffers 列表里传出去（配合 ZMQ 多帧发送）。
#        主消息里只放一个**索引**。这样张量数据一次都不用拷。
#        小张量（< 阈值）反而内联，因为多帧发送本身也有开销。
#
#     ② **带外传输（OOB）扩展点**：OOBTensorConsumer / Provider 这对接口
#        允许把张量完全交给外部通道（见 tensor_ipc.py 的共享内存方案）。
#        编码器只往主消息里塞一个轻量句柄，解码器再按句柄取回真张量。
#
#     ③ **默认拒绝 pickle**：任意对象序列化（pickle）有安全风险，
#        所以默认遇到不认识的类型直接抛 TypeError；
#        只有显式设置 VLLM_ALLOW_INSECURE_SERIALIZATION=1 才启用降级。
#
#     ===================== 三个 msgpack 扩展类型 =====================
#     1 = pickle / 2 = cloudpickle / 3 = 原始字节视图。
#     自定义类型走 Ext，靠 ext_hook 在解码端还原。
#
#     另外还有两块「周边工具」：
#     · run_method：在远端对象上调用方法（用于 collective_rpc 这类场景）；
#     · PydanticMsgspecMixin：让 msgspec.Struct 能被 Pydantic 校验与序列化，
#       这样 API 层可以直接把它当请求体用。
import dataclasses
import importlib
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from inspect import isclass
from types import FunctionType
from typing import Any, ClassVar, TypeAlias, cast, get_type_hints

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    BaseMultiModalField,
    MultiModalBatchedField,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    NestedTensors,
)
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.utils import tensor_data

logger = init_logger(__name__)

# [CN] msgpack 扩展类型码：1 = pickle 序列化。
CUSTOM_TYPE_PICKLE = 1
# [CN] 2 = cloudpickle 序列化（能处理 pickle 搞不定的函数/闭包）。
CUSTOM_TYPE_CLOUDPICKLE = 2
# [CN] 3 = 原始字节视图（零拷贝小数据的载体）。
CUSTOM_TYPE_RAW_VIEW = 3

# [CN] 多模态字段类型 → 工厂方法名的映射。
#      序列化时只存名字，反序列化时按名字找回 MultiModalFieldConfig 上的工厂。
# MultiModalField class serialization type map.
# These need to list all possible field types and match them
# to factory methods in `MultiModalFieldConfig`.
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str] = {
    MultiModalFlatField: "flat",
    MultiModalSharedField: "shared",
    MultiModalBatchedField: "batched",
}

# [CN] 「字节串」的联合类型：bytes / bytearray / memoryview / zmq.Frame 都算。
bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


# [CN] 带外（OOB）张量**消费方**接口：编码器遇到张量时先问它要不要接走。
class OOBTensorConsumer(ABC):
    @abstractmethod
    # [CN] 返回 None 表示「我不要，你按常规方式序列化」；
    #      返回 dict 表示「我接走了，把这个句柄写进消息」。
    def __call__(self, tensor: torch.Tensor) -> dict | None:
        """
        Called with tensors for the current message.
        Returns None to reject the tensor (falls back to regular serialization),
        otherwise a dict with arbitrary placeholder data to be included
        in the serialized message.
        """
        return None

    @abstractmethod
    # [CN] 每条新消息开始时回调一次，让消费方重置自己的序号。
    def new_message(self) -> None:
        """Called at the start of each new encoded message."""
        pass


# [CN] 带外张量**提供方**接口：解码端按 (dtype, shape, 句柄) 取回真张量。
#      它必须与编码端的 Consumer 配对使用。
# dtype, shape, metadata -> tensor
OOBTensorProvider = Callable[[str, tuple[int, ...], dict], torch.Tensor]


def _log_insecure_serialization_warning():
    logger.warning_once(
        "Allowing insecure serialization using pickle due to "
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
    )


# [CN] 取对象的「模块名 + 限定名」二元组，用于跨进程还原类型。
def _typestr(val: Any) -> tuple[str, str] | None:
    if val is None:
        return None
    t = type(val)
    return t.__module__, t.__qualname__


# [CN] 递归地为嵌套 list/dict 里的每个叶子记录类型信息。
#      为什么要这个：UtilityResult 的内容是**弱类型**的，纯 msgpack 解回来
#      只有 dict/list/标量，靠这份类型信息才能还原成真正的类实例。
def _encode_type_info_recursive(obj: Any) -> Any:
    """Recursively encode type information for nested structures of
    lists/dicts."""
    if obj is None:
        return None
    if type(obj) is list:
        return [_encode_type_info_recursive(item) for item in obj]
    if type(obj) is dict:
        return {k: _encode_type_info_recursive(v) for k, v in obj.items()}
    return _typestr(obj)


# [CN] 与上面配对：按类型信息递归地把原始数据还原成对应类型的实例。
def _decode_type_info_recursive(
    type_info: Any, data: Any, convert_fn: Callable[[Sequence[str], Any], Any]
) -> Any:
    """Recursively decode type information for nested structures of
    lists/dicts."""
    if type_info is None:
        return data
    if isinstance(type_info, dict):
        assert isinstance(data, dict)
        return {
            k: _decode_type_info_recursive(type_info[k], data[k], convert_fn)
            for k in type_info
        }
    # [CN] 排除「看起来像 list 但其实是 (dtype, shape) 或张量」的情况，
    #      只有真正的 list 才递归下去。
    if isinstance(type_info, list) and (
        # Exclude serialized tensors/numpy arrays.
        len(type_info) != 2 or not isinstance(type_info[0], str)
    ):
        assert isinstance(data, list)
        return [
            _decode_type_info_recursive(ti, d, convert_fn)
            for ti, d in zip(type_info, data)
        ]
    return convert_fn(type_info, data)


# [CN] 包装器：标记「这个返回值需要特殊序列化处理」。
class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""

    def __init__(self, r: Any = None):
        self.result = r


# [CN] **编码器主体**。
#      注意：与原生 msgspec.Encoder 不同，它在处理张量/numpy 时**不是线程安全**的，
#      因为要用 self.aux_buffers 暂存 buffer 指针。
#
#      大小策略：小于 size_threshold 的数组内联进主消息；
#      更大的走独立帧（每个张量单独判断，不是整条消息统一判断）。
class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.

    When a ``oob_tensor_consumer`` is provided, tensors (CUDA and CPU) will be
    offered to it for out-of-band handling.
    """

    def __init__(
        self,
        size_threshold: int | None = None,
        oob_tensor_consumer: OOBTensorConsumer | None = None,
    ):
        # [CN] 阈值默认从环境变量取。
        if size_threshold is None:
            size_threshold = envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # [CN] 关键技巧：msgspec 的 enc_hook 没有传递自定义数据的通道，
        #      所以把暂存区挂在 self 上，由 hook 闭包访问。这也是**不线程安全**的根源。
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: list[bytestr] | None = None
        self.size_threshold = size_threshold
        self.oob_tensor_consumer = oob_tensor_consumer
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    # [CN] 编码：返回**一串** buffer，第 0 个是主消息，其余是张量的底层 buffer。
    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            # [CN] 新消息开始，通知 OOB 消费方。
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            # [CN] bufs[0] 占位，稍后填主消息；其余由 enc_hook 追加。
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # [CN] 这个列表让我们能直接收集张量的**底层 buffer 指针**，
            #      随主消息一起返回，从而完全避免把张量数据拷进新 buffer。
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        # [CN] 用完即清：aux_buffers 是每条消息一次性的，留着会串味。
        finally:
            self.aux_buffers = None

    # [CN] 编码到**调用方提供**的 buffer（省一次分配），语义同 encode。
    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    # [CN] 自定义类型的总入口：msgspec 遇到不认识的类型就回调这里。
    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Fall back to pickle for object or void kind ndarrays.
        # [CN] object / void 类型的 ndarray 走不了裸字节，直接降级（见下）。
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        # [CN] slice 转成 (start, stop, step) 三元组。
        #      这里假定只会用到 int 型边界。
        if isinstance(obj, slice):
            # We are assuming only int-based values will be used here.
            return tuple(
                int(v) if v is not None else None
                for v in (obj.start, obj.stop, obj.step)
            )

        if isinstance(obj, MultiModalKwargsItem):
            return self._encode_mm_item(obj)

        if isinstance(obj, MultiModalKwargsItems):
            return self._encode_mm_items(obj)

        # [CN] UtilityResult：未开启不安全序列化时**不记录类型信息**（返回 None）。
        if isinstance(obj, UtilityResult):
            result = obj.result
            # [CN] 安全优先：默认不写类型信息，等于放弃还原成自定义类。
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                return None, result
            # Since utility results are not strongly typed, we recursively
            # encode type information for nested structures of lists/dicts
            # to help with correct msgspec deserialization.
            return _encode_type_info_recursive(result), result

        # [CN] 兜底：默认直接拒绝，而不是悄悄用 pickle。
        #      pickle 反序列化可以执行任意代码，属于真实的安全风险。
        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable"
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based serialization."
            )

        # [CN] 函数走 cloudpickle：pickle 虽快但处理不了方法/闭包。
        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        # [CN] 其它对象走 pickle，用最高协议。
        return msgpack.Ext(
            CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        )

    # [CN] 编码 numpy 数组：返回 (dtype 字符串, shape, 数据)。
    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.aux_buffers is not None
        # [CN] 非连续数组必须先拷成连续的，否则底层 buffer 不是有效布局。
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        # [CN] 标量或小于阈值的数组内联。用 Ext 包装是为了解码时也能零拷贝。
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        # [CN] 大数组：主消息里只放**下标**，真实数据追加到 aux_buffers。
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # [CN] 对外呈现为纯原生类型的三元组，便于跨语言/跨版本兼容。
        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data

    # [CN] 编码 torch 张量：三选一（内联 / OOB / 独立帧）。
    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | dict | memoryview]:
        oob_consumer = self.oob_tensor_consumer
        # [CN] 统一按「连续的一维字节数组」看待，dtype 与 shape 另存。
        # view the tensor as a contiguous 1D array of bytes
        # [CN] 小的 **CPU** 张量内联。GPU 张量不能内联（要先搬回 CPU，反而更贵）。
        if obj.nbytes < self.size_threshold and obj.is_cpu:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
        # [CN] 交给带外通道：消息里只留句柄 dict。
        elif oob_consumer is not None and (data := oob_consumer(obj)) is not None:
            assert isinstance(data, dict)
        # [CN] 走独立帧：主消息放下标，数据追加到 aux_buffers。
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            assert self.aux_buffers is not None
            data = len(self.aux_buffers)
            self.aux_buffers.append(tensor_data(obj))
        # [CN] dtype 去掉 "torch." 前缀，解码时再拼回去。
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    # [CN] 多模态：按 modality 分组，逐项编码。
    def _encode_mm_items(self, items: MultiModalKwargsItems) -> dict[str, Any]:
        return {
            modality: [self._encode_mm_item(item) for item in itemlist]
            for modality, itemlist in items.items()
        }

    def _encode_mm_item(self, item: MultiModalKwargsItem) -> dict[str, Any]:
        return {key: self._encode_mm_field_elem(elem) for key, elem in item.items()}

    # [CN] 每个字段元素编码成 {data, field} 两部分。
    def _encode_mm_field_elem(self, elem: MultiModalFieldElem) -> dict[str, Any]:
        return {
            "data": (
                None if elem.data is None else self._encode_nested_tensors(elem.data)
            ),
            "field": self._encode_mm_field(elem.field),
        }

    # [CN] 嵌套张量可能是张量、标量、或嵌套 list，递归处理。
    def _encode_nested_tensors(self, nt: NestedTensors) -> Any:
        if isinstance(nt, torch.Tensor):
            return self._encode_tensor(nt)
        # [CN] 虽然违反 NestedTensors 的类型定义，但实际数据里确实会有 float，
        #      这里放行而不是报错。
        if isinstance(nt, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return nt
        return [self._encode_nested_tensors(x) for x in nt]

    # [CN] 多模态字段是**数据类**，按「工厂名 + 字段值」编码。
    def _encode_mm_field(self, field: BaseMultiModalField):
        # Figure out the factory name for the field type.
        # [CN] 只认登记表里的类型，遇到未知类型明确报错。
        name = MMF_CLASS_TO_FACTORY.get(field.__class__)
        if not name:
            raise TypeError(f"Unsupported field type: {field.__class__}")

        # We just need to copy all of the field values in order
        # which will be then used to reconstruct the field.
        # [CN] 用 dataclasses.fields 按声明顺序取值，解码时按同名关键字重建。
        factory_kw = {f.name: getattr(field, f.name) for f in dataclasses.fields(field)}
        return name, factory_kw


# [CN] **解码器主体**，与 MsgpackEncoder 严格配对。
class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    ``oob_tensor_provider`` must be used when an OOBTensorConsumer is used on the
    encoder side.
    """

    def __init__(
        self,
        t: Any | None = None,
        share_mem: bool = True,
        oob_tensor_provider: OOBTensorProvider | None = None,
    ):
        # [CN] share_mem=True 时解出的数组/张量**共享**接收缓冲区的内存，
        #      零拷贝但会锁住整条消息 buffer，所以调用方不能长期持有。
        self.share_mem = share_mem
        # [CN] 是否对解出的张量做 pin_memory（提升后续 CPU→GPU 传输效率）。
        self.pin_tensors = PIN_MEMORY
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )
        self.aux_buffers: Sequence[bytestr] = ()
        self.oob_tensor_provider = oob_tensor_provider
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    # [CN] 解码：单个 buffer 直接解；多个时第 0 个是主消息，其余作为 aux。
    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        # [CN] 单 buffer 的情况说明这条消息没有带外数据。
        if isinstance(bufs, bytestr):  # type: ignore
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    # [CN] 与 enc_hook 对称：按目标类型 t 把原生数据还原成对象。
    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if t is slice:
                return slice(*obj)
            if issubclass(t, MultiModalKwargsItem):
                return self._decode_mm_item(obj)
            if issubclass(t, MultiModalKwargsItems):
                return self._decode_mm_items(obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
        return obj

    # [CN] 还原 UtilityResult：有类型信息才做递归还原。
    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        result_type, result = obj
        if result_type is not None:
            # [CN] 类型还原需要 import 任意模块，属不安全操作，必须显式开启。
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise TypeError(
                    "VLLM_ALLOW_INSECURE_SERIALIZATION must "
                    "be set to use custom utility result types"
                )
            # Use recursive decoding to handle nested structures
            result = _decode_type_info_recursive(
                result_type, result, self._convert_result
            )
        return UtilityResult(result)

    # [CN] 按 (模块名, 类名) 动态导入并 msgspec.convert 成目标类型。
    def _convert_result(self, result_type: Sequence[str], result: Any) -> Any:
        if result_type is None:
            return result
        mod_name, name = result_type
        mod = importlib.import_module(mod_name)
        result_type = getattr(mod, name)
        return msgspec.convert(result, result_type, dec_hook=self.dec_hook)

    # [CN] 解码 numpy 数组。
    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # [CN] 零拷贝：直接用 frombuffer 建视图，不复制数据。
        #      代价是这个 ndarray 会**锁住整条消息 buffer**，不能长期持有。
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        # [CN] data 是 int 就取 aux 帧，否则是内联的字节视图。
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        arr = np.frombuffer(buffer, dtype=dtype)
        # [CN] 不共享内存时拷一份，让调用方可以长期持有。
        if not self.share_mem:
            arr = arr.copy()
        return arr.reshape(shape)

    # [CN] 解码 torch 张量：dict → 带外；int → 独立帧；否则内联。
    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        # [CN] 句柄是 dict 说明编码端用了 OOB，必须配了 provider 才能解。
        if isinstance(data, dict):
            assert self.oob_tensor_provider, (
                "Received OOB tensor but tensor provider is not set"
            )
            return self.oob_tensor_provider(dtype, shape, data)

        # [CN] 区分「来自独立帧」（可安全共享）与「内联」（必须拷出来）。
        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        buffer = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        # [CN] 空 buffer 特殊处理：frombuffer 不接受空缓冲，且 shape 里必有 0。
        if not buffer.nbytes:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        # [CN] 先按 uint8 建张量，最后再 view 回真实 dtype 和 shape。
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # [CN] 内联数据必须 clone：它背后的内存不属于 PyTorch，
        #      直接拿去做异步 CPU→GPU 传输会有生命周期问题。
        # Clone ensures tensor is backed by pytorch-owned memory for safe
        # future async CPU->GPU transfer.
        # Pin larger tensors for more efficient CPU->GPU transfer.
        # [CN] 内联的 → 必须克隆到 PyTorch 自有内存。
        if not is_aux:
            arr = arr.clone()
        # [CN] 独立帧的 → 不共享时按需 pin 或克隆（pin 比 clone 更划算）。
        elif not self.share_mem:
            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
        # Convert back to proper shape & type
        # [CN] 两步 view：先按真实 dtype 解释，再reshape 成原 shape。
        return arr.view(torch_dtype).view(shape)

    def _decode_mm_items(self, obj: dict[str, Any]) -> MultiModalKwargsItems:
        return MultiModalKwargsItems(
            {
                modality: [self._decode_mm_item(item) for item in itemlist]
                for modality, itemlist in obj.items()
            }
        )

    def _decode_mm_item(self, obj: dict[str, Any]) -> MultiModalKwargsItem:
        return MultiModalKwargsItem(
            {key: self._decode_mm_field_elem(elem) for key, elem in obj.items()}
        )

    # [CN] 还原多模态字段元素：data 递归解码，field 按工厂方法重建。
    def _decode_mm_field_elem(self, obj: dict[str, Any]) -> MultiModalFieldElem:
        if obj["data"] is not None:
            obj["data"] = self._decode_nested_tensors(obj["data"])

        # Reconstruct the field processor using MultiModalFieldConfig
        factory_meth_name, factory_kw = obj["field"]
        factory_meth = getattr(MultiModalFieldConfig, factory_meth_name)

        # [CN] 特例：flat 字段的 slices 是嵌套 slice，需要单独还原。
        # Special case: decode the union "slices" field of
        # MultiModalFlatField
        if factory_meth_name == "flat":
            factory_kw["slices"] = self._decode_nested_slices(factory_kw["slices"])

        obj["field"] = factory_meth("", **factory_kw).field
        return MultiModalFieldElem(**obj)

    # [CN] 嵌套张量解码：靠「首元素是不是字符串」判断这是张量还是 list。
    #      张量被编码成 (dtype_str, shape, data)，所以首元素是 str。
    def _decode_nested_tensors(self, obj: Any) -> NestedTensors:
        if isinstance(obj, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return obj
        if not isinstance(obj, list):
            raise TypeError(f"Unexpected NestedTensors contents: {type(obj)}")
        if obj and isinstance(obj[0], str):
            return self._decode_tensor(obj)
        return [self._decode_nested_tensors(x) for x in obj]

    # [CN] 嵌套 slice 解码：单层是 (start, stop, step)，多层继续递归。
    def _decode_nested_slices(self, obj: Any) -> Any:
        assert isinstance(obj, (list, tuple))
        if obj and not isinstance(obj[0], (list, tuple)):
            return slice(*obj)
        return [self._decode_nested_slices(x) for x in obj]

    # [CN] msgpack 扩展类型钩子：按 code 还原。
    def ext_hook(self, code: int, data: memoryview) -> Any:
        # [CN] 原始字节视图：直接返回 memoryview（零拷贝）。
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data

        # [CN] pickle / cloudpickle 只在显式开启不安全序列化时才允许解。
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            if code == CUSTOM_TYPE_PICKLE:
                return pickle.loads(data)
            if code == CUSTOM_TYPE_CLOUDPICKLE:
                return cloudpickle.loads(data)

        # [CN] 未知扩展码或未开启 → 抛异常，绝不猜测。
        raise NotImplementedError(f"Extension type code {code} is not supported")


# [CN] 在远端对象上调用方法：method 可以是名字、序列化后的字节、或可调用对象。
#      用于 collective_rpc 这类「把一段逻辑发到对端执行」的场景。
def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    # [CN] bytes → cloudpickle 反序列化成函数，再把 obj 绑成第一个参数。
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    # [CN] str → 直接 getattr。
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    # [CN] 可调用对象 → 直接 partial 绑定。
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


# [CN] 让 msgspec.Struct 能被 Pydantic 双向使用。
#
#     为什么需要它：vLLM 内部数据结构用 msgspec.Struct（快、省内存），
#     但 API 层是 Pydantic（要生成 OpenAPI、要做校验）。
#     这个 mixin 桥接两者：
#       · 校验方向：JSON/dict → Struct；
#       · 序列化方向：Struct → JSON 安全的 dict。
class PydanticMsgspecMixin:
    """Make a ``msgspec.Struct`` compatible with Pydantic for both
    **validation** (JSON/dict -> Struct) and **serialization**
    (Struct -> JSON-safe dict).

    Subclasses may set ``__pydantic_msgspec_exclude__`` (a ``set[str]``)
    to list non-underscore field names that should also be stripped from
    serialized output.  Fields whose names start with ``_`` are always
    excluded automatically.
    """

    # Subclasses can override to exclude additional public-but-internal keys.
    # [CN] 子类可覆写：除了下划线开头的私有字段，还想额外剔除的公开字段。
    __pydantic_msgspec_exclude__: ClassVar[set[str]] = set()

    # [CN] Pydantic 的钩子：返回一个同时支持「直接传实例」和「传 dict」的 union schema。
    #      注意这个方法会被 Pydantic **缓存**，不会每次校验都跑。
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Make msgspec.Struct compatible with Pydantic, respecting defaults.
        Handle JSON=>msgspec.Struct. Used when exposing msgspec.Struct to the
        API as input or in `/docs`. Note this is cached by Pydantic and not
        called on every validation.
        """
        # [CN] 取 msgspec 的字段定义与类型注解。
        msgspec_fields = {f.name: f for f in msgspec.structs.fields(source_type)}
        type_hints = get_type_hints(source_type)

        # Build the Pydantic typed_dict_field for each msgspec field
        fields = {}
        # [CN] 逐字段构造 Pydantic 的 typed_dict_field。
        for name, hint in type_hints.items():
            # [CN] 跳过 ClassVar 等非 Struct 字段。
            if name not in msgspec_fields:
                # Skip ClassVar and other non-struct annotations.
                continue
            # Skip private fields — they are excluded from serialization
            # and should not appear in the generated JSON/OpenAPI schema.
            # [CN] 私有字段不进 schema，也不进 OpenAPI 文档。
            if name.startswith("_"):
                continue
            msgspec_field = msgspec_fields[name]

            # typed_dict_field using the handler to get the schema
            field_schema = handler(hint)

            # Add default value to the schema.
            # [CN] 有默认值的字段标记为**不必填**，
            #      这样生成的 JSON Schema 与 omit_defaults=True 的序列化行为一致
            #      （处于默认值的字段可能根本不会出现）。
            # Mark fields with defaults as not required so the generated
            # JSON Schema stays consistent with ``omit_defaults=True``
            # serialization (fields at their default value may be absent).
            # [CN] default_factory 形式（每次调用生成一个新默认值）。
            if msgspec_field.default_factory is not msgspec.NODEFAULT:
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default_factory=msgspec_field.default_factory,
                )
                fields[name] = core_schema.typed_dict_field(
                    wrapped_schema, required=False
                )
            # [CN] 普通默认值形式。
            elif msgspec_field.default is not msgspec.NODEFAULT:
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default=msgspec_field.default,
                )
                fields[name] = core_schema.typed_dict_field(
                    wrapped_schema, required=False
                )
            # [CN] 没有默认值 → 必填。
            else:
                # No default, so Pydantic will treat it as required
                fields[name] = core_schema.typed_dict_field(field_schema)
        # [CN] 先用 typed_dict 校验，再转成真正的 Struct。
        typed_dict_then_convert = core_schema.no_info_after_validator_function(
            cls._validate_msgspec,
            core_schema.typed_dict_schema(fields),
        )

        # [CN] 序列化方向：用自定义函数剥离私有/排除字段。
        # Build a serializer that strips private / excluded fields.
        serializer = core_schema.plain_serializer_function_ser_schema(
            cls._serialize_msgspec,
            info_arg=False,
        )

        # Accept either an already-constructed msgspec.Struct instance or a
        # JSON/dict-like payload.
        # [CN] 联合：既接受已构造好的 Struct 实例，也接受 JSON/dict 载荷。
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(source_type),
                typed_dict_then_convert,
            ],
            serialization=serializer,
        )

    # [CN] 校验并转换输入为 Struct 实例。
    @classmethod
    def _validate_msgspec(cls, value: Any) -> Any:
        """Validate and convert input to msgspec.Struct instance."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return msgspec.convert(value, type=cls)

    # [CN] 序列化为 JSON 兼容 dict，剥离私有字段与显式排除的字段。
    @staticmethod
    def _serialize_msgspec(value: Any) -> Any:
        """Serialize a msgspec.Struct to a JSON-compatible dict, stripping
        private (``_``-prefixed) and explicitly excluded fields.

        Uses ``msgspec.to_builtins`` which respects ``omit_defaults=True``,
        so only fields that differ from their declared defaults are included.
        """
        # [CN] to_builtins 配合 omit_defaults=True：
        #      只有与默认值不同的字段才会出现，输出更紧凑。
        raw = msgspec.to_builtins(value)
        if not isinstance(raw, dict):
            return raw

        exclude: set[str] = cast(
            set[str],
            getattr(type(value), "__pydantic_msgspec_exclude__", set()),
        )
        # [CN] 就地删除要隐藏的键。
        for key in list(raw):
            if key.startswith("_") or key in exclude:
                del raw[key]

        return raw
