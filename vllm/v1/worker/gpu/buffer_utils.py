# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)

# [CN] 文件总览：V2 runner 的「缓冲与写入」基础设施。
# [CN] 核心矛盾：CUDA graph 要求张量地址固定，但每步都要更新内容；
# [CN] 同时 async scheduling 下 CPU 侧写入与 GPU 侧读取会重叠。
# [CN] 本文件的解法：
# [CN]   1) UVA（统一虚拟寻址）—— CPU 写、GPU 读同一块物理内存，无需显式拷贝；
# [CN]   2) 轮转缓冲池 —— 多个 UVA buffer 轮着用，避免本步写覆盖上步未读完的；
# [CN]   3) StagedWrite —— CPU 先攒「差量写」，再由一个 Triton kernel 批量落盘，
# [CN]      把 N 次小 kernel launch 合并成 1 次。
# [CN] 容易看错的点：UVA 不是「免拷贝」，而是「免显式拷贝」——
# [CN] 数据仍要走 PCIe，只是地址统一、不需要 copy_ 调用。
# Default round-robin depth for the UVA buffer pools. Must be >= the number of
# concurrent in-flight steps (engine batch_queue_size).
_DEFAULT_MAX_CONCURRENCY = 2


# [CN] 轮转深度必须 >= 同时在飞的 step 数（即 engine 的 batch_queue_size），
# [CN] 否则第 N+1 步的写入会覆盖第 N 步 GPU 还没读完的缓冲。
# [CN] 下限取 2：至少要让「本步写」与「上步读」分开。
def set_default_max_concurrency(n: int) -> None:
    global _DEFAULT_MAX_CONCURRENCY
    _DEFAULT_MAX_CONCURRENCY = max(2, n)


# [CN] 异步 H2D 的封装：先 pin（已 pinned 则是 no-op），再 non_blocking copy_。
def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # pin_memory() is no-op if the memory is already pinned.
    pinned = x.pin_memory()
    return out.copy_(pinned, non_blocking=True)


# [CN] 一块 UVA 内存的三视图：cpu（torch）、np（numpy）、uva（GPU 可寻址）。
# [CN] 三者共享同一块物理内存，写 np 等价于写 GPU 可见数据。
class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)


# [CN] 多块 UVA buffer 的轮转池。
# [CN] 为什么需要轮转：async scheduling 下 CPU 侧第 N+1 步的写入，
# [CN] 与 GPU 侧第 N 步的读取是并发的，用同一块 buffer 会读到脏数据。
class UvaBufferPool:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0

    # [CN] 先切到「下一块」再写——即写入的永远是「最老的」那块，
    # [CN] 保证当前正在被 GPU 读的那块不会被改。
    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


# [CN] 「真值在 CPU（普通 pinned=False 内存），GPU 侧通过 UVA 池看它」。
# [CN] 与 UvaBuffer 的区别：UvaBuffer 本身就在 UVA 上，
# [CN] 而这里的 cpu 是普通内存，每次 copy_to_uva 才同步一份到 UVA 池。
class UvaBackedTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)
        self.gpu = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


# [CN] 支持「攒一批写、一次性落到 GPU」的张量。
# [CN] 只支持 int32 / int64 / float32：因为要写 Triton kernel，
# [CN] dtype 必须是编译期可枚举的。
# [CN] uva_instead_of_gpu=True 用于 all_token_ids 这类超大但访问稀疏的张量。
class StagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int | None = None,
        uva_instead_of_gpu: bool = False,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    # [CN] 攒一次「从 index 行的 start 列开始，写入 x 序列」的写操作。
    # [CN] 注意这里只记元数据 + 把内容 append 进一个扁平 list，
    # [CN] 真正的 GPU 写入推迟到 apply_write。
    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    # [CN] 用一个 Triton kernel 把攒下的所有写一次性落到 GPU。
    # [CN] cu_lens 是前缀和，kernel 里靠它定位每段内容的起止。
    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        write_contents = async_tensor_h2d(
            self._staged_write_contents, device=self.device, dtype=self.dtype
        )

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            None,
            BLOCK_SIZE=1024,
            MULTI_GROUP=False,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


# [CN] 把多个 StagedWriteTensor（例如多个 KV cache group）的差量写
# [CN] 合并到「一个」kernel 里执行：多一次 launch 换 N 次，省 launch 开销。
# [CN] 实现手法：给每次写带上 group_id，kernel 里按 group_id 解析基址与 stride。
class FusedStagedWriter:
    """Applies the staged writes of several `StagedWriteTensor`s at once."""

    def __init__(
        self, device: torch.device, max_writes: int, max_concurrency: int | None = None
    ):
        new_pool = partial(
            UvaBufferPool, dtype=torch.int32, max_concurrency=max_concurrency
        )
        self.group_ids = new_pool(max_writes)
        self.indices = new_pool(max_writes)
        self.starts = new_pool(max_writes)
        self.cu_lens = new_pool(max_writes)
        self.device = device

    def apply(
        self,
        tensors: Sequence[StagedWriteTensor],
        output_ptrs: torch.Tensor,
        output_strides: torch.Tensor,
    ) -> None:
        """Apply and clear the staged writes of `tensors` with one kernel."""
        group_ids: list[int] = []
        indices: list[int] = []
        starts: list[int] = []
        contents: list[int | float] = []
        cu_lens: list[int] = []

        for group_id, t in enumerate(tensors):
            n = len(t._staged_write_indices)
            if n == 0:
                continue

            group_ids.extend([group_id] * n)
            indices.extend(t._staged_write_indices)
            starts.extend(t._staged_write_starts)
            content_base = len(contents)
            contents.extend(t._staged_write_contents)
            cu_lens.extend(content_base + cu_len for cu_len in t._staged_write_cu_lens)

        if not group_ids:
            return

        group_ids_uva = self.group_ids.copy_to_uva(group_ids)
        indices_uva = self.indices.copy_to_uva(indices)
        starts_uva = self.starts.copy_to_uva(starts)
        cu_lens_uva = self.cu_lens.copy_to_uva(cu_lens)
        contents_gpu = async_tensor_h2d(contents, device=self.device, dtype=torch.int32)

        _apply_write_kernel[(len(group_ids),)](
            output_ptrs,
            output_strides,
            indices_uva,
            starts_uva,
            contents_gpu,
            cu_lens_uva,
            group_ids_uva,
            BLOCK_SIZE=1024,
            MULTI_GROUP=True,
        )
        for t in tensors:
            t.clear_staged_writes()


@triton.jit
# [CN] 一个 program 对应一次写（不是一行）。
# [CN] MULTI_GROUP=True 时 output_ptr 是「指针的数组」，需要先解引用两次
# [CN] 才能拿到真正的行地址（见 _load_ptr）。
def _apply_write_kernel(
    output_ptr,  # MULTI_GROUP: ptr-to-ptrs [num_groups]; else: data ptr
    output_stride,  # MULTI_GROUP: ptr-to-strides [num_groups]; else: row stride
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    write_group_ids_ptr,  # [num_writes], used only when MULTI_GROUP
    BLOCK_SIZE: tl.constexpr,
    MULTI_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    if MULTI_GROUP:
        # Each write targets a different output tensor (KV cache group);
        # resolve its base pointer and row stride per write.
        group_id = tl.load(write_group_ids_ptr + pid)
        row_ptr = _load_ptr(output_ptr + group_id, tl.int32)
        row_stride = tl.load(output_stride + group_id)
    else:
        row_ptr = output_ptr
        row_stride = output_stride
    row_ptr += row_idx * row_stride + start_idx

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(row_ptr + block, content, mask=mask)


@triton.jit
# [CN] 从 int64 加载出指针并转型。
# [CN] multiple_of(ptr, 16) 是给编译器的对齐提示，能生成更好的访存指令。
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)
