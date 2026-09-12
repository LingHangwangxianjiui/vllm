# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：worker 侧的「临时显存工作区」管理器。
# [CN] 要解决的问题：MoE 的 all-to-all、量化的中间结果等算子需要一块
# [CN]   临时显存，若每次现分配会产生大量 cudaMalloc（同步 + 碎片）。
# [CN] 解法：预分配一块 uint8 大 buffer，按需切成多个视图复用。
# [CN] 核心机制：
# [CN]   1) 每个 (ubatch, lane) 组合独占一块 buffer，互不干扰；
# [CN]   2) 首次分配后 lock()，此后只允许「取不大于当前大小」的工作区；
# [CN]   3) 一旦越界就抛 AssertionError —— 这是刻意的，把 OOM 风险前移到
# [CN]      warmup/profile 阶段暴露，而不是在运行时随机炸。
# [CN] 核心类：WorkspaceManager；模块级单例 _manager。
# [CN] 最容易看错的点：get_simultaneous 返回的都是同一块 buffer 的视图，
# [CN]   调用方必须保证这些张量的生命周期不重叠，否则会互相踩数据。


import inspect
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import accumulate
from math import prod

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


# [CN] shape * itemsize = 字节数。
def _compute_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return prod(shape) * dtype.itemsize


# Constants
_MB = 1024**2
_GiB = 1024**3

# Global workspace manager instance
# [CN] 模块级单例与 lane 上下文变量。
# [CN] lane 用 ContextVar 而非参数传递：算子调用链很深，
# [CN] 用上下文变量避免污染每一层函数签名。
_manager: "WorkspaceManager | None" = None
_workspace_lane: ContextVar[int] = ContextVar("vllm_workspace_lane", default=0)


# [CN] 切换 lane：让同一 ubatch 内的不同执行流（如不同请求组）各有独立工作区。
@contextmanager
def use_workspace_lane(lane: int) -> Iterator[None]:
    """Select an independent workspace owner for this execution context."""
    if lane < 0:
        raise ValueError(f"Workspace lane must be non-negative, got {lane}.")
    token = _workspace_lane.set(lane)
    try:
        yield
    finally:
        _workspace_lane.reset(token)


# [CN] 工作区管理器：为每个 (ubatch, lane) 槽位维护一块 buffer。
class WorkspaceManager:
    """Manager for workspace allocation.

    Manages one workspace buffer per active ``(ubatch, lane)`` slot.
    Can be locked to prevent further growth during execution.
    """

    # [CN] num_ubatches 在初始化时固定（DBO 打开时为 2），不支持动态扩容。
    def __init__(
        self,
        device: torch.device,
        num_ubatches: int | None = None,
        num_lanes: int = 1,
    ):
        self._device = device
        # Cache num ubatches at init based on configuration (default to 1)
        self._num_ubatches = num_ubatches if num_ubatches is not None else 1
        if num_lanes < 1:
            raise ValueError(f"num_lanes must be at least one, got {num_lanes}.")
        self._num_lanes = num_lanes
        self._current_workspaces: list[torch.Tensor | None] = [None] * (
            self._num_ubatches * self._num_lanes
        )
        self._locked: bool = False

    # [CN] 未分配的槽位大小为 0，而非报错。
    @staticmethod
    def _workspace_size_bytes(workspace: torch.Tensor | None) -> int:
        """Get size of workspace in bytes."""
        if workspace is None:
            return 0
        return workspace.numel() * workspace.element_size()

    # [CN] 锁定后禁止增长。调用时机：warmup / CUDA Graph 捕获完成后。
    def lock(self) -> None:
        """Lock the workspace to prevent further growth.

        After locking, any attempt to allocate a larger workspace will raise
        an assertion error. This ensures workspace size is fixed during execution.
        """
        self._locked = True
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] Workspace locked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    # [CN] 解锁：弹性 EP 扩容时专家数变化，工作区需求会变大，必须允许重新增长。
    def unlock(self) -> None:
        """Unlock the workspace to allow growth.

        This is used during elastic EP scaling when the workspace size
        needs to grow due to changes in the number of experts.
        """
        self._locked = False
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] Workspace unlocked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def is_locked(self) -> bool:
        """Check if workspace is locked."""
        return self._locked

    # [CN] 一次性切出多个视图：先按 256 字节对齐累加出总需求，再算各自偏移。
    # [CN] 对齐是为了满足 CUDA 算子对地址对齐的要求（否则可能报错或掉速）。
    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        """Get multiple workspace tensors simultaneously from a single allocation.

        Args:
            *shapes_and_dtypes: One or more (shape, dtype) tuples.

        Returns:
            List of tensor views into the workspace buffer, one per shape/dtype pair.
        """
        # [CN] round_up 到 256：对齐后切分，避免前一个张量尾部与后一个头部重叠。
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        total_bytes = sum(aligned_bytes)

        # Calculate cumulative offsets using itertools.accumulate
        # [CN] accumulate 求前缀和，得到每个张量在 buffer 中的起始偏移。
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        current_workspace = self._ensure_workspace_size(total_bytes)

        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    # [CN] 确保当前槽位的 buffer 至少 required_bytes 大，必要时重新分配。
    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        """Ensure workspace is allocated and large enough, return current workspace.

        Args:
            required_bytes: The number of bytes required.

        Returns:
            The current workspace tensor.
        """
        # [CN] 槽位下标 = ubatch_id * num_lanes + lane（行优先展平）。
        ubatch_id = dbo_current_ubatch_id()
        lane = _workspace_lane.get()
        if lane >= self._num_lanes:
            raise RuntimeError(
                f"Workspace lane {lane} is not configured; manager has "
                f"{self._num_lanes} lane(s)."
            )
        workspace_id = ubatch_id * self._num_lanes + lane
        current_workspace = self._current_workspaces[workspace_id]
        current_size = self._workspace_size_bytes(current_workspace)

        # [CN] 空间不够时才走下面的扩容路径。
        if current_size < required_bytes:

            # [CN] 辅助函数：回溯调用栈找到「真正是谁」触发了扩容，便于定位。
            def get_caller_info() -> str:
                """Find first frame outside WorkspaceManager."""
                curr_frame = inspect.currentframe()
                if curr_frame is None:
                    return "unknown"
                # Walk up the stack skipping WorkspaceManager frames
                curr_frame = curr_frame.f_back
                while curr_frame is not None:
                    # TODO: This only catches instance methods (self), missing
                    # classmethods and staticmethods. Once Python 3.11+ is the
                    # minimum supported version, use co_qualname instead:
                    #   qualname = curr_frame.f_code.co_qualname
                    #   if qualname.startswith("WorkspaceManager."):
                    if isinstance(curr_frame.f_locals.get("self"), WorkspaceManager):
                        curr_frame = curr_frame.f_back
                        continue
                    filename = os.path.basename(curr_frame.f_code.co_filename)
                    return (
                        f"{filename}:{curr_frame.f_lineno}:{curr_frame.f_code.co_name}"
                    )
                return "unknown"

            # [CN] 锁定状态下的越界是硬错误：宁可启动失败，也不要运行时偷偷分配。
            if self._locked:
                raise AssertionError(
                    f"Workspace is locked but allocation from '{get_caller_info()}' "
                    f"requires {required_bytes / _MB:.2f} MB, current size is "
                    f"{current_size / _MB:.2f} MB. "
                    "Workspace growth is not allowed after locking."
                )

            # [CN] 只给「当前请求的槽位」扩容。若把其他 ubatch 一起扩了，
            # [CN] 那些 ubatch 还持有旧 tensor 的视图，旧 tensor 会被提前释放（DBO 下的泄漏/悬垂）。
            # Only resize the requesting ubatch/lane workspace. Other slots
            # resize lazily on their next get_simultaneous call.
            # Resizing all ubatches here would orphan the other ubatch's
            # old tensor when it still holds views into it (DBO leak).
            self._current_workspaces[workspace_id] = None
            # [CN] 先置 None、del 引用，再 empty_cache()，把旧段真正还给 CUDA 缓存分配器，
            # [CN] 否则每次扩容都会在 reserved 里留下一块死段，峰值显存被抬高。
            del current_workspace
            # Release the freed segment back to CUDA so the caching
            # allocator can reuse the GPU memory for the larger
            # allocation below. Without this, each resize may leave a
            # dead segment in reserved memory which can cause higher peak
            # memory usage.
            torch.accelerator.empty_cache()
            self._current_workspaces[workspace_id] = torch.empty(
                (required_bytes,), dtype=torch.uint8, device=self._device
            )
            current_workspace = self._current_workspaces[workspace_id]

            if envs.VLLM_DEBUG_WORKSPACE:
                logger.info(
                    "[WORKSPACE DEBUG] Resized workspace from '%s': %.2f MB -> "
                    "%.2f MB (ubatch %d, lane %d)",
                    get_caller_info(),
                    current_size / _MB,
                    required_bytes / _MB,
                    ubatch_id,
                    lane,
                )

        return current_workspace


# [CN] 单例状态的查询与操作入口。
def is_workspace_manager_initialized() -> bool:
    """Check if workspace manager has been initialized.

    Returns:
        True if workspace manager is initialized, False otherwise.
    """
    return _manager is not None


def current_workspace_manager() -> "WorkspaceManager":
    """Get the current workspace manager instance.

    Raises:
        AssertionError: If workspace manager has not been initialized.
    """
    assert _manager is not None, (
        "WorkspaceManager not initialized. Call init_workspace_manager() "
        "with a device before using workspace functions."
    )
    return _manager


# [CN] 初始化单例。典型调用点在 GPUModelRunner.__init__。
def init_workspace_manager(
    device: torch.device,
    num_ubatches: int | None = None,
    num_lanes: int = 1,
) -> None:
    """Initialize the workspace manager with a device.

    Must be called before using any workspace functions. Typically called
    from GPUModelRunner.__init__.

    Args:
        device: The device to allocate workspace on.
        num_ubatches: Number of workspace ubatch slots. Defaults to 1.
        num_lanes: Number of independent execution lanes per ubatch. Defaults to 1.
    """
    global _manager
    if _manager is not None:
        logger.warning(
            "WorkspaceManager already initialized on device %s, "
            "reinitializing on device %s",
            _manager._device,
            device,
        )
    _manager = WorkspaceManager(device, num_ubatches, num_lanes)


# [CN] 锁定全局工作区：warmup 结束后调用，把「运行时不再分配显存」变成硬约束。
def lock_workspace() -> None:
    """Lock the workspace to prevent further growth.

    After calling this function, any attempt to allocate a workspace larger
    than the current size will raise an AssertionError. This ensures that
    workspace size is fixed during execution and prevents unexpected memory
    allocations in the hot path.

    Example:
        # During initialization
        init_workspace_manager(device)
        reserve_workspace(shape1, dtype1)
        reserve_workspace(shape2, dtype2)

        # Lock after warmup/profiling
        lock_workspace()

        # Now all get_workspace calls must fit in pre-allocated size
    """
    current_workspace_manager().lock()


def unlock_workspace() -> None:
    """Unlock the workspace to allow growth.

    This is used during elastic EP scaling when the workspace size
    needs to grow due to changes in the number of experts.
    After scaling operations complete, lock_workspace() should be
    called again to prevent unexpected allocations.
    """
    current_workspace_manager().unlock()


# [CN] 仅供测试：把单例清空以便重新初始化。
def reset_workspace_manager() -> None:
    """Reset the workspace manager to uninitialized state.

    This is primarily intended for testing purposes to allow tests
    to reinitialize the workspace manager cleanly.
    """
    global _manager
    _manager = None
