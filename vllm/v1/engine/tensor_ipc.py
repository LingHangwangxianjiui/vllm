# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor IPC transport via torch.multiprocessing.Queue.

This module contains the queue-based transport logic for sharing tensors
between processes (e.g., API server -> engine core). The msgpack layer
emits/consumes lightweight :class:`TensorIpcData` values, while transport
state such as request association, handle generation, queue routing, buffering,
and cleanup lives here.
"""


# [CN] 文件总览：**张量的跨进程零拷贝传输（IPC）**。
#
#     问题背景：多模态请求要把像素值、音频特征这类大张量从 API 进程送进
#     EngineCore 进程。走 msgpack 序列化等于在 CPU 上整块拷一遍，几 MB 的图
#     在高频请求下开销可观。
#
#     方案：**带外传输（out-of-band, OOB）**。msgpack 里只放一个轻量句柄
#     dict（sender_id / message_id / tensor_id 三个字段），真正的张量走
#     torch.multiprocessing.Queue + **共享内存**。解码器看到句柄后回调这里，
#     把真张量取回来。
#
#     为什么敢零拷贝：share_memory_() 把张量放进共享内存段，两个进程映射同一
#     段物理页，传递的是指针而不是字节。
#
#     ------------------------- 三条设计约束 -------------------------
#     ① **只服务 rank 0**：TP>1 / PP>1 时只有 rank 0 会消费多模态张量，
#        所以只需要一条队列；DP>1 直接不支持（见 set_target_engine）。
#     ② **允许失败降级**：发送失败返回 None，上层退回普通序列化，
#        绝不因为 IPC 出问题就让请求失败。
#     ③ **必须容忍乱序与迟到**：多个生产者并发 put，到达顺序不保证。
#        所以接收端用「排空并缓冲（drain-and-buffer）」模式：一直 get，
#        直到拿到想要的那个，顺路拿到的先存起来。
#
#     代码组织：TensorIpcData（线路上的数据单元）→ TensorIpcSender（发送侧，
#     实现 OOBTensorConsumer 接口）→ TensorIpcReceiver（接收侧，按句柄取回）。
import dataclasses
import uuid
from collections import defaultdict
from dataclasses import field
from multiprocessing.queues import Queue as MPQueue
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.serial_utils import OOBTensorConsumer

logger = init_logger(__name__)

TensorIpcQueue = MPQueue


# [CN] 队列上传输的最小数据单元：三元组定位 + 张量本体。
@dataclasses.dataclass
class TensorIpcData:
    """
    Data sent via torch.multiprocessing.Queue for zero-copy IPC.

    Contains the tensor_id and the actual tensor. The tensor is
    shared in memory (GPU or CPU) for efficient inter-process communication.
    """

    # [CN] 发送方标识。多生产者场景下用它在接收端区分「这批张量是谁发的」。
    sender_id: str
    # [CN] 消息序号，随 new_message() 递增。用于识别并丢弃上一批的陈旧张量。
    message_id: int
    # [CN] 张量在单条消息内部的序号，每条消息从 0 重新计数。
    tensor_id: int
    # [CN] 张量本体。已 share_memory_，跨进程共享同一段物理内存。
    tensor: torch.Tensor


# [CN] 发送侧：把张量推入共享内存队列，返回可序列化的轻量句柄。
class TensorIpcSender(OOBTensorConsumer):
    """Send-side logic for tensor IPC via torch.multiprocessing.Queue.

    Uses a single queue targeting rank 0 (the only rank that consumes
    multimodal tensors during TP>1 / PP>1. Note: DP>1 not supported).
    """

    # [CN] 初始化：随机 sender_id，避免多个前端进程的序号互相撞车。
    def __init__(self, queue: TensorIpcQueue):
        # [CN] 张量序号：每条消息内从 0 重新计数。
        self.queue = queue
        # [CN] 消息序号：每次 new_message() 递增。
        self._tensor_id_counter = 0
        self._message_counter = 0
        self._sender_id = uuid.uuid4().hex[:8]

    # [CN] 只支持单队列（rank 0）；指定其它 engine 直接报错而非静默走错路。
    def set_target_engine(self, target_engine: int) -> None:
        if target_engine != 0:
            raise IndexError(
                "TensorIpcSender only supports a single queue; "
                f"got target engine {target_engine}"
            )

    # [CN] 开始一条新消息：消息号 +1、张量号归零（相当于开启新的命名空间）。
    def new_message(self) -> None:
        self._message_counter += 1
        self._tensor_id_counter = 0

    # [CN] 发送一个张量：成功返回句柄 dict，失败返回 None 让上层降级。
    def __call__(self, tensor: torch.Tensor) -> dict[str, Any] | None:
        """Send tensor via queue, return its handle. Returns None if failed."""
        # [CN] 任何异常都吞掉并降级 —— IPC 失败不应该拖垮整个请求。
        try:
            # Move tensor to shared memory for IPC
            # This is required for proper inter-process communication
            # [CN] 不在共享内存的先搬进去。这是零拷贝的前提，也是唯一一次可能的拷贝。
            if not tensor.is_shared():
                tensor = tensor.share_memory_()

            # [CN] 构造句柄：只有三个小字段，序列化成本可忽略。
            metadata = {
                "sender_id": self._sender_id,
                "message_id": self._message_counter,
                "tensor_id": self._tensor_id_counter,
            }

            self._tensor_id_counter += 1

            ipc_data = TensorIpcData(**metadata, tensor=tensor)  # type: ignore[arg-type]

            # [CN] 带超时的 put：队列满时最多等 10s，避免消费端挂掉时永久阻塞。
            # Use a timeout to avoid blocking indefinitely
            self.queue.put(ipc_data, timeout=10.0)

            logger.debug(
                "Sent tensor %s for (shape=%s, device=%s) "
                "via IPC queue (shared memory)",
                metadata,
                tensor.shape,
                tensor.device,
            )

            return metadata
        except Exception as e:
            # [CN] 失败路径：记录告警并返回 None，调用方据此退回普通序列化。
            logger.warning(
                "Failed to send tensor via IPC queue: %s. "
                "Falling back to standard serialization.",
                e,
            )
            return None


# [CN] 单个发送方的接收缓冲：当前水位消息号 + 两级张量字典。
@dataclasses.dataclass
class _Sender:
    # [CN] 已处理到的最新消息号（水位线），用于丢弃比它更旧的张量。
    current_message_id: int = -1
    # [CN] 两级缓冲：先按 message_id 分组，再按 tensor_id 索引。
    tensors: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)


# [CN] 接收侧：按句柄从队列取回张量，必要时阻塞等待。
class TensorIpcReceiver:
    """Receive-side logic for tensor IPC via torch.multiprocessing.Queue.

    Wraps the queue receive logic previously embedded in MsgpackDecoder.
    """

    # [CN] defaultdict：首次见到某个 sender 自动建缓冲，省掉存在性判断。
    def __init__(self, queue: TensorIpcQueue):
        self.queue = queue
        # [CN] 每个 sender 一份独立缓冲，互不干扰。
        self._tensor_buffers = defaultdict[str, _Sender](_Sender)

    # [CN] 按 (sender_id, message_id, tensor_id) 取回张量。
    #      采用「排空并缓冲」：不停 get 直到拿到要的那个，顺路的先存起来。
    def __call__(
        self, dtype: str, shape: tuple[int, ...], meta: dict[str, Any]
    ) -> torch.Tensor:
        """Retrieve a tensor from torch.multiprocessing.Queue.

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        """

        # [CN] 从句柄里解出三元组作为查找键。
        # Create lookup key from handle
        sender_id: str = meta["sender_id"]
        message_id: int = meta["message_id"]
        tensor_id: int = meta["tensor_id"]

        # Drain all available tensors. We save them regardless if this is
        # the one we're waiting for as they may arrive out of order from
        # multiple producers.
        # [CN] 排空循环：每轮先查缓冲区，命中即返回；未命中则从队列 get 一个再查。
        while True:
            # [CN] 先看这个 sender 的缓冲区里有没有现成的。
            sender = self._tensor_buffers.get(sender_id)
            if sender is not None:
                tensors = sender.tensors
                # [CN] 用 pop 而非 get：取走就不再保留，避免缓冲区无限增长。
                tensor = tensors.get(message_id, {}).pop(tensor_id, None)
                if tensor is not None:
                    # [CN] 换了一条新消息 → 顺手清掉比当前消息更旧的所有分组。
                    if sender.current_message_id != message_id:
                        # [CN] 逐个弹出 message_id 小于当前消息的分组（海象运算符边取边比）。
                        while tensors and (mid := next(iter(tensors))) < message_id:
                            if sender.tensors.pop(mid):
                                logger.warning(
                                    "Discarding %d stale tensors from sender %s",
                                    sender_id,
                                )
                        # [CN] 推进水位线，之后更旧的张量会被直接丢弃而不是缓冲。
                        sender.current_message_id = message_id
                    logger.debug(
                        "Received tensor %s from sender %s for (shape=%s, device=%s) "
                        "via IPC queue (shared memory)",
                        (message_id, tensor_id),
                        sender_id,
                        tensor.shape,
                        tensor.device,
                    )
                    return tensor

            # [CN] 缓冲区没有 → 阻塞取下一个；10s 超时是为了防止死锁变成永久挂起。
            ipc_data: TensorIpcData = self.queue.get(timeout=10.0)

            # Store tensor
            # [CN] 按 sender 归档到对应缓冲区。
            sender = self._tensor_buffers[ipc_data.sender_id]
            # [CN] 比当前水位还旧 → 这是迟到包（已被跳过），直接丢弃。
            if sender.current_message_id > ipc_data.message_id:
                logger.warning(
                    "Ignoring stale tensor from sender %s", ipc_data.sender_id
                )
                continue

            # [CN] 存入两级字典，等真正被请求时再 pop 出来。
            sender.tensors.setdefault(ipc_data.message_id, {})[ipc_data.tensor_id] = (
                ipc_data.tensor
            )
