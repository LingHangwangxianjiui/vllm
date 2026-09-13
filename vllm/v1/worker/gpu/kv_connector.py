# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


# [CN] 文件总览：把「KV 传输」（P/D 分离、offloading 等）封装成 runner 可插拔接口。
# [CN] 设计要点：用空实现（NO_OP）替代到处写 if has_kv_transfer_group()，
# [CN] 让主链路代码保持干净。
class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    # [CN] 前向之前：处理抢占、绑定本步元数据、按需启动 KV 加载。
    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        pass

    # [CN] 前向之后：补发异步加载、等保存完成、收集完成/失败块与统计，
    # [CN] 最后清空本步元数据（否则下一步会读到陈旧数据）。
    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass


# [CN] 真正干活的实现，持有全局 kv_transfer_group。
class ActiveKVConnector(KVConnector):
    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ):
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._pending_load_start = False
        self._disabled = False

    # [CN] 同步加载必须在本步前向「之前」完成（否则读不到）；
    # [CN] 异步加载则推迟到 post_forward 再发起，把主机侧提交开销挪出关键路径。
    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)

        if scheduler_output.has_sync_kv_loads:
            # Sync loads need to run before this step's forward.
            self._start_load_kv()
        else:
            # Start any async loads in post-forward instead, keeping
            # their host-side submission cost off the critical path.
            self._pending_load_start = True

    def _start_load_kv(self) -> None:
        self._pending_load_start = False
        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        if self._pending_load_start:
            self._start_load_kv()

        output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        self.kv_connector.clear_connector_metadata()
        return output

    # [CN] 本步没有任何 token 要算（纯 KV 传输步）：只做收发，不走模型。
    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids, wait_for_save=False)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    # [CN] 禁用时必须同时清掉 KV_CONNECTOR_AGENT 全局量，
    # [CN] 否则各层里的 connector 钩子仍会被调用。
    def set_disabled(self, disabled: bool) -> None:
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


# [CN] 工厂：没配置传输组就返回空实现，主链路无感。
def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict)
