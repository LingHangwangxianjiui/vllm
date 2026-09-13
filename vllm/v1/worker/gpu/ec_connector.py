# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ECConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


# [CN] 文件总览：EC = Encoder Cache。把多模态 encoder 的输出在 P/D 之间
# [CN] （或跨实例）传输，避免重复跑 vision tower。
# [CN] 与 kv_connector.py 同构：空实现 + 真实实现 + 工厂。
class ECConnector:
    """EC connector interface used by the V2 GPU model runner."""

    @contextmanager
    def maybe_get_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> Generator[ECConnectorOutput | None, None, None]:
        yield None

    def no_forward(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT


# [CN] EC 传输的三种角色：producer（产出并 offload）、consumer（加载）、
# [CN] ec_both（既产出又加载）。
class ActiveECConnector(ECConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        encoder_cache: dict[str, torch.Tensor],
    ) -> None:
        self.encoder_cache = encoder_cache
        self.ec_connector = get_ec_transfer()
        assert isinstance(self.ec_connector, ECConnectorBase)
        # Every producer offloads freshly computed encoder outputs, including
        # an ec_both node that also reloads them.
        self.save_new_caches = self.ec_connector.is_producer

    @contextmanager
    # [CN] 上下文管理器形式：进入时按需发起收/发，
    # [CN] 退出时把「本步新算出来的」hash 落盘，并收集完成状态。
    # [CN] 用 set 差集（新 hash - 进入前已有 hash）识别新增项，简单且可靠。
    def maybe_get_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> Generator[ECConnectorOutput | None, None, None]:
        if scheduler_output.ec_connector_metadata is None:
            yield None
            return

        output = ECConnectorOutput()
        ec_connector = self.ec_connector
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        if ec_connector.is_producer:
            ec_connector.start_save_caches(encoder_cache=self.encoder_cache)

        if ec_connector.is_consumer:
            ec_connector.start_load_caches(self.encoder_cache)

        cached_hashes = set(self.encoder_cache) if self.save_new_caches else None
        try:
            yield output
            if cached_hashes is not None:
                for mm_hash in self.encoder_cache.keys() - cached_hashes:
                    ec_connector.save_caches(
                        encoder_cache=self.encoder_cache, mm_hash=mm_hash
                    )
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.ec_connector_worker_meta = ec_connector.build_connector_worker_meta()
            ec_connector.clear_connector_metadata()

    def no_forward(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # EC send/recv even if no work to do.
        with self.maybe_get_output(scheduler_output) as ec_connector_output:
            pass

        return ModelRunnerOutput.with_ec_conn_output_only(ec_connector_output)


NO_OP_EC_CONNECTOR = ECConnector()


# [CN] encoder-decoder 模型不走这条路（它的 encoder 输出直接进 decoder），
# [CN] 因此这里显式排除。
def get_ec_connector(
    vllm_config: VllmConfig,
    encoder_cache: "EncoderCache | None",
) -> ECConnector:
    if (
        not has_ec_transfer()
        or vllm_config.model_config.is_encoder_decoder
        or encoder_cache is None
    ):
        return NO_OP_EC_CONNECTOR

    return ActiveECConnector(vllm_config, encoder_cache.encoder_outputs)
