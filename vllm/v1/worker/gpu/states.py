# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：新一代 runner（V2）的请求状态容器。
# [CN] 与 v1/worker/gpu_input_batch.py 的 InputBatch 对比：
# [CN]   这里把「每请求的持久状态」单独抽成 RequestState，
# [CN]   且大张量（all_token_ids）刻意放在 UVA（统一虚拟寻址）内存而非显存，
# [CN]   因为 max_num_reqs × max_model_len 可能有好几 GB。
# [CN] 关键抽象：
# [CN]   StagedWriteTensor —— 先在 CPU 侧 stage，再一次性 apply 到 GPU；
# [CN]   UvaBackedTensor   —— 直接映射到 UVA，CPU/GPU 同址访问。
import numpy as np
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


# [CN] 全部请求的持久状态（按 slot 索引，不是按 req_id 连续存放）。
# [CN] slot 通过 free_indices 池复用，避免增删请求时移动数据。
class RequestState:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
        num_prefill_lookahead: int = 1,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # [CN] all_token_ids 可能数 GB，用 UVA 而非显存，省显存给 KV cache。
        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # depending on the configured max_num_reqs and max_model_len.
        # To save GPU memory, we use UVA instead of GPU for this tensor.
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        # [CN] prompt_len 与 prefill_len 必须分清：
        # [CN]   prompt_len  —— 用户给的 prompt 长度；
        # [CN]   prefill_len —— 真正送进 runner 的长度（含抢占恢复时续上的输出 token）。
        # [CN] prompt logprobs、频率惩罚等特性必须按 prompt 与输出分开处理，
        # [CN] 混用会导致语义错误。
        # NOTE(woosuk): Distinguish clearly between prompt_len and prefill_len:
        # - prompt_len: Number of tokens in the user-provided prompt.
        # - prefill_len: Number of tokens passed into the model runner.
        #   This can include the prompt and additional partial output tokens,
        #   so prefill_len >= prompt_len.
        # Usually, prefill_len equals prompt_len, but in cases such as resumption after
        # preemption, prefill_len may be greater. Differentiating between these values
        # is crucial, as certain features such as prompt logprobs or frequency penalties
        # must treat prompt and output tokens separately.
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # total_len = prompt_len + output_len. It grows as the request progresses.
        self.total_len = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # [CN] CPU 侧的「乐观镜像」：它是 GPU 真值的上界。
        # [CN] async scheduling 下 GPU 真值可能更小（草稿被拒），CPU 先按上界算，
        # [CN] 再由 GPU 侧修正，从而全程无需同步。
        # Optimistic CPU mirror of num_computed_tokens (upper bound on GPU value).
        self.num_computed_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Last sampled tokens.
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )

        # Max total seq length (prompt_len + max_tokens).
        self.max_seq_len = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Draft tokens.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        self.next_prefill_tokens = torch.zeros(
            num_prefill_lookahead,
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    # [CN] 从 free_indices 取一个 slot 挂上；prefill_len 来自 all_token_ids 长度。
    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.max_seq_len[req_idx] = prompt_len + max_tokens
        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.total_len.stage_write_elem(req_idx, prefill_len)
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

        self.draft_tokens[req_idx].zero_()

    # [CN] 把本步 stage 的所有写一次性提交（H2D / UVA 拷贝）。
    # [CN] 集中提交的目的是把多次小拷贝合并，减少 launch 开销。
    def apply_staged_writes(self) -> None:
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    # [CN] 归还 slot 到 free_indices；返回被释放的槽位下标供调用方清理其它数组。
    def remove_request(self, req_id: str) -> int | None:
        """Return the freed slot index, or None if the request was not found."""
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            return None
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)
        return req_idx
