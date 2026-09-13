# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：DBO 的「切分」侧 —— 把一步的 batch 切成 N 个微批次。
# [CN] 与本目录 ubatching.py 的分工：
# [CN]   - ubatching.py 管「怎么让两个 ubatch 交替执行」（线程与 stream 同步）；
# [CN]   - 本文件管「怎么切」（请求/token 切片、attention metadata 重建、SM 划分）。
# [CN] 核心概念：
# [CN]   - UBatchSlice：一对 slice，同时描述「包含哪些请求」和「包含哪些 token」；
# [CN]   - split_point：切分点，默认按 padding 后的 token 数均分；
# [CN]   - splits_first/last_request：一个请求被切到两个 ubatch 时的边界处理。
# [CN] 最容易看错的点：
# [CN]   1) 存在 padded 与 unpadded 两套切片 —— padded 版用于 CUDA Graph（形状固定），
# [CN]      unpadded 版用于真实计算；
# [CN]   2) _make_metadata_with_slice 里凡是涉及 seq_lens 的修改都必须先 clone，
# [CN]      否则会就地改掉原 tensor，CUDA Graph 捕获到的形状就错了。

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.distributed import get_ep_group
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import set_num_sms as deep_gemm_set_num_sms
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import CommonAttentionMetadata


# [CN] 一个微批次的切片：请求维度与 token 维度各一个 slice。
@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice

    # [CN] 空切片判定：任一维度长度为 0 即为空（DP 下最后一片经常是空的）。
    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start


UBatchSlices: TypeAlias = list[UBatchSlice]


# [CN] SM 划分上下文：进入时把 SM 切成「通信用 comm_sms」+「计算用 total-comm」，
# [CN] 退出时全部归还。目的是让通信 kernel 与计算 kernel 真正并发而不互相抢 SM。
class SMControlContextManager:
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ):
        """
        Context manager for controlling SM (Streaming Multiprocessor)
        allocation. Upon entering the context, it sets the number of SMs
        allocated for communication and computation to comm_sms and
        total_sms - comm_sms respectively. Upon exiting, it restores the
        allocation to use all available SMs (i.e. total_sms).

        Args:
            comm_sms (int): The number of SMs to allocate for communication.
                (The remainder will be used for computation.)
            set_comm_sms (Callable[[int], None]):
                A function that sets the number of SMs for communication.
            set_compute_sms (Callable[[int], None]):
                A function that sets the number of SMs for computation.
        """

        # [CN] 仅 CUDA/ROCm 支持 SM 控制；其他平台直接断言失败。
        assert current_platform.is_cuda() or current_platform.is_rocm(), (
            "SM/CU control is supported on CUDA and ROCm platforms"
        )
        device = torch.accelerator.current_device_index()
        total_sms = num_compute_units(device)

        assert comm_sms < total_sms
        self.total_sms = total_sms
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms

    # [CN] 进入/退出只改两个 setter，不做实际通信 —— setter 由外部注入（见下）。
    def __enter__(self):
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, exc_type, exc_value, traceback):
        self.set_comm_sms(self.total_sms)
        self.set_compute_sms(self.total_sms)


# [CN] 根据并行配置构造 SM 控制器。默认两个 setter 都是 no-op（lambda sms: None）。
def create_sm_control_context(
    parallel_config: ParallelConfig,
) -> SMControlContextManager:
    """Reserve SMs for communication kernels while microbatches overlap."""
    # [CN] 默认预留的通信 SM 数来自环境变量 VLLM_DBO_COMM_SMS。
    comm_sms: int = envs.VLLM_DBO_COMM_SMS
    rocm_deepep_ht_dbo = (
        current_platform.is_rocm()
        and parallel_config.enable_dbo
        and parallel_config.all2all_backend == "deepep_high_throughput"
    )
    # [CN] ROCm + DeepEP 高吞吐 + DBO 的组合下，预留 CU 会导致精度错误，
    # [CN] 这里直接置 0（保留后端但不预留），是一种保守降级。
    if rocm_deepep_ht_dbo:
        # On ROCm, reserving CUs for DeepEP HT communication under DBO
        # corrupts DP+EP generation accuracy. Keep the backend active, but
        # leave all CUs visible to the compute and communication kernels.
        comm_sms = 0

    set_comm_sms = lambda sms: None
    # [CN] 目前只有 DeepEP high-throughput 后端支持 SM 控制，因此通信侧划分只影响它。
    if parallel_config.enable_expert_parallel:
        # Currently only DeepEP highthroughput supports SM control so this
        # only affects that case.
        ep_group = get_ep_group()
        device_communicator = ep_group.device_communicator
        all2all_manager = None
        if device_communicator is not None:
            all2all_manager = device_communicator.all2all_manager

        # [CN] 不能超过后端实际使用的最大 SM 数，取 min 做上界。
        if all2all_manager is not None:
            max_sms_used = all2all_manager.max_sms_used()
            if max_sms_used is not None:
                comm_sms = min(comm_sms, max_sms_used)

        if comm_sms > 0 and all2all_manager is not None:
            set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

    # [CN] 计算侧目前只接了 DeepGEMM（TODO：扩展到更多 kernel）。
    # [CN] 若没有 deep_gemm，计算侧划分就是 no-op，只有通信侧被限制。
    # TODO(lucas): support other kernels besides DeepGEMM
    set_compute_sms = lambda sms: None
    if has_deep_gemm() and comm_sms > 0:
        set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)

    return SMControlContextManager(
        comm_sms=comm_sms,
        set_comm_sms=set_comm_sms,
        set_compute_sms=set_compute_sms,
    )


# [CN] 判断最后一个 ubatch 是否为空：均分后前 N-1 片是否已覆盖全部真实 token。
def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


# [CN] 微批次数：未开启 ubatching 时恒为 1。
def get_num_ubatches(parallel_config: ParallelConfig) -> int:
    """How many microbatches a step is split into; 1 when microbatching is off."""
    return parallel_config.num_ubatches if parallel_config.use_ubatching else 1


# [CN] 是否值得开 DBO：token 数要达到阈值才划算，
# [CN] 且 decode（均匀 1 token/请求）与 prefill 用不同阈值。
def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
# [CN] 把最后一片的边界扩到 padding 后的总量。
# [CN] 注意调用顺序：切分发生在「DP padding 之前」，所以要在这里补回来。
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


# [CN] 主入口：返回 (真实切片, padding 后切片) 两个版本；未开启 DBO 时返回 (None, None)。
def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    # [CN] 未指定切分点时按 padding 后 token 数均分。
    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    # [CN] 由「每请求 token 数」前缀和出 query_start_loc（即 cu_num_tokens）。
    # [CN] TODO：理想情况是让 runner 直接把 cu_num_tokens 传进来，省一次计算。
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0

    # Add the end point to the split points to make iteration easier
    all_points = token_split_points + [cu_num_tokens[-1]]

    # [CN] 逐段生成切片。请求边界用 searchsorted 在 cu_num_tokens 上二分。
    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)

        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        # [CN] side='right' -1：找到「包含 start_token 的那个请求」的下标；
        # [CN] side='left'：找到「第一个起点 >= end_token 的请求」。两者构成左闭右开区间。
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    # [CN] 不变式：padding 版各片 token 数之和必须等于 padding 后的总 token 数。
    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded


# [CN] 切出子区间的 query_start_loc，并重新归零起点（减掉首元素）。
# [CN] 注意 stop+1：query_start_loc 比请求数多一个元素（含末端哨兵）。
# [CN] 警告：本函数会新建 tensor，因此不能用在 CUDA Graph 捕获路径里。
def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


# [CN] 核心函数：为单个 ubatch 切片重建一份 CommonAttentionMetadata。
# [CN] 不修改传入的原始 metadata（涉及修改的一律先 clone）。
def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """

    # [CN] 空切片无法构造合法 metadata（query_start_loc 至少要 2 个元素）。
    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    # [CN] 校验切分点确实落在第一个请求的 token 区间内。
    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # [CN] 注意：last token 可以越出最后一个请求 —— CUDA Graph padding 会导致这种情况。
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    # [CN] splits_first_request：本片第一个请求是从上一片延续过来的（前半段在别处）；
    # [CN] splits_last_request：本片最后一个请求还要延续到下一片。
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    # [CN] 首请求被切断：本片的 query_start_loc 要整体偏移，减掉「前面已算过的 token 数」。
    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    # [CN] 读原始字段（带下划线前缀）而不是 property：property 会触发 D2H 同步，
    # [CN] 在热路径上是灾难性的性能问题。
    seq_lens = attn_metadata.seq_lens[request_slice]
    # Read raw fields to avoid triggering the deprecated D2H-syncing properties.
    seq_lens_cpu = (
        attn_metadata._seq_lens_cpu[request_slice]
        if attn_metadata._seq_lens_cpu is not None
        else None
    )
    seq_lens_cpu_upper_bound = (
        attn_metadata.seq_lens_cpu_upper_bound[request_slice]
        if attn_metadata.seq_lens_cpu_upper_bound is not None
        else None
    )
    num_computed_tokens_cpu = (
        attn_metadata._num_computed_tokens_cpu[request_slice]
        if attn_metadata._num_computed_tokens_cpu is not None
        else None
    )

    # [CN] 尾请求被切断：把末尾多出来的 token 从本片剔除。
    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        # [CN] 必须用原始 start_locs 计算 —— query_start_loc_cpu 可能已被上一步修改过。
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # [CN] clone 后再改：就地修改会破坏原 tensor，而 CUDA Graph 捕获依赖形状固定。
        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped
        if seq_lens_cpu_upper_bound is not None:
            seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound.clone()
            seq_lens_cpu_upper_bound[-1] -= tokens_skipped

    # [CN] max_seq_len 取「本片实际上界」与「原 metadata 的 max_seq_len」的较大者，
    # [CN] 以保留 CUDA Graph 捕获时设的 override，否则 SWA 层会选错 kernel。
    assert seq_lens_cpu_upper_bound is not None
    # Preserve the max_seq_len override set during CUDA-graph capture so
    # the attention backend selects the correct kernel for SWA layers.
    max_seq_len = max(int(seq_lens_cpu_upper_bound.max()), attn_metadata.max_seq_len)

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # [CN] dummy run 时 query_start_loc_cpu 全是 0，max_query_len 会算成 0，
    # [CN] 此时退回用原 metadata 的值，避免构造出非法的 attention metadata。
    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


# [CN] 批量版：为每个切片各生成一份 metadata，供两个 ubatch 线程各自使用。
def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results
