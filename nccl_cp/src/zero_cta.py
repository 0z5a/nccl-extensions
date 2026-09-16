# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Python wrapper for the prebuilt NCCL zero-CTA TorchBind runtime."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from nccl.cp.work import GeneralWork, WorkWithPostProcessFn
from nccl.cp._types import GroupReduceOp, ZeroCtaPlan, ZeroCtaRuntime
from nccl.cp._build_info import validate_manifest

if TYPE_CHECKING:
    from nccl.cp.comm_meta import ZeroCTACollectiveArg


_native_architectures: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _get_extension():
    global _native_architectures
    explicit = os.environ.get("NCCL_CP_LIBRARY")
    candidate = Path(explicit).expanduser() if explicit else Path(__file__).with_name("libnccl_cp.so")
    if not candidate.is_file():
        raise RuntimeError(
            f"NCCL CP native library not found: {candidate}. "
            "Build with CMake, then use the build-tree launcher or set NCCL_CP_LIBRARY."
        )
    library = candidate.resolve()
    manifest = validate_manifest(library)
    torch.ops.load_library(str(library))
    _native_architectures = tuple(str(value) for value in manifest.get("cuda_architectures", ()))
    return torch.ops.nccl_cp_zero_cta_full


def get_or_create_zero_cta_runtime(
    group: dist.ProcessGroup,
    max_per_peer_slot: int,
    nvl_domain_size: int,
    max_per_token_bytes: int,
    runtime_slot: int,
) -> ZeroCtaRuntime:
    """Get/create native Runtime for (group identity, runtime_slot).

    max_per_peer_slot is capacity in tokens; max_per_token_bytes is capacity
    in bytes per token; nvl_domain_size is a number of ranks. Reusing a slot
    requires identical token/domain capacities and sufficient byte capacity.
    See ZeroCtaRuntime in _types.pyi for buffer sizing and lifetime rules.

    Internal caller contract: all group members use matching slot/configuration
    and initialization order on their fixed CUDA devices. First creation may
    allocate/register shared memory and must stay outside data calls/capture.
    The ProcessGroup remains alive until clear_zero_cta_cpp_state completes.
    """

    group_arg = group.boxed() if hasattr(group, "boxed") else group
    return _get_extension().get_or_create_runtime(
        int(max_per_peer_slot),
        int(nvl_domain_size),
        int(max_per_token_bytes),
        int(runtime_slot),
        group_arg,
    )


def _make_device_metadata(
    values: Sequence[tuple[int, ...]],
    op: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    # Explicit CPU staging is required even if the application changes PyTorch's
    # default device. Inputs here are host records produced by the planner.
    if isinstance(values, torch.Tensor) and values.device.type != "cpu":
        raise ValueError("Metadata values must reside on CPU")
    host = torch.tensor(values, dtype=torch.int64, device="cpu", pin_memory=True)
    return op(host)


def _make_device_tensor_gather_tiles(
    values: Sequence[tuple[int, ...]],
) -> torch.Tensor:
    return _make_device_metadata(
        values, _get_extension().make_device_tensor_gather_tiles
    )


def _make_device_symmetric_gather_tiles(
    values: Sequence[tuple[int, ...]],
) -> torch.Tensor:
    return _make_device_metadata(
        values, _get_extension().make_device_symmetric_gather_tiles
    )


def _make_device_reduce_ranges(
    values: Sequence[tuple[int, ...]],
) -> torch.Tensor:
    return _make_device_metadata(values, _get_extension().make_device_reduce_ranges)


def create_zero_cta_cast_plan(
    send_token_counts: tuple[int, ...],
    remote_wait_peers: tuple[int, ...],
    pack_gather_tiles: torch.Tensor,
    post_gather_tiles: torch.Tensor,
    local_ready_signal_peers: tuple[int, ...] = (),
    local_ready_wait_peers: tuple[int, ...] = (),
) -> ZeroCtaPlan:
    """Create a cast Plan from group-relative counts, peers and CUDA metadata.

    send_token_counts has one token count per group rank. Peer tuples identify
    readiness dependencies. pack_gather_tiles is int64[P, 3]; post_gather_tiles
    is int64[G, 3] for direct or int64[G, 4] for hierarchical communication.
    Column meanings and token units are defined by ZeroCtaPlan in _types.pyi.
    """
    return _get_extension().create_cast_plan(
        send_token_counts,
        remote_wait_peers,
        pack_gather_tiles,
        post_gather_tiles,
        local_ready_signal_peers,
        local_ready_wait_peers,
    )


def create_zero_cta_reduce_plan(
    send_token_counts: tuple[int, ...],
    remote_wait_peers: tuple[int, ...],
    pack_gather_tiles: torch.Tensor,
    post_reduce_ranges: torch.Tensor,
    post_reduce_src_token_offsets: torch.Tensor,
    local_reduce_ranges: torch.Tensor | None = None,
    local_reduce_symmetric_srcs: torch.Tensor | None = None,
    local_ready_signal_peers: tuple[int, ...] = (),
    local_ready_wait_peers: tuple[int, ...] = (),
) -> ZeroCtaPlan:
    """Create a sum-reduce Plan from group-relative counts and CUDA metadata.

    pack_gather_tiles is int64[P, 3]; post_reduce_ranges is int64[R, 5];
    post_reduce_src_token_offsets is int64[S]. Optional local_reduce_ranges
    int64[L, 5] and local_reduce_symmetric_srcs int64[T, 2] must be supplied
    together (the builder uses int64[0] for an empty local source list).
    All tensor metadata resides on CUDA. Token starts/lengths use
    token units; source indices/counts refer to source metadata records.
    See ZeroCtaPlan in _types.pyi for the complete field/column definitions.
    """
    return _get_extension().create_reduce_plan(
        send_token_counts,
        remote_wait_peers,
        pack_gather_tiles,
        post_reduce_ranges,
        post_reduce_src_token_offsets,
        local_reduce_ranges,
        local_reduce_symmetric_srcs,
        local_ready_signal_peers,
        local_ready_wait_peers,
    )


def clear_zero_cta_cpp_state(group: dist.ProcessGroup | None = None) -> None:
    """Close C++ registry entries before their ProcessGroups are destroyed.

    Internal caller contract: stop submissions and coordinate teardown across
    all participating ranks. This may wait for GPU work and is not a recovery
    primitive for a failed peer. Public callers use close_runtime instead.
    """

    if _get_extension.cache_info().currsize == 0:
        return
    if group is None:
        _get_extension().close_all_runtimes()
        return
    group_arg = group.boxed() if hasattr(group, "boxed") else group
    _get_extension().close_runtime(group_arg)


def _run(
    input: torch.Tensor,
    output: torch.Tensor,
    collective_arg: ZeroCTACollectiveArg,
    plan: ZeroCtaPlan,
    async_op: bool,
) -> WorkWithPostProcessFn:
    # Direct internal callers must order this stream after metadata/input
    # producers and retain work/data/plans until completion. The immediate
    # public API invokes wait_post_process itself; *_async returns this
    # work adapter so the caller chooses when to submit post-processing.
    work = _get_extension().run(input, output, collective_arg.runtime, plan)
    return WorkWithPostProcessFn(
        work=GeneralWork(work=work),
        post_process_fn=lambda *_args, **_kwargs: output,
        async_op=async_op,
    )


def zero_cta_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor,
    collective_arg: ZeroCTACollectiveArg,
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> WorkWithPostProcessFn:
    if cast_lse or input_lse is not None or output_lse is not None:
        raise NotImplementedError("zero-CTA group-cast does not support LSE")
    if output is None:
        raise ValueError("zero-CTA group-cast requires an output tensor")
    return _run(input, output, collective_arg, collective_arg.cast_plan, async_op)


def zero_cta_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor,
    collective_arg: ZeroCTACollectiveArg,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> WorkWithPostProcessFn:
    if reduce_op != "sum":
        raise NotImplementedError("zero-CTA group-reduce supports only sum")
    if not acc_reduce:
        raise NotImplementedError("zero-CTA group-reduce requires acc_reduce=True")
    if comm_dtype is not None and comm_dtype != input.dtype:
        raise NotImplementedError(
            "zero-CTA group-reduce does not support dtype conversion"
        )
    if input_lse is not None or output_lse is not None:
        raise NotImplementedError("zero-CTA group-reduce does not support LSE")
    if output is None:
        raise ValueError("zero-CTA group-reduce requires an output tensor")
    return _run(input, output, collective_arg, collective_arg.reduce_plan, async_op)


__all__ = [
    "clear_zero_cta_cpp_state",
    "get_or_create_zero_cta_runtime",
    "zero_cta_group_cast_impl",
    "zero_cta_group_reduce_impl",
]
