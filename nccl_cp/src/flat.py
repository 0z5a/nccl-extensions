"""Flat all2allv execution using caller-owned input/output tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from .routing import Segment
from .work import WorkWithPostProcessFn


def run(input: torch.Tensor, output: torch.Tensor, send: Sequence[Sequence[Segment]],
        recv: Sequence[Sequence[Segment]], group: dist.ProcessGroup, *, reduce: bool):
    # Internal caller contract: peer counts/order and payload formats must match
    # across ranks, and the current stream must already depend on data producers.
    # Packing here is device-local; it does not establish producer dependencies.
    # Temporaries are allocated/consumed on this stream. The NCCL dependency
    # below orders subsequent use/reuse on their allocation stream; the caller
    # retains external input/output storage until completion.
    send_counts = [sum(s.count for s in parts) for parts in send]
    recv_counts = [sum(s.count for s in parts) for parts in recv]
    packed = torch.empty((sum(send_counts), *input.shape[1:]), device=input.device, dtype=input.dtype)
    received = torch.empty((sum(recv_counts), *input.shape[1:]), device=input.device, dtype=input.dtype)
    offset = 0
    for parts in send:
        for part in parts:
            packed.narrow(0, offset, part.count).copy_(input.narrow(0, part.start, part.count))
            offset += part.count
    work = dist.all_to_all_single(received, packed, recv_counts, send_counts, group=group, async_op=True)
    if input.device.type == "cuda":
        # Establish a GPU stream dependency even when NCCL blocking wait is enabled.
        work.block_current_stream()
    else:
        work.wait()
    offset = 0
    for parts in recv:
        for part in parts:
            target = output.narrow(0, part.start, part.count)
            source = received.narrow(0, offset, part.count)
            if reduce:
                # Caller initialized this accumulator. Floating-point rounding
                # can differ from native/hierarchical reduction grouping.
                target.add_(source)
            else:
                target.copy_(source)
            offset += part.count
    return packed, received, work


def run_async(input: torch.Tensor, output: torch.Tensor, send: Sequence[Sequence[Segment]],
              recv: Sequence[Sequence[Segment]], group: dist.ProcessGroup, *,
              reduce: bool) -> WorkWithPostProcessFn:
    """Pack and launch all2allv; defer its dependency and output writes to wait.

    Caller keeps one current stream for launch/wait and retains external data
    through GPU completion. The callback owns communication temporaries until
    it submits post-processing on their allocation stream. The run()
    path is unchanged, so immediate calls do not allocate this work adapter.
    """
    send_counts = [sum(s.count for s in parts) for parts in send]
    recv_counts = [sum(s.count for s in parts) for parts in recv]
    packed = torch.empty((sum(send_counts), *input.shape[1:]), device=input.device, dtype=input.dtype)
    received = torch.empty((sum(recv_counts), *input.shape[1:]), device=input.device, dtype=input.dtype)
    offset = 0
    for parts in send:
        for part in parts:
            packed.narrow(0, offset, part.count).copy_(input.narrow(0, part.start, part.count))
            offset += part.count
    work = dist.all_to_all_single(received, packed, recv_counts, send_counts, group=group, async_op=True)

    def post_process(*_args, **_kwargs):
        # Keep the send buffer captured until the transfer dependency is joined.
        # All temporaries were allocated on the caller's fixed launch/wait stream.
        _ = packed
        with torch.no_grad():
            if input.device.type == "cuda":
                work.block_current_stream()
            else:
                work.wait()
            offset = 0
            for parts in recv:
                for part in parts:
                    target = output.narrow(0, part.start, part.count)
                    source = received.narrow(0, offset, part.count)
                    if reduce:
                        target.add_(source)
                    else:
                        target.copy_(source)
                    offset += part.count
        return output

    return WorkWithPostProcessFn(post_process_fn=post_process, async_op=True)
