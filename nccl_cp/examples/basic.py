"""Minimal NCCL CP example with caller-owned tensors and an explicit stream."""

import argparse
import os

import torch
import torch.distributed as dist
from nccl.cp import (
    CpConfig,
    RowRange,
    close_runtime,
    create_group,
    create_handle,
    group_cast,
    group_reduce,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--nvl-domain-size", type=int, required=True,
                    help="Ranks per NVL domain for this ProcessGroup")
args = parser.parse_args()
if args.nvl_domain_size <= 0:
    parser.error("--nvl-domain-size must be positive")

# The launcher assigns LOCAL_RANK; the example does not choose physical GPUs.
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
stream = torch.cuda.current_stream()
options = dist.ProcessGroupNCCL.Options()
if hasattr(options.config, "cta_policy") and hasattr(dist.ProcessGroupNCCL, "NCCL_CTA_POLICY_ZERO"):
    options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
dist.init_process_group("nccl", pg_options=options)
nccl_group = dist.group.WORLD
rank, world = dist.get_rank(), dist.get_world_size()
cp_config = CpConfig(max_per_peer_slot=2, payload_shape=(4,), dtype=torch.float32,
                     nvl_domain_size=args.nvl_domain_size)
cp_group = create_group(nccl_group, cp_config)
# All group members prepare matching layouts/order. The tensors below must
# contain the described communication rows, including the correct payload width.
handle = create_handle(
    cp_group,
    local_owned_layout=[RowRange(2 * rank, 2 * rank + 2)],
    local_required_layout=[RowRange(0, 2 * world)],
    stream=stream,
)
input = torch.arange(2 * rank, 2 * rank + 2, device="cuda", dtype=torch.float32)[:, None].expand(-1, 4).contiguous()
output = torch.empty((2 * world, 4), device="cuda")
grad_input = torch.ones_like(output)
grad_output = torch.zeros_like(input)
try:
    for _ in range(2):
        # Producers, CP operations and checks use this same stream here. With
        # separate streams, the caller adds producer/consumer wait dependencies
        # and makes the CP stream current around the whole submission loop.
        group_cast(handle, input, output, stream=stream)
        expected = torch.arange(2 * world, device="cuda", dtype=input.dtype)[:, None].expand_as(output)
        torch.testing.assert_close(output, expected)
        grad_output.zero_()  # Caller initialization: group_reduce accumulates.
        group_reduce(handle, grad_input, grad_output, stream=stream)
        torch.testing.assert_close(grad_output, torch.full_like(input, world))
    if rank == 0:
        print(f"NCCL CP example passed; backend={handle.backend}")
finally:
    # Caller keeps data and plans alive until this stream finishes, then
    # coordinates native teardown on every rank after all submissions stop.
    stream.synchronize()
    handle.close()
    close_runtime(cp_group)
    dist.destroy_process_group()
