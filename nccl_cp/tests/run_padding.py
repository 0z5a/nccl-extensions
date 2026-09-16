"""Focused multi-rank padding checks for prepared and explicit CP entry points."""
import argparse
import json
import os
from dataclasses import replace

import torch
import torch.distributed as dist
from nccl.cp import (
    CpConfig,
    close_runtime,
    create_group,
    create_handle,
    group_cast,
    group_cast_explicit,
    group_reduce,
    group_reduce_explicit,
)
from nccl.cp.routing import layout_intervals, local_route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['gloo', 'nccl'], required=True)
    parser.add_argument('--require-zero', action='store_true')
    parser.add_argument("--nvl-domain-size", type=int, required=True,
                        help="Ranks per NVL domain; supplied by the caller")
    args = parser.parse_args()
    if args.nvl_domain_size <= 0:
        parser.error("--nvl-domain-size must be positive")
    options = None
    if args.backend == 'nccl':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        options = dist.ProcessGroupNCCL.Options()
        options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        device = torch.device('cuda', torch.cuda.current_device())
        stream = torch.cuda.current_stream()
    else:
        device, stream = torch.device('cpu'), None
    dist.init_process_group(args.backend, pg_options=options)
    pg = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    relay = args.nvl_domain_size if world > args.nvl_domain_size > 1 else None
    owned = [[100 * r + i for i in range(3)] if r != relay else [] for r in range(world)]
    requested = [[token for tokens in owned for token in tokens] if r != relay else [] for r in range(world)]
    owned[0].append(99)  # Unrequested owner row must retain its accumulator.
    layouts = [(layout_intervals(a), layout_intervals(b)) for a, b in zip(owned, requested)]
    routes = [local_route(r, layouts) for r in range(world)]
    ins = [list(r.input_splits) for r in routes]
    outs = [list(r.output_splits) for r in routes]
    dst = [[list(peers) for peers in r.destinations] for r in routes]
    src = [list(r.sources) for r in routes]
    config = CpConfig(max_per_peer_slot=4, payload_shape=(8,), dtype=torch.float32,
                      nvl_domain_size=args.nvl_domain_size)
    group = create_group(pg, config)
    flat_group = create_group(pg, replace(config, backend='all2allv'))
    handle = create_handle(group, owned[rank], requested[rank], stream=stream)
    flat_handle = create_handle(flat_group, owned[rank], requested[rank], stream=stream)
    handles = [handle, flat_handle]
    checks = 0
    if args.require_zero:
        assert handle.backend == 'zero_cta', handle.fallback_reason
    def values(tokens, dtype):
        ids = torch.tensor(tokens, device=device, dtype=torch.float32).reshape(-1, 1)
        return (ids.remainder(97) * .5 + torch.arange(8, device=device) * .125).to(dtype)
    try:
        for dtype in (torch.float32, torch.bfloat16):
            input = values(owned[rank], dtype)
            expected = values(requested[rank], dtype)
            count = len(requested[rank])
            contribution = torch.tensor([sum(token in wanted for wanted in requested) for token in owned[rank]],
                                        device=device, dtype=dtype).reshape(-1, 1).expand_as(input)
            for explicit, cp_group, prepared in ((False, group, handle), (False, flat_group, flat_handle),
                                                  (True, group, handle), (True, flat_group, flat_handle)):
                output = torch.full((count + world * config.max_per_peer_slot + rank + 1, 8), -19., device=device, dtype=dtype)
                pointer = output.data_ptr()
                grad_input = torch.full_like(output, float('nan'))
                grad_input[:count].fill_(1)
                grad_output = torch.full_like(input, 7)
                for iteration in range(2):
                    if explicit:
                        ret = group_cast_explicit(input, output, ins, outs, dst, src, group=cp_group, stream=stream)
                    else:
                        ret = group_cast(prepared, input, output, stream=stream)
                    assert ret is None and output.data_ptr() == pointer
                    torch.testing.assert_close(output[:count], expected, rtol=0, atol=0)
                    torch.testing.assert_close(output[count:], torch.full_like(output[count:], -19.), rtol=0, atol=0)
                    if explicit:
                        ret = group_reduce_explicit(grad_input, grad_output, outs, ins, src, dst, group=cp_group, stream=stream)
                    else:
                        ret = group_reduce(prepared, grad_input, grad_output, stream=stream)
                    assert ret is None
                    torch.testing.assert_close(grad_output, 7 + (iteration + 1) * contribution, rtol=0, atol=0)
                    torch.testing.assert_close(grad_input[count:], torch.full_like(grad_input[count:], float('nan')),
                                               rtol=0, atol=0, equal_nan=True)
                    checks += 1
        # Globally empty route with nonempty receive-buffer capacity.
        for cp_group in (group, flat_group):
            empty = create_handle(cp_group, [], [], stream=stream)
            handles.append(empty)
            input = torch.empty((0, 8), device=device)
            output = torch.full((3, 8), -23., device=device)
            assert group_cast(empty, input, output, stream=stream) is None
            assert group_cast_explicit(input, output, *([[] for _ in range(world)] for _ in range(4)),
                                       group=cp_group, stream=stream) is None
            torch.testing.assert_close(output, torch.full_like(output, -23.), rtol=0, atol=0)
            poisoned = torch.full_like(output, float('nan'))
            assert group_reduce(empty, poisoned, input, stream=stream) is None
            assert group_reduce_explicit(poisoned, input, *([[] for _ in range(world)] for _ in range(4)),
                                         group=cp_group, stream=stream) is None
            checks += 1
        if rank == 0:
            print(json.dumps(dict(passed=True, backend=handle.backend,
                                  checks_per_rank=checks, dtypes=['float32', 'bfloat16'],
                                  scope='cast tail preserved; reduce ignores poisoned tail; prepared/explicit; native/flat; empty prefixes')), flush=True)
    finally:
        if stream is not None:
            stream.synchronize()
        for item in handles:
            item.close()
        close_runtime(group)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
