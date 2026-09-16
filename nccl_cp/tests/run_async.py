"""Verify launch/compute/wait ordering and deferred output on real CP backends."""
import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.distributed as dist
from nccl.cp import (
    CpConfig,
    WorkWithPostProcessFn,
    close_runtime,
    create_group,
    create_handle,
    group_cast,
    group_cast_async,
    group_cast_explicit_async,
    group_reduce,
    group_reduce_async,
    group_reduce_explicit_async,
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
        device, stream = torch.device('cuda', torch.cuda.current_device()), torch.cuda.Stream()
    else:
        device, stream = torch.device('cpu'), None
    dist.init_process_group(args.backend, pg_options=options)
    pg, rank, world = dist.group.WORLD, dist.get_rank(), dist.get_world_size()
    handles = []
    try:
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            relay = args.nvl_domain_size if world > args.nvl_domain_size > 1 else None
            owned = [[100 * r + i for i in range(3)] if r != relay else [] for r in range(world)]
            requested = [[token for tokens in owned for token in tokens] if r != relay else [] for r in range(world)]
            owned[0].append(99)
            layouts = [(layout_intervals(a), layout_intervals(b)) for a, b in zip(owned, requested)]
            routes = [local_route(r, layouts) for r in range(world)]
            ins, outs = [list(r.input_splits) for r in routes], [list(r.output_splits) for r in routes]
            dst, src = [[list(p) for p in r.destinations] for r in routes], [list(r.sources) for r in routes]
            config = CpConfig(max_per_peer_slot=4, payload_shape=(8,), dtype=torch.float32,
                              nvl_domain_size=args.nvl_domain_size)
            group = create_group(pg, config)
            flat_group = create_group(pg, replace(config, backend='all2allv'))
            explicit_group = create_group(pg, replace(config, runtime_slot=1))
            handle = create_handle(group, owned[rank], requested[rank], stream=stream)
            flat_handle = create_handle(flat_group, owned[rank], requested[rank], stream=stream)
            handles.extend((handle, flat_handle))
            if args.require_zero:
                assert handle.backend == 'zero_cta', handle.fallback_reason
            checks = 0
            def values(tokens, dtype):
                ids = torch.tensor(tokens, device=device, dtype=torch.float32).reshape(-1, 1)
                return (ids.remainder(97) * .5 + torch.arange(8, device=device) * .125).to(dtype)
            for dtype in (torch.float32, torch.bfloat16):
                input, expected = values(owned[rank], dtype), values(requested[rank], dtype)
                count = len(requested[rank])
                contribution = torch.tensor([sum(token in rows for rows in requested) for token in owned[rank]],
                                            device=device, dtype=dtype).reshape(-1, 1).expand_as(input)
                for mode, cp_group, prepared in [('handle', group, handle), ('handle', flat_group, flat_handle),
                                                 ('explicit', explicit_group, handle), ('explicit', flat_group, flat_handle)]:
                    output = torch.full((count + rank + 3, 8), -19., device=device, dtype=dtype)
                    pointer = output.data_ptr()
                    if mode == 'handle':
                        work = prepared.group_cast_async(input, output, stream=stream)
                    else:
                        work = group_cast_explicit_async(input, output, ins, outs, dst, src, group=cp_group, stream=stream)
                    assert isinstance(work, WorkWithPostProcessFn) and work.async_op
                    # The native check must reject reusing a pending slot.
                    if mode == 'handle' and prepared.backend == 'zero_cta':
                        try:
                            group_cast_async(prepared, input, output, stream=stream)
                        except RuntimeError as error:
                            assert 'requires waiting the previous work' in str(error)
                        else:
                            raise AssertionError('Native pending-work guard did not fire')
                    # Independent work is submitted before the completion call.
                    independent = torch.arange(4096, device=device, dtype=torch.float32).mul_(2).add_(3)
                    scratch = [torch.empty((n,), device=device, dtype=torch.int64) for n in (1, 8, 64, 512, 4096)]
                    for buffer in scratch:
                        buffer.fill_(-1)
                    # This observes output on the caller stream before any final gather.
                    torch.testing.assert_close(output, torch.full_like(output, -19.), rtol=0, atol=0)
                    assert work.wait_post_process() is output and output.data_ptr() == pointer
                    torch.testing.assert_close(output[:count], expected, rtol=0, atol=0)
                    torch.testing.assert_close(output[count:], torch.full_like(output[count:], -19.), rtol=0, atol=0)
                    torch.testing.assert_close(independent, torch.arange(4096, device=device) * 2 + 3, check_dtype=False)
                    try:
                        work.wait_post_process()
                    except RuntimeError as error:
                        assert 'already been done' in str(error)
                    else:
                        raise AssertionError('Work was used twice')
                    grad_input = torch.full_like(output, float('nan'))
                    grad_input[:count].fill_(1)
                    grad_output = torch.full_like(input, 7)
                    for iteration in range(2):
                        if mode == 'handle':
                            work = group_reduce_async(prepared, grad_input, grad_output, stream=stream,
                                                      reduce_op="sum", acc_reduce=True, comm_dtype=dtype)
                        else:
                            work = group_reduce_explicit_async(grad_input, grad_output, outs, ins, src, dst,
                                                               group=cp_group, stream=stream, reduce_op="sum",
                                                               acc_reduce=True, comm_dtype=dtype)
                        before = 7 + iteration * contribution
                        torch.testing.assert_close(grad_output, before, rtol=0, atol=0)
                        independent.add_(1)
                        assert work.wait_post_process() is grad_output
                        torch.testing.assert_close(grad_output, 7 + (iteration + 1) * contribution, rtol=0, atol=0)
                    # Rejected reserved options must leave the buffer unchanged
                    # and must not occupy the slot or start a collective.
                    for options in ({'reduce_op': 'avg'}, {'reduce_op': 'lse'},
                                    {'acc_reduce': False}, {'comm_dtype': torch.float64},
                                    {'input_lse': torch.empty(0, device=device)},
                                    {'output_lse': torch.empty(0, device=device)}):
                        try:
                            if mode == 'handle':
                                prepared.group_reduce_async(grad_input, grad_output, stream=stream, **options)
                            else:
                                group_reduce_explicit_async(grad_input, grad_output, outs, ins, src, dst,
                                                            group=cp_group, stream=stream, **options)
                        except NotImplementedError:
                            pass
                        else:
                            raise AssertionError('Unsupported reduce option was accepted')
                    torch.testing.assert_close(grad_output, 7 + 2 * contribution, rtol=0, atol=0)
                    checks += 1
            if args.require_zero:
                assert 1 in explicit_group._state.runtimes
            # Immediate calls remain usable after async completion.
            assert group_cast(handle, input, output, stream=stream) is None
            assert group_reduce(handle, grad_input, grad_output, stream=stream) is None
            # Empty global routes still return a single-use work, including flat no-ops.
            for cp_group in (group, flat_group):
                empty = create_handle(cp_group, [], [], stream=stream)
                handles.append(empty)
                input = torch.empty((0, 8), device=device)
                output = torch.full((3, 8), -23., device=device)
                for mode in ('handle', 'explicit'):
                    work = (group_cast_async(empty, input, output, stream=stream) if mode == 'handle' else
                            group_cast_explicit_async(input, output, *([[] for _ in range(world)] for _ in range(4)), group=cp_group, stream=stream))
                    assert work.wait_post_process() is output
                    torch.testing.assert_close(output, torch.full_like(output, -23.), rtol=0, atol=0)
                    work = (group_reduce_async(empty, output, input, stream=stream) if mode == 'handle' else
                            group_reduce_explicit_async(output, input, *([[] for _ in range(world)] for _ in range(4)), group=cp_group, stream=stream))
                    assert work.wait_post_process() is input
                checks += 1
            if rank == 0:
                print(json.dumps(dict(passed=True, backend=handle.backend,
                                      checks_per_rank=checks, dtypes=['float32','bfloat16'],
                                      scope='deferred output/accumulation; single-use wait; pending-slot guard; nondefault stream; handle/explicit native/flat; padding; empty')), flush=True)
    finally:
        if stream is not None:
            stream.synchronize()
        for handle in handles:
            handle.close()
        if handles:
            close_runtime(handles[0].group)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
