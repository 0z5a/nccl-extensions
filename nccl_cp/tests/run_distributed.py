"""Validate distributed NCCL CP communication against known tensor values."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from nccl.cp import comm_meta as meta
from nccl.cp import zero_cta as backend
from route_cases import expected_cast, expected_reduce, kwargs_for, make_case, relay_case


def nccl_options():
    options = dist.ProcessGroupNCCL.Options()
    options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    return options


def run_collectives(module, backend, group, cases, capacity, rank, world, nvl, dtype, stream):
    with torch.cuda.stream(stream):
        for case_id, route in enumerate(cases):
            case = dict(route, capacity=capacity)
            arg = module.ZeroCTACollectiveArg(
                **kwargs_for(
                    module,
                    case,
                    rank,
                    world,
                    group,
                    slot=case_id % 2,
                    payload_bytes=64,
                ),
                nvl_domain_size=nvl,
            )
            tokens = sum(case["inputs"][rank])
            received = sum(case["outputs"][rank])
            for width in (3, 5):
                # Exercise supported non-dense token strides.
                storage = torch.empty((tokens, width + 2), dtype=dtype, device="cuda")
                input = storage[:, :width]
                values = torch.arange(tokens, dtype=torch.float32, device="cuda") + rank * 10000
                input.copy_(values.to(dtype).view(-1, 1).expand(-1, width))
                out_storage = torch.empty((received, width + 2), dtype=dtype, device="cuda")
                output = out_storage[:, :width]
                expected = (
                    torch.tensor(expected_cast(case, rank), dtype=dtype, device="cuda")
                    .view(-1, 1)
                    .expand(-1, width)
                )
                # Reuse the exact prepared metadata, both asynchronous and default work paths.
                for async_op in (True, False, True):
                    work = backend.zero_cta_group_cast_impl(
                        input, output, arg, group, async_op=async_op
                    )
                    assert work.wait_post_process(output) is output
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)
                    grad = torch.full_like(output, rank + 1)
                    local_grad = torch.full_like(input, 7)
                    work = backend.zero_cta_group_reduce_impl(
                        grad, local_grad, arg, group, async_op=async_op
                    )
                    assert work.wait_post_process(local_grad) is local_grad
                    expected_grad = (
                        torch.tensor(expected_reduce(case, rank, 7), dtype=dtype, device="cuda")
                        .view(-1, 1)
                        .expand(-1, width)
                    )
                    torch.testing.assert_close(local_grad, expected_grad, atol=0, rtol=0)
        empty_case = dict(
            inputs=[[] for _ in range(world)],
            outputs=[[] for _ in range(world)],
            destinations=[[] for _ in range(world)],
            sources=[[] for _ in range(world)],
            entries=[],
            capacity=0,
        )
        empty = module.ZeroCTACollectiveArg(
            **kwargs_for(module, empty_case, rank, world, group, slot=2, payload_bytes=64),
            nvl_domain_size=nvl,
        )
        input = torch.empty((0, 3), device="cuda", dtype=dtype)
        output = torch.empty_like(input)
        work = backend.zero_cta_group_cast_impl(input, output, empty, group, async_op=True)
        assert work.wait_post_process(output) is output
        work = backend.zero_cta_group_reduce_impl(input, output, empty, group, async_op=True)
        assert work.wait_post_process(output) is output
    stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--nvl-domain-size", type=int, required=True,
                        help="Ranks per NVL domain; supplied by the caller")
    args = parser.parse_args()
    if args.nvl_domain_size <= 0:
        parser.error("--nvl-domain-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this test requires CUDA; no CPU fallback is provided")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    nvl = args.nvl_domain_size
    dist.init_process_group("nccl", pg_options=nccl_options())
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    cases = [make_case(world, seed) for seed in (900, 903, 917)]
    if world > nvl > 1:
        cases.append(relay_case(world, nvl))
    capacity = max(case["capacity"] for case in cases)
    report = dict(
        rank=rank,
        success=False,
    )
    try:
        for dtype in (torch.float32, torch.bfloat16):
            run_collectives(
                meta, backend, group, cases, capacity, rank, world, nvl, dtype, torch.cuda.Stream()
            )
        report["success"] = True
        if rank == 0:
            print(json.dumps(report, indent=2))
    finally:
        if report["success"]:
            torch.cuda.synchronize()
            dist.barrier()
            backend.clear_zero_cta_cpp_state(group)
            dist.destroy_process_group()
        if args.report_dir is not None:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
