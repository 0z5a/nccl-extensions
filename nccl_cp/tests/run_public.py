"""Exercise public CP collectives with real Gloo or NCCL communication."""

import argparse
import json
import os
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.distributed as dist
from nccl.cp import (
    CpConfig,
    RowRange,
    group_cast,
    group_cast_explicit,
    group_reduce,
    group_reduce_explicit,
)
from nccl.cp import (
    close_runtime as _close_runtime,
)
from nccl.cp import (
    create_group as _create_group,
)
from nccl.cp import (
    create_handle as _create_handle,
)


def checked_control_count(function, expected, *args, **kwargs):
    # Group binding/release use zero control collectives; each handle uses one.
    gather = dist.all_gather
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return gather(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Object metadata collective must not be used")

    with patch.object(dist, "all_gather", counted), patch.object(dist, "all_gather_object", forbidden):
        try:
            return function(*args, **kwargs)
        finally:
            assert len(calls) == expected, f"{function.__name__}: expected {expected} tensor AllGathers"


def create_group(*args, **kwargs):
    return checked_control_count(_create_group, 0, *args, **kwargs)


def create_handle(*args, **kwargs):
    return checked_control_count(_create_handle, 1, *args, **kwargs)


_cast_explicit = group_cast_explicit
_reduce_explicit = group_reduce_explicit


def group_cast_explicit(*args, **kwargs):
    return checked_control_count(_cast_explicit, 0, *args, **kwargs)


def group_reduce_explicit(*args, **kwargs):
    return checked_control_count(_reduce_explicit, 0, *args, **kwargs)


def close_runtime(*args, **kwargs):
    return checked_control_count(_close_runtime, 0, *args, **kwargs)


def layouts(world, nvl):
    relay = nvl if world > nvl > 1 else None
    owned = [[100 * rank + i for i in range(3)] for rank in range(world)]
    if relay is not None:
        owned[relay] = []
    requested = [[tokens[i] for i in range(3) for tokens in owned if tokens] for _ in range(world)]
    if relay is not None:
        requested[relay] = []
    owned[0].append(99)  # Never requested; reduce must preserve its initial output.
    return owned, requested


def payload(tokens, device, dtype=torch.float32):
    return torch.tensor(tokens, device=device, dtype=dtype).reshape(-1, 1) * 10 + torch.arange(6, device=device, dtype=dtype).reshape(1, 6)


def expect_error(function, errors=(TypeError, ValueError)):
    try:
        function()
    except errors:
        return
    raise AssertionError("Expected a parameter error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--force-flat", action="store_true")
    selection.add_argument("--require-zero", action="store_true")
    parser.add_argument("--nvl-domain-size", type=int, required=True,
                        help="Ranks per NVL domain; supplied by the caller")
    args = parser.parse_args()
    if args.nvl_domain_size <= 0:
        parser.error("--nvl-domain-size must be positive")
    if args.force_flat:
        os.environ["NCCL_CP_DISABLE_ZERO_CTA"] = "1"
    options = None
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        options = dist.ProcessGroupNCCL.Options()
        options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dist.init_process_group(args.backend, pg_options=options)
    nccl_group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        dist.destroy_process_group()
        raise ValueError("Cross-rank validation cases require at least two ranks")
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None

    def close_handle(target):
        if stream is not None:
            stream.synchronize()  # Caller keeps plans alive through GPU completion.
        target.close()

    owned, requested = layouts(world, args.nvl_domain_size)
    # Test caller already knows the global layout and supplies every rank's lists.
    from nccl.cp.routing import layout_intervals, local_route
    all_layouts = [(layout_intervals(a), layout_intervals(b)) for a, b in zip(owned, requested)]
    supplied = [local_route(r, all_layouts) for r in range(world)]
    input_splits = [list(route.input_splits) for route in supplied]
    output_splits = [list(route.output_splits) for route in supplied]
    destinations = [[list(peers) for peers in route.destinations] for route in supplied]
    sources = [list(route.sources) for route in supplied]
    cp_config = CpConfig(max_per_peer_slot=4, payload_shape=(6,), dtype=torch.float32,
                         nvl_domain_size=args.nvl_domain_size)
    group = create_group(nccl_group, cp_config)
    assert group.nccl_group is nccl_group and group.cp_config is cp_config
    assert cp_config.max_per_token_bytes == 24
    handle = create_handle(group, owned[rank], requested[rank], stream=stream)
    if handle.backend == "zero_cta":
        assert handle._native is not None
    flat_group = create_group(nccl_group, replace(cp_config, backend="all2allv"))
    flat_handle = create_handle(flat_group, owned[rank], requested[rank], stream=stream)
    assert handle.group is group and handle.nccl_group is nccl_group
    input = payload(owned[rank], device)
    output = torch.empty((len(requested[rank]), 6), device=device)
    output_pointer = output.data_ptr()
    expected = payload(requested[rank], device)
    selected = set()
    try:
        for _ in range(2):
            assert handle.group_cast(input, output, stream=stream) is None
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert output.data_ptr() == output_pointer
            selected.add(handle.backend)
        if args.require_zero:
            assert handle.backend == "zero_cta", handle.fallback_reason
        assert group_cast(handle, input, output, stream=stream) is None
        grad_input = torch.full_like(output, rank + 1)
        grad_output = torch.full_like(input, 7)
        contribution = torch.tensor([
            sum(peer + 1 for peer in range(world) if token in requested[peer]) for token in owned[rank]
        ], device=device, dtype=input.dtype).reshape(-1, 1).expand_as(input)
        grad_pointer = grad_output.data_ptr()
        for count in (1, 2):
            assert handle.group_reduce(grad_input, grad_output, stream=stream) is None
            torch.testing.assert_close(grad_output, 7 + count * contribution, rtol=0, atol=0)
            assert grad_output.data_ptr() == grad_pointer
        grad_output.zero_()
        assert group_reduce(handle, grad_input, grad_output, stream=stream) is None
        grad_output.zero_()
        handle.group_reduce(grad_input, grad_output, stream=stream)
        torch.testing.assert_close(grad_output, contribution, rtol=0, atol=0)

        # First explicit preparation uses its own native slot without a handle.
        explicit_group = create_group(nccl_group, replace(cp_config, runtime_slot=3))
        from contextlib import nullcontext
        explicit_stream = torch.cuda.Stream() if device.type == "cuda" else None
        for _ in range(3):
            grad_output.fill_(7)
            if explicit_stream is not None:
                explicit_stream.wait_stream(stream)
            group_cast_explicit(input, output, input_splits, output_splits, destinations, sources, group=explicit_group, stream=explicit_stream)
            group_reduce_explicit(grad_input, grad_output, output_splits, input_splits, sources, destinations, group=explicit_group, stream=explicit_stream)
            # Allocation churn follows temporary-plan release on its stream.
            with torch.cuda.stream(explicit_stream) if explicit_stream is not None else nullcontext():
                scratch = [torch.empty((n,), device=device, dtype=torch.int64) for n in (1, 3, 8, 32, 128, 512, 2048, 4096)]
                for buffer in scratch:
                    buffer.fill_(-1)
            if explicit_stream is not None:
                stream.wait_stream(explicit_stream)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(grad_output, 7 + contribution, rtol=0, atol=0)
        if args.require_zero:
            assert 3 in explicit_group._state.runtimes
        # Explicit lists use the same route and execution contract.
        assert group_cast_explicit(input, output, input_splits, output_splits, destinations, sources, group=group, stream=stream) is None
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        grad_output.zero_()
        assert group_reduce_explicit(grad_input, grad_output, output_splits, input_splits, sources, destinations, group=group, stream=stream) is None
        torch.testing.assert_close(grad_output, contribution, rtol=0, atol=0)

        # General payloads use a backend selected before data submission.
        double_input, double_output = input.double(), output.double()
        if handle.backend == "zero_cta":
            expect_error(lambda: handle.group_cast(double_input, double_output, stream=stream), (RuntimeError,))
            assert handle.backend == "zero_cta"
        flat_handle.group_cast(double_input, double_output, stream=stream)
        assert flat_handle.backend == "all2allv"
        torch.testing.assert_close(double_output, expected.double(), rtol=0, atol=0)
        if device.type == "cuda":
            other_stream = torch.cuda.Stream()
            other_stream.wait_stream(stream)
            # The slot keeps its original CP stream. The caller establishes
            # producer order and explicitly waits before another stream consumes.
            stream.wait_stream(other_stream)
            with torch.cuda.stream(other_stream):
                # Caller selects the CP stream outside the submission loop.
                with torch.cuda.stream(stream):
                    handle.group_cast(input, output, stream=stream)
                assert torch.cuda.current_stream() == other_stream
            other_stream.wait_stream(stream)
        else:
            handle.group_cast(input, output, stream=stream)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        bf_input = payload(owned[rank], device, torch.bfloat16)
        bf_output = torch.empty(output.shape, device=device, dtype=torch.bfloat16)
        handle.group_cast(bf_input, bf_output, stream=stream)
        torch.testing.assert_close(bf_output, payload(requested[rank], device, torch.bfloat16), rtol=0, atol=0)
        bf_grad = torch.full_like(bf_input, 3)
        handle.group_reduce(torch.full_like(bf_output, rank + 1), bf_grad, stream=stream)
        torch.testing.assert_close(bf_grad, (3 + contribution).to(torch.bfloat16), rtol=0, atol=0)

        strided = input.t().contiguous().t() if rank == 0 else input
        if handle.backend == "zero_cta" and rank == 0:
            expect_error(lambda: handle.group_cast(strided, output, stream=stream), (RuntimeError,))
        flat_handle.group_cast(strided, output, stream=stream)
        assert flat_handle.backend == "all2allv"
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        # Invalid arguments fail locally; other ranks do not enter a payload call.
        if rank == 0:
            expect_error(lambda: handle.group_cast(input, None, stream=stream))
            if device.type == "cpu":
                expect_error(lambda: handle.group_cast(input, output, stream=0))
            bad_splits = [list(parts) for parts in input_splits]
            bad_splits[0][0] += 1
            expect_error(lambda: group_cast_explicit(input, output, bad_splits, output_splits, destinations, sources, group=group, stream=stream))
        # Preparation still synchronizes route errors across the group.
        bad_request = list(reversed(requested[rank])) if rank == 0 else requested[rank]
        expect_error(lambda: create_handle(group, owned[rank], bad_request, stream=stream))
        handle.group_cast(input, output, stream=stream)

        # Equal-sized handle frames can still report differing config values.
        # The caller must keep max_layout_tokens equal before this exchange.
        for invalid_config in (
            replace(cp_config, payload_shape=(2, 3) if rank == 0 else (6,)),
            replace(cp_config, runtime_slot=2, dtype=torch.bfloat16 if rank == 0 else torch.float32),
        ):
            invalid_group = create_group(nccl_group, invalid_config)
            expect_error(lambda: create_handle(invalid_group, owned[rank], requested[rank], stream=stream))

        duplicated_owner = [*owned[rank], owned[0][0]] if rank == 1 else owned[rank]
        expect_error(lambda: create_handle(group, duplicated_owner, requested[rank], stream=stream))

        # Metadata overflow is reported within the single handle exchange.
        oversized = [RowRange(0, 5)] if rank == 0 else owned[rank]
        expect_error(lambda: create_handle(group, oversized, requested[rank], stream=stream))
        oversized_request = [RowRange(0, 4 * world + 1)] if rank == 0 else requested[rank]
        expect_error(lambda: create_handle(group, owned[rank], oversized_request, stream=stream))
        out_of_int64 = [RowRange((1 << 63) - 1, 1 << 63)] if rank == 0 else owned[rank]
        expect_error(lambda: create_handle(group, out_of_int64, requested[rank], stream=stream))
        oversized_group = create_group(nccl_group, replace(
            cp_config, payload_shape=(1,) * 10000 if rank == 0 else (6,)))
        expect_error(lambda: create_handle(oversized_group, owned[rank], requested[rank], stream=stream))
        handle.group_cast(input, output, stream=stream)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        # New layout and explicit capacity limits use a separate slot.
        smaller_group = create_group(nccl_group, replace(
            cp_config, max_per_peer_slot=1, max_layout_tokens=4, runtime_slot=1))
        smaller = create_handle(smaller_group, owned[rank], requested[rank], stream=stream)
        smaller.group_cast(input, output, stream=stream)
        assert smaller.backend == "all2allv"
        close_handle(smaller)

        empty = create_handle(group, [], [], stream=stream)
        empty_input = torch.empty((0, 6), device=device)
        assert empty.group_cast(empty_input, torch.empty_like(empty_input), stream=stream) is None
        assert empty.group_reduce(empty_input, torch.empty_like(empty_input), stream=stream) is None
        close_handle(empty)
        unused = create_handle(group, owned[rank], [], stream=stream)
        untouched = torch.full_like(input, 3)
        unused.group_reduce(empty_input, untouched, stream=stream)
        torch.testing.assert_close(untouched, torch.full_like(input, 3), rtol=0, atol=0)
        close_handle(unused)

        # Equivalent compact owned layout; required lists retain rank-local ordering.
        compact = [RowRange(tokens[0], tokens[0] + 3)] if (tokens := owned[rank]) else []
        if rank == 0:
            compact.append(99)
        replacement = create_handle(
            group, local_owned_layout=compact,
            local_required_layout=requested[rank], stream=stream,
        )
        original_runtime_id = handle._native.runtime.runtime_id() if handle._native is not None else None
        replacement.group_cast(input, output, stream=stream)
        if replacement.backend == "zero_cta" and original_runtime_id is not None:
            assert replacement._native.runtime.runtime_id() == original_runtime_id
        close_handle(replacement)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        if args.backend == "gloo":
            from nccl.cp import collectives as implementation
            original = implementation._probe_zero
            implementation._probe_zero = lambda _group: dict(
                ok=rank != 0, reason="fixture extension unavailable" if rank == 0 else "",
                host="fixture", uuid=str(rank), accessible=tuple(str(p) for p in range(world)))
            with patch.object(torch.cuda, "mem_get_info", return_value=(1024 ** 3, 1024 ** 3)):
                coordinated = create_handle(group, owned[rank], requested[rank], stream=stream)
            implementation._probe_zero = original
            coordinated.group_cast(input, output, stream=stream)
            assert coordinated.backend == "all2allv"
            assert "fixture extension unavailable" in coordinated.fallback_reason
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            close_handle(coordinated)

        # Guard only the data-call interval; preparation and result checks are outside it.
        from contextlib import ExitStack

        from nccl.cp import collectives as implementation

        def forbidden(*args, **kwargs):
            raise AssertionError("Control traffic or host/device transfer during payload submission")

        grad_output.zero_()
        with ExitStack() as guards:
            for name in ("_exchange", "create_handle", "_probe_zero"):
                guards.enter_context(patch.object(implementation, name, forbidden))
            for name in ("all_gather_object", "all_gather", "all_reduce", "barrier"):
                guards.enter_context(patch.object(dist, name, forbidden))
            for name in ("cpu", "cuda", "to", "item", "tolist", "numpy"):
                guards.enter_context(patch.object(torch.Tensor, name, forbidden))
            guards.enter_context(patch.object(torch.cuda, "mem_get_info", forbidden))
            guards.enter_context(patch.object(torch.cuda, "synchronize", forbidden))
            handle.group_cast(input, output, stream=stream)
            handle.group_reduce(grad_input, grad_output, stream=stream)
            flat_handle.group_cast(input, output, stream=stream)
        group_cast_explicit(input, output, input_splits, output_splits, destinations, sources, group=group, stream=stream)
        group_reduce_explicit(grad_input, grad_output, output_splits, input_splits, sources, destinations, group=group, stream=stream)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(grad_output, 2 * contribution, rtol=0, atol=0)
        if rank == 0:
            close_handle(handle)
            if handle.backend == "all2allv":
                expect_error(lambda: handle.group_cast(input, output, stream=stream))
            # Calling a closed native handle violates the caller-owned lifetime
            # contract; no extra public per-call liveness check is installed.
        close_handle(handle)
        close_handle(flat_handle)
        if rank == 0:
            print(json.dumps({"success": True, "transport": args.backend,
                              "backends_observed": sorted(selected), "public_contract": "passed"}))
    finally:
        if device.type == "cuda":
            stream.synchronize()  # Caller-owned completion before releasing data/plans.
        close_runtime(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
