"""Prepared execution must not rebuild routes or issue host/device control traffic."""

from types import SimpleNamespace

import pytest
import torch
from nccl.cp import (
    CpConfig,
    CpGroup,
    collectives,
    create_group,
    flat,
)
from nccl.cp.routing import Route, Segment


def forbidden(*args, **kwargs):
    raise AssertionError("Control exchange, preparation or host/device transfer in data path")


@pytest.fixture
def local_group(monkeypatch):
    group = object()
    state = collectives._GroupState(group, torch.device("cpu"))
    monkeypatch.setattr(collectives, "_group", lambda value: group)
    monkeypatch.setattr(collectives, "_state", lambda value: state)
    monkeypatch.setattr(collectives.dist, "get_rank", lambda value: 0)
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda value: 1)
    monkeypatch.setattr(collectives.dist, "get_backend", lambda value: "gloo")
    return group, state


@pytest.mark.parametrize("backend", ["zero_cta", "all2allv"])
def test_data_entry_points_never_prepare_or_exchange(monkeypatch, local_group, backend):
    from nccl.cp import comm_meta, zero_cta

    group, state = local_group
    route = Route((2,), (2,), ((0,),), (0,))
    cp_group = CpGroup(group, CpConfig(max_per_peer_slot=2, payload_shape=(3,), dtype=torch.float32), state)
    handle = collectives.Handle(cp_group, state, route, True, backend == "zero_cta", "")
    native = SimpleNamespace(runtime=SimpleNamespace(max_per_token_bytes=lambda: 12))
    handle._native = native if backend == "zero_cta" else None
    operations, waits = [], []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(wait_event=waits.append))

    def cast(input, output, arg, group, **kwargs):
        assert arg is native and kwargs == {
            "async_op": True, "cast_lse": False, "input_lse": None, "output_lse": None,
        }
        operations.append("native cast")
        return SimpleNamespace(wait_post_process=lambda target: target.copy_(input))

    def reduce(input, output, arg, group, **kwargs):
        assert arg is native and kwargs == {
            "async_op": True, "reduce_op": "sum", "acc_reduce": True,
            "comm_dtype": None, "input_lse": None, "output_lse": None,
        }
        operations.append("native reduce")
        return SimpleNamespace(wait_post_process=lambda target: target.add_(input))

    def all2all(received, packed, recv_counts, send_counts, **kwargs):
        operations.append("all2allv")
        received.copy_(packed)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(zero_cta, "zero_cta_group_cast_impl", cast)
    monkeypatch.setattr(zero_cta, "zero_cta_group_reduce_impl", reduce)
    monkeypatch.setattr(comm_meta, "ZeroCTACollectiveArg", forbidden)
    monkeypatch.setattr(flat.dist, "all_to_all_single", all2all)
    for name in ("_exchange", "_probe_zero", "create_handle"):
        monkeypatch.setattr(collectives, name, forbidden)
    for name in ("all_gather_object", "all_gather", "all_reduce", "barrier", "broadcast_object_list"):
        monkeypatch.setattr(collectives.dist, name, forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    for name in ("cpu", "cuda", "to", "item", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)

    input, output = torch.ones(2, 3), torch.zeros(2, 3)
    # Handle calls use the already-prepared Route, without rebuilding its lists.
    with monkeypatch.context() as guard:
        guard.setattr(Route, "from_lists", forbidden)
        for _ in range(2):
            assert collectives.group_cast(handle, input, output, stream=None) is None
            assert collectives.group_reduce(handle, input, output, stream=None) is None
    assert len(operations) == 4 and handle.backend == backend
    assert torch.equal(output, input * 2)
    if backend == "zero_cta":
        assert handle._native is native and not waits

    before = len(operations)
    if backend == "all2allv":
        for operation in (collectives.group_cast, collectives.group_reduce):
            for source, target in ((None, output), (input, None)):
                with pytest.raises(TypeError):
                    operation(handle, source, target, stream=None)
    # Native argument rejection belongs to the native wrapper/TorchBind path,
    # covered by test_wrappers and the GPU runner, not these mocked operations.
    assert len(operations) == before


def test_handle_directly_builds_native_argument_and_reuses_native_runtime(monkeypatch, local_group):
    from nccl.cp import comm_meta

    group, state = local_group
    phases, builds, waits = [], [], []
    runtime = SimpleNamespace(max_per_peer_slot=lambda: 2, nvl_domain_size=lambda: 1,
                              max_per_token_bytes=lambda: 12)
    cuda_stream = SimpleNamespace(wait_event=waits.append)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: cuda_stream)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1024, 1024))
    monkeypatch.setattr(collectives, "_probe_zero", lambda group: dict(ok=True, host="node", reason=""))

    def exchange(group, phase, value=None, error="", **kwargs):
        phases.append(phase)
        if error:
            raise ValueError(error)
        return [value]

    def build(**kwargs):
        builds.append(kwargs)
        return SimpleNamespace(runtime=runtime, cast_plan=object(), reduce_plan=object())

    monkeypatch.setattr(collectives, "_exchange", exchange)
    monkeypatch.setattr(comm_meta, "ZeroCTACollectiveArg", build)
    config = CpConfig(max_per_peer_slot=2, payload_shape=(3,), dtype=torch.float32, nvl_domain_size=1)
    cp_group = create_group(group, config)
    handle = collectives.create_handle(cp_group, [0, 1], [0, 1], stream=None)
    assert phases == ["create_handle/layouts"]
    assert builds[0]["group"] is group and builds[0]["max_per_peer_slot"] == 2
    assert builds[0]["max_per_token_bytes"] == 12
    assert handle._native.cast_plan is not None and handle._native.reduce_plan is not None
    assert state.runtimes == {0: runtime}

    # Original native runtime permits reuse when its byte capacity is sufficient.
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    smaller_group = create_group(group, CpConfig(max_per_peer_slot=2, payload_shape=(3,), dtype=torch.bfloat16, nvl_domain_size=1))
    replacement = collectives.create_handle(smaller_group, [0, 1], [0, 1], stream=None)
    assert replacement._native.runtime is runtime and builds[1]["max_per_token_bytes"] == 6
    assert not waits
    handle.close()
    assert state.runtimes[0] is runtime
    phases.clear()
    with pytest.raises(ValueError, match="owner"):
        collectives.create_handle(cp_group, [0, 1], [10], stream=None)
    assert phases == ["create_handle/layouts"]
    assert len(builds) == 2


def test_flat_cuda_uses_stream_dependency_instead_of_blocking_wait(monkeypatch):
    calls = []

    class Buffer:
        shape = (1, 1)
        device = torch.device("cuda", 0)
        dtype = torch.float32

        def narrow(self, *args):
            return self

        def copy_(self, other):
            assert isinstance(other, Buffer)

    monkeypatch.setattr(flat.torch, "empty", lambda *args, **kwargs: Buffer())
    monkeypatch.setattr(flat.dist, "all_to_all_single", lambda *args, **kwargs: SimpleNamespace(
        wait=forbidden, block_current_stream=lambda: calls.append("stream dependency")))
    flat.run(Buffer(), Buffer(), ((Segment(0, 1),),), ((Segment(0, 1),),), object(), reduce=False)
    assert calls == ["stream dependency"]


@pytest.mark.parametrize("native", [True, False])
def test_cuda_shaped_handle_calls_add_no_python_events_or_locks(monkeypatch, native):
    from contextlib import nullcontext

    from nccl.cp import zero_cta

    class ForbiddenLock:
        def __enter__(self):
            forbidden()
        def __exit__(self, *args):
            pass

    class Stream:
        device = torch.device("cuda", 0)
        wait_event = forbidden

    class Tensor:
        shape = (2, 3)
        ndim = 2
        layout = torch.strided
        dtype = torch.float32
        device = torch.device("cuda", 0)
        cpu = cuda = to = item = tolist = numpy = record_stream = forbidden
        def stride(self, dim=None):
            return (3, 1) if dim is None else (3, 1)[dim]
        def numel(self):
            return 6
        def element_size(self):
            return 4

    class NoRescan(tuple):
        def __iter__(self):
            forbidden()

    group = object()
    state = collectives._GroupState(group, torch.device("cuda", 0), lock=ForbiddenLock())
    config = CpConfig(max_per_peer_slot=2, payload_shape=(3,), dtype=torch.float32)
    cp_group = CpGroup(group, config, state)
    monkeypatch.setattr(collectives.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(collectives.dist, "get_backend", lambda group: "nccl")
    route = Route((2,), (2,), ((0,),), (0,))
    handle = collectives.Handle(cp_group, state, route, True, native, "")
    handle._native = SimpleNamespace(runtime=SimpleNamespace(max_per_token_bytes=lambda: 12)) if native else None
    # Once prepared, validation must use cached row counts instead of rescanning.
    object.__setattr__(route, "input_splits", NoRescan((2,)))
    object.__setattr__(route, "output_splits", NoRescan((2,)))
    monkeypatch.setattr(torch, "Tensor", Tensor)
    monkeypatch.setattr(torch, "_debug_has_internal_overlap", lambda tensor: 0)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(collectives, "_state", forbidden)
    monkeypatch.setattr(collectives, "_states_lock", ForbiddenLock())
    monkeypatch.setattr(collectives, "_exchange", forbidden)
    calls = []

    def submit(*args, **kwargs):
        calls.append("submit")
        return SimpleNamespace(wait_post_process=lambda target: calls.append("postprocess"))

    monkeypatch.setattr(zero_cta, "zero_cta_group_cast_impl", submit)
    monkeypatch.setattr(zero_cta, "zero_cta_group_reduce_impl", submit)
    monkeypatch.setattr(flat, "run", lambda *args, **kwargs: calls.append("flat"))
    if native:
        # Prepared native calls may only reach the native wrapper and wait.
        # No extra Python validation, route/group lookup, stream or grad scope.
        handle._native.runtime.max_per_token_bytes = forbidden
        for name in ("_execute", "_validate_tensors", "_resolve_stream", "_group"):
            monkeypatch.setattr(collectives, name, forbidden)
        monkeypatch.setattr(collectives.dist, "get_backend", forbidden)
        monkeypatch.setattr(torch, "no_grad", forbidden)
        monkeypatch.setattr(torch.cuda, "stream", forbidden)
        monkeypatch.setattr(torch.cuda, "current_device", forbidden)
        monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    for _ in range(3):
        collectives.group_cast(handle, Tensor(), Tensor(), stream=Stream())
        collectives.group_reduce(handle, Tensor(), Tensor(), stream=Stream())
    assert calls == (["submit", "postprocess"] * 6 if native else ["flat"] * 6)
