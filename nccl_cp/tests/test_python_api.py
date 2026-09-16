from contextlib import contextmanager
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from nccl.cp import (
    CpConfig,
    collectives,
)
from nccl.cp.routing import Route


def fake_handle():
    handle = object.__new__(collectives.Handle)
    handle._group = object()
    handle._backend = "all2allv"
    handle._route = Route((2,), (2,), ((0,),), (0,))
    return handle


def test_handle_first_and_method_calls_share_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(collectives, "_execute", lambda *args: calls.append(args))
    handle = fake_handle()
    input, output = torch.ones(2, 3), torch.zeros(2, 3)
    assert collectives.group_cast(handle, input=input, output=output, stream=None) is None
    assert handle.group_cast(input=input, output=output, stream=None) is None
    assert calls[0][1] is input and calls[0][2] is output
    assert calls[0][3] is handle._route and calls[0][4] is handle and calls[0][5:] == (False, None)
    assert calls[1][1] is input and calls[1][2] is output
    assert collectives.group_reduce(handle, grad_input=input, grad_output=output, stream=None) is None
    assert calls[-1][5] is True
    assert handle.group_reduce(grad_input=input, grad_output=output, stream=None) is None
    assert calls[-1][1] is input and calls[-1][2] is output and calls[-1][5] is True


def test_stream_and_output_are_required():
    with pytest.raises(TypeError):
        collectives.group_cast(fake_handle(), torch.ones(2), stream=None)
    with pytest.raises(TypeError):
        collectives.group_reduce(fake_handle(), torch.ones(2), stream=None)
    with pytest.raises(TypeError):
        collectives.group_cast(fake_handle(), torch.ones(2), torch.zeros(2))
    with pytest.raises(TypeError):
        collectives.create_handle(None, [], [])
    with pytest.raises(TypeError, match="CpGroup"):
        collectives.create_handle(None, [], [], stream=None)
    with pytest.raises(TypeError):
        CpConfig(max_per_peer_slot=2, payload_shape=(128,))


def test_cpu_stream_is_explicit_none():
    state = SimpleNamespace(device=torch.device("cpu"))
    assert collectives._resolve_stream(state, None) is None
    with pytest.raises(ValueError, match="stream=None"):
        collectives._resolve_stream(state, 0)


def test_cuda_stream_selection_and_scope_restore(monkeypatch):
    class Stream:
        def __init__(self, device):
            self.device = torch.device(device)

    state = SimpleNamespace(device=torch.device("cuda", 0), lock=RLock())
    selected = Stream("cuda:0")
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device: selected)
    assert collectives._resolve_stream(state, selected) is selected
    assert collectives._resolve_stream(state, 0) is selected
    with pytest.raises(ValueError, match="ProcessGroup CUDA device"):
        collectives._resolve_stream(state, Stream("cuda:1"))
    with pytest.raises(TypeError):
        collectives._resolve_stream(state, None)
    with pytest.raises(TypeError):
        collectives._resolve_stream(state, -1)
    events = []

    @contextmanager
    def scope(stream):
        events.append(("enter", stream))
        try:
            yield
        finally:
            events.append(("restore", stream))

    monkeypatch.setattr(torch.cuda, "stream", scope)
    def fail_validation(*args):
        events.append(("body", selected))
        raise RuntimeError("fixture")

    monkeypatch.setattr(collectives, "_group", lambda group: group)
    monkeypatch.setattr(collectives, "_state", lambda group: state)
    monkeypatch.setattr(collectives, "_validate_tensors", fail_validation)
    with pytest.raises(RuntimeError):
        collectives._execute(object(), torch.ones(2), torch.zeros(2), fake_handle()._route, None, False, selected)
    assert [event[0] for event in events] == ["enter", "body", "restore"]
