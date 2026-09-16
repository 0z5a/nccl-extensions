"""Fixed metadata frames combine lengths, errors and contents in one collective."""

from types import SimpleNamespace

import pytest
import torch
from nccl.cp import CpConfig, collectives


@pytest.fixture
def transport(monkeypatch):
    calls = []
    group = object()
    monkeypatch.setattr(collectives, "_state", lambda group: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: 2)

    def gather(outputs, input, *, group):
        calls.append(input.clone())
        assert input.device.type == "cpu" and input.dtype == torch.int64
        assert len(outputs) == 2 and all(output.shape == input.shape for output in outputs)
        for output in outputs:
            output.copy_(input)

    def forbidden(*args, **kwargs):
        raise AssertionError("Object/size collective must not be used")

    monkeypatch.setattr(collectives.dist, "all_gather", gather)
    for name in ("all_gather_object", "all_reduce", "barrier", "broadcast_object_list"):
        monkeypatch.setattr(collectives.dist, name, forbidden)
    return group, calls


def record(owned, requested, limit=4):
    config = CpConfig(max_per_peer_slot=1, max_layout_tokens=limit,
                      payload_shape=(3,), dtype=torch.float32)
    return ((owned, requested), config, (1, 8, 12),
            dict(ok=True, reason="", host="node", uuid="gpu0", accessible=("gpu0", "gpu1")))


@pytest.mark.parametrize("owned,requested", [
    ((), ()),
    (((0, 4),), ((0, 4),)),
    (((0, 1), (7, 10)), ((0, 1), (4, 7), (7, 10))),
    (((-(1 << 63), -(1 << 63) + 1), ((1 << 63) - 2, (1 << 63) - 1)), ()),
])
def test_layout_round_trip_uses_one_fixed_collective(transport, owned, requested):
    group, calls = transport
    value = record(owned, requested)
    # No factory may inherit a non-CPU default device for host metadata.
    with torch.device("meta"):
        result = collectives._exchange(group, "create_handle/layouts", value, max_layout_tokens=4)
    assert result == [value, value]
    assert len(calls) == 1
    assert calls[0].numel() == collectives._HEADER_WORDS + collectives._CONTROL_BYTES // 8 + 2 * (4 + 2 * 4)


@pytest.mark.parametrize("owned,requested,match", [
    (((0, 5),), (), "max_layout_tokens"),
    ((), ((0, 9),), "max_layout_tokens"),
    ((((1 << 63) - 1, 1 << 63),), (), "int64"),
])
def test_invalid_layout_still_exchanges_one_same_sized_frame(transport, owned, requested, match):
    group, calls = transport
    with pytest.raises(ValueError, match=match):
        collectives._exchange(group, "create_handle/layouts", record(owned, requested), max_layout_tokens=4)
    assert len(calls) == 1
    assert calls[0][1:3].tolist() == [0, 0]
    assert calls[0].numel() == collectives._HEADER_WORDS + collectives._CONTROL_BYTES // 8 + 24


def test_control_envelope_overflow_is_reported_in_the_same_collective(transport):
    group, calls = transport
    with pytest.raises(ValueError, match="control metadata exceeds"):
        value = (*record((), ())[:3], dict(reason="x" * collectives._CONTROL_BYTES))
        collectives._exchange(group, "create_handle/layouts", value, max_layout_tokens=4)
    assert len(calls) == 1


def test_preparation_error_does_not_require_a_layout_or_second_exchange(transport):
    group, calls = transport
    value = (None, record((), ())[1], None, {})
    with pytest.raises(ValueError, match="rank 0: invalid layout; rank 1: invalid layout"):
        collectives._exchange(group, "create_handle/layouts", value, "invalid layout", max_layout_tokens=4)
    assert len(calls) == 1
