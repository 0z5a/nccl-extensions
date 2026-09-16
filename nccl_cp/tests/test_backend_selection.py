import pytest
import torch
from nccl.cp import collectives
from nccl.cp.routing import Route


def test_image_coverage_distinguishes_cubin_and_ptx():
    assert collectives._image_supports(("90",), 100)
    assert collectives._image_supports(("90-real",), 90)
    assert not collectives._image_supports(("90-real",), 100)
    assert not collectives._image_supports(("100",), 90)
    assert not collectives._image_supports(("native",), 90)


def test_flat_tensor_validation_accepts_general_payloads(monkeypatch):
    route = Route((4,), (4,), ((0,),), (0,))
    monkeypatch.setattr(collectives.dist, "get_backend", lambda group: "gloo")
    input = torch.empty((4, 6), dtype=torch.float64)
    collectives._validate_tensors(input.t().contiguous().t(), input, route, None, False)
    with pytest.raises(ValueError, match="shapes must match"):
        collectives._validate_tensors(input[:2], input, route, None, False)
    with pytest.raises(ValueError, match="overlapping"):
        collectives._validate_tensors(input, torch.empty(1, 6, dtype=torch.float64).expand(4, 6), route, None, False)


def test_topology_requires_same_host_and_accessible_distinct_peers():
    first = dict(host="node", uuid="gpu0", accessible=("gpu0", "gpu1"))
    second = dict(host="node", uuid="gpu1", accessible=("gpu0", "gpu1"))
    assert not collectives._topology_reason([first, second], 2)
    assert collectives._topology_reason([first, {**second, "host": "other"}], 2)
    assert collectives._topology_reason([first, {**second, "uuid": "gpu0"}], 2)
    assert collectives._topology_reason([{**first, "accessible": ("gpu0",)}, second], 2)
