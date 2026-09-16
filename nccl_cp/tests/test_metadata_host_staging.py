"""Host metadata stays on CPU regardless of the application's default device."""

import pytest
import torch
from nccl.cp import _utils, zero_cta


def test_device_tensor_stages_on_cpu_under_non_cpu_default(monkeypatch):
    # Pinning is unavailable in the CPU-only test environment. The test checks
    # the actual staging device/value behavior, not CUDA transfer completion.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    with torch.device("meta"):
        result = _utils._make_device_tensor([3, 7], dtype=torch.int64, device=torch.device("cpu"))
    assert result.device.type == "cpu" and result.dtype == torch.int64
    assert result.tolist() == [3, 7]


def test_native_metadata_stages_on_cpu_under_non_cpu_default(monkeypatch):
    original = torch.tensor

    def without_pinning(*args, **kwargs):
        assert kwargs.pop("pin_memory") is True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", without_pinning)
    with torch.device("meta"):
        result = zero_cta._make_device_metadata([(1, 2, 3)], lambda host: host)
    assert result.device.type == "cpu" and result.dtype == torch.int64
    assert result.tolist() == [[1, 2, 3]]


@pytest.mark.parametrize("helper", ["device_tensor", "native_metadata"])
def test_non_cpu_metadata_is_rejected_without_copy(monkeypatch, helper):
    values = torch.empty((1, 3), dtype=torch.int64, device="meta")

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject non-CPU metadata before attempting a copy")

    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "as_tensor", forbidden)
    with pytest.raises(ValueError, match="must reside on CPU"):
        if helper == "device_tensor":
            _utils._make_device_tensor(values, dtype=torch.int64)
        else:
            zero_cta._make_device_metadata(values, forbidden)
