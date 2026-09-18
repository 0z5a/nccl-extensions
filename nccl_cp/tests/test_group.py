"""CP group/config binding and host-only payload sizing."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch
from nccl.cp import CpConfig, CpGroup, RowRange, collectives, create_group


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected tensor operation, native setup or configuration mutation")


@pytest.fixture
def local_runtime(monkeypatch):
    nccl_group = object()
    state = collectives._GroupState(nccl_group, torch.device("cpu"))
    monkeypatch.setattr(collectives, "_group", lambda value: nccl_group)
    monkeypatch.setattr(collectives, "_state", lambda value: state)
    monkeypatch.setattr(collectives.dist, "get_rank", lambda value: 0)
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda value: 1)
    monkeypatch.setattr(collectives, "_exchange", lambda group, phase, value=None, error="", **kwargs: [value])
    return nccl_group, state


@pytest.mark.parametrize("shape,dtype,expected", [
    ((32, 128), torch.bfloat16, 8192),
    ([32, 128], torch.float32, 16384),
    (torch.Size([4096]), torch.bfloat16, 8192),
    ((), torch.float32, 4),
    ((0, 128), torch.bfloat16, 0),
])
def test_payload_budget_uses_only_host_metadata(monkeypatch, shape, dtype, expected):
    for name in ("tensor", "empty", "zeros"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("cpu", "cuda", "to", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    config = CpConfig(max_per_peer_slot=0 if expected == 0 else 4, payload_shape=shape, dtype=dtype)
    assert config.max_per_token_bytes == expected


@pytest.mark.parametrize("shape,dtype", [
    ((-1, 128), torch.bfloat16), ((True,), torch.bfloat16), ((2.0,), torch.float32),
    ((128,), torch.int64), ((128,), "bfloat16"), (torch.tensor([128]), torch.bfloat16),
])
def test_config_rejects_invalid_payload(shape, dtype):
    with pytest.raises((TypeError, ValueError), match="payload_shape|dtype"):
        CpConfig(max_per_peer_slot=4, payload_shape=shape, dtype=dtype)


def test_config_is_an_immutable_snapshot():
    shape = [32, 128]
    config = CpConfig(max_per_peer_slot=4, payload_shape=shape, dtype=torch.bfloat16)
    shape[0] = 100
    assert config.payload_shape == (32, 128) and config.max_per_token_bytes == 8192
    with pytest.raises(FrozenInstanceError):
        config.dtype = torch.float32


def test_create_group_stores_config_without_runtime_allocation(monkeypatch, local_runtime):
    from nccl.cp import zero_cta

    nccl_group, state = local_runtime
    monkeypatch.setenv("NVL_DOMAIN_SIZE", "2")
    monkeypatch.setattr(collectives, "_exchange", forbidden)
    for name in ("all_gather", "all_gather_object", "all_reduce", "barrier"):
        monkeypatch.setattr(collectives.dist, name, forbidden)
    monkeypatch.setattr(collectives, "_probe_zero", forbidden)
    monkeypatch.setattr(zero_cta, "get_or_create_zero_cta_runtime", forbidden)
    config = CpConfig(max_per_peer_slot=4096, payload_shape=(32, 128), dtype=torch.bfloat16)
    group = create_group(nccl_group, config)
    assert group.nccl_group is nccl_group and group.cp_config is config and group._state is state
    assert config.nvl_domain_size == 2 and config.max_per_token_bytes == 8192
    assert not state.runtimes


def test_create_group_rejects_invalid_configuration_locally(monkeypatch, local_runtime):
    nccl_group, state = local_runtime
    monkeypatch.setattr(collectives, "_exchange", forbidden)
    with pytest.raises(TypeError, match="cp_config must be a CpConfig"):
        create_group(nccl_group, None)
    assert not state.runtimes


@pytest.mark.parametrize("native", [False, True])
def test_close_runtime_uses_native_release_without_python_tracking(monkeypatch, local_runtime, native):
    from nccl.cp import zero_cta

    nccl_group, state = local_runtime
    calls = []
    if native:
        state.runtimes[0] = object()
    collectives._states[nccl_group] = state
    monkeypatch.setattr(collectives, "_exchange", forbidden)
    for name in ("all_gather", "all_gather_object", "all_reduce", "barrier"):
        monkeypatch.setattr(collectives.dist, name, forbidden)

    def release(group):
        assert group is nccl_group and not calls
        assert state.runtimes
        calls.append("native release")

    monkeypatch.setattr(zero_cta, "clear_zero_cta_cpp_state", release)
    collectives.close_runtime(nccl_group)
    assert calls == (["native release"] if native else [])
    assert state.closed and not state.runtimes
    assert nccl_group not in collectives._states


def test_capacity_check_reads_native_runtime_without_reconfiguring_it(monkeypatch, local_runtime):
    nccl_group, state = local_runtime
    runtime = SimpleNamespace(max_per_peer_slot=lambda: 4, nvl_domain_size=lambda: 1,
                              max_per_token_bytes=lambda: 256)
    state.runtimes[0] = runtime
    monkeypatch.setattr(collectives, "_probe_zero", lambda group: dict(ok=True, host="node", reason=""))
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    config = CpConfig(max_per_peer_slot=8, payload_shape=(128,), dtype=torch.bfloat16, nvl_domain_size=1)
    group = create_group(nccl_group, config)
    handle = collectives.create_handle(group, [0], [0], stream=None)
    assert handle.backend == "all2allv" and "capacity" in handle.fallback_reason
    assert state.runtimes[0] is runtime and runtime.max_per_peer_slot() == 4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_handle_reads_group_payload_and_selects_flat(monkeypatch, local_runtime, dtype):
    nccl_group, _ = local_runtime
    monkeypatch.setattr(collectives, "_probe_zero", forbidden)
    config = CpConfig(max_per_peer_slot=2, payload_shape=(128,), dtype=dtype)
    group = create_group(nccl_group, config)
    handle = collectives.create_handle(
        group,
        local_owned_layout=[RowRange(0, 2)],
        local_required_layout=[RowRange(0, 2)],
        stream=None,
    )
    assert handle.group is group and handle.nccl_group is nccl_group
    assert handle.backend == "all2allv" and "dtype" in handle.fallback_reason


def test_closed_runtime_invalidates_cp_group(monkeypatch):
    state = collectives._GroupState(object(), torch.device("cpu"), closed=True)
    group = CpGroup(state.group, CpConfig(max_per_peer_slot=1, payload_shape=(4,), dtype=torch.float32), state)
    monkeypatch.setattr(collectives.dist, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="closed runtime"):
        collectives._group(group)


def test_layout_bound_is_separate_from_native_workspace_capacity():
    config = CpConfig(max_per_peer_slot=4, payload_shape=(3,), dtype=torch.float32)
    assert config.max_layout_tokens == 4
    smaller_native = replace(config, max_per_peer_slot=1)
    assert smaller_native.max_layout_tokens == 4 and smaller_native.max_per_peer_slot == 1
    larger_layout = CpConfig(max_per_peer_slot=1, max_layout_tokens=100,
                             payload_shape=(3,), dtype=torch.float32)
    assert larger_layout.max_layout_tokens == 100 and larger_layout.max_per_peer_slot == 1


@pytest.mark.parametrize("bound", [-1, True, 1.5])
def test_layout_token_bound_requires_a_nonnegative_integer(bound):
    with pytest.raises(ValueError, match="max_layout_tokens"):
        CpConfig(max_per_peer_slot=1, max_layout_tokens=bound, payload_shape=(3,), dtype=torch.float32)
