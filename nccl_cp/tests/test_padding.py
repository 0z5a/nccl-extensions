"""Receive capacity rows must stay outside the logical communication route."""

from types import SimpleNamespace

import pytest
import torch
from nccl.cp import CpConfig, CpGroup, collectives, flat
from nccl.cp.routing import Route


@pytest.mark.parametrize('reduce,input_rows,output_rows,accepted', [
    (False, 2, 2, True), (False, 2, 5, True),
    (False, 2, 1, False), (False, 3, 5, False),
    (True, 2, 2, True), (True, 5, 2, True),
    (True, 1, 2, False), (True, 5, 3, False),
])
def test_row_capacity_contract(monkeypatch, reduce, input_rows, output_rows, accepted):
    monkeypatch.setattr(collectives.dist, 'get_backend', lambda group: 'gloo')
    route = Route((2,), (2,), ((0,),), (0,))
    args = (torch.empty(input_rows, 3), torch.empty(output_rows, 3), route, None, reduce)
    if accepted:
        collectives._validate_tensors(*args)
    else:
        with pytest.raises(ValueError, match='shapes must match'):
            collectives._validate_tensors(*args)


def test_padding_does_not_relax_payload_shape(monkeypatch):
    monkeypatch.setattr(collectives.dist, 'get_backend', lambda group: 'gloo')
    route = Route((2,), (2,), ((0,),), (0,))
    with pytest.raises(ValueError, match='shapes must match'):
        collectives._validate_tensors(torch.empty(2, 3), torch.empty(5, 4), route, None, False)


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('rows', [0, 2])
def test_prefix_write_and_reverse_ignore_poisoned_tail(monkeypatch, explicit, rows):
    pg = object()
    state = collectives._GroupState(pg, torch.device('cpu'))
    group = CpGroup(pg, CpConfig(max_per_peer_slot=2, payload_shape=(3,),
                                dtype=torch.float32, backend='all2allv'), state)
    monkeypatch.setattr(collectives, '_group', lambda group: pg)
    monkeypatch.setattr(collectives.dist, 'get_rank', lambda group: 0)
    monkeypatch.setattr(collectives.dist, 'get_world_size', lambda group: 1)
    monkeypatch.setattr(collectives.dist, 'get_backend', lambda group: 'gloo')
    counts = []

    def exchange(recv, send, recv_counts, send_counts, **kwargs):
        counts.append((tuple(send_counts), tuple(recv_counts)))
        recv.copy_(send)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(flat.dist, 'all_to_all_single', exchange)
    route = Route((rows,), (rows,), ((0,),), (0,))
    handle = collectives.Handle(group, state, route, bool(rows), False, '')
    data = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    padded = torch.full((rows + 3, 3), -19.)
    pointer = padded.data_ptr()
    if explicit:
        result = collectives.group_cast_explicit(data, padded, [[rows]], [[rows]], [[[0]]], [[0]], group=group, stream=None)
    else:
        result = collectives.group_cast(handle, data, padded, stream=None)
    assert result is None and padded.data_ptr() == pointer
    torch.testing.assert_close(padded[:rows], data, rtol=0, atol=0)
    torch.testing.assert_close(padded[rows:], torch.full((3, 3), -19.), rtol=0, atol=0)
    grad_input = torch.full_like(padded, float('nan'))
    grad_input[:rows].fill_(2)
    grad_output = torch.full_like(data, 7)
    if explicit:
        result = collectives.group_reduce_explicit(grad_input, grad_output, [[rows]], [[rows]], [[0]], [[[0]]], group=group, stream=None)
    else:
        result = collectives.group_reduce(handle, grad_input, grad_output, stream=None)
    assert result is None
    torch.testing.assert_close(grad_output, torch.full_like(data, 9), rtol=0, atol=0)
    assert torch.isnan(grad_input[rows:]).all()
    assert counts == ([((rows,), (rows,))] * 2 if rows else [])
