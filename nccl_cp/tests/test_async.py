"""Async launch must defer the transfer join and output post-processing."""
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from nccl.cp import (
    CpConfig,
    CpGroup,
    WorkWithPostProcessFn,
    collectives,
    flat,
    group_cast_async,
    group_cast_explicit_async,
    group_reduce_async,
    group_reduce_explicit_async,
    zero_cta,
)
from nccl.cp.routing import Route, Segment


def forbidden(*args, **kwargs):
    raise AssertionError('Extra preparation, synchronization or native-path validation')


@pytest.mark.parametrize('reduce', [False, True])
@pytest.mark.parametrize('method', [False, True])
def test_native_launch_defers_native_work_wait(monkeypatch, reduce, method):
    handle = object.__new__(collectives.Handle)
    handle._backend = 'zero_cta'
    handle._group = SimpleNamespace(nccl_group=object())
    handle._native = SimpleNamespace(runtime=object(), cast_plan='cast', reduce_plan='reduce')
    events = []
    def run(input, output, runtime, plan):
        events.append(('launch', plan))
        def wait():
            events.append('wait')
            output.add_(input) if reduce else output.copy_(input)
        return SimpleNamespace(wait=wait)
    monkeypatch.setattr(zero_cta, '_get_extension', lambda: SimpleNamespace(run=run))
    for name in ('_execute', '_validate_tensors', '_resolve_stream', '_group', '_exchange'):
        monkeypatch.setattr(collectives, name, forbidden)
    monkeypatch.setattr(torch.cuda, 'stream', forbidden)
    monkeypatch.setattr(torch.cuda, 'Event', forbidden)
    input, output = torch.ones(2, 3), torch.full((2, 3), 7.)
    if method:
        fn = handle.group_reduce_async if reduce else handle.group_cast_async
        work = fn(input, output, stream=None)
    else:
        fn = group_reduce_async if reduce else group_cast_async
        work = fn(handle, input, output, stream=None)
    assert isinstance(work, WorkWithPostProcessFn) and work.async_op
    assert events == [('launch', 'reduce' if reduce else 'cast')]
    assert torch.equal(output, torch.full_like(output, 7))
    events.append('independent computation')
    assert work.wait_post_process() is output
    assert events[-2:] == ['independent computation', 'wait']
    assert torch.equal(output, torch.full_like(output, 8 if reduce else 1))
    with pytest.raises(RuntimeError, match='already been done'):
        work.wait_post_process()


@pytest.mark.parametrize('reduce', [False, True])
def test_flat_wait_owns_buffers_and_defers_output(monkeypatch, reduce):
    events, refs = [], []
    def exchange(received, packed, recv_counts, send_counts, **kwargs):
        assert recv_counts == send_counts == [2]
        events.append('launch')
        r, p = weakref.ref(received), weakref.ref(packed)
        refs.extend((r, p))
        def wait():
            assert r() is not None and p() is not None
            events.append('wait')
            r().copy_(p())
        return SimpleNamespace(wait=wait)
    monkeypatch.setattr(flat.dist, 'all_to_all_single', exchange)
    data = torch.ones(2, 3)
    source = torch.cat((data, torch.full((3, 3), float('nan')))) if reduce else data
    output = torch.full((2 if reduce else 5, 3), 7., requires_grad=True)
    segments = ((Segment(0, 2),),)
    work = flat.run_async(source, output, segments, segments, object(), reduce=reduce)
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert events == ['launch'] and torch.equal(output, torch.full_like(output, 7))
    events.append('compute')
    assert work.wait_post_process() is output
    assert events == ['launch', 'compute', 'wait']
    torch.testing.assert_close(output[:2], torch.full_like(output[:2], 8 if reduce else 1))
    if not reduce:
        torch.testing.assert_close(output[2:], torch.full_like(output[2:], 7))
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('reduce', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_public_flat_async_and_empty_work(monkeypatch, explicit, reduce, empty):
    pg = object()
    state = collectives._GroupState(pg, torch.device('cpu'))
    group = CpGroup(pg, CpConfig(max_per_peer_slot=2, payload_shape=(3,),
                                dtype=torch.float32, backend='all2allv'), state)
    monkeypatch.setattr(collectives, '_group', lambda value: pg)
    monkeypatch.setattr(collectives.dist, 'get_rank', lambda value: 0)
    monkeypatch.setattr(collectives.dist, 'get_world_size', lambda value: 1)
    monkeypatch.setattr(collectives.dist, 'get_backend', lambda value: 'gloo')
    events = []
    def exchange(received, packed, *args, **kwargs):
        events.append('launch')
        return SimpleNamespace(wait=lambda: (events.append('wait'), received.copy_(packed)))
    monkeypatch.setattr(flat.dist, 'all_to_all_single', exchange)
    count = 0 if empty else 2
    route = Route((count,), (count,), ((0,),), (0,))
    handle = collectives.Handle(group, state, route, not empty, False, '')
    input, output = torch.ones(count, 3), torch.full((count, 3), 7.)
    if explicit:
        fn = group_reduce_explicit_async if reduce else group_cast_explicit_async
        peers = ([[0]], [[[0]]]) if reduce else ([[[0]]], [[0]])
        work = fn(input, output, [[count]], [[count]], *peers, group=group, stream=None)
    else:
        fn = group_reduce_async if reduce else group_cast_async
        work = fn(handle, input, output, stream=None)
    assert isinstance(work, WorkWithPostProcessFn)
    assert events == ([] if empty else ['launch'])
    assert work.wait_post_process() is output
    assert events == ([] if empty else ['launch', 'wait'])
    torch.testing.assert_close(output, torch.full_like(output, 8 if reduce else 1))


def test_async_lse_rejected_before_native_launch_or_explicit_preparation(monkeypatch):
    handle = SimpleNamespace(_backend='zero_cta', _native=None,
                             _group=SimpleNamespace(nccl_group=None))
    monkeypatch.setattr(zero_cta, '_get_extension', forbidden)
    monkeypatch.setattr(collectives, '_group', forbidden)
    for kwargs in ({'cast_lse': True}, {'input_lse': torch.empty(0)}, {'output_lse': torch.empty(0)}):
        with pytest.raises(NotImplementedError):
            group_cast_async(handle, torch.ones(1), torch.zeros(1), stream=None, **kwargs)
        with pytest.raises(NotImplementedError):
            group_cast_explicit_async(torch.ones(1), torch.zeros(1), [], [], [], [], group=None, stream=None, **kwargs)
