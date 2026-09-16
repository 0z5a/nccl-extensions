"""Unsupported communication options must fail before preparation or submission."""
import inspect
from types import SimpleNamespace

import pytest
import torch
from nccl.cp import CpConfig, CpGroup, collectives, flat, zero_cta
from nccl.cp.routing import Route


def forbidden(*args, **kwargs):
    raise AssertionError('Unsupported LSE must not prepare, load or submit communication')


def handle_for(backend):
    handle = object.__new__(collectives.Handle)
    handle._backend = backend
    handle._group = SimpleNamespace(nccl_group=object())
    handle._native = SimpleNamespace(runtime=object(), cast_plan=object(), reduce_plan=object())
    return handle


@pytest.mark.parametrize('backend', ['zero_cta', 'all2allv'])
@pytest.mark.parametrize('method', [False, True])
@pytest.mark.parametrize('field', ['cast_lse', 'input_lse', 'output_lse'])
def test_handle_cast_lse_rejected_before_submission(monkeypatch, backend, method, field):
    monkeypatch.setattr(zero_cta, '_get_extension', forbidden)
    monkeypatch.setattr(collectives, '_execute', forbidden)
    handle = handle_for(backend)
    # Empty tensors are still non-None requests, even with cast_lse=False.
    kwargs = {field: True if field == 'cast_lse' else torch.empty(0)}
    args = (torch.ones(1), torch.zeros(1))
    with pytest.raises(NotImplementedError, match='does not support LSE'):
        if method:
            handle.group_cast(*args, stream=None, **kwargs)
        else:
            collectives.group_cast(handle, *args, stream=None, **kwargs)


@pytest.mark.parametrize('field', ['cast_lse', 'input_lse', 'output_lse'])
def test_explicit_cast_lse_rejected_before_preparation(monkeypatch, field):
    for name in ('_group', 'supplied_routes', '_execute_explicit'):
        monkeypatch.setattr(collectives, name, forbidden)
    kwargs = {field: True if field == 'cast_lse' else torch.empty(0)}
    with pytest.raises(NotImplementedError, match='does not support LSE'):
        collectives.group_cast_explicit(torch.ones(1), torch.zeros(1), [], [], [], [],
                                        group=None, stream=None, **kwargs)


def test_reserved_defaults_and_native_wrapper(monkeypatch):
    for fn in (collectives.group_cast, collectives.Handle.group_cast, collectives.group_cast_explicit):
        params = inspect.signature(fn).parameters
        assert params['cast_lse'].default is False
        assert params['input_lse'].default is params['output_lse'].default is None
    events = []
    def run(input, output, runtime, plan):
        events.append('run')
        output.copy_(input)
        return SimpleNamespace(wait=lambda: events.append('wait'))
    monkeypatch.setattr(zero_cta, '_get_extension', lambda: SimpleNamespace(run=run))
    handle = handle_for('zero_cta')
    input, output = torch.ones(2, 3), torch.zeros(2, 3)
    assert collectives.group_cast(handle, input, output, stream=None) is None
    assert handle.group_cast(input, output, stream=None, cast_lse=False,
                             input_lse=None, output_lse=None) is None
    assert events == ['run', 'wait', 'run', 'wait']
    torch.testing.assert_close(output, input)


@pytest.mark.parametrize('backend', ['zero_cta', 'all2allv'])
@pytest.mark.parametrize('async_op', [False, True])
def test_reduce_options_rejected_before_preparation_or_submission(monkeypatch, backend, async_op):
    monkeypatch.setattr(zero_cta, '_get_extension', forbidden)
    for name in ('_execute', '_group', 'supplied_routes', '_execute_explicit'):
        monkeypatch.setattr(collectives, name, forbidden)
    handle = handle_for(backend)
    suffix = '_async' if async_op else ''
    function = getattr(collectives, 'group_reduce' + suffix)
    method = getattr(handle, 'group_reduce' + suffix)
    explicit = getattr(collectives, 'group_reduce_explicit' + suffix)
    input, output = torch.ones(1), torch.zeros(1)
    for kwargs, error in (({'reduce_op': 'avg'}, 'only sum'),
                          ({'reduce_op': 'lse'}, 'only sum'),
                          ({'acc_reduce': False}, 'acc_reduce=True'),
                          ({'comm_dtype': torch.float64}, 'dtype conversion'),
                          ({'input_lse': torch.empty(0)}, 'LSE'),
                          ({'output_lse': torch.empty(0)}, 'LSE')):
        with pytest.raises(NotImplementedError, match=error):
            function(handle, input, output, stream=None, **kwargs)
        with pytest.raises(NotImplementedError, match=error):
            method(input, output, stream=None, **kwargs)
        with pytest.raises(NotImplementedError, match=error):
            explicit(input, output, [], [], [], [], group=None, stream=None, **kwargs)
    assert torch.equal(output, torch.zeros_like(output))


def test_reduce_option_defaults_on_all_public_entries():
    for suffix in ('', '_async'):
        for function in (getattr(collectives, 'group_reduce' + suffix),
                         getattr(collectives.Handle, 'group_reduce' + suffix),
                         getattr(collectives, 'group_reduce_explicit' + suffix)):
            params = inspect.signature(function).parameters
            for name, default in dict(reduce_op='sum', acc_reduce=True, comm_dtype=None,
                                      input_lse=None, output_lse=None).items():
                assert params[name].default == default
                assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize('backend', ['zero_cta', 'all2allv'])
@pytest.mark.parametrize('async_op', [False, True])
def test_reduce_matching_dtype_preserves_accumulation_and_wait(monkeypatch, backend, async_op):
    pg = object()
    state = collectives._GroupState(pg, torch.device('cpu'))
    group = CpGroup(pg, CpConfig(max_per_peer_slot=2, payload_shape=(3,),
                                dtype=torch.float32, backend='all2allv'), state)
    monkeypatch.setattr(collectives, '_group', lambda value: pg)
    monkeypatch.setattr(collectives.dist, 'get_rank', lambda value: 0)
    monkeypatch.setattr(collectives.dist, 'get_world_size', lambda value: 1)
    monkeypatch.setattr(collectives.dist, 'get_backend', lambda value: 'gloo')
    route = Route((2,), (2,), ((0,),), (0,))
    handle = collectives.Handle(group, state, route, True, backend == 'zero_cta', '')
    handle._native = SimpleNamespace(runtime=object(), reduce_plan=object())
    events = []

    def native_run(input, output, runtime, plan):
        assert runtime is handle._native.runtime and plan is handle._native.reduce_plan
        events.append('launch')
        return SimpleNamespace(wait=lambda: (events.append('wait'), output.add_(input)))

    def exchange(received, packed, *args, **kwargs):
        events.append('launch')
        return SimpleNamespace(wait=lambda: (events.append('wait'), received.copy_(packed)))

    monkeypatch.setattr(zero_cta, '_get_extension', lambda: SimpleNamespace(run=native_run))
    monkeypatch.setattr(flat.dist, 'all_to_all_single', exchange)
    suffix = '_async' if async_op else ''
    input = torch.ones(2, 3)
    for entry in ('function', 'method', 'explicit'):
        # Explicit preparation uses real CPU/flat routing; native preparation
        # requires CUDA and is outside this local public-option check.
        for dtype in (None, input.dtype):
            events.clear()
            output = torch.full_like(input, 7)
            kwargs = dict(stream=None, reduce_op='sum', acc_reduce=True,
                          comm_dtype=dtype, input_lse=None, output_lse=None)
            if entry == 'function':
                result = getattr(collectives, 'group_reduce' + suffix)(handle, input, output, **kwargs)
            elif entry == 'method':
                result = getattr(handle, 'group_reduce' + suffix)(input, output, **kwargs)
            else:
                result = getattr(collectives, 'group_reduce_explicit' + suffix)(
                    input, output, [[2]], [[2]], [[0]], [[[0]]], group=group, **kwargs)
            if async_op:
                assert events == ['launch']
                torch.testing.assert_close(output, torch.full_like(input, 7))
                assert result.wait_post_process() is output
            else:
                assert result is None
            assert events == ['launch', 'wait']
            torch.testing.assert_close(output, torch.full_like(input, 8))
