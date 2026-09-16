"""Global-list public calls prepare local plans without exchanging route data."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from nccl.cp import CpConfig, CpGroup, collectives, comm_meta, routing, zero_cta
from route_cases import expected_cast, expected_reduce, make_case, relay_case
from route_reference import validate_routes
from test_metadata import capture, simulate


def lists(case):
    return case['inputs'], case['outputs'], case['destinations'], case['sources']


@pytest.mark.parametrize('world,nvl', [(2, 8), (4, 2), (6, 2), (8, 4)])
def test_global_lists_build_only_local_plans_matching_reference(monkeypatch, world, nvl):
    events = capture(monkeypatch, comm_meta, zero_cta)
    for seed in range(5):
        case = make_case(world, seed + 5700)
        routes = routing.supplied_routes(*lists(case), world)
        full = validate_routes(routes)
        args = []
        for rank, route in enumerate(routes):
            kwargs = dict(input_split_size_list=list(route.input_splits),
                          output_split_size_list=list(route.output_splits),
                          dst_indices_list=[list(x) for x in route.destinations],
                          src_index_list=list(route.sources), rank=rank, world_size=world,
                          group='fixture', max_per_peer_slot=case['capacity'],
                          max_per_token_bytes=24, nvl_domain_size=nvl)
            events.clear()
            reference = comm_meta.ZeroCTACollectiveArg(**kwargs, hierarchy_meta=tuple(
                comm_meta.GroupCollectiveEntry(e.owner, e.input_start, e.n_tokens, e.output_rank_starts)
                for e in full))
            expected_events = list(events)
            events.clear()
            entries = routing.local_hierarchy_from_routes(rank, routes, nvl)
            actual = comm_meta.ZeroCTACollectiveArg(**kwargs, hierarchy_meta=tuple(
                comm_meta.GroupCollectiveEntry(e.owner, e.input_start, e.n_tokens, e.output_rank_starts)
                for e in entries))
            assert events == expected_events
            assert vars(actual.cast_plan) == vars(reference.cast_plan)
            assert vars(actual.reduce_plan) == vars(reference.reduce_plan)
            args.append(actual)
        assert simulate(case, args, nvl) == [expected_cast(case, r) for r in range(world)]
        assert simulate(case, args, nvl, reduce=True) == [expected_reduce(case, r, 7) for r in range(world)]


def test_peer_stream_boundaries_and_relay_prefixes():
    routes = routing.supplied_routes(
        [[3, 5], [], [], []], [[], [2, 6], [1, 2], []],
        [[[1, 2], [1]], [], [], []], [[], [0, 0], [0, 0], []], 4)
    reference = validate_routes(routes)
    assert routing.local_hierarchy_from_routes(0, routes, 2) == reference
    actual = routing.local_hierarchy_from_routes(2, routes, 2)
    assert actual == tuple(routing.Entry(e.owner, e.input_start, e.n_tokens,
                                        tuple((r, offset) for r, offset in e.output_rank_starts if r // 2 == 1))
                           for e in reference if any(r // 2 == 1 for r, _ in e.output_rank_starts))
    case = relay_case(4, 2)
    routes = routing.supplied_routes(*lists(case), 4)
    assert routes[2].input_rows == routes[2].output_rows == 0
    entries = routing.local_hierarchy_from_routes(2, routes, 2)
    assert [(e.owner, e.input_start, e.n_tokens) for e in entries] == [(0, 0, 60), (0, 80, 40)]


@pytest.mark.parametrize('values,pattern', [
    (([[1]], [[1]], [[[0]]], [[0]]), 'one entry per'),
    (([[1], []], [[], [2]], [[[1]], []], [[], [0]]), 'Route mismatch'),
    (([[1], []], [[1], []], [[[0, 0]], []], [[0], []]), 'repeat'),
    (([[-1], []], [[], []], [[[]], []], [[], []]), 'nonnegative'),
])
def test_bad_global_descriptions_fail_locally(values, pattern):
    with pytest.raises(ValueError, match=pattern):
        routing.supplied_routes(*values, 2)


def test_global_public_api_has_no_handle_parameter():
    for fn in (collectives.group_cast_explicit, collectives.group_reduce_explicit):
        signature = inspect.signature(fn)
        assert 'handle' not in signature.parameters
        for name in ('input', 'output', 'group', 'stream'):
            assert signature.parameters[name].default is inspect.Parameter.empty
        with pytest.raises(TypeError, match='handle'):
            fn(None, None, [], [], [], [], group=None, stream=None, handle=object())


def test_explicit_native_prepares_relay_without_handle_or_control_exchange(monkeypatch):
    case = relay_case(4, 2)
    pg = object()
    state = collectives._GroupState(pg, torch.device('cuda', 0))
    cp_group = CpGroup(pg, CpConfig(max_per_peer_slot=120, payload_shape=(6,),
                                   dtype=torch.float32, nvl_domain_size=2), state)
    monkeypatch.setattr(collectives, '_group', lambda group: pg)
    monkeypatch.setattr(collectives, '_resolve_stream', lambda state, stream: None)
    monkeypatch.setattr(collectives.dist, 'get_world_size', lambda group: 4)
    monkeypatch.setattr(collectives.dist, 'get_rank', lambda group: 2)
    monkeypatch.setattr(collectives.dist, 'get_backend', lambda group: 'gloo')
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(collectives, '_probe_zero', lambda group: {'ok': True})
    events = capture(monkeypatch, comm_meta, zero_cta)

    def forbidden(*args, **kwargs):
        raise AssertionError('No hidden handle, route exchange, or new Python event/sync')

    for name in ('Handle', 'create_handle', '_exchange'):
        monkeypatch.setattr(collectives, name, forbidden)
    for name in ('all_gather', 'all_gather_object', 'all_reduce', 'barrier'):
        monkeypatch.setattr(collectives.dist, name, forbidden)
    for name in ('Event', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    submitted = []

    def submit(input, output, argument, group, **kwargs):
        assert group is pg and kwargs == {'async_op': True}
        assert argument.rank == 2
        assert argument.cast_plan.remote_wait_peers == (0,)
        submitted.append(argument)
        return SimpleNamespace(wait_post_process=lambda target: None)

    monkeypatch.setattr(zero_cta, 'zero_cta_group_cast_impl', submit)
    monkeypatch.setattr(zero_cta, 'zero_cta_group_reduce_impl', submit)
    data = torch.empty((0, 6))
    for _ in range(2):
        assert collectives.group_cast_explicit(data, data, *lists(case), group=cp_group, stream=None) is None
        assert collectives.group_reduce_explicit(data, data, case['outputs'], case['inputs'],
                                                case['sources'], case['destinations'], group=cp_group, stream=None) is None
    assert len(submitted) == 4 and len({id(arg) for arg in submitted}) == 4
    assert [event[0] for event in events].count('runtime') == 4
    assert state.runtimes[0] == submitted[-1].runtime
    monkeypatch.setattr(collectives, '_probe_zero', lambda group: {'ok': False, 'reason': 'missing extension'})
    monkeypatch.setattr(collectives.flat, 'run', forbidden)
    with pytest.raises(RuntimeError, match='missing extension'):
        collectives.group_cast_explicit(data, data, *lists(case), group=cp_group, stream=None)
