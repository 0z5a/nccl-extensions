"""Rank-local metadata must match the full reference's submitted work exactly."""


import pytest
from nccl.cp import comm_meta as meta
from nccl.cp import routing
from nccl.cp import zero_cta as backend
from route_cases import expected_cast, expected_reduce, make_case, relay_case
from route_reference import reference_local_route, validate_routes
from test_metadata import capture, simulate


def case_layouts(case):
    world = len(case["inputs"])
    return [(routing.layout_intervals([rank * 10000 + i for i in range(sum(case["inputs"][rank]))]),
             routing.layout_intervals(expected_cast(case, rank))) for rank in range(world)]


@pytest.mark.parametrize("world,nvl", [(1, 1), (2, 8), (4, 1), (4, 2), (6, 2), (8, 4)])
def test_owned_entries_and_local_plans_match_full_reference(monkeypatch, world, nvl):
    events = capture(monkeypatch, meta, backend)
    for seed in range(12):
        case = make_case(world, seed + 2700)
        layouts = case_layouts(case)
        ownership = routing.validate_layouts(layouts)
        routes = [routing.local_route(rank, layouts, build_entries=True, ownership=ownership) for rank in range(world)]
        full_entries = validate_routes(routes)
        assert tuple(entry for route in routes for entry in route.entries) == full_entries
        local_args = []
        for rank, route in enumerate(routes):
            assert route == reference_local_route(rank, layouts)
            assert all(entry.owner == rank for entry in route.entries)
            kwargs = dict(input_split_size_list=list(route.input_splits), output_split_size_list=list(route.output_splits),
                          dst_indices_list=[list(peers) for peers in route.destinations], src_index_list=list(route.sources),
                          rank=rank, world_size=world, group="cpu-fixture", max_per_peer_slot=case["capacity"],
                          max_per_token_bytes=24, nvl_domain_size=nvl)
            events.clear()
            reference = meta.ZeroCTACollectiveArg(**kwargs, hierarchy_meta=tuple(
                meta.GroupCollectiveEntry(e.owner, e.input_start, e.n_tokens, e.output_rank_starts) for e in full_entries))
            reference_events = list(events)
            local_entries = routing.local_hierarchy_entries(rank, layouts, route, nvl)
            events.clear()
            local = meta.ZeroCTACollectiveArg(**kwargs, hierarchy_meta=tuple(
                meta.GroupCollectiveEntry(e.owner, e.input_start, e.n_tokens, e.output_rank_starts) for e in local_entries))
            assert events == reference_events
            assert vars(local.cast_plan) == vars(reference.cast_plan)
            assert vars(local.reduce_plan) == vars(reference.reduce_plan)
            local_args.append(local)
        assert simulate(case, local_args, nvl) == [expected_cast(case, r) for r in range(world)]
        assert simulate(case, local_args, nvl, reduce=True) == [expected_reduce(case, r, 7) for r in range(world)]


def test_consumer_keeps_local_node_prefix_and_skips_unrelated_owner(monkeypatch):
    layouts = [(routing.layout_intervals(owned), routing.layout_intervals(wanted)) for owned, wanted in [
        ([0, 1, 2, 3], []), ([100, 101], []), ([], [0, 2]), ([], [1, 3]),
        ([400, 401], []), ([], []), ([], [400, 401]), ([], []),
    ]]
    routes = [routing.local_route(rank, layouts, build_entries=True) for rank in range(8)]

    original = routing._input_segments
    owners = []

    def relevant_segments(owner, layouts):
        assert owner != 4, "rank 3 must not derive unrelated owner 4's mapping"
        owners.append(owner)
        return original(owner, layouts)

    monkeypatch.setattr(routing, "_input_segments", relevant_segments)
    entries = routing.local_hierarchy_entries(3, layouts, routes[3], 2)
    assert 3 not in owners  # This rank's own mapping is already cached.
    assert [(e.owner, e.input_start, e.n_tokens, e.output_rank_starts) for e in entries] == [
        (0, 0, 1, ((2, 0),)), (0, 1, 1, ((3, 0),)),
        (0, 2, 1, ((2, 1),)), (0, 3, 1, ((3, 1),)),
    ]
    # The two prefix-only entries keep rank 3's receive offsets at 1 and 3.
    capture(monkeypatch, meta, backend)
    route = routes[3]
    arg = meta.ZeroCTACollectiveArg(
        input_split_size_list=list(route.input_splits), output_split_size_list=list(route.output_splits),
        dst_indices_list=[list(peers) for peers in route.destinations], src_index_list=list(route.sources),
        rank=3, world_size=8, group="cpu-fixture", max_per_peer_slot=4, max_per_token_bytes=24, nvl_domain_size=2,
        hierarchy_meta=tuple(meta.GroupCollectiveEntry(e.owner, e.input_start, e.n_tokens, e.output_rank_starts) for e in entries))
    assert arg.cast_plan.post_gather_tiles == ((2, 1, 0, 1), (2, 3, 1, 1))


def test_relay_only_rank_preserves_owner_gaps(monkeypatch):
    events = capture(monkeypatch, meta, backend)
    case = relay_case(4, 2)
    layouts = case_layouts(case)
    routes = [routing.local_route(rank, layouts, build_entries=True) for rank in range(4)]
    assert routes[2].input_rows == routes[2].output_rows == 0 and not routes[2].entries
    local = routing.local_hierarchy_entries(2, layouts, routes[2], 2)
    assert [(entry.input_start, entry.n_tokens) for entry in local] == [(0, 60), (80, 40)]
    full = validate_routes(routes)
    kwargs = dict(input_split_size_list=[], output_split_size_list=[], dst_indices_list=[], src_index_list=[],
                  rank=2, world_size=4, group="cpu-fixture", max_per_peer_slot=120, max_per_token_bytes=24, nvl_domain_size=2)
    def build(entries):
        return meta.ZeroCTACollectiveArg(**kwargs, hierarchy_meta=tuple(
            meta.GroupCollectiveEntry(e.owner, e.input_start, e.n_tokens, e.output_rank_starts) for e in entries))
    build(full)
    reference_events = list(events)
    events.clear()
    build(local)
    assert events == reference_events


def test_local_route_indexes_only_intersecting_ids(monkeypatch):
    layouts = [(((1000 * rank, 1000 * rank + 4),), ()) for rank in range(128)]
    layouts[0] = (((0, 4),), ((7000, 7002),))
    original = routing.bisect_right
    sizes = []
    def search(index, value):
        sizes.append(len(index))
        return original(index, value)
    monkeypatch.setattr(routing, "bisect_right", search)
    route = routing.local_route(0, layouts, build_entries=True)
    assert route.sources == (7,) and route.output_splits == (2,)
    assert max(sizes) <= 2  # Local input/request ranges, not all 128 owners.
    assert not route.entries


def test_nonmonotonic_ids_and_consumer_boundaries():
    layouts = [(routing.layout_intervals(owned), routing.layout_intervals(wanted)) for owned, wanted in [
        ([10, 11, 0, 1], [10, 100, 11, 0, 1]), ([100, 101], [10, 11, 0, 101, 1]),
    ]]
    routes = [routing.local_route(rank, layouts, build_entries=True) for rank in range(2)]
    assert tuple(e for route in routes for e in route.entries) == validate_routes(routes)
    assert all(routes[r] == reference_local_route(r, layouts) for r in range(2))


def test_empty_rank_leaves_duplicate_owner_check_to_owners():
    layouts = [((), ()), (((0, 2),), ()), (((1, 3),), ())]
    assert routing.local_route(0, layouts).input_rows == 0
    for rank in (1, 2):
        with pytest.raises(ValueError, match="more than one owner"):
            routing.local_route(rank, layouts)


@pytest.mark.parametrize("layouts", [
    [((), ()), (((0, 2),), ()), (((1, 3),), ())],
    [((), ()), (((0, 2),), ((10, 11),))],
    [((), ()), (((0, 3),), ((2, 3), (0, 1)))],
])
def test_shared_validation_needs_no_remote_routes(monkeypatch, layouts):
    def forbidden(*args, **kwargs):
        raise AssertionError("Global validation must not construct rank routes or owner mappings")
    monkeypatch.setattr(routing.Route, "from_lists", forbidden)
    monkeypatch.setattr(routing, "_input_segments", forbidden)
    with pytest.raises(ValueError, match="owner|preserve input order"):
        routing.validate_layouts(layouts)


def test_create_handle_uses_one_exchange_and_only_one_rank_route(monkeypatch):
    from types import SimpleNamespace

    import torch
    from nccl.cp import CpConfig, CpGroup, collectives

    case = relay_case(4, 2)
    layouts = case_layouts(case)
    rank, world, nvl = 2, 4, 2
    nccl_group = object()
    state = collectives._GroupState(nccl_group, torch.device("cpu"))
    config = CpConfig(max_per_peer_slot=120, payload_shape=(6,), dtype=torch.float32, nvl_domain_size=nvl)
    group = CpGroup(nccl_group, config, state)
    profiles = [dict(ok=True, reason="", host=str(r // nvl), uuid=str(r),
                     accessible=tuple(str(p) for p in range(r // nvl * nvl, (r // nvl + 1) * nvl))) for r in range(world)]
    monkeypatch.setattr(collectives, "_group", lambda group: nccl_group)
    monkeypatch.setattr(collectives, "_state", lambda group: state)
    monkeypatch.setattr(collectives.dist, "get_rank", lambda group: rank)
    monkeypatch.setattr(collectives.dist, "get_world_size", lambda group: world)
    monkeypatch.setattr(collectives, "_probe_zero", lambda group: profiles[rank])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 << 30, 1 << 30))
    stream = SimpleNamespace(wait_event=lambda event: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None, query=lambda: False))
    calls, computed, native_arguments = [], [], []

    def exchange(group, phase, value=None, error="", **kwargs):
        assert not calls, "create_handle must not issue a second metadata exchange"
        assert not error and phase == "create_handle/layouts"
        calls.append(phase)
        assert value[0] == layouts[rank]  # Only descriptors, never a derived Route.
        return [(layout, config, None, profiles[r]) for r, layout in enumerate(layouts)]

    original = routing.local_route
    def local_only(r, descriptions, **kwargs):
        assert r == rank, "must not compute another rank's four lists"
        computed.append(r)
        return original(r, descriptions, **kwargs)

    def build(**kwargs):
        native_arguments.append(kwargs)
        return SimpleNamespace(runtime=object())

    monkeypatch.setattr(collectives, "_exchange", exchange)
    monkeypatch.setattr(collectives, "local_route", local_only)
    monkeypatch.setattr(meta, "ZeroCTACollectiveArg", build)
    handle = collectives.create_handle(group, [], [], stream=None)
    assert calls == ["create_handle/layouts"] and computed == [rank]
    assert handle.backend == "zero_cta" and handle._has_transfers
    assert [(e.input_start, e.n_tokens) for e in native_arguments[0]["hierarchy_meta"]] == [(0, 60), (80, 40)]
