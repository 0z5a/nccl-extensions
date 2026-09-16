import pytest
from nccl.cp.routing import (
    Route,
    TokenRange,
    layout_intervals,
    local_route,
    peer_segments,
)
from route_reference import validate_routes


def test_local_lists_and_ordered_pairing():
    layouts = [(layout_intervals([0, 1, 2]), layout_intervals([0, 10, 2])),
               (layout_intervals([10, 11]), layout_intervals([1, 2, 10, 11]))]
    routes = [local_route(rank, layouts) for rank in range(2)]
    assert routes[0] == Route((1, 1, 1), (1, 1, 1), ((0,), (1,), (0, 1)), (0, 1, 0))
    assert routes[1] == Route((1, 1), (2, 2), ((0, 1), (1,)), (0, 1))
    entries = validate_routes(routes)
    for entry in entries:
        owner_tokens = [0, 1, 2] if entry.owner == 0 else [10, 11]
        for dest, offset in entry.output_rank_starts:
            output_tokens = [0, 10, 2] if dest == 0 else [1, 2, 10, 11]
            assert owner_tokens[entry.input_start:entry.input_start + entry.n_tokens] == output_tokens[offset:offset + entry.n_tokens]


def test_compact_ranges_and_unused_tokens():
    assert layout_intervals([TokenRange(0, 3), 3, TokenRange(9, 9)]) == ((0, 4),)
    route = local_route(0, [(((0, 4),), ((1, 3),))])
    assert route == Route((1, 2, 1), (2,), ((), (0,), ()), (0,))
    assert sum(e.n_tokens for e in validate_routes([route])) == 2
    assert len(peer_segments(route, 1)[0][0]) == 1


@pytest.mark.parametrize("input_layout,output_layout", [
    ([1, 1], []), ([1], [1, 1]), ([True], []),
])
def test_invalid_layout_description(input_layout, output_layout):
    with pytest.raises((ValueError, TypeError)):
        layout_intervals(input_layout)
        layout_intervals(output_layout)


def test_unknown_owner_duplicate_owner_and_reordering():
    for layouts in [
        [(((0, 1),), ((1, 2),))],
        [(((0, 1),), ()), (((0, 1),), ())],
        [(((0, 3),), ((2, 3), (0, 1)))],
    ]:
        with pytest.raises(ValueError):
            local_route(0, layouts)


@pytest.mark.parametrize("args", [
    ([1], [], [], []), ([1], [], [[0, 0]], []),
    ([1], [1], [[1]], [0]), ([-1], [], [[]], []),
])
def test_invalid_explicit_lists(args):
    with pytest.raises(ValueError):
        Route.from_lists(*args, world=1)


def test_pair_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="sends"):
        validate_routes([Route((2,), (1,), ((0,),), (0,))])


def test_large_compact_ranges_do_not_expand_tokens():
    size = 10**12
    layouts = [(((0, size),), ((size - 2, size + 2),)), (((size, 2 * size),), ())]
    first = local_route(0, layouts)
    second = local_route(1, layouts)
    assert first.input_splits == (size - 2, 2)
    assert first.output_splits == (2, 2)
    assert first.sources == (0, 1)
    assert second.input_splits == (2, size - 2)
    entries = validate_routes([first, second])
    assert [entry.n_tokens for entry in entries] == [2, 2]
