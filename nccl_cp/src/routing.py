"""Rank-local route descriptions and host execution plans."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import accumulate


@dataclass(frozen=True)
class RowRange:
    """Half-open range of global row IDs [start, stop), in tensor axis-0 order.

    Caller-assigned IDs identify communication rows, such as tokens or heads.
    They are not vocabulary IDs or local tensor offsets. The caller defines
    each row's payload and keeps the layout consistent with tensor contents.
    """
    start: int
    stop: int

    def __post_init__(self):
        if type(self.start) is not int or type(self.stop) is not int or self.stop < self.start:
            raise ValueError("RowRange requires integer start <= stop")


Layout = Sequence[int | RowRange]
Intervals = tuple[tuple[int, int], ...]


def layout_intervals(layout: Layout) -> Intervals:
    ranges: list[tuple[int, int]] = []
    for value in layout:
        if type(value) is int:
            start, stop = value, value + 1
        elif isinstance(value, RowRange):
            start, stop = value.start, value.stop
        else:
            raise TypeError("Layouts contain integer row IDs or RowRange values")
        if start == stop:
            continue
        if ranges and ranges[-1][1] == start:
            ranges[-1] = ranges[-1][0], stop
        else:
            ranges.append((start, stop))
    ordered = sorted(ranges)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("A local layout must not repeat a row ID")
    return tuple(ranges)


def expand_layout(layout: Intervals) -> tuple[int, ...]:
    return tuple(row for start, stop in layout for row in range(start, stop))


@dataclass(frozen=True)
class Route:
    """Serializable four-list payload of one rank's GroupCollectiveArg.

    Only layouts are exchanged. The four lists and optional local-owner entries
    stay on this rank. Entry caching avoids deriving the same local mapping again
    when constructing the hierarchical plan; equality compares only the lists.
    """
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    destinations: tuple[tuple[int, ...], ...]
    sources: tuple[int, ...]
    entries: tuple[Entry, ...] = field(default=(), compare=False)
    input_rows: int = field(init=False, compare=False)
    output_rows: int = field(init=False, compare=False)

    def __post_init__(self):
        # Route counts are immutable; compute once during preparation, not for
        # every payload validation. Explicit-list calls still construct a route.
        object.__setattr__(self, "input_rows", sum(self.input_splits))
        object.__setattr__(self, "output_rows", sum(self.output_splits))

    @classmethod
    def from_lists(cls, input_splits, output_splits, destinations, sources, world: int, *, entries: tuple[Entry, ...] = ()):
        route = cls(tuple(input_splits), tuple(output_splits),
                    tuple(tuple(peers) for peers in destinations), tuple(sources), entries)
        if len(route.input_splits) != len(route.destinations):
            raise ValueError("input_split_size_list and dst_indices_list lengths differ")
        if len(route.output_splits) != len(route.sources):
            raise ValueError("output_split_size_list and src_index_list lengths differ")
        if any(type(n) is not int or n < 0 for n in route.input_splits + route.output_splits):
            raise ValueError("Split sizes must be nonnegative integers")
        for peers in route.destinations:
            if len(set(peers)) != len(peers):
                raise ValueError("A destination list must not repeat a rank")
        if any(type(rank) is not int or not 0 <= rank < world
               for rank in (*route.sources, *(r for peers in route.destinations for r in peers))):
            raise ValueError("Route ranks must be relative to the ProcessGroup")
        return route


def validate_layouts(layouts: Sequence[tuple[Intervals, Intervals]]) -> tuple[tuple[int, int, int, int], ...]:
    """Validate exchanged descriptions identically on all ranks, without routes.

    This host scan replaces a second collective for rank-local routing errors.
    It creates no peer split lists or execution plans. The returned ownership
    index is reused for the current rank's output lookup.
    It validates the descriptions only; callers still bind actual tensor rows
    to those descriptions and coordinate which handle each rank executes.
    """
    ownership = []
    for peer, (owned, _) in enumerate(layouts):
        offset = 0
        for start, stop in owned:
            ownership.append((start, stop, peer, offset))
            offset += stop - start
    ownership.sort()
    if any(left[1] > right[0] for left, right in zip(ownership, ownership[1:])):
        raise ValueError("A row has more than one owner")
    starts = [part[0] for part in ownership]
    for rank, (_, requested) in enumerate(layouts):
        last_position: dict[int, int] = {}
        for start, stop in requested:
            position = start
            while position < stop:
                index = bisect_right(starts, position) - 1
                if index < 0 or not ownership[index][0] <= position < ownership[index][1]:
                    raise ValueError(f"rank {rank}: requested row {position} has no owner")
                first, end, owner, offset = ownership[index]
                count = min(stop, end) - position
                source_position = offset + position - first
                if source_position <= last_position.get(owner, -1):
                    raise ValueError(f"rank {rank}: output must preserve input order within each source rank")
                last_position[owner] = source_position + count - 1
                position += count
    return tuple(ownership)


def _input_segments(owner: int, layouts: Sequence[tuple[Intervals, Intervals]]):
    """Yield one needed owner's canonical (row, count, consumer positions).

    Boundaries include all consumers to preserve the native metadata,
    but this generator creates neither a peer's four-list Route nor its plan.
    """
    owned = layouts[owner][0]
    input_ranges = sorted((start, stop, index) for index, (start, stop) in enumerate(owned))
    input_starts = [part[0] for part in input_ranges]
    events = [{start: {}, stop: {}} for start, stop in owned]
    if input_ranges:
        for peer, (_, requested) in enumerate(layouts):
            offset = 0
            for start, stop in requested:
                index = max(0, bisect_right(input_starts, start) - 1)
                while index < len(input_ranges) and input_ranges[index][0] < stop:
                    first, last, local_index = input_ranges[index]
                    left, right = max(start, first), min(stop, last)
                    if left < right:
                        for position, delta in ((left, 1), (right, -1)):
                            changes = events[local_index].setdefault(position, {})
                            previous, origin = changes.get(peer, (0, None))
                            changes[peer] = previous + delta, offset - start if delta > 0 else origin
                    index += 1
                offset += stop - start
    input_offset = 0
    pending = None
    for (start, stop), points_map in zip(owned, events):
        points = sorted(points_map)
        active: dict[int, int] = {}
        for position, following in zip(points, points[1:]):
            for peer, (delta, origin) in points_map[position].items():
                if origin is not None:
                    active[peer] = origin
                elif delta < 0:
                    active.pop(peer, None)
            row, count = input_offset + position - start, following - position
            outputs = tuple((peer, position + active[peer]) for peer in sorted(active))
            if (pending is not None and pending[0] + pending[1] == row and
                    tuple((peer, offset + pending[1]) for peer, offset in pending[2]) == outputs):
                pending = pending[0], pending[1] + count, pending[2]
            else:
                if pending is not None:
                    yield pending
                pending = row, count, outputs
        input_offset += stop - start
    if pending is not None:
        yield pending


def local_route(rank: int, layouts: Sequence[tuple[Intervals, Intervals]], *, build_entries: bool = False,
                ownership: Sequence[tuple[int, int, int, int]] | None = None) -> Route:
    """Compute only this rank's four lists and optional local-owner entries."""
    owned, requested = layouts[rank]
    if ownership is None:
        # Restrict ownership lookup/validation to locally owned or requested IDs.
        interests: list[tuple[int, int]] = []
        for start, stop in sorted((*owned, *requested)):
            if interests and start <= interests[-1][1]:
                interests[-1] = interests[-1][0], max(stop, interests[-1][1])
            else:
                interests.append((start, stop))
        interest_starts = [part[0] for part in interests]
        ownership: list[tuple[int, int, int, int]] = []
        if interests:
            for peer, (peer_owned, _) in enumerate(layouts):
                offset = 0
                for start, stop in peer_owned:
                    index = max(0, bisect_right(interest_starts, start) - 1)
                    while index < len(interests) and interests[index][0] < stop:
                        left, right = max(start, interests[index][0]), min(stop, interests[index][1])
                        if left < right:
                            ownership.append((left, right, peer, offset + left - start))
                        index += 1
                    offset += stop - start
        ownership.sort()
        if any(left[1] > right[0] for left, right in zip(ownership, ownership[1:])):
            raise ValueError("A row has more than one owner")
    owner_starts = [item[0] for item in ownership]
    input_splits: list[int] = []
    destinations: list[list[int]] = []
    entries: list[Entry] = []
    for row, count, outputs in _input_segments(rank, layouts):
        peers = [peer for peer, _ in outputs]
        if destinations and destinations[-1] == peers:
            input_splits[-1] += count
        else:
            input_splits.append(count)
            destinations.append(peers)
        if build_entries and outputs:
            entries.append(Entry(rank, row, count, outputs))
    output_splits: list[int] = []
    sources: list[int] = []
    last_position: dict[int, int] = {}
    for start, stop in requested:
        position = start
        while position < stop:
            index = bisect_right(owner_starts, position) - 1
            if index < 0 or not ownership[index][0] <= position < ownership[index][1]:
                raise ValueError(f"Requested row {position} has no owner")
            first, end, peer, offset = ownership[index]
            count = min(stop, end) - position
            source_position = offset + position - first
            if source_position <= last_position.get(peer, -1):
                raise ValueError("Output must preserve input order within each source rank")
            last_position[peer] = source_position + count - 1
            if sources and sources[-1] == peer:
                output_splits[-1] += count
            else:
                sources.append(peer)
                output_splits.append(count)
            position += count
    return Route.from_lists(input_splits, output_splits, destinations, sources, len(layouts), entries=tuple(entries))


@dataclass(frozen=True)
class Segment:
    start: int
    count: int


@dataclass(frozen=True)
class Entry:
    """A source interval and its destination offsets, all in token rows."""
    owner: int
    input_start: int
    n_tokens: int
    output_rank_starts: tuple[tuple[int, int], ...]


def peer_segments(route: Route, world: int):
    send: list[list[Segment]] = [[] for _ in range(world)]
    recv: list[list[Segment]] = [[] for _ in range(world)]
    for start, count, peers in zip(accumulate((0, *route.input_splits)), route.input_splits,
                                   route.destinations):
        if count:
            for peer in peers:
                send[peer].append(Segment(start, count))
    for start, count, peer in zip(accumulate((0, *route.output_splits)), route.output_splits,
                                  route.sources):
        if count:
            recv[peer].append(Segment(start, count))
    return tuple(tuple(parts) for parts in send), tuple(tuple(parts) for parts in recv)


def local_hierarchy_entries(rank: int, layouts: Sequence[tuple[Intervals, Intervals]],
                            route: Route, nvl: int) -> tuple[Entry, ...]:
    """Derive only this rank's owner/consumer/relay mappings from shared layouts.

    Local-owner entries are already cached by local_route. For other relevant
    owners, retain only local-node rows, including prefix rows needed for the
    node offsets. No other rank's four-list Route or plan is computed.
    """
    if len(layouts) <= nvl:
        return ()
    node, lane = rank // nvl, rank % nvl
    sources = set(route.sources)
    result: list[Entry] = []
    for owner in range(len(layouts)):
        if owner == rank:
            result.extend(route.entries)
        elif owner in sources or owner % nvl == lane:
            for row, count, outputs in _input_segments(owner, layouts):
                local_outputs = tuple((peer, offset) for peer, offset in outputs if peer // nvl == node)
                if local_outputs:
                    result.append(Entry(owner, row, count, local_outputs))
    return tuple(result)


def supplied_routes(
    input_split_size_list: list[list[int]],
    output_split_size_list: list[list[int]],
    dst_indices_list: list[list[list[int]]],
    src_index_list: list[list[int]],
    world: int,
) -> tuple[Route, ...]:
    """Validate caller-supplied all-rank lists; do not derive any rank's lists.

    Outer index is the ProcessGroup-relative rank. Peer streams pair in source
    order; token identity and actual tensor contents remain caller obligations.
    No device access, token expansion or control collective is needed here.
    """
    if any(len(values) != world for values in (
        input_split_size_list, output_split_size_list, dst_indices_list, src_index_list
    )):
        raise ValueError("Each global route list must contain one entry per group rank")
    routes = tuple(Route.from_lists(*values, world) for values in zip(
        input_split_size_list, output_split_size_list, dst_indices_list, src_index_list
    ))
    sent = [[0] * world for _ in range(world)]
    received = [[0] * world for _ in range(world)]
    for rank, route in enumerate(routes):
        for count, peers in zip(route.input_splits, route.destinations):
            for peer in peers:
                sent[rank][peer] += count
        for count, owner in zip(route.output_splits, route.sources):
            received[owner][rank] += count
    for owner in range(world):
        for consumer in range(world):
            if sent[owner][consumer] != received[owner][consumer]:
                raise ValueError(
                    f"Route mismatch {owner}->{consumer}: sends {sent[owner][consumer]}, "
                    f"receives {received[owner][consumer]}"
                )
    return routes


def local_hierarchy_from_routes(rank: int, routes: Sequence[Route], nvl: int) -> tuple[Entry, ...]:
    """Pair supplied peer streams for this rank's owner/consumer/relay work.

    Only relevant owners are matched. Keep local-node prefix entries even when
    this rank is not a consumer, so relay receive offsets match the owner.
    Other ranks' four lists are already supplied; no remote plans are built.
    """
    world = len(routes)
    if world <= nvl:
        return ()
    received: list[list[list[Segment]]] = [[[] for _ in range(world)] for _ in range(world)]
    for consumer, route in enumerate(routes):
        start = 0
        for count, owner in zip(route.output_splits, route.sources):
            if count:
                received[consumer][owner].append(Segment(start, count))
            start += count
    node, lane = rank // nvl, rank % nvl
    sources = set(routes[rank].sources)
    entries: list[Entry] = []
    for owner, route in enumerate(routes):
        if owner != rank and owner not in sources and owner % nvl != lane:
            continue
        offsets, chunks = [0] * world, [0] * world
        start = 0
        for count, peers in zip(route.input_splits, route.destinations):
            consumed = 0
            while peers and consumed < count:
                current = {peer: received[peer][owner][chunks[peer]] for peer in peers}
                size = min(count - consumed, *(current[peer].count - offsets[peer] for peer in peers))
                outputs = tuple((peer, current[peer].start + offsets[peer]) for peer in peers
                                if owner == rank or peer // nvl == node)
                if outputs:
                    entries.append(Entry(owner, start + consumed, size, outputs))
                consumed += size
                for peer in peers:
                    offsets[peer] += size
                    if offsets[peer] == current[peer].count:
                        offsets[peer] = 0
                        chunks[peer] += 1
            start += count
    return tuple(entries)
