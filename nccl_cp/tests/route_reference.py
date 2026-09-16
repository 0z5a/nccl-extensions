"""Full-route reference used only for metadata-equivalence tests."""

from bisect import bisect_right
from collections.abc import Sequence

from nccl.cp.routing import Entry, Intervals, Route, peer_segments


def validate_routes(routes: Sequence[Route]) -> tuple[Entry, ...]:
    """Pair ordered peer streams, splitting entries at all output boundaries."""
    world = len(routes)
    segments = [peer_segments(route, world) for route in routes]
    for owner in range(world):
        for dest in range(world):
            sent = sum(s.count for s in segments[owner][0][dest])
            received = sum(s.count for s in segments[dest][1][owner])
            if sent != received:
                raise ValueError(f"Route mismatch {owner}->{dest}: sends {sent}, receives {received}")
    entries: list[Entry] = []
    for owner, route in enumerate(routes):
        offsets = [0] * world
        chunk_ids = [0] * world
        start = 0
        for count, peers in zip(route.input_splits, route.destinations):
            consumed = 0
            while peers and consumed < count:
                chunks = {peer: segments[peer][1][owner][chunk_ids[peer]] for peer in peers}
                size = min(count - consumed, *(chunks[peer].count - offsets[peer] for peer in peers))
                entries.append(Entry(owner, start + consumed, size,
                    tuple((peer, chunks[peer].start + offsets[peer]) for peer in peers)))
                consumed += size
                for peer in peers:
                    offsets[peer] += size
                    if offsets[peer] == chunks[peer].count:
                        offsets[peer] = 0
                        chunk_ids[peer] += 1
            start += count
    return tuple(entries)


def reference_local_route(rank: int, layouts: Sequence[tuple[Intervals, Intervals]]) -> Route:
    """Compute this rank's lists by interval intersections, without expanding IDs."""
    ownership: list[tuple[int, int, int, int]] = []
    for peer, (owned, _requested) in enumerate(layouts):
        offset = 0
        for start, stop in owned:
            ownership.append((start, stop, peer, offset))
            offset += stop - start
    ownership.sort()
    if any(left[1] > right[0] for left, right in zip(ownership, ownership[1:])):
        raise ValueError("A token has more than one owner")
    owner_starts = [item[0] for item in ownership]
    requests = [sorted(pair[1]) for pair in layouts]
    request_starts = [[part[0] for part in request] for request in requests]
    input_splits: list[int] = []
    destinations: list[list[int]] = []
    for start, stop in layouts[rank][0]:
        events: dict[int, dict[int, int]] = {start: {}, stop: {}}
        for peer, intervals in enumerate(requests):
            index = max(0, bisect_right(request_starts[peer], start) - 1)
            while index < len(intervals) and intervals[index][0] < stop:
                left, right = max(start, intervals[index][0]), min(stop, intervals[index][1])
                if left < right:
                    for position, delta in ((left, 1), (right, -1)):
                        changes = events.setdefault(position, {})
                        changes[peer] = changes.get(peer, 0) + delta
                index += 1
        points = sorted(events)
        active: dict[int, int] = {}
        for position, following in zip(points, points[1:]):
            for peer, delta in events[position].items():
                active[peer] = active.get(peer, 0) + delta
            peers = sorted(peer for peer, count in active.items() if count)
            count = following - position
            if destinations and destinations[-1] == peers:
                input_splits[-1] += count
            else:
                input_splits.append(count)
                destinations.append(peers)
    output_splits: list[int] = []
    sources: list[int] = []
    last_position: dict[int, int] = {}
    for start, stop in layouts[rank][1]:
        position = start
        while position < stop:
            index = bisect_right(owner_starts, position) - 1
            if index < 0 or not ownership[index][0] <= position < ownership[index][1]:
                raise ValueError(f"Requested token {position} has no owner")
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
    return Route.from_lists(input_splits, output_splits, destinations, sources, len(layouts))
