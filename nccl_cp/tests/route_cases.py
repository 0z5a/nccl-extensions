"""Shared, deterministic input fixtures for CPU and real GPU comparisons."""

import random


def make_case(world, seed):
    rng = random.Random(seed)
    inputs = [[] for _ in range(world)]
    destinations = [[] for _ in range(world)]
    outputs = [[] for _ in range(world)]
    sources = [[] for _ in range(world)]
    entries = []
    output_cursors = [0] * world
    for owner in range(world):
        input_cursor = 0
        for _ in range(rng.randrange(0, 6)):
            count = rng.choice([0, 1, 2, 17, 31, 32, 33, 127, 128, 129, 300])
            consumers = [peer for peer in range(world) if rng.random() < 0.5]
            inputs[owner].append(count)
            destinations[owner].append(consumers)
            positions = []
            for peer in consumers:
                outputs[peer].append(count)
                sources[peer].append(owner)
                positions.append((peer, output_cursors[peer]))
                output_cursors[peer] += count
            if count:
                entries.append((owner, input_cursor, count, tuple(positions)))
            input_cursor += count
    return dict(
        inputs=inputs,
        outputs=outputs,
        destinations=destinations,
        sources=sources,
        entries=entries,
        capacity=max((sum(x) for x in inputs), default=0),
    )


def relay_case(world, nvl):
    """Synthetic relay route for the caller's logical domain geometry."""
    if not (world > nvl > 1 and world % nvl == 0):
        raise ValueError("Relay fixture requires complete domains with multiple ranks")
    remote_consumer = nvl + 1
    inputs, outputs = [[] for _ in range(world)], [[] for _ in range(world)]
    destinations, sources = [[] for _ in range(world)], [[] for _ in range(world)]
    inputs[0] = [60, 20, 40]
    outputs[1], outputs[remote_consumer] = [20], [60, 40]
    destinations[0] = [[remote_consumer], [1], [remote_consumer]]
    sources[1], sources[remote_consumer] = [0], [0, 0]
    return dict(
        inputs=inputs, outputs=outputs, destinations=destinations, sources=sources,
        entries=[(0, 0, 60, ((remote_consumer, 0),)), (0, 60, 20, ((1, 0),)),
                 (0, 80, 40, ((remote_consumer, 60),))],
        capacity=120,
    )


def kwargs_for(meta, case, rank, world, group, *, slot=0, payload_bytes=4096):
    return dict(
        input_split_size_list=case["inputs"][rank],
        output_split_size_list=case["outputs"][rank],
        dst_indices_list=case["destinations"][rank],
        src_index_list=case["sources"][rank],
        rank=rank,
        world_size=world,
        group=group,
        hierarchy_meta=tuple(meta.GroupCollectiveEntry(*row) for row in case["entries"]),
        max_per_peer_slot=case["capacity"],
        max_per_token_bytes=payload_bytes,
        runtime_slot=slot,
    )


def expected_cast(case, rank):
    result = [None] * sum(case["outputs"][rank])
    for owner, start, count, positions in case["entries"]:
        for consumer, offset in positions:
            if consumer == rank:
                result[offset : offset + count] = [owner * 10000 + start + i for i in range(count)]
    assert all(value is not None for value in result)
    return result


def expected_reduce(case, rank, initial=0):
    result = [initial] * sum(case["inputs"][rank])
    for owner, start, count, positions in case["entries"]:
        if owner == rank:
            value = sum(peer + 1 for peer, _ in positions)
            for i in range(start, start + count):
                result[i] += value
    return result
