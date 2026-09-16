from types import SimpleNamespace

import pytest
from nccl.cp import comm_meta as meta
from nccl.cp import zero_cta as backend
from route_cases import expected_cast, expected_reduce, kwargs_for, make_case, relay_case


def freeze(value):
    if isinstance(value, (tuple, list)):
        return tuple(freeze(x) for x in value)
    return value


def capture(monkeypatch, target_meta, target_backend):
    events = []

    def upload(name):
        def call(values, **kwargs):
            value = freeze(values)
            events.append((name, value, kwargs))
            return value

        return call

    monkeypatch.setattr(target_meta, "_make_device_tensor", upload("tensor"))
    for name in (
        "_make_device_tensor_gather_tiles",
        "_make_device_symmetric_gather_tiles",
        "_make_device_reduce_ranges",
    ):
        monkeypatch.setattr(target_backend, name, upload(name))
    for operation, name in (
        ("cast", "create_zero_cta_cast_plan"),
        ("reduce", "create_zero_cta_reduce_plan"),
    ):

        def create(operation=operation, **kwargs):
            events.append((operation, kwargs))
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(target_backend, name, create)

    def runtime(group, capacity, nvl, bytes_per_token, slot):
        value = (group, capacity, nvl, bytes_per_token, slot)
        events.append(("runtime", value))
        return value

    monkeypatch.setattr(target_backend, "get_or_create_zero_cta_runtime", runtime)
    return events


@pytest.fixture
def prepared(monkeypatch):
    return capture(monkeypatch, meta, backend)


@pytest.mark.parametrize("world,nvl", [(1, 1), (2, 8), (2, 2), (4, 1), (4, 2), (6, 2), (8, 4)])
def test_routes_metadata_and_runtime_parameters(prepared, monkeypatch, world, nvl):
    monkeypatch.setenv("NVL_DOMAIN_SIZE", str(nvl))
    for seed in range(24):
        case = make_case(world, seed + 900)
        arguments = []
        for rank in range(world):
            prepared.clear()
            arg = meta.ZeroCTACollectiveArg(
                **kwargs_for(meta, case, rank, world, "cpu-fixture", slot=seed % 2)
            )
            arguments.append(arg)
            assert arg.runtime == ("cpu-fixture", case["capacity"], nvl, 4096, seed % 2)
            assert prepared[-1] == ("runtime", arg.runtime)
            kinds = [entry[0] for entry in prepared]
            assert kinds.count("cast") == kinds.count("reduce") == 1
            assert kinds.index("cast") < kinds.index("reduce") < kinds.index("runtime")
            assert arg.to_group_cast_args()["zero_cta_arg"] is arg
            assert arg.to_group_reduce_args()["zero_cta_arg"] is arg
            assert meta.GroupCollectiveArg.to_packed_group_cast_args(arg, 2) == dict(
                input_split_sizes=case["inputs"][rank] * 2,
                output_split_sizes=case["outputs"][rank] * 2,
                dst_indices=case["destinations"][rank] * 2,
                src_index=case["sources"][rank] * 2,
            )
            assert meta.GroupCollectiveArg.to_packed_group_reduce_args(arg, 2) == dict(
                input_split_sizes=case["outputs"][rank] * 2,
                output_split_sizes=case["inputs"][rank] * 2,
                dst_index=case["sources"][rank] * 2,
                src_indices=case["destinations"][rank] * 2,
            )
            sent = sum(
                n * len(peers) for n, peers in zip(case["inputs"][rank], case["destinations"][rank])
            )
            received = sum(case["outputs"][rank])
            for operation in ("max", "sum"):
                arg.compute_send_recv_token_counts(operation)
                expected = max(sent, received) if operation == "max" else sent + received
                assert arg.group_cast_comm_tokens == arg.group_reduce_comm_tokens == expected
        assert simulate(case, arguments, nvl) == [expected_cast(case, r) for r in range(world)]
        assert simulate(case, arguments, nvl, reduce=True) == [
            expected_reduce(case, r, 7) for r in range(world)
        ]


def simulate(case, args, nvl, reduce=False):
    """Execute host segments on CPU with bounds/uninitialized-read checks."""
    world = len(args)
    capacity = case["capacity"]
    hierarchical = world > nvl
    nodes = world // nvl if hierarchical else world
    plans = [arg.reduce_plan if reduce else arg.cast_plan for arg in args]
    raw = [[None] * (world * capacity) for _ in args]
    sent = [[None] * (nodes * capacity) for _ in args]
    received = [[None] * (nodes * capacity) for _ in args]
    payload = [
        [rank + 1] * sum(case["outputs"][rank])
        if reduce
        else [rank * 10000 + i for i in range(sum(case["inputs"][rank]))]
        for rank in range(world)
    ]

    def read(values, offset, count):
        assert 0 <= offset <= offset + count <= len(values)
        result = values[offset : offset + count]
        assert all(x is not None for x in result)
        return result

    def write(values, offset, data):
        assert 0 <= offset <= offset + len(data) <= len(values)
        values[offset : offset + len(data)] = data

    for rank, plan in enumerate(plans):
        target = sent[rank] if hierarchical and not reduce else raw[rank]
        for source, destination, count in plan.pack_gather_tiles:
            write(target, destination, read(payload[rank], source, count))
        for peer in getattr(plan, "local_ready_signal_peers", ()):
            assert rank in plans[peer].local_ready_wait_peers
        for peer in getattr(plan, "local_ready_wait_peers", ()):
            assert rank in plans[peer].local_ready_signal_peers
    if hierarchical and reduce:
        for rank, plan in enumerate(plans):
            for destination, first, count, tokens in plan.local_reduce_ranges:
                values = [0] * tokens
                for peer, source in plan.local_reduce_symmetric_srcs[first : first + count]:
                    assert peer // nvl == rank // nvl
                    values = [a + b for a, b in zip(values, read(raw[peer], source, tokens))]
                target = received[rank] if destination // capacity == rank // nvl else sent[rank]
                write(target, destination, values)
    for rank, plan in enumerate(plans):
        for peer, tokens in enumerate(plan.send_token_counts):
            assert 0 <= tokens <= capacity
            if not tokens:
                continue
            if peer != rank:
                assert rank in plans[peer].remote_wait_peers
            source = peer // nvl if hierarchical else peer
            destination = rank // nvl if hierarchical else rank
            data = sent[rank] if hierarchical else raw[rank]
            write(received[peer], destination * capacity, read(data, source * capacity, tokens))
    result = []
    for rank, plan in enumerate(plans):
        output = [7 if reduce else None] * sum(case["inputs" if reduce else "outputs"][rank])
        if reduce:
            for destination, first, count, tokens in plan.post_reduce_ranges:
                values = [7] * tokens
                for source in plan.post_reduce_src_token_offsets[first : first + count]:
                    values = [a + b for a, b in zip(values, read(received[rank], source, tokens))]
                write(output, destination, values)
        else:
            for tile in plan.post_gather_tiles:
                if hierarchical:
                    peer, source, destination, count = tile
                    assert peer // nvl == rank // nvl
                else:
                    source, destination, count = tile
                    peer = rank
                write(output, destination, read(received[peer], source, count))
        assert all(x is not None for x in output)
        result.append(output)
    return result


@pytest.mark.parametrize("world,nvl", [(1, 1), (2, 2), (4, 2), (6, 2), (8, 4)])
def test_compiled_segments_preserve_token_values_and_sum(prepared, monkeypatch, world, nvl):
    monkeypatch.setenv("NVL_DOMAIN_SIZE", str(nvl))
    for seed in range(12):
        case = make_case(world, seed + 300)
        args = [
            meta.ZeroCTACollectiveArg(**kwargs_for(meta, case, rank, world, "cpu-fixture"))
            for rank in range(world)
        ]
        assert simulate(case, args, nvl) == [expected_cast(case, r) for r in range(world)]
        assert simulate(case, args, nvl, reduce=True) == [
            expected_reduce(case, r, 7) for r in range(world)
        ]


def test_relay_only_rank_and_owner_offset_gaps(prepared, monkeypatch):
    monkeypatch.setenv("NVL_DOMAIN_SIZE", "2")
    case = relay_case(4, 2)
    args = [
        meta.ZeroCTACollectiveArg(**kwargs_for(meta, case, r, 4, "cpu-fixture")) for r in range(4)
    ]
    assert args[2].cast_plan.pack_gather_tiles == ()
    assert args[2].cast_plan.post_gather_tiles == ()
    assert args[2].cast_plan.remote_wait_peers == (0,)
    assert args[3].reduce_plan.pack_gather_tiles == ((0, 0, 60), (60, 80, 40))
    assert simulate(case, args, 2) == [expected_cast(case, r) for r in range(4)]
    assert simulate(case, args, 2, True) == [expected_reduce(case, r, 7) for r in range(4)]


@pytest.mark.parametrize(
    "kind,exception,message",
    [
        ("bad_nvl", ValueError, "NVL_DOMAIN_SIZE must divide"),
        ("missing_hier", ValueError, "requires hierarchy metadata"),
        ("capacity", ValueError, "exceeds its global token capacity"),
    ],
)
def test_constructor_rejections(prepared, monkeypatch, kind, exception, message):
    monkeypatch.setenv("NVL_DOMAIN_SIZE", "2")
    case = relay_case(4, 2)
    if kind == "bad_nvl":
        monkeypatch.setenv("NVL_DOMAIN_SIZE", "3")
    if kind == "capacity":
        monkeypatch.setenv("NVL_DOMAIN_SIZE", "4")
    kwargs = kwargs_for(meta, case, 0, 4, "cpu-fixture")
    if kind == "missing_hier":
        kwargs["hierarchy_meta"] = None
    if kind == "capacity":
        kwargs["max_per_peer_slot"] = 1
    with pytest.raises(exception, match=message):
        meta.ZeroCTACollectiveArg(**kwargs)
