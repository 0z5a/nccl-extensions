from nccl.cp.routing import Route


def test_route_rows_are_computed_once_and_do_not_affect_route_equality():
    calls = []

    class Counted(tuple):
        def __iter__(self):
            calls.append(1)
            return super().__iter__()

    route = Route(Counted((1, 3)), Counted((2, 2)), ((0,), (0,)), (0, 0))
    assert len(calls) == 2
    for _ in range(100):
        assert route.input_rows == route.output_rows == 4
    assert len(calls) == 2
    assert route == Route((1, 3), (2, 2), ((0,), (0,)), (0, 0))
