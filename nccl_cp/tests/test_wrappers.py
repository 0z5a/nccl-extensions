from types import SimpleNamespace

import pytest
import torch
from nccl.cp import work as work_module
from nccl.cp import zero_cta as backend


def install_fake_run(monkeypatch):
    events = []

    class NativeWork:
        def wait(self):
            events.append("wait")

    def run(input, output, runtime, plan):
        events.append(("run", input, output, runtime, plan))
        return NativeWork()

    monkeypatch.setattr(backend, "_get_extension", lambda: SimpleNamespace(run=run))
    return events


@pytest.mark.parametrize("operation", ["cast", "reduce"])
@pytest.mark.parametrize("async_op", [False, True])
def test_work_and_output_identity(monkeypatch, operation, async_op):
    events = install_fake_run(monkeypatch)
    arg = SimpleNamespace(
        runtime="runtime",
        cast_plan="cast-plan",
        reduce_plan="reduce-plan",
        output_split_size_list=[2],
    )
    input = torch.ones((2, 4))
    output = torch.full_like(input, 7)
    fn = (
        backend.zero_cta_group_cast_impl
        if operation == "cast"
        else backend.zero_cta_group_reduce_impl
    )
    work = fn(input, output, arg, group=None, async_op=async_op)
    assert events[0][1] is input
    assert events[0][2] is output
    assert events[0][3:] == ("runtime", operation + "-plan")
    assert events.count("wait") == (0 if async_op else 1)
    assert work.wait_post_process(output) is output
    assert events.count("wait") == 1
    assert torch.equal(output, torch.full_like(output, 7))
    with pytest.raises(RuntimeError, match="already been done"):
        work.wait_post_process(output)
    assert work._work_done and work.work is None


def test_internal_cast_requires_output(monkeypatch):
    events = install_fake_run(monkeypatch)
    arg = SimpleNamespace(runtime="runtime", cast_plan="cast", output_split_size_list=[2, 3])
    with pytest.raises(ValueError, match="requires an output tensor"):
        backend.zero_cta_group_cast_impl(torch.ones((4, 2, 3)), None, arg, None)
    assert events == []


@pytest.mark.parametrize(
    "operation,kwargs,exception",
    [
        ("cast", {"cast_lse": True}, NotImplementedError),
        ("cast", {"input_lse": "lse"}, NotImplementedError),
        ("cast", {"output_lse": "lse"}, NotImplementedError),
        ("reduce", {"acc_reduce": False}, NotImplementedError),
        ("reduce", {"reduce_op": "avg"}, NotImplementedError),
        ("reduce", {"reduce_op": "lse"}, NotImplementedError),
        ("reduce", {"comm_dtype": torch.bfloat16}, NotImplementedError),
        ("reduce", {"input_lse": "lse"}, NotImplementedError),
        ("reduce", {"output_lse": "lse"}, NotImplementedError),
        ("reduce", {"output": None}, ValueError),
    ],
)
def test_rejections_without_launch(monkeypatch, operation, kwargs, exception):
    events = install_fake_run(monkeypatch)
    fn = (
        backend.zero_cta_group_cast_impl
        if operation == "cast"
        else backend.zero_cta_group_reduce_impl
    )
    arguments = dict(
        input=torch.ones(2), output=torch.zeros(2), collective_arg=object(), group=None
    )
    arguments.update(kwargs)
    with pytest.raises(exception):
        fn(**arguments)
    assert events == []


@pytest.mark.parametrize("async_op", [False, True])
def test_work_postprocess_order(async_op):
    events = []
    native = SimpleNamespace(wait=lambda: events.append("native-wait"))
    nested = work_module.GeneralWork(work_module.GeneralWork([native, None]))
    wrapped = work_module.WorkWithPostProcessFn(
        work=nested,
        post_process_fn=lambda value: events.append(("post", value)),
        async_op=async_op,
    )
    events.append("returned")
    wrapped.wait_post_process(9)
    assert events == (
        ["returned", "native-wait", ("post", 9)]
        if async_op
        else ["native-wait", "returned", ("post", 9)]
    )


def test_clear_without_extension_load(monkeypatch):
    def unexpected():
        raise AssertionError("must not load C++ while clearing an unused runtime")

    unexpected.cache_info = lambda: SimpleNamespace(currsize=0)
    monkeypatch.setattr(backend, "_get_extension", unexpected)
    backend.clear_zero_cta_cpp_state()
