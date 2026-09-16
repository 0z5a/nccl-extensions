"""Distributed entry points take caller configuration rather than host presets."""
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from nccl.cp.comm_meta import GroupCollectiveEntry
from route_cases import make_case, relay_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', [
    'tests/run_public.py', 'tests/run_padding.py', 'tests/run_async.py',
    'tests/run_distributed.py', 'examples/basic.py',
])
def test_missing_domain_fails_before_device_or_group_setup(monkeypatch, capsys, script):
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing configuration must not initialize devices or communication')
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    monkeypatch.setattr(torch.distributed, 'init_process_group', forbidden)
    monkeypatch.setenv('NVL_DOMAIN_SIZE', '9')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '9')
    monkeypatch.setattr(sys, 'argv', [script])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT/script), run_name='__main__')
    assert error.value.code == 2
    assert '--nvl-domain-size' in capsys.readouterr().err


@pytest.mark.parametrize('empty', [False, True])
def test_native_runner_passes_explicit_domain_to_plans(monkeypatch, empty):
    import run_distributed

    class Prepared(Exception):
        pass

    def argument(**kwargs):
        assert kwargs['nvl_domain_size'] == 3
        assert kwargs['world_size'] == 6
        raise Prepared

    monkeypatch.setenv('NVL_DOMAIN_SIZE', '9')
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    module = SimpleNamespace(ZeroCTACollectiveArg=argument, GroupCollectiveEntry=GroupCollectiveEntry)
    cases = [] if empty else [make_case(6, 11)]
    with pytest.raises(Prepared):
        run_distributed.run_collectives(module, None, object(), cases, 1024, 0, 6, 3,
                                        torch.float32, object())


@pytest.mark.parametrize('world,nvl', [(4, 2), (6, 3), (9, 3)])
def test_relay_fixture_follows_logical_domains(world, nvl):
    case = relay_case(world, nvl)
    assert all(len(case[name]) == world for name in ('inputs', 'outputs', 'sources', 'destinations'))
    assert case['inputs'][nvl] == case['outputs'][nvl] == []
    assert case['destinations'][0] == [[nvl + 1], [1], [nvl + 1]]
    assert case['outputs'][nvl + 1] == [60, 40]
