"""Exercise the no-install launcher in fresh Python processes."""

import os
import subprocess
import sys
from pathlib import Path

import nccl.cp
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/run.py"
PACKAGE = Path(nccl.cp.__file__).resolve().parent


def run_launcher(tmp_path, *args, env=None):
    return subprocess.run(
        [sys.executable, str(LAUNCHER), "--package-dir", str(PACKAGE), *args],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )


def test_script_arguments_and_local_imports(tmp_path):
    (tmp_path / "helper.py").write_text("value = 42\n")
    script = tmp_path / "script.py"
    script.write_text(
        "import nccl.cp, helper, sys\n"
        "assert helper.value == 42\n"
        "assert sys.argv[1:] == ['--value', 'two words']\n"
        "print(nccl.cp.__file__)\n"
    )
    result = run_launcher(tmp_path, str(script), "--value", "two words")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == PACKAGE / "__init__.py"


def test_module_forwards_leading_option(tmp_path):
    result = run_launcher(tmp_path, "--module", "pytest", "--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("pytest ")


def test_preserves_existing_nccl_parent(tmp_path):
    parent = tmp_path / "nccl"
    parent.mkdir()
    (parent / "__init__.py").write_text("sentinel = 'existing parent'\n")
    (parent / "ep.py").write_text("sentinel = 'existing sibling'\n")
    script = tmp_path / "script.py"
    script.write_text(
        "import nccl, nccl.ep, nccl.cp\n"
        "assert nccl.sentinel == 'existing parent'\n"
        "assert nccl.ep.sentinel == 'existing sibling'\n"
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = run_launcher(tmp_path, str(script), env=env)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("broken_parent", [False, True])
def test_rejects_conflicting_modules(tmp_path, broken_parent):
    script = tmp_path / "conflict.py"
    script.write_text(
        "import runpy, sys, types\n"
        f"bind = runpy.run_path({str(LAUNCHER)!r})['bind_package']\n"
        "from pathlib import Path\n"
        + ("sys.modules['nccl'] = types.ModuleType('nccl')\n" if broken_parent else
           "sys.modules['nccl.cp'] = types.ModuleType('nccl.cp')\n")
        + f"bind(Path({str(PACKAGE)!r}))\n"
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert ("not a package" if broken_parent else "already loaded") in result.stderr
