"""Check the bootstrap using a local miniature submodule and a CPU Make target."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make")
pytestmark = pytest.mark.skipif(not MAKE or not shutil.which("git"), reason="Git and Make required")


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), "-c", "protocol.file.allow=always",
         "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    source = tmp_path / "dependency"
    source.mkdir()
    git(source, "init")
    (source / "makefiles").mkdir()
    (source / "makefiles/version.mk").write_text("NCCL_MAJOR := 2\n")
    (source / ".gitignore").write_text("build/\n")
    (source / "Makefile").write_text(
        ".PHONY: src.build\nsrc.build:\n"
        "\t@mkdir -p \"$(BUILDDIR)\"\n"
        "\t@printf '%s\\n' built >> \"$(BUILDDIR)/runs\"\n"
    )
    git(source, "add", ".")
    git(source, "commit", "-m", "Fixture dependency")
    project = tmp_path / "project"
    project.mkdir()
    git(project, "init")
    git(project, "submodule", "add", str(source), "third_party/nccl")
    return project


def bootstrap(project):
    return subprocess.run(
        ["sh", str(ROOT / "tools/build_nccl.sh"), str(project), MAKE, "2"],
        env=dict(os.environ, GIT_ALLOW_PROTOCOL="file"),
        capture_output=True, text=True,
    )


def test_initialize_and_repeat_build_without_removing_outputs(checkout):
    git(checkout, "submodule", "deinit", "-f", "third_party/nccl")
    assert not (checkout / "third_party/nccl/.git").exists()
    result = bootstrap(checkout)
    assert result.returncode == 0, result.stderr
    output = checkout / "third_party/nccl/build"
    (output / "keep").write_text("existing output")
    result = bootstrap(checkout)
    assert result.returncode == 0, result.stderr
    assert (output / "runs").read_text().splitlines() == ["built", "built"]
    assert (output / "keep").read_text() == "existing output"


@pytest.mark.parametrize("kind", ["tracked", "untracked", "staged", "revision"])
def test_local_changes_are_preserved(checkout, kind):
    dependency = checkout / "third_party/nccl"
    if kind == "revision":
        git(dependency, "commit", "--allow-empty", "-m", "Local revision")
    else:
        path = dependency / ("local.txt" if kind == "untracked" else "makefiles/version.mk")
        path.write_text("local change")
        if kind == "staged":
            git(dependency, "add", "makefiles/version.mk")
    head = git(dependency, "rev-parse", "HEAD")
    status = git(dependency, "status", "--porcelain")
    result = bootstrap(checkout)
    assert result.returncode != 0
    assert ("another revision" if kind == "revision" else "local changes") in result.stderr
    assert git(dependency, "rev-parse", "HEAD") == head
    assert git(dependency, "status", "--porcelain") == status
    assert not (dependency / "build/runs").exists()
