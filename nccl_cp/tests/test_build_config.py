"""Run dependency selection with synthetic toolchains, without compiling CUDA."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CMAKE = os.environ.get("NCCL_CP_CMAKE_COMMAND") or shutil.which("cmake")
pytestmark = pytest.mark.skipif(not CMAKE, reason="CMake is required for configuration checks")


def run_config(tmp_path, body, **environment):
    script = tmp_path / "check.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.24)\n'
        f'include("{ROOT}/cmake/Dependencies.cmake")\n' + body
    )
    env = dict(os.environ)
    for name in ("CUDAARCHS", "NCCL_HOME", "NCCL_CP_NCCL_HOME"):
        env.pop(name, None)
    env.update(environment)
    return subprocess.run([CMAKE, "-P", str(script)], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("codes,expected", [
    ("sm_70 sm_80 sm_90", "80;90"),
    ("sm_80 sm_90 sm_100 sm_100a sm_120", "80;90;100;120"),
    ("sm_90 sm_100a", "90"),
])
def test_default_architectures_from_compiler(tmp_path, codes, expected):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(f"#!/bin/sh\n[ \"$1\" = --list-gpu-code ] || exit 9\nprintf '%s\\n' '{codes}'\n")
    nvcc.chmod(0o755)
    result = run_config(tmp_path,
        f'nccl_cp_select_architectures("{nvcc}")\n'
        f'if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "{expected}")\n'
        '  message(FATAL_ERROR "wrong targets: ${CMAKE_CUDA_ARCHITECTURES}")\nendif()\n')
    assert result.returncode == 0, result.stderr


def test_explicit_architectures_override_environment_and_discovery(tmp_path):
    result = run_config(tmp_path,
        'set(CMAKE_CUDA_ARCHITECTURES "90-real;100-virtual")\n'
        'nccl_cp_select_architectures("/missing/nvcc")\n'
        'if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "90-real;100-virtual")\n'
        '  message(FATAL_ERROR "override lost")\nendif()\n', CUDAARCHS="80")
    assert result.returncode == 0, result.stderr


def test_environment_architectures_override_discovery(tmp_path):
    result = run_config(tmp_path,
        'nccl_cp_select_architectures("/missing/nvcc")\n'
        'if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "80;90")\n'
        '  message(FATAL_ERROR "override lost")\nendif()\n', CUDAARCHS="80;90")
    assert result.returncode == 0, result.stderr


def test_unusable_architecture_list_fails(tmp_path):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("#!/bin/sh\nprintf '%s\\n' sm_70\n")
    nvcc.chmod(0o755)
    result = run_config(tmp_path, f'nccl_cp_select_architectures("{nvcc}")\n')
    assert result.returncode != 0
    assert "none of the default CP architectures" in result.stderr


def nccl_root(path, libdir="lib"):
    (path / "include").mkdir(parents=True)
    (path / "include/nccl.h").write_text("// synthetic header\n")
    (path / libdir).mkdir()
    library = path / libdir / "libnccl.so.0.0.0"
    library.write_bytes(b"synthetic library")
    return library


@pytest.mark.parametrize("selection", ["vendor", "environment", "explicit"])
def test_nccl_selection_priority(tmp_path, selection):
    vendor = tmp_path / "third_party/nccl/build"
    env_root = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    libraries = {
        "vendor": nccl_root(vendor),
        "environment": nccl_root(env_root, "lib64"),
        "explicit": nccl_root(explicit),
    }
    prefix = f'set(NCCL_HOME "{explicit}" CACHE PATH "")\n' if selection == "explicit" else ""
    env = {"NCCL_HOME": str(env_root)} if selection != "vendor" else {}
    result = run_config(tmp_path, prefix
        + f'nccl_cp_find_nccl("{tmp_path}")\n'
        + f'if(NOT NCCL_CP_NCCL_LIBRARY STREQUAL "{libraries[selection]}")\n'
        + '  message(FATAL_ERROR "wrong NCCL library: ${NCCL_CP_NCCL_LIBRARY}")\nendif()\n', **env)
    assert result.returncode == 0, result.stderr


def test_changing_nccl_root_does_not_reuse_old_library(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    nccl_root(first)
    expected = nccl_root(second)
    result = run_config(tmp_path,
        f'set(NCCL_HOME "{first}" CACHE PATH "")\n'
        f'nccl_cp_find_nccl("{tmp_path}")\n'
        f'set(NCCL_HOME "{second}" CACHE PATH "" FORCE)\n'
        f'nccl_cp_find_nccl("{tmp_path}")\n'
        f'if(NOT NCCL_CP_NCCL_LIBRARY STREQUAL "{expected}")\n'
        '  message(FATAL_ERROR "cached library reused")\nendif()\n')
    assert result.returncode == 0, result.stderr


def test_missing_nccl_fails_with_preparation_hint(tmp_path):
    result = run_config(tmp_path, f'nccl_cp_find_nccl("{tmp_path}")\n')
    assert result.returncode != 0
    assert "make nccl-submodule" in result.stderr
