"""Prebuilt loading checks; native loading is mocked on CPU hosts."""

import json
from pathlib import Path

import pytest
import torch
from nccl.cp import _build_info, zero_cta


@pytest.fixture
def library(tmp_path, monkeypatch):
    path = tmp_path / "libnccl_cp.so"
    path.write_bytes(b"test artifact: not a native library")
    _build_info.write_manifest(path, "13.0", "2.30.0", "90;100")
    monkeypatch.setenv("NCCL_CP_LIBRARY", str(path))
    zero_cta._get_extension.cache_clear()
    yield path
    zero_cta._get_extension.cache_clear()


def test_explicit_prebuilt_load_is_cached(library, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.ops, "load_library", calls.append)
    first = zero_cta._get_extension()
    assert zero_cta._get_extension() is first
    assert calls == [str(library.resolve())]
    record = _build_info.validate_manifest(library)
    assert record["cuda_architectures"] == ["90", "100"]


def test_bundled_library_is_default(library, monkeypatch):
    monkeypatch.delenv("NCCL_CP_LIBRARY")
    monkeypatch.setattr(zero_cta, "__file__", str(library.with_name("zero_cta.py")))
    calls = []
    monkeypatch.setattr(torch.ops, "load_library", calls.append)
    zero_cta._get_extension()
    assert calls == [str(library.resolve())]


@pytest.mark.parametrize("failure", ["missing", "json", "version", "name", "checksum", "identity", "runtime"])
def test_invalid_artifact_fails_before_load(library, monkeypatch, failure):
    manifest = _build_info.manifest_path(library)
    record = json.loads(manifest.read_text())
    if failure == "missing":
        manifest.unlink()
    elif failure == "json":
        manifest.write_text("{")
    else:
        if failure == "version":
            record["format_version"] = 99
        elif failure == "name":
            record["library"] = "different.so"
        elif failure == "checksum":
            library.write_bytes(b"changed artifact")
        elif failure == "identity":
            record["runtime"]["torch_version"] = "incompatible build"
        elif failure == "runtime":
            del record["runtime"]
        manifest.write_text(json.dumps(record))
    calls = []
    monkeypatch.setattr(torch.ops, "load_library", calls.append)
    with pytest.raises(RuntimeError):
        zero_cta._get_extension()
    assert not calls
    assert zero_cta._get_extension.cache_info().currsize == 0


def test_missing_explicit_library_does_not_use_bundled_copy(library, monkeypatch):
    monkeypatch.setattr(zero_cta, "__file__", str(library.with_name("zero_cta.py")))
    monkeypatch.setenv("NCCL_CP_LIBRARY", str(library.with_name("missing.so")))
    with pytest.raises(RuntimeError, match="Build with CMake"):
        zero_cta._get_extension()


def test_load_failure_can_be_retried(library, monkeypatch):
    def fail(_path):
        raise OSError("unresolved native symbol")

    monkeypatch.setattr(torch.ops, "load_library", fail)
    with pytest.raises(OSError, match="unresolved native symbol"):
        zero_cta._get_extension()
    assert zero_cta._get_extension.cache_info().currsize == 0
    calls = []
    monkeypatch.setattr(torch.ops, "load_library", calls.append)
    zero_cta._get_extension()
    assert calls == [str(library.resolve())]


def test_runtime_sources_do_not_invoke_build_helpers():
    root = Path(zero_cta.__file__).parent
    for source in root.glob("*.py"):
        text = source.read_text()
        assert "torch.utils.cpp_extension" not in text
        assert "subprocess" not in text
