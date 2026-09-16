"""Software identity and manifest support for the native library."""

from __future__ import annotations

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

import torch
import torch.distributed as dist

IDENTITY_FIELDS = (
    "torch_version", "torch_cuda_version", "python_soabi", "cxx11_abi", "nccl_version",
)


def runtime_identity():
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
        "nccl_version": list(torch.cuda.nccl.version()) if dist.is_nccl_available() else None,
    }


def describe():
    root = Path(torch.__file__).resolve().parent
    return {
        **runtime_identity(),
        "torch_root": str(root),
        "nccl_available": dist.is_nccl_available(),
        "pybind_compiler_type": getattr(torch._C, "_PYBIND11_COMPILER_TYPE", None),
        "pybind_stdlib": getattr(torch._C, "_PYBIND11_STDLIB", None),
        "pybind_build_abi": getattr(torch._C, "_PYBIND11_BUILD_ABI", None),
    }


def manifest_path(library: Path):
    return library.with_suffix(".build.json")


def validate_manifest(library: Path):
    """Check the native artifact against the active Python/PyTorch identity.

    Deployment callers still supply a compatible driver and dynamic-library
    environment. This guard does not install dependencies or prove GPU operation.
    """
    path = manifest_path(library)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read NCCL CP build manifest: {path}") from error
    if not isinstance(record, dict) or record.get("format_version") != 1:
        raise RuntimeError(f"Unsupported NCCL CP build manifest: {path}")
    if record.get("library") != library.name:
        raise RuntimeError("NCCL CP build manifest names a different library")
    if hashlib.sha256(library.read_bytes()).hexdigest() != record.get("sha256"):
        raise RuntimeError("NCCL CP native library checksum does not match its build manifest")
    built = record.get("runtime")
    current = runtime_identity()
    if not isinstance(built, dict):
        raise RuntimeError("NCCL CP build manifest is missing runtime identity")
    mismatches = [key for key in IDENTITY_FIELDS if key not in built or built[key] != current[key]]
    if mismatches:
        raise RuntimeError(
            "NCCL CP native library was built for a different runtime: "
            + ", ".join(mismatches)
            + ". Rebuild with the Python/PyTorch environment used for execution."
        )
    return record


def write_manifest(library: Path, cuda_toolkit: str, nccl_header_version: str, architectures: str):
    library = library.resolve(strict=True)
    record = {
        "format_version": 1,
        "library": library.name,
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "runtime": runtime_identity(),
        "cuda_toolkit": cuda_toolkit,
        "nccl_header_version": nccl_header_version,
        "cuda_architectures": architectures.split(";"),
    }
    destination = manifest_path(library)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cuda-toolkit", default="")
    parser.add_argument("--nccl-header-version", default="")
    parser.add_argument("--architectures", default="")
    args = parser.parse_args()
    if args.library is None:
        print(json.dumps(describe()))
    else:
        write_manifest(args.library, args.cuda_toolkit, args.nccl_header_version, args.architectures)
