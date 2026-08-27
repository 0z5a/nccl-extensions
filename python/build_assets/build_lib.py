#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "packaging",
# ]
# ///

"""Build and stage selected nccl-extensions native artifacts.

Targets are selected as ``<library>[@<version>]``. A version selects that
target's conventional release tag; without a version, the current checkout is
built. Only explicitly selected targets are staged in the Python project.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from packaging.version import Version

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_SOURCE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PYTHON_SOURCE_ROOT.parent

logger = logging.getLogger(__name__)
_TARGET_VERSION_RE = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")
_NVCC_RELEASE_RE = re.compile(r"\brelease\s+(\d+)\.")


@dataclass(frozen=True)
class TargetSpec:
    """Build recipe for one independently versioned library."""

    tag_format: str
    version_header: str
    version_prefix: str
    library: str
    header_patterns: tuple[str, ...]
    package_relpath: str

    def tag_for(self, version: Version) -> str:
        return self.tag_format.format(version=version)


# This is the only registry to edit when another library is added.
TARGETS: dict[str, TargetSpec] = {
    "nccl_ep": TargetSpec(
        tag_format="nccl-ep-v{version}",
        version_header="nccl_ep/include/nccl_ep.h",
        version_prefix="NCCL_EP",
        library="libnccl_ep.so",
        header_patterns=(
            "nccl_ep.h",
            "nccl_ep/*.h",
            "nccl_ep/*.hpp",
            "nccl_ep/device/*.cuh",
        ),
        package_relpath="nccl/ep",
    ),
    "nccl_m2n": TargetSpec(
        tag_format="nccl-m2n-v{version}",
        version_header="nccl_m2n/src/nccl_m2n.h",
        version_prefix="NCCL_M2N",
        library="libnccl_m2n.so",
        header_patterns=("nccl_m2n.h",),
        package_relpath="nccl/m2n",
    ),
}


class Target:
    """One selected target and its resolved build state."""

    name: str
    spec: TargetSpec
    requested_version: Version | None
    resolved_version: Version
    build_dir: Path

    def __init__(self, selection: str, staging_dir: Path) -> None:
        """Create a target from a validated ``LIB[@VERSION]`` selector."""
        name, separator, version_text = selection.partition("@")
        self.name = name
        self.spec = TARGETS[name]
        self.requested_version = Version(version_text) if separator else None
        self.build_dir = staging_dir / f"{self.name}_build"

    def resolve_version(self, root: Path) -> None:
        """Resolve and validate this target's version from its source header."""
        header = root / self.spec.version_header
        text = header.read_text(encoding="utf-8")
        version_parts = []
        for part in ("MAJOR", "MINOR", "PATCH"):
            macro = f"{self.spec.version_prefix}_{part}"
            match = re.search(
                rf"^\s*#define\s+{re.escape(macro)}\s+(\d+)(?:\s|$)",
                text,
                re.MULTILINE,
            )
            if match is None:
                raise RuntimeError(f"No {macro} #define found in {header}")
            version_parts.append(match.group(1))
        header_version = Version(".".join(version_parts))
        if (
            self.requested_version is not None
            and header_version != self.requested_version
        ):
            raise RuntimeError(
                f"{self.tag} contains {self.name} version {header_version}, "
                f"expected {self.requested_version}"
            )
        self.resolved_version = (
            self.requested_version
            if self.requested_version is not None
            else header_version
        )

    def build(self, source_root: Path, nccl_home: Path, cuda_home: Path) -> None:
        """Build this target and verify its shared library."""
        logger.info(f">>> [{self.name}] building {self.spec.library}")
        subprocess.run(
            [
                "make",
                "-C",
                str(source_root / self.name),
                "build",
                f"BUILDDIR={self.build_dir}",
                f"NCCL_HOME={nccl_home}",
                f"CUDA_HOME={cuda_home}",
            ],
            check=True,
        )
        if not self.library.is_file():
            raise RuntimeError(
                f"Build succeeded but shared library was not found: {self.library}"
            )

    def installed_headers(self) -> tuple[Path, ...]:
        """Return owned headers relative to the build include directory."""
        headers = []
        for pattern in self.spec.header_patterns:
            matches = sorted(
                path.relative_to(self.include_dir)
                for path in self.include_dir.glob(pattern)
                if path.is_file()
            )
            if not matches:
                raise RuntimeError(
                    "Build succeeded but no installed headers matched "
                    f"{pattern!r} under {self.include_dir}"
                )
            headers.extend(matches)
        return tuple(dict.fromkeys(headers))

    def copy_artifacts(self, output_dir: Path, cuda_major: int) -> None:
        """Copy this target's owned artifacts into its Python package."""
        headers = self.installed_headers()
        package_dir = output_dir / self.spec.package_relpath
        library_destination = (
            package_dir / "lib" / f"cu{cuda_major}" / self.spec.library
        )
        library_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.library, library_destination, follow_symlinks=True)

        headers_destination = package_dir / "include"
        if headers_destination.is_symlink() or headers_destination.is_file():
            headers_destination.unlink()
        elif headers_destination.is_dir():
            shutil.rmtree(headers_destination)
        for header in headers:
            destination = headers_destination / header
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.include_dir / header, destination)
        logger.info(f">>> [{self.name}] staged {package_dir}")

    @property
    def include_dir(self) -> Path:
        return self.build_dir / "include"

    @property
    def library(self) -> Path:
        return self.build_dir / "lib" / self.spec.library

    @property
    def tag(self) -> str:
        if self.requested_version is None:
            raise RuntimeError(f"No version tag requested for {self.name}")
        return self.spec.tag_for(self.requested_version)


def _validate_selections(items: list[str]) -> None:
    """Validate all ``LIB[@VERSION]`` target selectors."""
    seen = set()
    for item in items:
        name, separator, version_text = item.partition("@")
        if name not in TARGETS:
            raise RuntimeError(f"Unknown library {name!r}; known: {', '.join(TARGETS)}")
        if name in seen:
            raise RuntimeError(f"Library specified more than once: {name}")
        seen.add(name)
        if not separator:
            continue
        if not version_text:
            raise RuntimeError(f"Missing version after '@' in {item!r}")
        if _TARGET_VERSION_RE.fullmatch(version_text) is None:
            raise RuntimeError(
                f"Invalid version in {item!r}: expected MAJOR.MINOR.PATCH "
                "without leading zeros"
            )
        tag = TARGETS[name].tag_for(Version(version_text))
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Tag not found: {tag}")


def _cuda_major(cuda_home: Path) -> int:
    """Return the CUDA Toolkit major reported by ``cuda_home/bin/nvcc``."""
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise RuntimeError(f"CUDA compiler is not executable: {nvcc}")
    result = subprocess.run(
        [str(nvcc), "--version"], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run {nvcc}: {result.stderr.strip()}")
    match = _NVCC_RELEASE_RE.search(result.stdout + result.stderr)
    if match is None:
        raise RuntimeError(f"Cannot determine CUDA version from {nvcc} --version")
    major = int(match.group(1))
    if major not in {12, 13}:
        raise RuntimeError(f"CUDA {major} is not supported; expected CUDA 12 or 13")
    return major


@contextmanager
def git_worktree(tag: str, path: Path) -> Generator[Path, None, None]:
    """Check ``tag`` out into a detached worktree at ``path``."""
    logger.info(f">>> Creating worktree for {tag} at {path}")
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), tag],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create worktree for {tag}: {e.stderr}") from e
    try:
        yield path
    finally:
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                f"Failed to remove temporary worktree {path}: {result.stderr}"
            )


def prepare_nccl(source_root: Path, build_root: Path, cuda_home: Path) -> None:
    """Build the NCCL submodule selected by ``source_root`` when necessary."""
    header = build_root / "include" / "nccl_device.h"
    library = build_root / "lib" / "libnccl.so"
    if header.is_file() and library.is_file():
        return

    logger.info(f">>> Building vendored NCCL at {build_root}")
    subprocess.run(
        [
            "make",
            "nccl-submodule",
            f"NCCL_SUBMODULE_HOME={source_root / 'third_party' / 'nccl'}",
            f"NCCL_BUILDDIR={build_root}",
            f"CUDA_HOME={cuda_home}",
        ],
        cwd=source_root,
        check=True,
    )
    if not header.is_file() or not library.is_file():
        raise RuntimeError(
            f"NCCL build succeeded but artifacts are missing under {build_root}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and stage selected nccl-extensions native artifacts"
    )
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="LIB[@VERSION]",
        help=(
            "Libraries to build, optionally from a release version, e.g. "
            "'nccl_ep@0.2.0 nccl_m2n@0.1.0'. A version selects the library's "
            "release tag; without one, use the current checkout. "
            f"Known libraries: {', '.join(TARGETS)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Python project root that receives staged package artifacts",
    )
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=None,
        help="Path to CUDA installation (default: $CUDA_PATH or $CUDA_HOME)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    try:
        _validate_selections(args.targets)
    except RuntimeError as e:
        parser.error(str(e))

    cuda_home_value = (
        args.cuda_home or os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    )
    if cuda_home_value is None:
        raise RuntimeError(
            "Provide --cuda-home or set CUDA_PATH or CUDA_HOME to a directory"
        )
    cuda_home = Path(cuda_home_value).expanduser().resolve()
    if not cuda_home.is_dir():
        raise RuntimeError(
            "Provide --cuda-home or set CUDA_PATH or CUDA_HOME to a directory"
        )
    cuda_major = _cuda_major(cuda_home)
    configured_nccl_home = os.environ.get("NCCL_HOME")
    external_nccl_home = (
        Path(configured_nccl_home).expanduser().resolve()
        if configured_nccl_home
        else None
    )
    if external_nccl_home is not None and not external_nccl_home.is_dir():
        raise RuntimeError(f"NCCL_HOME is not a directory: {external_nccl_home}")
    if external_nccl_home is not None:
        logger.info(f">>> Using NCCL_HOME={external_nccl_home} for all targets")

    output_dir = args.output_dir.expanduser().resolve()
    targets: list[Target] = []
    prepared_nccl_homes: dict[Path, Path] = {}
    with tempfile.TemporaryDirectory(prefix="nccl_extensions_build_") as tmp:
        staging_dir = Path(tmp)
        for selection in args.targets:
            target = Target(selection, staging_dir)
            if target.requested_version is None:
                logger.info(f">>> [{target.name}] using current checkout")
                source: AbstractContextManager[Path] = nullcontext(REPO_ROOT)
            else:
                logger.info(
                    f">>> [{target.name}] version "
                    f"{target.requested_version} -> {target.tag}"
                )
                source = git_worktree(target.tag, staging_dir / f"{target.name}_source")

            with source as source_root:
                source_root = source_root.resolve()
                target.resolve_version(source_root)
                if external_nccl_home is not None:
                    nccl_home = external_nccl_home
                elif source_root not in prepared_nccl_homes:
                    nccl_home = staging_dir / f"{target.name}_nccl"
                    prepare_nccl(source_root, nccl_home, cuda_home)
                    prepared_nccl_homes[source_root] = nccl_home
                else:
                    nccl_home = prepared_nccl_homes[source_root]
                target.build(source_root, nccl_home, cuda_home)
            target.copy_artifacts(output_dir, cuda_major)
            targets.append(target)

    logger.info(f"Artifacts location: {output_dir}")
    logger.info(
        "Built: "
        + ", ".join(f"{target.name}@{target.resolved_version}" for target in targets)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
