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
#   "cybind==0.3.1.dev2125+gfd8ec2066",
#   "packaging",
#   "pyyaml",
# ]
#
# [[tool.uv.index]]
# name = "cybind"
# url = "https://gitlab-master.nvidia.com/api/v4/projects/xiakunl%2Fcybind/packages/pypi/simple"
# explicit = true
# authenticate = "never"
#
# [tool.uv.sources]
# cybind = { index = "cybind" }
# ///

"""Generate selected nccl-extensions Cython bindings using cybind.

Targets are selected as ``<library>[@<version>]``. A version selects that
target's conventional release tag; without a version, headers are read from
the current checkout. Only explicitly selected targets are generated.
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
from typing import Any
from unittest.mock import patch

import cybind.__main__ as cybind_cli
import yaml
from packaging.version import Version

# Repository layout configuration. These are the main values to adjust when
# reusing this driver in another repository or Python package.
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_SOURCE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PYTHON_SOURCE_ROOT.parent
CYBIND_ASSETS_DIR = SCRIPT_DIR / "cybind"
NCCL_SOURCE_RELPATH = Path("third_party/nccl")
NCCL_HEADER_RELPATH = Path("build/include/nccl.h")

# Headers shared by all targets but owned outside their source directories.
SHARED_EXTERNAL_HEADERS = (
    (
        "nccl.h",
        str(NCCL_SOURCE_RELPATH / NCCL_HEADER_RELPATH),
    ),
)

# Shared files relative to each target's generated bindings package. Cybind does
# not emit them, so they are installed whenever that package is generated.
STATIC_TEMPLATE_FILES = (
    Path("__init__.py"),
    Path("_internal/__init__.py"),
    Path("_internal/utils.pxd"),
    Path("_internal/utils.pyx"),
)

logger = logging.getLogger(__name__)
_CONFIG_VERSION_RE = re.compile(r"(versions:\n\s*- - )\S+")
_TARGET_VERSION_RE = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")


class ConfigError(RuntimeError):
    """A cybind config has invalid syntax or an unsupported structure."""


@dataclass(frozen=True)
class TargetSpec:
    """Hand-written recipe for one independently generated library."""

    tag_format: str
    version_header: str
    version_prefix: str
    # (path in cybind's per-target header asset, path in the source repo)
    headers: tuple[tuple[str, str], ...]
    # (path in the per-target asset, source owned outside this target)
    external_headers: tuple[tuple[str, str], ...] = SHARED_EXTERNAL_HEADERS

    def tag_for(self, version: Version) -> str:
        return self.tag_format.format(version=version)


# This is the only target registry to edit when adding another library.
TARGETS: dict[str, TargetSpec] = {
    "nccl_ep": TargetSpec(
        tag_format="nccl-ep-v{version}",
        version_header="nccl_ep/include/nccl_ep.h",
        version_prefix="NCCL_EP",
        headers=(
            ("nccl_ep.h", "nccl_ep/include/nccl_ep.h"),
            ("nccl_ep/ep_enums.h", "nccl_ep/include/ep_enums.h"),
        ),
    ),
    "nccl_m2n": TargetSpec(
        tag_format="nccl-m2n-v{version}",
        version_header="nccl_m2n/src/nccl_m2n.h",
        version_prefix="NCCL_M2N",
        headers=(("nccl_m2n.h", "nccl_m2n/src/nccl_m2n.h"),),
    ),
}


class Target:
    """One selected target and its resolved generation state."""

    name: str
    spec: TargetSpec
    requested_version: Version | None
    staging_assets: Path
    resolved_version: Version
    bindings_relpath: Path

    def __init__(self, selection: str, staging_assets: Path) -> None:
        """Create a target from a validated ``LIB[@VERSION]`` selector."""
        name, separator, version_text = selection.partition("@")
        self.name = name
        self.spec = TARGETS[name]
        self.requested_version = Version(version_text) if separator else None
        self.staging_assets = staging_assets.resolve()

        try:
            config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {self.config_path}: {e}") from e
        if not isinstance(config, dict):
            raise ConfigError(f"Expected a mapping in {self.config_path}")
        self._resolve_config(config)

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

    def _resolve_config(self, config: dict[str, Any]) -> None:
        """Resolve generated module paths from this target's cybind config."""
        try:
            module = config[self.name]["module"]
        except (KeyError, TypeError) as e:
            raise ConfigError(f"No module configured for {self.name!r}") from e
        if not isinstance(module, str):
            raise ConfigError(f"Module for {self.name!r} must be a dotted string")
        module_parts = module.split(".")
        if len(module_parts) < 2 or any(not part for part in module_parts):
            raise ConfigError(f"Invalid module for {self.name!r}: {module!r}")
        self.bindings_relpath = Path(*module_parts[:-1])

    def update_config(self) -> None:
        """Update this target's staged config version."""
        text = self.config_path.read_text(encoding="utf-8")
        updated, count = _CONFIG_VERSION_RE.subn(rf"\g<1>{self.resolved_version}", text)
        if count != 1:
            raise ConfigError(
                f"Expected exactly one `versions:` block in {self.config_path}, "
                f"found {count}"
            )
        self.config_path.write_text(updated, encoding="utf-8")

    def copy_headers(self, root: Path) -> None:
        """Copy this target's complete header tree into staged assets."""
        if self.headers_dir.exists():
            shutil.rmtree(self.headers_dir)
        headers = (*self.spec.headers, *self.spec.external_headers)
        for relpath, source_relpath in headers:
            source = root / source_relpath
            destination = self.headers_dir / relpath
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    @property
    def config_path(self) -> Path:
        return self.staging_assets / "configs" / f"{self.name}.cybind.yaml"

    @property
    def headers_dir(self) -> Path:
        return self.staging_assets / "headers" / self.name / str(self.resolved_version)

    @property
    def templates_dir(self) -> Path:
        return self.staging_assets / "templates"

    @property
    def tag(self) -> str:
        if self.requested_version is None:
            raise RuntimeError(f"No version tag requested for {self.name}")
        return self.spec.tag_for(self.requested_version)


def _validate_selections(
    items: list[str],
) -> None:
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


def prepare_nccl_headers(root: Path) -> None:
    """Initialize NCCL when needed and generate its public headers."""
    nccl_root = root / NCCL_SOURCE_RELPATH
    if not (nccl_root / "Makefile").is_file():
        subprocess.run(
            ["git", "submodule", "update", "--init", str(NCCL_SOURCE_RELPATH)],
            cwd=root,
            check=True,
        )
    build_dir = nccl_root / "build"
    subprocess.run(
        [
            "make",
            "-C",
            "src",
            str(nccl_root / NCCL_HEADER_RELPATH),
            f"BUILDDIR={build_dir}",
        ],
        cwd=nccl_root,
        check=True,
    )


def run_cybind(targets: list[Target], output_dir: Path) -> None:
    """Run cybind's Python entry point once for all selected libraries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--generate",
        *(target.name for target in targets),
        "--output-dir",
        str(output_dir),
    ]
    logger.debug(f"cybind arguments: {' '.join(args)}")
    with patch.object(
        cybind_cli,
        "_get_assets_dir",
        return_value=str(targets[0].staging_assets),
    ):
        result = cybind_cli.main(args)
    if result != 0:
        raise RuntimeError(f"cybind exited with status {result}")


def generate_bindings(
    targets: list[Target],
    staging_dir: Path,
) -> None:
    """Generate and install the resolved targets."""
    for target in targets:
        target.update_config()

    output_dir = staging_dir / "generated"
    run_cybind(targets, output_dir)
    static_templates = {
        target.bindings_relpath / filename: (
            target.templates_dir / target.bindings_relpath / filename
        )
        for target in targets
        for filename in STATIC_TEMPLATE_FILES
    }
    for relpath, source in static_templates.items():
        destination = output_dir / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    shutil.copytree(output_dir, PYTHON_SOURCE_ROOT, dirs_exist_ok=True)
    for target in targets:
        destination = CYBIND_ASSETS_DIR / "configs" / target.config_path.name
        shutil.copy2(target.config_path, destination)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate selected nccl-extensions Cython bindings using cybind"
    )
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="LIB[@VERSION]",
        help=(
            "Libraries to generate, optionally from a release version, e.g. "
            "'nccl_ep@0.2.0 nccl_m2n@0.1.0'. A version selects the library's "
            "release tag; without one, use the current checkout. "
            f"Known libraries: {', '.join(TARGETS)}."
        ),
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
    os.environ["CUDA_PATH"] = str(cuda_home)
    os.environ["CUDA_HOME"] = str(cuda_home)

    targets: list[Target] = []
    prepared_source_roots: set[Path] = set()
    with tempfile.TemporaryDirectory(prefix="nccl_extensions_generate_") as tmp:
        staging_dir = Path(tmp)
        staging_assets = staging_dir / "assets"
        shutil.copytree(CYBIND_ASSETS_DIR, staging_assets)

        for selection in args.targets:
            target = Target(selection, staging_assets)
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
                if source_root not in prepared_source_roots:
                    prepare_nccl_headers(source_root)
                    prepared_source_roots.add(source_root)
                target.resolve_version(source_root)
                target.copy_headers(source_root)
            targets.append(target)

        generate_bindings(targets, staging_dir)

    logger.info(f"Bindings location: {PYTHON_SOURCE_ROOT}")
    logger.info(
        "Generated: "
        + ", ".join(f"{target.name}@{target.resolved_version}" for target in targets)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
