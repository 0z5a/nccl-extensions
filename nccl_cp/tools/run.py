"""Run a script/module against a build-tree or installed NCCL CP package."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import runpy
import sys
import types
from pathlib import Path


def bind_package(package_directory: Path):
    directory = package_directory.resolve()
    entrypoint = directory / "__init__.py"
    if not entrypoint.is_file():
        raise RuntimeError(f"NCCL CP Python package is missing: {directory}. Run cmake --build first.")
    existing = sys.modules.get("nccl.cp")
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if not existing_path or Path(existing_path).resolve() != entrypoint:
            raise RuntimeError("A different nccl.cp package is already loaded")
        return existing
    try:
        parent = importlib.import_module("nccl")
    except ModuleNotFoundError as error:
        if error.name != "nccl":
            raise
        parent = types.ModuleType("nccl")
        parent.__path__ = []
        sys.modules["nccl"] = parent
    if not hasattr(parent, "__path__"):
        raise RuntimeError("The existing nccl module is not a package")
    spec = importlib.util.spec_from_file_location(
        "nccl.cp", entrypoint, submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nccl.cp"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["nccl.cp"]
        raise
    parent.cp = module
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "python/nccl/cp")
    parser.add_argument("--module", action="store_true", help="Run the target as a module")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("provide a script or --module followed by a module name")
    bind_package(args.package_dir)
    target, *arguments = args.command
    if args.module:
        sys.argv = [target, *arguments]
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        script = Path(target).resolve()
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), *arguments]
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
