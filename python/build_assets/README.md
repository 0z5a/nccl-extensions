# Build Assets for the nccl-extensions Python package

> **⚠️ Internal Use Only**: This directory is **not released** to the public
> nccl-extensions GitHub repository.

## Purpose

The Cython bindings under `python/nccl/_extensions/bindings/` are **generated**,
not hand-written. This directory holds the tooling that generates them: the
cybind config, the Cython templates, and the driver script. The M2N low-level
surface is generated into the shared `nccl._extensions.bindings` package while
its public facade remains under `python/nccl/m2n/`.

The generated `.pyx`/`.pxd` files are checked into the repository so users can
build the package without internal tooling; the generation tooling here stays
internal.

## Why not public?

Generation relies on [cybind](https://gitlab-master.nvidia.com/xiakunl/cybind), an
internal NVIDIA tool. To keep the package buildable without access to it, we:

1. Run `generate_cython.py` when a bound library's headers change
2. Commit the generated Cython sources
3. Exclude this `build_assets/` directory from public releases

Ported from nccl4py's `bindings/nccl4py/build_assets/`, trimmed to the targets
this repo owns. nccl4py's `generate_header.py` (which flattens the NCCL *device*
API headers) has no nccl-extensions equivalent and was not ported.

## Prerequisites

- CUDA installation, with `CUDA_HOME` or `CUDA_PATH` set (cybind needs `cuda.h`)
- [`uv`](https://docs.astral.sh/uv/) on `PATH`

The script declares its Python dependencies using PEP 723, so no separate
Python environment setup is required.

## Usage

Select exactly the libraries whose bindings should be regenerated. Add
`@VERSION` to generate from that library's release tag, or omit it to use the
current checkout:

```bash
uv run python/build_assets/generate_cython.py nccl_ep@0.2.0 nccl_m2n@0.1.0
uv run python/build_assets/generate_cython.py nccl_ep
```

For a versioned target, the conventional tag rule maps the version to a tag
(`nccl_ep@0.2.0` becomes `nccl-ep-v0.2.0`). The script creates a detached
worktree for that tag; an unversioned target uses the current checkout. Only
the selected targets are regenerated. Output is written under
`python/nccl/_extensions/bindings/`.

See `generate_cython.py --help` for all options.

## Adding a library

Add its target definition to `TARGETS` in `generate_cython.py`, then add the
matching cybind config and any required templates under `cybind/`.

After changing a bound header, config, or template, regenerate the affected
targets and review the generated diff.
