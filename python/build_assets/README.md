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

## Building shared libraries

`build_lib.py` accepts one or more `LIB[@VERSION]` selectors and stages only
the selected libraries and their installed headers into a Python project:

```bash
uv run build_assets/build_lib.py \
  nccl_ep@0.2.0 nccl_m2n@0.1.0 \
  --output-dir /path/to/python-project
```

An unversioned target uses the current checkout; a versioned target uses its
conventional release tag. When `NCCL_HOME` is set, every selected target builds
against that NCCL installation. Otherwise, each source checkout builds and uses
its own vendored NCCL revision in a temporary directory. Each target is staged
after its build succeeds.

The installed build layout is preserved below each target's package:

```text
build/lib/libnccl_ep.so   -> nccl/ep/lib/cu<major>/libnccl_ep.so
build/include/**          -> nccl/ep/include/**
build/lib/libnccl_m2n.so  -> nccl/m2n/lib/cu<major>/libnccl_m2n.so
build/include/nccl_m2n.h  -> nccl/m2n/include/nccl_m2n.h
```

`build_lib.py` obtains `<major>` from the Toolkit selected by `--cuda-home`
(or `CUDA_PATH` / `CUDA_HOME`) and currently accepts CUDA 12 and CUDA 13.

## Building wheels

`build_wheels.sh` builds CUDA 12 and CUDA 13 variants of both current-checkout
libraries by default. Optional selectors choose each library's source version
and are forwarded to `build_lib.py`; both libraries must be selected:

```bash
CUDA12_HOME=/usr/local/cuda-12.8 CUDA13_HOME=/usr/local/cuda-13.0 \
  ./build_assets/build_wheels.sh
CUDA12_HOME=/usr/local/cuda-12.8 CUDA13_HOME=/usr/local/cuda-13.0 \
  ./build_assets/build_wheels.sh \
  nccl_ep@0.2.0 nccl_m2n@0.1.0
```

The script first builds a source-only sdist and passes it directly to
cibuildwheel, preventing ignored host build artifacts from entering a wheel.
The native libraries are built once with each Toolkit inside cibuildwheel's
manylinux container and staged under each facade's `lib/cu12` and `lib/cu13`
directories before the Python extensions are compiled. Headers from both builds
must match, except for EP's generated `config.h`, and are stored once. Native
build output is not written into the source checkout.
Wheels and the sdist are written to `<repo>/build/dist`; set `BUILDDIR` to choose
another output root. Docker and `uv` must be available on the host.
Production wheel builds set `NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1`, so every
packaged native library must have been selected and staged. Direct developer
builds default this variable to `0` and warn instead, allowing an external
library to be supplied at runtime.

## Adding a library

Add its target definition to `TARGETS` in `generate_cython.py`, then add the
matching cybind config and any required templates under `cybind/`.

After changing a bound header, config, or template, regenerate the affected
targets and review the generated diff.
