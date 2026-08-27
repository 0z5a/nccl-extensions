#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Build wheels for nccl-extensions
# This script is intended for CI use with uv and cibuildwheel
#
# Usage: build_wheels.sh [LIB[@VERSION] ...]
#
# Targets are passed through to build_lib.py, which stages each library's
# shared object and public headers into the package before the wheels are
# built. A version selects that library's release tag; without one the current
# checkout is used. Defaults to every library the wheel packages.
#
# Requires: CUDA12_HOME and CUDA13_HOME environment variables set

set -euo pipefail

# Libraries the wheel packages; also the default set to build
DEFAULT_TARGETS=(nccl_ep nccl_m2n)

if [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

# A production wheel always carries both libraries. Selectors may choose each
# library's version independently, but neither library may be omitted.
for required_target in "${DEFAULT_TARGETS[@]}"; do
    count=0
    for selection in "${TARGETS[@]}"; do
        if [ "${selection%%@*}" = "$required_target" ]; then
            count=$((count + 1))
        fi
    done
    if [ "$count" -ne 1 ]; then
        echo "Error: specify $required_target exactly once" >&2
        exit 1
    fi
done

# Quote each selector for cibuildwheel's shell command. build_lib.py remains
# the single source of truth for selector validation.
TARGET_ARGUMENTS=()
for target in "${TARGETS[@]}"; do
    printf -v quoted_target '%q' "$target"
    TARGET_ARGUMENTS+=("$quoted_target")
done
TARGET_COMMAND="${TARGET_ARGUMENTS[*]}"

resolve_cuda_home() {
    local variable=$1
    local value=${!variable:-}
    if [ -z "$value" ]; then
        echo "Error: $variable is not set" >&2
        return 1
    fi
    if [ "$value" = "~" ]; then
        value="$HOME"
    elif [[ "$value" = "~/"* ]]; then
        value="$HOME/${value:2}"
    fi
    if ! value="$(cd "$value" 2>/dev/null && pwd -P)"; then
        echo "Error: $variable is not a directory: $value" >&2
        return 1
    fi
    if [ ! -x "$value/bin/nvcc" ]; then
        echo "Error: $variable/bin/nvcc is not executable" >&2
        return 1
    fi
    printf '%s\n' "$value"
}

validate_cuda_major() {
    local cuda_home=$1
    local expected_major=$2
    local nvcc_version
    if ! nvcc_version="$("$cuda_home/bin/nvcc" --version 2>&1)"; then
        echo "Error: failed to run $cuda_home/bin/nvcc" >&2
        return 1
    fi
    if ! grep -Eq "release ${expected_major}\\." <<<"$nvcc_version"; then
        echo "Error: expected CUDA $expected_major at $cuda_home" >&2
        echo "$nvcc_version" >&2
        return 1
    fi
}

CUDA12_HOME="$(resolve_cuda_home CUDA12_HOME)"
CUDA13_HOME="$(resolve_cuda_home CUDA13_HOME)"
validate_cuda_major "$CUDA12_HOME" 12
validate_cuda_major "$CUDA13_HOME" 13
export CUDA12_HOME CUDA13_HOME

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install from: https://docs.astral.sh/uv/" >&2
    exit 1
fi

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PYTHON_DIR/.." && pwd)"
BUILDDIR="${BUILDDIR:-$REPO_ROOT/build}"
if [ "$BUILDDIR" = "~" ]; then
    BUILDDIR="$HOME"
elif [[ "$BUILDDIR" = "~/"* ]]; then
    BUILDDIR="$HOME/${BUILDDIR:2}"
fi
mkdir -p "$BUILDDIR"
BUILDDIR="$(cd "$BUILDDIR" && pwd -P)"
DIST_DIR="$BUILDDIR/dist"

mkdir -p "$DIST_DIR"
SDIST_DIR="$(mktemp -d)"
trap 'rm -rf "$SDIST_DIR"' EXIT

echo "========================================="
echo "Building nccl-extensions wheels"
echo "========================================="
echo "CUDA12_HOME: $CUDA12_HOME"
echo "CUDA13_HOME: $CUDA13_HOME"
echo "Targets: ${TARGETS[*]}"
echo "Output directory: $DIST_DIR"
echo ""

cd "$PYTHON_DIR"

# Build through the source distribution so ignored host artifacts do not enter
# cibuildwheel's package tree.
echo ">>> Building source distribution"
uv build --sdist --out-dir "$SDIST_DIR" .

SDISTS=("$SDIST_DIR"/*.tar.gz)
if [ ${#SDISTS[@]} -ne 1 ] || [ ! -f "${SDISTS[0]}" ]; then
    echo "Error: expected exactly one source distribution under $SDIST_DIR" >&2
    exit 1
fi
SDIST="${SDISTS[0]}"
echo ">>> Source distribution: $SDIST"

# Build wheels using cibuildwheel. CUDA's libcuda stub satisfies the native
# link; the real driver library is supplied at runtime. Runtime libraries are
# excluded from auditwheel repair via pyproject.toml's repair-wheel-command.
printf -v QUOTED_CUDA12_HOME '%q' "$CUDA12_HOME"
printf -v QUOTED_CUDA13_HOME '%q' "$CUDA13_HOME"
export CIBW_ENVIRONMENT="CUDA_HOME=$QUOTED_CUDA12_HOME CUDA12_HOME=$QUOTED_CUDA12_HOME CUDA13_HOME=$QUOTED_CUDA13_HOME NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1"

# The repository is mounted because cibuildwheel copies only the Python package
# into the container, while build_lib.py needs the native sources next to it.
printf -v CUDA12_MOUNT '%q' "$CUDA12_HOME:$CUDA12_HOME:ro"
printf -v CUDA13_MOUNT '%q' "$CUDA13_HOME:$CUDA13_HOME:ro"
printf -v REPO_MOUNT '%q' "$REPO_ROOT:$REPO_ROOT"
export CIBW_CONTAINER_ENGINE="docker; create_args: -v $CUDA12_MOUNT -v $CUDA13_MOUNT -v $REPO_MOUNT"

# Build the native libraries inside the manylinux container and stage each .so
# and its owned headers into the package cibuildwheel is about to build. The
# libraries ship in the wheel, so building them on the host would tie the wheel
# to the host glibc and fail auditwheel's manylinux check.
# EP config.h records build-toolkit paths; runtime environment variables
# override those defaults, so only EP's other installed headers must match.
# The container runs as root, so git rejects the mounted checkout as
# dubiously owned; a LIB@VERSION target needs git to create its worktree.
CONTAINER_PYTHON=/opt/python/cp312-cp312/bin/python
printf -v QUOTED_REPO_ROOT '%q' "$REPO_ROOT"
printf -v QUOTED_BUILD_SCRIPT '%q' "$SCRIPT_DIR/build_lib.py"
export CIBW_BEFORE_ALL="git config --global --add safe.directory $QUOTED_REPO_ROOT && \
    $CONTAINER_PYTHON -m pip install -q packaging && \
    NATIVE_STAGING=\$(mktemp -d) && \
    echo \">>> Starting native build with CUDA 12: \$CUDA12_HOME\" && \
    $CONTAINER_PYTHON $QUOTED_BUILD_SCRIPT $TARGET_COMMAND \
        --cuda-home \"\$CUDA12_HOME\" --output-dir \"\$NATIVE_STAGING/cu12\" && \
    echo \">>> Finished native build with CUDA 12: \$CUDA12_HOME\" && \
    echo \">>> Starting native build with CUDA 13: \$CUDA13_HOME\" && \
    $CONTAINER_PYTHON $QUOTED_BUILD_SCRIPT $TARGET_COMMAND \
        --cuda-home \"\$CUDA13_HOME\" --output-dir \"\$NATIVE_STAGING/cu13\" && \
    echo \">>> Finished native build with CUDA 13: \$CUDA13_HOME\" && \
    if readelf -d \
        \"\$NATIVE_STAGING/cu12/nccl/ep/lib/cu12/libnccl_ep.so\" \
        \"\$NATIVE_STAGING/cu12/nccl/m2n/lib/cu12/libnccl_m2n.so\" \
        \"\$NATIVE_STAGING/cu13/nccl/ep/lib/cu13/libnccl_ep.so\" \
        \"\$NATIVE_STAGING/cu13/nccl/m2n/lib/cu13/libnccl_m2n.so\" | \
        grep -q 'libcudart\\.so'; then \
        echo 'Error: native libraries must not dynamically link libcudart' >&2; \
        exit 1; \
    fi && \
    diff -qr --exclude=config.h \
        \"\$NATIVE_STAGING/cu12/nccl/ep/include\" \
        \"\$NATIVE_STAGING/cu13/nccl/ep/include\" && \
    diff -qr \
        \"\$NATIVE_STAGING/cu12/nccl/m2n/include\" \
        \"\$NATIVE_STAGING/cu13/nccl/m2n/include\" && \
    mkdir -p \"{package}/nccl/ep/lib/cu12\" \"{package}/nccl/ep/lib/cu13\" \
        \"{package}/nccl/m2n/lib/cu12\" \"{package}/nccl/m2n/lib/cu13\" && \
    cp \"\$NATIVE_STAGING/cu12/nccl/ep/lib/cu12/libnccl_ep.so\" \
        \"{package}/nccl/ep/lib/cu12/libnccl_ep.so\" && \
    cp \"\$NATIVE_STAGING/cu13/nccl/ep/lib/cu13/libnccl_ep.so\" \
        \"{package}/nccl/ep/lib/cu13/libnccl_ep.so\" && \
    cp \"\$NATIVE_STAGING/cu12/nccl/m2n/lib/cu12/libnccl_m2n.so\" \
        \"{package}/nccl/m2n/lib/cu12/libnccl_m2n.so\" && \
    cp \"\$NATIVE_STAGING/cu13/nccl/m2n/lib/cu13/libnccl_m2n.so\" \
        \"{package}/nccl/m2n/lib/cu13/libnccl_m2n.so\" && \
    cp -a \"\$NATIVE_STAGING/cu12/nccl/ep/include\" \"{package}/nccl/ep/\" && \
    cp -a \"\$NATIVE_STAGING/cu12/nccl/m2n/include\" \"{package}/nccl/m2n/\""

uv tool run cibuildwheel --output-dir "$DIST_DIR" --platform linux "$SDIST"
cp "$SDIST" "$DIST_DIR/"

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo "CUDA12_HOME: $CUDA12_HOME"
echo "CUDA13_HOME: $CUDA13_HOME"
echo "Output directory: $DIST_DIR"
echo ""
echo "Built distributions:"
find "$DIST_DIR" -maxdepth 1 -type f \( -name "*.whl" -o -name "*.tar.gz" \) | sort
