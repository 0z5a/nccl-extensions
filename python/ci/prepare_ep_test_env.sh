#!/usr/bin/env bash
# Build an NCCL Extensions wheel with the CI-produced NCCL EP artifacts and
# install it, plus the CUDA 12 and MPI test dependencies, into an isolated venv.

set -euo pipefail

: "${CUDA_HOME:?CUDA_HOME must be set}"
: "${NCCL_HOME:?NCCL_HOME must be set}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="${ROOT}/python"
EP_PACKAGE_DIR="${PACKAGE_DIR}/nccl/ep"
DIST_DIR="${PACKAGE_DIR}/dist"
WORK_DIR="${PACKAGE_DIR}/.ci-ep-python-${CI_JOB_ID:-local}"
BOOTSTRAP_DIR="${WORK_DIR}/bootstrap"
TOOLS_VENV_DIR="${WORK_DIR}/tools"
RUNTIME_VENV_DIR="${WORK_DIR}/runtime"
UV_CACHE_DIR="${WORK_DIR}/uv-cache"
UV_PYTHON_INSTALL_DIR="${WORK_DIR}/uv-python"
ENV_FILE="${WORK_DIR}/test-env.sh"

LIBRARY="${NCCL_HOME}/lib/libnccl_ep.so"
HEADER="${NCCL_HOME}/include/nccl_ep.h"
JIT_HEADERS="${NCCL_HOME}/include/nccl_ep"

test -s "${LIBRARY}"
test -s "${HEADER}"
test -d "${JIT_HEADERS}"

cleanup_package_artifacts() {
    rm -rf "${EP_PACKAGE_DIR}/lib" "${EP_PACKAGE_DIR}/include"
}
trap cleanup_package_artifacts EXIT

rm -rf "${WORK_DIR}" "${DIST_DIR}"
cleanup_package_artifacts
mkdir -p \
    "${EP_PACKAGE_DIR}/lib" \
    "${EP_PACKAGE_DIR}/include" \
    "${DIST_DIR}" \
    "${WORK_DIR}"
cp "${LIBRARY}" "${EP_PACKAGE_DIR}/lib/libnccl_ep.so"
cp "${HEADER}" "${EP_PACKAGE_DIR}/include/nccl_ep.h"
cp -a "${JIT_HEADERS}" "${EP_PACKAGE_DIR}/include/nccl_ep"

# The EOS runner's system Python can be older than this package supports.
# Bootstrap uv locally, then provision a supported build interpreter.
python3 -m pip install --disable-pip-version-check --target "${BOOTSTRAP_DIR}" uv
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
PYTHONPATH="${BOOTSTRAP_DIR}" python3 -m uv venv \
    --python 3.12 \
    --seed \
    "${TOOLS_VENV_DIR}"
"${TOOLS_VENV_DIR}/bin/python" -m pip install \
    --disable-pip-version-check \
    build \
    uv
"${TOOLS_VENV_DIR}/bin/python" -m build \
    --wheel \
    --outdir "${DIST_DIR}" \
    "${PACKAGE_DIR}"

wheel=("${DIST_DIR}"/nccl_extensions-*.whl)
if [[ ! -f "${wheel[0]}" || "${#wheel[@]}" -ne 1 ]]; then
    echo "ERROR: expected exactly one NCCL Extensions wheel in ${DIST_DIR}" >&2
    exit 1
fi

"${TOOLS_VENV_DIR}/bin/python" -m venv "${RUNTIME_VENV_DIR}"
"${RUNTIME_VENV_DIR}/bin/pip" install \
    --disable-pip-version-check \
    "${wheel[0]}[cu12,bench]"

LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${RUNTIME_VENV_DIR}/bin/python" -c \
    'import nccl.ep; from nccl._extensions.bindings import nccl_ep'

printf 'export PYTHON_BIN=%q\n' "${RUNTIME_VENV_DIR}/bin/python" > "${ENV_FILE}"
printf '%s\n' "${ENV_FILE}"
