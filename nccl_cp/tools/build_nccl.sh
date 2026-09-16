#!/bin/sh
# Prepare the recorded NCCL dependency without changing an existing checkout.
set -eu

project_root=$1
make_program=$2
jobs=$3
dependency="$project_root/third_party/nccl"

fail() {
    printf '%s\n' "NCCL dependency: $*" >&2
    exit 1
}

case "$jobs" in
    ''|*[!0-9]*|0) fail "JOBS must be a positive integer" ;;
esac

root=$(git -C "$project_root" rev-parse --show-toplevel 2>/dev/null) ||
    fail "use a Git checkout, or set NCCL_HOME to an existing NCCL build"
[ "$(cd "$root" && pwd -P)" = "$(cd "$project_root" && pwd -P)" ] ||
    fail "the project must be its own Git checkout"
entry=$(git -C "$project_root" ls-files --stage -- third_party/nccl)
[ -n "$entry" ] || fail "third_party/nccl is not recorded as a submodule"
mode=$(printf '%s\n' "$entry" | awk '{print $1}')
revision=$(printf '%s\n' "$entry" | awk '{print $2}')
index_stage=$(printf '%s\n' "$entry" | awk '{print $3}')
[ "$mode" = 160000 ] && [ "$index_stage" = 0 ] ||
    fail "third_party/nccl has no unambiguous submodule revision"

if [ -e "$dependency/.git" ]; then
    [ -z "$(git -C "$dependency" status --porcelain --untracked-files=normal)" ] ||
        fail "third_party/nccl has local changes; preserve or resolve them before building"
    [ "$(git -C "$dependency" rev-parse HEAD)" = "$revision" ] ||
        fail "third_party/nccl is at another revision; select the recorded revision explicitly"
else
    git -C "$project_root" submodule update --init -- third_party/nccl
fi

[ "$(git -C "$dependency" rev-parse HEAD)" = "$revision" ] ||
    fail "the checkout does not match the recorded revision"
[ -f "$dependency/makefiles/version.mk" ] || fail "the NCCL checkout is incomplete"

exec "$make_program" -C "$dependency" -j "$jobs" src.build "BUILDDIR=$dependency/build"
