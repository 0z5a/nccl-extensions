# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated $version_span. Do not modify it directly.

from .utils import FunctionNotFoundError, NotSupportedError

import os
from nccl._extensions._runtime import bundled_library


cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()

    enum:
        RTLD_NOW
        RTLD_GLOBAL

    ctypedef struct Dl_info:
        const char* dli_fname
        void* dli_fbase
        const char* dli_sname
        void* dli_saddr
    int dladdr(const void*, Dl_info*)


###############################################################################
# Library resolution. libnccl_ep.so is not an NVIDIA wheel library, so it is
# located here instead of through cuda.pathfinder.
###############################################################################

# The matching CUDA variant ships under this library's facade package.
# _resolve_library_path() runs on the first call that needs a symbol, not at
# import; after that the generated init guard holds the resolved pointers.
def _resolve_library_path() -> str:
    # 1. CUDA-specific nccl-extensions package path.
    pkg_lib = bundled_library("${libname}")
    if pkg_lib is not None:
        return pkg_lib

    # 2. CONDA_PREFIX/lib[64]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for sub in ("lib", "lib64"):
            candidate = os.path.join(conda_prefix, sub, "lib${libname}.so")
            if os.path.exists(candidate):
                return candidate

    # 3. CUDA_HOME / CUDA_PATH lib[64]
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            for sub in ("lib", "lib64"):
                candidate = os.path.join(root, sub, "lib${libname}.so")
                if os.path.exists(candidate):
                    return candidate

    # 4. SONAME fallback — let dlopen perform its own search across
    # LD_LIBRARY_PATH, /etc/ld.so.cache, and /lib, /usr/lib, /lib64,
    # /usr/lib64. If it fails the caller surfaces a clear error.
    return "lib${libname}.so"


###############################################################################
# Wrapper init
###############################################################################

$wrapper_init


cdef void* load_library() except* with gil:
    # libnccl_ep.so has NEEDED libnccl.so.2. Forcing nccl4py's loader to run
    # maps that SONAME RTLD_GLOBAL first, so the NEEDED resolves without a
    # filesystem search and nccl4py stays the one place that locates libnccl.
    from nccl.bindings._internal import nccl as _nccl_loader
    _nccl_loader._inspect_function_pointers()

    cdef bytes path_bytes = _resolve_library_path().encode()
    cdef void* handle = dlopen(path_bytes, RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise RuntimeError(
            f'Failed to dlopen lib${libname} ({err_msg.decode()}); '
            f'tried path {path_bytes.decode()!r}'
        )
    return handle


cdef object __${libname}_loaded_so_path = None


cpdef object _inspect_loaded_library_path():
    import os
    # Path of the .so backing the loaded symbols, via dladdr() on a
    # resolved entry point. None if it cannot be determined.
    global __${libname}_loaded_so_path
    if __${libname}_loaded_so_path is not None:
        return __${libname}_loaded_so_path

    cdef dict ptrs = _inspect_function_pointers()
    # Any resolved symbol maps to the same .so.
    cdef intptr_t addr = 0
    for value in ptrs.values():
        if value:
            addr = value
            break

    cdef Dl_info info
    if addr == 0:
        return None
    if dladdr(<void*>addr, &info) == 0 or info.dli_fname == NULL:
        return None
    __${libname}_loaded_so_path = os.fsdecode(<bytes>info.dli_fname)
    return __${libname}_loaded_so_path


###############################################################################
# Wrapper functions
###############################################################################

$wrapper_defs
