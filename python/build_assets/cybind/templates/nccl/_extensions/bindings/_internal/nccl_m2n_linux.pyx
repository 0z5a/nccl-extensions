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
# Library resolution
###############################################################################

def _candidate_library_paths() -> list[str]:
    explicit = os.environ.get("NCCL_M2N_LIBRARY")
    if explicit:
        return [explicit]

    # With no explicit override, prefer the native library bundled with this
    # facade before environment and SONAME fallbacks.
    bundled = bundled_library("${libname}")
    candidates = [bundled] if bundled is not None else []

    home = os.environ.get("NCCL_M2N_HOME")
    if home:
        candidates.append(os.path.join(home, "lib", "libnccl_m2n.so"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for subdir in ("lib", "lib64"):
            candidates.append(os.path.join(conda_prefix, subdir, "libnccl_m2n.so"))

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            for subdir in ("lib", "lib64"):
                candidates.append(os.path.join(root, subdir, "libnccl_m2n.so"))

    candidates.append("libnccl_m2n.so")
    return candidates


###############################################################################
# Wrapper init
###############################################################################

$wrapper_init


cdef void* load_library() except* with gil:
    # libnccl_m2n.so has NEEDED libnccl.so.2. Forcing nccl4py's loader to run
    # maps that SONAME RTLD_GLOBAL first, so the NEEDED resolves without a
    # filesystem search and nccl4py stays the one place that locates libnccl.
    from nccl.bindings._internal import nccl as _nccl_loader
    _nccl_loader._inspect_function_pointers()

    cdef void* handle = NULL
    cdef bytes path_bytes
    cdef char* err_msg
    errors = []

    for path in _candidate_library_paths():
        if path != "libnccl_m2n.so" and not os.path.exists(path):
            errors.append(f"{path}: not found")
            continue
        path_bytes = path.encode()
        handle = dlopen(path_bytes, RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            return handle
        err_msg = dlerror()
        if err_msg != NULL:
            errors.append(f"{path}: {err_msg.decode()}")
        else:
            errors.append(f"{path}: dlopen failed")

    raise RuntimeError(
        "Failed to dlopen libnccl_m2n.so. Set NCCL_M2N_LIBRARY to the "
        "shared library path or NCCL_M2N_HOME to an install prefix. Tried: "
        + "; ".join(errors)
    )


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
