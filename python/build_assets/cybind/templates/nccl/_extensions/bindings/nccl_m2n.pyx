# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated $version_span. Do not modify it directly.

cimport cython  # NOQA
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr,
                              get_resource_ptr, get_nested_resource_ptr)

_version_span = "$version_span"
__version__ = _version_span.split()[-1]

# NCCL_VERSION(X,Y,Z) = X*10000 + Y*100 + Z (NCCL >= 2.9).
_version_parts = __version__.split(".")
__version_code__ = (
    int(_version_parts[0]) * 10000 + int(_version_parts[1]) * 100 + int(_version_parts[2])
)


###############################################################################
# POD
###############################################################################

$pod_defs


###############################################################################
# Enum
###############################################################################

$enum_defs


# Hand-written: these are #defines in nccl_m2n.h, so cybind's enum extraction
# cannot see them; cynccl_m2n.pxd re-declares them for the ABI.
MESH_NDIMS = NCCL_RESHARD_MAX_MESH_DIMS
MAX_TENSOR_DIMS = NCCL_RESHARD_MAX_TENSOR_DIMS
REPLICATE = NCCL_RESHARD_REPLICATE


###############################################################################
# Error handling
###############################################################################

from nccl.bindings.nccl import NCCLError as _NCCLError
from ._internal.utils import FunctionNotFoundError


# Hand-written: derives from nccl4py's NCCLError and attaches the detail string
# from ncclM2nGetLastError(), which the generated exception defs cannot express.

class NCCLReshardError(_NCCLError):

    def __init__(self, status, detail=None):
        self.status = int(status)
        self.detail = detail
        message = f"NCCL Reshard error code {self.status}"
        if detail:
            message += f": {detail}"
        Exception.__init__(self, message)

    def __reduce__(self):
        return (type(self), (self.status, self.detail))


@cython.profile(False)
cpdef inline check_status(int status):
    cdef const char* detail = NULL
    cdef bytes detail_bytes
    cdef object detail_text = None
    if status != 0:
        detail = ncclM2nGetLastError()
        if detail != NULL:
            detail_bytes = <bytes>detail
            if detail_bytes:
                detail_text = detail_bytes.decode("utf-8", "replace")
        raise NCCLReshardError(status, detail_text)


###############################################################################
# Wrapper functions
###############################################################################

$wrapper_defs



# Hand-written: reports the .so the symbols resolved to; not a C entry point.
cpdef object get_library_path():
    from ._internal.nccl_m2n import _inspect_loaded_library_path
    return _inspect_loaded_library_path()
