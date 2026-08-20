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


###############################################################################
# Error handling
###############################################################################

class NCCLEpError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = f"NCCL EP error code {status}"
        super(NCCLEpError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise NCCLEpError(status)


###############################################################################
# Wrapper functions
###############################################################################

$wrapper_defs


# Hand-written: reports the .so the symbols resolved to; not a C entry point.
cpdef object get_library_path():
    from ._internal.nccl_ep import _inspect_loaded_library_path
    return _inspect_loaded_library_path()
