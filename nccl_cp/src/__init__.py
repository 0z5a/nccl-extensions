"""NCCL context-parallel communication."""

from importlib import import_module

__all__ = [
    "RowRange", "Handle", "create_handle", "group_cast", "group_reduce",
    "close_runtime", "StreamSpec",
    "group_cast_explicit", "group_reduce_explicit",
    "CpConfig", "CpGroup", "create_group", "WorkWithPostProcessFn",
    "group_cast_async", "group_reduce_async",
    "group_cast_explicit_async", "group_reduce_explicit_async",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module_name = (".group" if name in ("CpConfig", "CpGroup", "create_group") else
                   ".routing" if name == "RowRange" else
                   ".work" if name == "WorkWithPostProcessFn" else ".collectives")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
