"""CP group configuration layered on an existing PyTorch ProcessGroup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .collectives import _GroupState


@dataclass(frozen=True, kw_only=True)
class CpConfig:
    """Host configuration shared by all handles of a CP group.

    max_per_peer_slot: Row capacity C reserved for each native peer slot.
    max_layout_tokens: Maximum owned rows per rank for handle metadata,
        for either backend. None resolves to C at configuration construction.
        Requests may contain up to world_size * max_layout_tokens rows. Use a
        larger value when flat routes must exceed the native workspace capacity.
        This sizes metadata only and does not enlarge native workspace.
    payload_shape: One communication row's dimensions, excluding tensor axis 0.
        A row can represent a token, a head, or another caller-defined unit.
        A tuple/list of nonnegative Python integers; () means one scalar.
    dtype: Storage dtype used to size the payload budget, not a conversion.
        Supported: float16, bfloat16, float32, float64. Use a description that
        covers the largest intended cast/reduce row.
    max_per_token_bytes: Derived per-row budget B = prod(payload_shape) * dtype.itemsize.
    runtime_slot: Nonnegative workspace slot within the underlying ProcessGroup.
    nvl_domain_size: Ranks per NVL domain; None resolves NVL_DOMAIN_SIZE when
        constructing this configuration (default 8).
    backend: auto selects an eligible backend during handle preparation or
        explicit global-list preparation;
        all2allv explicitly selects the flat backend.

    Caller contract: size C/B for the intended zero-CTA traffic and keep
    per-call payload formats compatible across communicating ranks. This
    configuration reserves capacity; it does not convert tensors or verify
    their row contents. A reused native runtime may have a larger B.
    NVL domains must describe contiguous group-rank blocks with mutually
    device-accessible symmetric buffers. Preparation checks available peer
    identity/access information, not the application's placement intent.
    """

    max_per_peer_slot: int
    payload_shape: tuple[int, ...] | list[int]
    dtype: torch.dtype
    max_layout_tokens: int | None = None
    runtime_slot: int = 0
    nvl_domain_size: int | None = None
    backend: Literal["auto", "all2allv"] = "auto"
    max_per_token_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload_shape, (tuple, list)):
            raise TypeError("payload_shape must be a tuple or list excluding tensor axis 0")
        if any(type(size) is not int or size < 0 for size in self.payload_shape):
            raise ValueError("payload_shape dimensions must be nonnegative Python integers")
        if not isinstance(self.dtype, torch.dtype) or self.dtype not in (
            torch.float16, torch.bfloat16, torch.float32, torch.float64
        ):
            raise ValueError("dtype must be torch.float16, torch.bfloat16, torch.float32 or torch.float64")
        if any(type(value) is not int or value < 0 for value in (self.max_per_peer_slot, self.runtime_slot)):
            raise ValueError("max_per_peer_slot and runtime_slot must be nonnegative integers")
        layout_tokens = self.max_per_peer_slot if self.max_layout_tokens is None else self.max_layout_tokens
        if type(layout_tokens) is not int or layout_tokens < 0:
            raise ValueError("max_layout_tokens must be a nonnegative integer")
        nvl = int(os.environ.get("NVL_DOMAIN_SIZE", "8")) if self.nvl_domain_size is None else self.nvl_domain_size
        if type(nvl) is not int or nvl <= 0:
            raise ValueError("nvl_domain_size must be positive")
        if self.backend not in ("auto", "all2allv"):
            raise ValueError("backend must be 'auto' or 'all2allv'")
        shape = tuple(self.payload_shape)
        byte_capacity = prod(shape) * self.dtype.itemsize
        if self.max_per_peer_slot and not byte_capacity:
            raise ValueError("A nonempty runtime requires positive payload byte capacity")
        object.__setattr__(self, "max_layout_tokens", layout_tokens)
        object.__setattr__(self, "payload_shape", shape)
        object.__setattr__(self, "max_per_token_bytes", byte_capacity)
        object.__setattr__(self, "nvl_domain_size", nvl)


@dataclass(frozen=True, eq=False)
class CpGroup:
    """A configured CP group returned by create_group.

    nccl_group is the caller-created ProcessGroup (Gloo for CPU verification).
    cp_config is the immutable payload/capacity configuration. The caller owns
    the ProcessGroup; close_runtime drains CP resources before its destruction.
    """

    nccl_group: dist.ProcessGroup
    cp_config: CpConfig
    _state: _GroupState = field(repr=False)


def create_group(nccl_group: dist.ProcessGroup | None, cp_config: CpConfig) -> CpGroup:
    """Locally bind runtime configuration to an existing ProcessGroup.

    None selects WORLD. Handle/global-list preparation passes this config directly to
    ZeroCTACollectiveArg; its C++ runtime registry owns workspace reuse.

    Caller contract: initialize a supported ProcessGroup (NCCL, or Gloo for
    CPU verification), select this rank's assigned CUDA device first, and
    keep that device fixed for the group's lifetime. All group members must
    use the same CpConfig and enter handle preparation in matching order,
    outside graph capture. In particular max_layout_tokens determines the
    AllGather frame size and must already agree before create_handle starts;
    a size mismatch cannot be repaired by checks after that exchange.
    This factory performs local checks only; it does not contact other ranks.
    The caller must close CP runtimes before destroying the ProcessGroup.
    """
    from .collectives import _group, _state

    if not isinstance(cp_config, CpConfig):
        raise TypeError("cp_config must be a CpConfig")
    nccl_group = _group(nccl_group)
    state = _state(nccl_group)
    with state.lock:
        return CpGroup(nccl_group=nccl_group, cp_config=cp_config, _state=state)
