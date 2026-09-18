"""Public context-parallel collectives and reusable route handles."""

from __future__ import annotations

import os
import pickle
import socket
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, TypeAlias, cast

import torch
import torch.distributed as dist

from . import flat, zero_cta
from ._types import GroupReduceOp
from .group import CpGroup
from .work import WorkWithPostProcessFn
from .routing import (
    Layout,
    Route,
    layout_intervals,
    local_hierarchy_entries,
    local_hierarchy_from_routes,
    local_route,
    peer_segments,
    supplied_routes,
    validate_layouts,
)

if TYPE_CHECKING:
    from ._types import ZeroCtaRuntime
    from .comm_meta import ZeroCTACollectiveArg


StreamSpec: TypeAlias = torch.cuda.Stream | int | None


@dataclass
class _GroupState:
    group: dist.ProcessGroup
    device: torch.device
    lock: threading.RLock = field(default_factory=threading.RLock)
    # References to the native runtimes; C++ owns allocation and reuse.
    runtimes: dict[int, ZeroCtaRuntime] = field(default_factory=dict)
    closed: bool = False


_states: dict[dist.ProcessGroup, _GroupState] = {}
_states_lock = threading.RLock()


def _group(group: CpGroup | dist.ProcessGroup | None) -> dist.ProcessGroup:
    if not dist.is_initialized():
        raise RuntimeError("Initialize torch.distributed before using NCCL CP")
    if isinstance(group, CpGroup):
        if group._state.closed:
            raise ValueError("CP group belongs to a closed runtime")
        group = group.nccl_group
    resolved = dist.group.WORLD if group is None else group
    if dist.get_rank(resolved) < 0:
        raise ValueError("The calling rank is not a member of this ProcessGroup")
    return resolved


def _state(group):
    with _states_lock:
        if group not in _states:
            device = torch.device("cuda", torch.cuda.current_device()) if str(dist.get_backend(group)).lower() == "nccl" else torch.device("cpu")
            _states[group] = _GroupState(group, device)
        return _states[group]


# Fixed control envelope for configuration, topology and diagnostic fields.
# Layout endpoints use separate int64 slots derived from the token bound.
_CONTROL_BYTES = 8192
_HEADER_WORDS = 3  # Control byte count, input interval count, output interval count.


def _exchange(group, phase: str, value=None, error: str = "", *,
              max_layout_tokens: int):
    """Exchange one fixed-size tensor during handle preparation only.

    The caller guarantees the same token bound on every rank before sizing;
    create_group performs no cross-rank check. Lengths,
    errors and contents travel together; no preceding size collective is needed.
    CPU staging/readback is confined to this control path, never payload calls.
    """
    state = _state(group)
    if state.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CP control collectives must run outside CUDA graph capture")
    world = dist.get_world_size(group)
    layout_capacity = max_layout_tokens * (1 + world)
    layout_start = _HEADER_WORDS + _CONTROL_BYTES // 8
    # Explicit host allocation avoids inheriting an application's default device.
    host = torch.zeros(layout_start + 2 * layout_capacity, dtype=torch.int64, device="cpu")
    try:
        layout, *settings = value
        control_value = tuple(settings)
        if not error:
            owned, requested = layout
            if (sum(stop - start for start, stop in owned) > max_layout_tokens or
                    sum(stop - start for start, stop in requested) > world * max_layout_tokens):
                raise ValueError("layout exceeds max_layout_tokens; increase the metadata token bound")
            endpoints = [point for ranges in layout for interval in ranges for point in interval]
            if any(not -(1 << 63) <= point < (1 << 63) for point in endpoints):
                raise ValueError("layout interval endpoints must fit signed int64")
            host[1], host[2] = len(owned), len(requested)
            if endpoints:
                host[layout_start:layout_start + len(endpoints)] = torch.tensor(
                    endpoints, dtype=torch.int64, device="cpu")
        control = pickle.dumps((phase, error, None if error else control_value), protocol=4)
        if len(control) > _CONTROL_BYTES:
            raise ValueError(f"CP control metadata exceeds the {_CONTROL_BYTES}-byte envelope")
    except (TypeError, ValueError, OverflowError, pickle.PickleError) as exc:
        # An overflowing/malformed local record still participates with the same
        # frame size, so every rank sees the error before planning/native setup.
        error = str(exc)[:1024]
        control = pickle.dumps((phase, error, None), protocol=4)
        host[1:3] = 0
    host[0] = len(control)
    host[_HEADER_WORDS:layout_start].view(torch.uint8)[:len(control)] = torch.frombuffer(
        bytearray(control), dtype=torch.uint8)
    with torch.cuda.device(state.device) if state.device.type == "cuda" else nullcontext():
        sending = host.to(state.device) if state.device.type == "cuda" else host
        received = torch.empty((world, host.numel()), dtype=torch.int64, device=state.device)
        dist.all_gather(list(received.unbind(0)), sending, group=group)
        # One readback after the contents arrive; no intermediate length readback.
        received = received.cpu()
    records = [pickle.loads(bytes(row[_HEADER_WORDS:layout_start].view(torch.uint8)[:int(row[0])].tolist()))
               for row in received]
    if any(record[0] != phase for record in records):
        raise ValueError("All ranks must call the same CP operation in the same order")
    failures = [f"rank {rank}: {record[1]}" for rank, record in enumerate(records) if record[1]]
    if failures:
        raise ValueError("; ".join(failures))
    values = []
    for row, record in zip(received, records):
        n_input, n_output = int(row[1]), int(row[2])
        pairs = row[layout_start:layout_start + 2 * (n_input + n_output)].view(-1, 2).tolist()
        layout = (tuple(map(tuple, pairs[:n_input])), tuple(map(tuple, pairs[n_input:])))
        values.append((layout, *record[2]))
    return values


def _resolve_stream(state: _GroupState, stream: StreamSpec) -> torch.cuda.Stream | None:
    if state.device.type == "cpu":
        if stream is not None:
            raise ValueError("CPU/Gloo calls require stream=None")
        return None
    if isinstance(stream, torch.cuda.Stream):
        if torch.device(stream.device) != state.device:
            raise ValueError("stream must belong to the ProcessGroup CUDA device")
        return stream
    if type(stream) is int and stream >= 0:
        # Caller-owned raw handles must name a valid stream on this device and
        # remain alive until queued work completes; ExternalStream does not own it.
        return (torch.cuda.default_stream(state.device) if stream == 0 else
                torch.cuda.ExternalStream(stream, device=state.device))
    raise TypeError("CUDA stream must be a torch.cuda.Stream or a nonnegative stream pointer")


def _image_supports(architectures: tuple[str, ...], sm: int) -> bool:
    for target in architectures:
        number, _, kind = target.partition("-")
        if number.isdecimal():
            architecture = int(number)
            if kind == "real" and sm == architecture:
                return True
            if kind in ("", "virtual") and sm >= architecture:
                return True
    return False


def _probe_zero(group):
    """Read detectable prerequisites; importing the flat backend never calls this loader."""
    result = {"ok": False, "reason": "", "host": socket.gethostname(), "uuid": "", "accessible": ()}
    try:
        if os.environ.get("NCCL_CP_DISABLE_ZERO_CTA") == "1":
            result["reason"] = "zero-CTA disabled"
            return result
        if str(dist.get_backend(group)).lower() != "nccl" or not torch.cuda.is_available():
            result["reason"] = "zero-CTA requires a CUDA NCCL ProcessGroup"
            return result
        if tuple(torch.cuda.nccl.version()) < (2, 30, 0):
            result["reason"] = "zero-CTA requires NCCL 2.30+"
            return result
        device = torch.cuda.current_device()
        backend = group._get_backend(torch.device("cuda", device))
        options = backend.options
        zero = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        if not (int(options.config.cta_policy) & int(zero)):
            result["reason"] = "ProcessGroup CTA policy is not ZERO"
            return result
        root = os.path.dirname(torch.__file__)
        if not os.path.isfile(os.path.join(root, "include/torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp")):
            result["reason"] = "PyTorch lacks NCCLSymmetricMemory"
            return result
        from . import zero_cta
        extension = zero_cta._get_extension()
        properties = torch.cuda.get_device_properties(device)
        if not _image_supports(zero_cta._native_architectures, properties.major * 10 + properties.minor):
            result["reason"] = "native image targets do not cover the current GPU"
            return result
        for name in ("get_or_create_runtime", "create_cast_plan", "create_reduce_plan", "run"):
            if not hasattr(extension, name):
                raise RuntimeError(f"Native extension is missing {name}")
        uuids = [str(getattr(torch.cuda.get_device_properties(i), "uuid", ""))
                 for i in range(torch.cuda.device_count())]
        result["uuid"] = uuids[device]
        result["accessible"] = tuple(uuid for i, uuid in enumerate(uuids)
                                     if i == device or torch.cuda.can_device_access_peer(device, i))
        result["ok"] = True
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as error:
        result["reason"] = str(error).split("\n")[0]
    return result


def _topology_reason(profiles, nvl: int):
    world = len(profiles)
    if world > nvl and world % nvl:
        return "NVL domain size does not divide the group"
    for start in range(0, world, nvl):
        block = profiles[start:min(start + nvl, world)]
        if len({item["host"] for item in block}) != 1:
            return "Ranks within an NVL domain must be on the same host"
        if len(block) > 1:
            ids = [item["uuid"] for item in block]
            if any(not uuid for uuid in ids) or len(set(ids)) != len(ids):
                return "NVL domain GPU identities are unavailable or repeated"
            if any(not set(ids).issubset(item["accessible"]) for item in block):
                return "NVL peers are not visible with peer access enabled"
    return ""


class Handle:
    """Reusable rank-local route; close() drops its plan reference, not workspace.

    Caller contract: reuse only while row identities and axis-0 row order
    still match the prepared layouts. Returned list properties are copies;
    editing them does not update the plan. Keep this handle and all submitted
    tensors alive until GPU work completes, and only then close/release them.
    Data calls do not retain submission references or manage completion events.
    """

    def __init__(self, group: CpGroup, state: _GroupState, route: Route,
                 has_transfers: bool, zero_allowed: bool, reason: str):
        self._group = group
        self._rank = dist.get_rank(group.nccl_group)
        self._state = state
        self._route = route
        self._has_transfers = has_transfers
        self._send, self._recv = peer_segments(route, dist.get_world_size(group.nccl_group))
        self._backend = "zero_cta" if zero_allowed else "all2allv"
        self._reason = reason
        self._native: ZeroCTACollectiveArg | None = None
        self._closed = False

    @property
    def group(self) -> CpGroup:
        return self._group

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self._group.nccl_group

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def input_split_size_list(self) -> list[int]:
        return list(self._route.input_splits)

    @property
    def output_split_size_list(self) -> list[int]:
        return list(self._route.output_splits)

    @property
    def dst_indices_list(self) -> list[list[int]]:
        return [list(peers) for peers in self._route.destinations]

    @property
    def src_index_list(self) -> list[int]:
        return list(self._route.sources)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def fallback_reason(self) -> str:
        return self._reason

    def group_cast(
        self, input: torch.Tensor, output: torch.Tensor, *, stream: StreamSpec,
        cast_lse: bool = False, input_lse: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
    ) -> None:
        """Call group_cast with this handle; see its Args/Returns/Notes contract."""
        group_cast(self, input, output, stream=stream, cast_lse=cast_lse,
                   input_lse=input_lse, output_lse=output_lse)

    def group_reduce(
        self, grad_input: torch.Tensor, grad_output: torch.Tensor, *, stream: StreamSpec,
        reduce_op: GroupReduceOp = "sum", acc_reduce: bool = True,
        comm_dtype: torch.dtype | None = None,
        input_lse: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
    ) -> None:
        """Call group_reduce with this handle; see its Args/Returns/Notes contract."""
        group_reduce(
            self, grad_input, grad_output, stream=stream, reduce_op=reduce_op,
            acc_reduce=acc_reduce, comm_dtype=comm_dtype,
            input_lse=input_lse, output_lse=output_lse,
        )

    def group_cast_async(
        self, input: torch.Tensor, output: torch.Tensor, *, stream: StreamSpec,
        cast_lse: bool = False, input_lse: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
    ) -> WorkWithPostProcessFn:
        """Launch cast; call the returned work.wait_post_process() on stream."""
        return group_cast_async(self, input, output, stream=stream, cast_lse=cast_lse,
                                input_lse=input_lse, output_lse=output_lse)

    def group_reduce_async(
        self, grad_input: torch.Tensor, grad_output: torch.Tensor, *, stream: StreamSpec,
        reduce_op: GroupReduceOp = "sum", acc_reduce: bool = True,
        comm_dtype: torch.dtype | None = None,
        input_lse: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
    ) -> WorkWithPostProcessFn:
        """Launch reduce; call the returned work.wait_post_process() on stream."""
        return group_reduce_async(
            self, grad_input, grad_output, stream=stream, reduce_op=reduce_op,
            acc_reduce=acc_reduce, comm_dtype=comm_dtype,
            input_lse=input_lse, output_lse=output_lse,
        )

    def close(self) -> None:
        """Release native plans after GPU completion; shared workspace stays live.

        The caller completes all uses first, including async post-processing.
        This method neither synchronizes the GPU nor closes the ProcessGroup.
        close_runtime releases the ProcessGroup's shared runtime slots.
        """
        with self._state.lock:
            self._closed = True
            self._native = None
            self._send = self._recv = ()

    def __enter__(self) -> Handle:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def create_handle(group: CpGroup, local_owned_layout: Layout, local_required_layout: Layout, *,
                  stream: StreamSpec) -> Handle:
    """Prepare the rank-local route and zero-CTA plans before data calls.

    local_owned_layout describes this rank's owned rows: cast input and reduce
    output. local_required_layout describes all rows this rank needs, including
    any local rows: cast output and reduce input. Row IDs can represent tokens,
    heads, or other caller-defined axis-0 units.

    Caller contract: all ranks, including empty/relay-only ranks, prepare the
    same logical communication in matching order. Layout IDs describe global
    row identities and tensor row order, not local offsets. Shared descriptors
    are validated here, but their correspondence to actual tensor contents is
    the caller's responsibility. A changed mapping requires a new handle.
    max_layout_tokens bounds owned rows per rank; requests may contain at most
    world_size times that count. This bound covers both communication backends.
    Preparation performs one tensor AllGather and metadata uploads outside graph
    capture. Prepare and serially submit every handle sharing a native runtime
    slot on the same caller-owned CUDA stream. The caller also keeps input,
    output and handles alive until GPU completion; CP adds no Python events or
    per-call cross-stream ordering to enforce this contract.
    """
    if not isinstance(group, CpGroup):
        raise TypeError("group must be a CpGroup returned by create_group")
    cp_group, config = group, group.cp_config
    group = _group(group)
    state = _state(group)
    with state.lock:
        error, selected = "", None
        try:
            selected = _resolve_stream(state, stream)
        except (TypeError, ValueError, RuntimeError) as exc:
            error = str(exc)
        with torch.cuda.stream(selected) if selected is not None else nullcontext():
            layout = None
            try:
                layout = layout_intervals(local_owned_layout), layout_intervals(local_required_layout)
            except (TypeError, ValueError) as exc:
                error = error or str(exc)
            existing = None
            profile = {"ok": False, "reason": "invalid preparation arguments"}
            try:
                if state.device.type == "cuda" and torch.cuda.current_device() != state.device.index:
                    raise ValueError("Keep the CUDA device fixed for the ProcessGroup lifetime")
                runtime = state.runtimes.get(config.runtime_slot)
                if runtime is not None:
                    existing = (runtime.max_per_peer_slot(), runtime.nvl_domain_size(), runtime.max_per_token_bytes())
                if config.backend == "all2allv":
                    profile = {"ok": False, "reason": "all2allv selected in CpConfig"}
                elif config.dtype not in (torch.float32, torch.bfloat16):
                    profile = {"ok": False, "reason": "payload dtype requires all2allv"}
                elif config.max_per_token_bytes == 0:
                    profile = {"ok": False, "reason": "empty payload"}
                else:
                    profile = _probe_zero(group)
                if profile["ok"]:
                    if existing is not None:
                        if (existing[:2] != (config.max_per_peer_slot, config.nvl_domain_size) or
                                existing[2] < config.max_per_token_bytes):
                            profile.update(ok=False, reason="existing runtime slot capacity does not match CpConfig")
                    else:
                        world, nvl = dist.get_world_size(group), config.nvl_domain_size
                        slots = 2 * world if world <= nvl else world + 2 * (world // nvl)
                        if slots * config.max_per_peer_slot * config.max_per_token_bytes > torch.cuda.mem_get_info(state.device)[0]:
                            profile.update(ok=False, reason="insufficient available memory for symmetric workspace")
            except (TypeError, ValueError, RuntimeError) as exc:
                error = error or str(exc)
            gathered = _exchange(group, "create_handle/layouts", (layout, config, existing, profile), error,
                                 max_layout_tokens=config.max_layout_tokens)
            if any(item[1:3] != gathered[0][1:3] for item in gathered):
                raise ValueError("Runtime settings must match across ranks")
            layouts = tuple(item[0] for item in gathered)
            ownership = validate_layouts(layouts)
            profiles = [item[3] for item in gathered]
            reasons = [f"rank {rank}: {profile['reason']}" for rank, profile in enumerate(profiles)
                       if not profile["ok"]]
            reason = "; ".join(reasons) if reasons else _topology_reason(profiles, config.nvl_domain_size)
            if not reason and config.max_per_peer_slot < max(sum(stop - start for start, stop in owned)
                                                            for owned, _ in layouts):
                reason = "route exceeds the runtime slot capacity"
            rank = dist.get_rank(group)
            route = local_route(rank, layouts, ownership=ownership,
                                build_entries=not reason and len(layouts) > config.nvl_domain_size)
            handle = Handle(cp_group, state, route, any(requested for _, requested in layouts), not reason, reason)
            if not reason:
                from .comm_meta import GroupCollectiveEntry, ZeroCTACollectiveArg

                handle._native = ZeroCTACollectiveArg(
                    input_split_size_list=handle.input_split_size_list,
                    output_split_size_list=handle.output_split_size_list,
                    dst_indices_list=handle.dst_indices_list,
                    src_index_list=handle.src_index_list,
                    rank=handle.rank, world_size=len(layouts), group=group,
                    hierarchy_meta=tuple(GroupCollectiveEntry(e.owner, e.input_start, e.n_tokens, e.output_rank_starts)
                                         for e in local_hierarchy_entries(handle.rank, layouts, route, config.nvl_domain_size)),
                    max_per_peer_slot=config.max_per_peer_slot,
                    max_per_token_bytes=config.max_per_token_bytes,
                    runtime_slot=config.runtime_slot, nvl_domain_size=config.nvl_domain_size,
                )
                state.runtimes[config.runtime_slot] = handle._native.runtime
            return handle


def _nonoverlapping(tensor: torch.Tensor) -> bool:
    if tensor.numel() <= 1 or torch._debug_has_internal_overlap(tensor) == 0:
        return True
    span = 1
    for stride, size in sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride()) if size > 1):
        if stride < span:
            return False
        span += (size - 1) * stride
    return True


def _validate_tensors(input, output, route: Route, group, reduce: bool,
                      state: _GroupState | None = None):
    """Validate flat-backend tensors; native calls retain their C++ checks."""
    if not isinstance(input, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("input and output must both be tensors")
    if input.layout != torch.strided or output.layout != torch.strided or input.ndim < 1:
        raise ValueError("input/output must be strided tensors with a row dimension")
    rows = (route.output_rows, route.input_rows) if reduce else (route.input_rows, route.output_rows)
    # The cast receive buffer (and reverse reduce input) may reserve extra
    # rows. Plans/segments use logical route counts, never this capacity.
    input_rows_ok = input.shape[0] >= rows[0] if reduce else input.shape[0] == rows[0]
    output_rows_ok = output.shape[0] == rows[1] if reduce else output.shape[0] >= rows[1]
    if not input_rows_ok or not output_rows_ok or tuple(input.shape[1:]) != tuple(output.shape[1:]):
        raise ValueError(
            "Tensor shapes must match the route and per-row payload shape; "
            "cast output/reduce input may have extra trailing rows but cannot be smaller than the route"
        )
    if input.dtype != output.dtype or input.device != output.device:
        raise ValueError("input/output must have the same dtype and device")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Supported payload dtypes are float16, bfloat16, float32 and float64")
    backend = str(dist.get_backend(group)).lower()
    if backend == "nccl":
        device = state.device if state is not None else _state(group).device
        if input.device.type != "cuda" or input.device != device or input.device.index != torch.cuda.current_device():
            raise ValueError("NCCL tensors must use the rank's current CUDA device")
    elif backend == "gloo":
        if input.device.type != "cpu":
            raise ValueError("The Gloo verification path requires CPU tensors")
    else:
        raise ValueError("NCCL CP supports NCCL and Gloo ProcessGroups")
    if not _nonoverlapping(output):
        raise ValueError("output must not contain overlapping tensor elements")


def _execute(group, input, output, route: Route, handle: Handle | None, reduce: bool, stream: StreamSpec,
             *, async_op: bool = False) -> WorkWithPostProcessFn | None:
    """Submit a flat route using cached peer segments, without control traffic.

    Caller contract: match operation/handle order and per-row shape/dtype
    across ranks. A local argument error is not broadcast to peers; recovery
    must be coordinated by the application rather than ignored on one rank.
    Use one fixed stream per native slot and serialize host submissions with
    preparation/close. The caller owns stream ordering and all data/plan lifetimes.
    """
    supplied_group = group
    if isinstance(group, CpGroup):
        state = group._state
        if state.closed:
            raise ValueError("CP group belongs to a closed runtime")
        group = group.nccl_group
    else:
        group = _group(group)
        state = _state(group)
    if handle is not None:
        if not isinstance(handle, Handle) or handle.nccl_group is not group or handle._state is not state:
            raise ValueError("handle belongs to another group or a closed runtime")
        if isinstance(supplied_group, CpGroup) and handle.group is not supplied_group:
            raise ValueError("handle belongs to another CP group")
        if handle._closed:
            raise ValueError("handle is closed")
    selected = _resolve_stream(state, stream)
    with torch.no_grad(), (torch.cuda.stream(selected) if selected is not None else nullcontext()):
        _validate_tensors(input, output, route, group, reduce, state)
        if (handle is None or handle._has_transfers) and prod(input.shape[1:]) != 0:
            send, recv = (peer_segments(route, dist.get_world_size(group)) if handle is None else
                          (handle._send, handle._recv))
            if reduce:
                send, recv = recv, send
            if async_op:
                return flat.run_async(input, output, send, recv, group, reduce=reduce)
            flat.run(input, output, send, recv, group, reduce=reduce)
        elif async_op:
            return WorkWithPostProcessFn(post_process_fn=lambda *_args, **_kwargs: output, async_op=True)


def group_cast(
    handle: Handle, input: torch.Tensor, output: torch.Tensor, *, stream: StreamSpec,
    cast_lse: bool = False, input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> None:
    """Group cast interface using a prepared handle.

    Args:
        handle (Handle): Route and execution plans returned by create_handle.
            Its owned/required layouts describe this rank's tensor row order;
            its CpGroup supplies the ProcessGroup, backend and runtime slot.
        input (torch.Tensor): Local source tensor shaped
            [input_seqlen, *payload_shape], with input_seqlen equal to
            sum(handle.input_split_size_list). Each input segment is sent to
            the destination ranks stored in the handle.
        output (torch.Tensor): Required, caller-allocated destination buffer
            shaped [output_capacity, *payload_shape], where
            sum(handle.output_split_size_list) <= output_capacity. Only the
            valid prefix is written; extra trailing rows keep their existing
            values and are not sent or zeroed. Must have the same dtype,
            device and per-row shape as input. The supplied storage is
            written directly; None and automatic output allocation are not
            part of this interface.
        stream (StreamSpec): torch.cuda.Stream or a nonnegative raw CUDA
            stream pointer on this rank's fixed CUDA device; 0 denotes the
            default stream. CPU/Gloo verification calls use None.
            For zero-CTA, the caller must already have made this stream
            current outside the submission loop; this argument is a caller
            assertion, with no per-call validation or stream switching.
            The flat backend selects the supplied stream internally.
        cast_lse (bool): Reserved LSE-cast switch, default False. True raises
            NotImplementedError; LSE communication is not implemented.
        input_lse (torch.Tensor | None): Reserved input LSE tensor, default
            None. Any non-None value raises NotImplementedError, even when
            cast_lse is False; its shape/dtype are not inspected.
        output_lse (torch.Tensor | None): Reserved output LSE buffer, default
            None. Any non-None value raises NotImplementedError; the buffer
            is not allocated or written by this interface.

    Returns:
        None: Communication and post-processing have been submitted. CUDA
        completion is not implied, and no WorkWithPostProcessFn is returned.

    Notes:
        Output segments from the same source preserve that source's input
        order. Changing token ownership or row order requires a new handle.
        zero-CTA supports float32/bfloat16 with dense trailing-dimension
        strides, including singleton dimensions; token-axis stride padding
        is allowed. The flat backend also supports float16/float64 and strided
        tensors, subject to its output nonoverlap check.
        The actual payload bytes per token must fit the prepared runtime.
        Per-call input/output formats must agree across communicating ranks.

        Caller contract: match operations/handles across ranks, use a live
        handle and one fixed stream per native runtime slot, and serialize
        submissions with preparation/teardown. Native C++ tensor/work checks
        remain; Python does not recheck route row counts or runtime capacity.
        The caller orders input producers and output consumers, supplies any
        needed no_grad scope, and schedules reverse communication through its
        own autograd integration. Keep input, output, handle and external
        streams alive until GPU completion. Ranks with no valid rows keep
        the agreed payload shape/dtype and still enter matching calls;
        a receive buffer may have capacity even when its valid prefix is empty.

        LSE arguments are reserved and rejected when requested, before data
        submission. There is no async_op or kwargs option.
        This prepared entry adds no route exchange or metadata upload.
    """
    if handle._backend == "zero_cta":
        zero_cta.zero_cta_group_cast_impl(
            input, output, handle._native, handle._group.nccl_group, async_op=True,
            cast_lse=cast_lse, input_lse=input_lse, output_lse=output_lse,
        ).wait_post_process(output)
    else:
        if cast_lse or input_lse is not None or output_lse is not None:
            raise NotImplementedError("flat group-cast does not support LSE")
        _execute(handle.group, input, output, handle._route, handle, False, stream)


def group_reduce(
    handle: Handle, grad_input: torch.Tensor, grad_output: torch.Tensor, *, stream: StreamSpec,
    reduce_op: GroupReduceOp = "sum", acc_reduce: bool = True,
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> None:
    """Group reduce interface using a prepared handle; accumulate sums.

    Args:
        handle (Handle): The prepared cast route used in reverse. Its required
            layout describes grad_input; its owned layout describes grad_output.
        grad_input (torch.Tensor): Local contributions shaped
            [input_capacity, *payload_shape], where
            sum(handle.output_split_size_list) <= input_capacity. Only the
            valid prefix is read; trailing capacity rows are ignored. This
            permits the reverse of a cast with a capacity-padded output.
            Contributions are sent back to their owners in per-owner order.
        grad_output (torch.Tensor): Required, caller-allocated accumulator
            shaped [output_seqlen, *payload_shape], with output_seqlen equal
            to sum(handle.input_split_size_list). Must have the same dtype,
            device and per-row shape as grad_input. The result is the
            initial grad_output plus the sum of received contributions.
            Rows without contributions retain their initial values. Zero the
            buffer before calling when a plain sum is required; None, output
            allocation and an overwrite mode are not supported.
        stream (StreamSpec): torch.cuda.Stream or a nonnegative raw CUDA
            stream pointer on the group's fixed device; 0 denotes the default
            stream. CPU/Gloo verification uses None. For zero-CTA the caller
            must make this stream current before the call; no per-call stream
            switching/checking is added. The flat backend selects it internally.
        reduce_op (GroupReduceOp): Reserved reduction selector, default
            "sum". Any other value raises NotImplementedError.
        acc_reduce (bool): Accumulate into the supplied output, default
            True. False raises NotImplementedError; zero the output
            before calling when a plain sum is required.
        comm_dtype (torch.dtype | None): Communication dtype, default None
            (use grad_input.dtype). An explicit matching dtype is accepted;
            any different dtype raises NotImplementedError. No conversion
            or separate communication buffer is created.
        input_lse (torch.Tensor | None): Reserved input LSE tensor, default
            None. Any non-None value raises NotImplementedError, including
            an empty tensor; its shape/dtype are not inspected.
        output_lse (torch.Tensor | None): Reserved output LSE buffer, default
            None. Any non-None value raises NotImplementedError; the buffer
            is not allocated or written by this interface.

    Returns:
        None: Communication and accumulation post-processing are submitted.
        Returning does not establish GPU completion or return a work object.

    Notes:
        Only sum accumulation is implemented. Other reduction modes,
        overwrite mode, LSE and dtype conversion are rejected before
        submission. There is no async_op or kwargs parameter.
        Floating-point results can differ with backend/reduction grouping.

        grad_output row count must exactly match the route; grad_input may
        have unused trailing capacity. The dtype/stride/capacity rules of
        group_cast apply. Per-call payload formats and operation order must
        agree across ranks, including empty and relay-only ranks.

        Caller contract: order grad_input production and grad_output
        initialization before this stream, and make consumers on other
        streams wait for it. Use a live handle, one fixed native-slot stream
        and serialized host submissions. Retain tensors, handle and external
        streams until GPU completion. The caller supplies any required
        no_grad scope and autograd integration; the native fast path retains
        its C++ checks without additional Python row-count checks.
        This prepared entry adds no route exchange or metadata upload.
    """
    if handle._backend == "zero_cta":
        zero_cta.zero_cta_group_reduce_impl(
            grad_input, grad_output, handle._native, handle._group.nccl_group, async_op=True,
            reduce_op=reduce_op, acc_reduce=acc_reduce,
            comm_dtype=comm_dtype, input_lse=input_lse, output_lse=output_lse,
        ).wait_post_process(grad_output)
    else:
        if reduce_op != "sum":
            raise NotImplementedError("flat group-reduce supports only sum")
        if not acc_reduce:
            raise NotImplementedError("flat group-reduce requires acc_reduce=True")
        if comm_dtype is not None and comm_dtype != grad_input.dtype:
            raise NotImplementedError("flat group-reduce does not support dtype conversion")
        if input_lse is not None or output_lse is not None:
            raise NotImplementedError("flat group-reduce does not support LSE")
        _execute(handle.group, grad_input, grad_output, handle._route, handle, True, stream)


def _execute_explicit(input: torch.Tensor, output: torch.Tensor, routes: tuple[Route, ...],
                      group: CpGroup, stream: StreamSpec, *, reduce: bool,
                      async_op: bool = False) -> WorkWithPostProcessFn | None:
    """Prepare and submit one all-rank route snapshot without a public handle.

    Caller supplies identical global routes/configuration and compatible native
    prerequisites on every rank. No route/backend exchange is added here; a
    local native prerequisite failure raises instead of switching one rank.
    """
    process_group = _group(group)
    state, config = group._state, group.cp_config
    rank, world = dist.get_rank(process_group), len(routes)
    route = routes[rank]
    selected = _resolve_stream(state, stream)
    with state.lock, torch.no_grad(), (torch.cuda.stream(selected) if selected is not None else nullcontext()):
        _validate_tensors(input, output, route, process_group, reduce, state)
        if state.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Explicit route preparation must run outside CUDA graph capture")
        use_zero = (
            config.backend == "auto" and state.device.type == "cuda"
            and os.environ.get("NCCL_CP_DISABLE_ZERO_CTA") != "1"
            and config.dtype in (torch.float32, torch.bfloat16)
            and input.dtype in (torch.float32, torch.bfloat16)
            and config.max_per_token_bytes > 0
            and (world <= config.nvl_domain_size or world % config.nvl_domain_size == 0)
            and max(item.input_rows for item in routes) <= config.max_per_peer_slot
        )
        if not use_zero:
            if any(item.output_rows for item in routes) and prod(input.shape[1:]):
                send, recv = peer_segments(route, world)
                if reduce:
                    send, recv = recv, send
                if async_op:
                    return flat.run_async(input, output, send, recv, process_group, reduce=reduce)
                flat.run(input, output, send, recv, process_group, reduce=reduce)
            elif async_op:
                return WorkWithPostProcessFn(post_process_fn=lambda *_args, **_kwargs: output, async_op=True)
            return
        profile = _probe_zero(process_group)
        if not profile["ok"]:
            raise RuntimeError(
                f"Explicit zero-CTA prerequisites are unavailable: {profile['reason']}. "
                "Select backend='all2allv' consistently on all ranks to use the flat backend."
            )
        from .comm_meta import GroupCollectiveEntry, ZeroCTACollectiveArg

        entries = local_hierarchy_from_routes(rank, routes, config.nvl_domain_size)
        argument = ZeroCTACollectiveArg(
            input_split_size_list=list(route.input_splits),
            output_split_size_list=list(route.output_splits),
            dst_indices_list=[list(peers) for peers in route.destinations],
            src_index_list=list(route.sources), rank=rank, world_size=world,
            group=process_group,
            hierarchy_meta=tuple(GroupCollectiveEntry(
                entry.owner, entry.input_start, entry.n_tokens, entry.output_rank_starts
            ) for entry in entries),
            max_per_peer_slot=config.max_per_peer_slot,
            max_per_token_bytes=config.max_per_token_bytes,
            runtime_slot=config.runtime_slot, nvl_domain_size=config.nvl_domain_size,
        )
        state.runtimes[config.runtime_slot] = argument.runtime
        function = zero_cta.zero_cta_group_reduce_impl if reduce else zero_cta.zero_cta_group_cast_impl
        work = function(input, output, argument, process_group, async_op=True)
        if async_op:
            # The native work owns its active C++ plan until wait.
            # Completion must run on this same allocation stream before release.
            return work
        work.wait_post_process(output)
        # Metadata is allocated/uploaded on selected and the work wait
        # joins its NCCL-stream users back onto selected before we release it.
        # Stream-ordered allocator reuse therefore protects these temporary
        # plans without a host sync, Python event queue or persistent handle.
        # Caller-owned input/output must still live through GPU completion.


def group_cast_explicit(
    input: torch.Tensor, output: torch.Tensor,
    input_split_size_list: list[list[int]],
    output_split_size_list: list[list[int]],
    dst_indices_list: list[list[list[int]]],
    src_index_list: list[list[int]], *, group: CpGroup, stream: StreamSpec,
    cast_lse: bool = False, input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> None:
    """Group cast interface using the caller's complete all-rank route lists.

    Let W = group.nccl_group.size() and r be this process's group-relative
    rank. Every outer routing list contains W entries, indexed by rank.
    input/output are local tensors; the four routing arguments are global.

    Args:
        input (torch.Tensor): Local source tensor shaped
            [input_seqlen, *payload_shape]. Its logical rows are partitioned
            consecutively by input_split_size_list[r].
        output (torch.Tensor): Required, caller-allocated destination buffer
            shaped [output_seqlen, *payload_shape]. Must match input's dtype,
            device and trailing shape. Its storage is written directly.
            None and automatic output allocation are not supported.
        input_split_size_list (list[list[int]]): For every rank q,
            input_split_size_list[q][i] is the nonnegative Python-int row
            count of input segment i. Segment starts are prefix sums.
            sum(input_split_size_list[r]) == input.shape[0].
        output_split_size_list (list[list[int]]): For every rank q,
            output_split_size_list[q][j] is the nonnegative Python-int row
            count of output segment j, in tensor row order.
            sum(output_split_size_list[r]) <= output.shape[0] is required.
            Only output[:sum(output_split_size_list[r])] is written; the
            right-padded capacity rows are ignored and left unchanged. The
            buffer is not resized or replaced, and no padding copy is added.
        dst_indices_list (list[list[list[int]]]): Destination ranks for each
            input segment of every rank q:
            len(dst_indices_list[q]) == len(input_split_size_list[q]).
            dst_indices_list[q][i] contains distinct group-relative ranks
            in [0, W). An empty list means that input segment is not sent.
            Rank indices are Python ints; -1 padding is not supported.
        src_index_list (list[list[int]]): One source rank per output segment
            of every rank q:
            len(src_index_list[q]) == len(output_split_size_list[q]).
            src_index_list[q][j] is a Python int in [0, W). Segments from the
            same source preserve that source's input order. A sentinel W
            (or -1) is invalid, including for a zero-length output segment;
            there is no padded-rank-list convention.
        group (CpGroup): Configured group returned by create_group, containing
            the caller's ProcessGroup and CpConfig. Rank indices refer to
            that ProcessGroup, not necessarily WORLD. The config supplies
            payload byte capacity, token capacity, NVL domain size, runtime
            slot and backend policy. A raw ProcessGroup is not accepted here.
        stream (StreamSpec): torch.cuda.Stream or a nonnegative raw CUDA
            stream pointer on the group's device; 0 selects the default
            stream. CPU/Gloo verification uses None. This explicit entry
            makes the selected stream current for preparation/submission,
            then restores the previous stream. The caller owns external
            streams and orders data producers/consumers around this stream.
        cast_lse (bool): Reserved LSE-cast switch, default False. True raises
            NotImplementedError; LSE communication is not implemented.
        input_lse (torch.Tensor | None): Reserved input LSE tensor, default
            None. Any non-None value raises NotImplementedError, even when
            cast_lse is False; its shape/dtype are not inspected.
        output_lse (torch.Tensor | None): Reserved output LSE buffer, default
            None. Any non-None value raises NotImplementedError; the buffer
            is not allocated or written by this interface.

    Returns:
        None: Communication and post-processing are submitted into the
        supplied output. GPU completion is not implied; no work is returned.

    Notes:
        Routing arguments are host Python lists, not torch.Tensor metadata.
        Each sender/receiver pair must describe the same token count and
        preserve source order. The supplied global descriptions are checked
        locally, but their agreement with peer calls and tensor contents is
        the caller's responsibility. input row count is exact; output row
        count may be an upper bound. The valid prefix comes from host route
        metadata, without reading a token count back from the GPU. Capacity
        padding is separate from zero-CTA's allowed token-axis stride padding;
        its trailing dimensions must remain contiguous.

        No handle is accepted. Each call computes only this rank's direct/
        hierarchical plan and uploads metadata, then submits communication.
        There is no route AllGather or backend-consensus collective. Native
        first-use initialization/rendezvous still applies; workspace
        is reused until close_runtime. Preparation must be outside capture.
        Temporary plans follow the selected stream's allocator lifetime;
        caller-owned input/output must remain alive until GPU completion.

        Caller contract: supply identical global lists/CpConfig/backend policy,
        matching operation order and compatible per-row shape/dtype on all
        ranks. NVL domain placement and peer access must be valid. Use one
        fixed stream per native runtime slot and serialize submissions with
        preparation/teardown. A local native prerequisite failure raises;
        coordinate recovery or all2allv selection across the group.
        zero-CTA supports float32/bfloat16; the flat backend also supports
        float16/float64. Input/output are not converted to another dtype.
        This entry runs under no_grad and provides no automatic backward.

        LSE arguments are reserved and rejected when requested, before data
        submission. There is no async_op or kwargs option.
        For reusable plans without per-call preparation, use create_handle
        followed by group_cast instead.
    """
    # Reject unsupported LSE before planning, uploads or runtime initialization.
    if cast_lse or input_lse is not None or output_lse is not None:
        raise NotImplementedError("group-cast does not support LSE")
    if not isinstance(group, CpGroup):
        raise TypeError("group must be a CpGroup returned by create_group")
    world = dist.get_world_size(_group(group))
    routes = supplied_routes(input_split_size_list, output_split_size_list,
                             dst_indices_list, src_index_list, world)
    _execute_explicit(input, output, routes, group, stream, reduce=False)


def group_reduce_explicit(
    input: torch.Tensor, output: torch.Tensor,
    input_split_size_list: list[list[int]],
    output_split_size_list: list[list[int]],
    dst_index_list: list[list[int]],
    src_indices_list: list[list[list[int]]], *, group: CpGroup, stream: StreamSpec,
    reduce_op: GroupReduceOp = "sum", acc_reduce: bool = True,
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> None:
    """Group reduce interface using global reverse-route lists; sum and accumulate.

    Let W = group.nccl_group.size() and r be this process's group-relative
    rank. Every outer routing list contains W entries. input/output are local
    tensors; all four routing lists describe the complete group.

    Args:
        input (torch.Tensor): Local contributions shaped
            [input_seqlen, *payload_shape], with its valid prefix partitioned
            consecutively by input_split_size_list[r]. Contributions to the same owner keep
            their order so they match that owner's output segments.
        output (torch.Tensor): Required, caller-allocated accumulator shaped
            [output_seqlen, *payload_shape]. Must have the same dtype, device
            and trailing shape as input. Each output segment accumulates the
            sum from its listed sources into its existing values. Segments
            with no sources remain unchanged. Zero output before calling for
            a plain sum. None, output allocation and overwrite mode are absent.
        input_split_size_list (list[list[int]]): For every rank q,
            input_split_size_list[q][i] is the nonnegative Python-int row
            count of input segment i. Segment starts are prefix sums.
            sum(input_split_size_list[r]) <= input.shape[0] is required.
            Only the described prefix is read; extra trailing input rows are
            ignored and cannot contribute to the reduction.
        output_split_size_list (list[list[int]]): For every rank q,
            output_split_size_list[q][j] is the nonnegative Python-int row
            count of accumulator segment j, in tensor row order.
            sum(output_split_size_list[r]) == output.shape[0].
        dst_index_list (list[list[int]]): One destination/owner rank per input
            segment of every rank q:
            len(dst_index_list[q]) == len(input_split_size_list[q]).
            dst_index_list[q][i] is a Python int in [0, W). Segments sent to
            the same owner preserve their source order. Sentinel W and -1
            are invalid even when the corresponding input split has size 0.
        src_indices_list (list[list[list[int]]]): Contribution source ranks
            for each output segment of every rank q:
            len(src_indices_list[q]) == len(output_split_size_list[q]).
            src_indices_list[q][j] contains distinct Python-int ranks in
            [0, W); an empty list leaves that output segment unchanged.
            Tensor metadata and -1-padded rank lists are not supported.
            Source enumeration does not promise a bitwise reduction order;
            floating-point results can differ with reduction grouping/backend.
        group (CpGroup): Configured group returned by create_group. Its
            nccl_group defines the rank numbering; its CpConfig provides
            payload/token capacity, NVL domain size, runtime slot and backend
            policy. Pass CpGroup, not a raw ProcessGroup.
        stream (StreamSpec): torch.cuda.Stream or nonnegative raw CUDA stream
            pointer on this rank's fixed device; 0 selects the default stream.
            CPU/Gloo verification uses None. Preparation, communication and
            post-processing use this stream, then the previous stream is
            restored. The caller orders input production, output initialization
            and later consumers; external streams stay caller-owned.
        reduce_op (GroupReduceOp): Reserved reduction selector, default
            "sum". Any other value raises NotImplementedError.
        acc_reduce (bool): Accumulate into the supplied output, default
            True. False raises NotImplementedError; zero the output
            before calling when a plain sum is required.
        comm_dtype (torch.dtype | None): Communication dtype, default None
            (use input.dtype). An explicit matching dtype is accepted;
            any different dtype raises NotImplementedError. No conversion
            or separate communication buffer is created.
        input_lse (torch.Tensor | None): Reserved input LSE tensor, default
            None. Any non-None value raises NotImplementedError, including
            an empty tensor; its shape/dtype are not inspected.
        output_lse (torch.Tensor | None): Reserved output LSE buffer, default
            None. Any non-None value raises NotImplementedError; the buffer
            is not allocated or written by this interface.

    Returns:
        None: Communication and sum accumulation are submitted. Returning
        does not establish GPU completion or expose WorkWithPostProcessFn.

    Notes:
        Only sum accumulation is implemented. Other reduction modes,
        overwrite mode, LSE and dtype conversion are rejected before
        submission. There is no async_op or kwargs parameter.
        All routing arguments are host Python lists. output row count is
        exact; input may have unused trailing capacity. The valid prefix is
        determined from host route metadata, without a GPU count readback.
        Rank-list sentinel padding remains unsupported.

        For the reverse of a cast, pass its output/input split lists as this
        function's input/output split lists, its src_index_list as dst_index_list,
        and its dst_indices_list as src_indices_list. Peer counts must pair,
        and each sender's order must match the receiver's segment order.

        No handle is accepted. Each call locally builds this rank's reverse
        direct/hierarchical plan and uploads metadata. No route exchange or
        backend-consensus collective is added. Native initialization
        may still communicate; workspace is reused until close_runtime.
        Preparation is outside capture and runs under no_grad. No automatic
        backward is registered. For prepared reuse, call group_reduce instead.

        Caller contract: provide identical global routes/configuration and
        matching calls/formats across ranks, valid NVL placement/peer access,
        and compatible native prerequisites. Use one fixed stream per native
        runtime slot and serialized host submissions. Local errors are not
        broadcast; recovery/backend changes must be coordinated by the caller.
        Retain input/output and external streams until GPU completion. The
        zero-CTA dtype/stride/capacity rules and flat dtype support described
        by group_cast_explicit also apply.
    """
    # Reject unsupported options before planning, uploads or runtime initialization.
    if reduce_op != "sum":
        raise NotImplementedError("group-reduce supports only sum")
    if not acc_reduce:
        raise NotImplementedError("group-reduce requires acc_reduce=True")
    if comm_dtype is not None and comm_dtype != input.dtype:
        raise NotImplementedError("group-reduce does not support dtype conversion")
    if input_lse is not None or output_lse is not None:
        raise NotImplementedError("group-reduce does not support LSE")
    if not isinstance(group, CpGroup):
        raise TypeError("group must be a CpGroup returned by create_group")
    world = dist.get_world_size(_group(group))
    routes = supplied_routes(output_split_size_list, input_split_size_list,
                             src_indices_list, dst_index_list, world)
    _execute_explicit(input, output, routes, group, stream, reduce=True)


def group_cast_async(
    handle: Handle, input: torch.Tensor, output: torch.Tensor, *, stream: StreamSpec,
    cast_lse: bool = False, input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> WorkWithPostProcessFn:
    """Launch a prepared cast and defer its output post-processing.

    Args:
        handle (Handle): A live prepared route, as for group_cast.
        input (torch.Tensor): Caller-owned input in the handle's input order.
        output (torch.Tensor): Required caller-owned buffer. Only its valid
            route prefix will be written; extra capacity rows stay unchanged.
        stream (StreamSpec): Same stream contract as group_cast. Native calls
            require this stream to be current already. Call wait_post_process
            on this same current stream after submitting independent work.
        cast_lse (bool): Reserved; True raises NotImplementedError.
        input_lse (torch.Tensor | None): Reserved; non-None raises.
        output_lse (torch.Tensor | None): Reserved; non-None raises.

    Returns:
        WorkWithPostProcessFn: The work adapter. Call its
        wait_post_process() exactly once to submit the completion dependency
        and post-processing; it returns the same caller-supplied output tensor.
        CUDA completion is not implied by either return. This is not a Python
        coroutine and does not need await.

    Notes:
        Launch -> independent computation -> wait_post_process -> consume.
        Output is not ready to consume before wait_post_process. A native
        runtime slot permits only one work whose post-processing is not yet
        submitted; the C++ pending-work check enforces this. Matching
        host submission order is still required across ranks/slots.
        Retain work until wait_post_process, and tensors/handle/external stream
        until GPU completion. Keep launch and wait on the same fixed stream;
        no automatic stream switching, completion queue or new events are added.
        group_cast's payload/capacity/LSE and caller-owned autograd rules apply.
        Prepared native launch adds no validation or metadata exchange.
    """
    if handle._backend == "zero_cta":
        return zero_cta.zero_cta_group_cast_impl(
            input, output, handle._native, handle._group.nccl_group, async_op=True,
            cast_lse=cast_lse, input_lse=input_lse, output_lse=output_lse,
        )
    if cast_lse or input_lse is not None or output_lse is not None:
        raise NotImplementedError("flat group-cast does not support LSE")
    return cast(WorkWithPostProcessFn, _execute(
        handle.group, input, output, handle._route, handle, False, stream, async_op=True,
    ))


def group_reduce_async(
    handle: Handle, grad_input: torch.Tensor, grad_output: torch.Tensor, *, stream: StreamSpec,
    reduce_op: GroupReduceOp = "sum", acc_reduce: bool = True,
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> WorkWithPostProcessFn:
    """Launch prepared sum-reduce; defer accumulation until wait_post_process.

    Args:
        handle (Handle): A live cast route used in reverse.
        grad_input (torch.Tensor): Contributions in the cast output order;
            any trailing capacity rows beyond the route are ignored.
        grad_output (torch.Tensor): Required accumulator in the cast input
            order. Initialize it before the call; zero it for a plain sum.
        stream (StreamSpec): The caller's fixed launch/completion stream,
            already current for native calls. CPU/Gloo uses None.
        reduce_op (GroupReduceOp): Reserved reduction selector, default
            "sum". Any other value raises NotImplementedError.
        acc_reduce (bool): Accumulate into the supplied output, default
            True. False raises NotImplementedError; zero the output
            before calling when a plain sum is required.
        comm_dtype (torch.dtype | None): Communication dtype, default None
            (use grad_input.dtype). An explicit matching dtype is accepted;
            any different dtype raises NotImplementedError. No conversion
            or separate communication buffer is created.
        input_lse (torch.Tensor | None): Reserved input LSE tensor, default
            None. Any non-None value raises NotImplementedError, including
            an empty tensor; its shape/dtype are not inspected.
        output_lse (torch.Tensor | None): Reserved output LSE buffer, default
            None. Any non-None value raises NotImplementedError; the buffer
            is not allocated or written by this interface.

    Returns:
        WorkWithPostProcessFn: Call wait_post_process() once, on the same
        current stream, before consuming grad_output or reusing the native
        slot. It returns grad_output after submitting accumulation, not after
        waiting for GPU completion.

    Notes:
        The work/pending-slot checks and group_reduce semantics apply.
        Keep work until post-processing is submitted, and retain tensors,
        handle and stream through GPU completion. Launch/completion must share
        their current stream. No new native checks, events or control traffic
        are added; the caller may insert independent computation before wait.
    """
    if handle._backend == "zero_cta":
        return zero_cta.zero_cta_group_reduce_impl(
            grad_input, grad_output, handle._native, handle._group.nccl_group, async_op=True,
            reduce_op=reduce_op, acc_reduce=acc_reduce,
            comm_dtype=comm_dtype, input_lse=input_lse, output_lse=output_lse,
        )
    if reduce_op != "sum":
        raise NotImplementedError("flat group-reduce supports only sum")
    if not acc_reduce:
        raise NotImplementedError("flat group-reduce requires acc_reduce=True")
    if comm_dtype is not None and comm_dtype != grad_input.dtype:
        raise NotImplementedError("flat group-reduce does not support dtype conversion")
    if input_lse is not None or output_lse is not None:
        raise NotImplementedError("flat group-reduce does not support LSE")
    return cast(WorkWithPostProcessFn, _execute(
        handle.group, grad_input, grad_output, handle._route, handle, True, stream, async_op=True,
    ))


def group_cast_explicit_async(
    input: torch.Tensor, output: torch.Tensor,
    input_split_size_list: list[list[int]],
    output_split_size_list: list[list[int]],
    dst_indices_list: list[list[list[int]]],
    src_index_list: list[list[int]], *, group: CpGroup, stream: StreamSpec,
    cast_lse: bool = False, input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> WorkWithPostProcessFn:
    """Prepare all-rank lists and launch cast without its final output gather.

    Tensor, global-list, CpGroup, stream and reserved LSE arguments have the
    same meanings as group_cast_explicit. No handle is accepted. Preparation
    and metadata upload occur at launch, with no route AllGather. The returned
    WorkWithPostProcessFn retains the active native plan (or flat
    buffers) until its single wait_post_process() call.

    Make stream current when calling wait_post_process(); launch and completion
    must share this stream so temporary metadata uses its allocation stream's
    lifetime. The selected stream is restored after launch, as for the immediate
    explicit entry. Retain work until wait, and input/output/stream through GPU
    completion. Native slots still allow only one work awaiting post-processing.
    wait_post_process() returns the supplied output after submitting the final
    gather, without a CPU wait for GPU completion.
    """
    if cast_lse or input_lse is not None or output_lse is not None:
        raise NotImplementedError("group-cast does not support LSE")
    if not isinstance(group, CpGroup):
        raise TypeError("group must be a CpGroup returned by create_group")
    world = dist.get_world_size(_group(group))
    routes = supplied_routes(input_split_size_list, output_split_size_list,
                             dst_indices_list, src_index_list, world)
    return cast(WorkWithPostProcessFn, _execute_explicit(
        input, output, routes, group, stream, reduce=False, async_op=True,
    ))


def group_reduce_explicit_async(
    input: torch.Tensor, output: torch.Tensor,
    input_split_size_list: list[list[int]],
    output_split_size_list: list[list[int]],
    dst_index_list: list[list[int]],
    src_indices_list: list[list[list[int]]], *, group: CpGroup, stream: StreamSpec,
    reduce_op: GroupReduceOp = "sum", acc_reduce: bool = True,
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> WorkWithPostProcessFn:
    """Prepare global reverse lists and launch reduce with deferred accumulation.

    Arguments and sum/capacity rules match group_reduce_explicit; no handle is
    accepted. This entry returns the WorkWithPostProcessFn. Insert
    independent computation, then call wait_post_process() once with stream
    current. It submits final accumulation and returns the caller's output.
    Launch and completion use the same stream; retain work until wait and keep
    tensors/external stream alive through GPU completion. Native slots cannot
    accept another communication until the work has been waited.
    reduce_op, acc_reduce, comm_dtype and reserved LSE arguments follow
    group_reduce_explicit;
    unsupported requests raise before preparation.
    Preparation and metadata uploads occur at launch, never at completion.
    The global consistency/native prerequisite obligations of the immediate
    explicit entry still apply. There is no route/backend negotiation added.
    """
    # Reject unsupported options before planning, uploads or runtime initialization.
    if reduce_op != "sum":
        raise NotImplementedError("group-reduce supports only sum")
    if not acc_reduce:
        raise NotImplementedError("group-reduce requires acc_reduce=True")
    if comm_dtype is not None and comm_dtype != input.dtype:
        raise NotImplementedError("group-reduce does not support dtype conversion")
    if input_lse is not None or output_lse is not None:
        raise NotImplementedError("group-reduce does not support LSE")
    if not isinstance(group, CpGroup):
        raise TypeError("group must be a CpGroup returned by create_group")
    world = dist.get_world_size(_group(group))
    routes = supplied_routes(output_split_size_list, input_split_size_list,
                             src_indices_list, dst_index_list, world)
    return cast(WorkWithPostProcessFn, _execute_explicit(
        input, output, routes, group, stream, reduce=True, async_op=True,
    ))


def close_runtime(group: CpGroup | dist.ProcessGroup | None) -> None:
    """Drain submitted work and close all slots of the underlying ProcessGroup.

    Caller contract: stop submissions through every CP group sharing this
    ProcessGroup, then have all members enter teardown in matching order before
    destroying it. This closes sibling slots/groups too and may block the host
    while draining GPU work. It is not an abort/recovery primitive for a failed
    or missing peer. Deleting a handle/group alone does not perform this cleanup.
    The caller guarantees matching runtime creation/teardown across ranks;
    there is no control collective to check peer state before releasing locally
    registered slots. Native runtime draining/release still apply. The caller
    must finish flat-backend GPU work and retain its data/handles until then;
    CP does not track per-submission events or references in Python.
    """
    group = _group(group)
    state = _state(group)
    with state.lock, (torch.cuda.device(state.device) if state.device.type == "cuda" else nullcontext()):
        if state.runtimes:
            from .zero_cta import clear_zero_cta_cpp_state
            clear_zero_cta_cpp_state(group)
        state.runtimes.clear()
        state.closed = True
        with _states_lock:
            _states.pop(group, None)
