# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import InitVar, dataclass, field
from itertools import accumulate
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from ._types import ZeroCtaPlan, ZeroCtaRuntime
from ._utils import _make_device_tensor

@dataclass(frozen=True)
class GroupCollectiveEntry:
    """One logical owner-to-consumer segment for hierarchical collectives.

    Offsets are measured in tokens in the owner/consumer collective tensors,
    not in the symmetric communication workspace.
    """

    owner: int
    input_start: int
    n_tokens: int
    output_rank_starts: tuple[tuple[int, int], ...]


@dataclass(repr=False)
class GroupCollectiveArg:
    """The basic comm args for group collective implementations.

    Internal caller contract: ranks are group-relative and lists describe the
    supplied tensors' row order. Matching peer counts/order are prepared before
    execution; this object does not negotiate or verify peer tensor contents.
    """

    input_split_size_list: list[int]
    output_split_size_list: list[int]
    dst_indices_list: list[list[int]]
    src_index_list: list[int]

    rank: int
    world_size: int
    group: dist.ProcessGroup
    device_mesh: DeviceMesh | None = None

    deterministic: bool = False

    split_alignment: int = 1

    hierarchy_meta: tuple[GroupCollectiveEntry, ...] | None = None

    def __post_init__(self):
        pass

    def to_group_cast_args(self) -> dict:
        return self.to_packed_group_cast_args()

    def to_group_reduce_args(self) -> dict:
        return self.to_packed_group_reduce_args()

    def to_packed_group_cast_args(self, packed_times: int = 1) -> dict:
        # pack args along split dim by `packed_times` times
        return dict(
            input_split_sizes=self.input_split_size_list * packed_times,
            output_split_sizes=self.output_split_size_list * packed_times,
            dst_indices=self.dst_indices_list * packed_times,
            src_index=self.src_index_list * packed_times,
        )

    def to_packed_group_reduce_args(self, packed_times: int = 1) -> dict:
        # symmetric to group-cast
        # pack args along split dim by `packed_times` times
        return dict(
            input_split_sizes=self.output_split_size_list * packed_times,
            output_split_sizes=self.input_split_size_list * packed_times,
            dst_index=self.src_index_list * packed_times,
            src_indices=self.dst_indices_list * packed_times,
        )

    def compute_send_recv_token_counts(
        self, reduce_op: Literal["sum", "max"] = "max"
    ) -> None:
        self._compute_group_cast_send_recv_token_counts(reduce_op=reduce_op)
        self._compute_group_reduce_send_recv_token_counts(reduce_op=reduce_op)

    def _compute_group_cast_send_recv_token_counts(
        self, reduce_op: Literal["sum", "max"] = "max"
    ) -> None:
        group_cast_args = self.to_group_cast_args()

        # calculate for group cast
        cast_input_split_size_list: list[int] = group_cast_args["input_split_sizes"]
        cast_output_split_size_list: list[int] = group_cast_args["output_split_sizes"]
        cast_dst_indices_list: list[list[int]] = group_cast_args["dst_indices"]

        cast_send_tokens = sum(
            [
                split_size * len(dst_indice)
                for split_size, dst_indice in zip(
                    cast_input_split_size_list, cast_dst_indices_list
                )
            ]
        )
        cast_recv_tokens = sum(cast_output_split_size_list)

        self.group_cast_comm_tokens = self._reduce_send_recv_tokens(
            cast_send_tokens, cast_recv_tokens, reduce_op
        )

    def _compute_group_reduce_send_recv_token_counts(
        self, reduce_op: Literal["sum", "max"] = "max"
    ) -> None:
        group_reduce_args = self.to_group_reduce_args()

        # calculate for group reduce
        reduce_input_split_size_list: list[int] = group_reduce_args["input_split_sizes"]
        reduce_output_split_size_list: list[int] = group_reduce_args[
            "output_split_sizes"
        ]
        reduce_src_indices_list: list[list[int]] = group_reduce_args["src_indices"]

        reduce_send_tokens = sum(reduce_input_split_size_list)
        reduce_recv_tokens = sum(
            [
                split_size * len(dst_indice)
                for split_size, dst_indice in zip(
                    reduce_output_split_size_list, reduce_src_indices_list
                )
            ]
        )

        self.group_reduce_comm_tokens = self._reduce_send_recv_tokens(
            reduce_send_tokens, reduce_recv_tokens, reduce_op
        )

    def _reduce_send_recv_tokens(
        self,
        send_tokens: int,
        recv_tokens: int,
        reduce_op: Literal["sum", "max"] = "max",
    ) -> int:
        match reduce_op:
            case "sum":
                return send_tokens + recv_tokens
            case "max":
                return max(send_tokens, recv_tokens)
            case _:
                raise ValueError(f"Invalid reduce_op: {reduce_op}")

    def __repr__(self) -> str:  # pragma: no cover
        indent = ""
        repr_str = "GroupCollectiveArg(\n"

        repr_str += f"{indent}    rank={self.rank},\n"
        repr_str += f"{indent}    world_size={self.world_size},\n"
        repr_str += f"{indent}    device_mesh={repr(self.device_mesh)},\n"
        repr_str += f"{indent}    deterministic={self.deterministic},\n"
        repr_str += f"{indent}    split_alignment={self.split_alignment},\n"

        repr_str += f"{indent}    input_split_size_list={self.input_split_size_list},\n"
        repr_str += (
            f"{indent}    output_split_size_list={self.output_split_size_list},\n"
        )
        repr_str += f"{indent}    dst_indices_list={self.dst_indices_list},\n"
        repr_str += f"{indent}    src_index_list={self.src_index_list},\n"

        repr_str = repr_str.rstrip(",\n") + "\n)"
        return repr_str


@dataclass(repr=False, kw_only=True)
class ZeroCTACollectiveArg(GroupCollectiveArg):
    """Prepared NCCL zero-CTA group-cast and group-reduce metadata.

    Hierarchy planning and metadata upload happen once during construction. The
    collective hot path receives this object through ``zero_cta_arg`` and only
    launches the prebuilt device work metadata. The first pack must use this
    construction stream or a stream that the caller made dependent on it.

    Direct internal callers provide validated split/peer lists and compatible
    hierarchy_meta, group and capacities. In hierarchical mode the metadata
    must retain the relevant owner/node prefixes; dropping prefix-only entries
    changes receive offsets. Do not mutate a prepared argument or its metadata
    while it is in use. Public create_handle prepares these inputs on the
    caller's fixed slot stream; the caller preserves stream order and lifetime.
    """

    max_per_peer_slot: InitVar[int]
    max_per_token_bytes: InitVar[int]
    runtime_slot: InitVar[int] = 0
    nvl_domain_size: InitVar[int | None] = None

    # Complete native types and storage schemas are declared in _types.pyi.
    # Shared communication resources for (ProcessGroup, runtime_slot): NCCL
    # communicator/stream, symmetric workspace, capacity and reclaim state.
    # The C++ registry owns its lifetime; several route objects may reuse it.
    runtime: ZeroCtaRuntime = field(init=False, repr=False)

    # Forward route: input -> pack -> peer transfers -> gather into output.
    # Contains counts, peer dependencies and uploaded gather metadata.
    cast_plan: ZeroCtaPlan = field(init=False, repr=False)

    # Reverse route: grad_input -> pack/local reduction -> peer transfers ->
    # sum into grad_output. Contains uploaded reduction ranges/source offsets.
    reduce_plan: ZeroCtaPlan = field(init=False, repr=False)

    def __post_init__(
        self,
        max_per_peer_slot: int,
        max_per_token_bytes: int,
        runtime_slot: int,
        nvl_domain_size: int | None,
    ) -> None:
        from nccl.cp.zero_cta import (
            get_or_create_zero_cta_runtime,
        )

        super().__post_init__()

        input_splits = list(map(int, self.input_split_size_list))
        output_splits = list(map(int, self.output_split_size_list))
        if nvl_domain_size is None:
            nvl_domain_size = int(os.environ.get("NVL_DOMAIN_SIZE", "8"))
        if self.world_size > nvl_domain_size:
            cast_plan, reduce_plan = self._build_hierarchical_meta(
                max_per_peer_slot, nvl_domain_size
            )
        else:
            cast_plan, reduce_plan = self._build_direct_meta(
                input_splits, output_splits, max_per_peer_slot
            )
        self.runtime = get_or_create_zero_cta_runtime(
            self.group,
            max_per_peer_slot,
            nvl_domain_size,
            max_per_token_bytes,
            runtime_slot,
        )

        self.cast_plan = cast_plan
        self.reduce_plan = reduce_plan

    def _build_direct_meta(
        self,
        input_splits: list[int],
        output_splits: list[int],
        max_per_peer_slot: int,
    ) -> tuple[ZeroCtaPlan, ZeroCtaPlan]:
        from nccl.cp.zero_cta import (
            _make_device_reduce_ranges as make_reduce_ranges,
            _make_device_tensor_gather_tiles as make_tensor_gather_tiles,
            create_zero_cta_cast_plan,
            create_zero_cta_reduce_plan,
        )

        world = self.world_size
        send_token_counts, recv_token_counts = [0] * world, [0] * world
        for tokens, destinations in zip(input_splits, self.dst_indices_list):
            for peer in destinations:
                send_token_counts[peer] += tokens
        for tokens, peer in zip(output_splits, self.src_index_list):
            recv_token_counts[peer] += tokens
        if max((*send_token_counts, *recv_token_counts), default=0) > max_per_peer_slot:
            raise ValueError("zero-CTA route exceeds its global token capacity")

        input_starts = list(accumulate(input_splits, initial=0))
        output_starts = list(accumulate(output_splits, initial=0))
        cast_pack_segments, cast_post_gather_segments = [], []
        reduce_pack_segments = []
        cursor = [0] * world
        for peer in range(world):
            for start, tokens, destinations in zip(
                input_starts, input_splits, self.dst_indices_list
            ):
                if tokens and peer in destinations:
                    packed = peer * max_per_peer_slot + cursor[peer]
                    cast_pack_segments.append((start, packed, tokens))
                    cursor[peer] += tokens
        cursor = [0] * world
        for start, tokens, peer in zip(
            output_starts, output_splits, self.src_index_list
        ):
            packed = peer * max_per_peer_slot + cursor[peer]
            if tokens:
                cast_post_gather_segments.append((packed, start, tokens))
                reduce_pack_segments.append((start, packed, tokens))
            cursor[peer] += tokens

        post_reduce_segments = []
        post_reduce_src_token_offsets: list[int] = []
        cursor = [0] * world
        for start, tokens, consumers in zip(
            input_starts, input_splits, self.dst_indices_list
        ):
            first = len(post_reduce_src_token_offsets)
            for peer in consumers:
                post_reduce_src_token_offsets.append(
                    peer * max_per_peer_slot + cursor[peer]
                )
                cursor[peer] += tokens
            if consumers:
                post_reduce_segments.append((start, first, len(consumers), tokens))

        return (
            create_zero_cta_cast_plan(
                send_token_counts=tuple(send_token_counts),
                remote_wait_peers=tuple(
                    i for i, n in enumerate(recv_token_counts) if i != self.rank and n
                ),
                pack_gather_tiles=make_tensor_gather_tiles(cast_pack_segments),
                post_gather_tiles=make_tensor_gather_tiles(
                    cast_post_gather_segments
                ),
            ),
            create_zero_cta_reduce_plan(
                send_token_counts=tuple(recv_token_counts),
                remote_wait_peers=tuple(
                    i for i, n in enumerate(send_token_counts) if i != self.rank and n
                ),
                pack_gather_tiles=make_tensor_gather_tiles(reduce_pack_segments),
                post_reduce_ranges=make_reduce_ranges(post_reduce_segments),
                post_reduce_src_token_offsets=_make_device_tensor(
                    post_reduce_src_token_offsets,
                    dtype=torch.int64,
                ),
            ),
        )

    def _build_hierarchical_meta(
        self,
        max_per_peer_slot: int,
        nvl: int,
    ) -> tuple[ZeroCtaPlan, ZeroCtaPlan]:
        from nccl.cp.zero_cta import (
            _make_device_reduce_ranges as make_reduce_ranges,
            _make_device_symmetric_gather_tiles as make_symmetric_gather_tiles,
            _make_device_tensor_gather_tiles as make_tensor_gather_tiles,
            create_zero_cta_cast_plan,
            create_zero_cta_reduce_plan,
        )

        if nvl <= 0 or self.world_size % nvl:
            raise ValueError("NVL_DOMAIN_SIZE must divide the zero-CTA world size")
        if self.hierarchy_meta is None:
            raise ValueError("hierarchical zero-CTA requires hierarchy metadata")

        hierarchy_meta = self.hierarchy_meta
        world = self.world_size
        num_nodes = world // nvl
        node_offsets: dict[tuple[int, int], int] = {}
        node_sizes: dict[tuple[int, int], int] = {}

        def nodes_for(entry: GroupCollectiveEntry) -> list[int]:
            return sorted({consumer // nvl for consumer, _ in entry.output_rank_starts})

        for entry_id, entry in enumerate(hierarchy_meta):
            for node in nodes_for(entry):
                key = (entry.owner, node)
                node_offsets[(entry_id, node)] = node_sizes.get(key, 0)
                node_sizes[key] = node_sizes.get(key, 0) + entry.n_tokens

        local_node = self.rank // nvl
        local_lane = self.rank % nvl
        node_capacity = max_per_peer_slot

        cast_pack_segments: list[tuple[int, int, int]] = []
        for entry_id, entry in enumerate(hierarchy_meta):
            if entry.owner != self.rank:
                continue
            for node in nodes_for(entry):
                cast_pack_segments.append(
                    (
                        entry.input_start,
                        node * node_capacity + node_offsets[(entry_id, node)],
                        entry.n_tokens,
                    )
                )
        cast_send_token_counts = [
            (
                node_sizes.get((self.rank, peer // nvl), 0)
                if peer % nvl == local_lane
                else 0
            )
            for peer in range(world)
        ]

        cast_post_gather_segments: list[tuple[int, int, int, int]] = []
        reduce_pack_segments: list[tuple[int, int, int]] = []
        local_relays: set[int] = set()
        for entry_id, entry in enumerate(hierarchy_meta):
            for consumer, output_start in entry.output_rank_starts:
                if consumer != self.rank:
                    continue
                relay = local_node * nvl + entry.owner % nvl
                if relay != self.rank:
                    local_relays.add(relay)
                recv_start = (
                    entry.owner // nvl * node_capacity
                    + node_offsets[(entry_id, local_node)]
                )
                cast_post_gather_segments.append(
                    (relay, recv_start, output_start, entry.n_tokens)
                )
                reduce_pack_segments.append(
                    (
                        output_start,
                        entry.owner * max_per_peer_slot + entry.input_start,
                        entry.n_tokens,
                    )
                )

        cast_wait_peers = [
            owner
            for owner in range(local_lane, world, nvl)
            if owner != self.rank and node_sizes.get((owner, local_node), 0)
        ]
        cast_signal_peers = {
            consumer
            for entry_id, entry in enumerate(hierarchy_meta)
            if entry.owner % nvl == local_lane
            and (entry_id, local_node) in node_offsets
            for consumer, _ in entry.output_rank_starts
            if consumer != self.rank and consumer // nvl == local_node
        }

        local_reduce_symmetric_srcs: list[tuple[int, int]] = []
        local_reduce_segments: list[tuple[int, int, int, int]] = []
        reduce_wait_consumers: set[int] = set()
        for entry_id, entry in enumerate(hierarchy_meta):
            key = (entry_id, local_node)
            if entry.owner % nvl != local_lane or key not in node_offsets:
                continue
            consumers = tuple(
                peer
                for peer, _ in entry.output_rank_starts
                if peer // nvl == local_node
            )
            first = len(local_reduce_symmetric_srcs)
            local_reduce_symmetric_srcs.extend(
                (peer, entry.owner * max_per_peer_slot + entry.input_start)
                for peer in consumers
            )
            reduce_wait_consumers.update(consumers)
            local_reduce_segments.append(
                (
                    entry.owner // nvl * node_capacity + node_offsets[key],
                    first,
                    len(consumers),
                    entry.n_tokens,
                )
            )
        reduce_send_token_counts = [
            (
                node_sizes.get((peer, local_node), 0)
                if peer % nvl == local_lane and peer != self.rank
                else 0
            )
            for peer in range(world)
        ]
        reduce_wait_consumers.discard(self.rank)

        post_reduce_segments: list[tuple[int, int, int, int]] = []
        post_reduce_src_token_offsets: list[int] = []
        reduce_wait_peers = [
            node * nvl + local_lane
            for node in range(num_nodes)
            if node * nvl + local_lane != self.rank
            and node_sizes.get((self.rank, node), 0)
        ]
        for entry_id, entry in enumerate(hierarchy_meta):
            if entry.owner != self.rank:
                continue
            source_nodes = nodes_for(entry)
            first_source = len(post_reduce_src_token_offsets)
            for node in source_nodes:
                post_reduce_src_token_offsets.append(
                    node * node_capacity + node_offsets[(entry_id, node)]
                )
            post_reduce_segments.append(
                (
                    entry.input_start,
                    first_source,
                    len(source_nodes),
                    entry.n_tokens,
                )
            )

        return (
            create_zero_cta_cast_plan(
                send_token_counts=tuple(cast_send_token_counts),
                remote_wait_peers=tuple(cast_wait_peers),
                pack_gather_tiles=make_tensor_gather_tiles(cast_pack_segments),
                post_gather_tiles=make_symmetric_gather_tiles(
                    cast_post_gather_segments
                ),
                local_ready_signal_peers=tuple(sorted(cast_signal_peers)),
                local_ready_wait_peers=tuple(sorted(local_relays)),
            ),
            create_zero_cta_reduce_plan(
                send_token_counts=tuple(reduce_send_token_counts),
                remote_wait_peers=tuple(reduce_wait_peers),
                pack_gather_tiles=make_tensor_gather_tiles(reduce_pack_segments),
                post_reduce_ranges=make_reduce_ranges(post_reduce_segments),
                post_reduce_src_token_offsets=_make_device_tensor(
                    post_reduce_src_token_offsets,
                    dtype=torch.int64,
                ),
                local_reduce_ranges=make_reduce_ranges(local_reduce_segments),
                local_reduce_symmetric_srcs=_make_device_tensor(
                    local_reduce_symmetric_srcs,
                    dtype=torch.int64,
                ),
                local_ready_signal_peers=tuple(sorted(local_relays)),
                local_ready_wait_peers=tuple(sorted(reduce_wait_consumers)),
            ),
        )

    def to_group_cast_args(self) -> dict:
        args = super().to_group_cast_args()
        args["zero_cta_arg"] = self
        return args

    def to_group_reduce_args(self) -> dict:
        args = super().to_group_reduce_args()
        args["zero_cta_arg"] = self
        return args
