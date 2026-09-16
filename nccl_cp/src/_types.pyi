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

"""Static declarations for the native objects registered by NCCL CP."""

from typing import Literal, TypeAlias

GroupReduceOp: TypeAlias = Literal["sum", "avg", "lse"]


class ZeroCtaRuntime:
    """Communication resources shared by prepared routes in one runtime slot.

    Native type:
        C++ nccl_cp_zero_cta::ZeroCtaRuntime in zero_cta_collective.cpp.
        TorchBind torch.classes.nccl_cp_zero_cta_full.Runtime.

    Creation and lifetime:
        get_or_create_zero_cta_runtime(group, K, D, B, slot) returns the
        registry entry keyed by (ProcessGroup identity, runtime_slot).
        K = max_per_peer_slot, in tokens; D = NVL domain size, in ranks;
        B = max_per_token_bytes, in bytes. Reuse requires the same K and D,
        and the existing B must be at least the requested B.
        The ProcessGroup must use the NCCL ZERO CTA policy. Create the
        runtime before CUDA graph capture. For a nonempty route K > 0,
        B must be positive; K and slot are nonnegative, and D is positive.

    Native resources (not exposed as Python attributes):
        group: intrusive_ptr<ProcessGroup>; nccl_backend: ProcessGroupNCCL*.
        rank/world_size: rank and size within that ProcessGroup.
        comm: ncclComm_t and comm_stream: CUDAStream from the NCCL backend.
        max_per_peer_slot_value/nvl_domain_size_value/
        max_per_token_bytes_value/runtime_slot_value: int64_t configuration.
        workspace: optional Workspace owning byte tensors send_bytes and
            recv_bytes, symmetric-memory handles, device peer-pointer
            tables, and the registered receive window plus its offset.
        signal_state: SharedSignalState shared by slots in this group;
            contains a mutex and queued completion/reclaim records.
        mutex: serializes operations on this runtime.
        peer_post_process_peers: ncclWaitSignalDesc_t entries for every
            other rank, with opCnt=1.
        gpu_completion_markers_enabled: optional profiling flag.
        collective_pending/closed: work-submission and shutdown state.

    Workspace capacities:
        Let W = world_size. Direct mode is W <= D:
            send_tokens = recv_tokens = W * K.
        Hierarchical mode is W > D and requires W % D == 0. Let N = W // D:
            send_tokens = (W + N) * K; recv_tokens = N * K.
        Each byte buffer reserves its token capacity multiplied by B.
        These are capacity bounds, not the current payload size.

    One slot requires waiting the previous work before another operation.
    Deleting a route object does not close the registry-owned runtime.
    clear_zero_cta_cpp_state(group) closes all of that group's slots after
    their work is waited, before the caller destroys the ProcessGroup.
    The five methods below are those registered for Runtime by this extension.
    """

    def max_per_peer_slot(self) -> int:
        """Return K, the token capacity of each peer/node slot."""
        ...

    def node_slot_capacity(self) -> int:
        """Return K in hierarchical mode, otherwise 0."""
        ...

    def nvl_domain_size(self) -> int:
        """Return D, the configured number of ranks per NVL domain."""
        ...

    def max_per_token_bytes(self) -> int:
        """Return B, the reserved byte capacity per token."""
        ...

    def runtime_id(self) -> int:
        """Return this process's native object identity, not rank or slot."""
        ...


class ZeroCtaPlan:
    """Prepared cast or sum-reduce route metadata owned by C++.

    Native type:
        C++ nccl_cp_zero_cta::ZeroCtaPlan in zero_cta_collective.cpp.
        TorchBind torch.classes.nccl_cp_zero_cta_full.Plan.

    Both cast_plan and reduce_plan use this same native type. The cast
    factory sets post_gather_tiles; the reduce factory sets post_reduce.
    C++ selects reduce when post_reduce.has_value() is true.
    Plan has no registered Python constructor, methods or field accessors.
    The fields below describe C++ storage, not Python-accessible attributes.

    Complete native field schema:
        send_token_counts: vector<int64_t>, length W (group size).
            Entry p is the actual number of tokens sent to rank p.
        remote_data_waits: vector<ncclWaitSignalDesc_t>.
            Converted from remote_wait_peers; each entry has peer=p,
            opCnt=1 and describes a remote-data readiness dependency.
        pack_gather_tiles: CUDA int64 Tensor[P, 3].
            Rows are (src_token_start, dst_token_start, n_tokens).
        post_gather_tiles: optional CUDA int64 Tensor[G, 3 or 4].
            Cast only. Direct rows use the same 3-column gather format.
            Hierarchical rows prepend src_rank for a same-domain peer.
        post_reduce: optional PostReduceMetadata.
            reduce_ranges: CUDA int64 Tensor[R, 5], with rows
                (dst_token_start, first_src_index, n_srcs,
                 src_range_token_offset, n_tokens).
            src_token_offsets: CUDA int64 Tensor[S], indexing source
                token starts in the receive buffer.
        local_reduce: optional LocalReduceMetadata.
            Used for hierarchical reduce before remote transfer.
            reduce_ranges: CUDA int64 Tensor[L, 5], same range format.
            symmetric_srcs: CUDA int64 Tensor[T, 2], with rows
                (src_rank, src_token_start) for same-domain peers.
                The current builder represents an empty source list as
                Tensor[0]; it is unused when there are no local ranges.
        local_ready_signals/local_ready_waits:
            vector<ncclWaitSignalDesc_t>, created from peer lists with
            opCnt=1, for intra-domain relay readiness.

    All ranks are ProcessGroup-relative. src_token_start, dst_token_start,
    src_range_token_offset, n_tokens and send_token_counts use token units.
    first_src_index and n_srcs index/count source metadata records. Signal
    opCnt counts signal increments. Gather rows are split into tiles of at most
    32 tokens; reduce segments are expanded into ranges of at most 128 tokens.
    Exact column constants are defined in zero_cta_kernels.h.

    Plans contain route metadata, not input/output data or a workspace.
    Reuse with the compatible runtime that was prepared with the route.
    Metadata upload runs on the construction stream; the caller establishes
    a dependency before first use on another stream. The native work keeps
    the plan, runtime and input/output tensors alive while it owns them.
    """
