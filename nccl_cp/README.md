# NCCL CP

NCCL CP provides context-parallel group cast and sum reduction through `nccl.cp`.
Prepare a route once, then reuse its handle for zero-CTA direct/hierarchical
communication or flat all2allv. The native library uses PyTorch, ProcessGroup
and TorchBind.

## Dependencies

| Component | Requirement | Purpose |
|---|---|---|
| Python | 3.10+ | Public API |
| PyTorch | 2.12+ with the required distributed interfaces | Tensors, ProcessGroup and symmetric memory |
| Linux / NVIDIA driver | Compatible with CUDA/PyTorch | GPU execution |
| CUDA toolkit / C++ compiler | Matching PyTorch's CUDA major/minor; C++17 | Native compilation |
| NCCL | 2.30+ for zero-CTA | Host RMA and collectives |
| CMake | 3.24+ | Build and installation |
| Make / Git | Available on the build host | NCCL dependency preparation |
| pytest | Required for local tests | CPU/API validation |

Zero-CTA requires PyTorch's `NCCLSymmetricMemory` interfaces, an NCCL
ProcessGroup with ZERO CTA policy, and peer-accessible GPUs within each NVL
domain. A version number alone does not establish compatibility with these
PyTorch internal interfaces. Build and run with the same Python/PyTorch/CUDA/NCCL
environment. CPU tests use Gloo.

## Build

Run from `nccl_cp/` in an NCCL Extensions checkout with the execution
Python/PyTorch environment active:

```bash
export CUDA_HOME="<cuda-toolkit-prefix>"
make nccl-submodule
export NCCL_HOME="$PWD/../third_party/nccl/build"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build -G "Unix Makefiles"
cmake --build build --parallel
```

`make nccl-submodule` delegates to the enclosing repository and builds its
shared pinned NCCL dependency. For an existing NCCL build, skip this
step and set `NCCL_HOME` to its prefix. Source archives require this external
dependency path. Use `lib64` in the library path when applicable.

CMake discovers Python/CUDA from the environment and checks NCCL headers against
`torch.cuda.nccl.version()`. Compiler-supported targets are selected by the build configuration;
`CUDAARCHS` or `CMAKE_CUDA_ARCHITECTURES` overrides selection.

| Artifact | Location |
|---|---|
| Native library and manifest | `build/lib/libnccl_cp.so`, `build/lib/libnccl_cp.build.json` |
| Staged Python package | `build/python/nccl/cp/` |
| Build-tree launcher | `build/bin/nccl_cp_run.py` |

For CPU tests or a flat-only package, configure `-DNCCL_CP_BUILD_NATIVE=OFF`.
To install the CMake artifacts:

```bash
cmake --install build --prefix /path/to/nccl-cp-install
python /path/to/nccl-cp-install/bin/nccl_cp_run.py /path/to/application.py
```

The launcher loads the package alongside other installed `nccl` extensions.
Keep the native library, its manifest and runtime dependencies together.

## Quick Start

[examples/basic.py](examples/basic.py) initializes the ProcessGroup, allocates
buffers, reuses a handle and checks cast/reduce results:

```bash
torchrun <launcher-options> \
  build/bin/nccl_cp_run.py examples/basic.py \
  --nvl-domain-size <ranks-per-domain>
```

Supply launcher options for process allocation and rendezvous. Set the domain
size from the communicator topology; the example does not select devices or
infer NVL domains from the number of processes on a host.

For an application with an initialized `nccl_group`, local layouts and allocated
input/output tensors:

```python
import torch
from nccl.cp import (
    CpConfig, create_group, create_handle, group_cast, group_reduce, close_runtime,
)

stream = torch.cuda.current_stream()
config = CpConfig(
    max_per_peer_slot=4096,
    payload_shape=(32, 128),  # Dimensions of one token, excluding axis 0.
    dtype=torch.bfloat16,
    nvl_domain_size=ranks_per_domain,  # Supplied by the application.
)
group = create_group(nccl_group, config)
handle = create_handle(group, local_input_layout, local_output_layout, stream=stream)

with torch.no_grad(), torch.cuda.stream(stream):
    group_cast(handle, input, output, stream=stream)
    grad_output.zero_()  # Omit when accumulating into existing values.
    group_reduce(handle, grad_input, grad_output, stream=stream)

stream.synchronize()  # Teardown boundary, outside the communication loop.
handle.close()
close_runtime(group)  # Before destroying the ProcessGroup.
```

The method forms `handle.group_cast(...)` and `handle.group_reduce(...)` have
the same tensor/stream arguments. Immediate APIs write the supplied output and
return `None`; they do not wait for GPU completion.

## Configuration and Layouts

`create_group(nccl_group, config)` binds a caller-owned ProcessGroup and immutable
configuration without communication. `None` selects WORLD. All ranks supply the
same configuration and keep their assigned CUDA device fixed. `CpGroup.nccl_group`
and `CpGroup.cp_config` expose those objects.

| `CpConfig` field | Configuration | Meaning |
|---|---|---|
| `max_per_peer_slot` | Required | Native peer-slot capacity C, in token rows |
| `payload_shape` | Required | Per-token dimensions; `()` describes a scalar |
| `dtype` | Required | Storage dtype used to size capacity |
| `max_per_token_bytes` | Derived | B = `prod(payload_shape) * dtype.itemsize` |
| `max_layout_tokens` | C | Owned-token bound per rank for handle metadata; independent of workspace capacity |
| `runtime_slot` | `0` | Workspace slot within the ProcessGroup |
| `nvl_domain_size` | Set explicitly for the communicator | Ranks per NVL domain |
| `backend` | `"auto"` | Prepare zero-CTA when eligible; `"all2allv"` selects flat |

For `(32, 128)` and BF16, B is 8192 bytes. Size C/B for the largest intended
traffic. Configuration does not convert tensors. A slot can reuse a larger byte
capacity, but token/domain capacities must match; use another slot for different
bounds.

### Handle input and output layouts

Both layouts are host-side `Sequence[int | TokenRange]` descriptions. Each integer
identifies one token occurrence; `TokenRange(start, stop)` describes the half-open
interval `[start, stop)` in ascending order. IDs are shared across the group and
identify token occurrences, not vocabulary entries. They carry no tensor data.

| Argument | What this rank declares | Tensor row mapping |
|---|---|---|
| `local_input_layout` | Tokens owned by this rank, in local input order | Cast input and reduce output |
| `local_output_layout` | Tokens this rank requests, in the desired result order | Valid cast output prefix and valid reduce input prefix |

`local_input_layout` describes **every row of the supplied cast input**, in
axis-0 order. Its expanded token count must equal both the cast input row count
and the reduce output row count. Each declared token ID belongs to exactly one
rank in the group; IDs can be nonconsecutive and need not be numerically sorted.
Include each input row even when no rank requests it. Rows unrequested by every
rank do not appear in cast outputs, and their reduce output values stay unchanged.

`local_output_layout` describes the **complete requested output**, including
remote tokens and any locally owned tokens that should appear in the result.
The caller does not need to know the remote owners: handle preparation resolves
them from the input layouts exchanged across the group.

For example, suppose this rank owns `[100, 101, 102]` and another owner declares
`[200, 201]` in that order:

```python
from nccl.cp import TokenRange

local_input_layout = [TokenRange(100, 103)]
local_output_layout = [100, 200, 101, 201]
```

Cast writes rows for tokens `100, 200, 101, 201` in exactly that order. Reduce
reads contributions in the same order and sends each back to its owner;
this rank's accumulator rows still correspond to `100, 101, 102`. If no rank
requests token `102`, its accumulator row is left unchanged.

- Every requested ID must have exactly one owner. Neither local layout may
  repeat an ID, including through overlapping ranges.
- Requests from the same owner must preserve that owner's declared input order;
  different owners may interleave. This is not a numeric sorting requirement.
  In the example, requesting `101` before `100` is invalid.
- The logical row count is the sum of interval lengths and individual IDs,
  not the number of layout entries. The example has three input rows and four
  output rows. Additional output buffer capacity is not listed in the layout.
- `local_input_layout=[]` declares no owned rows: cast input and reduce output
  have zero rows. The output layout may still request tokens owned by other ranks.
- `local_output_layout=[]` requests no rows. The rank still participates in
  matching calls and may have relay work.

The caller keeps actual tensor rows consistent with these descriptions.

`create_handle(group, local_input_layout, local_output_layout, *, stream)` exchanges
layouts and prepares this rank's route and relay work. Changed ownership, requests
or row order require a new handle. Preparation/uploads run outside CUDA Graph capture.

There is one fixed-size tensor AllGather containing both lengths and contents.
With W ranks and C = `max_layout_tokens`, each frame occupies
`8216 + 16*C*(1+W)` bytes, and every rank receives W frames. Owned rows are bounded
by C and requested rows by W*C. Endpoints fit signed int64. Even compact layouts
exchange the padded frame, so choose C deliberately and identically across ranks.
Each rank computes its own four lists and the relay metadata it needs.

## Tensor and Reduction Contract

Input/output are required caller-owned tensors with matching dtype, device and
per-token shape. CP never allocates or replaces the output buffer.

| Operation | Input rows | Output rows |
|---|---|---|
| Cast | Exactly the local input layout | At least the local output layout |
| Reduce | At least the local output layout | Exactly the local input layout |

Cast writes the valid output prefix and leaves trailing capacity unchanged.
Reduce ignores the input tail and **adds** contributions to existing output.
Zero output first for a plain sum. Unrequested owner rows retain their values.
Native callers guarantee adequate row capacity; flat/explicit paths validate it locally.

These keyword options apply to functions, handle methods and their async variants:

| API | Option | Supported value |
|---|---|---|
| Cast | `cast_lse` | `False` |
| Reduce | `reduce_op` | `"sum"` |
| Reduce | `acc_reduce` | `True` |
| Reduce | `comm_dtype` | `None` or the input dtype |
| Both | `input_lse`, `output_lse` | `None` |

Other values raise `NotImplementedError` before submission; explicit-list calls
reject them before preparation. LSE, dtype conversion, average reduction and
overwrite mode are not implemented. Parameter definitions are also documented
in [src/collectives.py](src/collectives.py).

## Explicit Global Lists

Callers with complete routing information can use the following signatures:

```python
def group_cast_explicit(
    input, output, input_split_size_list, output_split_size_list,
    dst_indices_list, src_index_list, *, group, stream,
    cast_lse=False, input_lse=None, output_lse=None,
) -> None: ...

def group_reduce_explicit(
    input, output, input_split_size_list, output_split_size_list,
    dst_index_list, src_indices_list, *, group, stream,
    reduce_op="sum", acc_reduce=True, comm_dtype=None,
    input_lse=None, output_lse=None,
) -> None: ...
```

Tensors are local. Each routing argument is a Python list with W outer entries,
indexed by group-relative rank r:

| Argument | Cast | Reduce |
|---|---|---|
| `input_split_size_list[r]` | Input segment lengths | Contribution segment lengths |
| `output_split_size_list[r]` | Output segment lengths | Accumulator segment lengths |
| Destinations | `dst_indices_list[r][i]`: distinct recipients of segment i | `dst_index_list[r][i]`: owner of contribution i |
| Sources | `src_index_list[r][j]`: owner of segment j | `src_indices_list[r][j]`: distinct contributors to segment j |

Starts are prefix sums; lengths are nonnegative integers; ranks lie in `[0, W)`.
Tensor metadata and sentinel rank padding are unsupported. Peer token counts
must pair and preserve within-source order. Tensor capacity rules apply to the
local split sums. Empty destination/source lists are allowed.

To reverse cast, swap its input/output split lists, use its source list as reduce
destinations and its destination lists as reduce sources. These APIs accept no
handle: each call prepares this rank's plan and uploads metadata without a route
AllGather. All ranks supply identical global routes, configuration and backend
policy. For repeated calls, prepare a handle from layouts instead.

## Streams, Async Completion and Lifetimes

`stream` accepts a CUDA stream object or raw pointer (`0` means the default
stream); CPU/Gloo uses `None`. Preparation, explicit-list and flat calls select
it internally. **Prepared zero-CTA calls require this stream to be current already**;
select it outside the submission loop. Raw stream pointers remain caller-owned.

`group_cast_async`, `group_reduce_async` and their handle methods return
`WorkWithPostProcessFn`. The `*_explicit_async` functions take the same arguments
as their immediate explicit counterparts.

```python
from nccl.cp import group_cast_async, group_reduce_async

with torch.no_grad(), torch.cuda.stream(stream):
    work = group_cast_async(handle, input, output, stream=stream)
    independent_compute()  # Must not consume output or modify CP buffers.
    work.wait_post_process()  # Submits completion dependencies and output gather.
    consume(output)

    work = group_reduce_async(handle, grad_input, grad_output, stream=stream)
    independent_backward_compute()
    work.wait_post_process()  # Submits accumulation into grad_output.
```

Call `wait_post_process()` exactly once on the same current stream as launch,
before consuming output or reusing the native slot. It returns the supplied
output after submitting post-processing; CUDA completion is not implied.
A native slot permits only one work awaiting post-processing. Discarding work
does not complete it. These are not Python coroutines.

Caller responsibilities:

- Match operation order, routes and per-token formats across ranks, including
  empty/relay-only ranks. CP does not negotiate them in data calls.
- Keep one fixed stream per native slot and serialize submissions with preparation
  and teardown. Order producers before communication and consumers after completion;
  explicitly establish dependencies for other streams.
- Retain work until `wait_post_process`, and tensors, handles and external streams
  until GPU completion. Do not resize/rebind buffers while they are in use.
- Supply any required `no_grad` scope and autograd integration. CP does not
  register automatic backward operations.
- Stop submissions through all CP groups sharing a ProcessGroup before teardown.
  Finish post-processing and GPU uses, close handles, then call `close_runtime`
  before destroying that ProcessGroup.

`handle.close()` releases native plans without synchronization or freeing shared
workspace. The native registry owns workspace per ProcessGroup/slot.
`close_runtime(group)` drains/releases every slot of that ProcessGroup, including
slots shared by sibling CP groups, and may block the host. Deleting a group/handle
does not perform this cleanup. Local errors are not broadcast; recovery must be
coordinated by the application.

## Backends

| Backend | Payload dtype | Layout / topology |
|---|---|---|
| zero-CTA | FP32, BF16 | Dense trailing dimensions; token-axis stride padding allowed; contiguous peer-accessible rank blocks per NVL domain |
| Flat all2allv | FP16, BF16, FP32, FP64 | Strided tensors with nonoverlapping output; NCCL GPU or Gloo CPU |

Handle preparation checks native availability, configuration, topology and capacity
across ranks. `handle.backend` reports the choice; `handle.fallback_reason` explains
fallback. `backend="all2allv"` or `NCCL_CP_DISABLE_ZERO_CTA=1` selects flat.
A prepared handle never switches backend or grows workspace during a data call;
payload bytes must fit its runtime. Rounding can differ with reduction grouping.

Explicit-list preparation uses shared routes/configuration to determine eligibility.
If a required local native prerequisite is unavailable, it raises; callers select
flat consistently across ranks. Running communication failures propagate and are
never retried through another backend.

Group creation and runtime teardown perform no CP control AllGather. Handle creation
performs one; prepared data calls perform none and upload no metadata. Native slot
initialization separately connects/rendezvous/registers symmetric buffers. This
setup can communicate and is reused by compatible handles.

## Tests

Run the complete local suite through the built package:

```bash
python build/bin/nccl_cp_run.py --module pytest tests -q
```

Alternatively configure `-DNCCL_CP_BUILD_TESTS=ON` and run
`ctest --test-dir build --output-on-failure`. Distributed runners are separate:

| Runner | Coverage |
|---|---|
| [tests/run_distributed.py](tests/run_distributed.py) | Native direct/hierarchical plans, FP32/BF16 numerical results |
| [tests/run_public.py](tests/run_public.py) | Public routes, reuse, lifecycle, fallback, errors |
| [tests/run_padding.py](tests/run_padding.py) | Capacity tails and reverse reduction |
| [tests/run_async.py](tests/run_async.py) | Deferred output, accumulation, stream/slot ordering, reserved options |

The public runners require explicit `--backend` and `--nvl-domain-size` values.
The native-only runner requires `--nvl-domain-size` and uses NCCL. Launcher
options control rank allocation and rendezvous; there are no fixed device IDs,
process counts, addresses or ports in these examples:

```bash
torchrun <launcher-options> \
  build/bin/nccl_cp_run.py tests/run_public.py \
  --backend <gloo-or-nccl> --nvl-domain-size <ranks-per-domain>

torchrun <launcher-options> \
  build/bin/nccl_cp_run.py tests/run_distributed.py \
  --nvl-domain-size <ranks-per-domain>
```

Use the same explicit arguments for `run_padding.py` and `run_async.py`.
`--require-zero` requires native selection; the public runner's `--force-flat`
selects the fallback. The public runner needs multiple ranks for its cross-rank
error cases. Direct/hierarchical selection follows communicator size and domain
size. Relay-only cases are generated when multiple multi-rank domains exist.

Unit tests use synthetic routes, temporary directories and mocked compiler/device
properties. Those fixtures validate contracts without requiring matching hardware.
Distributed results report test outcomes; they do not collect host names, GPU
identifiers or software inventory.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
