/**********************************************************************************
 * Copyright (c) 2025-2026 SandAI. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *********************************************************************************/

#include "zero_cta_kernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
// ABI NOTE: the NGC PyTorch 2.12 build used by this extension exposes the
// NCCL communicator through getCommPtr(), but has no public accessor for the
// matching ProcessGroupNCCL stream. NCCL RMA must use that exact stream.
#define private public
#define protected public
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#undef protected
#undef private
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NCCL_CP_ZERO_CTA_HAS_NVTX 1
#else
#define NCCL_CP_ZERO_CTA_HAS_NVTX 0
#endif

namespace nccl_cp_zero_cta {

using c10d::ProcessGroup;
using c10d::ProcessGroupNCCL;
using c10d::symmetric_memory::NCCLSymmetricMemory;
using c10d::symmetric_memory::SymmetricMemory;

namespace {

constexpr int64_t kGatherTokensPerTile = 32;
constexpr int64_t kReduceTokensPerRange = 128;

at::Device current_cuda_device() {
  // Distributed workers set their single CUDA device before creating CP state.
  return at::Device(at::kCUDA, at::cuda::current_device());
}

struct Workspace {
  // One byte-addressable send/recv pair is shared by every CP payload dtype.
  at::Tensor send_bytes;
  at::Tensor recv_bytes;
  c10::intrusive_ptr<SymmetricMemory> send_handle;
  c10::intrusive_ptr<SymmetricMemory> recv_handle;
  void** send_buffer_ptrs = nullptr;
  void** recv_buffer_ptrs = nullptr;
  ncclWindow_t recv_window = nullptr;
  size_t recv_window_offset = 0;
};

void nvtx_push(const char* name) {
#if NCCL_CP_ZERO_CTA_HAS_NVTX
  nvtxRangePushA(name);
#else
  (void)name;
#endif
}

void nvtx_pop() {
#if NCCL_CP_ZERO_CTA_HAS_NVTX
  nvtxRangePop();
#endif
}

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtx_push(name); }
  ~NvtxRange() { nvtx_pop(); }
};

void check_nccl(ncclResult_t result, const char* op) {
  TORCH_CHECK(
      result == ncclSuccess,
      "zero-CTA ",
      op,
      " failed: ",
      ncclGetErrorString(result));
}

at::cuda::CUDAStream get_process_group_nccl_stream(
    ProcessGroupNCCL* backend) {
  TORCH_CHECK(backend != nullptr, "zero-CTA NCCL backend is null");
  const int device_index = at::cuda::current_device();
  std::lock_guard<std::mutex> lock(backend->mutex_);
  const auto it = backend->ncclStreams_.find(std::to_string(device_index));
  TORCH_CHECK(
      it != backend->ncclStreams_.end(),
      "zero-CTA ProcessGroup NCCL stream is not initialized for cuda:",
      device_index);
  return it->second;
}

ProcessGroupNCCL* validate_zero_cta_backend(
    const c10::intrusive_ptr<ProcessGroup>& group) {
  auto backend = group->getBackend(ProcessGroup::BackendType::NCCL);
  TORCH_CHECK(backend != nullptr, "zero-CTA runtime requires an NCCL backend");
  auto* nccl_backend = dynamic_cast<ProcessGroupNCCL*>(backend.get());
  TORCH_CHECK(
      nccl_backend != nullptr,
      "zero-CTA runtime requires ProcessGroupNCCL");
#if defined(NCCL_HAS_CTA_POLICY) && defined(NCCL_CTA_POLICY_ZERO)
  auto options = nccl_backend->getOptions();
  TORCH_CHECK(
      (options->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) != 0,
      "zero-CTA runtime requires ProcessGroupNCCL CTA policy ZERO");
#else
  TORCH_CHECK(false, "PyTorch/NCCL build lacks NCCL CTA policy support");
#endif
  int version = 0;
  check_nccl(ncclGetVersion(&version), "ncclGetVersion");
  TORCH_CHECK(version >= 23000, "zero-CTA runtime requires NCCL >= 2.30");
  return nccl_backend;
}

int64_t per_token_elements(const at::Tensor& tensor) {
  int64_t result = 1;
  for (int64_t dim = 1; dim < tensor.dim(); ++dim) result *= tensor.size(dim);
  return result;
}

void validate_shapes(
    const at::Tensor& input,
    const at::Tensor& output) {
  // Local metadata checks only. The caller coordinates operation/plan order
  // and compatible per-token dtype/shape across ranks; no peer negotiation
  // or validation of the token identities stored in these tensors occurs here.
  TORCH_CHECK(
      input.is_cuda() && output.is_cuda() && input.dim() >= 1 &&
          input.dim() == output.dim() &&
          input.scalar_type() == output.scalar_type() &&
          input.device() == output.device() &&
          (input.scalar_type() == at::kFloat ||
           input.scalar_type() == at::kBFloat16),
      "zero-CTA input/output tensor mismatch");
  int64_t stride = 1;
  for (int64_t dim = 1; dim < input.dim(); ++dim) {
    TORCH_CHECK(
        input.size(dim) == output.size(dim), "per-token shapes differ");
  }
  for (int64_t dim = input.dim() - 1; dim >= 1; --dim) {
    TORCH_CHECK(
        input.stride(dim) == stride && output.stride(dim) == stride,
        "per-token payload must be contiguous");
    stride *= input.size(dim);
  }
  TORCH_CHECK(
      input.stride(0) >= stride && output.stride(0) >= stride,
      "invalid token stride");
}

template <typename T, typename = void>
struct has_window_offset : std::false_type {};

template <typename T>
struct has_window_offset<
    T,
    std::void_t<decltype(std::declval<T&>().get_window_offset())>>
    : std::true_type {};

template <typename T>
size_t get_window_offset_compat(T* memory) {
  if constexpr (has_window_offset<NCCLSymmetricMemory>::value) {
    return memory->get_window_offset();
  }
  return memory->get_offset();
}

template <typename F>
void nccl_group(ProcessGroupNCCL& backend, F&& body) {
  backend.groupStart();
  try {
    body();
    backend.groupEnd();
  } catch (...) {
    try {
      backend.groupEnd();
    } catch (...) {
    }
    throw;
  }
}

int64_t expanded_row_count(
    const at::Tensor& host,
    int64_t n_tokens_field,
    int64_t tokens_per_row) {
  if (host.numel() == 0) return 0;
  auto input = host.accessor<int64_t, 2>();
  int64_t rows = 0;
  for (int64_t row = 0; row < host.size(0); ++row) {
    rows +=
        (input[row][n_tokens_field] + tokens_per_row - 1) / tokens_per_row;
  }
  return rows;
}

at::Tensor make_pinned_i64(int64_t rows, int64_t fields) {
  return at::empty(
      {rows, fields},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
}

at::Tensor copy_to_device(at::Tensor host) {
  return host.to(current_cuda_device(), at::kLong, true, true);
}

template <bool kSymmetric>
at::Tensor make_device_gather_tiles_impl(const at::Tensor& host) {
  constexpr int64_t kFields = kSymmetric
      ? symmetric_gather_segment::kFields
      : tensor_gather_segment::kFields;
  constexpr int64_t kFirstTokenField = kSymmetric
      ? symmetric_gather_segment::kSrcTokenStart
      : tensor_gather_segment::kSrcTokenStart;
  constexpr int64_t kNTokensField = kSymmetric
      ? symmetric_gather_segment::kNTokens
      : tensor_gather_segment::kNTokens;
  auto expanded = make_pinned_i64(
      expanded_row_count(host, kNTokensField, kGatherTokensPerTile), kFields);
  if (host.numel() == 0) return copy_to_device(std::move(expanded));

  auto input = host.accessor<int64_t, 2>();
  auto output = expanded.accessor<int64_t, 2>();
  int64_t output_row = 0;
  for (int64_t row = 0; row < host.size(0); ++row) {
    const int64_t tokens = input[row][kNTokensField];
    for (int64_t offset = 0; offset < tokens;
         offset += kGatherTokensPerTile) {
      for (int64_t field = 0; field < kFields; ++field) {
        output[output_row][field] = input[row][field];
      }
      for (int64_t field = kFirstTokenField; field < kNTokensField; ++field) {
        output[output_row][field] += offset;
      }
      output[output_row][kNTokensField] =
          std::min(kGatherTokensPerTile, tokens - offset);
      ++output_row;
    }
  }
  return copy_to_device(std::move(expanded));
}

} // namespace

at::Tensor make_device_tensor_gather_tiles(const at::Tensor& host) {
  return make_device_gather_tiles_impl<false>(host);
}

at::Tensor make_device_symmetric_gather_tiles(const at::Tensor& host) {
  return make_device_gather_tiles_impl<true>(host);
}

at::Tensor make_device_reduce_ranges(const at::Tensor& host) {
  auto expanded = make_pinned_i64(
      expanded_row_count(
          host, reduce_segment::kNTokens, kReduceTokensPerRange),
      reduce_range::kFields);
  if (host.numel() == 0) return copy_to_device(std::move(expanded));

  auto input = host.accessor<int64_t, 2>();
  auto output = expanded.accessor<int64_t, 2>();
  int64_t output_row = 0;
  for (int64_t row = 0; row < host.size(0); ++row) {
    const int64_t tokens = input[row][reduce_segment::kNTokens];
    for (int64_t offset = 0; offset < tokens;
         offset += kReduceTokensPerRange) {
      output[output_row][reduce_range::kDstTokenStart] =
          input[row][reduce_segment::kDstTokenStart] + offset;
      output[output_row][reduce_range::kFirstSrcIndex] =
          input[row][reduce_segment::kFirstSrcIndex];
      output[output_row][reduce_range::kNSrcs] =
          input[row][reduce_segment::kNSrcs];
      output[output_row][reduce_range::kSrcRangeTokenOffset] = offset;
      output[output_row][reduce_range::kNTokens] =
          std::min(kReduceTokensPerRange, tokens - offset);
      ++output_row;
    }
  }
  return copy_to_device(std::move(expanded));
}

struct PendingReclaim {
  int64_t runtime_slot;
  std::shared_ptr<at::cuda::CUDAEvent> completion;
  std::vector<ncclWaitSignalDesc_t> signal_peers;
  std::vector<ncclWaitSignalDesc_t> wait_peers;
};

struct SharedSignalState {
  // NCCL RMA signal ordering is shared by every workspace on a communicator.
  std::mutex mutex;
  std::deque<PendingReclaim> pending_reclaims;
};

class ZeroCtaRuntime final : public torch::CustomClassHolder {
 public:
  c10::intrusive_ptr<ProcessGroup> group;
  ProcessGroupNCCL* nccl_backend = nullptr;
  int rank = 0;
  int world_size = 0;
  int64_t max_per_peer_slot_value = 0;
  int64_t nvl_domain_size_value = 1;
  int64_t max_per_token_bytes_value = 0;
  int64_t runtime_slot_value = 0;
  std::shared_ptr<SharedSignalState> signal_state;
  std::optional<at::cuda::CUDAStream> comm_stream;
  ncclComm_t comm = nullptr;
  std::mutex mutex;
  std::optional<Workspace> workspace;
  std::vector<ncclWaitSignalDesc_t> peer_post_process_peers;
  // Profiling-only marker kernels, enabled before runtime creation with
  // NCCL_CP_GPU_MARKERS=1.
  bool gpu_completion_markers_enabled = false;
  bool collective_pending = false;
  bool closed = false;

  int64_t max_per_peer_slot() const { return max_per_peer_slot_value; }
  int64_t node_slot_capacity() const {
    return hierarchical() ? max_per_peer_slot_value : 0;
  }
  int64_t nvl_domain_size() const { return nvl_domain_size_value; }
  int64_t max_per_token_bytes() const { return max_per_token_bytes_value; }
  int64_t runtime_id() const {
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(this));
  }

  bool hierarchical() const { return world_size > nvl_domain_size_value; }
  int64_t num_nodes() const { return world_size / nvl_domain_size_value; }
  int64_t raw_token_capacity() const {
    return static_cast<int64_t>(world_size) * max_per_peer_slot_value;
  }
  int64_t node_token_capacity() const {
    return num_nodes() * max_per_peer_slot_value;
  }
  int64_t send_token_capacity() const {
    return raw_token_capacity() +
        (hierarchical() ? node_token_capacity() : 0);
  }
  int64_t recv_token_capacity() const {
    return hierarchical() ? node_token_capacity() : raw_token_capacity();
  }

  void close() {
    // Caller contract: no further submissions; all participating ranks close
    // compatible runtimes before their ProcessGroups are destroyed.
    std::lock_guard<std::mutex> lock(mutex);
    if (closed) return;
    TORCH_CHECK(
        !collective_pending,
        "zero-CTA runtime cannot close before pending work is waited");
    // Cleanup is outside the communication hot path. Waiting here guarantees
    // that symmetric windows are deregistered before ProcessGroup destruction.
    if (signal_state != nullptr) {
      std::lock_guard<std::mutex> signal_lock(signal_state->mutex);
      C10_CUDA_CHECK(cudaStreamSynchronize(comm_stream->stream()));
      auto& pending = signal_state->pending_reclaims;
      for (auto it = pending.begin(); it != pending.end();) {
        if (it->runtime_slot == runtime_slot_value) {
          it->completion->synchronize();
          it = pending.erase(it);
        } else {
          ++it;
        }
      }
    }
    workspace.reset();
    closed = true;
  }
};

void mark_gpu_completion(
    ZeroCtaRuntime& runtime,
    GpuCompletionMarker marker) {
  if (runtime.gpu_completion_markers_enabled) {
    launch_gpu_completion_marker(marker, runtime.comm_stream->stream());
  }
}

struct PostReduceMetadata {
  at::Tensor reduce_ranges;
  at::Tensor src_token_offsets;
};

struct LocalReduceMetadata {
  at::Tensor reduce_ranges;
  at::Tensor symmetric_srcs;
};

class ZeroCtaPlan final : public torch::CustomClassHolder {
 public:
  std::vector<int64_t> send_token_counts;
  std::vector<ncclWaitSignalDesc_t> remote_data_waits;
  at::Tensor pack_gather_tiles;
  std::optional<at::Tensor> post_gather_tiles;
  std::optional<PostReduceMetadata> post_reduce;
  std::optional<LocalReduceMetadata> local_reduce;
  std::vector<ncclWaitSignalDesc_t> local_ready_signals;
  std::vector<ncclWaitSignalDesc_t> local_ready_waits;

  bool is_reduce() const { return post_reduce.has_value(); }
};

std::vector<ncclWaitSignalDesc_t> make_signal_descs(
    const std::vector<int64_t>& peers) {
  std::vector<ncclWaitSignalDesc_t> result(peers.size());
  for (size_t i = 0; i < peers.size(); ++i) {
    result[i].opCnt = 1;
    result[i].peer = peers[i];
  }
  return result;
}

c10::intrusive_ptr<ZeroCtaPlan> create_plan_base(
    std::vector<int64_t> send_token_counts,
    std::vector<int64_t> remote_wait_peers,
    at::Tensor pack_gather_tiles,
    std::vector<int64_t> local_ready_signal_peers,
    std::vector<int64_t> local_ready_wait_peers) {
  auto plan = c10::make_intrusive<ZeroCtaPlan>();
  plan->send_token_counts = std::move(send_token_counts);
  plan->remote_data_waits = make_signal_descs(remote_wait_peers);
  plan->pack_gather_tiles = std::move(pack_gather_tiles);
  plan->local_ready_signals = make_signal_descs(local_ready_signal_peers);
  plan->local_ready_waits = make_signal_descs(local_ready_wait_peers);
  return plan;
}

c10::intrusive_ptr<ZeroCtaPlan> create_cast_plan(
    std::vector<int64_t> send_token_counts,
    std::vector<int64_t> remote_wait_peers,
    at::Tensor pack_gather_tiles,
    at::Tensor post_gather_tiles,
    std::vector<int64_t> local_ready_signal_peers,
    std::vector<int64_t> local_ready_wait_peers) {
  auto plan = create_plan_base(
      std::move(send_token_counts),
      std::move(remote_wait_peers),
      std::move(pack_gather_tiles),
      std::move(local_ready_signal_peers),
      std::move(local_ready_wait_peers));
  plan->post_gather_tiles = std::move(post_gather_tiles);
  return plan;
}

c10::intrusive_ptr<ZeroCtaPlan> create_reduce_plan(
    std::vector<int64_t> send_token_counts,
    std::vector<int64_t> remote_wait_peers,
    at::Tensor pack_gather_tiles,
    at::Tensor post_reduce_ranges,
    at::Tensor post_reduce_src_token_offsets,
    std::optional<at::Tensor> local_reduce_ranges,
    std::optional<at::Tensor> local_reduce_symmetric_srcs,
    std::vector<int64_t> local_ready_signal_peers,
    std::vector<int64_t> local_ready_wait_peers) {
  TORCH_CHECK(
      local_reduce_ranges.has_value() ==
          local_reduce_symmetric_srcs.has_value(),
      "zero-CTA local reduce metadata is incomplete");
  auto plan = create_plan_base(
      std::move(send_token_counts),
      std::move(remote_wait_peers),
      std::move(pack_gather_tiles),
      std::move(local_ready_signal_peers),
      std::move(local_ready_wait_peers));
  plan->post_reduce.emplace(PostReduceMetadata{
      std::move(post_reduce_ranges),
      std::move(post_reduce_src_token_offsets),
  });
  if (local_reduce_ranges.has_value()) {
    plan->local_reduce.emplace(LocalReduceMetadata{
        std::move(*local_reduce_ranges),
        std::move(*local_reduce_symmetric_srcs),
    });
  }
  return plan;
}

struct RuntimeGroup {
  c10::intrusive_ptr<ProcessGroup> group;
  std::shared_ptr<SharedSignalState> signal_state;
  std::unordered_map<int64_t, c10::intrusive_ptr<ZeroCtaRuntime>> slots;
};

struct RuntimeRegistry {
  std::mutex mutex;
  std::unordered_map<ProcessGroup*, RuntimeGroup> groups;
};

RuntimeRegistry& runtime_registry() {
  // Explicit close APIs release CUDA/NCCL resources before ProcessGroup
  // teardown. Deliberately leak the empty registry shell so static destruction
  // never touches CUDA state during interpreter shutdown.
  static auto* registry = new RuntimeRegistry();
  return *registry;
}

namespace {

Workspace& prepare_workspace(ZeroCtaRuntime& runtime) {
  const int64_t send_bytes_size =
      runtime.send_token_capacity() * runtime.max_per_token_bytes_value;
  const int64_t recv_bytes_size =
      runtime.recv_token_capacity() * runtime.max_per_token_bytes_value;
  const std::vector<int64_t> strides{1};
  const std::string group_name = runtime.group->getGroupName();
  const auto device = current_cuda_device();
  auto send_bytes = c10d::symmetric_memory::empty_strided_p2p(
      {send_bytes_size},
      strides,
      at::kByte,
      device,
      std::nullopt,
      std::nullopt);
  auto recv_bytes = c10d::symmetric_memory::empty_strided_p2p(
      {recv_bytes_size},
      strides,
      at::kByte,
      device,
      std::nullopt,
      std::nullopt);
  auto send_handle =
      c10d::symmetric_memory::rendezvous(send_bytes, group_name);
  auto recv_handle =
      c10d::symmetric_memory::rendezvous(recv_bytes, group_name);
  auto* send_nccl = dynamic_cast<NCCLSymmetricMemory*>(send_handle.get());
  auto* recv_nccl = dynamic_cast<NCCLSymmetricMemory*>(recv_handle.get());
  TORCH_CHECK(
      send_nccl != nullptr && recv_nccl != nullptr,
      "zero-CTA requires NCCL symmetric-memory handles");
  TORCH_CHECK(
      recv_nccl->get_window() != nullptr,
      "zero-CTA NCCL receive window is null");
  const size_t recv_window_offset = get_window_offset_compat(recv_nccl);
  runtime.workspace.emplace(Workspace{
      send_bytes,
      recv_bytes,
      send_handle,
      recv_handle,
      send_nccl->get_buffer_ptrs_dev(),
      recv_nccl->get_buffer_ptrs_dev(),
      recv_nccl->get_window(),
      recv_window_offset,
  });
  return *runtime.workspace;
}

void wait_signals(
    ZeroCtaRuntime& runtime,
    const std::vector<ncclWaitSignalDesc_t>& waits,
    const char* op) {
  if (waits.empty()) return;
  check_nccl(
      ncclWaitSignal(
          static_cast<int>(waits.size()),
          const_cast<ncclWaitSignalDesc_t*>(waits.data()),
          runtime.comm,
          runtime.comm_stream->stream()),
      op);
}

void signal_peers(
    ZeroCtaRuntime& runtime,
    const std::vector<ncclWaitSignalDesc_t>& peers,
    const char* op) {
  if (peers.empty()) return;
  // Fuse peer signals so dense CP routes pay one NCCL enqueue per batch.
  nccl_group(*runtime.nccl_backend, [&] {
    for (const auto& peer : peers) {
      check_nccl(
          ncclSignal(
              peer.peer,
              0,
              0,
              0,
              runtime.comm,
              runtime.comm_stream->stream()),
          op);
    }
  });
}

void reclaim_workspace(ZeroCtaRuntime& runtime) {
  TORCH_CHECK(runtime.signal_state != nullptr, "zero-CTA signal state is missing");
  std::lock_guard<std::mutex> lock(runtime.signal_state->mutex);
  auto& pending = runtime.signal_state->pending_reclaims;
  const auto target = std::find_if(
      pending.begin(), pending.end(), [&](const PendingReclaim& reclaim) {
        return reclaim.runtime_slot == runtime.runtime_slot_value;
      });
  if (target == pending.end()) return;

  NvtxRange range("zero_cta_cpp.reclaim_workspace");
  const size_t count = static_cast<size_t>(target - pending.begin()) + 1;
  for (size_t index = 0; index < count; ++index) {
    auto& reclaim = pending.front();
    reclaim.completion->block(*runtime.comm_stream);
    signal_peers(
        runtime,
        reclaim.signal_peers,
        "ncclSignal(peer-post-process-done)");
    wait_signals(
        runtime,
        reclaim.wait_peers,
        "ncclWaitSignal(peer-post-process-done)");
    pending.pop_front();
  }
  mark_gpu_completion(runtime, GpuCompletionMarker::kPeerPostProcessDone);
}

at::Tensor workspace_view(
    const at::Tensor& bytes,
    int64_t tokens,
    int64_t token_elements,
    size_t per_token_bytes,
    at::ScalarType dtype) {
  return bytes
      .narrow(0, 0, tokens * static_cast<int64_t>(per_token_bytes))
      .view(dtype)
      .view({tokens, token_elements});
}

struct HierarchicalViews {
  at::Tensor raw_send_buffer;
  at::Tensor node_send_buffer;
  at::Tensor node_recv_buffer;
};

HierarchicalViews make_hierarchical_views(
    ZeroCtaRuntime& runtime,
    Workspace& workspace,
    const at::Tensor& like) {
  const int64_t token_elements = per_token_elements(like);
  const size_t per_token_bytes =
      static_cast<size_t>(token_elements) * like.element_size();
  auto send = workspace_view(
      workspace.send_bytes,
      runtime.send_token_capacity(),
      token_elements,
      per_token_bytes,
      like.scalar_type());
  auto recv = workspace_view(
      workspace.recv_bytes,
      runtime.recv_token_capacity(),
      token_elements,
      per_token_bytes,
      like.scalar_type());
  const int64_t raw_tokens = runtime.raw_token_capacity();
  return {
      send.narrow(0, 0, raw_tokens),
      send.narrow(0, raw_tokens, runtime.node_token_capacity()),
      recv,
  };
}

void symmetric_memory_put(
    ZeroCtaRuntime& runtime,
    Workspace& workspace,
    const at::Tensor& send_buffer,
    const at::Tensor& recv_buffer,
    const std::vector<int64_t>& send_token_counts,
    size_t per_token_bytes,
    bool hierarchical) {
  const int64_t capacity = runtime.max_per_peer_slot_value;
  const int64_t local_slot = hierarchical
      ? runtime.rank / runtime.nvl_domain_size_value : runtime.rank;
  const int64_t source_base = hierarchical ? runtime.raw_token_capacity() : 0;
  nccl_group(*runtime.nccl_backend, [&] {
    for (int peer = 0; peer < runtime.world_size; ++peer) {
      const int64_t tokens = send_token_counts[peer];
      if (tokens == 0) continue;
      const int64_t target_slot = hierarchical
          ? peer / runtime.nvl_domain_size_value : peer;
      if (peer == runtime.rank) {
        // Remote peers use ncclPutSignal; the local peer uses an async D2D copy.
        recv_buffer.narrow(0, local_slot * capacity, tokens)
            .copy_(
                send_buffer.narrow(0, target_slot * capacity, tokens), true);
        continue;
      }
      const size_t source_offset = static_cast<size_t>(
          source_base + target_slot * capacity) * per_token_bytes;
      const size_t destination_offset =
          static_cast<size_t>(local_slot * capacity) * per_token_bytes;
      check_nccl(
          ncclPutSignal(
              static_cast<char*>(workspace.send_bytes.data_ptr()) +
                  source_offset,
              static_cast<size_t>(tokens) * per_token_bytes,
              ncclChar,
              peer,
              workspace.recv_window,
              workspace.recv_window_offset + destination_offset,
              0,
              0,
              0,
              runtime.comm,
              runtime.comm_stream->stream()),
          "ncclPutSignal");
    }
  });
}

std::shared_ptr<at::cuda::CUDAEvent> record_event(
    at::cuda::CUDAStream stream) {
  auto event = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
  event->record(stream);
  return event;
}

void wait_for_workspace_reclaim_before_pack(
    ZeroCtaRuntime& runtime,
    at::cuda::CUDAStream process_stream) {
  c10::cuda::CUDAStreamGuard comm_stream_guard(*runtime.comm_stream);
  reclaim_workspace(runtime);

  // Another slot may have dequeued this reclaim before its GPU work completes.
  // Always fence the comm stream before pack overwrites the shared workspace.
  auto workspace_ready = record_event(*runtime.comm_stream);
  workspace_ready->block(process_stream);
}

struct PendingCollective {
  at::Tensor recv_buffer;
  std::shared_ptr<at::cuda::CUDAEvent> rma_ready;
};

void launch_pack(
    const at::Tensor& input,
    const at::Tensor& dst_symmetric_buffer,
    const ZeroCtaPlan& plan,
    at::cuda::CUDAEvent& pack_ready) {
  const auto process_stream =
      at::cuda::getCurrentCUDAStream();
  {
    NvtxRange range("zero_cta_cpp.pack");
    launch_gather_from_tensor(
        input,
        dst_symmetric_buffer,
        TensorGatherMetadata{plan.pack_gather_tiles},
        process_stream.stream());
  }
  pack_ready.record(process_stream);
}

PendingCollective enqueue_direct(
    ZeroCtaRuntime& runtime,
    Workspace& workspace,
    const at::Tensor& input,
    int64_t token_elements,
    size_t per_token_bytes,
    const ZeroCtaPlan& plan) {
  auto send_buffer = workspace_view(
      workspace.send_bytes,
      runtime.raw_token_capacity(),
      token_elements,
      per_token_bytes,
      input.scalar_type());
  auto recv_buffer = workspace_view(
      workspace.recv_bytes,
      runtime.raw_token_capacity(),
      token_elements,
      per_token_bytes,
      input.scalar_type());
  const auto process_stream = at::cuda::getCurrentCUDAStream();
  wait_for_workspace_reclaim_before_pack(runtime, process_stream);
  at::cuda::CUDAEvent pack_ready(cudaEventDisableTiming);
  launch_pack(input, send_buffer, plan, pack_ready);

  c10::cuda::CUDAStreamGuard comm_stream_guard(*runtime.comm_stream);
  pack_ready.block(*runtime.comm_stream);
  mark_gpu_completion(runtime, GpuCompletionMarker::kPackReady);
  {
    NvtxRange range("zero_cta_cpp.put_signal_group");
    symmetric_memory_put(
        runtime, workspace, send_buffer, recv_buffer,
        plan.send_token_counts, per_token_bytes, false);
  }
  {
    NvtxRange range("zero_cta_cpp.wait_signal_array");
    wait_signals(runtime, plan.remote_data_waits, "ncclWaitSignal");
  }
  mark_gpu_completion(runtime, GpuCompletionMarker::kRemoteDataReady);
  mark_gpu_completion(runtime, GpuCompletionMarker::kRmaReady);
  return {recv_buffer, record_event(*runtime.comm_stream)};
}

PendingCollective enqueue_hierarchical_cast(
    ZeroCtaRuntime& runtime,
    Workspace& workspace,
    const at::Tensor& input,
    size_t per_token_bytes,
    const ZeroCtaPlan& plan,
    const HierarchicalViews& buffers) {
  const auto process_stream = at::cuda::getCurrentCUDAStream();
  wait_for_workspace_reclaim_before_pack(runtime, process_stream);
  at::cuda::CUDAEvent pack_ready(cudaEventDisableTiming);
  launch_pack(input, buffers.node_send_buffer, plan, pack_ready);

  c10::cuda::CUDAStreamGuard comm_stream_guard(*runtime.comm_stream);
  pack_ready.block(*runtime.comm_stream);
  mark_gpu_completion(runtime, GpuCompletionMarker::kPackReady);
  {
    NvtxRange range("zero_cta_cpp.put_signal_group");
    symmetric_memory_put(
        runtime, workspace, buffers.node_send_buffer,
        buffers.node_recv_buffer, plan.send_token_counts, per_token_bytes,
        true);
  }
  {
    NvtxRange range("zero_cta_cpp.wait_signal_array");
    wait_signals(runtime, plan.remote_data_waits, "ncclWaitSignal");
  }
  mark_gpu_completion(runtime, GpuCompletionMarker::kRemoteDataReady);
  {
    NvtxRange range("zero_cta_cpp.signal_local_ready");
    signal_peers(runtime, plan.local_ready_signals, "ncclSignal(local-ready)");
  }
  {
    NvtxRange range("zero_cta_cpp.wait_local_ready");
    wait_signals(
        runtime, plan.local_ready_waits, "ncclWaitSignal(local-ready)");
  }
  mark_gpu_completion(runtime, GpuCompletionMarker::kLocalReady);
  mark_gpu_completion(runtime, GpuCompletionMarker::kRmaReady);
  return {buffers.node_recv_buffer, record_event(*runtime.comm_stream)};
}

PendingCollective enqueue_hierarchical_reduce(
    ZeroCtaRuntime& runtime,
    Workspace& workspace,
    const at::Tensor& input,
    size_t per_token_bytes,
    const ZeroCtaPlan& plan,
    const HierarchicalViews& buffers) {
  const auto process_stream = at::cuda::getCurrentCUDAStream();
  wait_for_workspace_reclaim_before_pack(runtime, process_stream);
  at::cuda::CUDAEvent pack_ready(cudaEventDisableTiming);
  launch_pack(input, buffers.raw_send_buffer, plan, pack_ready);

  c10::cuda::CUDAStreamGuard comm_stream_guard(*runtime.comm_stream);
  pack_ready.block(*runtime.comm_stream);
  mark_gpu_completion(runtime, GpuCompletionMarker::kPackReady);
  {
    NvtxRange range("zero_cta_cpp.signal_local_ready");
    signal_peers(runtime, plan.local_ready_signals, "ncclSignal(local-ready)");
  }
  {
    NvtxRange range("zero_cta_cpp.wait_local_ready");
    wait_signals(
        runtime, plan.local_ready_waits, "ncclWaitSignal(local-ready)");
  }
  mark_gpu_completion(runtime, GpuCompletionMarker::kLocalReady);
  {
    NvtxRange range("zero_cta_cpp.local_reduce");
    const int64_t local_output_begin =
        runtime.rank / runtime.nvl_domain_size_value *
        runtime.max_per_peer_slot_value;
    const auto& local_reduce = *plan.local_reduce;
    launch_reduce_from_symmetric(
        buffers.node_send_buffer,
        buffers.node_recv_buffer,
        local_output_begin,
        local_output_begin + runtime.max_per_peer_slot_value,
        SymmetricReduceMetadata{
            local_reduce.reduce_ranges,
            local_reduce.symmetric_srcs,
            workspace.send_buffer_ptrs},
        runtime.comm_stream->stream());
  }
  {
    NvtxRange range("zero_cta_cpp.signal_local_consumed");
    signal_peers(
        runtime, plan.local_ready_waits, "ncclSignal(local-consumed)");
  }
  {
    NvtxRange range("zero_cta_cpp.put_signal_group");
    symmetric_memory_put(
        runtime, workspace, buffers.node_send_buffer,
        buffers.node_recv_buffer, plan.send_token_counts, per_token_bytes,
        true);
  }
  {
    NvtxRange range("zero_cta_cpp.wait_local_consumed");
    wait_signals(
        runtime, plan.local_ready_signals,
        "ncclWaitSignal(local-consumed)");
  }
  mark_gpu_completion(runtime, GpuCompletionMarker::kLocalConsumed);
  {
    NvtxRange range("zero_cta_cpp.wait_signal_array");
    wait_signals(runtime, plan.remote_data_waits, "ncclWaitSignal");
  }
  mark_gpu_completion(runtime, GpuCompletionMarker::kRemoteDataReady);
  mark_gpu_completion(runtime, GpuCompletionMarker::kRmaReady);
  return {buffers.node_recv_buffer, record_event(*runtime.comm_stream)};
}

PendingCollective enqueue_collective(
    ZeroCtaRuntime& runtime,
    Workspace& workspace,
    const at::Tensor& input,
    const ZeroCtaPlan& plan) {
  const int64_t token_elements = per_token_elements(input);
  const size_t per_token_bytes =
      static_cast<size_t>(token_elements) * input.element_size();
  TORCH_CHECK(
      per_token_bytes <=
          static_cast<size_t>(runtime.max_per_token_bytes_value),
      "zero-CTA per-token payload exceeds shared workspace capacity");
  if (!runtime.hierarchical()) {
    return enqueue_direct(
        runtime, workspace, input, token_elements, per_token_bytes, plan);
  }
  const auto buffers = make_hierarchical_views(runtime, workspace, input);
  if (plan.is_reduce()) {
    return enqueue_hierarchical_reduce(
        runtime, workspace, input, per_token_bytes, plan, buffers);
  }
  return enqueue_hierarchical_cast(
      runtime, workspace, input, per_token_bytes, plan, buffers);
}

class ZeroCtaWork final : public c10d::Work {
 public:
  ZeroCtaWork(
      c10::intrusive_ptr<ZeroCtaRuntime> runtime,
      c10::intrusive_ptr<ZeroCtaPlan> plan,
      at::Tensor input,
      at::Tensor output,
      std::shared_ptr<at::cuda::CUDAEvent> completion_event,
      std::optional<PendingCollective> pending)
      : c10d::Work(runtime->rank, c10d::OpType::UNKNOWN),
        event_(std::move(completion_event)),
        pending_(std::move(pending)),
        input_(std::move(input)),
        output_(std::move(output)),
        plan_(std::move(plan)),
        runtime_(std::move(runtime)) {
    TORCH_CHECK(
        event_ != nullptr || pending_.has_value(),
        "zero-CTA work has neither completion nor pending post-process");
  }

  bool isCompleted() override { return poll_completion(); }

  bool isSuccess() const override {
    const_cast<ZeroCtaWork*>(this)->poll_completion();
    return exception() == nullptr;
  }

  void synchronize() override { blockCurrentStream(); }

  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    blockCurrentStream();
    if (event_->query()) {
      poll_completion();
      throw_if_error();
    }
    return true;
  }

  void blockCurrentStream() override {
    throw_if_error();
    enqueue_post_process();
    event_->block(at::cuda::getCurrentCUDAStream());
  }

 private:
  std::exception_ptr query_nccl_error() const {
    ncclResult_t async_error = ncclSuccess;
    const ncclResult_t query_result =
        ncclCommGetAsyncError(runtime_->comm, &async_error);
    ncclResult_t error = query_result;
    if (error == ncclSuccess) error = async_error;
    if (error == ncclSuccess || error == ncclInProgress) return nullptr;
    return std::make_exception_ptr(std::runtime_error(
        std::string("zero-CTA asynchronous NCCL error: ") +
        ncclGetErrorString(error)));
  }

  void enqueue_post_process() {
    std::lock_guard<std::mutex> status_lock(status_mutex_);
    if (!pending_.has_value()) return;

    std::lock_guard<std::mutex> runtime_lock(runtime_->mutex);
    TORCH_CHECK(
        !runtime_->closed,
        "zero-CTA runtime was closed before work was waited");
    TORCH_CHECK(
        runtime_->collective_pending,
        "zero-CTA runtime has no pending collective");

    const auto process_stream = at::cuda::getCurrentCUDAStream();
    auto& pending = *pending_;
    pending.rma_ready->block(process_stream);
    const bool reduce = plan_->is_reduce();
    {
      NvtxRange range(
          reduce ? "zero_cta_cpp.reduce" : "zero_cta_cpp.unpack");
      if (reduce) {
        const auto& post_reduce = *plan_->post_reduce;
        launch_reduce_from_tensor(
            pending.recv_buffer,
            output_,
            TensorReduceMetadata{
                post_reduce.reduce_ranges,
                post_reduce.src_token_offsets},
            process_stream.stream());
      } else if (runtime_->hierarchical()) {
        launch_gather_from_symmetric(
            output_,
            SymmetricGatherMetadata{
                *plan_->post_gather_tiles,
                runtime_->workspace->recv_buffer_ptrs},
            process_stream.stream());
      } else {
        launch_gather_from_tensor(
            pending.recv_buffer,
            output_,
            TensorGatherMetadata{*plan_->post_gather_tiles},
            process_stream.stream());
      }
    }

    auto completion = record_event(process_stream);

    if (!runtime_->peer_post_process_peers.empty()) {
      TORCH_CHECK(
          runtime_->signal_state != nullptr,
          "zero-CTA signal state is missing");
      std::lock_guard<std::mutex> signal_lock(runtime_->signal_state->mutex);
      const auto duplicate = std::find_if(
          runtime_->signal_state->pending_reclaims.begin(),
          runtime_->signal_state->pending_reclaims.end(),
          [&](const PendingReclaim& reclaim) {
            return reclaim.runtime_slot == runtime_->runtime_slot_value;
          });
      TORCH_CHECK(
          duplicate == runtime_->signal_state->pending_reclaims.end(),
          "zero-CTA runtime has an unreclaimed receive window");

      auto reclaim_signals = runtime_->peer_post_process_peers;
      auto reclaim_waits = runtime_->peer_post_process_peers;
      if (reduce && runtime_->hierarchical()) {
        // Routes and byte strides can change on the next collective. Fence
        // every same-lane peer before reusing the receive window.
        reclaim_signals.clear();
        for (const auto& peer : runtime_->peer_post_process_peers) {
          if (peer.peer % runtime_->nvl_domain_size_value ==
              runtime_->rank % runtime_->nvl_domain_size_value) {
            reclaim_signals.push_back(peer);
          }
        }
        reclaim_waits = reclaim_signals;
      }
      runtime_->signal_state->pending_reclaims.push_back(
          PendingReclaim{
              runtime_->runtime_slot_value,
              completion,
              std::move(reclaim_signals),
              std::move(reclaim_waits)});
    }

    event_ = std::move(completion);
    pending_.reset();
    runtime_->collective_pending = false;
  }

  bool poll_completion() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (finished_) return true;
    auto error = query_nccl_error();
    if (error != nullptr) {
      finish(error);
      finished_ = true;
      return true;
    }
    if (pending_.has_value() || event_ == nullptr || !event_->query()) {
      return false;
    }
    error = query_nccl_error();
    finish(error);
    finished_ = true;
    return true;
  }

  void throw_if_error() {
    poll_completion();
    auto error = exception();
    if (error != nullptr) std::rethrow_exception(error);
  }

  std::shared_ptr<at::cuda::CUDAEvent> event_;
  std::optional<PendingCollective> pending_;
  at::Tensor input_;
  at::Tensor output_;
  c10::intrusive_ptr<ZeroCtaPlan> plan_;
  c10::intrusive_ptr<ZeroCtaRuntime> runtime_;
  std::mutex status_mutex_;
  bool finished_ = false;
};

} // namespace

c10::intrusive_ptr<ZeroCtaRuntime> create_runtime_impl(
    int64_t max_per_peer_slot,
    int64_t nvl_domain_size,
    int64_t max_per_token_bytes,
    int64_t runtime_slot,
    std::shared_ptr<SharedSignalState> signal_state,
    bool register_group,
    const c10::intrusive_ptr<ProcessGroup>& group) {
  // Caller contract: each rank selected its assigned CUDA device and enters
  // matching runtime initialization outside capture. Device and ProcessGroup
  // identity remain fixed while the registry owns these symmetric windows.
  TORCH_CHECK(group != nullptr, "zero-CTA ProcessGroup is null");
  TORCH_CHECK(
      max_per_peer_slot >= 0 && max_per_token_bytes >= 0 &&
          nvl_domain_size > 0 &&
          (group->getSize() <= nvl_domain_size ||
           group->getSize() % nvl_domain_size == 0),
      "invalid zero-CTA runtime capacity");
  TORCH_CHECK(
      max_per_peer_slot == 0 || max_per_token_bytes > 0,
      "a non-empty zero-CTA route requires per-token byte capacity");
  TORCH_CHECK(
      at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None,
      "zero-CTA runtime must be created before CUDA graph capture");
  auto* nccl_backend = validate_zero_cta_backend(group);

  const auto device = current_cuda_device();
  auto comm = reinterpret_cast<ncclComm_t>(nccl_backend->getCommPtr());
  if (comm == nullptr) {
    // Eager connection is not idempotent in some PyTorch builds. Repeating it
    // can publish a different communicator for symmetric-memory registration
    // while the ProcessGroup retains its original communicator.
    nccl_backend->eagerConnectSingleDevice(device);
    comm = reinterpret_cast<ncclComm_t>(nccl_backend->getCommPtr());
  }

  auto current_backend = c10d::symmetric_memory::get_backend(device);
  TORCH_CHECK(
      !current_backend.has_value() || current_backend == "NCCL" ||
          current_backend == "CUDA",
      "symmetric-memory backend must be NCCL for zero-CTA");
  c10d::symmetric_memory::set_backend("NCCL");
  if (register_group) {
    c10d::symmetric_memory::set_group_info(
        group->getGroupName(),
        group->getRank(),
        group->getSize(),
        group->getStore());
  }

  auto runtime = c10::make_intrusive<ZeroCtaRuntime>();
  runtime->group = group;
  runtime->nccl_backend = nccl_backend;
  runtime->rank = group->getRank();
  runtime->world_size = group->getSize();
  runtime->max_per_peer_slot_value = max_per_peer_slot;
  runtime->nvl_domain_size_value = nvl_domain_size;
  runtime->max_per_token_bytes_value = max_per_token_bytes;
  runtime->runtime_slot_value = runtime_slot;
  runtime->signal_state = std::move(signal_state);
  TORCH_CHECK(runtime->signal_state != nullptr, "zero-CTA signal state is missing");

  runtime->comm = comm;
  TORCH_CHECK(runtime->comm != nullptr, "NCCL communicator is not initialized");
  runtime->comm_stream = get_process_group_nccl_stream(nccl_backend);
  const char* gpu_markers =
      std::getenv("NCCL_CP_GPU_MARKERS");
  runtime->gpu_completion_markers_enabled =
      gpu_markers != nullptr && std::string(gpu_markers) == "1";
  for (int peer = 0; peer < runtime->world_size; ++peer) {
    if (peer == runtime->rank) continue;
    ncclWaitSignalDesc_t desc{};
    desc.opCnt = 1;
    desc.peer = peer;
    runtime->peer_post_process_peers.push_back(desc);
  }
  if (runtime->send_token_capacity() > 0) prepare_workspace(*runtime);
  return runtime;
}

c10::intrusive_ptr<ZeroCtaRuntime> get_or_create_runtime(
    int64_t max_per_peer_slot,
    int64_t nvl_domain_size,
    int64_t max_per_token_bytes,
    int64_t runtime_slot,
    const c10::intrusive_ptr<ProcessGroup>& group) {
  TORCH_CHECK(runtime_slot >= 0, "zero-CTA runtime slot is negative");
  auto& registry = runtime_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  ProcessGroup* const key = group.get();
  const auto group_found = registry.groups.find(key);
  if (group_found != registry.groups.end()) {
    const auto runtime_found = group_found->second.slots.find(runtime_slot);
    if (runtime_found != group_found->second.slots.end()) {
      const auto& runtime = runtime_found->second;
      TORCH_CHECK(
          runtime->max_per_peer_slot_value == max_per_peer_slot &&
              runtime->nvl_domain_size_value == nvl_domain_size &&
              runtime->max_per_token_bytes_value >= max_per_token_bytes,
          "zero-CTA runtime configuration does not match this slot");
      return runtime;
    }
  }

  const bool register_group = group_found == registry.groups.end();
  auto signal_state = register_group
      ? std::make_shared<SharedSignalState>()
      : group_found->second.signal_state;
  auto runtime = create_runtime_impl(
      max_per_peer_slot,
      nvl_domain_size,
      max_per_token_bytes,
      runtime_slot,
      signal_state,
      register_group,
      group);
  if (register_group) {
    RuntimeGroup runtime_group;
    runtime_group.group = group;
    runtime_group.signal_state = std::move(signal_state);
    runtime_group.slots.emplace(runtime_slot, runtime);
    registry.groups.emplace(key, std::move(runtime_group));
  } else {
    group_found->second.slots.emplace(runtime_slot, runtime);
  }
  return runtime;
}

void erase_runtime(
    ProcessGroup* key,
    int64_t runtime_slot,
    const c10::intrusive_ptr<ZeroCtaRuntime>& runtime) {
  auto& registry = runtime_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto group_found = registry.groups.find(key);
  if (group_found == registry.groups.end()) return;
  auto& slots = group_found->second.slots;
  const auto runtime_found = slots.find(runtime_slot);
  if (runtime_found != slots.end() &&
      runtime_found->second == runtime) {
    slots.erase(runtime_found);
  }
  if (slots.empty()) registry.groups.erase(group_found);
}

void close_runtime(const c10::intrusive_ptr<ProcessGroup>& group) {
  if (group == nullptr) return;
  ProcessGroup* const key = group.get();
  std::vector<std::pair<int64_t, c10::intrusive_ptr<ZeroCtaRuntime>>>
      runtimes;
  {
    auto& registry = runtime_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.groups.find(key);
    if (found == registry.groups.end()) return;
    runtimes.reserve(found->second.slots.size());
    for (const auto& entry : found->second.slots) runtimes.push_back(entry);
  }
  std::sort(
      runtimes.begin(),
      runtimes.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
      });
  for (const auto& [runtime_slot, runtime] : runtimes) {
    runtime->close();
    erase_runtime(key, runtime_slot, runtime);
  }
}

void close_all_runtimes() {
  std::vector<c10::intrusive_ptr<ProcessGroup>> groups;
  {
    auto& registry = runtime_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    groups.reserve(registry.groups.size());
    for (const auto& [key, runtime_group] : registry.groups) {
      (void)key;
      groups.push_back(runtime_group.group);
    }
  }
  for (const auto& group : groups) close_runtime(group);
}

c10::intrusive_ptr<c10d::Work> run_collective(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::intrusive_ptr<ZeroCtaRuntime>& runtime,
    const c10::intrusive_ptr<ZeroCtaPlan>& plan) {
  // Internal caller contract: plan offsets/peer counts are already validated,
  // and the current stream is ordered after input and metadata producers.
  // Direct callers wait this Work to submit post-processing and protect buffer
  // lifetimes. The Python public API submits post-processing before returning
  // None; its caller keeps data and prepared plans alive until GPU completion.
  TORCH_CHECK(runtime != nullptr && plan != nullptr, "null zero-CTA state");
  validate_shapes(input, output);
  const int64_t capacity = runtime->max_per_peer_slot_value;

  std::lock_guard<std::mutex> lock(runtime->mutex);
  TORCH_CHECK(!runtime->closed, "zero-CTA runtime was already closed");
  TORCH_CHECK(
      !runtime->collective_pending,
      "zero-CTA single workspace requires waiting the previous work");
  std::shared_ptr<at::cuda::CUDAEvent> completion;
  std::optional<PendingCollective> pending;
  if (capacity == 0) {
    completion = record_event(at::cuda::getCurrentCUDAStream());
  } else {
    TORCH_CHECK(runtime->workspace.has_value(), "zero-CTA workspace is missing");
    pending = enqueue_collective(*runtime, *runtime->workspace, input, *plan);
    runtime->collective_pending = true;
  }
  return c10::make_intrusive<ZeroCtaWork>(
      runtime, plan, input, output, completion, std::move(pending));
}

} // namespace nccl_cp_zero_cta

TORCH_LIBRARY(nccl_cp_zero_cta_full, m) {
  m.def(
      "make_device_tensor_gather_tiles(Tensor host) -> Tensor",
      nccl_cp_zero_cta::make_device_tensor_gather_tiles);
  m.def(
      "make_device_symmetric_gather_tiles(Tensor host) -> Tensor",
      nccl_cp_zero_cta::make_device_symmetric_gather_tiles);
  m.def(
      "make_device_reduce_ranges(Tensor host) -> Tensor",
      nccl_cp_zero_cta::make_device_reduce_ranges);
  m.class_<nccl_cp_zero_cta::ZeroCtaPlan>("Plan");
  m.def(
      "create_cast_plan(int[] send_token_counts, int[] remote_wait_peers, Tensor pack_gather_tiles, Tensor post_gather_tiles, int[] local_ready_signal_peers, int[] local_ready_wait_peers) -> __torch__.torch.classes.nccl_cp_zero_cta_full.Plan",
      nccl_cp_zero_cta::create_cast_plan);
  m.def(
      "create_reduce_plan(int[] send_token_counts, int[] remote_wait_peers, Tensor pack_gather_tiles, Tensor post_reduce_ranges, Tensor post_reduce_src_token_offsets, Tensor? local_reduce_ranges, Tensor? local_reduce_symmetric_srcs, int[] local_ready_signal_peers, int[] local_ready_wait_peers) -> __torch__.torch.classes.nccl_cp_zero_cta_full.Plan",
      nccl_cp_zero_cta::create_reduce_plan);
  m.class_<nccl_cp_zero_cta::ZeroCtaRuntime>("Runtime")
      .def(
          "max_per_peer_slot",
          &nccl_cp_zero_cta::ZeroCtaRuntime::max_per_peer_slot)
      .def(
          "node_slot_capacity",
          &nccl_cp_zero_cta::ZeroCtaRuntime::node_slot_capacity)
      .def(
          "nvl_domain_size",
          &nccl_cp_zero_cta::ZeroCtaRuntime::nvl_domain_size)
      .def(
          "max_per_token_bytes",
          &nccl_cp_zero_cta::ZeroCtaRuntime::max_per_token_bytes)
      .def("runtime_id", &nccl_cp_zero_cta::ZeroCtaRuntime::runtime_id);
  m.def(
      "get_or_create_runtime(int max_per_peer_slot, int nvl_domain_size, int max_per_token_bytes, int runtime_slot, __torch__.torch.classes.c10d.ProcessGroup group) -> __torch__.torch.classes.nccl_cp_zero_cta_full.Runtime",
      nccl_cp_zero_cta::get_or_create_runtime);
  m.def(
      "close_runtime(__torch__.torch.classes.c10d.ProcessGroup group) -> ()",
      nccl_cp_zero_cta::close_runtime);
  m.def("close_all_runtimes() -> ()", nccl_cp_zero_cta::close_all_runtimes);
  m.def(
      "run(Tensor input, Tensor(a!) output, __torch__.torch.classes.nccl_cp_zero_cta_full.Runtime runtime, __torch__.torch.classes.nccl_cp_zero_cta_full.Plan plan) -> __torch__.torch.classes.c10d.Work");
}

TORCH_LIBRARY_IMPL(nccl_cp_zero_cta_full, CUDA, m) {
  m.impl("run", nccl_cp_zero_cta::run_collective);
}
