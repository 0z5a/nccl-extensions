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

#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace nccl_cp_zero_cta {

namespace tensor_gather_segment {
constexpr int kSrcTokenStart = 0;
constexpr int kDstTokenStart = 1;
constexpr int kNTokens = 2;
constexpr int kFields = 3;
} // namespace tensor_gather_segment

namespace symmetric_gather_segment {
constexpr int kSrcRank = 0;
constexpr int kSrcTokenStart = 1;
constexpr int kDstTokenStart = 2;
constexpr int kNTokens = 3;
constexpr int kFields = 4;
} // namespace symmetric_gather_segment

namespace reduce_segment {
constexpr int kDstTokenStart = 0;
constexpr int kFirstSrcIndex = 1;
constexpr int kNSrcs = 2;
constexpr int kNTokens = 3;
constexpr int kFields = 4;
} // namespace reduce_segment

namespace reduce_range {
constexpr int kDstTokenStart = 0;
constexpr int kFirstSrcIndex = 1;
constexpr int kNSrcs = 2;
constexpr int kSrcRangeTokenOffset = 3;
constexpr int kNTokens = 4;
constexpr int kFields = 5;
} // namespace reduce_range

namespace symmetric_src {
constexpr int kRank = 0;
constexpr int kTokenStart = 1;
constexpr int kFields = 2;
} // namespace symmetric_src

struct TensorGatherMetadata {
  at::Tensor gather_tiles;
};

struct SymmetricGatherMetadata {
  at::Tensor gather_tiles;
  void** symmetric_buffer_ptrs;
};

struct TensorReduceMetadata {
  at::Tensor reduce_ranges;
  at::Tensor src_token_offsets;
};

struct SymmetricReduceMetadata {
  at::Tensor reduce_ranges;
  at::Tensor symmetric_srcs;
  void** symmetric_buffer_ptrs;
};

enum class GpuCompletionMarker {
  kPackReady,
  kPeerPostProcessDone,
  kRemoteDataReady,
  kLocalReady,
  kLocalConsumed,
  kRmaReady,
};

// Internal launch contract: metadata comes from a validated prepared plan;
// offsets/counts and symmetric peer pointers fit the corresponding buffers.
// Launch on the runtime/tensor device after data and metadata are ready, and
// retain tensors, metadata and windows until the submitted GPU work completes.
// Device kernels trust these bounds and do not perform cross-rank validation.
void launch_gather_from_tensor(
    const at::Tensor&, const at::Tensor&, const TensorGatherMetadata&,
    cudaStream_t);

void launch_gather_from_symmetric(
    const at::Tensor&, const SymmetricGatherMetadata&, cudaStream_t);

void launch_reduce_from_tensor(
    const at::Tensor&, const at::Tensor&, const TensorReduceMetadata&,
    cudaStream_t);

void launch_reduce_from_symmetric(
    const at::Tensor&, const at::Tensor&, int64_t, int64_t,
    const SymmetricReduceMetadata&, cudaStream_t);

void launch_gpu_completion_marker(GpuCompletionMarker, cudaStream_t);

} // namespace nccl_cp_zero_cta
