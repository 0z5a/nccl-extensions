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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace nccl_cp_zero_cta {
namespace {

constexpr int kWarpSize = 32;
constexpr int kGatherThreads = 512;
constexpr int kGatherTokensPerWarp = 2;
constexpr int kReduceThreads = 256;
constexpr int kReduceTokensPerBlock = kReduceThreads / kWarpSize;
constexpr int kReduceTokensPerWorkRange = 128;
constexpr int kReduceBlocksPerWorkRange =
    kReduceTokensPerWorkRange / kReduceTokensPerBlock;

__global__ void zero_cta_pack_ready_marker_kernel() {}
__global__ void zero_cta_peer_post_process_done_marker_kernel() {}
__global__ void zero_cta_remote_data_ready_marker_kernel() {}
__global__ void zero_cta_local_ready_marker_kernel() {}
__global__ void zero_cta_local_consumed_marker_kernel() {}
__global__ void zero_cta_rma_ready_marker_kernel() {}

int64_t per_token_elements(const at::Tensor& tensor) {
  int64_t result = 1;
  for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
    result *= tensor.size(dim);
  }
  return result;
}

bool is_bf162_compatible(const at::Tensor& tensor) {
  const auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  return tensor.stride(0) % 2 == 0 &&
      address % alignof(__nv_bfloat162) == 0;
}

template <typename scalar_t, typename unit_t, bool kSymmetric>
__global__ void gather_tiles_kernel(
    const scalar_t* __restrict__ input,
    void* const* __restrict__ symmetric_buffer_ptrs,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ gather_tiles,
    int64_t input_stride,
    int64_t output_stride,
    int64_t units_per_token) {
  constexpr int kFields =
      kSymmetric ? symmetric_gather_segment::kFields
                 : tensor_gather_segment::kFields;
  const int64_t* tile =
      gather_tiles + static_cast<int64_t>(blockIdx.x) * kFields;
  const int64_t input_start = tile[
      kSymmetric ? symmetric_gather_segment::kSrcTokenStart
                 : tensor_gather_segment::kSrcTokenStart];
  const int64_t output_start = tile[
      kSymmetric ? symmetric_gather_segment::kDstTokenStart
                 : tensor_gather_segment::kDstTokenStart];
  const int64_t tile_tokens = tile[
      kSymmetric ? symmetric_gather_segment::kNTokens
                 : tensor_gather_segment::kNTokens];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  constexpr int kLanesPerToken = kWarpSize / kGatherTokensPerWarp;
  const int token_in_warp = lane / kLanesPerToken;
  const int lane_in_token = lane % kLanesPerToken;
  const int token_in_tile = warp * kGatherTokensPerWarp + token_in_warp;
  if (token_in_tile >= tile_tokens) return;

  const unit_t* input_units;
  if constexpr (kSymmetric) {
    input_units = reinterpret_cast<const unit_t*>(
                      symmetric_buffer_ptrs[
                          tile[symmetric_gather_segment::kSrcRank]]) +
        (input_start + token_in_tile) * units_per_token;
  } else {
    input_units = reinterpret_cast<const unit_t*>(input) +
        (input_start + token_in_tile) * input_stride;
  }
  auto* output_units = reinterpret_cast<unit_t*>(output) +
      (output_start + token_in_tile) * output_stride;
  for (int64_t unit = lane_in_token; unit < units_per_token;
       unit += kLanesPerToken) {
    output_units[unit] = input_units[unit];
  }
}

template <typename unit_t>
struct ReduceOps;

template <>
struct ReduceOps<float> {
  using acc_t = float;

  __device__ static __forceinline__ acc_t zero() { return 0.0f; }
  __device__ static __forceinline__ acc_t load(float value) { return value; }
  __device__ static __forceinline__ void add(acc_t& acc, float value) {
    acc += value;
  }
  __device__ static __forceinline__ float store(acc_t value) { return value; }
};

template <>
struct ReduceOps<__nv_bfloat16> {
  using acc_t = float;

  __device__ static __forceinline__ acc_t zero() { return 0.0f; }
  __device__ static __forceinline__ acc_t load(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __forceinline__ void add(
      acc_t& acc, __nv_bfloat16 value) {
    acc += __bfloat162float(value);
  }
  __device__ static __forceinline__ __nv_bfloat16 store(acc_t value) {
    return __float2bfloat16(value);
  }
};

template <>
struct ReduceOps<__nv_bfloat162> {
  using acc_t = float2;

  __device__ static __forceinline__ acc_t zero() {
    return make_float2(0.0f, 0.0f);
  }
  __device__ static __forceinline__ acc_t load(__nv_bfloat162 value) {
    return __bfloat1622float2(value);
  }
  __device__ static __forceinline__ void add(
      acc_t& acc, __nv_bfloat162 value) {
    const float2 converted = __bfloat1622float2(value);
    acc.x += converted.x;
    acc.y += converted.y;
  }
  __device__ static __forceinline__ __nv_bfloat162 store(acc_t value) {
    return __floats2bfloat162_rn(value.x, value.y);
  }
};

template <typename unit_t, bool kSymmetric>
__global__ void reduce_ranges_kernel(
    const unit_t* __restrict__ input,
    void* const* __restrict__ symmetric_buffer_ptrs,
    unit_t* __restrict__ remote_output,
    unit_t* __restrict__ local_output,
    const int64_t* __restrict__ reduce_ranges,
    const int64_t* __restrict__ source_descriptors,
    int64_t input_stride,
    int64_t output_stride,
    int64_t local_output_start,
    int64_t local_output_end,
    int64_t units_per_token) {
  const int64_t* range = reduce_ranges +
      static_cast<int64_t>(blockIdx.y) * reduce_range::kFields;
  const int64_t output_start = range[reduce_range::kDstTokenStart];
  const int64_t first_source = range[reduce_range::kFirstSrcIndex];
  const int64_t source_count = range[reduce_range::kNSrcs];
  const int64_t source_range_offset =
      range[reduce_range::kSrcRangeTokenOffset];
  const int64_t range_tokens = range[reduce_range::kNTokens];
  const int64_t tile_offset =
      static_cast<int64_t>(blockIdx.x) * kReduceTokensPerBlock;
  if (tile_offset >= range_tokens) return;
  const int token_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t token_in_range = tile_offset + token_in_block;
  if (token_in_range >= range_tokens) return;

  const int64_t output_token = output_start + token_in_range;
  const int64_t input_token_offset =
      source_range_offset + token_in_range;
  auto* output = remote_output;
  if constexpr (kSymmetric) {
    // output_start is constant for every block handling this work range.
    if (output_start >= local_output_start && output_start < local_output_end) {
      output = local_output;
    }
  }
  using Ops = ReduceOps<unit_t>;
  for (int64_t unit = lane; unit < units_per_token; unit += kWarpSize) {
    typename Ops::acc_t acc = kSymmetric
        ? Ops::zero()
        : Ops::load(output[output_token * output_stride + unit]);
    for (int64_t source = 0; source < source_count; ++source) {
      const unit_t* source_input = input;
      int64_t input_token;
      if constexpr (kSymmetric) {
        const int64_t* source_desc =
            source_descriptors +
            (first_source + source) * symmetric_src::kFields;
        source_input =
            static_cast<const unit_t*>(
                symmetric_buffer_ptrs[
                    source_desc[symmetric_src::kRank]]);
        input_token =
            source_desc[symmetric_src::kTokenStart] + input_token_offset;
      } else {
        input_token =
            source_descriptors[first_source + source] + input_token_offset;
      }
      Ops::add(
          acc,
          source_input[input_token *
                           (kSymmetric ? units_per_token : input_stride) +
                       unit]);
    }
    output[output_token * output_stride + unit] = Ops::store(acc);
  }
}

template <typename scalar_t, bool kSymmetric>
void launch_gather_impl(
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& gather_tiles,
    void** symmetric_buffer_ptrs,
    cudaStream_t stream) {
  const int64_t token_elements =
      per_token_elements(kSymmetric ? output : input);
  if (gather_tiles.size(0) == 0 || token_elements == 0) return;
  const dim3 grid(static_cast<unsigned int>(gather_tiles.size(0)), 1, 1);
  constexpr int64_t elements_per_vector =
      std::is_same_v<scalar_t, __nv_bfloat16> ? 8 : 4;
  const bool vectorized = token_elements % elements_per_vector == 0 &&
      output.stride(0) % elements_per_vector == 0 &&
      reinterpret_cast<uintptr_t>(output.data_ptr()) % sizeof(uint4) == 0 &&
      (kSymmetric ||
       (input.stride(0) % elements_per_vector == 0 &&
        reinterpret_cast<uintptr_t>(input.data_ptr()) % sizeof(uint4) == 0));
  if (vectorized) {
    gather_tiles_kernel<scalar_t, uint4, kSymmetric>
        <<<grid, kGatherThreads, 0, stream>>>(
            kSymmetric
                ? nullptr
                : reinterpret_cast<const scalar_t*>(input.data_ptr()),
            symmetric_buffer_ptrs,
            reinterpret_cast<scalar_t*>(output.data_ptr()),
            gather_tiles.data_ptr<int64_t>(),
            kSymmetric ? 0 : input.stride(0) / elements_per_vector,
            output.stride(0) / elements_per_vector,
            token_elements / elements_per_vector);
    return;
  }
  gather_tiles_kernel<scalar_t, scalar_t, kSymmetric>
      <<<grid, kGatherThreads, 0, stream>>>(
          kSymmetric
              ? nullptr
              : reinterpret_cast<const scalar_t*>(input.data_ptr()),
          symmetric_buffer_ptrs,
          reinterpret_cast<scalar_t*>(output.data_ptr()),
          gather_tiles.data_ptr<int64_t>(),
          kSymmetric ? 0 : input.stride(0),
          output.stride(0),
          token_elements);
}

template <typename unit_t, bool kSymmetric>
void launch_reduce_kernel(
    const at::Tensor& input,
    const at::Tensor& remote_output,
    const at::Tensor& local_output,
    int64_t local_begin,
    int64_t local_end,
    const at::Tensor& reduce_ranges,
    const at::Tensor& source_descriptors,
    void** symmetric_buffer_ptrs,
    int64_t token_elements,
    int64_t elements_per_unit,
    cudaStream_t stream) {
  const dim3 grid(
      kReduceBlocksPerWorkRange,
      static_cast<unsigned int>(reduce_ranges.size(0)),
      1);
  reduce_ranges_kernel<unit_t, kSymmetric>
      <<<grid, kReduceThreads, 0, stream>>>(
      kSymmetric ? nullptr
                 : reinterpret_cast<const unit_t*>(input.data_ptr()),
      symmetric_buffer_ptrs,
      reinterpret_cast<unit_t*>(remote_output.data_ptr()),
      kSymmetric ? reinterpret_cast<unit_t*>(local_output.data_ptr())
                 : nullptr,
      reduce_ranges.data_ptr<int64_t>(),
      source_descriptors.data_ptr<int64_t>(),
      kSymmetric ? 0 : input.stride(0) / elements_per_unit,
      remote_output.stride(0) / elements_per_unit,
      local_begin,
      local_end,
      token_elements / elements_per_unit);
}

template <typename scalar_t, bool kSymmetric>
void launch_reduce_impl(
    const at::Tensor& input,
    const at::Tensor& remote_output,
    const at::Tensor& local_output,
    int64_t local_begin,
    int64_t local_end,
    const at::Tensor& reduce_ranges,
    const at::Tensor& source_descriptors,
    void** symmetric_buffer_ptrs,
    cudaStream_t stream) {
  const int64_t token_elements = per_token_elements(remote_output);
  if (reduce_ranges.size(0) == 0 || token_elements == 0) return;
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
    if (token_elements % 2 == 0 &&
        is_bf162_compatible(remote_output) &&
        is_bf162_compatible(kSymmetric ? local_output : input)) {
      launch_reduce_kernel<__nv_bfloat162, kSymmetric>(
          input,
          remote_output,
          local_output,
          local_begin,
          local_end,
          reduce_ranges,
          source_descriptors,
          symmetric_buffer_ptrs,
          token_elements,
          2,
          stream);
      return;
    }
  }
  launch_reduce_kernel<scalar_t, kSymmetric>(
      input,
      remote_output,
      local_output,
      local_begin,
      local_end,
      reduce_ranges,
      source_descriptors,
      symmetric_buffer_ptrs,
      token_elements,
      1,
      stream);
}

void check_launch() {
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "zero-CTA CUDA kernel launch failed: ",
              cudaGetErrorString(error));
}

template <typename F>
void dispatch(const at::Tensor& tensor, F&& launch) {
  if (tensor.scalar_type() == at::kFloat) {
    launch(float{});
  } else if (tensor.scalar_type() == at::kBFloat16) {
    launch(__nv_bfloat16{});
  } else {
    TORCH_CHECK(false, "zero-CTA supports only float32 and bfloat16");
  }
  check_launch();
}

} // namespace

void launch_gather_from_tensor(
    const at::Tensor& input,
    const at::Tensor& output,
    const TensorGatherMetadata& meta,
    cudaStream_t stream) {
  dispatch(input, [&](auto scalar) {
    launch_gather_impl<decltype(scalar), false>(
        input, output, meta.gather_tiles, nullptr, stream);
  });
}

void launch_gather_from_symmetric(
    const at::Tensor& output,
    const SymmetricGatherMetadata& meta,
    cudaStream_t stream) {
  dispatch(output, [&](auto scalar) {
    launch_gather_impl<decltype(scalar), true>(
        {}, output, meta.gather_tiles, meta.symmetric_buffer_ptrs, stream);
  });
}

void launch_reduce_from_tensor(
    const at::Tensor& input,
    const at::Tensor& output,
    const TensorReduceMetadata& meta,
    cudaStream_t stream) {
  dispatch(input, [&](auto scalar) {
    launch_reduce_impl<decltype(scalar), false>(
        input, output, {}, 0, 0, meta.reduce_ranges,
        meta.src_token_offsets, nullptr, stream);
  });
}

void launch_reduce_from_symmetric(
    const at::Tensor& remote_output,
    const at::Tensor& local_output,
    int64_t local_output_begin,
    int64_t local_output_end,
    const SymmetricReduceMetadata& meta,
    cudaStream_t stream) {
  dispatch(remote_output, [&](auto scalar) {
    launch_reduce_impl<decltype(scalar), true>(
        {}, remote_output, local_output, local_output_begin, local_output_end,
        meta.reduce_ranges, meta.symmetric_srcs,
        meta.symmetric_buffer_ptrs, stream);
  });
}

void launch_gpu_completion_marker(
    GpuCompletionMarker marker,
    cudaStream_t stream) {
  switch (marker) {
    case GpuCompletionMarker::kPackReady:
      zero_cta_pack_ready_marker_kernel<<<1, 1, 0, stream>>>();
      break;
    case GpuCompletionMarker::kPeerPostProcessDone:
      zero_cta_peer_post_process_done_marker_kernel<<<1, 1, 0, stream>>>();
      break;
    case GpuCompletionMarker::kRemoteDataReady:
      zero_cta_remote_data_ready_marker_kernel<<<1, 1, 0, stream>>>();
      break;
    case GpuCompletionMarker::kLocalReady:
      zero_cta_local_ready_marker_kernel<<<1, 1, 0, stream>>>();
      break;
    case GpuCompletionMarker::kLocalConsumed:
      zero_cta_local_consumed_marker_kernel<<<1, 1, 0, stream>>>();
      break;
    case GpuCompletionMarker::kRmaReady:
      zero_cta_rma_ready_marker_kernel<<<1, 1, 0, stream>>>();
      break;
  }
  check_launch();
}

} // namespace nccl_cp_zero_cta
