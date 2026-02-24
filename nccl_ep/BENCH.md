# NCCL EP Performance Benchmark

## Quick Start

## Binary: build and test nccl_ep/ep_bench

### Discover compute capabilities
Use `nvidia-smi` command to detect the compute capabilities of your NVIDIA GPU.
For example, on Hopper system with `compute_cap` of `90`, the output looks like below:
```bash
$ nvidia-smi --query-gpu=compute_cap --format=csv
compute_cap
9.0
...
```

### Set the environment
```bash
export COMPUTE_CAP=<discovered compute_cap>
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export NCCL_HOME=/path/to/nccl/build # The desired NCCL build
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:${MPI_HOME}/bin:$PATH"
```

### Build and Run
```bash
# Build (from nccl root; builds lib + ep_test + ep_bench)
make -C contrib/nccl_ep MPI=1 NVCC_GENCODE="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}" BUILDDIR=$NCCL_HOME

# Run Low Latency benchmark with validation (8 GPUs, single node)
mpirun -np 8 --oversubscribe --allow-run-as-root -x LD_LIBRARY_PATH $NCCL_HOME/test/nccl_ep/ep_bench --algorithm low-latency --validate

# Run High Throughput benchmark with validation (16 GPUs, multi-node)
mpirun -np 16 -x LD_LIBRARY_PATH $NCCL_HOME/test/nccl_ep/ep_bench --algorithm high-throughput --validate

# Common options
$NCCL_HOME/test/nccl_ep/ep_bench --algorithm low-latency --tokens 256 --hidden 7168 --top-k 8 --experts 256
$NCCL_HOME/test/nccl_ep/ep_bench --algorithm high-throughput --tokens 4096
```

---

## Overview

This benchmark measures the performance of NCCL EP (Expert Parallelism) dispatch and combine operations for MoE (Mixture of Experts) workloads. It supports two algorithm modes optimized for different scenarios:

- **Low Latency**: Optimized for small batch sizes and latency-sensitive inference
- **High Throughput**: Optimized for large batch sizes and maximum bandwidth utilization

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithm <mode>` | Algorithm mode: `low-latency` (or `ll`), `high-throughput` (or `ht`) | `low-latency` |
| `--tokens <num>` | Number of tokens per rank | LL=128, HT=4096 |
| `--hidden <num>` | Hidden dimension size | 7168 |
| `--top-k <num>` | Number of experts selected per token | 8 |
| `--experts <num>` | Total number of experts (must be divisible by num_ranks) | 256 |
| `--warmup <num>` | Number of warmup iterations | 10 |
| `--iters <num>` | Number of benchmark iterations | 50 |
| `--use-fp8` | Use FP8 for dispatch (default: BF16) | disabled |
| `--profile` | Enable NVTX profiling mode for use with `nsys` | disabled |
| `--disable-nvlink` | Force RDMA for intranode communication (low-latency mode only) | disabled |
| `--validate` | Enable data validation for dispatch/combine correctness | disabled |
| `--dynamic-tokens` | Enable dynamic token allocation via `NCCL_EP_AUTO` (HT only, **not yet supported**) | disabled |
| `--help` | Show help message | - |

## Algorithm Modes

### Low Latency Mode (`--algorithm low-latency`)

Designed for latency-sensitive inference with small batch sizes.

**Characteristics:**
- Per token-expert pair communication
- Supports both FP8 and BF16 dispatch
- Combine always uses BF16
- Requires `num_qp_per_rank >= num_local_experts`

**Throughput Calculation** (matches DeepEP `test_low_latency.py`):
- **FP8 dispatch bytes**: `(hidden + hidden/128*4 + 16)` per selection
- **BF16 dispatch bytes**: `hidden * 2` per selection
- **Combine bytes**: `hidden * 2` per selection (always BF16)

### High Throughput Mode (`--algorithm high-throughput`)

Designed for maximum bandwidth utilization with large batch sizes.

**Characteristics:**
- Batched token communication with RDMA/NVLink path separation
- Optimized for multi-node configurations
- Uses auto QP configuration

**Throughput Calculation** (matches DeepEP `test_internode.py`):
- **RDMA bytes**: Tokens sent to remote nodes × `hidden * 2`
- **NVL bytes**: Tokens sent within local node × `hidden * 2`
- **FP8 factor**: `(1 + 4/128) / 2 ≈ 0.516` of BF16 bytes

## Output Format

### Per-Rank Results

```
[Rank 0] Dispatch:         avg=50.00 us, min=48.00 us, max=52.00 us, throughput=25.00 GB/s
[Rank 0] Combine:          avg=45.00 us, min=43.00 us, max=47.00 us, throughput=30.00 GB/s
[Rank 0] Dispatch+Combine: avg=95.00 us, min=91.00 us, max=99.00 us, throughput=27.50 GB/s
```

### Summary (Low Latency)

```
=== Summary (Low Latency, across 8 ranks) ===
Dispatch (BF16):  avg=50.00 us, min=48.00 us, max=52.00 us
                  throughput: avg=25.00 GB/s, min=24.00 GB/s (rank 3), max=26.00 GB/s (rank 1)
Combine (BF16):   avg=45.00 us, min=43.00 us, max=47.00 us
                  throughput: avg=30.00 GB/s, min=29.00 GB/s (rank 5), max=31.00 GB/s (rank 2)
Total (D+C):      avg=95.00 us, min=91.00 us, max=99.00 us
                  throughput: avg=27.50 GB/s, min=26.00 GB/s (rank 4), max=29.00 GB/s (rank 0)

Byte counts: dispatch=1.80 MB (BF16), combine=1.80 MB (BF16), selections=1024
```

### Summary (High Throughput)

```
=== Summary (High Throughput BF16, across 16 ranks) ===
Dispatch:         avg=200.00 us, min=190.00 us, max=210.00 us
                  throughput: avg=80.00 GB/s (RDMA: 50.00, NVL: 30.00 GB/s)
                  min=75.00 GB/s (rank 5), max=85.00 GB/s (rank 2)
Combine:          avg=195.00 us, min=185.00 us, max=205.00 us
                  throughput: avg=82.00 GB/s (RDMA: 51.00, NVL: 31.00 GB/s)
                  min=77.00 GB/s (rank 6), max=87.00 GB/s (rank 1)
Total (D+C):      avg=395.00 us, min=375.00 us, max=415.00 us
                  throughput: avg=81.00 GB/s
                  min=76.00 GB/s (rank 5), max=86.00 GB/s (rank 2)

Byte breakdown (per rank avg): RDMA=8.00 MB (500 tokens), NVL=4.00 MB (250 tokens)
```

## NVTX Profiling

Use `--profile` to enable NVTX markers for detailed kernel analysis with NVIDIA Nsight Systems:

```bash
nsys profile -t cuda,nvtx mpirun -np 8 $NCCL_HOME/test/nccl_ep/ep_bench --algorithm low-latency --profile
```

This generates labeled ranges for:
- `Dispatch Benchmark` - Individual dispatch iterations
- `Combine Benchmark` - Individual combine iterations
- `Dispatch+Combine Benchmark` - Combined operation iterations

## Configuration Details

### Group Configuration

| Parameter | Low Latency | High Throughput |
|-----------|-------------|-----------------|
| `num_qp_per_rank` | `num_local_experts` | `0` (auto) |
| `nvl_buffer_size` | `0` (auto) | `0` (auto) |
| `rdma_buffer_size` | `0` (auto) | `0` (auto) |
| `num_channels` | `0` (auto) | `0` (auto) |

### Tensor Configuration

**Low Latency Mode:**
- Input: `[num_tokens × hidden]` tokens + `[num_tokens × top_k]` topk_idx
- Output: `[num_local_experts × num_recv_tokens × hidden]` packed expert tokens, where `num_recv_tokens = num_ranks × max_tokens_per_rank`

**High Throughput Mode:**
- Input: `[num_tokens × hidden]` tokens + `[num_tokens × top_k]` topk_weights + `[num_tokens × top_k]` topk_idx
- Output: `[num_recv_tokens × hidden]` received tokens, where `num_recv_tokens = num_ranks × max_tokens_per_rank`
- Separate RDMA and NVLink communication paths

**Tensor Creation:**
- Uses `ncclEpTensorCreate()` API for all dispatch/combine tensors
- Allocator callbacks (`cudaMalloc`/`cudaFree`) passed to `ncclEpCreateGroup`
- Tensors cleaned up with `ncclEpTensorDestroy()` before group destruction

## Example Configurations

### DeepSeek-V3 Style (256 experts, top-8)

```bash
# Single node (8 GPUs) - Low Latency with validation
mpirun -np 8 $NCCL_HOME/test/nccl_ep/ep_bench \
    --algorithm low-latency \
    --tokens 128 \
    --hidden 7168 \
    --top-k 8 \
    --experts 256 \
    --validate

# Multi-node (32 GPUs across 4 nodes) - High Throughput with validation
mpirun -np 32 $NCCL_HOME/test/nccl_ep/ep_bench \
    --algorithm high-throughput \
    --tokens 4096 \
    --hidden 7168 \
    --top-k 8 \
    --experts 256 \
    --validate

# High Throughput with FP8 (performance only)
mpirun -np 32 $NCCL_HOME/test/nccl_ep/ep_bench \
    --algorithm high-throughput \
    --tokens 4096 \
    --hidden 7168 \
    --top-k 8 \
    --experts 256 \
    --use-fp8
```

### Mixtral Style (8 experts, top-2)

```bash
mpirun -np 8 $NCCL_HOME/test/nccl_ep/ep_bench \
    --algorithm low-latency \
    --tokens 512 \
    --hidden 4096 \
    --top-k 2 \
    --experts 8
```

## Data Validation

The `--validate` flag enables correctness checking for dispatch and combine operations, following DeepEP's validation methodology.

### How It Works

1. **Token Initialization**: Each rank fills tokens with value `(rank - 128)` for BF16-safe representation
2. **Weights Initialization**: All `topk_weights` set to 1.0 for simpler validation math
3. **Dispatch Validation**: Verify received tokens contain expected source rank values
4. **Combine Validation**: Uses DeepEP formula: `check = combined / is_token_in_rank.sum()`
   - `is_token_in_rank.sum()` = count of unique ranks each token was sent to
   - For HT deterministic: all top_k experts on same rank → divide by 1
   - For LL random: experts spread across ranks → divide by unique rank count

### Example Output

```
=== Data Validation ===
Dispatch validation: PASSED
Combine validation:  PASSED (max_diff=0.0000)

Global validation: Dispatch=PASSED, Combine=PASSED
```

## Known Limitations

1. **Dynamic Token Allocation (`--dynamic-tokens`)**
   - `NCCL_EP_AUTO` for `max_tokens_per_rank` is not yet supported in the current release
   - `max_tokens_per_rank` must be explicitly set to the per-rank batch size for both LL and HT modes
   - The `--dynamic-tokens` flag is reserved for a future HT mode feature

2. **LL Mode Random Routing**
   - Uses random topk with -1 masking for invalid experts
   - Works correctly with validation enabled

## Methodology Notes

1. **Timing**: Uses CUDA events for accurate GPU timing
2. **Warmup**: Configurable warmup iterations (default 10) are excluded from measurements
3. **Trimming**: Top/bottom 10% of samples are trimmed for stable averages
4. **Synchronization**: MPI barriers ensure synchronized timing across ranks
5. **Token Generation**: Random top-k expert selection with 10 masked entries to simulate realistic workloads (LL mode), deterministic routing (HT mode)

## Comparison with DeepEP

This benchmark follows the same throughput calculation methodology as DeepEP tests:

| Mode | NCCL EP Benchmark | DeepEP Test |
|------|-------------------|-------------|
| Low Latency | `ep_bench --algorithm low-latency` | `test_low_latency.py` |
| High Throughput | `ep_bench --algorithm high-throughput` | `test_internode.py` |

The byte calculations and throughput metrics are designed to be directly comparable.

## Benchmark Results

The following results compare NCCL EP with DeepEP (NCCL backend) for Low-Latency kernels using **BF16 dispatch and combine**.

**Test Configuration:**
- Hidden: 7168
- Top-k: 8
- Experts: 256
- Tokens: 128 per rank
- 8 GPUs per node

### RDMA + NVLink Mode

Intra-node communication uses NVLink, inter-node uses RDMA.

| Nodes | NCCL EP ||| DeepEP (NCCL) |||
|-------|---------|---------|---------|---------|---------|---------|
| | Dispatch Lat (μs) | Combine Lat (μs) | Dispatch BW (GB/s) | Combine BW (GB/s) | Dispatch Lat (μs) | Combine Lat (μs) | Dispatch BW (GB/s) | Combine BW (GB/s) |
| 1 | 65.34 | 77.19 | 222.69 | 188.51 | 62.50 | 69.57 | 232.61 | 208.97 |
| 2 | 194.70 | 211.63 | 74.74 | 68.76 | 212.93 | 211.50 | 69.88 | 70.61 |
| 4 | 257.79 | 308.70 | 56.45 | 47.14 | 269.50 | 287.30 | 53.99 | 50.64 |
| 8 | 301.97 | 352.99 | 48.19 | 41.22 | 348.50 | 368.96 | 41.74 | 39.42 |

### Pure RDMA Mode

All communication uses RDMA (NVLink disabled via `--disable-nvlink`).

| Nodes | NCCL EP ||| DeepEP (NCCL) |||
|-------|---------|---------|---------|---------|---------|---------|
| | Dispatch Lat (μs) | Combine Lat (μs) | Dispatch BW (GB/s) | Combine BW (GB/s) | Dispatch Lat (μs) | Combine Lat (μs) | Dispatch BW (GB/s) | Combine BW (GB/s) |
| 1* | 4364.18 | 6358.43 | 3.33 | 2.29 | 295.68 | 314.98 | 49.17 | 46.16 |
| 2 | 261.23 | 290.20 | 55.70 | 50.14 | 325.88 | 337.35 | 44.65 | 43.12 |
| 4 | 294.30 | 336.54 | 49.44 | 43.24 | 342.42 | 365.07 | 42.48 | 39.84 |
| 8 | 328.55 | 388.26 | 44.29 | 37.48 | 382.39 | 390.36 | 38.04 | 37.25 |

\* Single-node RDMA mode for NCCL EP shows degraded performance due to lack of real RDMA hardware in loopback configuration.

### Key Observations

1. **RDMA + NVLink Mode**: NCCL EP shows competitive or better performance compared to DeepEP across all node counts.

2. **Pure RDMA Mode (multi-node)**: NCCL EP outperforms DeepEP by 10-25% in throughput for 2, 4, and 8 node configurations.

3. **Single-node NVLink**: Both implementations achieve ~200+ GB/s throughput, near NVLink bandwidth limits.

4. **Scaling**: Both implementations show expected throughput reduction as node count increases due to increased inter-node communication overhead.
