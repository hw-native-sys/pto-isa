# High-Performance GEMM AllReduce Fusion Operator Example

## Overview

This example demonstrates how to implement a multi-device GEMM + AllReduce fusion operator using PTO. It adopts a dual-stream (Compute Stream + Comm Stream) compute-communication overlap design, performing AllReduce via PTO communication ISA over HCCL RDMA windows.

## Supported AI Processors

- A2/A3

## Directory Structure

```
kernels/manual/a2a3/gemm_ar/
├── CMakeLists.txt              # Build config (3 targets: cube kernel, vec kernel, host exe)
├── run.sh                      # One-click build+run script (auto HCCL_BUFFSIZE, MPI discovery)
├── gemm_ar_config.h            # Global parameter config (matrix dims, tile sizes, block counts)
├── main.cpp                    # Entry: MPI init, data gen, HCCL init, window alloc, perf measurement, validation
├── gemm_compute_kernel.cpp     # GEMM compute kernel (Cube arch, L0C FP32→GM FP16 auto cast)
├── comm_kernel.cpp             # Comm kernel (Vector arch, single-kernel two-phase AllReduce)
├── common.hpp                  # HcclRemotePtr device-side wrapper (RDMA window address translation)
├── hccl_context.h              # HcclDeviceContext struct (per-rank RDMA window addresses)
├── ready_queue.hpp             # Multi-block lock-free tile queue (compute→comm signaling)
└── comm_mpi.h                  # MPI dynamic loading wrapper (dlopen/dlsym, no hard-link dependency)
```

## Operator Description

### Computation

This example implements multi-device GEMM + AllReduce:

$$
C_{final} = \sum_{i=0}^{nranks-1} A_i \times B
$$

Where:

- `A_i` is `M×K` (independent per rank)
- `B` is `K×N` (shared across all ranks)
- `C_i` is `M×N` (local GEMM result per rank)
- `C_final` is `M×N` (final output after AllReduce)

The default reference configuration in `gemm_ar_config.h` is `M=5416, K=6144, N=1408`, running on 8 devices.

### Specifications

| Item | Value |
| --- | --- |
| OpType | `GEMM + AllReduce` |
| Input | `A_i`: `M×K`, `float16`, `ND` (independent per rank); `B`: `K×N`, `float16`, `DN` (shared) |
| Output | `C_final`: `M×N`, `float16`, `ND` (AllReduce result) |
| Compute Kernel Name | `GemmComputeKernel` (Cube arch, `dav-c220-cube`) |
| Comm Kernel Name | `GemmCommAllKernel` (Vector arch, `dav-c220-vec`) |

## Optimization Details

This example uses the 8-device Ascend 910B (A2/A3) platform for performance validation. The 910B adopts a disaggregated architecture: 24 AICs (Cube Cores) handle matrix computation, 24 AIVs (Vector Cores) handle communication transfers. AICs and AIVs are physically independent and can run fully in parallel.

- **Dual-stream compute-communication overlap**: The compute kernel runs on the Compute Stream (24 AICs), the comm kernel runs on the Comm Stream (24 AIVs), achieving compute-communication parallelism through per-tile signaling.
- **ReduceScatter + AllGather two-phase communication**: The RS phase uses `TPUT<AtomicAdd>` to accumulate directly into the owner rank's `reduced_output`; hardware atomic adds perform the reduction on the target side, eliminating a separate Reduce phase. The AG phase broadcasts the reduced result from the owner rank to all other ranks.
- **Block Swizzle**: The compute kernel uses a zigzag tile traversal order (odd rows reversed), improving L1 cache reuse of the B matrix across adjacent tiles.
- **Two-level double-buffered pipeline**: L1 cache (stepK=4 batched TLOAD) + L0 double buffer (ping/pong), overlapping DMA transfers with Cube computation as much as possible.
- **Lock-free Ready Queue**: Each AIC has an independent queue (single-producer single-consumer). AIVs poll non-blockingly via the `TTEST` hardware instruction; when no data is ready, `TWAIT` hardware wait avoids busy-spinning.
- **RS double-buffered pipeline**: The RS phase of the comm kernel uses ping/pong tiles to double-buffer `TLOAD` and `TSTORE<AtomicAdd>`, overlapping the current tile's TLOAD with the previous tile's TSTORE.
- **AG row-level flattened decomposition**: The AG phase flattens all work into row granularity (`my_tile_count × (nranks-1) × G_BASE_M` rows), distributing evenly across all AIVs, eliminating ±1 load imbalance from tile-level assignment.

## Tiling Parameters

| Parameter | Value |
| --- | --- |
| `M` (original) | 5416 |
| `K` | 6144 |
| `N` (original) | 1408 |
| `M` (padded) | 5504 |
| `N` (padded) | 1536 |
| `baseM` | 128 |
| `baseK` | 64 |
| `baseN` | 256 |
| `stepKa` | 4 |
| `stepKb` | 4 |
| Tile count | 258 (43×6) |
| `COMPUTE_BLOCK_NUM` | 24 |
| `COMM_BLOCK_NUM` | 24 |

## Overall Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Compute Stream (24 AIC)                Comm Stream (24 AIV)                │
│                                                                              │
│  GemmComputeKernel:                     GemmCommAllKernel:                   │
│  ┌─────────────────────────┐            ┌──────────────────────────────┐     │
│  │ for each tile:          │            │ Phase 1: ReduceScatter       │     │
│  │   K-loop (L1→L0→Cube)  │            │   Poll Ready Queue           │     │
│  │   TSTORE → gemm_output │──Ready──→  │   TLOAD tile from gemm_output│     │
│  │   pipe_barrier(ALL)     │  Queue     │   TSTORE<AtomicAdd> → owner  │     │
│  │   Enqueue tile_idx      │            │       (ping/pong dbl-buffer) │     │
│  └─────────────────────────┘            │         ↓                    │     │
│                                          │   DeviceBarrier (cross-rank) │     │
│                                          │         ↓                    │     │
│                                          │ Phase 2: AllGather           │     │
│                                          │   Row-level flat assignment  │     │
│                                          │   TLOAD → TSTORE to remotes │     │
│                                          └──────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Compute Kernel Details

```
Time →
L1 (MTE2):  [TLOAD A0,B0]                [TLOAD A1,B1]              ...
L0 (MTE1):       [TEXTRACT k0] [k1] [k2] [k3] [TEXTRACT k0'] ...
Cube (M):             [TMATMUL k0] [ACC k1] [ACC k2] [ACC k3] [TMATMUL k0'] ...
                      ↑ Three-stage pipeline, fully parallel ↑
```

Each AIC handles a set of tiles (assigned by `block_idx × tiles_per_block`). For each tile:

1. **Block Swizzle mapping**: Remaps linear tile indices to zigzag traversal order (odd rows reversed), so consecutive tiles share B matrix columns, improving L1 reuse.
2. **K-loop**: Every `stepKa=4` iterations, one TLOAD is issued (L1 cache optimization); each iteration uses TEXTRACT to extract a single K-slice to L0, then TMATMUL/TMATMUL_ACC accumulates.
3. **TSTORE**: L0C FP32 is auto-cast to FP16 via FixPipe, then written to `gemm_output`.
4. **pipe_barrier(PIPE_ALL)**: Ensures GM writes are complete.
5. **MultiBlockEnqueueFast**: Enqueues `tile_idx` to notify the comm kernel.

## Comm Kernel Details

### Phase 1: ReduceScatter

Each AIV is 1:1 bound to an AIC's Ready Queue. The AIV polls the queue non-blockingly via the `TTEST` hardware instruction. Upon receiving a ready tile:

1. **TLOAD** from `gemm_output` into UB (ping/pong double buffer)
2. **TSTORE\<AtomicAdd\>** to the owner rank's `reduced_output` (local or remote via RDMA)

The double-buffered pipeline overlaps the current tile's TLOAD with the previous tile's TSTORE. When the queue has no data, the AIV uses `TWAIT` hardware wait to avoid busy-spinning.

Tile ownership is determined by `tile_idx % nranks`—this ensures each rank's tiles are evenly distributed across all ranks.

### DeviceBarrier: Two-Level Device-Side Synchronization

```
DeviceBarrier(phase):
  pipe_barrier(PIPE_ALL)                    // Ensure this block's pipeline is flushed

  if block_idx == 0:                        // Only block 0 performs cross-rank signaling
    for each remote rank r:
      TNOTIFY(remote signal_matrix[phase][my_rank], 1, AtomicAdd)   // Write to remote
    for each remote rank r:
      TWAIT(local signal_matrix[phase][r], 1, GE)                   // Wait for remote
    TNOTIFY(local_broadcast_flag[phase], 1, Set)                    // Notify other blocks on this rank
  else:
    TWAIT(local_broadcast_flag[phase], 1, GE)                       // Wait for block 0's broadcast

  pipe_barrier(PIPE_ALL)
```

### Phase 2: AllGather

All AG work is flattened into row granularity:

```
total_rows = my_tile_count × (nranks - 1) × G_BASE_M
rows_per_block = ceil(total_rows / num_comm_blocks)
```

Each AIV handles rows in the range `[row_start, row_end)`. For each contiguous row segment, the original `(tile_owner_idx, remote_rank, row_in_tile)` is recovered, then:

1. **TLOAD** from local `reduced_output` into UB
2. **TSTORE** to the remote rank's `reduced_output` (via RDMA)

Row-level assignment ensures all AIVs transfer strictly equal amounts of data, eliminating load imbalance that occurs with tile-level assignment when the tile count is not divisible by the number of AIVs.

## Ready Queue Mechanism

```
┌─────────────┐         ┌─────────────┐
│  AIC 0      │         │  AIV 0      │
│  (Compute)  │──Queue──│  (Comm)     │
│  block_idx=0│   0     │  block_idx=0│
└─────────────┘         └─────────────┘
┌─────────────┐         ┌─────────────┐
│  AIC 1      │         │  AIV 1      │
│  (Compute)  │──Queue──│  (Comm)     │
│  block_idx=1│   1     │  block_idx=1│
└─────────────┘         └─────────────┘
      ...                     ...
┌─────────────┐         ┌─────────────┐
│  AIC 23     │         │  AIV 23     │
│  (Compute)  │──Queue──│  (Comm)     │
│  block_idx=23│  23    │  block_idx=23│
└─────────────┘         └─────────────┘
```

- Each queue is a 64-byte aligned `PerBlockQueue` struct containing `count` (producer-incremented) and `data[]` (tile index array).
- **Producer** (AIC): `PerBlockQueueEnqueueFast` writes to `data[slot]` and increments `count`, using `dcci` to flush cache for AIV visibility.
- **Consumer** (AIV): `PerBlockQueueTryDequeue` uses the `TTEST` hardware instruction to check `count >= head+1`; returns -1 when no data is available. Uses `TWAIT` hardware wait for prolonged idle periods.
- Single-producer single-consumer design; no atomic operations required.

## Memory Layout and HCCL Window

Only buffers written by remote TPUT/TNOTIFY need to reside in the HCCL RDMA window; buffers with local-only read/write use standard `aclrtMalloc`.

| Buffer | Size | Location | Reason |
| --- | --- | --- | --- |
| `reduced_output` | M × N × 2B | **HCCL window** | RS AtomicAdd + AG remote TPUT writes (FP16) |
| `signal_matrix` | (MAX_RANKS+1) × 4B, aligned 64B | **HCCL window** | DeviceBarrier cross-rank TNOTIFY writes |
| `gemm_output` | M × N × 2B | **aclrtMalloc** | Local read/write only (FP16) |
| `src0_dev`, `src1_dev` | Input matrices (FP16) | **aclrtMalloc** | Local read/write only |

Window size is controlled by the `HCCL_BUFFSIZE` environment variable. `run.sh` computes it automatically: `M × N × 2 / 1MB + 64MB`.

## Measured Performance (Reference)

The following data was measured on 8 Ascend 910B devices with parameters M=5416, K=6144, N=1408 (padded 5504×1536), 258 tiles (43×6). Each rank computes the full GEMM C_i = A_i × B, and AllReduce sums the 8 C_i results.

| Metric | Value |
| --- | --- |
| Compute-only | 365 us (257 TFLOPS, 98%) |
| Sequential | 743 us (compute 368 us + comm 375 us @ 74 GB/s) |
| Pipelined | **631 us** (speedup 1.18x, overlap 31%) |
| Throughput | 1189 TFLOPS (total) |

### What These Numbers Mean

- **Compute-only**: Pure GEMM time (no communication), reflecting the upper bound of single-device Cube utilization. 257 TFLOPS at 98% of theoretical peak indicates the compute kernel is highly optimized.
- **Sequential**: Compute→communication executed serially with no overlap. Total time = compute time + communication time.
- **Pipelined**: Compute and communication running in dual-stream parallel. 631 us vs. Sequential 743 us yields 1.18× speedup with 31% overlap efficiency.
- **Speedup**: Sequential / Pipelined; higher values indicate more effective compute-communication overlap.
- **Overlap eff**: Percentage of the shorter phase's time saved by overlapping. 31% means roughly one-third of the communication time is successfully hidden behind computation.

### Optimization History

| Optimization | Pipelined (us) | Gain | Conclusion |
| --- | --- | --- | --- |
| Baseline | 808 | — | — |
| Block Swizzle | 793 | -1.8% | **Kept** |
| RS AtomicAdd eliminating Reduce phase | 736 | -6.6% | **Kept** |
| AG row-level flattened decomposition | 623 | -15.4% | **Kept** |
| 48 AIV (RS skip + AG participation) | 639 | RS 24 AIVs only, AG 48 AIVs | **Reverted** (AIC interference) |
| 48 AIV dual-queue (1 AIC : 2 AIV) | 667 | RS/AG both 48 AIVs | **Reverted** (AIC interference) |

## Performance Tuning Guide (How to Tune This Kernel)

### 1) Start with Multi-Core Partitioning

Each AIC is assigned a tile subset via `block_idx × tiles_per_block`; blocks do not interfere with each other.

Checklist:

- Adjust `COMPUTE_BLOCK_NUM` so each block handles a roughly equal number of tiles.
- For different matrix shapes, recalculate the total tile count: `G_NUM_TILES = (M_padded/128) × (N_padded/256)`.

### 2) Choose Appropriate Base Tile Sizes

L0A and L0B use double buffering (ping/pong), with each buffer capped at 32 KiB.

For FP16 input (2 bytes/elem):

- L0A tile bytes ≈ `baseM × baseK × 2` = `128 × 64 × 2 = 16 KiB`
- L0B tile bytes ≈ `baseK × baseN × 2` = `64 × 256 × 2 = 32 KiB`

Communication tile size is `baseM × baseN × sizeof(FP16) = 128 × 256 × 2 = 64 KB`.

### 3) Use L1 "stepK" Caching to Improve Reuse

`stepKa=stepKb=4`: One TLOAD fetches 4 K-slices into L1; subsequent TEXTRACT calls extract them one by one to L0.

L1 usage: `2×64KB(A) + 2×128KB(B) = 384KB ≤ 1024KB` (total L1 capacity).

Increasing `stepK` reduces DMA launch overhead, but ensure it does not exceed L1 capacity.

### 4) Maintain Pipeline Overlap

The double buffering inside the compute kernel (L1/L0A/L0B) plus the dual-stream overlap between compute and communication is the core of performance.

When you observe:

- **Communication time >> compute time** → The compute side is well-optimized; focus on improving communication efficiency or increasing overlap.
- **Compute time >> communication time** → Communication is fully hidden; focus on optimizing the compute side.

### 5) Adjust Communication Block Count

`COMM_BLOCK_NUM` controls the AIV parallelism of the comm kernel. Adjust via the `--comm-blocks` argument.

Note: On the 910B, benchmarks show that increasing `COMM_BLOCK_NUM` from 24 to 48 (using all AIVs) causes a significant increase in AIC compute time (+24%) due to HBM bandwidth contention and TSCH scheduling overhead. The current optimal configuration is 24 AIVs.

### 6) Constraints

- K must be divisible by `G_BASE_K × G_STEP_KA` (default 64×4=256).
- M is automatically padded to 128 alignment; N is automatically padded to 256 alignment.
- All in-window buffers must be allocated at the same offset on every rank.
- `signal_matrix` is zeroed via `aclrtMemset` at the beginning of each iteration.

## Build and Run

1. Set up the Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Activate conda environment (requires Python + NumPy):

```bash
conda activate <your-conda-env>
```

3. Run the example (8 devices):

```bash
cd ${git_clone_path}/kernels/manual/a2a3/gemm_ar
./run.sh --nranks 8 --soc-version Ascend910B1
```

4. Specify starting device ID:

```bash
FIRST_DEVICE=0 ./run.sh --nranks 8 --soc-version Ascend910B1
```

5. Customize block allocation:

```bash
./run.sh --nranks 8 --compute-blocks 20 --comm-blocks 4
```

On success, the output will be:

```text
GEMM AllReduce demo completed successfully.
```

### Environment Variables

| Variable | Purpose | Default Behavior |
| --- | --- | --- |
| `ASCEND_CANN_PATH` | Full path to CANN `set_env.sh` | Auto-globs `/usr/local/Ascend/cann-*/set_env.sh` and picks the latest |
| `MPI_SEARCH_DIRS` | MPI `bin/` search paths (space-separated) | Searches `/usr/local/mpich/bin`, `/home/mpich/bin`, and other common paths |
| `ASCEND_DRIVER_PATH` | Ascend driver path (used by CMake) | Defaults to `/usr/local/Ascend/driver` |
| `MPI_LIB_PATH` | Absolute path to `libmpi.so` (runtime dynamic loading) | Auto-set by `run.sh` based on discovered MPI |
| `HCCL_BUFFSIZE` | HCCL RDMA window size (MB) | Auto-computed by `run.sh` based on M/N |
| `FIRST_DEVICE` | Starting NPU device ID | Defaults to 0 |

## Changing Matrix Dimensions

Modify `CONFIG_G_M`, `CONFIG_G_K`, and `CONFIG_G_N` in `gemm_ar_config.h`—all source files share the configuration via include. You can also pass them as CMake arguments:

```bash
cmake -DCONFIG_G_M=8192 -DCONFIG_G_K=8192 -DCONFIG_G_N=2048 ..
```

Constraint: K must be divisible by `G_BASE_K × G_STEP_KA` (default 64×4=256). `HCCL_BUFFSIZE` is auto-computed by `run.sh`.

## FAQ

| Issue | Cause & Solution |
| --- | --- |
| `HCCL window too small` | Window is too small. Check `HCCL_BUFFSIZE`; formula: `M × N × 2 bytes + margin` |
| `HcclGetRootInfo failed: 7` | Stale state from a previous run. Run `rm -rf /dev/shm/sem.hccl*; ipcrm -a` or wait ~30s and retry |
| Hangs after HCCL initialization | Rank synchronization issue; verify all ranks reach `CommMpiBarrier` |
| Segfault in comm kernel | Usually an invalid window address; verify `windowsIn[]` values are non-zero |
| DeviceBarrier deadlock | signal_matrix not zeroed between iterations; check that `resetState` memsets signal_matrix |
| Validation failure with large max_diff | FP16 has limited precision; validation tolerance is atol=1.0, rtol=0.01. If diff is abnormally large, check DeviceBarrier synchronization logic |
| `aclInit repeat init` (100002) | Harmless; code guards against this by calling `aclInit` only once per process |
| `--allow-run-as-root` failure | This project uses MPICH; that flag is OpenMPI-specific |

## Build System

- **Compiler**: bisheng (CANN built-in clang 15.0.5)
- **Cube kernel**: `--cce-aicore-arch=dav-c220-cube -DMEMORY_BASE`
- **Vec kernel**: `--cce-aicore-arch=dav-c220-vec -DMEMORY_BASE`
- **Host executable**: `-xc++` standard compilation
- **Link libraries**: `runtime`, `ascendcl`, `hcomm`, `tiling_api`
- The pto-comm-isa include path **must come first** to override CANN's built-in `pto_tile.hpp`

## Changelog

| Date | Change |
| --- | --- |
| 2025-12-15 | Initial version: GEMM + AllReduce dual-stream fusion operator |
| 2026-04-01 | CANN 9.0.0 compatibility (removed deprecated hccl/hccl.h dependency) |
| 2026-04-02 | RS AtomicAdd eliminating Reduce phase; AG row-level flattened decomposition for load balancing; architecture doc update |
