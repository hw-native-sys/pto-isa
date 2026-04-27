# High-Performance GEMM + AllReduce Fusion Example

## Overview

This example shows how to implement a multi-rank GEMM + AllReduce fused operator with PTO. It uses a dual-stream design for communication-compute overlap: a Compute Stream runs the GEMM kernel, and a Comm Stream runs the communication kernel. PTO communication instructions operate directly on the HCCL RDMA window to complete the AllReduce.

## Supported AI Processors

- Ascend950PR

## Directory Layout

```text
kernels/manual/a5/gemm_ar/
├── CMakeLists.txt              # Build configuration (3 targets: cube kernel, vec kernel, host executable)
├── run.sh                      # One-click build + run script (auto-computes HCCL_BUFFSIZE and locates MPI)
├── gemm_ar_config.h            # Global configuration (matrix shape, tile sizes, block counts)
├── main.cpp                    # Entry: MPI init, data generation, HCCL init, window allocation, perf measurement, verification
├── gemm_compute_kernel.cpp     # GEMM compute kernel (Cube side, L0C FP32 -> GM FP16 auto cast)
├── comm_kernel.cpp             # Communication kernel (Vector side, two-phase AllReduce in one kernel)
├── common.hpp                  # Device-side HcclRemotePtr wrapper (RDMA window address translation)
├── hccl_context.h              # HcclDeviceContext structure (RDMA window addresses for each rank)
├── ready_queue.hpp             # Multi-block lock-free tile queue (compute -> comm signaling)
└── comm_mpi.h                  # MPI dynamic loading wrapper (dlopen/dlsym, no hard link dependency)
```

## Operator Description

### Functionality

This example implements multi-rank GEMM + AllReduce:

$$
C_{final} = \sum_{i=0}^{nranks-1} A_i \times B
$$

Where:

- `A_i` is `M x K` and is private to each rank
- `B` is `K x N` and shared by all ranks
- `C_i` is the local GEMM result of shape `M x N`
- `C_final` is the final `M x N` output after AllReduce

The default reference configuration in `gemm_ar_config.h` is `M=5416, K=6144, N=1408` with 2 ranks.

### Specification

| Item | Value |
| --- | --- |
| OpType | `GEMM + AllReduce` |
| Input | `A_i`: `M x K`, `float16`, `ND` (private to each rank); `B`: `K x N`, `float16`, `DN` (shared) |
| Output | `C_final`: `M x N`, `float16`, `ND` (AllReduce result) |
| Compute kernel name | `GemmComputeKernel` (Cube architecture, `dav-c220-cube`) |
| Comm kernel name | `GemmCommAllKernel` (Vector architecture, `dav-c220-vec`) |

## Optimization Notes

This example uses a 2-rank Ascend950PR platform as the performance validation target. Ascend950PR (`DAV_3510` / `arch35`) uses a split architecture where Cube (AIC) and Vector (AIV) are physically separate, which makes dual-stream communication-compute overlap practical.

> Use the CANN `platform_config` as the source of truth for core counts. For example, on `950PR_958b`:
>
> - `cube_core_cnt=32` (Cube / AIC parallelism)
> - `vector_core_cnt=64` (Vector / AIV parallelism)

- **Dual-stream overlap**: the compute kernel runs on the Compute Stream (Cube) and the communication kernel runs on the Comm Stream (Vector). Tile-level signaling allows communication and computation to run concurrently.
- **Two-phase communication: ReduceScatter + AllGather**: in the RS phase, `TPUT<AtomicAdd>` writes directly into the owner rank's `reduced_output`. Reduction is completed by hardware atomic add on the destination side, so no separate Reduce stage is needed. In the AG phase, the owner rank broadcasts the reduced result to all other ranks.
- **Block Swizzle**: the compute kernel uses a zigzag tile traversal order (odd rows reversed) to improve L1 reuse of neighboring `B` matrix tiles.
- **Two-level double-buffer pipeline**: L1 cache (`stepK=4` batched `TLOAD`) plus L0 ping/pong buffering lets DMA movement overlap with Cube compute as much as possible.
- **Lock-free Ready Queue**: each AIC has a dedicated single-producer single-consumer queue. AIV uses non-blocking `TTEST` polling and falls back to `TWAIT` when no tile is ready, avoiding busy spinning.
- **RS double buffering**: the RS phase of the communication kernel uses ping/pong tiles so that the `TLOAD` of the current tile overlaps with the `TSTORE<AtomicAdd>` of the previous tile.
- **AG row-level flattened scheduling**: the AG phase flattens all work to row granularity (`my_tile_count x (nranks-1) x G_BASE_M`) and distributes it evenly across AIV blocks, eliminating the +/-1 imbalance of tile-level scheduling.

## Tiling Parameters

| Parameter | Value |
| --- | --- |
| `M` (raw) | 5416 |
| `K` | 6144 |
| `N` (raw) | 1408 |
| `M` (padded) | 5504 |
| `N` (padded) | 1536 |
| `baseM` | 128 |
| `baseK` | 64 |
| `baseN` | 256 |
| `stepKa` | 4 |
| `stepKb` | 4 |
| Number of tiles | 258 (`43 x 6`) |
| `COMPUTE_BLOCK_NUM` | 24 |
| `COMM_BLOCK_NUM` | 24 |

## Overall Architecture

```text
Compute Stream (32 Cube)                 Comm Stream (64 Vector, 958b ini)
┌──────────────────────────────┐         ┌────────────────────────────────┐
│ GemmComputeKernel            │         │ GemmCommAllKernel              │
│ for each tile:               │         │ Phase 1: ReduceScatter         │
│ K-loop (L1 -> L0 -> Cube)    │         │ poll Ready Queue               │
│ TSTORE -> gemm_output        │──Ready─►│ TLOAD tile from gemm_output    │
│ pipe_barrier(ALL)            │  Queue  │ TSTORE<AtomicAdd> -> owner     │
│ Enqueue tile_idx             │         │ (ping/pong buffering)          │
└──────────────────────────────┘         │ DeviceBarrier (cross-rank)     │
                                         │ Phase 2: AllGather             │
                                         │ row-level flattened split      │
                                         │ TLOAD -> TSTORE to remote      │
                                         └────────────────────────────────┘
```

## Compute Kernel Details

```text
Time ->
L1 (MTE2):  [TLOAD A0,B0]                [TLOAD A1,B1]              ...
L0 (MTE1):       [TEXTRACT k0] [k1] [k2] [k3] [TEXTRACT k0'] ...
Cube (M):             [TMATMUL k0] [ACC k1] [ACC k2] [ACC k3] [TMATMUL k0'] ...
                      ^ full three-stage overlap ^
```

Each AIC is responsible for a subset of tiles assigned by `block_idx x tiles_per_block`. For each tile:

1. **Block Swizzle mapping**: remap the linear tile index into a zigzag traversal order, reversing odd rows so adjacent tiles reuse columns of matrix `B` in L1.
2. **K-loop**: every `stepKa=4` iterations, perform one batched `TLOAD` into L1. Each iteration then uses `TEXTRACT` to pull one K-slice into L0, followed by `TMATMUL` / `TMATMUL_ACC` accumulation.
3. **TSTORE**: FP32 values in L0C are automatically cast to FP16 by the FixPipe and stored to `gemm_output`.
4. **`pipe_barrier(PIPE_ALL)`**: guarantees that the GM write is complete.
5. **`MultiBlockEnqueueFast`**: enqueue `tile_idx` to notify the communication kernel.

## Communication Kernel Details

### Phase 1: ReduceScatter

Each **communication block** polls and consumes its corresponding Ready Queue. In this example, `COMPUTE_BLOCK_NUM` and `COMM_BLOCK_NUM` are often both 24, but they do not need to match the physical `cube_core_cnt=32` and `vector_core_cnt=64` in `Ascend950PR_958b.ini`.

AIV uses the hardware `TTEST` instruction to poll the queue without blocking. Once a ready tile is found:

1. **`TLOAD`** moves the tile from `gemm_output` into UB using ping/pong buffering.
2. **`TSTOREAtomicAdd`** accumulates the tile into the owner rank's `reduced_output` (local or remote RDMA).

The double-buffer pipeline overlaps the `TLOAD` of the current tile with the `TSTORE` of the previous tile. When the queue is empty, AIV enters hardware `TWAIT` instead of busy spinning.

The tile owner is determined by `tile_idx % nranks`, which distributes tiles evenly across ranks.

### DeviceBarrier: Two-Level Device-Side Synchronization

```text
DeviceBarrier(phase):
  pipe_barrier(PIPE_ALL)                    // flush this block's local pipeline

  if block_idx == 0:                        // only block 0 performs cross-rank signaling
    for each remote rank r:
      TNOTIFY(remote signal_matrix[phase][my_rank], 1, AtomicAdd)   // write remote signal
    for each remote rank r:
      TWAIT(local signal_matrix[phase][r], 1, GE)                   // wait for remote signal
    TNOTIFY(local_broadcast_flag[phase], 1, Set)                    // notify local blocks
  else:
    TWAIT(local_broadcast_flag[phase], 1, GE)                       // wait for block 0 broadcast

  pipe_barrier(PIPE_ALL)
```

### Phase 2: AllGather

All AG work is flattened into row-level tasks:

```text
total_rows = my_tile_count x (nranks - 1) x G_BASE_M
rows_per_block = ceil(total_rows / num_comm_blocks)
```

Each AIV handles a row range `[row_start, row_end)`. For every continuous row segment, it reconstructs `(tile_owner_idx, remote_rank, row_in_tile)` and then:

1. **`TLOAD`** from local `reduced_output` into UB
2. **`TSTORE`** into the remote rank's `reduced_output` through RDMA

Row-level scheduling ensures every AIV transfers the same amount of data, avoiding the imbalance that appears when tile counts cannot be evenly divided across AIV blocks.

## Ready Queue Mechanism

```text
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

This illustration uses the default logical block range `0...23` when `COMPUTE_BLOCK_NUM = COMM_BLOCK_NUM = 24`. These logical block IDs do not need to equal the physical `cube_core_cnt` and `vector_core_cnt` from `Ascend950PR_958b.ini`. If you use another SoC such as `Ascend950PR_9599`, follow the corresponding counts from its `.ini`.

- Each queue is a 64-byte-aligned `PerBlockQueue` structure containing `count` (producer-side monotonically increasing counter) and `data[]` (tile index array).
- **Producer** (AIC): `PerBlockQueueEnqueueFast` writes `data[slot]`, then increments `count`, and uses `dcci` to flush cache state so the entry becomes visible to AIV.
- **Consumer** (AIV): `PerBlockQueueTryDequeue` uses hardware `TTEST` to check whether `count >= head+1`. If no tile is ready, it returns `-1`; after a prolonged idle period it falls back to hardware `TWAIT`.
- The design is single-producer single-consumer, so no atomic operation is required inside the queue.

## Memory Layout and HCCL Window

Only buffers written by remote `TPUT` or `TNOTIFY` need to live in the HCCL RDMA window. Buffers used only for local read/write can be allocated with plain `aclrtMalloc`.

| Buffer | Size | Location | Why |
| --- | --- | --- | --- |
| `reduced_output` | `M x N x 2B` | **HCCL window** | RS `AtomicAdd` and AG remote `TPUT` writes (`FP16`) |
| `signal_matrix` | `(MAX_RANKS+1) x 4B`, aligned to 64B | **HCCL window** | Cross-rank `TNOTIFY` writes for `DeviceBarrier` |
| `gemm_output` | `M x N x 2B` | **aclrtMalloc** | Local read/write only (`FP16`) |
| `src0_dev`, `src1_dev` | input matrices (`FP16`) | **aclrtMalloc** | Local read/write only |

Window size is controlled by the `HCCL_BUFFSIZE` environment variable. `run.sh` computes it automatically as:

```text
M x N x 2 / 1MB + 64MB
```

## Measured Performance (Reference)

The following numbers were collected on 2-card Ascend950PR with `M=5416`, `K=6144`, `N=1408` (padded to `5504 x 1536`) and 258 tiles (`43 x 6`). Each rank computes a full GEMM `C_i = A_i x B`, and AllReduce sums the two `C_i` tensors.

| Metric | Value |
| --- | --- |
| Compute-only | 365 us (`257 TFLOPS`, `98%`) |
| Sequential | 743 us (compute `368 us` + comm `375 us @ 74 GB/s`) |
| Pipelined | **631 us** (speedup `1.18x`, overlap `31%`) |
| Throughput | `1189 TFLOPS` (total) |

### What These Numbers Mean

- **Compute-only**: pure GEMM execution time with no communication. It reflects the upper bound of single-card Cube utilization. `257 TFLOPS` reaches `98%` of the theoretical peak, which indicates the compute kernel is already highly optimized.
- **Sequential**: compute followed by communication with no overlap. Total time is simply compute time plus communication time.
- **Pipelined**: compute and communication run concurrently on two streams. `631 us` versus `743 us` gives a `1.18x` speedup, with `31%` overlap efficiency.
- **Speedup**: `Sequential / Pipelined`. A larger value means communication-compute overlap is more effective.
- **Overlap efficiency**: the fraction of the shorter phase that is hidden by overlap. `31%` means roughly one third of the communication time is successfully hidden under compute.

### Optimization History

> The table below records historical ablation data collected on **Ascend910B** (including 24/48 block experiments). For Ascend950PR, use the `cube_core_cnt` and `vector_core_cnt` values from the relevant `platform_config` as described above. Do not interpret the block counts in this table as physical core counts.

| Optimization | Pipelined (us) | Gain | Conclusion |
| --- | --- | --- | --- |
| Baseline | 808 | - | - |
| Block Swizzle | 793 | `-1.8%` | **Kept** |
| RS `AtomicAdd` removes the separate Reduce stage | 736 | `-6.6%` | **Kept** |
| AG row-level flattened scheduling | 623 | `-15.4%` | **Kept** |
| 48 AIV (`RS` skip + `AG` participate) | 639 | RS only on 24 AIV, AG on 48 AIV | **Reverted** (`AIC` interference) |
| 48 AIV dual-queue (`1 AIC : 2 AIV`) | 667 | both RS and AG on 48 AIV | **Reverted** (`AIC` interference) |

## Performance Tuning Guide

### 1. Prioritize Multi-Core Partitioning

Each AIC receives a tile subset according to `block_idx x tiles_per_block`, and blocks do not interfere with one another.

Checklist:

- Tune `COMPUTE_BLOCK_NUM` so each block gets a similar number of tiles.
- For different matrix shapes, recompute the total tile count as `G_NUM_TILES = (M_padded/128) x (N_padded/256)`.

### 2. Choose a Proper Base Tile

L0A and L0B use ping/pong double buffering, and each buffer is limited to 32 KiB.

For FP16 input (`2 bytes/elem`):

- L0A tile bytes ~= `baseM x baseK x 2` = `128 x 64 x 2 = 16 KiB`
- L0B tile bytes ~= `baseK x baseN x 2` = `64 x 256 x 2 = 32 KiB`

The communication tile size is:

```text
baseM x baseN x sizeof(FP16) = 128 x 256 x 2 = 64 KB
```

### 3. Use L1 `stepK` Caching to Increase Reuse

With `stepKa=stepKb=4`, one `TLOAD` brings 4 K-slices into L1, and subsequent `TEXTRACT` operations pull them into L0 one by one.

L1 usage:

```text
2 x 64KB (A) + 2 x 128KB (B) = 384KB <= 1024KB
```

Increasing `stepK` can reduce DMA launch overhead, but the total must still fit in L1.

### 4. Preserve Pipeline Overlap

The key to performance is the combination of:

- double buffering inside the compute kernel (`L1` / `L0A` / `L0B`)
- dual-stream overlap between compute and communication

When you observe:

- **communication time >> compute time**: the compute side is already efficient, so focus on improving communication or increasing overlap.
- **compute time >> communication time**: communication is fully hidden, so focus on the compute side.

### 5. Tune the Number of Communication Blocks

`COMM_BLOCK_NUM` controls AIV parallelism in the communication kernel and can be adjusted via `--comm-blocks`.

On **Ascend910B**, measurements showed that increasing `COMM_BLOCK_NUM` from 24 to 48 caused a significant increase in AIC compute time (about `+24%`) because of HBM bandwidth contention and TSCH scheduling overhead. A more stable default was therefore 24. After moving to **Ascend950PR**, the upper bound should be reconsidered based on the SoC-specific `vector_core_cnt` in the corresponding `.ini` file, for example **64 on 958b** and **72 on 9599**. Do not assume the old "24 best, 48 worse" conclusion still holds without profiling on the target SoC.

### 6. Constraints

- `K` must be divisible by `G_BASE_K x G_STEP_KA` (default `64 x 4 = 256`).
- `M` is padded automatically to a multiple of 128, and `N` is padded automatically to a multiple of 256.
- All HCCL-window buffers must be allocated at the same offset on every rank.
- `signal_matrix` must be reset with `aclrtMemset` before each iteration.

## Build and Run

1. Configure the Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

1. Activate a conda environment that provides Python and NumPy:

```bash
conda activate <your-conda-env>
```

1. Run the example with 2 ranks:

```bash
cd ${git_clone_path}/kernels/manual/a5/gemm_ar
./run.sh --nranks 2 --soc-version Ascend950PR_958b
```

1. Specify the starting device index:

```bash
FIRST_DEVICE=0 ./run.sh --nranks 2 --soc-version Ascend950PR_958b
```

1. Use custom compute/communication block counts:

```bash
./run.sh --nranks 2 --soc-version Ascend950PR_958b --compute-blocks 20 --comm-blocks 4
```

When successful, the program prints:

```text
GEMM AllReduce demo completed successfully.
```

### Environment Variables

| Environment Variable | Purpose | Default Behavior |
| --- | --- | --- |
| `ASCEND_CANN_PATH` | Full path to the CANN `set_env.sh` script | Auto-globs `/usr/local/Ascend/cann-*/set_env.sh` and picks the latest one |
| `MPI_SEARCH_DIRS` | Search paths for MPI `bin/` directories (space-separated) | Searches common locations such as `/usr/local/mpich/bin` and `/home/mpich/bin` |
| `ASCEND_DRIVER_PATH` | Ascend driver path used by CMake | Defaults to `/usr/local/Ascend/driver` |
| `MPI_LIB_PATH` | Absolute path to `libmpi.so` for runtime dynamic loading | Auto-set by `run.sh` according to the discovered MPI installation |
| `HCCL_BUFFSIZE` | HCCL RDMA window size in MB | Auto-computed by `run.sh` from `M` and `N` |
| `FIRST_DEVICE` | Starting NPU device index | Defaults to `0` |

## Changing Matrix Dimensions

Update `CONFIG_G_M`, `CONFIG_G_K`, and `CONFIG_G_N` in `gemm_ar_config.h`. All source files share the configuration through includes. You can also pass them from CMake:

```bash
cmake -DCONFIG_G_M=8192 -DCONFIG_G_K=8192 -DCONFIG_G_N=2048 ..
```

Constraint: `K` must be divisible by `G_BASE_K x G_STEP_KA` (default `64 x 4 = 256`). `HCCL_BUFFSIZE` is computed automatically by `run.sh`.

## FAQ

| Problem | Cause and Fix |
| --- | --- |
| `HCCL window too small` | The window is not large enough. Check `HCCL_BUFFSIZE`. Formula: `M x N x 2 bytes + margin` |
| `HcclGetRootInfo failed: 7` | Leftover dirty state from a previous run. Execute `rm -rf /dev/shm/sem.hccl*; ipcrm -a` or wait about 30 seconds and retry |
| Hangs after HCCL initialization | Usually a rank synchronization problem. Check that all ranks reached `CommMpiBarrier` |
| Segmentation fault in the communication kernel | Usually caused by an invalid window address. Verify that `windowsIn[]` entries are non-zero |
| Deadlock in `DeviceBarrier` | `signal_matrix` was not cleared between iterations. Check whether `resetState` calls `memset` on `signal_matrix` |
| Verification fails with large `max_diff` | FP16 precision is limited. The validation tolerance is `atol=1.0, rtol=0.01`. If the diff is abnormally large, check `DeviceBarrier` synchronization logic |
| `aclInit repeat init` (`100002`) | Harmless. The code already guards against repeated `aclInit` in one process |
| `--allow-run-as-root` fails | This project uses MPICH. That option is specific to OpenMPI |

## Build System

- **Compiler**: `bisheng` (CANN-bundled clang 15.0.5)
- **Cube kernel flags**: `--cce-aicore-arch=dav-c220-cube -DMEMORY_BASE`
- **Vector kernel flags**: `--cce-aicore-arch=dav-c220-vec -DMEMORY_BASE`
- **Host executable**: standard `-xc++` compilation
- **Linked libraries**: `runtime`, `ascendcl`, `hcomm`, `tiling_api`
- The include path for `pto-comm-isa` **must come first** so it overrides the `pto_tile.hpp` bundled with CANN

## Changelog

| Date | Change |
| --- | --- |
| 2026-04-15 | Added the A5 adaptation of `gemm_ar` |
