# AllGather + GEMM Communication-Compute Fusion Example

## Overview

This example demonstrates a fused AllGather + GEMM operator on Ascend AI Cores using **M-dimension splitting** and a **tile streaming pipeline**. In multi-card LLM inference, each rank holds a local slice of matrix `A` along the M dimension. Instead of completing the AllGather before starting GEMM, this implementation overlaps communication and computation at tile granularity — the compute kernel begins processing each tile as soon as the communication kernel signals its arrival, effectively hiding communication latency behind compute.

## Supported AI Processors

- Ascend950PR (A5 family)

## Directory Layout

```
kernels/manual/a5/allgather_gemm/
├── main.cpp                           # Host entry: HCCL init, dual-stream dispatch, warmup, verification, perf stats
├── allgather_gemm_comm_kernel.cpp     # AIV communication kernel: AllGather via TPUT
├── allgather_gemm_compute_kernel.cpp  # AIC compute kernel: streaming GEMM with tile-flag waiting
├── ready_queue.hpp                    # TileFlagMatrix / summary counter metadata
├── run.sh                             # Generate data, build, and launch mpirun execution
├── scripts/
│   ├── gen_data.py                    # Input data generation (FP16 A slices + B)
│   ├── test_shapes.csv                # Test shape configurations (M, K, N)
│   └── verify_result.py               # Golden comparison (FP32, rtol/atol=0.001)
└── CMakeLists.txt                     # Build configuration
```

## Operator Description

### Function

This example implements AllGather followed by GEMM:

$$
C = A \times B
$$

Where:

- Each of `n_ranks` ranks holds a local M-slice of `A`: rows `[rank * M/n_ranks, (rank+1) * M/n_ranks)`.
- `B` is replicated across all ranks (`K × N`, FP16).
- After AllGather collects the full `A` (`M × K`, FP16), each rank computes the full `C` (`M × N`, FP32).

The AllGather and GEMM are fused into a streaming pipeline so that computation begins before the full AllGather completes.

### Specification

| Item         | Value |
| ------------ | ----- |
| OpType       | `AllGather + GEMM` (communication-compute fusion) |
| Inputs       | `A`: `M × K`, `float16`, `ND` (M-sliced across ranks); `B`: `K × N`, `float16`, `ND` (replicated) |
| Output       | `C`: `M × N`, `float32`, `ND` |
| Comm kernel  | `RingCommStreamingKernel` (AIV) |
| Compute kernel | `AllGatherGemmComputeStreamingKernel` (AIC) |

## Architecture

### Dual-Stream Concurrency

The communication and compute kernels run on two independent AICPU streams, launched concurrently from the host:

- **Comm stream** → `RingCommStreamingKernel` runs on **AIV** (Vector) cores.
- **Compute stream** → `AllGatherGemmComputeStreamingKernel` runs on **AIC** (Cube) cores.

The host dispatches both kernels back-to-back and synchronizes after both complete.

### AI Core Resources

| Unit            | Hardware Engine | Role in This Example |
| --------------- | --------------- | -------------------- |
| **AIC (Cube)**  | Matrix engine   | Compute kernel: GEMM (`TMATMUL` / `TMATMUL_ACC`) |
| **AIV (Vector)**| Vector / DMA    | Comm kernel: RDMA data transfer (`TPUT`) + signaling (`TNOTIFY`) |

### Streaming Pipeline

```
Sequential execution:
  [ AllGather completes entirely ] ──► [ GEMM completes entirely ]

Streaming pipelined execution:
  Comm (AIV):   [tile0 TPUT][TNOTIFY] [tile1 TPUT][TNOTIFY] [tile2 TPUT][TNOTIFY] ...
                      │                     │                     │
                      ▼                     ▼                     ▼
  Compute (AIC): [local GEMM]  [TWAIT tile0][GEMM tile0] [TWAIT tile1][GEMM tile1] ...
                  (zero-wait)
```

The compute kernel operates in two phases:

1. **Phase 1 (local)**: Processes the local rank's row-groups immediately (data already resident in shared memory, no waiting).
2. **Phase 2 (remote)**: For each remote rank's row-groups, uses `TWAIT` on the summary counter to block until tiles arrive, then computes as soon as each tile is ready.

## Optimization Notes

- **Summary monotonic counter + TWAIT**: The comm kernel atomically increments a per-source summary counter (`TNOTIFY` AtomicAdd) after each tile transfer. The compute kernel uses hardware `TWAIT` (compare-and-block) to wait for the counter to reach the expected value — zero polling overhead, no busy-spin.
- **Local data zero-wait priority**: The compute kernel processes the local rank's row-groups first (Phase 1) with no flag checks, overlapping with remote tile transfers.
- **Send order aligned with consumption order**: The comm kernel transmits tiles in the same order the compute kernel consumes them, minimizing wait time.
- **Continuous K accumulation pipeline**: Within each row-group, K-blocks are processed with `TMATMUL` (first iteration) followed by `TMATMUL_ACC` (subsequent iterations), maintaining a continuous accumulation pipeline without intermediate store/reload.
- **L1/L0 two-level double buffering**: `aMatTile[2]` / `bMatTile[2]` in L1 and `aTile[2]` / `bTile[2]` in L0A/L0B enable overlapped DMA (`TLOAD`) ↔ extract (`TEXTRACT`) ↔ compute (`TMATMUL`).
- **Parallel AIV full-mesh communication**: In the full-mesh mode, each rank's AIV cores directly `TPUT` data to all other ranks simultaneously, with multiple AIV blocks assigned per destination for bandwidth utilization.
- **Dynamic tile size**: `ComputeOptimalTileSize()` automatically selects tile granularity to keep the number of tiles per source in the 64–128 range, balancing pipeline depth against polling overhead.
- **Flexible block allocation**: The comm kernel adapts to the available block count — when blocks outnumber destinations, blocks are evenly distributed per destination; otherwise, work items are round-robin scheduled across blocks.

## Build and Run

The current `run.sh` script does three things in one command:

1. Generates input data and golden output into `./out`
2. Recreates `build/` and rebuilds `allgather_gemm`
3. Launches `mpirun -n <n_ranks> ./allgather_gemm`

Before running it, configure your Ascend CANN environment so `ASCEND_HOME_PATH` is available:

```bash
source <cann-install>/set_env.sh
```

Then enter the example directory:

```bash
cd ${git_clone_path}/kernels/manual/a5/allgather_gemm
```

Run the default 2-rank example on A5:

```bash
bash run.sh -r npu -v Ascend950PR_958b
```

Run with a custom rank count and GEMM shape:

```bash
bash run.sh -r npu -v Ascend950PR_958b -n 4 --gm 4096 --gk 2048 --gn 1536
```

Run with custom tile and block settings:

```bash
bash run.sh -r npu -v Ascend950PR_958b -n 2 --gm 2048 --gk 2048 --gn 1024 --base-m 128 --base-n 256 --compute-blocks 32 --comm-blocks 24
```

Run in simulator mode:

```bash
bash run.sh -r sim -v Ascend950PR_958b -n 2 --gm 2048 --gk 2048 --gn 1024
```

Shape constraints enforced by `run.sh`:

- `--base-n` must be divisible by 4
- `G_M % G_BASE_M == 0`
- `G_K % G_BASE_N == 0`
- `G_N % G_BASE_N == 0`


### Command-Line Options

| Option | Description |
| ------ | ----------- |
| `-r/--run-mode` | Run mode: `npu` or `sim` |
| `-v/--soc-version` | SoC version string, for example `Ascend950PR_958b` |
| `-n/--n-ranks` | Number of MPI ranks passed to `mpirun` |
| `--gm` | Global M dimension used for data generation and build-time configuration |
| `--gk` | Global K dimension used for data generation and build-time configuration |
| `--gn` | Global N dimension used for data generation and build-time configuration |
| `--base-m` | Tile size on the M dimension |
| `--base-n` | Tile size on the N dimension (`--base-n` must be divisible by 4) |
| `--compute-blocks` | Override the compute kernel block count |
| `--comm-blocks` | Override the communication kernel block count |

## Changelog

| Date       | Change |
| ---------- | ------ |
| 2025-07-01 | Initial implementation: AllGather+GEMM fusion with M-split streaming pipeline |
