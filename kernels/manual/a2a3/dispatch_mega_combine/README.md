# PTO MegaMoE Dispatch + Combine Fusion Example

[中文文档](README_zh.md)

## Overview

This example implements an end-to-end MegaMoE fused operator with PTO Manual kernels. It fuses the traditional MoE reorder, AlltoAllV-style data exchange, grouped FFN compute, combine, and unpermute flow into one large kernel, and overlaps AIC and AIV work at local-expert granularity.

The main device pipeline is:

```text
FrontReorder -> Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine -> Unpermute
```

## Supported AI Processors

- Ascend910B1
- Ascend910B / Ascend910C
- Ascend910_93 / Ascend910_9391 / Ascend910_9381 / Ascend910_9372 / Ascend910_9392 / Ascend910_9382 / Ascend910_9362

This directory is under the `a2a3` manual-kernel path. The typical performance cases use the script defaults for A2/A3-style execution; users do not need to pass an explicit SoC option in the listed commands.

## Directory Layout

```text
kernels/manual/a2a3/dispatch_mega_combine/
├── CMakeLists.txt                  # Build configuration for host executable and device kernel shared library
├── run.sh                          # One-click data generation, build, and mpirun entry
├── main.cpp                        # Host entry: case loading, ACL/HCCL/MPI setup, windows, launch, verify, timing
├── kernel_launch.cpp               # Device kernel launch wrapper
├── runtime_context.*               # Single-rank runtime, HCCL window, device/context management
├── tiling_builder.*                # Host tiling and workspace planning
├── data_utils.*                    # Case file IO, validation, and performance helpers
├── comm_mpi.h                      # MPI dynamic loading wrapper
├── scripts/
│   ├── gen_data.py                 # Synthetic input, weight, and golden generation
│   └── tests/                      # Data distribution tests
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoE device pipeline entry
│   ├── front_reorder.h             # Front reorder shared postprocess and quant scatter
│   ├── front_fullload_sort.h       # FullLoad small-route sorting path
│   ├── front_vms_sort.h            # OneCore / MultiCore VMS sorting paths
│   ├── dispatch.h                  # Pull offsetA from source ranks and build GMM1 input
│   ├── gmm_common.h                # Shared GMM tile scheduling and helpers
│   ├── gmm1.h                      # First grouped matmul
│   ├── swiglu.h                    # SwiGLU activation and dynamic quantization
│   ├── gmm2.h                      # Second grouped matmul
│   ├── combine.h                   # Remote writeback of GMM2 output to offsetD
│   ├── unpermute.h                 # TopK weighted reduction and original token order restoration
│   └── utils/                      # PTO vector, sync, HCCL window, and GMM pipeline helpers
├── overview.md                     # Design overview, performance comparison, and stage pseudocode
├── front_reorder.md                # Front reorder / sort / count-as-flag details
├── dispatch.md                     # Dispatch contract and data movement strategy
├── gmm1.md / gmm2.md               # GMM tile scheduling, swizzle, sync, and pipeline details
├── swiglu.md                       # SwiGLU segmentation and quantization strategy
├── combine.md                      # Combine large/small paths and remote writeback protocol
├── unpermute.md                    # Unpermute restoration and accumulation strategy
└── glden.md                        # Python batch golden rewrite design
```

## Operator Description

### Functionality

The operator implements the multi-rank MoE FFN main path:

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> routed expert-major token rows
  -> grouped GMM1
  -> SwiGLU activation + dynamic quantization
  -> grouped GMM2
  -> combine back to source ranks
  -> TopK weighted reduction
  -> out[rank, M, K]
```

Conceptually:

```text
for each rank, token:
  out[token] = sum_{topK route} probs[token, route] * FFN_expert(x[token])
```

`FFN_expert` consists of int8 GMM1, SwiGLU, and int8 GMM2. Cross-rank data exchange uses the HCCL RDMA window and PTO communication/synchronization helpers.

### Specification

| Item | Value |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| Input | `x`: `[M, K]`, `half/bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`; `weight1/weight2`: per-local-expert int8 packed weights; `scale1/scale2`: fixpipe scales |
| Output | `out`: `[M, K]`, `half/bfloat16` |
| Kernel name | `dispatch_mega_combine_kernel` |
| Host executable | `dispatch_mega_combine` |
| Default script case | `worldSize=8, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## Optimization Notes

- **Expert-level overlap**: AIC-side GMM1/GMM2 and AIV-side Dispatch/SwiGLU/Combine progress by local expert group, connected by hard flags.
- **Three front reorder paths**: FullLoad, OneCore, and MultiCore are selected by UB working-set size. Small-route cases keep sort/count/inverse/quant work in UB as much as possible.
- **Count-as-flag**: FrontReorder publishes count rows with a marker value so peer ranks can wait on data arrival directly instead of adding a full count-exchange barrier.
- **PTO tile GMM optimization**: GMM1/GMM2 use output-tile swizzle, L1-to-L0 multi-level reuse, double buffering, and fixpipe quant/cast.
- **Segmented SwiGLU overlap**: SwiGLU is split into segments so the first segment can finish before GMM2 starts.
- **Dual combine paths**: large-token cases use full-row writeback, while small-token cases split GMM2 tiles into subtiles to improve AIV occupancy.

## Tiling Parameters

| Parameter | Default / Description |
| --- | --- |
| `M` | Set by `run.sh --m` or `case.json` |
| `K` | Hidden size for GMM input; must satisfy packed-row and GMM tile alignment |
| `N` | FFN intermediate size; GMM1 output and `N/2` after SwiGLU |
| `topK` | Number of routed experts per token |
| `expertPerRank` | Number of local experts per rank |
| `worldSize` | MPI/HCCL rank count |
| `maxOutputSize` | Per-rank routed-row workspace limit; typical performance cases pass a fixed workspace limit explicitly |
| `aicNum` | Logical AIC count; default script value is 24 |
| `aivNum` | Logical AIV count; default script value is 48 |
| `GMM baseM/baseN` | Main output tile shape is `128 x 256` |
| `Front FullLoad` | Selected by `routeElems`, `K`, `expertNum`, and the 192 KiB UB budget |
| `Combine small path` | Preferred when `problemM * topK <= 4096` |

## Supported Cases

Typical performance cases keep all major parameters fixed except `M`:

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=24
aivNum=48
```

| M | maxOutputSize | Command |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## Overall Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ FrontReorder (AIV)                                                          │
│   sort expertId routes -> offsetA + count/prefix metadata                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ D2C ready / count-as-flag
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Expert-level overlapped pipeline                                             │
│                                                                              │
│ AIV: Dispatch(group i) -> SwiGLU(segment/group i) -> Combine(group i)         │
│ AIC:                    GMM1(group i)      -> GMM2(group i)                  │
│                                                                              │
│ Stages communicate through hard flags: D2C, C2V, V2C, G2C/Combine ready       │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ final boundary
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Unpermute (AIV)                                                              │
│   offsetD + probs + expandedRowIdx -> TopK weighted reduce -> out[M, K]       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## FrontReorder Stage

FrontReorder sorts `[token, topK]` routes by global expert on the source rank and writes quantized token rows into the local remote window:

```text
x[M, K] + expertId[M, topK]
  -> offsetA[expert-major rows, K + 32]
  -> expandedRowIdx / localTokenPerExpert / preSumBeforeRank / cumsumMM
```

The three sort paths have the same semantics:

- **FullLoad**: route/count/quant working set fits in UB; active AIVs repeat the full sort and split quant/scatter by route range.
- **OneCore**: one core can sort the route list, but the full FullLoad working set does not fit; core0 writes `frontExpandedExpert/frontExpandDstToSrc`.
- **MultiCore**: large route list; multiple AIVs build VBS sorted runs, VMS merge them, and core0 writes the final sort-out.

FrontReorder publishes:

```text
tokenPerExpert[srcRank, globalExpert]
preSumBeforeRank[srcRank, localExpert]
cumsumMM[srcRank, localExpert]
expertTokenNums[localExpert]
```

## Dispatch Stage

Dispatch runs on the destination rank and pulls packed rows from all source ranks:

```text
srcRank.remoteWindow.offsetA[srcRowBase : srcRowBase + rows]
  -> workspace.gmA[dstRowBase : dstRowBase + rows, 0:K]
  -> workspace.perTokenScale1[dstRowBase : dstRowBase + rows]
```

The address calculation is driven by FrontReorder metadata:

```text
rows       = tokenPerExpert[srcRank, globalExpert]
srcRowBase = preSumBeforeRank[srcRank, localExpert]
dstRowBase = groupBase + (srcRank == 0 ? 0 : cumsumMM[srcRank - 1, localExpert])
```

After each local expert group is gathered, Dispatch sets the GMM1-ready flag for that group.

## GMM1 / SwiGLU / GMM2 Stages

### GMM1

GMM1 runs on AIC and performs the first grouped matmul:

```text
gmA[int8] x weight1[int8]
  -> int32 accumulator
  -> fixpipe scale1
  -> gmC[half]
```

Each local expert is split into `128 x 256` output tiles. Linear tile ids are mapped to `(blockM, blockN)` with swizzle to improve L1 reuse of the B-side weights.

### SwiGLU

SwiGLU runs on AIV and consumes GMM1 `gmC` plus Dispatch-generated `perTokenScale1`:

```text
gmC * perTokenScale1
  -> silu(up) * gate
  -> dynamic quantization
  -> gmPermutedToken[int8] + perTokenScale2[float]
```

SwiGLU is split into segments. Core0 writes segment metadata and the other AIVs split rows from that metadata.

### GMM2

GMM2 runs on AIC and performs the second grouped matmul:

```text
gmPermutedToken[int8] x weight2[int8]
  -> int32 accumulator
  -> fixpipe scale2
  -> gmm2Output[half]
```

After GMM2 finishes a local expert group, it sets the combine-ready flag for that group.

## Combine / Unpermute Stages

Combine runs on AIV and writes GMM2 output back to the source rank after applying `perTokenScale2`:

```text
gmm2Output[srcRow, 0:K] half
  -> fp32
  -> * perTokenScale2[srcRow]
  -> OutputElement
  -> srcRank.remoteWindow.offsetD[dstRow, 0:K]
```

Path selection:

- **DirectLarge**: full-row writeback for large-token cases.
- **DirectSmall**: subtile writeback for small-token cases to improve AIV occupancy.

Unpermute restores the source-rank token order:

```text
offsetD + probs + expandedRowIdx
  -> TopK weighted accumulation
  -> out[M, K]
```

## Memory Layout and HCCL Window

The HCCL remote window carries cross-rank visible data:

| Buffer | Location | Purpose |
| --- | --- | --- |
| `offsetA` | HCCL window | FrontReorder writes packed int8 token rows; Dispatch pulls from peer ranks |
| `offsetD` | HCCL window | Combine writes back to source ranks; Unpermute consumes locally |
| `tokenPerExpert` | HCCL window | count-as-flag cross-rank count rows |
| `gmA` | workspace GM | GMM1 input generated by Dispatch |
| `gmC` | workspace GM | GMM1 output and SwiGLU input |
| `gmPermutedToken` | workspace GM | SwiGLU dynamic-quant output and GMM2 input |
| `gmm2Output` | workspace GM | GMM2 output and Combine input |
| `expandedRowIdx` | workspace GM | inverse route index: route -> expert-major row |
| `cumsumMM / preSumBeforeRank` | workspace GM | address metadata for Dispatch and Combine |

`run.sh` estimates the HCCL window size from `maxOutputSize` and `K`, and raises `HCCL_BUFFSIZE` when needed.

## Measured Performance (Reference)

The performance comparison and pipeline diagram are in `overview.md`:

```text
overview.md
  - PTO-ISA measured performance comparison
  - PTO MegaMoE 2048-case overlap timeline
```

Current summary:

- In small-M cases, PTO MegaMoE is roughly on par with the AscendC implementation.
- As M increases, PTO GMM tiling and communication-compute overlap become more visible.
- In the 2048 case, AIC is nearly saturated; future work should reduce HBM contention between AIC and AIV.

## Performance Tuning Guide

### 1. Check the FrontReorder Case First

Small route counts should prefer FullLoad, which reduces GM intermediate traffic. If the path falls back to OneCore/MultiCore, focus on sort output visibility and `frontExpandedExpert/frontExpandDstToSrc` GM traffic.

### 2. Keep Expert-Level Stage Boundaries Clear

Dispatch, GMM1, SwiGLU, GMM2, and Combine depend on hard flags. Before removing any synchronization, verify the exact per-group set/wait relationship.

### 3. Prioritize GMM Tile Efficiency

GMM1/GMM2 dominate runtime. Check:

- whether output tile swizzle improves L1 reuse;
- whether L1 -> L0 ping/pong overlap is stable;
- whether small `currentM` causes AIC imbalance;
- whether AIV communication/writeback competes with GMM HBM traffic.

### 4. Use Subtile Combine for Small Token Counts

For small M, the direct row path may underuse AIVs. DirectSmall splits GMM2 tiles into subtiles, but requires the GMM2 tiling column width to align with the small-path subtile width.

### 5. Use Batch Golden Generation

Large synthetic cases use the `python-batch` golden backend by default. Use `python-naive` only for debugging against the old path.

## Build and Run

Configure the Ascend CANN environment:

```bash
export ASCEND_CANN_PATH=/usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann/cann
source /usr/local/Ascend/cann/cann/set_env.sh
```

Run the default 2048 case:

```bash
cd ${git_clone_path}/kernels/manual/a2a3/dispatch_mega_combine
./run.sh
```

The default case is currently `worldSize=8, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940`. A2/A3 users can use the script defaults and do not need to pass an explicit chip option in typical commands.

To run another typical M value, use the typical-case commands above directly. Example:

```bash
bash run.sh --world-size 8 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### Environment Variables

| Environment Variable | Purpose | Default Behavior |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN installation path | Must be set before running |
| `CMAKE_COMPILER` | Compiler used by CMake | `bisheng` |
| `MPI_ENV_BIN` | MPI/conda `bin` path | `/home/ntlab/miniconda3/envs/ltr_pto/bin` |
| `MPI_ENV_LIB` | MPI/conda `lib` path | `/home/ntlab/miniconda3/envs/ltr_pto/lib` |
| `MPI_LIB_PATH` | Absolute path to `libmpi.so` | `${MPI_ENV_LIB}/libmpi.so` |
| `MPI_RUNNER` | MPI launch command | `mpirun` |
| `HCCL_BUFFSIZE` | HCCL RDMA window size | Raised automatically by `run.sh` when needed |

## Changing Case Parameters

The typical performance data only compares different `M` values. Keep the other major parameters fixed and copy the full command from the supported-case table.

```bash
bash run.sh --world-size 8 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

Common constraints:

- `K` must satisfy packed-row, GMM1/GMM2 tile, and quantization-path alignment requirements.
- `N` is the GMM1 output dimension; GMM2 consumes `N / 2` after SwiGLU.
- `maxOutputSize` must cover the per-rank routed-row workspace limit; typical performance cases pass it explicitly.
- Synthetic `expert_idx` uses global token round-robin so small-M cases still cover global experts.

## FAQ

| Problem | Cause and Fix |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | Source the CANN environment and export `ASCEND_HOME_PATH` before running `run.sh` |
| HCCL window too small | The manually set `HCCL_BUFFSIZE` is below the case requirement; unset it or increase it |
| MPI launch fails | Check that `MPI_ENV_BIN`, `MPI_ENV_LIB`, and `MPI_LIB_PATH` point to the same conda/MPI environment |
| Golden generation is slow | Use the default `python-batch` backend; use `python-naive` only for debug comparison |
| Small-M performance is unstable | Check whether FullLoad is selected, whether Combine uses DirectSmall, and whether AIV concurrency is hurting GMM |
| Result diff is abnormal | Check whether old generated data was reused; do not reuse stale `out/` after changing expert distribution or key case parameters |

## Build System

- **Compiler**: `bisheng`
- **Device kernel flags**: `-xcce --cce-aicore-arch=${CCE_AICORE_ARCH}`
- **Host executable**: `-xc++ -std=c++17`
- **Targets**: `dispatch_mega_combine_kernel`, `dispatch_mega_combine`
- **Linked libraries**: `stdc++`, `ascendcl`, `hcomm`, `runtime`, `tiling_api`, `platform`, `nnopbase`, `pthread`, and others
- **PTO include**: repository root `include/` is added to the include path for PTO tile/communication helpers

## Changelog

| Date | Change |
| --- | --- |
| 2026-06-26 | Added `dispatch_mega_combine` README covering the MegaMoE operator, stage flow, build/run, and FAQ |
