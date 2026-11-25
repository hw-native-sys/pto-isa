# PTO MegaMoE Dispatch + Combine Fusion Example

Multi-server URMA is enabled with the same launcher by setting `--rank-num-per-server N`; the compute schedule
remains unchanged.

## Overview

This example implements an end-to-end MegaMoE fused operator with PTO Manual kernels. It combines token quantization,
route-mask exchange, grouped FFN compute, combine, and unpermute in one mixed-core kernel. The current implementation
uses MXFP8 data, direct AIC-to-AIV CV pipes, dynamic GMM task mailboxes, and token-ready RankStreaming Unpermute to
overlap communication, metadata preparation, AIC compute, and AIV processing.

The main device pipeline is:

```text
FrontReorder -> Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine -> Unpermute
```

## Supported AI Processors

- Ascend950PR (A5, arch35 / DAV_3510 family)
- Toolchain-compatible alias `Ascend910_9599`

This project always builds the mixed-core kernel with `dav-c310` and `PTO_NPU_ARCH_A5`. `Ascend910B` is an A3 target and is not a valid `--soc` value for this directory.

## Multi-server URMA Transport

The same binary supports both the existing MTE path and multi-server URMA. Use `--rank-num-per-server N` to enable
URMA for cross-server communication; same-server traffic continues to use MTE.

URMA uses HCCL SharedPool resources with independent queues owned by AIV0. Enabling URMA does not change the compute
pipeline, GMM scheduling, Combine flow, or Unpermute scheduling.

## Directory Layout

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # Build configuration for host executable and device kernel shared library
├── run.sh                          # One-click data generation, build, and mpirun entry
├── main.cpp                        # Host entry: case loading, ACL/HCCL/MPI setup, launch, verification, and timing
├── kernel_launch.cpp               # Device kernel launch wrapper
├── runtime_context.*               # Single-rank runtime, HCCL window, device/context management
├── tiling_builder.*                # Host tiling and workspace planning
├── data_utils.*                    # Case file IO and validation helpers
├── comm_mpi.h                      # MPI dynamic loading wrapper
├── scripts/
│   └── gen_data.py                 # Synthetic input, weight, and golden generation
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoE device pipeline entry
│   ├── front_reorder.h             # Token-order MXFP8 quantization and route-mask publication
│   ├── front_metadata_sort.h       # Expert-major route sort used by deferred metadata
│   ├── deferred_route_metadata.h   # Task descriptors, preSum, and inverse-route metadata
│   ├── dispatch.h                  # Mask compaction, remote token pull, and GMM1 input build
│   ├── gmm_common.h                # Shared GMM tile scheduling and helpers
│   ├── gmm_task_producer.h         # GMM1/GMM2 producer-consumer ticket scheduler
│   ├── gmm_expert_progress.h       # Combine completion to rank-progress coordinator
│   ├── gmm1.h                      # First grouped matmul and GMM1 mailbox consumer
│   ├── swiglu.h                    # SwiGLU activation and dynamic quantization
│   ├── gmm2.h                      # Producer-driven GMM2 mailbox consumer
│   ├── combine.h                   # CV-tile consumption and remote route-output writeback
│   ├── unpermute.h                 # Token-ready TopK reduction in original token order
│   └── utils/                      # PTO vector, sync, HCCL window, and GMM pipeline helpers
```

## Operator Description

### Functionality

The operator implements the multi-rank MoE FFN main path:

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> route-mask pull into destination expert-major rows
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

`FFN_expert` consists of MXFP8 GMM1, SwiGLU, and MXFP8 GMM2. Cross-rank data exchange uses the HCCL RDMA window and PTO communication/synchronization helpers.

### Specification

| Item | Value |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| Input | `x`: `[M, K]`, `bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`, `float32`; `weight1/weight2`: per-local-expert E4M3 data; `scale1/scale2`: E8M0 scales |
| Output | `out`: `[M, K]`, `bfloat16` |
| Kernel name | `dispatch_mega_combine_kernel` |
| Host executable | `dispatch_mega_combine` |
| Default script case | `worldSize=2, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## Optimization Notes

Front quantizes tokens and generates masks with one owner per expert. It publishes compact raw counts separately;
Dispatch and deferred metadata build expert-major cumulative row tables without restoring the old mask-tail protocol.

GMM1 uses only wave0 as the full-AIC direct prefix, followed by producer-assigned work. GMM2 uses mailbox tasks from its
first expert. The producer selects the GMM1/GMM2 split from the available work; `M <= 128` targets a 4-6 expert GMM2
lag, while larger cases target 1-2 experts. Internal GMM L1 panels adapt to the expert row count; production Gate/Up paired-weight panel loading and the interleaved CV handoff are retained.

| AIC | AIV | Default Dispatch AIV0 pool | Bootstrap GMM1:GMM2 | Large-M Unpermute phase1 AIV0 |
| ---: | ---: | ---: | --- | ---: |
| 28 | 56 | 18 | 20:8 | 18 |
| 32 | 64 | 21 | 21:11 | 21 |
| 36 | 72 | 24 | 24:12 | 24 |

Bootstrap groups are not the final runtime split. For `M <= 256`, Dispatch instead runs on all paired AIV1 workers
before SwiGLU; larger M uses the AIV0 pool above. Rank ownership uses quotient/remainder partitioning without dropping
workers. GMM grouping is independent of the Dispatch pool.

Combine retains the CV handoff and uses PTO `TSTORE`. Unpermute has one phase below M512; at M512 and above, the
released Dispatch AIV0 pool can process a frozen readiness snapshot before all AIVs join the final live phase.

`run.sh --aicore-num 0|28|32|36` chooses the effective core count; zero uses the device-reported value. Requests must not
exceed the physical core count. The existing rank limit is `min(32, aicNum - 2)`; wider rank configurations can expand
the Dispatch pool to retain at least one worker per rank.

## Tiling Parameters

| Parameter | Default / Description |
| --- | --- |
| `M` | Set by `run.sh --m` or `case.json` |
| `K` | Hidden size for GMM input; must satisfy packed-row and GMM tile alignment |
| `N` | FFN intermediate size; GMM1 output and `N/2` after SwiGLU |
| `topK` | Number of routed experts per token |
| `expertPerRank` | Runtime local-expert count: `4`, `8`, `16`, or `32`; all values share one kernel |
| `worldSize` | MPI/HCCL rank count, limited by `min(32, aicNum - 2)` |
| `maxOutputSize` | Per-rank routed-row workspace limit |
| `aicNum` | Effective AICore launch count; validated values are 28, 32, and 36 |
| `aivNum` | Derived as `2 * aicNum`: 56, 64, or 72 |
| `GMM baseM/baseN` | Main output tile shape is `256 x 256` |
| `Front Mask Pull` | The only Front implementation; host rejects clipping, invalid experts, inactive tokens, and insufficient receive capacity |
| `Fixed-role AIV UB` | Dispatch, SwiGLU, Combine, and Unpermute use a 216 KiB main region; the final 40 KiB is reserved for synchronization snapshots |
| `Dispatch / Unpermute tiles` | Dispatch uses an adaptive 2-6-slot packed-token ring; Unpermute keeps a full K row when possible, supports up to 8192 columns per tile, and uses an adaptive 2-6-slot input ring |
| `GMM / AIV CV tiles` | GMM1 sends paired `256 x 256` BF16 half-tiles through one CV slot; GMM2 sends `256 x 256` BF16 tiles through one CV slot |

## Supported Cases

The tuned eight-rank performance cases keep all major parameters fixed except `M`. The same shapes can be used as
two-rank smoke/correctness cases; generic tiling also supports other legal rank counts and runtime local-expert counts.

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=runtime (28, 32, or 36), or `run.sh --aicore-num 28|32|36`
aivNum=2 * aicNum
```

| M | maxOutputSize | Command |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## Overall Architecture

```text
Front: packed token records + route masks + raw expert counts
  -> Dispatch and deferred metadata in parallel
  -> GMM1: direct wave0 + mailbox work
  -> SwiGLU: paired AIV1 CV consumption and MXFP8 quantization
  -> GMM2: mailbox work with per-expert input readiness
  -> Combine: TSTORE to same-server output or cross-server URMA staging
  -> Unpermute: token-ready TopK accumulation
```

AIC work, AIV metadata, and communication overlap. The producer controls the runtime GMM split; the transport preserves
the same task and completion semantics in both MTE and URMA modes.

## FrontReorder Stage

All AIVs quantize their own tokens and then reuse their private buffers for route-mask generation. Each global expert
has one mask owner. Compact raw counts use `[localExpert][sourceRank]` order.

Deferred metadata builds the global expert-major `cumsumMM`, route ordering, inverse indices, and task descriptors.
GMM2 admission waits for the required metadata and preSum, not for all Dispatch workers to finish; input dependencies
are acquired by each consumer. URMA keeps its metadata exchange and materializes the counts in the same local format.

## Dispatch Stage

Dispatch pulls only routed source tokens and stores E4M3 data and E8M0 scales into the GMM1 workspace. Small cases use
rolling AIV1 workers; larger cases split AIV0 workers across ranks using quotient/remainder ownership. A 2-6-slot ring
overlaps loads and stores. Ready counts are published per expert and 256-row M tile after its payload stores complete.

Each small-M AIV1 publishes an expert-count row for its paired AIC. Cross-server token addressing continues to use the
URMA relay. Route records are retained only for optional debug capture, not for the computation.

## GMM1 / SwiGLU / GMM2 Stages

### GMM1

GMM1 runs on AIC and performs the first MXFP8 grouped matmul:

```text
gmA[E4M3] / gmAScale[E8M0] x weight1[E4M3] / scale1[E8M0]
  -> int32 accumulator
  -> fixpipe MX scaling and BF16 cast
  -> paired x/gate CV tiles
```

Each local expert is split into `256 x 256` half-output tiles; x and gate are computed as a pair and sent directly to
the paired AIV1 through a one-slot CV FIFO. All M cases execute direct wave0 and then consume mailbox work.
The internal L1 panel adapts to the expert row count while the logical output tile stays `256 x 256`.

### SwiGLU

SwiGLU runs on the paired AIV1 and consumes the BF16 x/gate CV tile:

```text
x/gate BF16 CV tile
  -> silu(x) * gate
  -> MXFP8 dynamic quantization
  -> gmSwigluA[E4M3] + gmSwigluScale[E8M0]
```

The CV control stream carries the exact GMM task sequence. SwiGLU drains each E4M3/E8M0 store before incrementing the
GMM2 dependency counter for that expert, and GMM2 starts the expert after the required SwiGLU tile count is complete.

### GMM2

GMM2 runs on AIC and performs the second MXFP8 grouped matmul:

```text
gmSwigluA[E4M3] / gmSwigluScale[E8M0] x weight2[E4M3] / scale2[E8M0]
  -> int32 accumulator
  -> fixpipe MX scaling and BF16 cast
  -> GMM2/Combine CV tile
```

Every GMM2 tile is a producer-assigned mailbox task. Released GMM1 AICs join according to the runtime split and
publication progress, without a global GMM1/GMM2 barrier. Each consumer acquires its input dependency before loading
and streams BF16 results to its paired Combine AIV1 through the CV FIFO.

## Combine / Unpermute Stages

Combine runs on the paired AIV1 and writes each BF16 GMM2 CV tile directly to source-rank compact output rows:

```text
GMM2 BF16 CV tile
  -> intersect expert rows with each source-rank cumsum range
  -> srcRank.combineOutputByRouteSlot[compactRow, 0:K]
```

Combine preloads `cumsumMM` and `preSumBeforeRank`, then uses strided PTO `TSTORE` for each source-rank
intersection. Before publishing an expert completion count, it drains prior remote MTE3 stores; the progress
coordinator therefore exposes an ordered expert prefix to each source rank.

Unpermute restores the source-rank token order:

```text
expandedRowIdx[token * topK + topk]
  -> combineOutputByRouteSlot[compactRow] * probs[token, topk]
  -> TopK weighted accumulation
  -> out[M, K]
```

For `M < 512`, all AIVs run the single live phase. For `M >= 512`, the released Dispatch AIV0 pool first consumes
tokens admitted by a frozen progress snapshot, then all AIVs consume remaining tokens from live expert-prefix progress. RankStreaming
is the only Unpermute implementation; unsupported shapes fail host validation instead of falling back to a barrier
kernel.

## Memory Layout and HCCL Window

The HCCL remote window carries cross-rank visible data:

| Buffer | Location | Purpose |
| --- | --- | --- |
| `sourceTokenRecords` | HCCL window | One packed E4M3/E8M0 record per source token; Dispatch pulls by `routeSlot / topK` |
| `routeMaskSlots` | HCCL window | Per-local-expert/source-rank mask plus aligned count record |
| `combineOutputByRouteSlot` | HCCL window | Expert-major compact BF16 rows written by Combine and indexed through `expandedRowIdx` |
| `preSumBeforeRank` | HCCL window | Source-rank compact-row base for each global expert |
| signal tail | HCCL window | Front/preSum/data-ready epochs and rank-progress slots |
| `gmA` / `gmAScale` | workspace GM | E4M3/E8M0 GMM1 input generated by Dispatch |
| `gmSwigluA` / `gmSwigluScale` | workspace GM | E4M3/E8M0 GMM2 input generated by SwiGLU |
| `cumsumMM` | workspace GM | source-major cumulative rows for Dispatch destination offsets |
| `sortedRouteSlot` / `expandedRowIdx` | workspace GM | Expert-major route order and its route-slot-to-compact-row inverse |
| GMM queues/mailbox | workspace GM | Runtime task descriptors, tickets, dependency counters, and completion counters |

Configure `HCCL_BUFFSIZE` in the external runtime environment.
C++ tiling owns the layout and rejects insufficient capacity; increase the variable for larger cases.

## Build and Run

Source `set_env.sh` from the CANN installation before building. It provides `ASCEND_HOME_PATH` and the toolchain
environment.
Configure MPI and CANN runtime variables externally, including library search paths, HCCL options, and device
visibility consistent with the rank-to-device mapping. `run.sh` inherits these settings without changing them.
Data generation requires Python 3.10 or later with NumPy.

Build the A5 mixed-core kernel and host without generating case data:

```bash
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
bash run.sh --build-only
```

Run on contiguous available A5 devices. The environment must provide MPICH; OpenMPI is not supported by the MPI
compatibility wrapper. Add the MPICH launcher directory to `PATH` or set `MPI_RUNNER`. Make its shared libraries
available through the system dynamic loader (for example, `LD_LIBRARY_PATH`), or set `MPI_LIB_PATH` explicitly.
The launcher and library must come from the same MPICH installation; installation paths belong in the deployment
environment configuration.

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 2048 --k 7168 --n 4096 \
  --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

This maps ranks 0 and 1 to physical devices 2 and 3. `Ascend910_9599` is the A5 toolchain alias here; it must not be
replaced with the A3 target `Ascend910B`.

For another `M` value, keep the initial two-rank setup, for example:

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### Environment Variables

| Environment Variable | Purpose | Default Behavior |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN installation path | Must be set externally before building |
| `CMAKE_COMPILER` | Compiler used by CMake | `bisheng` |
| `FIRST_DEVICE` | First physical device in the contiguous rank mapping | `0`; overridden by `--first-device` |
| `MPI_LIB_PATH` | Optional path to the MPICH shared library | Otherwise resolves `libmpi.so` or `libmpich.so` through the system dynamic loader |
| `MPI_RUNNER` | MPICH launch command | `mpirun` from `PATH` |
| `HCCL_BUFFSIZE` | HCCL RDMA window size | Supplied by the external runtime environment |
| `DISPATCH_MEGA_COMBINE_AICORE_NUM` | Effective AIC count | `0`, which uses the runtime-reported count |
| `DISPATCH_MEGA_COMBINE_REUSE_DATA` | Reuse compatible generated case data | Disabled; a nonzero value is equivalent to `--reuse-data` |
| `DISPATCH_MEGA_COMBINE_WARMUP_ITERS` | Warmup launches before timing | `3` |
| `DISPATCH_MEGA_COMBINE_MEASURE_ITERS` | Timed launches used for the kernel summary | `5` |

### Kernel Performance

After the timed launches, rank 0 prints one `[KERNEL_PERF]` summary for the complete kernel. Each AIC/AIV records only
its overall start/end system-counter values; the summary reports the maximum rank duration per iteration together with
average, minimum, maximum, standard deviation, token throughput, equivalent compute TFLOPS, and equivalent
communication bandwidth.

## Changing Case Parameters

When changing a case, keep its MXFP8 dimensions, rank topology, and receive capacity within host-validated limits.

```bash
bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

Common constraints:

- `K` must be a multiple of 128 and satisfy the Dispatch packed-row capacity.
- `N` must be even and `N / 2` must be a multiple of 128.
- `topK` must be in `1..32`.
- `expertPerRank` must be `4`, `8`, `16`, or `32`; all values use the same runtime kernel with capacity for 32 local
  experts.
- `worldSize` must not exceed `min(32, aicNum - 2)` for the selected 28/32/36-AIC topology.
- `maxOutputSize` must cover the per-rank routed-row workspace limit.
- Each RankStreaming phase is limited to 256 tokens per worker; unsupported shapes are rejected by host tiling.
- Synthetic `expert_idx` uses global token round-robin so small-M cases still cover global experts.

## FAQ

| Problem | Cause and Fix |
| --- | --- |
| CANN installation path is missing | Source `set_env.sh` from the CANN installation to set `ASCEND_HOME_PATH` |
| HCCL window too small | Increase `HCCL_BUFFSIZE` in the external runtime environment to meet the case requirement |
| MPI launch fails | Check `MPI_RUNNER`/`PATH` and the matching MPICH library configuration; the launcher version must report MPICH/HYDRA; OpenMPI is unsupported |
| Golden generation is slow | Reuse the generated files with `--reuse-data` after the first run; the chunk size is fixed internally |
| Shape or rank topology is rejected | Check the 28/32/36-AIC selection, rank limit, MXFP8 alignment, UB capacity, and RankStreaming token-per-worker limit |
| Result diff is abnormal | Check whether old generated data was reused; do not reuse stale `out/` after changing expert distribution or key case parameters |

## Build System

- **Compiler**: `bisheng`
- **Device kernel flags**: `-xcce --cce-aicore-arch=dav-c310` plus the A5 address-transform and explicit DCCI policy
- **Host executable**: `-xc++ -std=c++17`
- **Targets**: `dispatch_mega_combine_kernel`, `dispatch_mega_combine`
- **Linked libraries**: `stdc++`, `ascendcl`, `hcomm`, `runtime`, `tiling_api`, `platform`, `nnopbase`, `pthread`, and others
- **PTO include**: repository root `include/` is added to the include path for PTO tile/communication helpers

## Changelog

| Date | Change |
| --- | --- |
| 2026-06-26 | Added `dispatch_mega_combine` README covering the MegaMoE operator, stage flow, build/run, and FAQ |
| 2026-07-27 | Ported the optimized A3 schedule to the A5 backend and completed compile-only validation |
| 2026-08-14 | Removed development instrumentation and prepared the A5 target for production use |
| 2026-08-20 | Updated the production path with runtime expert counts, deferred metadata, hybrid GMM1, mailbox GMM2, CV handoff, RankStreaming-only Unpermute, and ordered Combine progress publication |
