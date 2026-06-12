# moe_combine - A2/A3 PTO MoE Combine Kernel

## Overview

This example implements the MoE combine stage with PTO on Ascend A2/A3-class chips. It is the return half of a
dispatch-compute-combine MoE pipeline: after local experts finish computing routed rows, the combine kernel returns those
rows to the original token owner rank and restores each token output with the gate weights.

The current kernel is a standalone combine kernel with an explicit low-level routing contract. Route information such
as `expert_ids`, `assist_info_for_combine`, and `ep_send_counts` is prepared by the upstream stage or host side and
passed in through `routeMeta`.

```text
expertOutput[local expert rows, K]
  -> variable-length return through HCCL peerWindow.ptrD
  -> cross-rank completion with TNOTIFY/TWAIT
  -> weighted restore: outputC[token, :] = sum(topK probs * returned rows)
```

## Supported AI Processors

- A2/A3, validated on Atlas 910B1

## Directory Layout

```text
kernels/manual/a2a3/moe_combine/
├── CMakeLists.txt           # Bisheng CCE + host build configuration
├── run.sh                   # One-click build/run wrapper, MPI discovery, HCCL_BUFFSIZE estimate
├── common.h                 # Shared ABI: shape, routeMeta layout, peerWindow layout, HCCL context
├── layout.h                 # Host-side layout calculators and HCCL_BUFFSIZE estimator
├── kernel_launchers.h       # Host-side kernel launcher declaration
├── moe_combine_kernel.cpp   # PTO AIV kernel: return + wait + weighted restore
├── main.cpp                 # Host orchestration: MPI, ACL, HCCL window, fixture, verify, profiling
├── golden.h                 # CPU golden data structures and public interface declarations
├── golden.cpp               # CPU golden route construction and output verification implementation
├── hccl_context.h           # HCCL window bootstrap and peer-window address exchange
├── comm_mpi.h               # MPI dynamic loading wrapper
├── DESIGN.md                # Split-from-dispatch design notes
├── PHASE2_DESIGN.md         # PTO style refactor plan
├── IMPLEMENTATION_PLAN.md   # Historical implementation tracker
├── README.md                # English README
└── README_zh.md             # Chinese README
```

## Operator Description

### Functionality

For each rank, the operator consumes expert outputs that are already laid out by local expert and source rank. It then:

1. Reads `routeMeta` to know how many rows each source rank sent to each expert and where those rows are located in
   `expertOutput`.
2. Uses PTO `TPUT` to return each expert row to the token owner rank's HCCL peer window.
3. Uses `TNOTIFY` / `TWAIT` to wait until every peer has completed its return writes.
4. Reads `routeMeta.expandedRowIdx` and `probs` to restore `outputC[M, K]`.

For token `t`, dispatch creates `topK` expert routes. After the combine return phase, the expert output rows for those
routes have been written back to this rank's `peerWindow.ptrD`. `expandedRowIdx[t * topK + slot]` records the row index
in `ptrD`, and `probs[t * topK + slot]` is the gate weight for that route.

For each output column `c`, restore is:

```text
outputC[t, c] = 0
for slot in 0..topK-1:
    row = expandedRowIdx[t * topK + slot]
    if row >= 0:
        outputC[t, c] += probs[t * topK + slot] * peerWindow.ptrD[row, c]
```

In other words, the kernel computes the gate-weighted sum of the `topK` expert rows for the same token and writes the
final `outputC[t, :]`.

### Scope

| Included | Not included |
| --- | --- |
| EP-domain combine return through HCCL window | Dispatch pack/gather kernel |
| Variable-length all-to-all-like return with `TPUT` | HCCL collective `AllToAllV` API |
| Weighted restore with `probs` | Expert FFN/GMM compute |
| Explicit low-level `routeMeta` contract | Quantization, TP ReduceScatterV, shared/copy/const experts |
| A2/A3 HCCL peer-window path | Upper-level public ABI adapter |

## Entry Contract

### Kernel Launcher ABI

```cpp
void LaunchMoeCombineKernel(MoeCombineShape shape, uint32_t myRank,
                            uint8_t *expertOutput,
                            uint8_t *probs,
                            uint8_t *outputC,
                            uint8_t *routeMeta,
                            uint8_t *peerWindow,
                            uint8_t *hcclCtx,
                            uint8_t *workspace,
                            void *stream,
                            uint32_t launchBlockCount);
```

### Runtime Inputs

| Argument | Direction | Storage | Meaning |
| --- | --- | --- | --- |
| `shape` | input | value | Static shape and AIV block count, such as `ep`, `m`, `k`, `topK`, `expertPerRank`, `aivBlocks` |
| `myRank` | input | value | Rank id in the EP domain |
| `expertOutput` | input | `aclrtMalloc` GM | Local expert result rows, shape `[maxOutputSize, K]`, fp16 |
| `probs` | input | `aclrtMalloc` GM | Gate weights, shape `[M, topK]`, fp32 |
| `outputC` | output | `aclrtMalloc` GM | Restored token output, shape `[M, K]`, fp16 |
| `routeMeta` | input | `aclrtMalloc` GM | Explicit combine routing ledger |
| `peerWindow` | input/output | HCCL RDMA window | Remote-visible `ptrD` return buffer and signal counters |
| `hcclCtx` | input | `aclrtMalloc` GM | Device-side HCCL window addresses for all ranks |
| `workspace` | temporary | `aclrtMalloc` GM | Local AIV soft-sync area |
| `stream` | input | ACL stream | Kernel launch stream |
| `launchBlockCount` | input | value | AIV block count for the kernel launch |

### `peerWindow` Contents

`localWindowBase` is the HCCL window base. On A2/A3, the `peerWindow` argument passed to the kernel points to the live
payload at `localWindowBase`; this layout does not reserve the A5 4096B head guard.

```text
A2/A3 localWindowBase
  peerWindow live payload:
    ptrD
    countReadySignal[ep]
    combineDoneSignal[ep]
```

| Field | Location | Meaning |
| --- | --- | --- |
| `ptrD` | HCCL window live payload | Return destination rows, remotely written by `TPUT` |
| `countReadySignal[ep]` | HCCL window live payload | Per-rank ready counter area |
| `combineDoneSignal[ep]` | HCCL window live payload | Per-rank completion counters; remote ranks `TNOTIFY` the corresponding slot after writing this rank's `ptrD` |

### `MoeCombineShape`

| Field | Meaning |
| --- | --- |
| `ep` | EP rank count |
| `m` | Tokens per rank |
| `k` | Hidden size |
| `topK` | Expert routes per token |
| `expertPerRank` | Local expert count per rank |
| `expertNum` | Global expert count, normally `ep * expertPerRank` |
| `maxOutputSize` | Per-rank expert-output row capacity |
| `aivBlocks` | Logical AIV block count; A3 default is `24`, can be overridden |

### `routeMeta` Layout

`routeMeta` is the explicit low-level combine ledger. It is local GM, not part of the HCCL window.

| Field | Shape | Meaning |
| --- | --- | --- |
| `peerTokenPerExpert` | `[ep, expertNumPadded]` int32 | Number of rows owned by each source rank for each global expert |
| `expandedRowIdx` | `[M * topK]` int32 | Token route to `peerWindow.ptrD` row mapping; `-1` means invalid route |
| `cumsumPerExpert` | `[ep, expertNumPadded]` int32 | Inclusive prefix by global expert for each source rank: `cumsum[src,e] = sum(peerTokenPerExpert[src,0..e])` |
| `dispatchOffset` | `[expertPerRank]` int32 | Base row in `expertOutput` for each local expert |
| `prevSumBeforeRank` | `[ep, expertPerRank]` int32 | Per-source offset inside a local expert's rows |

## Optimization Notes

This kernel is an AIV-only combine kernel. It is bandwidth-bound for large hidden sizes such as `K=7168`, where one fp16
row is 14 KiB. The main optimization goal is to keep GM/HCCL-window movement streaming while minimizing control-path
overhead.

- **Explicit routeMeta**: routing metadata is passed as a separate GM buffer. `peerWindow` is reserved for remote-visible
  return data and signals, and `workspace` only keeps the local AIV soft-sync area.
- **Chunked return sharding**: the return phase iterates `src_rank x local_expert` segments and shards row chunks across
  AIV blocks with `chunkBase % blockNum`.
- **PTO `TPUT` ping/pong path**: remote return uses `TPUT(remoteDst, localSrc, ping, pong)`, allowing MTE2 load and MTE3
  store movement to pipeline through UB.
- **Route cache for restore**: when `topK <= 16`, route rows and probabilities are cached in scalar arrays per token so
  the inner restore loop does not reload route metadata.
- **DCCI batched acquire before restore**: each token refreshes the returned `ptrD` rows before consuming them, then uses
  one `dsb(DSB_DDR)` for the cached route batch.
- **TLOAD to TAXPY event chain**: restore accumulates each returned row with a PTO `TLOAD -> TAXPY` event dependency.
- **Soft AIV sync**: `SoftSyncAiv` separates return, wait, and restore stages within the same kernel launch.

## Tiling and Default Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `PES` / `ep` | `2` | EP rank count |
| `M` | `64` | Tokens per rank |
| `K` | `7168` | Hidden size |
| `topK` | `8` | Expert routes per token |
| `expertPerPe` | `2` | Local experts per rank |
| `expertNum` | `4` | `PES * expertPerPe` |
| `maxOutputSize` | `PES * M * topK` | Default capacity, `1024` for the default shape |
| `aivBlocks` | `24` | A3 default logical AIV block count |
| Internal vector tile columns | `1024` | Fixed by the sample implementation |
| Internal return chunk | `8 rows` | Fixed return-stage row chunk |
| Internal metadata pad | `16` | Pads expert metadata rows |

For `PES=2, M=64, K=7168, topK=8, expertPerPe=2, aivBlocks=24`, the layouts are:

| Layout | Bytes |
| --- | --- |
| `workspace` | `2304` |
| `routeMeta` | `2432` |
| `peerWindow` | `7340160` |

## Overall Architecture

```text
Host:
  ParseArgs -> ComputeWorkspaceLayout / ComputeCombineRouteMetaLayout / ComputePeerWindowLayout
    -> PrepareHostData and CPU golden
    -> Init HCCL peer window
    -> AllocateLocalBuffers(routeMeta/workspace/expertOutput/probs/outputC)
    -> loop(warmup + measured):
         ClearDeviceState
         PrepareCombineFixture -> writes routeMeta + expertOutput
         LaunchMoeCombineKernel
         Verify outputC

Device:
  ReturnExpertRowsToOwners -> WaitCombinePhase -> RestoreOutputRows
```

```text
Return phase:
  routeMeta(peerToken/cumsum/offset) + expertOutput
    -> local or remote peerWindow.ptrD
    -> TNOTIFY peer combineDoneSignal[myRank]

Restore phase:
  routeMeta.expandedRowIdx + probs + peerWindow.ptrD
    -> outputC
```

## Kernel Details

### Stage 1: ReturnExpertRowsToOwners

The kernel walks all local expert segments:

```text
segment = src_rank * expertPerRank + localExpert
globalExpert = myRank * expertPerRank + localExpert
rows = routeMeta.peerTokenPerExpert[src_rank, globalExpert]
```

For each non-empty segment:

1. `srcStart` is computed from `dispatchOffset[localExpert] + prevSumBeforeRank[src_rank, localExpert]`.
2. `dstStart` is computed from `cumsumPerExpert[src_rank, globalExpert - 1]`; it is `0` when `globalExpert == 0`.
3. If `src_rank == myRank`, the row is copied locally into local `peerWindow.ptrD`.
4. Otherwise, PTO `TPUT` writes the row chunk into the source rank's remote peer window.

### Stage 2: WaitCombinePhase

After return writes finish, each rank notifies every token-owner rank:

```text
TNOTIFY(remotePeer.combineDoneSignal[myRank], AtomicAdd)
TWAIT(localPeer.combineDoneSignal[peer] >= 1)
```

The host clears `combineDoneSignal` before each iteration, so the kernel waits for one notify from every peer.

### Stage 3: RestoreOutputRows

Each AIV block owns a contiguous token shard. For each token and each column tile:

1. Initialize the output tile to zero with `TEXPANDS`.
2. For every valid route, load `ptrD[expandedRowIdx]`.
3. Accumulate with `TAXPY(outTile, ptrTile, prob)`.
4. Store the fp16 tile to `outputC`.

## Measured Performance

This project runs directly on A2/A3 machines (Atlas 910B1). The run script prints a profile block like this:

```text
[PROFILE] CombineTile
  M=64 K=7168 ranks=2 topK=8 expertPerPe=2 warmup=3 measured=5 samples=5
  prepare_fixture: avg=... us max=... us
  combine_e2e: avg=... us max=... us
  verify=PASS
```

Key metrics:

| Metric | Meaning |
| --- | --- |
| `combine_e2e` | Combine kernel launch to stream sync; excludes clear, fixture, verify, and MPI barriers outside the kernel launch window |
| `verify=PASS` | Device `outputC` matches CPU golden |

## Build and Run

### Environment

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
```

Load the CANN environment before invoking `run.sh`. Configure MPI in the shell before running if `mpirun` is not already
in `PATH`.

### Build Only

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh --skip-build 0 --clean-build 1
```

### Quick Verification

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --aiv-blocks 2
```

### Default Shape

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```

### Key CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `-pes` | `2` | Rank count |
| `-M` | `64` | Tokens per rank |
| `-K` | `7168` | Hidden size |
| `-topK` | `8` | Routes per token |
| `-expertPerPe` | `2` | Experts per rank |
| `--max-output-size` | `PES * M * topK` | Expert output row capacity |
| `--aiv-blocks` | `0 -> 24` | Logical AIV block count, override when matching a hardware resource plan |
| `--device-base` | `0` | First device id used by rank-to-device mapping |
| `--ndevices` | `PES` | Visible device count used by the sample launcher |

## Verification

The host builds a deterministic CPU golden route ledger, writes it to `routeMeta`, copies `expertOutput`, launches the
kernel, and compares `outputC` with the CPU golden output. Verification is enabled by default.

Expected successful output:

```text
verify=PASS
```
