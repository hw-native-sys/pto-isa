# Single-device Multi-block FFN GridPipe Demo

## Goal

`distributed_ffn_grid_reducesum` validates a single-device logical FFN grid on A2/A3. The host runs one process on the selected device and launches `gridRows * gridCols` blocks. Each block owns one logical cell:

- Rows are data-parallel token slices.
- Columns are model-parallel FFN intermediate shards.
- The mixed Cube/Vec kernel computes gate/up, activation, down projection, and row-local `EAST` reduce in one launch.
- The last column in each row writes the final fp32 `[T, H]` output tile, which the host compares with `golden.bin` using `1e-3` tolerance.

The `EAST` reduce uses the A2/A3 GridPipe mock backend: local SRAM windows backed by GM in the mock, fake `HcclDeviceContext` window pointers, ready/free counters, `dcci/dsb` fences, and spin waits. This validates the programming model and same-device mock path; it is not multi-card communication validation.

An AllGather variant is also provided in `run_allgather.sh` / `distributed_ffn_grid_allgather`. It gathers hidden shards across columns before down projection, so the post-down ReduceSum is removed and each column writes its own output-H shard.

Beyond the nearest-neighbor FFN demos, GridPipe also supports routed K-hop unicast (`TPUSH<dir, dist>` / `TPOP<dir, dist>`) and single-source row/column broadcast (`TPUSH<GridSpan>`). Each capability has a standalone Vec-only smoke test under `smoke/` (see [Smoke tests](#gridpipe-smoke-tests)).

## Files

| File | Purpose |
| --- | --- |
| `README.md` / `README_zh.md` | English / Chinese documentation. |
| `CMakeLists.txt` | Builds the host executables and mixed Cube/Vec device kernel shared libraries. |
| `run_reducesum.sh` | Sets up CANN, generates ReduceSum data, configures CMake, builds, and runs the ReduceSum demo. |
| `run_allgather.sh` | Sets up CANN, generates AllGather data, configures CMake, builds, and runs the AllGather demo. |
| `ffn_config.hpp` | Compile-time grid shape, tile shape, GridPipe window sizes, buffer sizes, and PReLU alpha. |
| `kernel_launch.hpp` | Host-side mixed kernel launch declaration. |
| `main_reducesum.cpp` | ReduceSum host driver: ACL setup, fake HCCL context/local GridPipe windows, device buffers, data loading, kernel launch, golden comparison, and cleanup. |
| `distributed_ffn_grid_reducesum_compute_kernel.cpp` | ReduceSum mixed Cube/Vec kernel. Cube computes GEMMs; Vec computes activation/cast and GridPipe `EAST` reduce. |
| `main_allgather.cpp` | AllGather host driver. |
| `distributed_ffn_grid_allgather_compute_kernel.cpp` | AllGather mixed Cube/Vec kernel. Vec gathers hidden shards; Cube writes output-H shards. |
| `tpipe_tmov_inl.hpp` | Directional `TMOV` overloads that lower Cube↔Vec C2V/V2C transfers to the existing `TPUSH`/`TPOP`, so the kernel body never spells out the handshake. |
| `gridpipe_payload_inl.hpp` | Local GridPipe payload hooks and fake-window remote pointer adapter. |
| `smoke/` | Standalone GridPipe feature smoke tests. `khop_smoke_{config,kernel,launch}` + `main_khop_smoke.cpp` + `run_khop_smoke.sh` cover routed K-hop unicast; `bcast_smoke_*` + `run_bcast_smoke.sh` cover single-source row/column broadcast. Both build through the parent `CMakeLists.txt`. |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | Consolidated GridPipe A2/A3 intrinsic layer. Section 3 is the V6 IPC_SCB scoreboard API (`sync_neighbor_scb`/`wait_local_spr`/`mov_local_spr`) used by GridPipe ready/free waits and notifications; Section 4 is the neighbor SRAM address + payload transfer API (`copy_ubuf_to_neighbor_ubuf`/`copy_local_slot_to_ubuf`), plus the `GmSramArena` address-segment SRAM model and the `sram_pop_is_local` TPOP read-locality guard. |
| `scripts/gen_data.py` | Generates per-cell fp16 X/weight shards and an fp32 golden reference. |
| `build/` | Ignored generated build directory. |
| `out/` | Ignored generated data directory. |

## Execution Flow

1. `run_reducesum.sh` parses arguments. Defaults are `gridRows=2`, `gridCols=2`, `T=16`, `H=64`, `Fi=64`, and `n-ranks=1`.
2. Unless `--build-only` is set, `scripts/gen_data.py` generates per-cell input and weight files plus `golden.bin`.
3. CMake builds two targets:
   - `distributed_ffn_grid_reducesum_mixed_kernel`: `dav-c220` mixed Cube/Vec.
   - `distributed_ffn_grid_reducesum`: host executable.
4. The host initializes ACL on the selected device.
5. The host allocates contiguous device buffers for `gridRows * gridCols` cells.
6. The host allocates one local GridPipe SRAM window per cell, backed by GM in the mock, and builds a fake `HcclDeviceContext`:

```text
windowsIn[cell] = reduce_pipe_windows_dev + cell * FFN_GRID_WINDOW_BYTES
rankNum = gridRows * gridCols
winSize = FFN_GRID_WINDOW_BYTES
```

7. The host loads X and weights into row-major per-cell buffers.
8. The host obtains the FFTS base address with `rtGetC2cCtrlAddr()` and launches `DistributedFfnGridMixedKernel` once with `gridRows * gridCols` blocks.
9. Inside each block, Cube and Vec branches exchange intermediate tiles through A2/A3 `TPipe` FIFOs. The kernel issues these C2V/V2C transfers as a directional `TMOV` (`TMOV(pipe, tile)` to produce, `TMOV(tile, pipe)` to consume); the underlying `TPUSH`/`TPOP` stays implicit (see `tpipe_tmov_inl.hpp`):

```text
Cube:
  X[row] @ W_gate[col] -> gatePartial[row,col] --TMOV C2V-->
  X[row] @ W_up[col]   -> upPartial[row,col]   --TMOV C2V-->

Vec:
  hidden[row,col] = fp16(PReLU(gatePartial) * upPartial)
  hidden[row,col] --TMOV V2C-->

Cube:
  hidden[row,col] @ W_down[col] -> downPartial[row,col] --TMOV C2V-->

Vec:
  downPartial --GridPipe EAST reduce across cols--> yOutput[row] on final col
```

The cross-cell `EAST`/`WEST` reduce and gather keep their explicit GridPipe `TPUSH`/`TPOP`; only the in-block Cube↔Vec C2V/V2C traffic is folded into `TMOV`.

10. The host synchronizes the stream, checks GridPipe fault flags, copies `yOutput` back, and compares it with `golden.bin`.

## Key Designs

### Mixed Cube/Vec launch

The device kernel is compiled for `dav-c220`. Cube and Vec code paths are guarded by `__DAV_CUBE__` and `__DAV_VEC__`, so both sides live in one kernel source and synchronize through regular A2/A3 `TPipe` ready/free handshakes.

### Implicit C2V/V2C `TMOV`

`tpipe_tmov_inl.hpp` adds two `TMOV` overloads so the kernel body expresses Cube↔Vec transfers as a single tile-move op instead of explicit `TPUSH`/`TPOP`:

- `TMOV(pipe, tile)` — producer side; forwards to `TPUSH` (write `tile` into the C2V/V2C FIFO).
- `TMOV(tile, pipe)` — consumer side; forwards to `TPOP` (read the next slot into `tile`).

Which physical core writes vs reads, and whether the pipe is C2V or V2C, stays encoded in the `TPipe` type and its `__DAV_CUBE__`/`__DAV_VEC__` guards, so the call site is direction-agnostic. The overloads take exactly two `(pipe, tile)`/`(tile, pipe)` arguments (no wait-event pack), which makes them strictly more specialized than the generic tile-to-tile `TMOV(dst, src, ...)`; overload resolution therefore selects them for any `TPipe`/tile pair and leaves every other `TMOV` use unchanged. This keeps the Cube↔Vec handshake implicit at the call site, the way a real WSE fabric move hides the producer/consumer split, while reusing the existing `TPUSH`/`TPOP` sync and record machinery verbatim.

### Single-device logical grid

`get_block_idx()` is the row-major cell id:

```text
cell = get_block_idx()
row  = cell / gridCols
col  = cell % gridCols
```

All cells run on one device. `gridRows` controls data-parallel token tiles, and `gridCols` controls model-parallel FFN shards.

### Local GridPipe mock

The host allocates `gridRows * gridCols` local SRAM windows, backed by GM in the mock. `TPUSH<EAST>` resolves the east neighbor's SRAM slot with `get_neighbor_sram_addr`, writes the payload, then publishes the ready counter; `TPOP<EAST>` waits on the local ready counter, loads the local SRAM slot, and returns free credit to the west neighbor.

The mock uses GM flag polling and cache maintenance to emulate the intended LPU WSE `SPR` / `WFE` behavior on A2/A3.

### NoC write-only address-segment SRAM model (`GmSramArena`)

To stay close to real silicon, the mock models future-hardware per-core SRAM as an explicit **GM address-segment arena**. The single contiguous `gridRows*gridCols * FFN_GRID_WINDOW_BYTES` window buffer is cut into equal per-core segments, so segment `c` (== `windowsIn[c]`) *is* core `c`'s private SRAM:

```text
segment c = [base + c*winSize, base + (c+1)*winSize)   // base == windowsIn[0]
```

`GmSramArena` (in `include/pto/npu/a2a3/grid_intrinsic.hpp`) carries `{base, segBytes, numSegs}` plus the `SegmentOf` / `InSegment` classifiers; the demo builds it on-device from the fake `HcclDeviceContext` window table (`SramArenaFromCtx`). It is the single source of truth for "which core owns this address".

This makes the NoC contract explicit and **enforced**: the fabric can only *write* across cores, never *read*.

- `TPUSH<dir>` writes a payload into the **neighbor's** segment — a cross-segment write, exactly what the fabric does.
- `TPOP<dir>` may only drain **this core's own** segment. `GRID_TRY_TPOP_IMPL` calls the `sram_pop_is_local` guard before the payload read; on a cross-segment read it raises `kFaultPopNonLocal` (`0x205`, "pop non-local segment") and aborts the pop. The host's `CheckGridPipeFaults` surfaces it.

On native hardware `sram_pop_is_local` is a no-op (`true`): a TPOP read address is local by construction because the fabric has no remote-read path. The guard exists only because the A2/A3 mock backs SRAM with a GM window that *can* physically read any address, so without it a demo could silently rely on a remote read the silicon cannot perform. A compile-time `static_assert(GmSramArenaSelfCheck())` is built into every A2/A3 kernel, so a regression in the segment math fails the build rather than mis-routing a pop.

> The `pto::comm` variants (`TREDUCE` / `TGATHER`) intentionally do **not** follow this rule: they are a root-pulls-from-every-rank collective (HCCL/RDMA-style remote reads), a different memory model from the WSE NoC. Only the GridPipe `TPUSH`/`TPOP` path is constrained to write-only.

### IPC_SCB scoreboard intrinsic API

GridPipe ready/free synchronization follows the V6 IPC_SCB scoreboard route, exposed as CCE-intrinsic-style APIs in `include/pto/npu/a2a3/grid_intrinsic.hpp` (Section 3). The canonical call form keeps hardware semantic operands first and the mock backend operand last:

- `sync_neighbor_scb(kind, dir, dist, abs_count, operand)` (V6 `SYNC_HSCB`/`ST_HSCB`) stores this core's new absolute count into the peer's `ready_scb`/`free_scb` IPC_SCB `dist` hops away.
- `wait_local_spr(kind, dir, threshold, operand, maxSpins)` (V6 `WAIT_SPR`) blocks until the **local** `ready_scb`/`free_scb` IPC_SCB reaches `threshold`.
- `mov_local_spr(kind, dir, operand)` (V6 `MOV_SPR2X`) non-blocking peek of the local scoreboard, used for the fast path before `wait_local_spr`.

GridPipe payload address resolution goes through the same header (Section 4):

- `get_neighbor_sram_addr(dst, src, dir, peerRank, operand)` resolves a local slot offset to the same offset in a neighbor SRAM slot address register.
- `copy_ubuf_to_neighbor_ubuf(dst, src, bytes, config)` (V6 `COPY_UBUF_TO_NBR`) writes a local UB payload into the neighbor's UB/L1 slot.
- `copy_local_slot_to_ubuf(dst, src, bytes, config)` drains this core's own slot into the tile. V6 has **no cross-core read** of payload: native lowering is the existing local TLOAD/TMOV (an interface placeholder here); the A2/A3 mock reads the GM-backed local slot.
- `sram_pop_is_local(slot, bytes, callerRank, operand)` is the TPOP read-locality guard (see the address-segment model above): native lowering returns `true`, while the A2/A3 mock checks `slot` against `callerRank`'s `GmSramArena` segment so a cross-segment read is rejected instead of silently serviced.

Native lowering targets the real CCE HSCB/IPC_SCB stack (`__sync_hscb`/`__st_hscb`, `get_ipc_scb_*`, and `try_wait(CROSS_CORE)` for the wait, since the header exposes no blocking `WAIT_SPR` on IPC_SCB yet); the current A2/A3 mock stands in for those IPC_SCB scoreboards with GM words + cache maintenance, and degrades `WAIT_SPR` to a spin-poll. Once native hardware provides neighbor-IPC_SCB addressing (V6 HW-DEP-1) and the `COPY_UBUF_TO_NBR` builtin (V6 HW-DEP-0), GridPipe call sites do not need to change.

`TPUSH<EAST>` peeks/waits on the local `free_scb` with `mov_local_spr`/`wait_local_spr`, writes the payload slot, then publishes its `prod_idx` to the downstream `ready_scb` with `sync_neighbor_scb`. `TPOP<EAST>` waits on the local `ready_scb`, reads the payload slot, then publishes its `cons_idx` to the upstream `free_scb`.

On current A2/A3 boards, `NeighborCounterOperand::addr` points to a GM-backed counter in the local/fake peer GridPipe window, and `NeighborSramOperand::runtimeCtx` points to the fake HCCL context. When real hardware supports neighbor SPR/WFE counters and neighbor SRAM address registers, this demo should be adapted by compiling GridPipe with `PTO_GRID_COUNTER_NATIVE_INTRINSIC` and `PTO_GRID_SRAM_NATIVE_INTRINSIC` and providing the compiler builtins. In that mode the mock operands are ignored, and the host/device setup should move from fake GM windows to hardware-provided per-neighbor counter/event registers and SRAM slot bases while keeping the GridPipe `TPUSH/TPOP` call sites unchanged.

### fp32 EAST reduction

The reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This keeps `downPartial`, `yOutput`, and `golden.bin` in fp32 for direct tolerance-based comparison.

### Routed K-hop unicast

`TPUSH<dir, dist>` / `TPOP<dir, dist>` extend the nearest-neighbor pipe to a routed unicast `dist` hops along `dir` (Scheme A). The payload write resolves the slot in the `dist`-hop neighbor's segment, and the ready/free scoreboard stores carry the same `dist` operand through `sync_neighbor_scb`, so the receiver `dist` hops downstream pops the tile with no relays in between. `dist == 1` reproduces the original nearest-neighbor behavior. Fan-in stays 1 (one upstream per direction/distance), so no slot/flag expansion is needed.

### Single-source row/column broadcast

`TPUSH<GridSpan>` (`ROW` or `COL` as the first template argument selects the multicast overload) lets one source cell broadcast its tile to every other cell in its row or column as one op: the per-target writes are batched with no inter-target fence, the whole broadcast pays a single publish fence, then all ready doorbells fire. It is not lowered to a per-hop `TPUSH` loop. Receivers drain with the ordinary `TPOP<dir, dist>` toward the source (`EAST`/`WEST` for a row span, `NORTH`/`SOUTH` for a column span).

### GridPipe smoke tests

The two capabilities above each have a Vec-only data-movement smoke test under `smoke/` (no Cube, no matmul, no data files; in-process verification on the same GM-backed mock as the FFN demos):

- `khop_smoke` — a `1 x cols` row; each cell pushes a stamped fp32 `[T, W]` tile `DIST` hops east, the receiver pops and stores it; the host checks `out[c] == in[c-DIST]`.
- `bcast_smoke` — one source cell (`--src`) broadcasts its stamped tile to its whole row (or column with `--span-col 1`); every other cell pops at its own direction/distance and stores it; the host checks `out[cell] == in[source]`. The default 1x5 row with the source at col 2 exercises both arms in one run.

### AllGather variant

`run_allgather.sh` uses `scripts/gen_data.py --split-mode allgather` to generate data. In this mode, `W_down` is split as `[F, Hc]`, GridPipe carries fp16 `hidden [T, Fi]`, and each column writes one `Y[:, Hc]` output shard. The host still compares the stitched full output with `golden.bin`. AllGather requires `--model-tile` to be divisible by `--grid-cols` so `Hc = H / gridCols` is an integer tile width.

## How to Run

### Build only

```bash
bash run_reducesum.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
bash run_allgather.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
```

### Run on NPU

```bash
bash run_reducesum.sh -r npu -v Ascend910B1 --device-id 0 --grid-rows 3 --grid-cols 3
bash run_allgather.sh -r npu -v Ascend910B1 --device-id 0 --grid-rows 3 --grid-cols 3 --model-tile 96
```

### GridPipe smoke tests

```bash
# Routed K-hop unicast: 1x4 row, shift-by-2
bash smoke/run_khop_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 4 --dist 2
# Single-source broadcast: 1x5 row, source at col 2 (use --span-col 1 + Rx1 grid for a column broadcast)
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2
```

Both accept `--build-only` and need no data generation.

### Common arguments

```text
-r, --run-mode      sim or npu, default npu
-v, --soc-version   default Ascend910B1
-n, --n-ranks       fixed to 1
-d, --device-id     selected ACL device id; defaults to FFN_GRID_DEVICE_ID, ASCEND_DEVICE_ID, DEVICE_ID, then 0
--grid-rows         logical grid row count, default 2
--grid-cols         logical grid column count, default 2
--token-tile        token tile T per cell, default 16
--model-tile        hidden dim H, default 64; AllGather requires H % gridCols == 0
--ffn-tile          intermediate dim Fi per column, default 64
--build-only        build only; skip data generation and execution
```

The smoke scripts reuse `-r/-v/-d`, `--grid-rows/--grid-cols`, `--token-tile/--model-tile` (tile `[T, W]`), and `--build-only`; `run_khop_smoke.sh` adds `--dist` (hop count, default 2) and `run_bcast_smoke.sh` adds `--src` (source index, default 2) and `--span-col` (1 = column broadcast, default 0).

## Expected Result

On success, the ReduceSum executable prints:

```text
[SUCCESS] Single-device multi-block FFN GridPipe PASS.
```

On success, the AllGather executable prints:

```text
[SUCCESS] Single-device multi-block FFN GridPipe AllGather PASS.
```

The smoke tests print:

```text
[SUCCESS] GridPipe K-hop unicast smoke PASS.
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```
