# Single-device Multi-block FFN GridPipe Demo

## Goal

`distributed_ffn_grid_reducesum` validates a single-device logical FFN grid on A2/A3. The host runs one process on the selected device and launches `gridRows * gridCols` blocks. Each block owns one logical cell:

- Rows are data-parallel token slices.
- Columns are model-parallel FFN intermediate shards.
- The mixed Cube/Vec kernel computes gate/up, activation, down projection, and row-local `EAST` reduce in one launch.
- The last column in each row writes the final fp32 `[T, H]` output tile, which the host compares with `golden.bin` using `1e-3` tolerance.

The `EAST` reduce uses the A2/A3 GridPipe mock backend: local SRAM windows backed by GM in the mock, fake `HcclDeviceContext` window pointers, ready/free counters, `dcci/dsb` fences, and spin waits. This validates the programming model and same-device mock path; it is not multi-card communication validation.

An AllGather variant is also provided in `run_allgather.sh` / `distributed_ffn_grid_allgather`. It gathers hidden shards across columns before down projection, so the post-down ReduceSum is removed and each column writes its own output-H shard.

Beyond the nearest-neighbor FFN demos, GridPipe also supports routed K-hop unicast (`TPUSH<dir, dist>` / `TPOP<dir, dist>`) and concurrent group broadcast (`TBROADCAST<GridGroup>` / `TPOP<GridGroup>`). Each capability has a standalone Vec-only smoke test under `smoke/` (see [Smoke tests](#gridpipe-smoke-tests)).

## Files

| File | Purpose |
| --- | --- |
| `README.md` / `README_zh.md` | English / Chinese documentation. |
| `CMakeLists.txt` | Builds the host executables and mixed Cube/Vec device kernel shared libraries. |
| `run_reducesum.sh` | Sets up CANN, generates ReduceSum data, configures CMake, builds, and runs the ReduceSum demo. |
| `run_allgather.sh` | Sets up CANN, generates AllGather data, configures CMake, builds, and runs the AllGather demo. |
| `ffn_config.hpp` | Compile-time grid shape, tile shape, GridPipe window sizes, buffer sizes, SwiGLU clamp bounds, the A3 precision-mapping table, and Batcher GM arena byte sizes. |
| `kernel_launch.hpp` | Host-side mixed kernel launch declaration. |
| `main_reducesum.cpp` | ReduceSum host driver: ACL setup, fake HCCL context/local GridPipe windows, working buffers, Batcher load/distribute/broadcast, kernel launch, golden comparison, and cleanup. |
| `distributed_ffn_grid_reducesum_compute_kernel.cpp` | ReduceSum mixed Cube/Vec kernel. Cube computes GEMMs; Vec computes the SwiGLU activation/cast and the GridPipe `EAST` reduce. |
| `main_allgather.cpp` | AllGather host driver. |
| `distributed_ffn_grid_allgather_compute_kernel.cpp` | AllGather mixed Cube/Vec kernel. Vec gathers hidden shards; Cube writes output-H shards. |
| `batcher.hpp` | Host-side GM-simulated **Batcher**: owns the full input + the full DRAM-resident weights in GM, splits them column-parallel into per-col shards, broadcasts x per-row, and exposes the output-collection region. Replaces the legacy per-cell X/weight load path. |
| `tpipe_tmov_inl.hpp` | Directional `TMOV` overloads that lower Cube↔Vec C2V/V2C transfers to the existing `TPUSH`/`TPOP`, so the kernel body never spells out the handshake. |
| `gridpipe_payload_inl.hpp` | Local GridPipe payload hooks and fake-window remote pointer adapter. |
| `smoke/` | Standalone GridPipe feature smoke tests. `khop_smoke_{config,kernel,launch}` + `main_khop_smoke.cpp` + `run_khop_smoke.sh` cover routed K-hop unicast; `bcast_smoke_*` + `run_bcast_smoke.sh` cover single-source row/column broadcast. Both build through the parent `CMakeLists.txt`. |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | V8 GridPipe CCE facade layer: `copy_ubuf_to_neighbor_ubuf` (G1 `COPY_UBUF_TO_NBR`), `sync_hscb` (G2 `SYNC_HSCB`/`ST_HSCB`), `wait_ipc_scb`/`wait_ipc_scb_sim` (G3 `WAIT_SPR`, read+block in one instruction, no `MOV_SPR2X` peek). Each forwards 1:1 to a `__builtin_cce_*` under `PTO_GRID_CCE_NATIVE`, else emulates the same semantics with a GM word + cache maintenance. |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 data model + mock support: Section 1 is the mesh model + neighbor / K-hop / group resolvers; Section 2 is the GM-mock boundary-fault sentinels; Section 3 is the `GmSramArena` address-segment SRAM model + the TPOP read-locality guard. |
| `scripts/gen_data.py` | Generates the FULL fp16 X/weight tensors (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) the Batcher consumes, plus an fp32 SwiGLU `golden` reference. |
| `build/` | Ignored generated build directory. |
| `out/` | Ignored generated data directory. |

## Execution Flow

1. `run_reducesum.sh` parses arguments. Defaults are `gridRows=2`, `gridCols=2`, `T=16`, `H=64`, `Fi=64`, and `n-ranks=1`.
2. Unless `--build-only` is set, `scripts/gen_data.py` generates the full-tensor Batcher inputs (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) plus the SwiGLU `golden.bin`.
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

7. The host **Batcher** (`batcher.hpp`) loads the full input + full DRAM-resident weights into GM, splits the weights column-parallel into per-col shards, and broadcasts x per-row (see [Batcher (GM-simulated)](#batcher-gm-simulated)).
8. The host obtains the FFTS base address with `rtGetC2cCtrlAddr()` and launches `DistributedFfnGridMixedKernel` once with `gridRows * gridCols` blocks.
9. Inside each block, Cube and Vec branches exchange intermediate tiles through A2/A3 `TPipe` FIFOs. The kernel issues these C2V/V2C transfers as a directional `TMOV` (`TMOV(pipe, tile)` to produce, `TMOV(tile, pipe)` to consume); the underlying `TPUSH`/`TPOP` stays implicit (see `tpipe_tmov_inl.hpp`):

```text
Cube:
  X[row] @ W_gate[col] -> gatePartial[row,col] --TMOV C2V-->
  X[row] @ W_up[col]   -> upPartial[row,col]   --TMOV C2V-->

Vec:
  hidden[row,col] = fp16(SwiGLU(gatePartial) * upPartial)   # SiLU(clamp(gate)) * up
  hidden[row,col] --TMOV V2C-->

Cube:
  hidden[row,col] @ W_down[col] -> downPartial[row,col] --TMOV C2V-->

Vec:
  downPartial --GridPipe EAST reduce across cols--> yOutput[row] on final col
```

The cross-cell `EAST`/`WEST` reduce and gather keep their explicit GridPipe `TPUSH`/`TPOP`; only the in-block Cube↔Vec C2V/V2C traffic is folded into `TMOV`.

10. The host synchronizes the stream, checks GridPipe fault flags, copies `yOutput` back, and compares it with `golden.bin`.

## Key Designs

### Batcher (GM-simulated), SwiGLU, and the A3 precision map

This demo is aligned to the WSE-FFN tile-level expansion (`WSE-FFN-tile级全展开图.svg`), which casts an external **Batcher** as the owner of the full input and the full DRAM-resident weights, responsible for splitting/distributing them to cores and collecting the output. A2/A3 has no such hardware, so `batcher.hpp` simulates the Batcher entirely in GM:

- **Full weights resident in GM** (`w_gate_full`/`w_up_full` `[H,F]`, `w_down_full` `[F,H]`), mirroring the SVG's `DRAM 常驻` store.
- **Distribute** slices those full weights column-parallel and writes a contiguous per-col shard (`[H,Fi]` gate/up; `[F,Hc]` AllGather or `[Fi,H]` ReduceSum for down) into a per-col GM region. Each core then TLOADs its own shard (DRAM→L1 stream), exactly like a core streaming its Batcher-delivered weight tile.
- **Broadcast** writes the full `x` into GM; every column in a row reads the same `x` (broadcast, "复制 broadcast → N 核"). This also drops the legacy per-cell duplication: `x` is per-row, weights are per-col.
- **Collect**: cores write their y shards (AllGather) / the EAST reduce writes the per-row sum (ReduceSum) straight into the Batcher `y` region of GM.

The kernel addresses Batcher storage by `(row, col)`: `x = xFull + row*…`, `w = wShards + col*…`, `y = yFull + row*… (+ col*Hc)`.

The SVG activation is **SwiGLU = SiLU(clamp(gate)) · up** ("SiLU + clamp(max=10)"). The Vec branch composes SiLU from existing intrinsics in fp32: `SiLU(g) = g / (1 + e^-g)` via `TMAXS`/`TMINS` (clamp ±10), `TMULS(-1)` → `TEXP` → `TADDS(1)` (denominator), `TDIV`, then `TMUL` with `up`. `gen_data.py` uses the identical clamp+SiLU for the golden reference.

The SVG also carries low precisions A3 does not support (FP4 weights, FP8 activations, BF16 I/O). Per the extension design every tile-graph precision is mapped to **one A3-supported dtype** (see the table in `ffn_config.hpp`): FP4/FP8/BF16 → `half`, FP32 accumulators/output stay `float`. The `act_quant` and weight-`unpack` stages therefore exist as named, zero-cost identity points in the kernel — they document where the SVG casts would live without adding any A3-unsupported conversion. The fp16/fp32 data path already *is* the mapped result.

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

The host allocates `gridRows * gridCols` local SRAM windows, backed by GM in the mock. `TPUSH<EAST>` resolves the east neighbor's SRAM slot with the `ResolvePeerSlotAddr` runtime helper, writes the payload, then publishes the ready counter; `TPOP<EAST>` waits on the local ready counter, loads the local SRAM slot, and returns free credit to the west neighbor.

The mock uses GM flag polling and cache maintenance to emulate the intended LPU WSE `SPR` / `WFE` behavior on A2/A3.

### NoC write-only address-segment SRAM model (`GmSramArena`)

To stay close to real silicon, the mock models future-hardware per-core SRAM as an explicit **GM address-segment arena**. The single contiguous `gridRows*gridCols * FFN_GRID_WINDOW_BYTES` window buffer is cut into equal per-core segments, so segment `c` (== `windowsIn[c]`) *is* core `c`'s private SRAM:

```text
segment c = [base + c*winSize, base + (c+1)*winSize)   // base == windowsIn[0]
```

`GmSramArena` (in `include/pto/npu/a2a3/grid_intrinsic.hpp`) carries `{base, segBytes, numSegs}` plus the `SegmentOf` / `InSegment` classifiers; the demo builds it on-device from the fake `HcclDeviceContext` window table (`SramArenaFromCtx`). It is the single source of truth for "which core owns this address".

This makes the NoC contract explicit and **enforced**: the fabric can only *write* across cores, never *read*.

- `TPUSH<dir>` writes a payload into the **neighbor's** segment — a cross-segment write, exactly what the fabric does.
- `TPOP<dir>` may only drain **this core's own** segment. `GRID_TRY_TPOP_IMPL` calls the `PopSlotIsLocal` guard before the payload read; on a cross-segment read it raises `kFaultPopNonLocal` (`0x205`, "pop non-local segment") and aborts the pop. The host's `CheckGridPipeFaults` surfaces it.

On native hardware `PopSlotIsLocal` is a no-op (`true`): a TPOP read address is local by construction because the fabric has no remote-read path. The guard exists only because the A2/A3 mock backs SRAM with a GM window that *can* physically read any address, so without it a demo could silently rely on a remote read the silicon cannot perform. A compile-time `static_assert(GmSramArenaSelfCheck())` is built into every A2/A3 kernel, so a regression in the segment math fails the build rather than mis-routing a pop.

> The `pto::comm` variants (`TREDUCE` / `TGATHER`) intentionally do **not** follow this rule: they are a root-pulls-from-every-rank collective (HCCL/RDMA-style remote reads), a different memory model from the WSE NoC. Only the GridPipe `TPUSH`/`TPOP` path is constrained to write-only.

### IPC_SCB scoreboard intrinsic API

GridPipe ready/free synchronization follows the V8 IPC_SCB scoreboard route. The handshake intrinsics live in `include/pto/npu/a2a3/grid_cce_intrinsic.hpp` as a thin CCE facade layer — each facade forwards 1:1 to a `__builtin_cce_*` under `PTO_GRID_CCE_NATIVE`, and otherwise emulates the same semantics in the A2/A3 mock with a GM word + cache maintenance (`dcci`/`dsb`):

- `copy_ubuf_to_neighbor_ubuf(dstNeighborSlot, src, bytes)` (V8 `COPY_UBUF_TO_NBR`, G1 — the only new machine instruction / HW-DEP-0) writes a local UB payload into the resolved neighbor L1/SRAM slot. Not self-syncing; data-ready is announced by the following `sync_hscb(READY)`.
- `sync_hscb(peerScb, absCount)` (V8 `SYNC_HSCB`/`ST_HSCB`, G2 — reused HSCB store + neighbor IPC_SCB addressing / HW-DEP-1) stores this core's new monotone absolute count into the peer's `ready_scb`/`free_scb` IPC_SCB. The `(kind, dir, dist)` machine operands are resolved into `peerScb` by the caller's `RemoteScbPtr` runtime helper, so the facade operates on the resolved target.
- `wait_ipc_scb(localScb, threshold, slot)` (V8 `WAIT_SPR`, G3 — reused IPC_SCB blocking wait) reads + blocks in **one** instruction: the entry reads the local IPC_SCB and proceeds if it is already `>= threshold`, else blocks the current pipe until the peer's `sync_hscb` store raises it. V8 dropped the V7 `MOV_SPR2X` non-blocking peek — there is no separate read step. The demo calls the `wait_ipc_scb_sim(..., maxSpins)` mock wrapper, which adds a spin-timeout fault sentinel so a handshake deadlock fails the test instead of hanging; the documented hardware interface is the void `wait_ipc_scb`.

Payload address resolution (turning a local slot / scoreboard word into the same byte offset in a peer's GM window) is a plain runtime helper in the demo's `gridpipe_payload_inl.hpp` (`ResolvePeerSlotAddr` / `RemoteScbPtr`), not an intrinsic. TPOP's local drain reuses the existing local `copy_gm_to_ubuf` — the NoC is write-only, so there is deliberately **no cross-core read** of payload — guarded by the `GmSramArena` segment check `PopSlotIsLocal` so a mis-wired cross-segment read is rejected instead of silently serviced.

Native lowering targets the real CCE HSCB/IPC_SCB stack (`__sync_hscb`/`__st_hscb`; `__builtin_cce___wait_ipc_scb`, or the closest-real `__wait_ast_scb`, for the blocking wait — the header exposes no blocking `WAIT_SPR` on IPC_SCB yet). The current A2/A3 mock stands in for those IPC_SCB scoreboards with GM words + cache maintenance. Once native hardware provides neighbor-IPC_SCB addressing (V8 HW-DEP-1) and the `COPY_UBUF_TO_NBR` builtin (V8 HW-DEP-0), GridPipe call sites do not change — flip `PTO_GRID_CCE_NATIVE` on and the facades route to the real builtins.

`TPUSH<EAST>` waits the local `free_scb` with `wait_ipc_scb`, writes the payload slot, then publishes `prod_idx` to the downstream `ready_scb` with `sync_hscb`. `TPOP<EAST>` waits the local `ready_scb`, reads the payload slot, then publishes `cons_idx` to the upstream `free_scb`.

### fp32 EAST reduction

The reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This keeps `downPartial`, `yOutput`, and `golden.bin` in fp32 for direct tolerance-based comparison.

### Routed K-hop unicast

`TPUSH<dir, dist>` / `TPOP<dir, dist>` extend the nearest-neighbor pipe to a routed unicast `dist` hops along `dir` (Scheme A). The payload write resolves the slot in the `dist`-hop neighbor's segment, and the ready/free scoreboard stores are routed to the `dist`-hop peer's IPC_SCB (resolved via `RemoteScbPtr`) and issued through `sync_hscb`, so the receiver `dist` hops downstream pops the tile with no relays in between. `dist == 1` reproduces the original nearest-neighbor behavior. Fan-in stays 1 (one upstream per direction/distance), so no slot/flag expansion is needed.

### Concurrent group broadcast (TBROADCAST)

`TBROADCAST<GridGroup>` (`ROW` or `COL` as the first template argument) broadcasts a cell's tile to every other cell in its row (`ROW`) or column (`COL`) as one op: the per-target writes into each receiver's shared ring are batched with no inter-target fence, the whole broadcast pays a single publish fence, then per-source ready lanes fire. It is not lowered to a per-hop `TPUSH` loop.

Unlike the old single-source `TPUSH<GridSpan>` multicast (fan-in 1, which forbade concurrent senders), `TBROADCAST` is a 真·同时 MPSC channel (design doc `Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移`): every member of the group may call it at the same instant. Each source writes only its own prefix-offset slot in each receiver's shared ring and rings only its own per-source ready lane, so K concurrent senders never clobber a shared counter. This is what makes the AllGather-of-shards ("every AICORE broadcasts its own shard") correct. Receivers drain member `srcRank`'s shard with `TPOP<GridGroup>(pipe, tile, srcRank)` (ascending srcRank, so the directed free-notification chain advances with consumption). See `Grid_TPUSH_TBROADCAST_TREDUCE_接口设计说明.md` for the full handshake.

### GridPipe smoke tests

The two capabilities above each have a Vec-only data-movement smoke test under `smoke/` (no Cube, no matmul, no data files; in-process verification on the same GM-backed mock as the FFN demos):

- `khop_smoke` — a `1 x cols` row; each cell pushes a stamped fp32 `[T, W]` tile `DIST` hops east, the receiver pops and stores it; the host checks `out[c] == in[c-DIST]`.
- `bcast_smoke` — one source cell (`--src`) broadcasts its stamped tile to its whole row (or column with `--span-col 1`) via `TBROADCAST<GridGroup>`; every other cell drains it with `TPOP<GridGroup>(pipe, tile, src)` and stores it; the host checks `out[cell] == in[source]`. The default 1x5 row with the source at col 2 exercises receivers on both sides of the source in one run.

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
