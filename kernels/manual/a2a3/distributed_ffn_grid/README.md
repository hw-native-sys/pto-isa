# Single-device Multi-block FFN GridPipe Demo

## Goal

This demo validates the three distributed-FFN GridPipe collective interfaces — **TPUSH**, **TBROADCAST**, **TREDUCE** — on a single-device logical FFN grid on A2/A3. The host runs one process on the selected device and launches `gridRows * gridCols` blocks; each block owns one logical cell. There are **four examples**, one per (interface, FFN pattern), all on the same pure 1D N-cut 4×8 = 32-cell topology with the real DeepSeek-v4 Pro shapes (M=T=8, H=7168, I=3072):

| Example (run script / executable) | Interface verified | FFN pattern | Cross-cell collective |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | explicit `TPOP<Dir>` + `TADD` + `TPUSH<Dir>` (the A3 lowering of `TREDUCE`) |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | nearest-neighbor `TPUSH`/`TPOP` relay gather (fan-in-1 DAG) |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` MPSC group broadcast |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | fused `TREDUCE<GridGroup, Sum>` N→1 group fan-in (`reduce_group_to_ubuf`) |

Each example compares its `[T, H]` output with `golden.bin` using a `1e-3` tolerance. All four pass **bit-exact** on the NPU (`max diff = 0`, run with `-r npu`); see [Bit-exactness notes](#bit-exactness-notes).

The cross-cell collectives use the A2/A3 GridPipe mock backend: local SRAM windows backed by GM in the mock, fake `HcclDeviceContext` window pointers, ready/free counters, `dcci/dsb` fences, and spin waits. This validates the programming model and same-device mock path; it is not multi-card communication validation.

Beyond these FFN examples, GridPipe also supports routed K-hop unicast (`TPUSH<dir, dist>` / `TPOP<dir, dist>`) and concurrent group broadcast (`TBROADCAST<GridGroup>` / `TPOP<GridGroup>`). Each capability has a standalone Vec-only smoke test under `smoke/` (see [Smoke tests](#gridpipe-smoke-tests)).

## Files

| File | Purpose |
| --- | --- |
| `README.md` / `README_zh.md` | English / Chinese documentation. |
| `CMakeLists.txt` | Builds the four host executables and their mixed Cube/Vec device kernel shared libraries. |
| `run_treduce_reducesum.sh` / `run_tpush_reducesum.sh` | Set up CANN, generate data, configure CMake, build, and run the TREDUCE / TPUSH ReduceSum examples. |
| `run_tbroadcast_allgather.sh` / `run_tpush_allgather.sh` | Set up CANN, generate data, configure CMake, build, and run the TBROADCAST / TPUSH AllGather examples. |
| `ffn_config.hpp` | Compile-time grid shape, tile shape, GridPipe window sizes, buffer sizes, SwiGLU clamp bounds, the A3 precision-mapping table, and Batcher GM arena byte sizes. |
| `kernel_launch.hpp` | Host-side mixed kernel launch declarations (one per example). |
| `main_treduce_reducesum.cpp` / `main_tpush_reducesum.cpp` | ReduceSum host drivers: ACL setup, fake HCCL context / local GridPipe windows, working buffers, Batcher load/distribute, kernel launch, golden comparison, cleanup. |
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel: the EAST+SOUTH reduce uses the fused `TREDUCE<GridGroup, Sum>` group fan-in (`reduce_group_to_ubuf`) at the sink. |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel: same compute, but the EAST+SOUTH reduce is spelled with explicit `TPOP<Dir>` + `TADD` + `TPUSH<Dir>`. |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host drivers. |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel: the two gather phases use `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`. |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel: the two gather phases use a bidirectional `TPUSH`/`TPOP` relay. |
| `batcher.hpp` | Host-side GM-simulated **Batcher**: owns the full input + the full DRAM-resident weights in GM, splits them column-parallel into per-cell shards, broadcasts x, and exposes the output-collection region. |
| `tpipe_tmov_inl.hpp` | Directional `TMOV` overloads that lower Cube↔Vec C2V/V2C transfers to the existing `TPUSH`/`TPOP`, so the kernel body never spells out the handshake. |
| `gridpipe_payload_inl.hpp` | Local GridPipe payload hooks and fake-window remote pointer adapter — peer-slot / scoreboard-word resolution (`ResolvePeerSlotAddr`/`RemoteScbPtr`), the `copy_ubuf_to_neighbor_ubuf`/`copy_gm_to_ubuf` tile adapters (`CopyTileToNeighborSramSlot`/`CopyLocalSlotToTile`), the NoC read-locality guard (`PopSlotIsLocal`), and `TileUbPtr` (extract a tile's `__ubuf__` pointer for the G4/G5 group intrinsics, which take raw UB pointers rather than tile objects). |
| `smoke/` | Standalone GridPipe feature smoke tests. `khop_smoke_{config,kernel,launch}` + `main_khop_smoke.cpp` + `run_khop_smoke.sh` cover routed K-hop unicast; `bcast_smoke_*` + `run_bcast_smoke.sh` cover single-source row/column broadcast. Both build through the parent `CMakeLists.txt`. |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | V8 GridPipe CCE facade layer: `copy_ubuf_to_neighbor_ubuf` (G1 `COPY_UBUF_TO_NBR`), `sync_hscb` (G2 `SYNC_HSCB`/`ST_HSCB`), `wait_ipc_scb`/`wait_ipc_scb_sim` (G3 `WAIT_SPR`, read+block in one instruction, no `MOV_SPR2X` peek) — plus the group-semantics `bcast_ubuf_to_group` (G4 `BCAST_UBUF_TO_GROUP`, single-instruction 1→N fan-out) and `reduce_group_to_ubuf<Group,Op,T>` (G5 `REDUCE_GROUP_TO_UBUF`, N→1 element-wise fan-in). Each forwards 1:1 to a `__builtin_cce_*` under `PTO_GRID_CCE_NATIVE`, else emulates the same semantics — G1–G3 with a GM word + cache maintenance; G4–G5 with chunked UB/GM copies and, for G5 on the A3 mock, an in-core `vadd`/`vmax`/`vmin` combine over a per-member scratch. |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 data model + mock support: Section 1 is the mesh model + neighbor / K-hop / group resolvers; Section 2 is the GM-mock boundary-fault sentinels; Section 3 is the `GmSramArena` address-segment SRAM model + the TPOP read-locality guard. |
| `scripts/gen_data.py` | Generates the FULL fp16 X/weight tensors (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) the Batcher consumes, plus an fp32 SwiGLU `golden` reference. |
| `build/` | Ignored generated build directory. |
| `out/` | Ignored generated data directory. |

## Bit-exactness notes

Run with `-r npu` (the `sim`/`camodel` modes fail `aclrtSetDevice` 507033); on a shared host every run goes through `task-submit`. All four examples produce `max diff = 0` vs `golden.bin` — bit-exact, not merely within the `1e-3` tolerance. Two real bugs once masked that, both now fixed:

- **Cache-line doorbell stride (TBROADCAST MPSC).** `TBROADCAST<GridGroup>` is a true concurrent MPSC channel — every member may publish at the same instant, each ringing only its *own* per-source ready lane. Those lanes were packed as consecutive `u32`s: `GroupMax` lanes × 4 B ≤ 32 B < one 64 B cache line. Several producers therefore each wrote a different word of the *same* line while the consumer `dcci`-invalidated + read it each poll; the AICore store is line-granular, so one producer's write-back clobbered another's word and that doorbell silently **dropped from GM** (proven by a D2H lane dump that bypasses the consumer's `dcci`). Symptom: sporadic `wait ready timeout`. Fix: give each ready/free lane its **own cache line** — `kBcastLaneStride = 64` / `kBcastLaneStrideU32 = 16` in `grid_intrinsic.hpp`, mirrored as `FFN_NCUT_LANE_STRIDE = 64` in `ffn_config.hpp` — at all four lane-index sites in `GridTBroadcast.hpp`. The phase-B/C handshake dropped from 10–20 s (retry waves) to ~40 µs.
- **Phase-D output T-stride (both AllGather kernels).** The AllGather y-shard `[T, Hc]` is written into the *full* `[T, H]` output, so its row stride must be the full output width `kHfull` (= `H` = 7168). A copy-paste from the `hidden_full` store had left it at `kIfull` (= `I` = 3072), scrambling y rows 1–7 (≈50 % zero output / large drift). One-line fix `kIfull` → `kHfull` in the `GY` store of both AllGather kernels.

The `treduce` ReduceSum additionally requires its per-cell partial buffers (`partialBuf` / `rowPartialBuf`) to be laid out **segment-major** — each `[T, kHBase]` H-segment contiguous at offset `h*(T*kHBase)` — so the group fan-in reads every row-mate's segment as one contiguous byte range; only the final `yFull` keeps the strided `[T, H]` golden layout.

A 32-block launch still cannot run in one wave on 24 physical AICores — a single-wave launch oversubscription-deadlocks phase C, whose COL groups span all 4 rows (first-batch cells spin on second-batch row-3 doorbells that never get a core). The host therefore launches in waves sized from `--phys-cores` (`rowsPerWave = physCores/cols`, `colsPerWave = physCores/rows` → 2 waves each for phases B and C, 6 launches total, ~5 ms). With the stride fix the wave split is purely a scheduling concern, not a reliability problem.

## Execution Flow

1. Each `run_*.sh` parses arguments. Defaults are the real DeepSeek-v4 Pro shapes on a 4×8 = 32-cell mesh: `gridRows=4`, `gridCols=8`, `T=8` (token tile), `H=7168`, `Fi=96` (per-cell I shard; full `I = Fi * cells = 3072`), `n-ranks=1`, and `phys-cores=24`.
2. Unless `--build-only` is set, `scripts/gen_data.py --pure-ncut` generates the flat full-tensor Batcher inputs (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) plus the SwiGLU `golden.bin`.
3. CMake builds two targets per example — a `..._mixed_kernel` `dav-c220` shared library and the matching host executable (e.g. `distributed_ffn_grid_treduce_reducesum_mixed_kernel` + `distributed_ffn_grid_treduce_reducesum`). The two AllGather kernels are additionally compiled with `-DCONFIG_FFN_GRID_ALLGATHER`.
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

### Group broadcast & reduce intrinsics (G4 / G5)

On top of the three handshake facades (G1–G3), the same header exposes two group *data-movement* intrinsics, each a single-instruction form of a Tier-2 collective:

- `bcast_ubuf_to_group<Group>(groupSlotBase, src, bytes, memberCount, rect, memberStride)` (G4 `BCAST_UBUF_TO_GROUP`) copies one local UB tile into every member's resolved receive slot — a dtype-agnostic byte copy (`void*` + bytes), not self-syncing (the caller still fires `sync_hscb(READY)` doorbells). Under `PTO_GRID_CCE_NATIVE` it is one `__builtin_cce_bcast_ubuf_to_group`; on the A3 mock it is a chunked UB→GM-window pump.
- `reduce_group_to_ubuf<Group, Op, T>(dst, groupSlotBase, bytes, memberCount, rect, memberStride, combineScratch)` (G5 `REDUCE_GROUP_TO_UBUF`) folds every member's resolved contribution slot element-wise with `Op` (Sum/Max/Min) into local UB. It must know the element width, so it is templated on `T` and dispatches the native builtin by `sizeof(T)` (`_b16` for half/bfloat16, `_b32` for float). It folds members in **ascending** order (member 0 seeds `dst`), so an SPMD row/column fan-in reproduces the directional relay's left-to-right accumulation bit-for-bit (IEEE-754 add is commutative). On the A3 mock it pulls each member GM→UB and runs an in-core `vadd`/`vmax`/`vmin` over a per-member `combineScratch` (required on the mock, ignored on native/`__CPU_SIM`).

`GRID_TBROADCAST`'s per-member copy loop now **collapses to one** `bcast_ubuf_to_group` whenever the group's receive slots form a uniform-stride arena — ROW and COL always do (consecutive ranks → `memberStride = resolved slot₁ − slot₀`); a SUBRECT is uniform when it is a single row/column, otherwise the multi-row rectangle falls back to the per-member `copy_ubuf_to_neighbor_ubuf` loop. The `treduce` ReduceSum's EAST (row) and SOUTH (column) phases fan in at the sink (`col == gridCols-1` / `row == gridRows-1`) via `GRID_TREDUCE_GROUP_IMPL<ROW/COL, Sum, float>` — a genuine N→1 fan-in, a different collective *shape* from the directional `TREDUCE<Dir>` relay that the `tpush` ReduceSum example spells out as `TPOP<Dir>` + `TADD` + `TPUSH<Dir>`. Both Tier-2 facades are tile-agnostic, so a new payload hook `TileUbPtr<T>` (in `gridpipe_payload_inl.hpp`) extracts the tile's `__ubuf__` pointer to hand to the intrinsics.

### fp32 reduction

The reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This keeps `downPartial`, `yOutput`, and `golden.bin` in fp32 for direct tolerance-based comparison. The ReduceSum reduce is H-chunked (`FFN_RS_REDUCE_SLOT_COUNT = kHSegs` = 7, one slot per H-segment): the `treduce` example folds the segment-h partials with `GRID_TREDUCE_GROUP_IMPL` at the sink, while the `tpush` example relays them hop-by-hop with `TPOP<EAST/SOUTH>` + `TADD` + `TPUSH<EAST/SOUTH>`. In the relay form the slot count must equal `kHSegs` — reusing slots across segments deadlocks on the cross-segment *free* doorbell.

### Routed K-hop unicast

`TPUSH<dir, dist>` / `TPOP<dir, dist>` extend the nearest-neighbor pipe to a routed unicast `dist` hops along `dir` (Scheme A). The payload write resolves the slot in the `dist`-hop neighbor's segment, and the ready/free scoreboard stores are routed to the `dist`-hop peer's IPC_SCB (resolved via `RemoteScbPtr`) and issued through `sync_hscb`, so the receiver `dist` hops downstream pops the tile with no relays in between. `dist == 1` reproduces the original nearest-neighbor behavior. Fan-in stays 1 (one upstream per direction/distance), so no slot/flag expansion is needed.

### Concurrent group broadcast (TBROADCAST)

`TBROADCAST<GridGroup>` (`ROW` or `COL` as the first template argument) broadcasts a cell's tile to every other cell in its row (`ROW`) or column (`COL`) as one op: the per-target writes into each receiver's shared ring are batched with no inter-target fence, the whole broadcast pays a single publish fence, then per-source ready lanes fire. It is not lowered to a per-hop `TPUSH` loop.

Unlike the old single-source `TPUSH<GridSpan>` multicast (fan-in 1, which forbade concurrent senders), `TBROADCAST` is a 真·同时 MPSC channel (design doc `Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移`): every member of the group may call it at the same instant. Each source writes only its own prefix-offset slot in each receiver's shared ring and rings only its own per-source ready lane, so K concurrent senders never clobber a shared counter. This is what makes the AllGather-of-shards ("every AICORE broadcasts its own shard") correct. Receivers drain member `srcRank`'s shard with `TPOP<GridGroup>(pipe, tile, srcRank)` (ascending srcRank, so the directed free-notification chain advances with consumption). The per-member payload fan-out itself collapses to a single `bcast_ubuf_to_group` intrinsic when the group is a uniform-stride arena (see [Group broadcast & reduce intrinsics](#group-broadcast--reduce-intrinsics-g4--g5)). See `Grid_TPUSH_TBROADCAST_TREDUCE_接口设计说明.md` for the full handshake.

### GridPipe smoke tests

The two capabilities above each have a Vec-only data-movement smoke test under `smoke/` (no Cube, no matmul, no data files; in-process verification on the same GM-backed mock as the FFN demos):

- `khop_smoke` — a `1 x cols` row; each cell pushes a stamped fp32 `[T, W]` tile `DIST` hops east, the receiver pops and stores it; the host checks `out[c] == in[c-DIST]`.
- `bcast_smoke` — one source cell (`--src`) broadcasts its stamped tile to its whole row (or column with `--span-col 1`) via `TBROADCAST<GridGroup>`; every other cell drains it with `TPOP<GridGroup>(pipe, tile, src)` and stores it; the host checks `out[cell] == in[source]`. The default 1x5 row with the source at col 2 exercises receivers on both sides of the source in one run.

### AllGather variant

The two AllGather examples (`run_tbroadcast_allgather.sh`, `run_tpush_allgather.sh`) share the same pure-N-cut data (`scripts/gen_data.py --pure-ncut`) as the ReduceSum examples; the kernel is compiled with `-DCONFIG_FFN_GRID_ALLGATHER`, which makes the host Batcher slice `W_down` along the output **H** (each cell gets an `[I_full, Hc]` shard, `Hc = H / cells`) and turns the cross-cell work into a two-phase gather that rebuilds the full fp16 `hidden [T, I_full]` on every cell before the down GEMM — each cell then writes one `Y[:, Hc]` output shard, so there is no post-down ReduceSum. The host stitches the shards and compares with `golden.bin`. Pure-N-cut requires `--model-tile` (H) divisible by the cell count (`grid-rows * grid-cols`) so `Hc` is an integer width, and full `I` (`ffn-tile * cells`) divisible by the cell count.

## How to Run

### Build only

```bash
bash run_treduce_reducesum.sh    --build-only
bash run_tbroadcast_allgather.sh --build-only
```

### Run on NPU

The scripts default to the DeepSeek-v4 Pro shapes (4×8 = 32 cells), so a plain invocation runs the real shape:

```bash
bash run_treduce_reducesum.sh    -r npu -v Ascend910B1 --device-id 0
bash run_tpush_reducesum.sh      -r npu -v Ascend910B1 --device-id 0
bash run_tbroadcast_allgather.sh -r npu -v Ascend910B1 --device-id 0
bash run_tpush_allgather.sh      -r npu -v Ascend910B1 --device-id 0
```

On a shared host, wrap each run in `task-submit` (e.g. `task-submit bash run_treduce_reducesum.sh -r npu --device-id 0`).

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
-r, --run-mode      sim or npu, default npu (sim/camodel fail aclrtSetDevice 507033)
-v, --soc-version   default Ascend910B1
-n, --n-ranks       fixed to 1
-d, --device-id     selected ACL device id; defaults to TASK_DEVICE, FFN_GRID_DEVICE_ID, ASCEND_DEVICE_ID, DEVICE_ID, then 0
--grid-rows         logical grid row count, default 4
--grid-cols         logical grid column count, default 8
--token-tile        token tile T (M) per cell, default 8
--model-tile        hidden dim H, default 7168; pure-N-cut requires H % (grid-rows*grid-cols) == 0
--ffn-tile          per-cell intermediate dim I_shard, default 96 (full I = ffn-tile*cells = 3072; must divide evenly by cells)
--phys-cores        physical AICores to emulate on, default 24 (waves are sized from this; <32 forces a multi-wave launch)
--build-only        build only; skip data generation and execution
```

The smoke scripts reuse `-r/-v/-d`, `--grid-rows/--grid-cols`, `--token-tile/--model-tile` (tile `[T, W]`), and `--build-only`; `run_khop_smoke.sh` adds `--dist` (hop count, default 2). `run_bcast_smoke.sh` adds `--src` (source index, default 2), `--span-col` (1 = column group, default 0 = row group), and `--subrect` (1 = sub-rectangle group, default 0) with `--rect-r0/r1/c0/c1` / `--rect-src` to scope the rectangle.

## Expected Result

On success, each FFN executable prints its bit-exact verdict:

```text
[SUCCESS] 32-cell N-cut FFN GridPipe TREDUCE ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TBROADCAST AllGather PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH AllGather PASS.
```

The smoke tests print:

```text
[SUCCESS] GridPipe K-hop unicast smoke PASS.
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```
