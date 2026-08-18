# Single-device Multi-block FFN GridPipe Demo

## Goal

This demo validates the three distributed-FFN GridPipe collective interfaces — **TPUSH**, **TBROADCAST**, **TREDUCE** — on a single-device logical FFN grid on A2/A3. The host runs one process on the selected device and launches `gridRows * gridCols` blocks; each block owns one logical cell. There are **four examples**, one per (interface, FFN pattern), all on the same pure 1D N-cut 4×8 = 32-cell topology with the real DeepSeek-v4 Pro shapes (M=T=8, H=7168, I=3072):

| Example (run script / executable) | Interface verified | FFN pattern | Cross-cell collective |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | explicit `TPOP<Dir>` + `TADD` + `TPUSH<Dir>` (the A3 lowering of `TREDUCE`) |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | nearest-neighbor `TPUSH`/`TPOP` relay gather (fan-in-1 DAG) |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` serialized group broadcast |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | fused `TREDUCE<GridGroup, Sum>` N→1 group fan-in (`mov_ubuf_group`, op=SUM) |

Each example compares its `[T, H]` output with `golden.bin` using a `1e-3` tolerance. All four pass **bit-exact** on the NPU (`max diff = 0`, run with `-r npu`); see [Bit-exactness notes](#bit-exactness-notes).

The cross-cell collectives use the A2/A3 GridPipe mock backend: local SRAM windows backed by GM in the mock, fake `HcclDeviceContext` window pointers, ready/free counters, `dcci/dsb` fences, and spin waits. This validates the programming model and same-device mock path; it is not multi-card communication validation.

Beyond these FFN examples, GridPipe also supports group broadcast (`TBROADCAST<GridGroup>` / `TPOP<GridGroup>`), which has a standalone Vec-only smoke test under `smoke/` (see [Smoke tests](#gridpipe-smoke-tests)).

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
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel: the EAST+SOUTH reduce uses the fused `TREDUCE<GridGroup, Sum>` group fan-in (`mov_ubuf_group`, op=SUM) at the sink. |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel: same compute, but the EAST+SOUTH reduce is spelled with explicit `TPOP<Dir>` + `TADD` + `TPUSH<Dir>`. |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host drivers. |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel: the two gather phases use `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`. |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel: the two gather phases use a bidirectional `TPUSH`/`TPOP` relay. |
| `batcher.hpp` | Host-side GM-simulated **Batcher**: owns the full input + the full DRAM-resident weights in GM, splits them column-parallel into per-cell shards, broadcasts x, and exposes the output-collection region. |
| `tpipe_tmov_inl.hpp` | Directional `TMOV` overloads that lower Cube↔Vec C2V/V2C transfers to the existing `TPUSH`/`TPOP`, so the kernel body never spells out the handshake. |
| `gridpipe_payload_inl.hpp` | Local GridPipe payload hooks and fake-window remote pointer adapter — peer-slot / scoreboard-word resolution (`ResolvePeerSlotAddr`/`RemoteScbPtr`), the `copy_ubuf_to_neighbor_ubuf`/`copy_gm_to_ubuf` tile adapters (`CopyTileToNeighborSramSlot`/`CopyLocalSlotToTile`), the NoC read-locality guard (`PopSlotIsLocal`), and `TileUbPtr` (extract a tile's `__ubuf__` pointer for the G4 group intrinsic `mov_ubuf_group`, which takes raw UB pointers rather than tile objects). |
| `smoke/` | Standalone GridPipe feature smoke tests. `bcast_smoke_*` + `run_bcast_smoke.sh` cover single-source and all-source (`--all-src 1`) row and column broadcast. They build through the parent `CMakeLists.txt`. |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | V8 GridPipe CCE facade layer: `copy_ubuf_to_neighbor_ubuf` (G1 `COPY_UBUF_TO_NBR`), `sync_hscb` (G2 `SYNC_HSCB`/`ST_HSCB`), `wait_ipc_scb`/`wait_ipc_scb_sim` (G3 `WAIT_SPR`, read+block in one instruction, no `MOV_SPR2X` peek) — plus the unified group-collective `mov_ubuf_group` (G4 `MOV_UBUF_GROUP`, template-free, all runtime parameters). A broadcast and a reduce are *the same action* from the issuing core's view — move a UB tile to/from the resolved group arena — so the NoC collective mode is a runtime `GridCollOp` operand (`COPY` = 1→N replicate fan-out, `SUM`/`MAX`/`MIN` = N→1 element-wise fan-in; datapath direction implied by `op`), not a second instruction. Each forwards 1:1 to a `__builtin_cce_*` under `PTO_GRID_CCE_NATIVE`, else emulates the same semantics — G1–G3 with a GM word + cache maintenance; G4 with chunked UB/GM copies and, for a reduce (`op != COPY`) on the A3 mock, an in-core `vadd`/`vmax`/`vmin` combine over a per-member scratch. (This collapses the former `bcast_ubuf_to_group` G4 `BCAST_UBUF_TO_GROUP` + `reduce_group_to_ubuf<Group,Op,T>` G5 `REDUCE_GROUP_TO_UBUF` — two machine instructions and two template facades — into one; the new-instruction count drops from "+2 +reuse 2" to "+1 +reuse 2". Design doc: `2026-07-24-bcast-reduce-合并mov_ubuf_group方案.md`.) |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 data model + mock support: Section 1 is the mesh model + nearest-neighbor / group resolvers; Section 2 is the GM-mock boundary-fault sentinels; Section 3 is the `GmSramArena` address-segment SRAM model + the TPOP read-locality guard. |
| `scripts/gen_data.py` | Generates the FULL fp16 X/weight tensors (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) the Batcher consumes, plus an fp32 SwiGLU `golden` reference. |
| `build/` | Ignored generated build directory. |
| `out/` | Ignored generated data directory. |

## Bit-exactness notes

Run with `-r npu` (the `sim`/`camodel` modes fail `aclrtSetDevice` 507033); on a shared host every run goes through `task-submit`. All four examples produce `max diff = 0` vs `golden.bin` — bit-exact, not merely within the `1e-3` tolerance. Two real bugs once masked that, both now fixed:

- **Cache-line doorbell stride.** Doorbell words were packed as consecutive `u32`s, several to one 64 B line. Two different cores writing two words of the *same* line lose each other's updates: the AICore store is line-granular, so one core's write-back puts back its own (stale) copy of the neighbour's word and that doorbell silently **dropped from GM** (proven by a D2H dump that bypasses the consumer's `dcci`). Symptom: sporadic `wait ready timeout`. Fix: every independently-written scoreboard gets its **own cache line** — `kScbLineStride = 64` / `kScbLineStrideU32 = 16` in `grid_intrinsic.hpp`, mirrored as `FFN_NCUT_SCB_LINE_STRIDE = 64` in `ffn_config.hpp`. The phase-B/C handshake dropped from 10–20 s (retry waves) to ~40 µs. First hit on TBROADCAST's per-source lanes (since retired, see below); the rule outlives them and now covers all eight scoreboards in a window.
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

### PTO instruction surface

One `GridPipe` holds the FIFO state of **every** channel a cell uses — the rings per *direction* (`slotBase[dir]`, `prodIndex[dir]`, `consIndex[dir]`, `pushWindow[dir]`, `popWindow[dir]`, 5 entries) and the scoreboards per mesh *edge* (`readyScb[edge]`, `freeScb[edge]`, 4 entries) — and it **binds no peer identity**. The group collectives own no state of their own: they ride the same directional rings and the same edge scoreboards, so there is no broadcast ring and no group round counter.

```cpp
GridPipe<Tile, SlotStride, SlotCount /*, DirMask = kGridDirAll */>

TPUSH<Dir>(pipe, tile)              TPOP<Dir>(pipe, tile)
TREDUCE<Dir, Op>(pipe, acc, recv)
TBROADCAST<Group>(pipe, tile)       TPOP<Group>(pipe, tile, srcBlockId)
TBWAIT<Group>(pipe)                 TBNOTIFY<Group>(pipe, dstBlockId)
TREDUCE<Group, Op, T>(pipe, acc, scratch, base, bytes, memberCount, sinkBlockId, memberStride)
```

Every peer operand in that surface is a **logical block id** — the integer `get_block_idx()` returns, `row * gridCols + col` — and never a position within the group: `srcBlockId` names the broadcasting core, `dstBlockId` the core the publish turn is handed to, `sinkBlockId` the core that collects a fan-in. There is no device-rank notion here, only blocks of one launch. `GroupMemberBlockId(Group, coord, shape, indexInGroup)` is the conversion when a kernel wants to walk a group by position (`IndexInGroup` is the inverse), and a block outside the group traps as `0x405 kFaultGroupBadPeer` on every member instead of resolving to a stranger.

Every one of those notifies through an **IPC_SCB scoreboard pair** — `ready_scb` (a monotone count the peer stores, the local `WAIT_SPR` tests) and `free_scb` (the credit going the other way). There are **eight scoreboards and no more**, one pair per mesh edge — `GridDirection` has five enumerators but only four are edges; `SOURCE` is a pseudo-direction for GM/host/runtime injection and owns no scoreboard (a `TPOP<SOURCE>` resolves its pair to null, so both its ready wait and its free store are gated by the runtime out-of-band). Index the rings with `GridDirectionIndex` and the scoreboards with `GridEdgeIndex`: the group collectives add none of their own and instead borrow the directional pair, attributing each (producer, consumer) edge to a direction by the **dominant axis** of the coordinate delta — `|dCol| > |dRow|` → `EAST`/`WEST` by which side the consumer is on, otherwise (ties included) `NORTH`/`SOUTH` — via `GroupFlowDirection`. The direction names the direction of *flow*, so a source whose receiver lies east writes that receiver's `ready_scb[EAST]` and the receiver waits on its own `ready_scb[EAST]`, exactly the `TPUSH<EAST>`/`TPOP<EAST>` contract, just several hops and possibly off-axis. Nothing uses a statically partitioned per-rank lane array or a group-private scoreboard, which is why the group collectives serialize their publishers instead of running them concurrently (see below).

The group collectives notify with the **increment form** of the store, `sync_hscb_add(peerScb, 1)`, not the absolute `sync_hscb(peerScb, count)` that `TPUSH`/`TPOP` use (V8 §2b). That is what makes a scoreboard with *several* writers-over-time expressible at all. An absolute count forces every writer to agree on the shared sequence, so it has to encode the participant set into the value — and that breaks the moment the participant set is not the whole group (a single-source broadcast computes a different value than its one receiver waits for). Increments commute and carry no assumption about who else writes: a source adds 1 to each receiver's `ready_scb`, a receiver adds 1 to the source's `free_scb`, and each side's threshold is plain `TPUSH`/`TPOP` arithmetic. It also retires the old "successor's doorbell **last**" discipline: HSCB ordering holds only between the *same* (producer, consumer) pair, so a core's stores to two different peers are mutually unordered — with absolute counts a straggling store could drive a shared count *backwards*, but an add cannot, whatever order the stores land in.

> **HW-DEP.** An add-form HSCB is a hardware ask, and the increment must be **atomic at the scoreboard**: unlike the overwrite store, its writers genuinely overlap — the K receivers of one broadcast all credit the same producer `free_scb` at the same instant. The A2/A3 mock therefore emulates it with the smallest atomic-accumulate DMA the backend has (`set_atomic_s32` + `set_atomic_add` + a 4-byte UB→GM burst, whose addend lives in a reserved word at the top of UB), not with a read-modify-write — an RMW drops credits (observed: a free count of 9 where 10 adds were issued). The mock also collapses the increment to sub-block 0, because one cell is one block but a mix-mode block runs the vector program on **both** of its AIV sub-blocks: the absolute store is idempotent under that duplication, an increment is not.

Because the group borrows the directional scoreboards, **a group collective and a unicast channel that resolve to the same direction on the same pipe corrupt each other's counts** — give them separate pipes. The demos' collective pipes are `DirMask = kGridDirNone`, so nothing else touches their scoreboards; the scoreboards themselves are wired regardless of `DirMask`, which is what makes a ring-less pipe usable.

`Dir` names an **edge of the mesh, not a core**: the target of a `TPUSH<Dir>` is the adjacent cell along `Dir` and the producer a `TPOP<Dir>` drains is the adjacent cell along `-Dir`, both resolved at the call site from `(Dir, coord, shape)`. There is **no hop-count argument** anywhere in the family — every grid transfer is exactly one hop, and a longer reach is a relay, one `TPUSH` per edge, so each edge keeps its own credit and back-pressure. Because no peer is baked into the type, the same pipe keeps serving the same direction across phases even when the core on the other end changes.

All four demos and both smokes call **that PTO instruction layer**, not the A2/A3 backend's `GRID_*_IMPL` / `GRID_TRY_*_IMPL` entry points, so one run exercises the instruction surface itself (overload selection, the `SOURCE` `static_assert`, the target-profile guard) and not just the lowering. The cost is that the instructions carry no spin bound — they block like the hardware `WAIT_SPR` — so a mis-wired handshake shows up as a hang rather than a `kFaultWaitReadyTimeout` sentinel. That is safe here because the host waves each relay group / collective group wholly into one launch (see `LaunchWave` in the `main_*.cpp`). To get the timeout sentinel back while debugging, call the matching `GRID_TRY_*_IMPL(..., maxSpins)` directly.

`TREDUCE` has two overloads for two different reduction *shapes*: `TREDUCE<Dir, Op>(pipe, acc, recv)` is the hop-by-hop relay (first template argument a `GridDirection`), and `TREDUCE<Group, Op, T>(pipe, acc, scratch, base, bytes, memberCount, sinkBlockId, memberStride)` is the N→1 group fan-in (first template argument a `GridGroup`, lowering to one `mov_ubuf_group`). The fan-in reads the contribution arena directly, so it uses no ring — but it does take the pipe, because the handshake that tells the sink "all N contributions have landed" and the credit that tells the contributors "the sink is done" are the pipe's edge scoreboards. They cannot be confused: each one's leading template argument fails substitution into the other.

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
- `sync_hscb(peerScb, absCount)` (V8 `SYNC_HSCB`/`ST_HSCB`, G2 — reused HSCB store + neighbor IPC_SCB addressing / HW-DEP-1) stores this core's new monotone absolute count into the peer's `ready_scb`/`free_scb` IPC_SCB. The `(kind, dir)` machine operands are resolved into `peerScb` by the caller's `RemoteScbPtr` runtime helper, so the facade operates on the resolved target.
- `wait_ipc_scb(localScb, threshold, slot)` (V8 `WAIT_SPR`, G3 — reused IPC_SCB blocking wait) reads + blocks in **one** instruction: the entry reads the local IPC_SCB and proceeds if it is already `>= threshold`, else blocks the current pipe until the peer's `sync_hscb` store raises it. V8 dropped the V7 `MOV_SPR2X` non-blocking peek — there is no separate read step. GridPipe's handshake sequences go through the `wait_ipc_scb_sim(..., maxSpins)` mock wrapper, which adds a spin-timeout fault sentinel (when `maxSpins > 0`) so a handshake deadlock fails the test instead of hanging. The PTO instructions lower with `maxSpins = 0` — block-forever, like the hardware `WAIT_SPR` — so the timeout sentinel is only reachable by calling `GRID_TRY_*_IMPL` directly. The documented hardware interface is the void `wait_ipc_scb`.

Payload address resolution (turning a local slot / scoreboard word into the same byte offset in a peer's GM window) is a plain runtime helper in the demo's `gridpipe_payload_inl.hpp` (`ResolvePeerSlotAddr` / `RemoteScbPtr`), not an intrinsic. TPOP's local drain reuses the existing local `copy_gm_to_ubuf` — the NoC is write-only, so there is deliberately **no cross-core read** of payload — guarded by the `GmSramArena` segment check `PopSlotIsLocal` so a mis-wired cross-segment read is rejected instead of silently serviced.

Native lowering targets the real CCE HSCB/IPC_SCB stack (`__sync_hscb`/`__st_hscb`; `__builtin_cce___wait_ipc_scb`, or the closest-real `__wait_ast_scb`, for the blocking wait — the header exposes no blocking `WAIT_SPR` on IPC_SCB yet). The current A2/A3 mock stands in for those IPC_SCB scoreboards with GM words + cache maintenance. Once native hardware provides neighbor-IPC_SCB addressing (V8 HW-DEP-1) and the `COPY_UBUF_TO_NBR` builtin (V8 HW-DEP-0), GridPipe call sites do not change — flip `PTO_GRID_CCE_NATIVE` on and the facades route to the real builtins.

`TPUSH<EAST>` waits the local `free_scb` with `wait_ipc_scb`, writes the payload slot, then publishes `prod_idx` to the downstream `ready_scb` with `sync_hscb`. `TPOP<EAST>` waits the local `ready_scb`, reads the payload slot, then publishes `cons_idx` to the upstream `free_scb`.

### Group broadcast & reduce intrinsic (G4 / MOV_UBUF_GROUP)

On top of the three handshake facades (G1–G3), the same header exposes one group *data-movement* intrinsic, `mov_ubuf_group` — the unified single-instruction form of *both* a Tier-2 broadcast and a Tier-2 reduce. From the issuing core's perspective the two are the same action — move a `bytes`-sized UB tile to/from the resolved per-member group arena — so what distinguishes them is the NoC collective mode, a **runtime** operand rather than a second instruction. The 2026-07-23 design lowered these as a `bcast_ubuf_to_group<Group>` (G4 `BCAST_UBUF_TO_GROUP`) + `reduce_group_to_ubuf<Group,Op,T>` (G5 `REDUCE_GROUP_TO_UBUF`) pair; the 2026-07-24 merge folds both into one machine instruction and one template-free facade (`mov_ubuf_group`, the new-instruction count drops from "+2 +reuse 2" to "+1 +reuse 2"; design doc `2026-07-24-bcast-reduce-合并mov_ubuf_group方案.md`):

- `mov_ubuf_group(ubTile, groupSlotBase, bytes, memberCount, memberStride, op, eltype, rect, combineScratch, groupDesc)` (G4 `MOV_UBUF_GROUP`, **template-free** — the `Group`/`Op`/`T` template parameters are gone; `Group` is resolved by the Tier-2 caller into `groupSlotBase`/`memberCount` or `groupDesc`). `op` is a runtime `GridCollOp` that selects the NoC collective mode **and** implies the datapath direction:
  - `op == COPY` — broadcast: this core is the **source**; its UB tile is replicated once into every member's slot (1→N fan-out, push). It is a byte-level pure copy (it does not read element values), so `eltype` is ignored — exactly the former `bcast_ubuf_to_group`.
  - `op == SUM/MAX/MIN` — reduce: this core is the **sink**; it reads every member's contribution slot and folds them element-wise into UB (N→1 fan-in, pull). The combine must know the element width, so `eltype` (1/2/4 bytes) selects the combine granularity — exactly the former `reduce_group_to_ubuf`, with the template `T` replaced by the runtime `eltype` = `sizeof(T)` dispatch.
  `GridCollOp` values are deliberately `comm::ReduceOp{Sum,Max,Min}` + 1, so the Tier-2 reduce caller maps with `static_cast<GridCollOp>(uint32_t(commOp) + 1)` (`COPY=0` stays reserved for broadcast). Not self-syncing: data-ready is still announced by the caller's `sync_hscb(READY)` after the publish fence. It folds members in **ascending** order (member 0 seeds `ubTile`), so an SPMD row/column fan-in reproduces the directional relay's left-to-right accumulation bit-for-bit (IEEE-754 add is commutative). On the A3 mock a reduce pulls each member GM→UB and runs an in-core `vadd`/`vmax`/`vmin` over a per-member `combineScratch` (required on the mock, ignored for `COPY` and on native/`__CPU_SIM`). The three-state body is byte-identical to the former two facades — only the template `T`/`Op` became the runtime `eltype`/`op` switch — so behavior is bit-exact by construction.

`GRID_TBROADCAST`'s payload fan-out is **one** `mov_ubuf_group(..., op=COPY, eltype=1)` per window row: a group is a whole row or a whole column, whose members occupy consecutive grid ranks, so the receive slots are always a uniform-stride arena (`memberStride = resolved slot₁ − slot₀`). There is no per-member `copy_ubuf_to_neighbor_ubuf` fallback — that existed only for a multi-row sub-rectangle group, which no longer exists. The `treduce` ReduceSum's EAST (row) and SOUTH (column) phases fan in at the sink (`col == gridCols-1` / `row == gridRows-1`) via `TREDUCE<ROW/COL, Sum, float>`, which lowers to `mov_ubuf_group(..., op=SUM, eltype=sizeof(float))` — a genuine N→1 fan-in, a different collective *shape* from the directional `TREDUCE<Dir, Op>` relay that the `tpush` ReduceSum example spells out as `TPOP<Dir>` + `TADD` + `TPUSH<Dir>`. **Every** group member calls it, not just the sink, and each picks its role by comparing its own position with the `sinkBlockId` operand: the block that value names collects, the rest are contributors whose half is the handshake alone (their data is already in the arena). The sink is a runtime operand rather than the old "last member" convention because where the result lands is a placement decision of the **caller** — it is wherever the value is next needed, which is not generally the end of the row — and the collecting core is by construction the one that issues the gather. Every member must pass the same value; a block outside the group traps as `0x405 kFaultGroupBadPeer`. Contributor → sink is a fan-in, and with the increment doorbell it needs no turn order at all: each contributor adds 1 to the sink's `ready_scb` and the sink waits for the *total*, `cons_idx + contributorCount`. Each contributor's pair is indexed by *its own* edge direction to the sink, so an **interior** sink has contributors on both sides and the fan-in lands on **two** scoreboards — it waits each side out separately, skipping an empty one, the same back/forward split `TBROADCAST` has. Each side keeps its own count, so the two never interfere; with the sink at the end of the group one side is empty and this degenerates to a single wait. It then credits each contributor back with one add on that contributor's `free_scb`, which is the back-pressure that lets the next H-segment's round start. Its pipe carries no ring at all (`DirMask = kGridDirNone`) — the window is just the flag header — and phases B and C get one sub-window each so neither inherits the other's counts. The Tier-2 facades (`GRID_TBROADCAST` / `GRID_TREDUCE_GROUP_IMPL`) keep their `<Group[,Op,T]>` templates — that structural info lives in the PTO layer, not the CCE instruction layer — but they are tile-agnostic, so the payload hook `TileUbPtr<T>` (in `gridpipe_payload_inl.hpp`) extracts the tile's `__ubuf__` pointer to hand to the intrinsic.

### fp32 reduction

The reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This keeps `downPartial`, `yOutput`, and `golden.bin` in fp32 for direct tolerance-based comparison. The ReduceSum reduce is H-chunked (`FFN_RS_REDUCE_SLOT_COUNT = kHSegs` = 7, one slot per H-segment): the `treduce` example folds the segment-h partials with `TREDUCE<Group, Sum, float>` at the sink, while the `tpush` example relays them hop-by-hop with `TPOP<EAST/SOUTH>` + `TADD` + `TPUSH<EAST/SOUTH>`. In the relay form the slot count must equal `kHSegs` — reusing slots across segments deadlocks on the cross-segment *free* doorbell.

### Serialized group broadcast (TBROADCAST)

`TBROADCAST<GridGroup>` (`ROW` or `COL` as the first template argument) broadcasts a cell's tile to every other cell in its row (`ROW`) or column (`COL`) as one op: the per-target writes into each receiver's shared ring are batched with no inter-target fence, the whole broadcast pays a single publish fence, then the ready doorbells fire. It is not lowered to a per-hop `TPUSH` loop.

There is **no group-private addressing anywhere in it**. A source lands in each receiver's ordinary directional ring at `prod_idx % SlotCount`, exactly where `TPUSH<dir>` would land, and rings that receiver's `ready_scb[dir]` — one of the same eight scoreboards `TPUSH` uses — with `dir` attributed per (source, receiver) edge as above. `SlotCount` is `TPUSH`'s `SlotCount`; there is no separate `BcastSlotCount`, no per-rank prefix offset and no per-source lane. `TPOP<GridGroup>(pipe, tile, srcBlockId)` is likewise plain `TPOP` arithmetic: `srcBlockId` is used for exactly two things — *which direction* the source's edge takes, and the address the retire credit goes back to — while the wait threshold and the slot are the receiver's own `cons_idx`, so the receiving side needs to know neither the schedule nor the participant set.

That works because every writer only ever **adds 1**, and it costs one obligation in return: the caller owes the collective an **SPSC schedule** — at any instant exactly one member of a group is inside `TBROADCAST`, and its tile is drained by every receiver before the next member publishes. A single-source broadcast (one member sends, the rest only `TPOP`, as in the smoke test) satisfies it trivially. Calling `TBROADCAST` from several members at once is a caller error, not a supported mode, and the receive half traps a violation it can see locally — a `ready_scb` that has run *past* `cons_idx + 1` means a later source published before an earlier one was drained, reported as fault `0x404 kFaultGroupOutOfOrder` instead of silently draining the wrong tile.

**Consumption is `TPOP` and nothing else, and `TBROADCAST` does not return until every receiver has done it.** After the doorbells it blocks on `rounds * peerCount` credits per side — one `+1` from each receiver at the end of each `TPOP`. Note the threshold carries **no `SlotCount` term**: this is not the pipelined `prod - SlotCount + 1` credit test a unicast `TPUSH` uses, because the group contract is stricter than slot reuse — *no* next broadcast may start until the previous one is consumed everywhere. The source therefore gives up cross-round pipelining; that is the price of one shared ring. (It is also what an absolute store could not express at all: the credit comes from *every* receiver back to the one source, and a minimum across K writers is not something a single `WAIT_SPR` can test — a counting total is.)

What the primitive cannot do on its own is tell the **next** source, which has no counter that moves when someone else's tile is drained and cannot read a peer's state. So that one fact travels one hop — but with no packet, no second pipe and nothing extra for the receivers to drain: once `TBROADCAST`'s drain wait has ended, **`TBNOTIFY<Group>(pipe, dstBlockId)`** adds `1` to the **turn scoreboard** of the block it names (one `sync_hscb_add`, 4 B, no payload), where that member is blocked in **`TBWAIT<Group>`**.

Notifying is a **separate instruction, not the tail of the publish**, because the two say different things. `TBROADCAST` establishes a *fact* about this source's tile — every receiver has drained it — while *who publishes next* is a **schedule**, and the schedule is the caller's: rank+1 with wrap-around is the AllGather shape of it, and a collective where only a subset publishes, or where the order is data-dependent, names its successor directly instead. The one ordering rule is that `TBNOTIFY` follows the `TBROADCAST` whose drain wait it speaks for.

`TBWAIT` is the other half: **it tests whether the next tile may be written and writes nothing**. The test belongs to the previous source (the receivers credit *its* `free_scb` when they drain), so that source evaluates it and `TBNOTIFY` forwards the verdict; a waiter consumes one verdict. The baton rides the axis the group does **not** span — NORTH for a ROW group, EAST for a COL group — which is idle on a group pipe, so turn-taking costs no scoreboard, no ring, no window and no packet, and it leaves the group's own EAST/WEST counts untouched (the receive half's order check stays exact). The two AllGather phases still walk their members in one ascending loop (publish on your own turn, `TPOP` on everyone else's) and the whole caller-side obligation is `TBWAIT<Group>` before the publish and `TBNOTIFY<Group>` after it; a single-source broadcast has no next source and calls neither.

**One `TBWAIT` is one `TBNOTIFY`.** The wait carries no exemption for any member, index or round — both counts are just the number of calls, which is what keeps the two sides in step without agreeing on an absolute value. (An earlier version did exempt "index-in-group 0 that has not published yet" as a stand-in for the group's first publish; that silently pinned the schedule to one starting at index 0 and would have released index 0 early in any schedule that does not, so it is gone.) In exchange the two **ends** of the walk are the caller's to leave open, because a token has to be minted before it can be consumed: the **first** publisher does not `TBWAIT` (nobody notified it — or it mints its own token with `TBNOTIFY<Group>(pipe, ownBlockId)` and then waits like everyone else), and the **last** publisher of a finite walk does not `TBNOTIFY`. That second one is not tidiness: an unconsumed token persists in the target's scoreboard — the pipe init re-zeroes `consIndex`, not the GM count — and would satisfy the first `TBWAIT` of a later round or of a later launch that reuses the window. Only a schedule that genuinely circulates round-to-round wraps, its last publisher notifying its first. A group of one member is both the first and the last publisher, so it calls neither half. See `Grid_TPUSH_TBROADCAST_TREDUCE_接口设计说明.md` for the full handshake.

**Ring configuration of a group collective.** A group owns no ring: it uses the two directional rings its `DirMask` names, each `SlotCount` deep. The sender addresses `prod_idx % SlotCount` (its OWN publish counter) and the receiver `cons_idx % SlotCount` (its count of arrivals from that direction). With several sources feeding one direction those two counters are not equal — every source's `prod_idx` starts at 0 while the receiver's `cons_idx` runs over all of them — so **`SlotCount` must be 1 for a multi-source group**, and one slot per direction is what the demos configure (`FFN_NCUT_GROUP_SLOTS_P1/P2 = 1`; `SlotStride` = `FFN_NCUT_SLOT_BYTES_P1` = 1536 B for the `[8,96]` shard and `FFN_NCUT_SLOT_BYTES_P2` = 12288 B for the `[8,768]` row block; window = 1024 B flags + 2 directions x 1 slot). A deeper ring only helps a **single**-source group, where the two counters do coincide.

### GridPipe smoke tests

The group-broadcast capability has a Vec-only data-movement smoke test under `smoke/` (no Cube, no matmul, no data files; in-process verification on the same GM-backed mock as the FFN demos):
- `bcast_smoke` — two modes over the same kernel:
  - **single source** (default): one cell (`--src`) broadcasts its stamped tile to its whole row (or column with `--span-col 1`) via `TBROADCAST<GridGroup>`; every other cell drains it with `TPOP<GridGroup>(pipe, tile, src)` and stores it; the host checks `out[cell] == in[source]`. The default 1x5 row with the source at col 2 exercises receivers on both sides of the source in one run. One publisher means no next source, so there is no turn to pass and neither `TBWAIT` nor `TBNOTIFY` is called.
  - **all sources** (`--all-src 1`): every member broadcasts in turn — an AllGather in miniature, and the mode that covers the caller-side obligation. Each member takes the turn with `TBWAIT<Group>` before its own `TBROADCAST` and hands it on with `TBNOTIFY<Group>` after, with the walk's two ends open (member 0 does not wait, the last member does not notify). Running it with `--span-col 1` on the default 1x5 grid makes the group a single member, which exercises that both ends can be the same cell. The host checks `out[cell][src] == in[src cell]` for every (receiver, source) pair, which is what makes a slot overwritten by a late source visible: a summed check could not tell "source 3 twice" from "sources 3 and 4 once each". Deleting that one `TBWAIT` turns the 32-cell FFN AllGather from `max diff 0` into `max diff 17648` — and note the demo's own PASS verdict is an error-count ratio, so that run still prints SUCCESS; read the max diff.

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
# Single-source broadcast: 1x5 row, source at col 2 (use --span-col 1 + Rx1 grid for a column broadcast)
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2
```

It accepts `--build-only` and needs no data generation.

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

The smoke script reuses `-r/-v/-d`, `--grid-rows/--grid-cols`, `--token-tile/--model-tile` (tile `[T, W]`), and `--build-only`. `run_bcast_smoke.sh` additionally accepts `--src` (source index, default 2), `--all-src` (1 = every member broadcasts, default 0 = single source), and `--span-col` (1 = column group, default 0 = row group).

## Expected Result

On success, each FFN executable prints its bit-exact verdict:

```text
[SUCCESS] 32-cell N-cut FFN GridPipe TREDUCE ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TBROADCAST AllGather PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH AllGather PASS.
```

The smoke test prints:

```text
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```
