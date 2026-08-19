# Single-device Multi-block FFN GridPipe Demo

## Goal

This demo validates the three distributed-FFN GridPipe collective interfaces — **TPUSH**, **TBROADCAST**, **TREDUCE** — on a single-device logical FFN grid on A2/A3. The host runs one process on the selected device and launches `gridRows * gridCols` blocks; each block owns one logical cell. There are **four examples**, one per (interface, FFN pattern), all on the same pure 1D N-cut 4×8 = 32-cell topology with the real DeepSeek-v4 Pro shapes (M=T=8, H=7168, I=3072):

| Example (run script / executable) | Interface verified | FFN pattern | Cross-cell collective |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | explicit `TPOP(pipe, ..., prodId)` + `TADD` + `TPUSH(pipe, ..., consId, isLastTransfer)` relay |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | nearest-neighbor `TPUSH`/`TPOP` relay gather (fan-in-1 DAG) |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` MPSC group broadcast |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | fused `TREDUCE<GridGroup, Sum>` N→1 group fan-in (`mov_ubuf_group`, op=SUM) |

Each example compares its `[T, H]` output with `golden.bin` using a `1e-3` tolerance. All four pass **bit-exact** on the NPU (`max diff = 0`, run with `-r npu`); see [Bit-exactness notes](#bit-exactness-notes).

The cross-cell collectives use the A2/A3 GridPipe mock backend: local SRAM windows backed by GM in the mock, fake `HcclDeviceContext` window pointers, per-channel ready/free/close counters, `dcci/dsb` fences, and spin waits. This validates the programming model and same-device mock path; it is not multi-card communication validation.

Unicast names peers at runtime (`consId` on TPUSH and `prodId` on TPOP), while concurrent group broadcast keeps its `TBROADCAST<GridGroup>` / `TPOP<GridGroup>` interface. The standalone broadcast smoke test lives under `smoke/`.

## Files

| File | Purpose |
| --- | --- |
| `README.md` / `README_zh.md` | English / Chinese documentation. |
| `CMakeLists.txt` | Builds the four host executables and their mixed Cube/Vec device kernel shared libraries, plus the three smoke targets. |
| `run_treduce_reducesum.sh` / `run_tpush_reducesum.sh` | Set up CANN, generate data, configure CMake, build, and run the TREDUCE / TPUSH ReduceSum examples. |
| `run_tbroadcast_allgather.sh` / `run_tpush_allgather.sh` | Set up CANN, generate data, configure CMake, build, and run the TBROADCAST / TPUSH AllGather examples. |
| `ffn_config.hpp` | Compile-time grid shape, tile shape, GridPipe window sizes, buffer sizes, SwiGLU clamp bounds, the A3 precision-mapping table, and Batcher GM arena byte sizes. |
| `kernel_launch.hpp` | Host-side mixed kernel launch declarations (one per example). |
| `main_treduce_reducesum.cpp` / `main_tpush_reducesum.cpp` | ReduceSum host drivers: ACL setup, fake HCCL context / local GridPipe windows, working buffers, Batcher load/distribute, kernel launch, golden comparison, cleanup. |
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel: the EAST+SOUTH reduce uses the fused `TREDUCE<GridGroup, Sum>` group fan-in (`mov_ubuf_group`, op=SUM) at the sink. |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel: EAST and SOUTH share one channel across launches, exercising close + relay-count rebinding with explicit TPOP + TADD + TPUSH. |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host drivers. |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel: the two gather phases use `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`. |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel: the two gather phases use a bidirectional `TPUSH`/`TPOP` relay. |
| `batcher.hpp` | Host-side GM-simulated **Batcher**: owns the full input + the full DRAM-resident weights in GM, splits them column-parallel into per-cell shards, broadcasts x, and exposes the output-collection region. |
| `tpipe_tmov_inl.hpp` | Directional `TMOV` overloads that lower Cube↔Vec C2V/V2C transfers to the existing `TPUSH`/`TPOP`, so the kernel body never spells out the handshake. |
| `gridpipe_payload_inl.hpp` | Local GridPipe payload hooks and fake-window adapter: peer-slot/SCB resolution, tile-to-producer-L1 staging, producer-L1-to-peer-ring copies, local receive-ring drains, and the TPOP locality guard. |
| `smoke/` | Standalone Vec-only GridPipe smoke tests: broadcast (`bcast_smoke_*` + `run_bcast_smoke.sh`), unicast handover (`unicast_smoke_*` + `run_unicast_smoke.sh`), reduce↔unicast channel relay (`relay_smoke_*` + `run_relay_smoke.sh`), and `run_bcast_smoke_matrix.sh` — the coverage matrix over group flavour, publisher count, ring depth, ticket batch and channel handover. |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | Grid CCE facades: `copy_l1_to_peer_l1` / `copy_l1_to_group` for outbound unified-L1 transfers, `sync_hscb` for peer publication, `wait_ipc_scb` for local blocking waits, and `mov_ubuf_group` for group reduction. The A3 mock represents L1 ranges with GM windows. |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 data model + mock support: per-channel ready/free/close SCBs, producer/consumer bindings, the three-state per-consumer FSM, durable relay counters, mesh/group resolvers, fault sentinels, and the `GmSramArena` TPOP locality guard. |
| `scripts/gen_data.py` | Generates the FULL fp16 X/weight tensors (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) the Batcher consumes, plus an fp32 SwiGLU `golden` reference. |
| `build/` | Ignored generated build directory. |
| `out/` | Ignored generated data directory. |

## Bit-exactness notes

Run with `-r npu` (the `sim`/`camodel` modes fail `aclrtSetDevice` 507033); on a shared host every run goes through `task-submit`. All four examples produce `max diff = 0` vs `golden.bin` — bit-exact, not merely within the `1e-3` tolerance. Two real bugs once masked that, both now fixed:

- **Cache-line doorbell stride (TBROADCAST MPSC).** `TBROADCAST<GridGroup>` is a true concurrent MPSC channel — every member may publish at the same instant, each ringing only its *own* doorbell. Those doorbells were once packed as consecutive `u32`s: `GroupMax` of them × 4 B ≤ 32 B < one 64 B cache line. Several producers therefore each wrote a different word of the *same* line while the consumer `dcci`-invalidated + read it each poll; the AICore store is line-granular, so one producer's write-back clobbered another's word and that doorbell silently **dropped from GM** (proven by a D2H dump that bypasses the consumer's `dcci`). Symptom: sporadic `wait ready timeout`. Fix: give every independently-written scoreboard its **own cache line** (`kScbLineStride = 64` in `grid_intrinsic.hpp`). The phase-B/C handshake dropped from 10–20 s (retry waves) to ~40 µs. The bespoke lane arrays are gone entirely now — a group collective atomic-adds into the ready scoreboard of the pipe's RESERVED BROADCAST CHANNEL (so the channel budget is independent of the group width), and those scoreboards have owned a line each from the start — and the same rule sizes both mailbox queues, one line per peer.
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

The host allocates `gridRows * gridCols` local SRAM windows, backed by GM in the mock. `TPUSH(..., consId)` resolves that consumer's SRAM slot with `ResolvePeerSlotAddr`, writes the payload, then publishes READY; `TPOP(..., prodId)` waits on local READY, loads the local slot, and returns FREE credit to that producer.

The mock uses GM flag polling and cache maintenance to emulate the intended LPU WSE `SPR` / `WFE` behavior on A2/A3.

### NoC write-only address-segment SRAM model (`GmSramArena`)

To stay close to real silicon, the mock models future-hardware per-core SRAM as an explicit **GM address-segment arena**. The single contiguous `gridRows*gridCols * FFN_GRID_WINDOW_BYTES` window buffer is cut into equal per-core segments, so segment `c` (== `windowsIn[c]`) *is* core `c`'s private SRAM:

```text
segment c = [base + c*winSize, base + (c+1)*winSize)   // base == windowsIn[0]
```

Each segment keeps receive and producer storage distinct:

```text
control + record | unicast receive rings | optional group-collective receive ring | producer staging slot
                                                                                    [SlotStride bytes]
```

The producer slot is appended after every receive-side region. `TPUSH` and `TBROADCAST` first stage the tile there, then map that local L1 source to the selected peer payload-ring address. This models real WSE, where vector data and L1 share one SRAM address space, without allowing the producer source to alias a receive ring.

`GmSramArena` (in `include/pto/npu/a2a3/grid_intrinsic.hpp`) carries `{base, segBytes, numSegs}` plus the `SegmentOf` / `InSegment` classifiers; the demo builds it on-device from the fake `HcclDeviceContext` window table (`SramArenaFromCtx`). It is the single source of truth for "which core owns this address".

This makes the NoC contract explicit and **enforced**: the fabric can only *write* across cores, never *read*.

- `TPUSH(pipe, tile, consId, ...)` reads this core's dedicated producer slot and writes the payload into the **consumer's receive ring** — a cross-segment write, exactly what the fabric does.
- `TPOP(pipe, tile, prodId)` may only drain **this core's own** segment. `GRID_TRY_TPOP_IMPL` calls the `PopSlotIsLocal` guard before the payload read; on a cross-segment read it raises `kFaultPopNonLocal` (`0x205`, "pop non-local segment") and aborts the pop. The host's `CheckGridPipeFaults` surfaces it.

On native hardware `PopSlotIsLocal` is a no-op (`true`): a TPOP read address is local by construction because the fabric has no remote-read path. The guard exists only because the A2/A3 mock backs SRAM with a GM window that *can* physically read any address, so without it a demo could silently rely on a remote read the silicon cannot perform. A compile-time `static_assert(GmSramArenaSelfCheck())` is built into every A2/A3 kernel, so a regression in the segment math fails the build rather than mis-routing a pop.

> The `pto::comm` variants (`TREDUCE` / `TGATHER`) intentionally do **not** follow this rule: they are a root-pulls-from-every-rank collective (HCCL/RDMA-style remote reads), a different memory model from the WSE NoC. Only the GridPipe `TPUSH`/`TPOP` path is constrained to write-only.

### IPC_SCB scoreboard intrinsic API

GridPipe ready/free/close synchronization follows the V8 IPC_SCB scoreboard route. Each channel owns one SCB of each kind. The handshake intrinsics live in `include/pto/npu/a2a3/grid_cce_intrinsic.hpp` as a thin CCE facade layer — each facade forwards 1:1 to a `__builtin_cce_*` under `PTO_GRID_CCE_NATIVE`, and otherwise emulates the same semantics in the A2/A3 mock with a GM word + cache maintenance (`dcci`/`dsb`):

- `copy_l1_to_peer_l1(dstPeerSlot, srcProducerSlot, transferScratch, bytes)` (G1 / HW-DEP-0) writes the isolated local producer L1 slot into the resolved neighbor receive-ring slot. `transferScratch` exists only for the A3 GM-backed DMA pump; it is not an architectural source address. The operation is not self-synchronizing; the following `sync_hscb(READY)` publishes data-ready.
- `sync_hscb(peerScb, absCount)` (V8 `SYNC_HSCB`/`ST_HSCB`, G2 — reused HSCB store + neighbor IPC_SCB addressing / HW-DEP-1) stores an absolute count into the peer's `ready_scb`, `free_scb`, or `close_scb`. The target kind and peer are resolved into `peerScb` by `RemoteScbPtr`.
- `wait_ipc_scb(localScb, threshold, slot)` (V8 `WAIT_SPR`, G3 — reused IPC_SCB blocking wait) reads + blocks in **one** instruction: the entry reads the local IPC_SCB and proceeds if it is already `>= threshold`, else blocks the current pipe until the peer's `sync_hscb` store raises it. V8 dropped the V7 `MOV_SPR2X` non-blocking peek — there is no separate read step. The demo calls the `wait_ipc_scb_sim(..., maxSpins)` mock wrapper, which adds a spin-timeout fault sentinel so a handshake deadlock fails the test instead of hanging; the documented hardware interface is the void `wait_ipc_scb`.

Payload destination resolution (turning a local receive-ring slot / scoreboard word into the same byte offset in a peer's GM window) is a plain runtime helper in `gridpipe_payload_inl.hpp` (`ResolvePeerSlotAddr` / `RemoteScbPtr`), not an intrinsic. The source is separately fixed at `producerSlotBase`. TPOP's local drain reuses `copy_gm_to_ubuf`; the NoC is write-only, so there is deliberately **no cross-core read** of payload. `PopSlotIsLocal` rejects a mis-wired cross-segment read.

Native lowering targets the real CCE HSCB/IPC_SCB stack. The compiler-facing copy builtin retains its historical `ubuf` spelling, but the facade always passes the dedicated unified-L1 producer address; the caller's tile address is never used as the NoC source mapping. The A3 mock represents scoreboards and both L1 ranges with GM plus cache maintenance.

`TPUSH(pipe, tile, consId)` waits the selected local producer channel's `free_scb`, stages the tile in `producerSlotBase`, copies that L1 range to the independently selected consumer channel's ring, then publishes `prod_idx`. `TPOP(pipe, tile, prodId)` waits its local consumer channel's `ready_scb`, reads only that local receive ring, then publishes `cons_idx` to the negotiated producer channel's `free_scb` at the peer. The two channel indices need not be equal.

### Time-division MPSC relay counting

The producer keeps one FSM entry per consumer (`UNBOUND`, `ACTIVE`, or `CLOSED`) plus a local producer-channel table (`UNBOUND`, `ACTIVE`, or producer-`CLOSED`). An `ACTIVE` consumer uses the normal TPUSH fast path. For an `UNBOUND` or `CLOSED` consumer, the producer first reserves an unused or producer-`CLOSED` local channel, waiting if none is available. It then posts `[mode, local producer channel, producer block id]` — one 32-bit store, so payload and commit are the same write — into **its own slot** of the consumer's bind-request queue; it does not wait for the consumer to choose the producer-side resource.

**The mailbox is a queue at both ends.** `kBindRequestOffset` (in the consumer's window) and `kBindResponseOffset` (in the producer's window) are arrays of `kGridBindQueueDepth` cache lines indexed by the **peer's logical block id**: request slot `p` is written only by producer `p`, response slot `c` only by consumer `c`. Several producers may therefore ask the same consumer at the same instant without overwriting each other — the single-line mailbox could not, which is precisely why the group collectives used to derive their channels instead of negotiating them. A consumer serves **at most one request per pass**, rotating where it starts scanning so nobody starves, and leaves the rest untouched until it can serve them; an unservable request (no free channel yet, or a live flow from that same producer still draining) simply stays pending instead of blocking the consumer. Indexing by block id rather than by a rotating head is structural: A2/A3 has no fetch-and-add and the fabric has no remote read, so there is no ticket a requester could be handed. A peer whose block id is outside the queue raises `0x50B`; raise `PTO_GRID_BIND_QUEUE_DEPTH` if a mesh needs more.

The consumer scans its receive channels, preferring never-used entries and otherwise requiring only `close_scb > closeBaseline` — the old producer has published CLOSE, i.e. no more items will arrive. **That is the whole condition: no drain is required, of the retiring producer or of anybody before it.** The new producer is handed the channel's surviving ready count as its `prod_idx` and this consumer's `cons_idx` as its `free_scb` baseline, so its very first slot-reuse test (`free >= prod_idx + 1 - SlotCount`) is stated in the *same absolute stream* as any tail still in the ring; it can no more overwrite an undrained item than the retiring producer could have. The reader needs nothing remembered about the retiring producer either — a channel is **one continuous stream** across the handover (one ring, one absolute count, writers in sequence), so a later `TPOP` does exactly what a `TPOP` always does: read the local slot at `cons_idx` and store the new `cons_idx` into the `free_scb` of whoever holds the channel *now*. Which core wrote those bytes never enters that arithmetic. The one consequence for the caller: after accepting a new producer on a channel, drain the leftovers under the **new** producer's name — a producer id only selects a channel, and the retired one owns none. Naming the retired producer instead waits for a binding that will not come and times out; it does not corrupt. It **clears the request slot before answering** (a producer only asks again once it has the answer, so the clear can never wipe the next round's request), writes its current `cons_idx` into `free_scb[producer channel]`, and returns `[ready_scb baseline, consumer channel + granted]` into the producer's response slot, baseline first and commit last. The producer polls only that commit word, then installs the ready baseline into `prod_idx` and records the local-producer-channel ↔ remote-consumer-channel mapping. Counts are absolute overwrite values and continue monotonically on each consumer ring across producer turns; they are not increments and no SCB is reset.

Every mailbox wait — on either side — **serves its own request queue between attempts**, so a core blocked waiting for a grant is still handing out grants. Two cores that ask each other at the same instant (the normal state of a group collective) would otherwise deadlock.

The public A2/A3 API is `TPUSH(pipe, tile, consId, isLastTransfer, events...)`; omitting the Boolean is equivalent to `isLastTransfer == false`. Set it to `true` only on the last tile of that producer→consumer turn. The final TPUSH publishes the same absolute count to the remote consumer channel's `close_scb` after payload and READY, then transitions both that consumer FSM entry and the local producer channel to `CLOSED`. A unicast flow is still time-division — one writer per channel at a time — but the *requests* for one are no longer serialized by the caller: they queue.

### Group broadcast and reduce data movement

The group COPY and reduce modes share the native group opcode, but their facades keep the local address contract explicit:

- `copy_l1_to_group(srcProducerSlot, groupSlot, transferScratch, ...)` is the broadcast path. It fans out from the isolated producer L1 slot; the A3 mock expands the operation per member, while native still emits one group COPY.
- `mov_ubuf_group(..., op=SUM/MAX/MIN, ...)` is the current group-reduce path. The sink reads each member contribution and folds members in ascending block-id order, preserving relay accumulation order.

`GRID_TBROADCAST` stages once, then uses `copy_l1_to_group` for an affine group arena or per-member `copy_l1_to_peer_l1` otherwise. `GRID_TREDUCE_GROUP_IMPL` continues to lower SUM/MAX/MIN to `mov_ubuf_group`; it is a genuine N→1 fan-in and is distinct from the peer-id TPOP + TADD + TPUSH relay.

**The reduce doorbell is pulled, not pushed** (scheme C of `2026-08-12-组归约门铃归属方案分析-全员调用与sink-only的五种形态.md`). `mov_ubuf_group(op=SUM)` is already the sink *pulling* every member's contribution; scheme C extends that to the ready flag:

- a member publishes "my round-`r` contribution is in place" by storing `r+1` into its own window's epoch word — an ordinary **local** store, no cross-core action at all;
- the sink pulls every member's epoch word until all have reached `r+1`, then folds;
- after the fold the sink **pushes** `r+1` into each member's `free_scb`, which is that member's "your contribution has been read, you may refill it" release. The sink is the only writer of those counters, so this is a plain absolute-count store — no atomic needed (the broadcast needs one because it has K writers per counter, this has one).

Where to push that credit is what the **bind mailbox** carries: each member binds to the sink once per collective (`GridBindMode::GROUP_PULL`), telling it which `free_scb` to credit, and gets back how many rounds the sink has already folded so it can rebase. Those requests queue like any other, one served per pass. A pull binding allocates **no ring** — its payload travels through the caller's own arena — but it does take a channel's *counters* out of the same pool a unicast flow draws from, and the two **take turns** on it:

- **reduce → unicast (the zero rule).** The retiring collective's `cons_idx` counts folds, not tiles, and it never wrote the ring or rang the doorbells, so there is no baseline worth relaying: the consumer zeroes `cons_idx` and that channel's own ready/close scoreboards (`MOVX2SPR`, legal here because at that instant the channel has no external writer), then answers `prod_idx = 0` and stores 0 into the producer's `free_scb`.
- **unicast → reduce (drain, then baseline + round).** The channel is granted only once the retiring flow has published CLOSE *and* been fully drained — the reduce abandons the ring, so a leftover would never be read and the old consumer would still owe a FREE store that would land in the collective's credit counter. Nothing is cleared afterwards: the surviving `cons_idx` becomes the collective's baseline, and the sink hands it to every member twice — as its round origin and as the starting value of its `free_scb` — so both sides count baseline + round from there. The member also clears its epoch word before it asks, because that third counter cannot be rebased by a baseline the sink states after the fact.

`smoke/relay_smoke_kernel.cpp` runs that sequence twice over one channel. Phases 0–2 use one stage per launch and cover
the compatibility backstop. Phase 3 runs reduce → unicast → reduce entirely inside one launch. On the final round of
each reduce, every participant passes `isLastRound = true`: each member waits for its final sink credit before releasing
its producer-side channel, while the sink releases its consumer-side channel after publishing all final credits. The
next unicast or reduce tenant can then bind immediately without waiting for another `InitGridPipeFromWindow`.

The price is stated in the analysis and is not an implementation detail: `WAIT_SPR` can only block on this core's own IPC_SCB, and a core may not write its own, so a *pulled* flag cannot be suspended on — the sink **spins** where it used to block. `HW-DEP-B` (see `mov_peer_word_to_gpr` in `grid_cce_intrinsic.hpp`) is the proposed ISA change that would give the suspend back: a scalar `ALL_GE`/`MIN` mode on the group instruction whose result may land in a `WAIT_SPR`-able IPC_SCB. A stalled epoch pull is reported as `0x303`.

The reduce is still **called by every member**, not only by the sink: the "don't refill your slot while the sink is still reading it" wait belongs to the member and a sink-only API has nowhere to put it, and it matches `TBROADCAST` and every collective API of this shape (`MPI_Reduce` included) — all participants call the same operation and the root argument selects the role. The member half moves no payload: one credit wait plus one local store, and one bind handshake for the whole collective. There are no turns to take, because with no ready doorbell there is no shared count to serialize.

### fp32 reduction

The reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This keeps `downPartial`, `yOutput`, and `golden.bin` in fp32 for direct tolerance-based comparison. The ReduceSum reduce is H-chunked (`FFN_RS_REDUCE_SLOT_COUNT = kHSegs` = 7, one slot per H-segment): the `treduce` example folds the segment-h partials with `GRID_TREDUCE_GROUP_IMPL` at the sink (one round per segment, so no member ever waits on free credit), while the `tpush` example relays them hop-by-hop with peer-id TPOP + TADD + TPUSH. The `tpush` variant's EAST and SOUTH launches intentionally share one physical channel; the last H segment closes the first turn and the next launch resumes from the relayed count. The `treduce` variant gives each phase its own window — its two phases form different groups over the same cells, and while the negotiated bind now rebases a reused channel correctly, a window per phase keeps the two collectives' bookkeeping visibly separate.

### Peer-id unicast

Unicast direction is derived by the schedule, not encoded in the instruction type. TPUSH receives the destination logical block id (`consId`), TPOP receives the source logical block id (`prodId`), and `RemoteScbPtr` resolves the symmetric slot/SCB offset in that peer's window. This keeps one GridPipe usable when the peer changes between phases.

### Concurrent group broadcast (TBROADCAST)

`TBROADCAST<GridGroup>` (`ROW`, `COL` or `SUBRECT` as the first template argument) broadcasts a cell's tile to every other cell of its group as one op: the per-target writes are batched with no inter-target fence, the whole broadcast pays a single publish fence, then the ready doorbells fire. It is not lowered to a per-hop `TPUSH` loop.

Unlike the old single-source `TPUSH<GridSpan>` multicast (fan-in 1, which forbade concurrent senders), `TBROADCAST` is a 真·同时 MPSC channel: every member of the group may call it at the same instant. Three independent decompositions make that safe — one for the address, one for the count, one for the credit:

- **address — the caller's sequence number.** `TBROADCAST(pipe, tile, basek, …)` writes ring slot `basek % SlotCount` of the reserved broadcast channel. No identity enters the address, so a receiver's ring is sized by **its own SRAM** rather than by the number of writers — the defect analysed in `2026-08-13-TBROADCAST写入地址由身份推导的可扩展性缺陷与前缀偏移改造方案.md` (判据 M2/M3), where the old `(r*K + rank) % BcastSlotCount` silently required `BcastSlotCount ≥ K`. `basek` is a producer-side value, identical in every receiver's window, so the fan-out is still ONE `copy_l1_to_group` (判据 M4). The caller must keep it **unique, increasing and dense** per collective (`round*K + rank`, or plain `round` for a single source): density is what lets every receiver derive the same grant order with no communication.
- **count — a ticket on a reserved channel.** Channels `[0, kGridBcastChanCount)` (default: channel 0 alone) are reserved for the collective; the rest serve unicast flows and the group reduce. A publisher asks each receiver through the **group mailbox** — one request + one response line per **rank-in-group**, depth `GroupMax`, not one per core in the mesh — and each granted publisher raises the channel's ready count with ONE `ATOM_ADD_HSCB`. The receiver knows the batch has landed when the count reaches `ticketEnd = ticketBase + grants`, which is the single comparison that makes several concurrent publishers on one scoreboard exact.
- **credit — atomic add, the other way.** Each receiver adds 1 to the publisher's `free_scb` when it drains a tile, and the publisher waits for `baseline + round × (K-1)` before starting a new round. One receiver can contribute at most `round`, so the sum reaches the threshold only when *every* receiver has — and no looser threshold is sound on a summed counter (a fast receiver would mask a slow one).

**The grant is the write permission.** A receiver only grants a slot whose previous tenant its own caller has drained, which covers the case a per-publisher credit counter is blind to: once the ring is shallower than the group is wide, a slot's previous tenant belongs to *somebody else*, and no counter the publisher owns can see that. So the order is: ask first, write once every receiver has said yes.

**Why holding some grants while waiting for the rest cannot deadlock.** Each receiver grants only inside the window `[grantHead, grantHead + n)` of the dense `basek` sequence (`n = PTO_GRID_BCAST_TICKET_BATCH`, clamped to `SlotCount`), and keeps at most `n` grants outstanding. Let `g` be the smallest basek not yet completed: every basek below it completed, hence was granted everywhere, so every receiver's `grantHead ≥ g`, and `== g` where `g` is still ungranted. Every grant that receiver ever issued was therefore below `g + n`, and those below `g` have already arrived — so at most `n-1` are outstanding and the cap cannot block `g`. `g` is thus granted or immediately grantable **everywhere**, its publisher completes, and the window slides. No priority protocol, no release-and-retry, and no communication between receivers.

`n` is therefore not just a throughput knob: it is the collective's **concurrency degree**. `n = 1` serialises publishers strictly by `basek`; `n = SlotCount` lets a whole ring's worth publish at once (what the demos configure).

With `SlotCount ≥ K` the caller has no ordering obligation at all:

```cpp
TBROADCAST<Group>(pipe, tile, /*basek=*/myRank);
for (each srcRank except self) { TPOP<Group>(pipe, tile, srcRank); }   // any order
```

A group **wider than `SlotCount`** must be published in waves of `SlotCount` with drains in between, because a receiver cannot free a ring slot while its own caller is blocked inside `TBROADCAST`. That is a property of the pattern, not of this implementation — the ring physically holds `SlotCount` undrained tiles — and it is exactly what the caller expresses by how it allocates `basek`. Violating it blocks (and times out into a fault sentinel); it does not corrupt. The bcast smoke covers both: `--slot-count` below the group width switches its kernel to the wave loop.

Every wait inside the collective **serves while it waits** (one grant pass per spin), so a core blocked publishing keeps handing out tickets — which is what lets two members grant each other. The price is that these waits poll rather than suspend on `WAIT_SPR`: a core that suspends stops serving, and what it waits for may be queued behind exactly that service (`HW-DEP-C` would be a `WAIT_SPR` that can wake on any of a set of scoreboards). The unicast TPUSH/TPOP path is unaffected and still blocks.

Drain order is free — `TPOP<Group>(src)` waits for *that source*, and reads the slot that source's own `basek` named — so `0x405` and `0x50A` stay retired. `0x406` is new: a request whose `basek` this receiver has already granted, i.e. a caller sequence that repeated or rewound.

**TBROADCAST has no `isLast` or CLOSE operation.** Its channel is reserved by index and each grant expires on
arrival, so a CLOSE store would only race the next batch's atomic adds on the same scoreboard. Group TREDUCE is
also a pull collective rather than a stream, but its optional `isLastRound` has a different purpose: it releases
the shared credit channel after the final fold so a unicast flow or another reduce can bind in the same launch; it
does not publish CLOSE. `TPUSH` keeps `isLastTransfer`, which is the only interface where CLOSE marks the end of a
unicast producer turn.

The per-member payload fan-out uses one `copy_l1_to_group` operation when the group arena is affine (see [Group broadcast and reduce data movement](#group-broadcast-and-reduce-data-movement)).

### GridPipe smoke tests

`bcast_smoke` is a Vec-only data-movement smoke test (no Cube, no matmul, no data files) on the same GM-backed mock. One source cell broadcasts a stamped fp32 `[T, W]` tile to its row, column, or sub-rectangle; each receiver drains and stores it, and the host checks `out[cell] == in[source]`.

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
# ONE configuration: single-source broadcast, 1x5 row, source at col 2
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2

# THE COVERAGE MATRIX: every knob that changes a different part of the protocol
bash smoke/run_bcast_smoke_matrix.sh --list          # the plan, builds nothing
bash smoke/run_bcast_smoke_matrix.sh --from 7 --to 9 # a slice, e.g. the 24-cell cases
```

`run_bcast_smoke.sh` (and `run_unicast_smoke.sh`) runs one compile-time configuration;
`run_bcast_smoke_matrix.sh` runs the set that together covers the interface — group flavour (ROW / COL / SUBRECT, plus a SUBRECT strictly *inside* the
mesh so the cells outside it are proved to stay no-ops), one source vs every member publishing at the same
instant, one round vs many (slot reuse and the producer-side credit), `SlotCount >= K` vs `< K` vs `== 1`
(no ordering obligation vs waves vs a slot changing hands on every tile), ticket batch `n = SlotCount` vs
`n = 1`, and scale: **3x8 = 24 cells all publishing at once**, which is what actually loads the doorbell —
24 concurrent atomic adds on one reserved channel's ready count. Each case is a full rebuild, so
`--from/--to` runs a slice that fits a bounded task-queue slot. All three scripts accept `--build-only` and
need no data generation.

The last two cases are the **unicast time-division handover** (`run_unicast_smoke.sh`), and they are the only
test in this tree that reaches it: two producers take turns on ONE consumer channel, and the second takes it
over *while the first one's tiles are still undrained*. A baton (`A -> B` on A's reopened producer channel)
makes "A first, then B" a property of the program rather than of the scheduler, and the consumer's only drain
names B — which is what pins `cons_idx` at 0 while the channel changes hands. The wrapping variant
(`--slot-count 2`) additionally forces B's first tile onto A's first slot, so B must wait on the credit
baseline it was handed at bind time before it may write: the payload-safety half of relay counting. Both are
regression tests with teeth — restoring the old "rebind only after the drain" gate makes them fail with
`0x504` at the consumer and `0x505` at producer B, which is the deadlock that gate causes here.

`run_relay_smoke.sh` covers the other kind of handover: a group REDUCE and a unicast flow taking turns on the
same channel, now that the two share one pool instead of each owning an index. One shared-pool channel, three
launches over one window — reduce, unicast, reduce — so both rules run, plus a concurrent side flow that
leaves the second member a credit leftover which does *not* match the fold baseline. It has teeth in the same
way: disabling the reduce→unicast zero rule fails phase 1 with `0x301` at the consumer (it waits for a ready
count the new producer will never reach), and dropping the sink's free-baseline store fails phase 2 with
`0x302` at both members and `0x303` at the sink — the members wait for credit that is not coming while the
sink waits for the epochs they have not published.

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

The broadcast smoke script reuses `-r/-v/-d`, `--grid-rows/--grid-cols`, `--token-tile/--model-tile` (tile `[T, W]`), and `--build-only`; it adds `--src`, `--span-col`, and `--subrect` with `--rect-r0/r1/c0/c1` / `--rect-src` to scope a sub-rectangle.

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
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```
