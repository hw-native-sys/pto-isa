# pto.tpush

`pto.tpush` is part of the [System Scheduling](../../scalar/ops/micro-instruction/README.md) instruction set.

## Summary

`TPUSH` moves a tile from a producer pipeline (Cube or Vector) into a ring FIFO for consumption by a paired pipeline. It is the producer half of the TPipe/TMPipe producer-consumer protocol.

## Mechanism

For every `TPUSH` call:

1. The producer **allocates** by waiting on the consumer's slot-free flag.
2. It then **pushes** the tile data into the FIFO — either via a `TSTORE` to a GM slot buffer, a direct `TMOV`/`TINSERT` into the consumer's local UB/MAT buffer, or a 32-bit control-signal write for `V2C_CTRL`.
3. Finally it **records** the data-ready signal so the consumer's `TPOP` can proceed.

The op runs on FIXP / MTE3 / system pipes depending on the direction; see [Three-Phase Protocol](#three-phase-protocol) for the per-backend signal table.

## What TPUSH Is Not

`TPUSH` is **not** a scalar stack push or a generic FIFO enqueue. It is a structured tile-movement protocol for Cube-Vector tile passing. It is not available on the CPU simulator.

## Architecture: The TPipe Abstraction

A `TPipe<FlagID, DirType, SlotSize, SlotNum, LocalSlotNum>` is a compile-time configured ring FIFO that connects a producer and a consumer. Each `TPipe` owns:

- A **RingFIFO** that manages GM slots and consumer-local buffers
- A **Producer** struct that implements the push protocol
- A **Consumer** struct that implements the pop protocol (for paired use)

The producer and consumer are independent halves of the same logical FIFO — they do not share the same struct instance, but they share the same GM slot buffer address and flag ID namespace.

### Direction Types

The `DirType` template parameter selects the communication direction:

| DirType Constant | Meaning | Producer Side | Consumer Side |
|-----------------|---------|-------------|--------------|
| `DIR_C2V` (1) | Cube → Vector | `TileType::Acc` | `TileType::Vec` |
| `DIR_V2C` (2) | Vector → Cube | `TileType::Vec` | `TileType::Mat` |
| `DIR_BOTH` (3) | Cube ↔ Vector (bidirectional) | Acc + Vec | Vec + Mat |
| `DIR_V2C_CTRL` (4) | Vector → Cube (control signal) | `TileType::Vec` | scalar control |
| `DIR_C2V_GM` (5, A5 only) | Cube → Vector via GM | `TileType::Acc` | `TileType::Vec` |
| `DIR_V2C_GM` (6, A5 only) | Vector → Cube via GM | `TileType::Vec` | `TileType::Mat` |
| `DIR_BOTH_GM` (7, A5 only) | Bidirectional via GM | Acc + Vec | Vec + Mat |

### FIFO Storage Paths

Depending on the direction and backend, data flows through different storage paths:

| Path | Description | Used By |
|------|-------------|---------|
| **GM FIFO** | Tile data written to global memory slots; consumer loads via TLOAD | `*_GM` directions |
| **Local UB FIFO** | Tile data written to consumer's local UB buffer directly; no GM traffic | `DIR_C2V`, `DIR_BOTH` on A2/A3 |
| **MAT FIFO** | Vector writes tile into Cube's local MAT buffer via TINSERT | `DIR_V2C`, `DIR_BOTH` on A2/A3 |
| **CTRL FIFO** | Scalar control signal (32-bit) written to shared control buffer | `DIR_V2C_CTRL` |

On **A5**, the distinction is sharper: `VEC_FIFO` / `MAT_FIFO` / `GM_FIFO` / `CTRL_FIFO` are explicitly typed.

## Three-Phase Protocol

Every `TPUSH` call executes three phases in order:

```
1. ALLOCATE  ──►  2. PUSH  ──►  3. RECORD
```

### Phase 1: Allocate (wait for free slot)

```
Producer ──wait_flag_dev(FlagID+1)──► Consumer
```

The producer waits until the consumer has freed a slot. This prevents overwriting data the consumer has not yet consumed.

- On A2/A3 C2V: `wait_flag_dev(FlagID + 1)` via `PIPE_MTE2`
- On A2/A3 V2C: `wait_flag_dev(FlagID + 1)` via `PIPE_MTE2`
- On A5 C2V: `wait_intra_block(PIPE_FIX, FlagID + 1)` (and +16 for subblock 1)
- On A5 V2C: `wait_intra_block(PIPE_MTE3, FlagID + 1)`
- On A5 V2C_CTRL: `wait_intra_block(PIPE_S, FlagID + 1)`

Skipped if `isAllocate = false` (controlled via `pipe.prod.setAllocateStatus(false)`).

### Phase 2: Push (write data to FIFO)

The actual data transfer depends on the FIFO type and direction:

**C2V (Acc → Vec):**
- A2/A3: `TSTORE_IMPL` writes accumulator tile to GM slot buffer
- A5: Either direct TMOV into consumer's local UB buffer (`C2V_CONSUMER_BUF`) **or** via GM, depending on `is_c2v_ub` vs `is_c2v_gm`

**V2C (Vec → Mat):**
- A2/A3: `TSTORE_IMPL` writes vector tile to GM slot buffer; sub-tile offsets computed via `get_subblockid()` when splitting
- A5: Either TINSERT into consumer's local MAT buffer **or** via GM

**V2C_CTRL:** Writes a single scalar control signal (32-bit) from the vector tile's first element to the control slot buffer.

### Phase 3: Record (signal data-ready to consumer)

```
Producer ──set_flag/ffts_cross_core_sync(FlagID)──► Consumer
```

The producer signals that data is ready. The consumer can now safely wait for and consume it.

- On A2/A3 C2V: `ffts_cross_core_sync(PIPE_FIX, ...)`
- On A2/A3 V2C: `ffts_cross_core_sync(PIPE_MTE3, ...)`
- On A5 C2V: `set_intra_block(PIPE_FIX, FlagID)` (+16 for subblock 1)
- On A5 V2C: `set_intra_block(PIPE_MTE3, FlagID)`
- On A5 V2C_CTRL: `set_intra_block(PIPE_S, FlagID)`

Skipped if `isRecord = false`.

## Tile Split Modes

When producer and consumer operate on different tile shapes (e.g., producer is 128×256, consumer is 64×256), `TileSplitAxis` controls how the tile is decomposed:

| Split Mode | Meaning | Offset Computation |
|-----------|---------|-----------------|
| `TILE_NO_SPLIT` | Single writer, no decomposition | `offset = 0` |
| `TILE_UP_DOWN` | Split along rows; each subblock writes a row block | `offset = subblock_id × ProdM × ProdN × sizeof(T)` |
| `TILE_LEFT_RIGHT` | Split along columns; each subblock writes a column block | `offset = subblock_id × ProdN × sizeof(T)` |

## TMPipe: Multi-Pipe Variant

`TMPipe<FlagID, FiFoType, FiFoDepth, FiFoSyncT, TileDataProd, TileDataCons, EN_UNIT_FLAG, LocalFiFoDepth, VCRatio>` is the multi-pipe version. Key differences:

- `FiFoType`: Selects FIFO implementation (`GM_FIFO`, `VEC_FIFO`, `MAT_FIFO`, `CTRL_FIFO`)
- `FiFoDepth`: Configurable FIFO depth (2–8 on A2/A3; up to 16 on A5)
- `LocalFiFoDepth`: Local UB buffer depth for GM-path consumers
- `VCRatio`: `V2C1_VECS` (1 Cube, 2 Vec cores) or `V2C2_VECS` (1 Cube, 1 Vec core)
- `EN_UNIT_FLAG`: Enables per-slot synchronization for fine-grained flow control

## Syntax

### IR Level 1 (SSA)

```mlir
%event = pto.tpush %tile, %pipe : (!pto.tile<f32, 64, 64>, !pto.tpipe<...>) -> !pto.record_event
```

### IR Level 2 (DPS)

```mlir
pto.tpush ins(%tile : !pto.tile_buf<f32, 64, 64>) pipe(%pipe : !pto.tpipe<...>)
```

## C++ Intrinsic

```cpp
#include <pto/common/fifo.hpp>

using namespace pto;

// Define a C2V pipe: Acc producer → Vec consumer
// FlagID=0, DirType=DIR_C2V, SlotSize=16384 bytes, SlotNum=4, LocalSlotNum=2
using MyPipe = TPipe<0, Direction::DIR_C2V, 16384, 4, 2>;

// Allocate the pipe with GM slot buffer and consumer-local buffers
MyPipe pipe(/* GM slot buffer */ reinterpret_cast<__gm__ void*>(0x100000),
            /* C2V consumer UB buf */ 0x8000,
            /* V2C consumer buf */   0x9000);

void producer_side(AccTile& accTile) {
    TPUSH_IMPL(pipe, accTile);
}

// Or with split mode:
TPUSH_IMPL<MyPipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(pipe, accTile);
```

## TMPipe Usage

```cpp
// Define a V2C multi-pipe: Vec producer → Mat consumer via GM FIFO
using MyMultiPipe = TMPipe<
    4,                    // FlagID
    FIFOType::GM_FIFO,    // GM FIFO
    4,                    // FiFoDepth
    1,                    // FiFoSyncT (sync every 1)
    VecTile,              // Producer tile type
    MatTile,              // Consumer tile type
    false,                // EN_UNIT_FLAG
    2,                    // LocalFiFoDepth
    VecCubeRatio::V2C1_VECS  // 2 Vec cores per Cube
>;

MyMultiPipe multiPipe(/* GM FIFO base */ reinterpret_cast<__gm__ float*>(0x200000),
                     /* local FIFO base */ 0xA000);

void producer_vec(VecTile& vTile) {
    TPUSH_IMPL(vTile, multiPipe);
}
```

## Inputs

| Operand | Description |
|---------|-------------|
| `pipe` | The `TPipe` / `TMPipe` instance shared with the consumer. Carries `FlagID`, `DirType`, slot pointers, and consumer-local buffer addresses. |
| `tile` | The source tile (Acc or Vec) whose contents are pushed into the FIFO. For `*_UB` / `*_MAT` paths, the consumer-local buffer is the destination; for GM paths the slot is in global memory. |
| `TileSplitAxis` (template) | Optional split mode (`TILE_NO_SPLIT`, `TILE_UP_DOWN`, `TILE_LEFT_RIGHT`) used to compute the per-subblock GM offset. Must match the consumer. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals completion of the allocate + push + record sequence. |
| `pipe` | state | A FIFO slot is filled with the pushed tile data, the slot index advances (`tileIndex % SlotNum`), and the data-ready flag for the consumer is raised. |
| `pipe.cons.fifo.ctrlSignal` | scalar (V2C_CTRL only) | Receives the 32-bit control word taken from the producer tile's first element. |

## Side Effects

- Issues a cross-core / intra-block flag wait in the allocate phase (blocks until the consumer frees a slot).
- Writes a data-ready flag for the consumer; this can unblock a consumer waiting in `TPOP`.
- For GM paths: issues an MTE3 store.
- For FIXP paths (`C2V` accumulator drain on A5): issues a fix-pipe drain into the destination buffer.
- For local-buffer paths (`C2V_UB`, `V2C_MAT`): writes directly into the consumer's UB/MAT buffer via `TMOV` / `TINSERT`.
- Does **not** implicitly fence unrelated tile traffic. Callers must use explicit events for cross-pipe ordering.

## Constraints

!!! warning "Constraints"
    - `TileProd::Loc` must be `TileType::Acc` or `TileType::Vec`.
    - `DirType` must be compatible with `TileProd::Loc` and `TileDataCons::Loc`.
    - `SlotNum × SlotSize` must not exceed the available GM region for the FIFO.
    - `FlagID` range: 0–7 per pipe type on A2/A3; 0–15 on A5 with intra-block flags.
    - When `isAllocate = false`, the producer skips the allocation wait; the caller must ensure the slot is free.
    - When `isRecord = false`, the producer skips the ready signal; the caller must ensure the consumer waits externally.
    - Pairing: each `TPUSH` should have a corresponding `TPOP` on the consumer side; skipping allocation or record breaks the protocol.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **CPU simulator**: Not available. `TPUSH` and `TPOP` require the NPU inter-core synchronization infrastructure.
    - **A2/A3**: Supports `DIR_C2V`, `DIR_V2C`, `DIR_BOTH`, `DIR_V2C_CTRL`. FIFO paths: GM and local UB/MAT. Does not support `DIR_*_GM` variants.
    - **A5**: Supports all direction types including `DIR_C2V_GM`, `DIR_V2C_GM`, `DIR_BOTH_GM`. FIFO paths: GM, VEC_FIFO, MAT_FIFO, CTRL_FIFO. Intra-block synchronization uses `set_intra_block`/`wait_intra_block` instead of cross-core `ffts_*`.

## Performance

### A2/A3 Cycle Count

`pto.tpush` is dominated by two phases: the consumer-wait (variable, depends on consumer drain rate) and the data-write phase. The data-ready signal phase is a single cross-core write.

**Cycle model**:

```
total ≈ alloc_wait_latency + push_phase + record_overhead

# push_phase by path:
  C2V_UB (A5 local):       TMOV into consumer UB — vector-pipe write
  C2V (A2/A3, A5 *_GM):    Acc → GM via FIXP + MTE3 — fix-pipe drain dominates
  V2C_MAT (A5 local):      TINSERT into consumer MAT buffer — layout-converting move
  V2C (A2/A3, A5 *_GM):    Vec → GM via TSTORE on MTE3 — SlotSize / mte3_throughput
  V2C_CTRL:                32-bit scalar write — ~startup
```

`alloc_wait_latency` is the steady-state latency between the consumer's release and the producer's wakeup; on a well-balanced pipeline it is hidden by previous-iteration compute.

### FIFO-Path Impact

| Direction | Path | Cost driver |
|-----------|------|-------------|
| `C2V_UB` (A5) | `TMOV` Acc → consumer UB | FIXP drain; ~4× read/write BW asymmetry on A5 |
| `C2V` (A2/A3) / `C2V_GM` (A5) | FIXP → GM via `TSTORE` | FIXP + MTE3 |
| `V2C_MAT` (A2/A3, A5) | `TINSERT` into consumer MAT | Layout-converting; vector-pipe |
| `V2C` / `V2C_GM` | Vec → GM via `TSTORE` | MTE3 bandwidth, `SlotSize` |
| `V2C_CTRL` | 32-bit control signal | Sync only; trivial write |

Deeper FIFOs (larger `SlotNum` / `FiFoDepth`) hide `alloc_wait_latency` better at the cost of GM footprint; `EN_UNIT_FLAG` enables per-slot synchronization for fine-grained flow control.

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Calling `TPUSH` without an eventual matching `TPOP` causes the producer to stall in the allocate phase once all `SlotNum` slots are full.
    - Calling `TPUSH` on the CPU simulator is rejected: the op requires NPU inter-core synchronization infrastructure.
    - `DirType` incompatible with `TileProd::Loc` / `TileDataCons::Loc` is rejected at compile time via `static_assert`.
    - `FlagID` reuse with another concurrently-active synchronization op is undefined behavior.
    - Setting `isAllocate = false` when the slot has not actually been freed overwrites in-flight data the consumer has not yet read.
    - Setting `isRecord = false` for too many iterations leaves the consumer waiting indefinitely (deadlock).
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

### Pattern 1: Acc → Vec Tile Passing (GEMM Post-Processing)

```cpp
// Producer (Cube): accumulator result → Vector consumer
using AccTile = Tile<TileType::Acc, float, 64, 64>;
using VecTile = Tile<TileType::Vec, float, 64, 64>;
using Acc2VecPipe = TPipe<0, Direction::DIR_C2V, 16384, 4, 2>;

Acc2VecPipe pipe(/* GM buffer */ 0x100000, /* C2V UB */ 0x8000, /* V2C buf */ 0x9000);

void cube_kernel(AccTile& acc) {
    // Acc contains result of TMATMUL
    TPUSH_IMPL(pipe, acc);  // Signal Vec that accumulator is ready
}

// Consumer (Vector): receives accumulator tile
void vec_kernel(VecTile& vec) {
    TPOP_IMPL(pipe, vec);   // Wait for and receive accumulator
    // Apply activation, quantization, etc.
}
```

### Pattern 2: Vec → Mat Tile Passing with Row Split

```cpp
// Producer: 128×256 vector tile; Consumer: 256×256 matrix tile
using ProdTile = Tile<TileType::Vec, half, 128, 256>;
using ConsTile = Tile<TileType::Mat, half, 256, 256>;
using Vec2MatPipe = TMPipe<
    2, FIFOType::MAT_FIFO, 4, 1,
    ProdTile, ConsTile,
    false, 2, VecCubeRatio::V2C1_VECS
>;

Vec2MatPipe pipe(/* MAT FIFO base */ 0x300000);

void vec_producer(ProdTile& vec) {
    TPUSH_IMPL(vec, pipe);  // Splits vec into two 128×256 blocks, inserts into mat
}
```

### Pattern 3: Sparse Sync (Skipping Allocation Wait)

```cpp
// After the initial startup phase, slots are guaranteed free at every period boundary.
// Only allocate every N iterations:
pipe.prod.setAllocateStatus(/* iteration % period == 0 */);
TPUSH_IMPL(pipe, tile);
```

## See Also

- [TPOP](./TPOP.md) — Consumer half of the FIFO protocol
- [TPipe and TMPipe source code reference](https://github.com/PTO-ISA/pto-isa/blob/main/include/pto/npu/a2a3/TPush.hpp)
- [System Scheduling](../../scalar/ops/micro-instruction/README.md)
