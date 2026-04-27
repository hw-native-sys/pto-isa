# pto.tpop

`pto.tpop` is part of the [System Scheduling](../../scalar/ops/micro-instruction/README.md) instruction set.

## Summary

`TPOP` retrieves a tile from a ring FIFO into a consumer pipeline (Vector or Cube). It is the consumer half of the TPipe/TMPipe producer-consumer protocol, paired with [`TPUSH`](./TPUSH.md).

## What TPOP Is Not

`TPOP` is **not** a scalar stack pop or a generic FIFO dequeue. It is a structured tile-movement protocol for Cube-Vector tile passing. It is not available on the CPU simulator.

## Architecture: The TPipe Abstraction

A `TPipe<FlagID, DirType, SlotSize, SlotNum, LocalSlotNum>` is a compile-time configured ring FIFO. The `Consumer` struct within the `TPipe` owns the pop-side protocol. See [`TPUSH`](./TPUSH.md) for the full `TPipe` abstraction description including direction types, FIFO paths, and storage layouts.

## Three-Phase Protocol

Every `TPOP` call executes three phases in order:

```
1. WAIT  ──►  2. POP  ──►  3. FREE
```

### Phase 1: Wait (block until data is ready)

```
Consumer ──wait_flag_dev(FlagID)──► Producer
```

The consumer blocks until the producer has written data and signaled via `set_flag` or `ffts_cross_core_sync`.

- On A2/A3 C2V: `wait_flag_dev(FlagID)` — Vector waits for Cube's FIXPIPE signal
- On A2/A3 V2C: `wait_flag_dev(FlagID)` — Cube waits for Vector's MTE3 signal
- On A5 C2V_UB: `wait_intra_block(PIPE_V, FlagID)` — Vector waits on VPipe
- On A5 C2V_GM: `wait_intra_block(PIPE_MTE2, FlagID)` — Vector waits on MTE2 after GM load
- On A5 V2C_GM: `wait_intra_block(PIPE_MTE2, FlagID)` (+16 for subblock 1) — Cube waits on MTE2
- On A5 V2C_CTRL: `wait_intra_block(PIPE_S, FlagID)` (+16 for subblock 1)

Skipped if `isWait = false` (controlled via `pipe.cons.setWaitStatus(false)`).

### Phase 2: Pop (load data from FIFO)

The data transfer depends on the FIFO type and direction:

**C2V (Acc → Vec):**
- A2/A3: `TLOAD_IMPL` from GM slot buffer into the Vector tile (after `TASSIGN` to local UB address)
- A5 C2V_UB: `TASSIGN` only — data is already in consumer's local UB buffer (direct push)
- A5 C2V_GM: `TLOAD_IMPL` from GM slot buffer (after `TASSIGN`)

**V2C (Vec → Mat):**
- A2/A3: `TLOAD_IMPL` from GM slot buffer into the Matrix tile
- A5 V2C_MAT: `TASSIGN` only — data is already in consumer's local MAT buffer (TINSERT by producer)
- A5 V2C_GM: `TLOAD_IMPL` from GM slot buffer

**V2C_CTRL:** Reads a 32-bit control signal from the control slot buffer and stores it in `fifo.ctrlSignal`.

For sub-tile splitting, the load address uses `get_subblockid()` to compute the correct GM offset, matching the split mode used by the producer.

### Phase 3: Free (release slot for producer)

```
Consumer ──set_flag/ffts_cross_core_sync(FlagID+1)──► Producer
```

The consumer signals that it has consumed the data, freeing the slot for the producer's next write.

- On A2/A3 C2V: `ffts_cross_core_sync(PIPE_MTE2, ...)` — Vector frees slot for Cube
- On A2/A3 V2C: `ffts_cross_core_sync(PIPE_MTE2, ...)` — Cube frees slot for Vector
- On A5 C2V: `set_intra_block(PIPE_MTE2, FlagID + 1)` — Vector signals back to Cube
- On A5 V2C: `set_intra_block(PIPE_MTE1, FlagID + 1)` (+16) — Cube signals back to Vector

Skipped if `isFree = false`.

## TMPipe: Multi-Pipe Variant

`TMPipe` extends the `TPipe` model with configurable FIFO depth, FIFO type, and local buffer depth. The consumer-side pop for `TMPipe`:

- Uses `DataFiFo<fifoDepth>` to index into the slot array with wraparound
- Optionally uses a local UB/MAT buffer (`useLocalFiFo = true`) to avoid GM traffic for the V2C case
- Supports `GM_FIFO`, `VEC_FIFO`, `MAT_FIFO`, and `CTRL_FIFO` types

## Syntax

### IR Level 1 (SSA)

```mlir
%event = pto.tpop %pipe, %tile : (!pto.tpipe<...>, !pto.tile<f32, 64, 64>) -> !pto.record_event
```

### IR Level 2 (DPS)

```mlir
pto.tpop pipe(%pipe : !pto.tpipe<...>) outs(%tile : !pto.tile_buf<f32, 64, 64>)
```

## C++ Intrinsic

```cpp
#include <pto/common/fifo.hpp>

using namespace pto;

// Same TPipe definition as the producer
using MyPipe = TPipe<0, Direction::DIR_C2V, 16384, 4, 2>;

MyPipe pipe(/* GM slot buffer */ reinterpret_cast<__gm__ void*>(0x100000),
            /* C2V consumer UB buf */ 0x8000,
            /* V2C consumer buf */   0x9000);

// Consumer (Vector) side:
void consumer_side(VecTile& vecTile) {
    TPOP_IMPL(pipe, vecTile);
}

// Or with split mode:
TPOP_IMPL<MyPipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, vecTile);
```

## TMPipe Consumer Usage

```cpp
using MyMultiPipe = TMPipe<
    4,                    // FlagID
    FIFOType::GM_FIFO,   // GM FIFO
    4,                    // FiFoDepth
    1,                    // FiFoSyncT
    VecTile,              // Producer tile type
    MatTile,              // Consumer tile type
    false,                // EN_UNIT_FLAG
    2,                    // LocalFiFoDepth
    VecCubeRatio::V2C1_VECS
>;

MyMultiPipe multiPipe(/* GM FIFO base */ reinterpret_cast<__gm__ float*>(0x200000),
                     /* local FIFO base */ 0xA000);

void consumer_mat(MatTile& matTile) {
    TPOP_IMPL(matTile, multiPipe);
}
```

## Constraints

- `TileCons::Loc` must be `TileType::Vec` or `TileType::Mat`.
- For GM-path consumers on A5 with `useLocalFiFo = true`: `TASSIGN` is issued to the local FIFO buffer base; `TLOAD` then reads from GM into that local address.
- `FlagID` must be in range and unused by any other synchronization operation on the same pipeline.
- The producer must have issued a matching `TPUSH` before the consumer calls `TPOP`; otherwise the consumer waits indefinitely.
- Slot index: `tileIndex % SlotNum` — the ring wraps around.
- When `isWait = false`, the consumer skips the wait; caller must ensure data is already ready.
- When `isFree = false`, the consumer skips the free signal; caller must ensure the producer's next allocation wait succeeds.

## Target-Profile Restrictions

- **CPU simulator**: Not available. Requires NPU inter-core synchronization infrastructure.
- **A2/A3**: `DIR_C2V`, `DIR_V2C`, `DIR_BOTH`, `DIR_V2C_CTRL`. Synchronization via `wait_flag_dev` and `ffts_cross_core_sync`.
- **A5**: All direction types. Synchronization via `wait_intra_block` and `set_intra_block`. Additional `*_GM` paths with GM load. Sub-block support (`FlagID + 16`) for 2-Vec-core configurations.

## Common Patterns

### Pattern 1: Consuming Accumulator Tile (GEMM Post-Processing)

```cpp
// Consumer (Vector): receives accumulator from Cube
using AccTile = Tile<TileType::Acc, float, 64, 64>;
using VecTile = Tile<TileType::Vec, float, 64, 64>;
using Acc2VecPipe = TPipe<0, Direction::DIR_C2V, 16384, 4, 2>;

Acc2VecPipe pipe(/* GM buffer */ 0x100000, /* C2V UB */ 0x8000, /* V2C buf */ 0x9000);

void vec_kernel(VecTile& vec) {
    // Block until accumulator is ready and loaded into vec tile
    TPOP_IMPL(pipe, vec);

    // Now apply activation — e.g., quantization via TMOV
    Tile<TileType::Acc, float, 64, 64> accTmp;
    TMOV(accTmp, vec);
}
```

### Pattern 2: Consuming Vector Tile into Matrix (Attention Accumulation)

```cpp
// Consumer (Cube): receives vector tiles and accumulates into matrix
using ProdTile = Tile<TileType::Vec, half, 128, 256>;
using ConsTile = Tile<TileType::Mat, half, 256, 256>;
using Vec2MatPipe = TMPipe<
    2, FIFOType::MAT_FIFO, 4, 1,
    ProdTile, ConsTile,
    false, 2, VecCubeRatio::V2C1_VECS
>;

Vec2MatPipe pipe(/* MAT FIFO base */ 0x300000);

void cube_kernel(ConsTile& mat) {
    for (int i = 0; i < numHeads; ++i) {
        TPOP_IMPL(mat, pipe);   // Receives and inserts each vector tile
        TMATMUL(mat, mat, scaleTile);  // Accumulate
    }
}
```

### Pattern 3: Sparse Sync (Consumer Skips Wait Periodically)

```cpp
// After startup, data arrives on every period boundary.
// Consumer only waits every N iterations:
pipe.cons.setWaitStatus(/* iteration % period == 0 */);
TPOP_IMPL(pipe, vec);
pipe.cons.setWaitStatus(true);  // Reset for next phase
```

## Relationship with TFREE

`TFREE_IMPL(pipe)` is a no-op on both A2/A3 and A5. It is a placeholder for future explicit free semantics and does not currently perform any action.

## See Also

- [TPUSH](./TPUSH.md) — Producer half of the FIFO protocol
- [TPipe and TMPipe source code reference (A2/A3)](https://github.com/PTO-ISA/pto-isa/blob/main/include/pto/npu/a2a3/TPop.hpp)
- [TPipe and TMPipe source code reference (A5)](https://github.com/PTO-ISA/pto-isa/blob/main/include/pto/npu/a5/TPop.hpp)
- [System Scheduling](../../scalar/ops/micro-instruction/README.md)
