# TPUSH

## Introduction

Push a producer tile into a `TPipe` FIFO for Cube-Vector communication.

This page describes both the TileData overload and the `GlobalData` commit overload. For GM-slot workflows that expose the FIFO entry as a `GlobalTensor`, use `TALLOC` to allocate a slot view, write the slot with normal memory instructions, then use `TPUSH(Pipe&, GlobalData&)` to commit the slot.

## Operation Semantics

For the TileData overload, `TPUSH` performs three steps:

1. Wait for FIFO space when `Pipe::shouldWaitFree(pipe.prod.tileIndex)` is true.
2. Store the producer tile into the current FIFO slot.
3. Record data-ready synchronization for the consumer.

The producer tile index is incremented after the FIFO slot address is computed.

For the `GlobalData` overload, `TPUSH` only records data-ready synchronization for a slot that was already allocated by `TALLOC`. It does not store tile data by itself.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileProd>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` is typically an A2A3 `TPipe` declared in `include/pto/npu/a2a3/TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **A2A3 TileData producer**:
    - `TileProd::Loc` must be `TileType::Acc` or `TileType::Vec`.
    - `Direction::DIR_C2V`: Cube produces an accumulator tile for vector consumption.
    - `Direction::DIR_V2C`: Vector produces a vector tile for cube consumption.
    - `Direction::DIR_BOTH`: both C2V and V2C producers are supported by the same pipe type.
- **FIFO slot**:
    - `SlotSize` must be large enough for one logical FIFO entry.
    - `SlotNum >= 1`.
    - `Pipe::SyncPeriod` is derived from `SlotNum`: `(SlotNum <= 2) ? SlotNum : SlotNum / 2`.
- **Split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: no sub-vector offset is applied.
    - `TileSplitAxis::TILE_UP_DOWN`: vector subblocks map to row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: vector subblocks map to column halves.
- **Synchronization**:
    - Free-space waits are sparse and controlled by `Pipe::SyncPeriod`.
    - Data-ready record is emitted for each `TPUSH`.
- **GlobalData commit**:
    - `gmTensor` must be a FIFO slot view returned by `TALLOC`.
    - Data must be written into `gmTensor` before calling `TPUSH(Pipe&, GlobalData&)`.
    - `TPUSH(Pipe&, GlobalData&)` ignores the tensor contents and only commits the FIFO slot to the consumer.

## Examples

### C2V Accumulator Push

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_c2v(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(float), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    // Fill acc with a cube computation before pushing.
    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, acc);
}
```

### V2C Vector Push

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_v2c(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    VecTile tile;
    TASSIGN(tile, 0x0);

    TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}
```

### GlobalData Slot Commit

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_globaldata(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
    TSTORE(slot, tile);
    TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
}
```

## ASM Form Examples

The current public assembly reference does not define a stable PTO-AS spelling for `TPUSH`. Use the C++ intrinsic form for manual CV FIFO programming.
