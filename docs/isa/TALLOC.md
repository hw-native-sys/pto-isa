# TALLOC

## Introduction

Allocate a producer FIFO slot from a `TPipe` and expose it as a `GlobalTensor` view.

`TALLOC` is used by the `GlobalData` split interface. It lets the producer get the current FIFO slot address, write data into that slot with normal memory instructions such as `TSTORE`, and then commit the slot with `TPUSH(Pipe&, GlobalData&)`.

## Operation Semantics

`TALLOC` performs three steps:

1. Wait for FIFO free space when `pipe.prod.getAllocateStatus()` and `Pipe::shouldWaitFree(pipe.prod.tileIndex)` are both true.
2. Compute the current FIFO slot address from `pipe.prod.tileIndex`.
3. Assign `gmTensor` to the FIFO slot address and increment the producer tile index.

`TALLOC` does not write any data and does not notify the consumer. The producer must write the slot contents before calling `TPUSH(Pipe&, GlobalData&)`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TALLOC(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` is typically an A2A3 `TPipe` declared in `include/pto/npu/a2a3/TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **A2A3 GlobalData producer**:
    - `GlobalData` must satisfy `is_global_data_v<GlobalData>`.
    - `Direction::DIR_C2V`: the producer sees the whole FIFO slot.
    - `Direction::DIR_V2C`: split offsets may be applied for vector subblocks according to `Split`.
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
    - `TALLOC` does not record data-ready; use `TPUSH(Pipe&, GlobalData&)` after writing the slot.

## Examples

### Allocate, Store, Commit

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_talloc(__gm__ void *fifoMem)
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

### V2C Split Allocation

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_v2c_split(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 128;
    constexpr uint32_t FullM = M * 2;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, FullM * N * sizeof(T), FifoDepth>;
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
    TSTORE(slot, tile);
    TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```

## ASM Form Examples

The current public assembly reference does not define a stable PTO-AS spelling for `TALLOC`. Use the C++ intrinsic form for manual CV FIFO programming.
