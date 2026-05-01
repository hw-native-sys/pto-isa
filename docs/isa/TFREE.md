# TFREE

## Introduction

Release FIFO space for a `TPipe` entry.

For the TileData `TPOP` flow, `TPOP` already performs the free-space notification step internally. Therefore the TileData-oriented `TFREE(Pipe &pipe)` interface is currently a no-op and exists only for API symmetry with the `GlobalData` flow.

For the `GlobalData` flow, `TFREE(Pipe&, GlobalData&)` releases a FIFO slot view returned by `TPOP(Pipe&, GlobalData&)`.

## Operation Semantics

For TileData push/pop:

1. `TPUSH(Pipe&, TileData&)` stores a tile into the FIFO and records data-ready.
2. `TPOP(Pipe&, TileData&)` waits for data-ready, loads the FIFO slot into a tile, and notifies free space according to `Pipe::SyncPeriod`.
3. `TFREE(Pipe&)` performs no additional action.

Use `TFREE(Pipe&, GlobalData&)` only when using the `GlobalData` split interface where `TPOP` returns a FIFO slot view and the caller explicitly decides when the slot can be released.

For `GlobalData`, `TFREE` checks `pipe.cons.getFreeStatus()` and `Pipe::shouldNotifyFree(...)` before emitting the free-space notification.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

The corresponding A2A3 implementation in `include/pto/npu/a2a3/TPop.hpp` is intentionally empty for this overload:

```cpp
template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}
```

## Constraints

- **TileData flow**:
    - Do not use `TFREE(Pipe&)` to release a tile popped by `TPOP(Pipe&, TileData&)`; the release is already handled inside TileData `TPOP`.
    - Calling `TFREE(Pipe&)` after TileData `TPOP` has no effect on A2A3.
- **GlobalData flow**:
    - Use `TFREE(Pipe&, GlobalData&)` after the data in the popped FIFO slot is no longer needed.
    - `gmTensor` is only used to select the overload; the implementation does not read or write tensor contents.
    - Free-space notifications are sparse and controlled by `Pipe::SyncPeriod`.

## Examples

### TileData Flow

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_tiledata(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    VecTile tile;

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);

    // No TFREE is required here. TileData TPOP already handles free-space notification.
}
```

### No-op API Symmetry

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename Pipe>
AICORE void example_noop(Pipe &pipe)
{
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
}
```

### GlobalData Slot Release

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
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M / 2, N>, Stride<1, 1, 1, N, 1>>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;

    TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
    // Load or otherwise consume data from slot here.
    TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```

## ASM Form Examples

The current public assembly reference does not define a stable PTO-AS spelling for `TFREE`. Use the C++ intrinsic form for manual CV FIFO programming.
