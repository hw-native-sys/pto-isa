# TFREE

## Introduction

Release FIFO slot space.

For the TileData `TPOP` flow, on the A2A3 platform `TPOP` already performs the free-space notification step internally. Therefore the TileData-oriented `TFREE(Pipe &pipe)` interface is currently a no-op and exists only for API symmetry with the `GlobalData` flow. On the A5 platform, TFREE releases the FIFO slot space used by TPOP.

For the `GlobalData` flow, `TFREE(Pipe&, GlobalData&)` releases a FIFO slot view returned by `TPOP(Pipe&, GlobalData&)`.

## Operation Semantics

For the TileData flow:

1. `TPUSH(Pipe&, TileData&, Split)` stores the producer tile into the current FIFO slot and records data-ready synchronization for the consumer. The producer tile index is incremented after the slot address is computed.
2. `TPOP(Pipe&, TileData&, Split)` waits for the producer's data-ready synchronization and loads the current FIFO slot into the consumer tile. The consumer tile index is incremented after the slot address is computed.
3. `TFREE(Pipe&, Split)` releases FIFO slot space. On the A2A3 platform this interface is a no-op (`TPOP` already performs free-space notification internally), while on the A5 platform it releases the FIFO slot space used by `TPOP`.

For the GlobalData flow:

1. `TALLOC(Pipe&, GlobalData&)` allocates a producer FIFO slot from `TPipe` and exposes it as a `GlobalTensor` view. The producer can write data to the slot using instructions such as `TSTORE`.
2. `TPUSH(Pipe&, GlobalData&)` records data-ready synchronization for a slot already allocated by `TALLOC`, committing the FIFO slot to the consumer. It does not store tile data by itself.
3. `TPOP(Pipe&, GlobalData&)` waits for data-ready, assigns `gmTensor` to the current FIFO slot address, and increments the consumer tile index. It does not load data into a local tile and does not release the slot. The consumer can read data from the slot using instructions such as `TLOAD`.
4. `TFREE(Pipe&, GlobalData&)` releases the FIFO slot view returned by `TPOP(Pipe&, GlobalData&)`, notifying the producer that the slot space is free.

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
    - Use `TFREE(Pipe&, GlobalData&)` when the data in the popped FIFO slot is no longer needed.
    - Use TPUSH/TPOP/TFREE together for inter-core synchronization and data transfer; the size ratio between the pushed tile shape and the popped tile shape must be 1:1 or 1:2.
- **GlobalData flow**:
    - Use `TFREE(Pipe&, GlobalData&)` when the data in the popped FIFO slot is no longer needed.
    - `gmTensor` is only used to select the overload; the implementation does not read or write tensor contents.
    - Free-space notifications are sparse and controlled by `Pipe::SyncPeriod`.
    - If the size ratio is not 1:1 or 1:2 (i.e., subtile data transfer exists), use TALLOC/TPUSH/TPOP/TFREE together for inter-core synchronization and data transfer.

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
    ...  // final use of VecTile
    TFREE<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
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
