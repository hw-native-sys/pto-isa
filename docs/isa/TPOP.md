# TPOP

## Introduction

Pop a consumer tile from a `TPipe` FIFO for Cube-Vector communication.

This page describes both the TileData overload and the `GlobalData` overload for popping data from a `TPipe` FIFO.

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
template <typename Pipe, typename TileCons, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileCons>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` is typically an A2A3 `TPipe` declared in `include/pto/npu/a2a3/TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **A2A3 TileData consumer**:
    - `TileCons::Loc` must be `TileType::Vec` or `TileType::Mat`.
    - `Direction::DIR_C2V`: vector consumes data produced by cube.
    - `Direction::DIR_V2C`: cube consumes data produced by vector.
    - `Direction::DIR_BOTH`: both C2V and V2C consumers are supported by the same pipe type.
- **Local consumer buffers**:
    - For C2V vector consumers, `TPipe` assigns the tile to `C2V_CONSUMER_BUF` with local FIFO rotation.
    - For V2C matrix consumers, `TPipe` assigns the tile to `V2C_CONSUMER_BUF` with local FIFO rotation.
- **Split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No sub-vector offset is applied. On A2A3, this mode requires AIV0/AIV1 to participate in inter-core synchronization.
    - `TileSplitAxis::TILE_UP_DOWN`: vector subblocks consume row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: vector subblocks consume column halves.
- **Synchronization**:
    - Data-ready wait is performed for each `TPOP`.
    - Free-space notifications are sparse and controlled by `Pipe::SyncPeriod`.
- **GlobalData slot view**:
    - `gmTensor` is assigned to the FIFO slot base address selected by `pipe.cons.tileIndex`.
    - For C2V split modes, vector subblock offsets are applied according to `Split`.
    - The caller must call `TFREE(Pipe&, GlobalData&)` after all loads from the slot view are complete.

## Examples

### C2V Vector Pop

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
    constexpr uint32_t LocalBase = 0x0;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, LocalBase, 0x0);
    VecTile tile;

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
}
```

### V2C Matrix Pop

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
    constexpr uint32_t LocalBase = 0x0;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, M * N * sizeof(T), FifoDepth>;
    using MatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;

    Pipe pipe(fifoMem, 0x0, LocalBase);
    MatTile tile;

    TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}
```

### GlobalData Slot Pop

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
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
    TLOAD(tile, slot);
    TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```

## ASM Form Examples

The current public assembly reference does not define a stable PTO-AS spelling for `TPOP`. Use the C++ intrinsic form for manual CV FIFO programming.
