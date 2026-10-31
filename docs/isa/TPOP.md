# TPOP

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->


## Introduction

Pop a consumer tile from a `TPipe` FIFO for Cube-Vector communication.

This page describes the TileData overloads with an explicit `TileSplitAxis`, an explicit subblock ID, or reversed parameters without `Split`, plus the `GlobalData` overload.

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

template <typename Pipe, typename TileCons, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, int32_t subBlockId,
                          WaitEvents &... events);

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPOP(TileData &tile, Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` is typically a `TPipe` declared by the selected backend's `TPush.hpp`, including
`include/pto/cpu/TPush.hpp`, `include/pto/npu/a2a3/TPush.hpp`, or `include/pto/npu/a5/TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **TileData consumer**:
    - `TileCons::Loc` must be `TileType::Vec`, `TileType::Mat`, or `TileType::Ctrl`.
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
- **A6 split behavior**:
    - Only `TileSplitAxis::TILE_NO_SPLIT` is supported, because A6 hardware allows one AIC to communicate with one AIV.
- **NPU synchronization**:
    - Data-ready wait is performed for each `TPOP`.
    - Free-space notifications are sparse and controlled by `Pipe::SyncPeriod`.
- **GlobalData slot view**:
    - `gmTensor` is assigned to the FIFO slot base address selected by `pipe.cons.tileIndex`.
    - For C2V split modes, vector subblock offsets are applied according to `Split`.
    - The caller must call `TFREE(Pipe&, GlobalData&)` after all loads from the slot view are complete.
- **CPU_SIM FIFO model**:
    - `TPOP` waits for a committed producer slot using the host FIFO's mutex and condition variable.
    - TileData payloads are loaded from storage owned by the host FIFO state, even when `TPipe` carries a non-null NPU GM workspace. The workspace is not accessed by the CPU_SIM TileData flow.
    - If the destination's `data()` returns null, CPU_SIM allocates separate thread-local backing storage; an already bound destination keeps its storage. This applies to both split and no-split TileData flows.
    - The separately allocated storage is registered with the calling thread's memory model. `TASSIGN` accepts its base and interior pointers, checks the complete view extent, and creates a shared-data alias. The alias does not own the allocation. See [TPOP-backed tile views](../coding/cpu_sim.md#tpop-backed-tile-views).
    - Split modes select the lane from the current subblock context. For a C2V pipe configured with `IsNoSplit`, `TILE_NO_SPLIT` coordinates one or two vector consumer subblocks according to the runtime subblock count. In other no-split vector cases, a nonzero inactive lane is zero-filled where required.
    - The overload with an explicit `int32_t subBlockId` is not currently implemented by CPU_SIM; use the simulated subblock execution context to select a split lane.
    - For `DIR_BOTH`, the consumer waits in its direction's independent ring. No-split and V2C consumers take the oldest committed slot by producer commit order, preserving FIFO order when delayed frees leave gaps in the ring. Overlapping pops reserve distinct slots; `TFREE` releases the calling consumer's oldest outstanding pop in the corresponding direction.
    - CPU_SIM releases a concrete occupied slot through each enabled `TFREE`; it does not model the NPU `SyncPeriod` notification cadence.
    - TileData payloads are read back with the consumer tile's declared shape, so that shape must match the window
      the producer pushed (its valid shape under `TILE_NO_SPLIT`).
    - CPU_SIM copies TileData by logical coordinates into the destination layout. A5 local-FIFO `TPOP` instead binds the tile to the FIFO address; it does not convert layouts. For NPU-equivalent V2C behavior, use matching producer and consumer layouts, such as NZ Vec to NZ Mat, and perform any required layout conversion before `TPUSH`.
    - The GlobalData overload is not currently available in CPU_SIM.

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
