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


## Tile Operation Diagram

> No `TPOP.svg` is provided. `TPOP` is a **FIFO control-plane** primitive; it waits for data-ready, binds the consumer's `GlobalData` to the current FIFO slot address, and advances the consumer tile index. It does not load data into a local tile.

## Introduction

Pop a consumer tile from a `TPipe` FIFO for Cube-Vector communication.

This page describes the TileData overloads with an explicit `TileSplitAxis`, an explicit subblock ID, or reversed parameters without `Split`, plus the `GlobalData` overload and the A6R2 MaxTrack multi-tile overloads.

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

### A6R2 MaxTrack (C2V Acc→UB)

On A6R2, when the producer used `TPUSH` with `FixpipeParams<..., MaxTrack<groupmax_en=true, ...>>`, the consumer pops the main vec tile plus MaxTrack sidebands:

1. Wait for data-ready on the C2V FIFO (same as the ordinary TileData path).
2. Bind `vecTile` to the current `C2V_CONSUMER_BUF` slot.
3. Bind `gMaxTile` to the matching `C2V_GROUPMAX_BUF` slot (`TPOP(pipe, vec, gMax)`).
4. When RowMax was also enabled, bind `rMaxTile` to `C2V_ROWMAX_BUF` (`TPOP(pipe, vec, gMax, rMax)`).
5. Call `TFREE` to release the FIFO slot after consuming the tiles.

These overloads are compiled only when `PTO_NPU_ARCH_A6R2` is defined. Implementation lives in `include/pto/npu/a6r2/TPop.hpp`.

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

#if defined(PTO_NPU_ARCH_A6R2)
// A6R2 MaxTrack: vec + gMax  /  vec + gMax + rMax
// TConfig supplies group_num (same FixpipeParams / MaxTrack cfg as the matching TPUSH).
template <typename Pipe, typename VecTile, typename GMaxTile, typename TConfig,
          TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<VecTile> && is_tile_data_v<GMaxTile>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, VecTile &vecTile, GMaxTile &gMaxTile,
                          WaitEvents &... events);

template <typename Pipe, typename VecTile, typename GMaxTile, typename RMaxTile,
          typename TConfig, TileSplitAxis Split, bool EndRowMaxCmp = false,
          std::enable_if_t<is_tile_data_v<VecTile> && is_tile_data_v<GMaxTile> &&
                           is_tile_data_v<RMaxTile>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, VecTile &vecTile, GMaxTile &gMaxTile,
                          RMaxTile &rMaxTile, WaitEvents &... events);
#endif
```

`Pipe` is typically a `TPipe` declared by the selected backend's `TPush.hpp`, including
`include/pto/cpu/TPush.hpp`, `include/pto/npu/a2a3/TPush.hpp`, `include/pto/npu/a5/TPush.hpp`,
or the A6R2 MaxTrack ctor in `include/pto/npu/a6r2/TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Scalar control FIFOs

The A5 NPU `TPipe` control path supports V2C predicate reduction (`OR`/`AND`) and
C2V scalar broadcast. `CtrlPipeConfig` selects the participating AIVs and
independently enables forward and backward dependencies. The shared A6
backend supports AIV0 only. Control FIFOs use per-message release credits.
See [Scalar control FIFOs](../../tests/npu/a5/src/st/testcase/tpushpop_ctrl/README.md) for the control-specific
template parameters, storage layout, and lifetime contract. The data-tile
constraints below continue to describe data FIFOs.

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
- **A6R2 MaxTrack overloads**:
    - Available only when `PTO_NPU_ARCH_A6R2` is defined; require a C2V UB FIFO (`Pipe::is_c2v_ub` or `Pipe::is_both`).
    - `TPOP(pipe, vecTile, gMaxTile)` matches producer `groupmax_en=true, rowmax_en=false`.
    - `TPOP(pipe, vecTile, gMaxTile, rMaxTile)` matches producer `groupmax_en=true, rowmax_en=true`.
    - Producer MaxTrack requires `LayoutMode::NZ2ND` or `LayoutMode::NZ2DN`; `vecTile` must have no fractal (NZ2ND RowMajor or NZ2DN ColMajor).
    - Dtype must be `float`, `half`, or `bfloat16_t`; `vec` / `gMax` / `rMax` dtypes must match.
    - Consumer tiles must match MaxTrack push shapes: NZ2ND/NZ2DN fixpipe vec; GroupMax ValidRow/ValidCol `DYNAMIC` (NZ2ND `[srcValidRow, srcValidCol/group_num]`, NZ2DN transposed); RowMax `[1, ceil32B(ProdM)]` with `DYNAMIC` validCol. `TPOP` takes `TConfig` (provides `group_num`) and sets gMax/rMax valid shapes from `popTile` + `TConfig::group_num`.
    - After pop, call `TFREE` as on the ordinary TileData path.
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

### A6R2 MaxTrack C2V Pop

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/fixpipe.hpp>

using namespace pto;

AICORE void example_a6r2_maxtrack_pop()
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint8_t kGroupNum = 8;
    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;

    using AccTile = TileAcc<float, M, N, M, N>;
    using VecTile = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using GMaxTile = Tile<TileType::Vec, float, M, N / kGroupNum, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    constexpr int kRMaxCols =
        ((M + static_cast<int>(C0_SIZE_BYTE / sizeof(float)) - 1) / static_cast<int>(C0_SIZE_BYTE / sizeof(float))) *
        static_cast<int>(C0_SIZE_BYTE / sizeof(float));
    using RMaxTile = Tile<TileType::Vec, float, 1, kRMaxCols, BLayout::RowMajor, 1, DYNAMIC>;
    using MaxTrackCfg = MaxTrack<true, true, true, false, true, kGroupNum>;
    using FixpipeConfig = FixpipeParams<
        LayoutMode_t::NZ2ND, QuantMode_t::NoQuant, ReluPreMode::NoRelu, STPhase::Unspecified, 0, AtomicType::AtomicNone,
        ClipReluMode_t::NOCLIP_RELU, false, MaxTrackCfg>;
    using Pipe = TPipe<FLAG_ID, Direction::DIR_C2V, M * N * sizeof(float), FIFO_DEPTH, 2, true>;

    constexpr uint32_t C2V_BUF = 0x0;
    constexpr uint32_t GMAX_BUF = 0x20000;
    constexpr uint32_t RMAX_BUF = 0x28000;
    Pipe pipe((__gm__ void*)(uint64_t)0x0, C2V_BUF, (uint32_t)0x0, GMAX_BUF, RMAX_BUF);

    VecTile vecTile;
    GMaxTile gMaxTile(M, N / kGroupNum);
    RMaxTile rMaxTile(M);

    // Producer must have pushed with MaxTrack groupmax_en && rowmax_en.
    TPOP<Pipe, VecTile, GMaxTile, RMaxTile, FixpipeConfig, TileSplitAxis::TILE_NO_SPLIT>(
        pipe, vecTile, gMaxTile, rMaxTile);
    // Consume tiles...
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
}
```
