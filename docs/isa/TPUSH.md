# TPUSH

## Introduction

Push a producer tile into a `TPipe` FIFO for Cube-Vector communication.

This page describes all `TPUSH` overloads for pushing data into a `TPipe` FIFO: the TileData overload with explicit `TileSplitAxis`, the simplified TileData overload (reversed parameters, no Split), the GlobalTensor overload, and the TConfig-based overload.

## Operation Semantics

For the TileData overload, `TPUSH` performs three steps:

1. Wait for FIFO space when `Pipe::shouldWaitFree(pipe.prod.tileIndex)` is true.
2. Store the producer tile into the current FIFO slot. In this step:
   - For Cube-to-Vector data push, the `AccTile` is pushed into the `TPipe` FIFO.
   - For Vector-to-Cube data push, the `VecTile` is pushed into the `TPipe` FIFO.
3. Record data-ready synchronization for the consumer.

The producer tile index is incremented after the FIFO slot address is computed.

For the `GlobalData` overload, `TPUSH` only records data-ready synchronization for a slot that was already allocated by `TALLOC`. It does not store tile data by itself.

For the `TConfig` overload `TPUSH(Pipe&, TileProd&, TConfig)`, the `TConfig` template parameters is used to configure fixpipe parameters from L0C->GM/UB.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileProd>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

template <typename Pipe, typename TileProd, typename TConfig, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);
```

`Pipe` is typically an `TPipe` declared in `TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **TileData producer**:
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
    - `Direction::DIR_C2V`: Cube produces an accumulator tile for vector consumption.
    - `Direction::DIR_V2C`: Vector produces a vector tile for cube consumption.
    - `Direction::DIR_BOTH`: both C2V and V2C producers are supported by the same pipe type.
- **FIFO slot**:
    - `SlotSize` must be large enough for one logical FIFO entry.
    - `SlotNum >= 1`.
- **A2A3 split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No sub-vector offset is applied. On A2A3, this mode requires AIV0 and AIV1 to participate in synchronization.
    - `TileSplitAxis::TILE_UP_DOWN`: Vector subblocks map to row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Vector subblocks map to column halves.
- **A5 split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No sub-vector offset is applied.
    - `TileSplitAxis::TILE_UP_DOWN`: Data is split into row halves. For C2V direction (L0C→UB path), this mode only supports b32 data type, and `validRows` must be a power of 2; for V2C direction (UB→L1 path), `validCols` must be a multiple of 32 bytes.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Data is split into two column halves. For C2V direction (L0C→UB path), this mode only supports b32 data type, and `validCols` must be a multiple of 32; for V2C direction (UB→L1 path), `validCols` must be a multiple of 32 bytes.
- **Simplified TileData overload**:
    - `TPUSH(TileData&, Pipe&)` uses `TileSplitAxis::TILE_NO_SPLIT` semantics internally.
    - `TileData::Loc` must be `TileType::Acc` or `TileType::Vec`.
- **TConfig overload**:
    - `TConfig` is a configuration type that determines push behavior (implementation-defined).
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
- **Synchronization**:
    - Free-space waits are sparse and controlled by `Pipe::SyncPeriod`.
    - Data-ready record is emitted for each `TPUSH`.
- **GlobalData producer**:
    - `gmTensor` must be a FIFO slot view returned by `TALLOC`.
    - Data must be written into `gmTensor` before calling `TPUSH(Pipe&, GlobalData&)`.
    - `TPUSH(Pipe&, GlobalData&)` ignores the tensor contents and only commits the FIFO slot to the consumer.
- **Tile Type Support**:
    - **TPUSH/TPOP Supported Tile Types**:
        - `TileType::Acc` (Accumulator Tile): Used by Cube core for C2V direction communication.
        - `TileType::Vec` (Vector Tile): Used by Vector core for V2C direction communication.
        - `TileType::Ctrl` (Control Tile): Used by Vector core for V2C_CTRL direction control signal transmission.

## Defining TConfig

The `TConfig` template parameter for the `TPUSH(Pipe&, TileProd&, TConfig)` overload is a configuration struct that controls fixpipe behavior during push. PTO provides the `FixpipeParams` struct for this purpose.

Declared in `include/pto/common/fixpipe.hpp`:

```cpp
template <LayoutMode_t layoutMode = LayoutMode_t::NZ2ND,
          QuantMode_t quantMode = QuantMode_t::NoQuant,
          ReluPreMode reluMode = ReluPreMode::NoRelu,
          STPhase phase = STPhase::Unspecified,
          uint8_t subBlockId = 0,
          AtomicType atomicT = AtomicType::AtomicNone,
          ClipReluMode_t clipReluMode = ClipReluMode_t::NOCLIP_RELU,
          bool isChannelSplit = false>
struct FixpipeParams {
    static constexpr LayoutMode_t LayoutMode = layoutMode;
    static constexpr QuantMode_t QuantPre = quantMode;
    static constexpr ReluPreMode ReluMode = reluMode;
    static constexpr STPhase Phase = phase;
    static constexpr uint8_t SubBlockId = subBlockId;
    static constexpr AtomicType AtomicT = atomicT;
    static constexpr ClipReluMode_t ClipReluMode = clipReluMode;
    static constexpr bool IsChannelSplit = isChannelSplit;
};
```

### TConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `LayoutMode` | `LayoutMode_t` | Output data layout: `NZ2NZ` (NZ→NZ), `NZ2ND` (NZ→row-major), `NZ2DN` (NZ→column-major). Default: `NZ2ND`. |
| `QuantPre` | `QuantMode_t` | Quantization/dequantization mode (defined in CANN). Controls the data type conversion during fixpipe push. Default: `NoQuant`. |
| `ReluMode` | `ReluPreMode` | ReLU activation mode: `NoRelu` or `NormalRelu`. Default: `NoRelu`. |
| `Phase` | `STPhase` | Store phase for unit-flag aware paths: `Unspecified`, `Partial`, or `Final`. Default: `Unspecified`. |
| `SubBlockId` | `uint8_t` | Sub-block identifier for accumulator-to-vector move mode mapping (A5 only). Default: `0`. |
| `AtomicT` | `AtomicType` | Atomic operation type for GM store: `AtomicNone` or `AtomicAdd`. Default: `AtomicNone`. |
| `ClipReluMode` | `ClipReluMode_t` | Clip ReLU mode: `NOCLIP_RELU` or `CLIP_RELU`. Default: `NOCLIP_RELU`. |
| `IsChannelSplit` | `bool` | Whether channel split is enabled. Default: `false`. |

### TConfig Usage Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_tconfig_push(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // Define TConfig: NZ→row-major layout, dequantize to half, with ReLU
    using MyConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::DEQF16, ReluPreMode::NormalRelu>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    TPUSH<Pipe, AccTile, MyConfig>(pipe, acc);
}
```

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
