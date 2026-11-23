# TPUSH

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

> No `TPUSH.svg` is provided. `TPUSH` is a **FIFO control-plane** primitive; it records data-ready synchronization for the slot allocated by `TALLOC`, committing the producer slot to the consumer. It does not store tile data itself.

## Introduction

Push a producer tile into a `TPipe` FIFO for Cube-Vector communication.

This page describes all `TPUSH` overloads for pushing data into a `TPipe` FIFO: TileData overloads with an explicit `TileSplitAxis`, an explicit subblock ID, or reversed parameters without `Split`, plus the GlobalData and `TConfig` overloads.

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

For the `TConfig` overload `TPUSH<Pipe, TileProd, TConfig>(pipe, tile)`, the `TConfig` template parameter configures fixpipe conversion and layout. NPU backends apply it to their L0C→GM/UB path, while CPU_SIM writes the converted result to the host FIFO slot.

### A6R2 MaxTrack (C2V Acc→UB)

On A6R2, when `TConfig` embeds `MaxTrack` with `groupmax_en=true`, the same `TPUSH<Pipe, TileProd, TConfig>` overload also writes MaxTrack sideband tiles into the C2V UB FIFO:

1. Construct `TPipe` with the MaxTrack ctor that binds `C2V_GROUPMAX_BUF` / `C2V_ROWMAX_BUF` in addition to the usual C2V consumer buffer.
2. `TPUSH<Pipe, AccTile, FixpipeParams<..., MaxTrack<...>>>` runs Acc→UB fixpipe (`TMovCcToUb`), programs `set_fixp_max_cfg`, and stores:
   - main result into `C2V_CONSUMER_BUF` (as usual),
   - GroupMax into `C2V_GROUPMAX_BUF` when `groupmax_en`,
   - RowMax into `C2V_ROWMAX_BUF` when `groupmax_en && rowmax_en`.
3. The matching consumer uses the A6R2 MaxTrack `TPOP` overloads (`vec`+`gMax`, or `vec`+`gMax`+`rMax`), then `TFREE`.

When `groupmax_en=false`, behavior matches the existing A5/A6 C2V Acc→UB path (vec only). Implementation lives in `include/pto/npu/a6r2/TPush.hpp`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileProd>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);

template <typename Pipe, typename TileProd, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, int32_t subBlockId,
                           WaitEvents &... events);

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(TileData &tile, Pipe &pipe, WaitEvents &... events);

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

## Scalar control FIFOs

The A5 NPU `TPipe` control path supports V2C predicate reduction (`OR`/`AND`) and
C2V scalar broadcast. `CtrlPipeConfig` selects the participating AIVs and
independently enables forward and backward dependencies. The shared A6
backend supports AIV0 only. Control FIFOs use per-message release credits.
See [Scalar control FIFOs](../../../tests/npu/a5/src/st/testcase/tpushpop_ctrl/README.md) for the control-specific
template parameters, storage layout, and lifetime contract. The data-tile
constraints below continue to describe data FIFOs.

## Constraints

- **TileData producer**:
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
    - `Direction::DIR_C2V`: Cube produces an accumulator tile for vector consumption.
    - `Direction::DIR_V2C`: Vector produces a vector tile for cube consumption.
    - `Direction::DIR_BOTH`: both C2V and V2C producers are supported by the same pipe type.
- **FIFO slot**:
    - `SlotSize` must be large enough for one logical FIFO entry.
    - `SlotNum >= 1`.
- **A5 synchronization flag**:
    - `FlagID + 1` must not exceed `MAX_SYC_ID`.
    - `DIR_BOTH` and `DIR_BOTH_GM` use flags through `FlagID + 3`, so `FlagID + 3` must not exceed `MAX_SYC_ID` when both directions are used.
- **A2A3 split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No sub-vector offset is applied. On A2A3, this mode requires AIV0 and AIV1 to participate in synchronization.
    - `TileSplitAxis::TILE_UP_DOWN`: Vector subblocks map to row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Vector subblocks map to column halves.
- **A5 split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No sub-vector offset is applied.
    - `TileSplitAxis::TILE_UP_DOWN`: Data is split into row halves. For C2V direction (L0C→UB path), this mode only supports b32 data type, and `validRows` must be even (a multiple of 2). For V2C direction (UB→L1 path), vector subblocks are inserted into the upper and lower row regions.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Data is split into two column halves. For C2V direction (L0C→UB path), this mode only supports b32 data type, and `validCols` must be a multiple of 32; for V2C direction (UB→L1 path), `validCols` must be a multiple of 32 bytes.
- **A6 split behavior**:
    - Only `TileSplitAxis::TILE_NO_SPLIT` is supported, because A6 hardware allows one AIC to communicate with one AIV.
- **Simplified TileData overload**:
    - `TPUSH(TileData&, Pipe&)` uses `TileSplitAxis::TILE_NO_SPLIT` semantics internally.
    - `TileData::Loc` must be `TileType::Acc` or `TileType::Vec`.
- **TConfig overload**:
    - `TConfig` is a configuration type that determines push behavior (implementation-defined).
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
- **A6R2 MaxTrack (`groupmax_en=true`)**:
    - Supported only on A6R2 C2V UB FIFO (`Pipe::is_c2v_ub` or `Pipe::is_both`).
    - Construct `TPipe` with `TPipe(gm, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF, C2V_GROUPMAX_BUF, C2V_ROWMAX_BUF)`.
    - `groupmax_en` / `rowmax_en` are effective only when `LayoutMode` is `NZ2ND` or `NZ2DN`.
    - Result dtype after `FixpipeConsDType` must be `float`, `half`, or `bfloat16_t`.
    - `TileProd::Cols` must be a multiple of `group_num`; GroupMax row bytes must be 32B-aligned.
    - When `rowmax_en=1`, or when `groupmax_en=1` and `NZ2DN`: `srcTile` `validRow` must be `<= 128`.
    - When `groupmax_en=1` and `NZ2ND`: `N / group_num` must be `<= 16`.
    - `groupmax_en=true, rowmax_en=false` → outputs vec + gMax; `groupmax_en=true, rowmax_en=true` → vec + gMax + rMax.
    - MaxTrack push uses `TILE_NO_SPLIT` allocation/record internally.
    - GroupMax / RowMax UB addresses programmed into `set_fixp_max_cfg` are in C0 (32B) units.
- **NPU synchronization**:
    - Free-space waits are sparse and controlled by `Pipe::SyncPeriod`.
    - Data-ready record is emitted for each `TPUSH`.
- **GlobalData producer**:
    - `gmTensor` must be a FIFO slot view returned by `TALLOC`.
    - Data must be written into `gmTensor` before calling `TPUSH(Pipe&, GlobalData&)`.
    - `TPUSH(Pipe&, GlobalData&)` ignores the tensor contents and only commits the FIFO slot to the consumer.
- **CPU_SIM FIFO model**:
    - FIFO state is shared by host threads. `TPUSH` waits for a free slot and commits it with mutex and condition-variable synchronization.
    - TileData payloads use storage owned by the host FIFO state, including when `TPipe` is constructed with a non-null NPU GM workspace. CPU_SIM does not access that workspace for the TileData flow; this keeps payload lifetime covered by the same synchronization as the slot state.
    - TileData producers support `DIR_C2V`, `DIR_V2C`, and `DIR_BOTH`. A `DIR_BOTH` pipe gives C2V and V2C independent rings, each with `SlotNum` slots and its own payload storage, cursors, and synchronization state. A full ring in one direction does not consume capacity in the other. Commit sequence numbers preserve FIFO order within each direction.
    - CPU_SIM waits for the concrete slot selected by the direction's producer cursor to become free; it does not model the NPU `SyncPeriod` credit cadence.
    - Split modes select the lane from the current subblock context. `TILE_NO_SPLIT` uses one producer lane. For a C2V pipe configured with `IsNoSplit`, one producer slot can be coordinated across one or two vector consumer subblocks according to the runtime subblock count.
    - The overload with an explicit `int32_t subBlockId` is not currently implemented by CPU_SIM; use the simulated subblock execution context to select a split lane.
    - The simplified overload uses `TILE_NO_SPLIT`. The `TConfig` overload also supports the CPU fixpipe path.
    - For `TILE_NO_SPLIT`, the slot payload is laid out with the shape of the window actually transferred, that is
      the pushed tile's valid shape, so a narrow view of a wider tile keeps its rows aligned with the tile the
      consumer pops. The split axes lay the payload out with the producer tile's declared shape.
    - The GlobalData overload is not currently available in CPU_SIM.
- **Tile Type Support**:
    - **TPUSH/TPOP Supported Tile Types**:
        - `TileType::Acc` (Accumulator Tile): Used by Cube core for C2V direction communication.
        - `TileType::Vec` (Vector Tile): Used by Vector core for V2C direction communication.
        - `TileType::Ctrl` (Control Tile): Used by Vector core for V2C_CTRL direction control signal transmission.

## Defining TConfig

The `TConfig` template parameter for the `TPUSH<Pipe, TileProd, TConfig>(pipe, tile)` overload is a configuration struct that controls fixpipe behavior during push. PTO provides the `FixpipeParams` struct for this purpose.

Declared in `include/pto/common/fixpipe.hpp`:

```cpp
template <bool absEn = false, bool groupmaxEn = false, bool rowmaxEn = false,
          bool rowmaxCompareEn = false, bool rowmaxOutput = false, uint8_t groupNum = 8>
struct MaxTrack {
    static constexpr bool abs_en = absEn;
    static constexpr bool groupmax_en = groupmaxEn;
    static constexpr bool rowmax_en = rowmaxEn;
    static constexpr bool rowmax_compare_en = rowmaxCompareEn;
    static constexpr bool rowmax_output = rowmaxOutput;
    static constexpr uint8_t group_num = groupNum;
};

template <LayoutMode_t layoutMode = LayoutMode_t::NZ2ND,
          QuantMode_t quantMode = QuantMode_t::NoQuant,
          ReluPreMode reluMode = ReluPreMode::NoRelu,
          STPhase phase = STPhase::Unspecified,
          uint8_t subBlockId = 0,
          AtomicType atomicT = AtomicType::AtomicNone,
          ClipReluMode_t clipReluMode = ClipReluMode_t::NOCLIP_RELU,
          bool isChannelSplit = false,
          typename MaxTrackCfg = MaxTrack<>>
struct FixpipeParams {
    static constexpr LayoutMode_t LayoutMode = layoutMode;
    static constexpr QuantMode_t QuantPre = quantMode;
    static constexpr ReluPreMode ReluMode = reluMode;
    static constexpr STPhase Phase = phase;
    static constexpr uint8_t SubBlockId = subBlockId;
    static constexpr AtomicType AtomicT = atomicT;
    static constexpr ClipReluMode_t ClipReluMode = clipReluMode;
    static constexpr bool IsChannelSplit = isChannelSplit;
    using MaxTrackT = MaxTrackCfg;
    static constexpr bool abs_en = MaxTrackCfg::abs_en;
    static constexpr bool groupmax_en = MaxTrackCfg::groupmax_en;
    static constexpr bool rowmax_en = MaxTrackCfg::rowmax_en;
    static constexpr bool rowmax_compare_en = MaxTrackCfg::rowmax_compare_en;
    static constexpr bool rowmax_output = MaxTrackCfg::rowmax_output;
    static constexpr uint8_t group_num = MaxTrackCfg::group_num;
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
| `MaxTrackCfg` | `typename` | A6R2 MaxTrack sideband config (`MaxTrack<...>`). Default: `MaxTrack<>` (disabled). |

### MaxTrack Fields (A6R2)

| Field | Type | Description |
|-------|------|-------------|
| `abs_en` | `bool` | Absolute-value before max tracking. Default: `false`. |
| `groupmax_en` | `bool` | Emit GroupMax sideband. NZ2ND shape `[srcValidRow, srcValidCol/group_num]`; NZ2DN transposed `[srcValidCol/group_num, srcValidRow]`. Enables MaxTrack C2V path. Default: `false`. |
| `rowmax_en` | `bool` | Emit RowMax sideband (1D, validCol = src validRow). Requires `groupmax_en`. Default: `false`. |
| `rowmax_compare_en` | `bool` | Hardware rowmax compare enable for `set_fixp_max_cfg`. Default: `false`. |
| `rowmax_output` | `bool` | Hardware rowmax output enable for `set_fixp_max_cfg`. Default: `false`. |
| `group_num` | `uint8_t` | Elements per GroupMax group. Default: `8`. |

MaxTrack push defines consumer tile shapes internally from `TileProd` / `TConfig`:

- Fixpipe vec tile: NZ2ND RowMajor / NZ2DN ColMajor / else NZ (same size as Acc).
- GroupMax (RowMajor, ValidRow/ValidCol `DYNAMIC`): NZ2ND `[srcValidRow, srcValidCol/group_num]`; NZ2DN transposed `[srcValidCol/group_num, srcValidRow]`.
- RowMax: `[1, ceil32B(ProdM)]`, RowMajor, `ValidCol=DYNAMIC` (construct / TPOP-set with `srcValidRow`).

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

### A6R2 MaxTrack C2V Push

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/fixpipe.hpp>

using namespace pto;

AICORE void example_a6r2_maxtrack_push()
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint8_t kGroupNum = 8;
    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;

    using AccTile = TileAcc<float, M, N, M, N>;
    using MaxTrackCfg = MaxTrack</*abs*/false, /*gmax*/true, /*rmax*/true,
                                 /*cmp*/false, /*out*/true, kGroupNum>;
    using FixpipeConfig = FixpipeParams<
        LayoutMode_t::NZ2ND, QuantMode_t::NoQuant, ReluPreMode::NoRelu, STPhase::Unspecified, 0,
        AtomicType::AtomicNone, ClipReluMode_t::NOCLIP_RELU, false, MaxTrackCfg>;
    using Pipe = TPipe<FLAG_ID, Direction::DIR_C2V, M * N * sizeof(float), FIFO_DEPTH, 2, true>;

    constexpr uint32_t C2V_BUF = 0x0;
    constexpr uint32_t GMAX_BUF = 0x20000;
    constexpr uint32_t RMAX_BUF = 0x28000;
    Pipe pipe((__gm__ void*)(uint64_t)0x0, C2V_BUF, (uint32_t)0x0, GMAX_BUF, RMAX_BUF);

    AccTile acc;
    TASSIGN(acc, 0x0);
    // Fill acc (e.g. TMATMUL) before pushing.
    TPUSH<Pipe, AccTile, FixpipeConfig>(pipe, acc);
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
