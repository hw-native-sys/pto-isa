# TGATHER


## Tile Operation Diagram

![TGATHER tile operation](../figures/isa/TGATHER.svg)

## Introduction

Gather/select elements using either an index tile or a compile-time mask pattern.

## Math Interpretation

Index-based gather (conceptual):

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}\!\left[\mathrm{indices}_{i,j}\right] $$

`indices[i,j]` is a flat element offset from `src0.data()`, measured in source elements, not bytes or row numbers; physical padding contributes to the offset. The caller must provide nonnegative indices within the allocated source tile and select initialized data. The implementation does not check index bounds.

### 64-bit pattern gather output (A5)

Each valid source row selects columns `offset, offset + factor, ...` below `C = src.GetValidCol()`:

| Pattern | factor | offset |
| --- | ---: | ---: |
| `P0101` / `P1010` | 2 | 0 / 1 |
| `P0001` / `P0010` / `P0100` / `P1000` | 4 | 0 / 1 / 2 / 3 |
| `P1111` | 1 | 0 |

The selected count per row is `N = C > offset ? ceil((C - offset) / factor) : 0`. Rows are concatenated at `dst.data()` into `src.GetValidRow() * N` consecutive elements, without inserting padding according to `DstTileData::Cols`. The caller must provide sufficient contiguous physical capacity and a valid shape that represents this packed output. No writes occur when the source valid row count is zero or `N == 0`.

For example, source physical shape `[2,8]`, valid shape `[2,5]` and `P0101` select columns 0, 2, 4 per row, producing 6 consecutive elements. A destination with physical shape `[1,8]` and valid shape `[1,6]` holds this output; its last 2 padding elements remain unchanged.

## Assembly Syntax

Index-based gather:

```text
%dst = tgather %src0, %indices : !pto.tile<...> -> !pto.tile<...>
```

Mask-pattern gather:

```text
%dst = tgather %src {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

### Index-based Gather

```cpp
template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

### Mask-pattern Gather

```cpp
template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern = MaskPattern::P1111,
          auto gatherType = GatherAxis::GATHER_ROW, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

### Comparison-based Gather (TGather_cmp)

Gather indices of elements that satisfy a comparison condition with a scalar threshold value per row.

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp, CmpMode cmpMode, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS &src0, TileDataS1 &k_value, TileDataC &cdst, TileDataTmp &tmp, int offset, WaitEvents &... events);
```

For each row `i` in `src0`, compare each element `src0[i, j]` against threshold `k_value[i]` using `cmpMode` (GT or EQ). The indices of matching elements are gathered into `dst[i]`. The count of matches per row is stored in `cdst[i]`. The `offset` parameter specifies the starting index value.

#### Comparison-based Gather Constraints

- **A2A3**:
    - `TileDataD::DType` must be `int32_t` or `uint32_t`.
    - `TileDataS::DType` must be `float`, `half`, or `int32_t` (EQ mode only).
    - `TileDataS1::DType` must be `int32_t` or `uint32_t`.
    - `cmpMode` must be `CmpMode::GT` or `CmpMode::EQ`.
- **A5**:
    - `TileDataD::DType` must be `int32_t` or `uint32_t`.
    - `TileDataS::DType` must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, or `float`.
    - `TileDataS1::DType` must be `uint16_t` or `uint32_t`.
    - `cmpMode` must be `CmpMode::GT` or `CmpMode::EQ`.

## Constraints

- **Index-based gather: implementation checks (A2A3)**:
    - `sizeof(DstTileData::DType)` must be 2 or 4 bytes (b16/b32).
    - `sizeof(Src1TileData::DType)` must be 4 bytes (b32: `int32_t`, `uint32_t`).
    - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
    - `TmpTileData::DType` must be the same type as `Src1TileData::DType`.
    - `src1.GetValidCol() == TmpTileData::Cols` and `src1.GetValidRow() == TmpTileData::Rows`.
    - `dst.GetValidCol() == DstTileData::Cols` (continuous dst storage).
- **Index-based gather: implementation checks (A5)**:
    - `sizeof(DstTileData::DType)` must be 1, 2, 4, or 8 bytes: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - `Src1TileData::DType` must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`.
    - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
    - 64-bit data requires 32-bit indices (`int32_t` or `uint32_t`).
    - The 64-bit indexed path supports RowMajor/ColMajor destination and index tiles using their physical row/column strides; valid columns may be smaller than physical columns. The index valid region must cover the destination valid region.
- **Mask-pattern gather: implementation checks (A2A3)**:
    - Source element size must be `2` or `4` bytes.
    - `SrcTileData::DType`/`DstTileData::DType` must be `int16_t` or `uint16_t` or `int32_t` or `uint32_t`
    or `half` or `bfloat16_t` or `float`.
    - `dst` and `src` must both be `TileType::Vec` and row-major.
    - `sizeof(dst element) == sizeof(src element)` and `dst.GetValidCol() == DstTileData::Cols` (continuous dst storage).
- **Mask-pattern gather: implementation checks (A5)**:
    - Source element size must be `1`, `2`, `4`, or `8` bytes.
    - `dst` and `src` must both be `TileType::Vec` and row-major.
    - `SrcTileData::DType`/`DstTileData::DType` must be `int8_t` or `uint8_t` or `int16_t` or `uint16_t` or `int32_t` or `uint32_t`
    or `int64_t` or `uint64_t` or `half` or `bfloat16_t` or `float` or `float8_e4m3_t`or `float8_e5m2_t` or `hifloat8_t`.
    - `sizeof(dst element) == sizeof(src element)`. Output is a continuous stream; destination capacity must be sufficient, and its valid shape does not truncate writes.
    - The 64-bit pattern path supports only `GatherAxis::GATHER_ROW`; see Math Interpretation for output length and packing.
- **Comparison-based gather: implementation checks**: type and `cmpMode` constraints are detailed in [C++ Built-in Interface → Comparison-based Gather Constraints](#comparison-based-gather-constraints).
- **Bounds / validity**:
    - Index bounds are not validated by explicit runtime assertions; out-of-range indices are target-defined.
- **Temporary tile**:
    - **Index-based gather (A2A3)**: The C++ API requires an explicit `tmp` tile. `TileDataTmp::DType` must be the same type as `TileDataS1::DType` (`int32_t` or `uint32_t`). `src1.GetValidRow() == TileDataTmp::Rows` and `src1.GetValidCol() == TileDataTmp::Cols`. The tmp tile holds intermediate `vmuls` results used by `vgather` for b16 source types; for b32 source types, the result is written directly to `dst` but the API still requires `tmp`.
    - **Index-based gather (A5)**: The `tmp` tile is accepted and ignored. A5 hardware handles index-based gather without a scratch buffer.
    - **Comparison-based gather (A2A3)**: The C++ API requires an explicit `tmp` tile that serves as a combined scratch buffer for three internal regions:
        1. **cmpsTmp** (comparison result bitmap): offset 0, stored as `uint8_t`, size = `TileDataTmp::Rows × TileDataTmp::Cols` bytes.
        2. **indexTmp** (index array): offset = `TileDataTmp::Rows × TileDataTmp::Cols × sizeof(uint8_t)`, stored as `TileDataD::DType`, size = `TileDataS::Rows × TileDataS::Cols × sizeof(TileDataD::DType)` bytes.
        3. **cvtTmp** (converted k-value array): offset = `TileDataTmp::Rows × TileDataTmp::Cols × sizeof(uint8_t)` + `TileDataS::Rows × TileDataS::Cols × sizeof(TileDataD::DType)`, stored as `TileDataS::DType`, size = `TileDataS::Rows × sizeof(TileDataS::DType)` bytes.
        The minimum tmp size (in bytes) must satisfy:
        $$ \text{tmpSize} \ge \text{Rows}_\text{tmp} \times \text{Cols}_\text{tmp} + \text{Rows}_\text{src} \times \text{Cols}_\text{src} \times \text{sizeof(DType}_\text{dst}\text{)} + \text{Rows}_\text{src} \times \text{sizeof(DType}_\text{src}\text{)} $$
    - **Comparison-based gather (A5)**: The `tmp` tile is accepted and ignored. A5 hardware handles comparison-based gather without a scratch buffer.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src0;
  IdxT idx;
  DstT dst;
  IdxT tmp;
  TGATHER(dst, src0, idx, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 128>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
