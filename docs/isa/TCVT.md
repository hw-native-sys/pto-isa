# TCVT


## Tile Operation Diagram

![TCVT tile operation](../figures/isa/TCVT.svg)

## Introduction

Elementwise type conversion with a specified rounding mode.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode}}\!\left(\mathrm{src}_{i,j}\right) $$

where `rmode` is a rounding policy (see `pto::RoundMode`).

## Assembly Syntax

Synchronous form:

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          WaitEvents &... events);
```

## Constraints

- `dst` and `src` must be compatible in shape/valid region as required by the implementation.
- The conversion `(src element type) -> (dst element type)` must be supported by the target for the given `RoundMode`.
- **Implementation notes (A2A3/A5)**:
    - One form accepts an explicit `SaturationMode`, and the specified saturation behavior is forwarded directly to the implementation.
    - The other form omits `SaturationMode`; in that case, the implementation chooses a target-defined default saturation behavior for the specific type pair.
    - On CPU, only the form without explicit `SaturationMode` is currently implemented.
- **Temporary tile**:
  - The C++ API provides overloads with an explicit `tmp` tile. On A2A3 this tmp tile is consumed by PyTorch-compatible non-saturating narrowing paths when `SaturationMode::OFF` is used for `float -> int16`, `half -> int16`, or `half -> int8`. Other conversions do not require tmp space.
  - The implementation casts `tmp` to `int32_t *`; size the tile by bytes, independent of the declared `TmpTileData::DType`.
  - The formulas below give the minimum allocation size rounded to the 32-byte vector block granularity used by the implementation. If `C = 0`, no tmp-backed conversion is issued and the required tmp size is `0`.
  - **Common parameters**:
    - `R = dst.GetValidRow()`.
    - `C = dst.GetValidCol()`.
    - `SS = TileDataS::RowStride`, in source elements.
    - `REPEAT_MAX = 255`, `REPEAT_BYTE = 256`, `BLOCK_BYTE_SIZE = 32`.
  - **`float -> int16`, non-saturating (`SaturationMode::OFF`)**:
    - The temporary result is an `int32_t` tile produced by the first `float -> int32` conversion step.
    - `float` source rows are 32-byte aligned by the tile constraints, so `SS / 8` is the source repeat stride in 32-byte blocks.
    - For the aligned main region, one call processes one row and up to `REPEAT_MAX` repeats, with `64` elements per repeat:
    $$ \text{tmpHeadBytes} = 4 \times 64 \times \min\left(\left\lfloor\frac{C}{64}\right\rfloor, 255\right) $$
    - For the tail region, one call processes up to `REPEAT_MAX` rows using the source row stride. The extent is computed in 32-byte blocks because the vector repeat stride is block-based:
    $$ \text{tmpTailBytes} =
    \begin{cases}
    32 \times \left((\min(R, 255) - 1) \times \frac{SS}{8} + \left\lceil\frac{C \bmod 64}{8}\right\rceil\right), & C \bmod 64 > 0 \\
    0, & C \bmod 64 = 0
    \end{cases} $$
    - Minimum required tmp size for this path:
    $$ \text{tmpFloatToInt16Bytes} = \max(\text{tmpHeadBytes}, \text{tmpTailBytes}) $$
    - A compact full-repeat upper bound for the main region is `REPEAT_MAX * REPEAT_BYTE = 65280` bytes, but tail sizing can be larger when `SS` is large because tail rows are written with source-row stride.
  - **`half -> int16`, non-saturating (`SaturationMode::OFF`)**:
    - The implementation processes each row in sub-chunks of at most `64` elements and reuses the same temp buffer for every sub-chunk. For `C > 0`, let:
    $$ H = \min(C, 64) $$
    - Minimum required tmp size for this path:
    $$ \text{tmpHalfToInt16Bytes} = 32 \times \left\lceil\frac{H}{8}\right\rceil $$
    - A shape-independent upper bound for any non-empty tile is `256` bytes.
  - **`half -> int8`, non-saturating (`SaturationMode::OFF`)**:
    - The implementation also processes sub-chunks of at most `64` elements and reuses the same 256-byte temp region. The first step can write up to `64` `int32_t` values into bytes `[0, 255]`; after the `int32 -> int16` narrow, bytes `[0, 127]` hold the `int16_t` values and bytes `[128, 255]` are reused as scratch.
    - `tempMaskBuf = tempAndBuf + 64` advances by `64 * sizeof(int16_t) = 128` bytes, so it points at the upper half of the same 256-byte temp region. It does not require an additional 256-byte allocation.
    - Minimum required tmp size for this path:
    $$ \text{tmpHalfToInt8Bytes} = \max\left(32 \times \left\lceil\frac{H}{8}\right\rceil,\ 128 + 32 \times \left\lceil\frac{H}{16}\right\rceil\right) $$
    - A shape-independent upper bound for any non-empty tile is `256` bytes.
  - **Overall minimum for all tmp-backed TCVT conversions**:
    - Since `tmpHalfToInt8Bytes >= tmpHalfToInt16Bytes`, the minimum tmp size that fits all tmp-backed TCVT conversion paths for the same shape is:
    $$ \text{tmpSizeAllBytes} = \max(\text{tmpFloatToInt16Bytes},\ \text{tmpHalfToInt8Bytes}) $$
    - Equivalently, if the tile is non-empty and a compact shape-independent bound for the half paths is acceptable:
    $$ \text{tmpSizeAllBytes} = \max(\text{tmpFloatToInt16Bytes},\ 256) $$
  - The no-`tmp` overload remains valid for conversions that do not need the PyTorch-compatible tmp-backed path, or when native saturation behavior is sufficient.

## Supported Conversions (Side-by-Side: A2A3 vs A5)

| Source Type | A2A3 Destinations | A5 Destinations | Difference |
|---|---|---|---|
| FP32 | FP16, FP32 (round-only), BF16, I16, I32, I64 | FP32, FP16, BF16, I16, I32, I64, FP8_E4M3, FP8_E5M2, H8 | A5 adds FP8/H8 targets |
| FP16 | FP32, I32, I16, I8, U8, S4 (int4b_t) | FP32, I32, I16, I8, U8, H8 | A2A3 has S4 path; A5 has H8 path |
| BF16 | FP32, I32 | FP32, I32, FP16, FP4_E1M2X2, FP4_E2M1X2 | A5 adds FP16/FP4 targets |
| I16 | FP16, FP32 | U8, FP16, FP32, U32, I32 | A5 adds U8/U32/I32 targets |
| I32 | FP32, I16, I64, FP16 (deq path) | FP32, I16, U16, I64, U8 | A2A3 supports I32 -> FP16 (half, deq); A5 does not |
| I64 | FP32, I32 | FP32, I32 | Same |
| U8 | FP16 | FP16, U16 | A5 adds U16 target |
| I8 | FP16 | FP16, I16, I32 | A5 adds I16/I32 targets |
| S4 (int4b_t) | FP16 | N/A | A2A3-only |
| U32 | N/A | U8, U16, I16 | A5-only source type |
| FP8_E4M3 | N/A | FP32 | A5-only source type |
| FP8_E5M2 | N/A | FP32 | A5-only source type |
| H8 | N/A | FP32 | A5-only source type |
| FP4_E1M2X2 | N/A | BF16 | A5-only source type |
| FP4_E2M1X2 | N/A | BF16 | A5-only source type |

Notes:
- Key gap: A2A3 supports I32 -> FP16 (half) via deq path, while A5 has no I32 -> FP16 conversion.
- On A5, FP16 -> FP8_E4M3 and FP16 -> FP8_E5M2 are not supported.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
