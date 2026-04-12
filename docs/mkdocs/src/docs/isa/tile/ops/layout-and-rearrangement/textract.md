<!-- Generated from `docs/isa/tile/ops/layout-and-rearrangement/textract.md` -->

# pto.textract

Standalone reference page for `pto.textract`. This page belongs to the [Layout And Rearrangement](../../layout-and-rearrangement.md) family in the PTO ISA manual.

## Summary

Extract a sub-tile from a source tile.

## Mechanism

Extract a smaller sub-tile from a larger source tile. It belongs to the tile surface and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Conceptually copies a smaller window starting at `(indexRow, indexCol)` from the larger `src` tile into `dst`. Exact mapping depends on tile layouts.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `indexRow` is the starting row offset in `src`.
- `indexCol` is the starting column offset in `src`.
- `dst` names the destination tile. The operation iterates over dst's valid region.
- `fp` (optional for TEXTRACT_FP): scaling tile for vector quantization.
- `reluMode` (optional): specifies ReLU mode.
- `preQuantScalar` (optional): scalar for pre-quantization.

## Expected Outputs

`dst` holds the extracted sub-tile from `src` at position (indexRow, indexCol), with optional conversion.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

### General constraints / checks

- `DstTileData::DType` must equal `SrcTileData::DType`.

- Supported element types: `int8_t`, `half`, `bfloat16_t`, `float`.

- In GEMV scenarios targeting `TileType::Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.

- Supported element types: `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t`.

- In GEMV scenarios targeting `Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.

- Destination supports `TileType::Mat -> TileType::Left/Right/Scale`, `TileType::Acc -> TileType::Mat` (including relu, scalar-quant, and vector-quantized forms), and specific `TileType::Vec -> TileType::Mat` extraction paths.

- The vector-quantized form additionally requires an `FpTileData` scaling operand, matching the `TEXTRACT_FP(...)` interface.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- Runtime bounds checks:
  - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
  - `indexCol + DstTileData::Cols <= SrcTileData::Cols`

### A2A3 implementation checks

- Source layout must satisfy one of the checked A2A3 extraction layouts:
  - `(SFractal == ColMajor && isRowMajor)`, or
  - `(SFractal == RowMajor && !isRowMajor)`.

- Destination must be `TileType::Left` or `TileType::Right` with a target-supported fractal configuration.

### A5 implementation checks

- Source layout must satisfy one of the checked A5 extraction layouts:
  - for `Left` / `Right`: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`
  - for `ScaleLeft`: `(SFractal == RowMajor && isRowMajor)`
  - for `ScaleRight`: `(SFractal == ColMajor && !isRowMajor)`

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Next op in family: [pto.textract_fp](./textract-fp.md)
