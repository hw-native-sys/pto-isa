# pto.textract

`pto.textract` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

`TEXTRACT` pulls a sub-tile out of a larger source tile, starting at a configurable `(indexRow, indexCol)` offset. It is the natural counterpart to `TINSERT` — where `TINSERT` writes into a destination, `TEXTRACT` reads out of a source.

Two variants are documented here:

| Variant | Suffix | Description | Typical Use |
|---------|---------|-------------|-------------|
| Standard extract | *(none)* | Extract a sub-window with optional ReLU | `Mat→Left/Right/Scale`, `Acc→Mat` |
| Fix-pipe extract | `_fp` | Extract through the fix-pipe quantization path | `Acc→Mat` with per-channel scaling |

## Mechanism

Conceptually copies a window of size `(DstRows, DstCols)` starting at `(indexRow, indexCol)` from the source tile into the destination. For `0 <= i < DstRows` and `0 <= j < DstCols`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

**Fix-pipe variant** (`TEXTRACT_FP`): Routes through the hardware fix-pipe quantization pipeline. The `indexCol` offset also serves as an index into the `fp` tile's quantization parameter array — the backend offsets the FPC configuration address by `indexCol` before setting up the fix-pipe:

$$ \mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j};\ \mathrm{fp}\right) $$

## Variants

### Variant 1: Standard Extract

`TEXTRACT(dst, src, indexRow, indexCol)` — plain sub-tile extraction.

### Variant 2: ReLU Extract

`TEXTRACT<..., reluMode>(dst, src, indexRow, indexCol)` — extract with ReLU pre-processing.

### Variant 3: Scalar-Quant Extract

`TEXTRACT<..., reluMode>(dst, src, preQuantScalar, indexRow, indexCol)` — extract with a scalar quantization parameter.

### Variant 4: Fix-Pipe Extract (`TEXTRACT_FP`)

`TEXTRACT_FP(dst, src, fp, indexRow, indexCol)` — extract through the fix-pipe quantization path. The `_fp` suffix means **fix pipe**, not floating point.

## Supported Tile-Type Pairs

### A2/A3

| Source Type | Destination Type | Notes |
|------------|----------------|-------|
| `TileType::Mat` | `TileType::Left/Right/Scale` | MX block-format extraction |
| `TileType::Acc` | `TileType::Mat` | Accumulator-to-matrix extraction |

**Layout requirements (A2/A3):** Source layout must satisfy either:
- `(SFractal == ColMajor && isRowMajor)`, or
- `(SFractal == RowMajor && !isRowMajor)`

### A5

| Source Type | Destination Type | Notes |
|------------|----------------|-------|
| `TileType::Mat` | `TileType::Left/Right/ScaleLeft/ScaleRight` | Extended MX formats |
| `TileType::Acc` | `TileType::Mat` | Accumulator-to-matrix extraction |

**Layout requirements (A5):**
- `Left/Right`: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`
- `ScaleLeft`: `(SFractal == RowMajor && isRowMajor)`
- `ScaleRight`: `(SFractal == ColMajor && !isRowMajor)`

## Supported Element Types

`int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t`

## Syntax

### PTO Assembly Form

Standard extract:

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

Fix-pipe extract:

```text
%dst = textract.fp %src, %fp[%r0, %r1] : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
// Standard
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>

// Fix-pipe
%dst = pto.textract_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
// Standard
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)

// Fix-pipe
pto.textract_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Variant 1: Plain extract
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src,
                              uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents &... events);

// Variant 2: ReLU extract
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src,
                              uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

// Variant 3: Scalar-quant extract
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src,
                              uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

// Variant 4: Fix-pipe extract (TEXTRACT_FP)
template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                                 uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

## Inputs

- `src` — the source tile to extract from.
- `indexRow` — starting row offset within `src`. The extraction window begins here.
- `indexCol` — starting column offset within `src`. In fix-pipe variants, this also indexes into the `fp` tile's parameter array.
- `dst` — the destination tile. The operation writes the extracted sub-window here.
- `fp` (fix-pipe variant only) — auxiliary fix-pipe tile. Must be `TileType::Scaling`.
- `reluMode` (optional) — `ReluPreMode::{NoRelu, NormalRelu}`.
- `preQuantScalar` (scalar-quant variant only) — scalar quantization factor.

## Constraints

- **Runtime bounds**: `indexRow + DstTileData::Rows <= SrcTileData::Rows` and `indexCol + DstTileData::Cols <= SrcTileData::Cols`
- **Type match**: `DstTileData::DType` must equal `SrcTileData::DType`
- **Fp tile location**: `FpTileData::Loc` must be `TileType::Scaling` (A2/A3 and A5 both enforce this via `static_assert`)
- **Fix-pipe path**: The backend offsets the FPC address by `indexCol` (counted in units of the fp tile's element width) before configuring the fix-pipe

## Common Patterns

### Pattern 1: Extract Left Block from Matrix (GEMM Setup)

```cpp
// Extract the Left block (rows 0-63, cols 0-63) from a wider matrix
using MatT = Tile<TileType::Mat, float, 64, 256, BLayout::RowMajor, 64, 256, SLayout::ColMajor>;
using LeftT = TileLeft<float, 64, 64>;

MatT mat;
LeftT left;
TASSIGN(mat, 0x1000);

// Extract the 64x64 block starting at row=0, col=0
TEXTRACT(left, mat, /*indexRow=*/0, /*indexCol=*/0);
```

### Pattern 2: Sliding Window Extraction

```cpp
// Extract overlapping windows from a large feature map tile
using MatT = Tile<TileType::Mat, half, 32, 128>;
using LeftT = TileLeft<half, 32, 32>;

MatT mat;
LeftT window;
TASSIGN(mat, 0x2000);

// Slide a 32x32 window across columns 0, 32, 64, 96
TEXTRACT(window, mat, 0, 0);   // col block 0
TEXTRACT(window, mat, 0, 32);  // col block 1
TEXTRACT(window, mat, 0, 64);  // col block 2
TEXTRACT(window, mat, 0, 96);  // col block 3
```

### Pattern 3: Accumulator Extraction with Quantization

```cpp
// Extract from accumulator tile to matrix tile, applying fix-pipe quantization
using AccT = TileAcc<float, 32, 64>;
using MatT = Tile<TileType::Mat, int8_t, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
using FpT = Tile<TileType::Scaling, uint64_t, 1, 64>;

AccT acc;
MatT mat;
FpT fp(64);  // 64 scale factors (one per output channel column)

TASSIGN(acc, 0x1000);
TASSIGN(mat, 0x2000);
TASSIGN(fp, 0x3000);

// Extract a 16x16 window starting at row=0, col=16 (into channel columns 16-31)
TEXTRACT_FP(mat, acc, fp, /*indexRow=*/0, /*indexCol=*/16);
```

## See Also

- [Layout And Rearrangement](../../layout-and-rearrangement.md)
- [pto.tinsert](./tinsert.md) — the inverse operation (write into a destination tile)
- [pto.tinsert_fp](./tinsert.md) — fix-pipe insertion variant (merged into `tinsert.md`; see Variant 4 there)
- [Assembly Spelling And Operands](../../../syntax-and-operands/assembly-model.md)
