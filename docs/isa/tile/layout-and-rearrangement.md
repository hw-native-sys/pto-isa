# Layout And Rearrangement Instruction Set

Layout operations change how tile data is organized within the unified buffer. These are **pure data-movement operations** that do not modify element values.

## Operations

| Operation | Description | Category | C++ Intrinsic |
|-----------|-------------|----------|---------------|
| [pto.tmov](./ops/layout-and-rearrangement/tmov.md) | Move/copy tile data | Copy | `TMOV(dst, src)` |
| [pto.tmov_fp](./ops/layout-and-rearrangement/tmov.md) | Move/copy with fill/pad | Copy | `TMOV_FP(dst, src, fp)` |
| [pto.treshape](./ops/layout-and-rearrangement/treshape.md) | Change tile shape | Transform | `TRESHAPE(dst, src, newShape)` |
| [pto.ttrans](./ops/layout-and-rearrangement/ttrans.md) | Transpose tile dimensions | Transform | `TTRANS(dst, src)` |
| [pto.tconcat](./ops/layout-and-rearrangement/tconcat.md) | Concatenate tile sequences along a dimension | Transform | `TCONCAT(dst, lhs, rhs)` |
| [pto.tpack](./ops/layout-and-rearrangement/tpack.md) | Pack tile elements into a compact destination representation | Pack | `TPACK(dst, src)` |
| [pto.textract](./ops/layout-and-rearrangement/textract.md) | Extract a subtile | Extract | `TEXTRACT(dst, src, offset)` |
| [pto.textract_fp](./ops/layout-and-rearrangement/textract.md) | Extract with fill/pad | Extract | `TEXTRACT_FP(dst, src, offset, fp)` |
| [pto.tinsert](./ops/layout-and-rearrangement/tinsert.md) | Insert a subtile into a tile | Insert | `TINSERT(dst, src, offset)` |
| [pto.tinsert_fp](./ops/layout-and-rearrangement/tinsert.md) | Insert with fill/pad | Insert | `TINSERT_FP(dst, src, offset, fp)` |
| [pto.tfillpad](./ops/layout-and-rearrangement/tfillpad.md) | Fill tile padding region | Fill | `TFILLPAD(dst, fp)` |
| [pto.tfillpad_inplace](./ops/layout-and-rearrangement/tfillpad-inplace.md) | Fill padding in place | Fill | `TFILLPAD_INPLACE(dst, fp)` |
| [pto.tfillpad_expand](./ops/layout-and-rearrangement/tfillpad-expand.md) | Fill padding and expand | Fill | `TFILLPAD_EXPAND(dst, fp)` |
| [pto.timg2col](./ops/layout-and-rearrangement/timg2col.md) | Image to column transformation | Transform | `TIMG2COL(dst, src, cfg)` |

## Mechanism

### Copy (TMOV, TMOV_FP)

Copy all elements from source tile to destination tile. The FP variant additionally fills padding regions with a specified fill value.

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} $$

### Transform (TRESHAPE, TTRANS, TIMG2COL)

Change the declared shape or layout without changing which logical elements are read/written:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{index}(i,j)} $$

- `TTRANS`: swaps row and column indices: `dst[i,j] = src[j,i]`
- `TRESHAPE`: reinterprets the flat element sequence with a new `(Rows, Cols)` shape
- `TCONCAT`: joins tile element sequences along the documented dimension
- `TPACK`: repacks tile elements into the destination layout or compact representation
- `TIMG2COL`: rearranges image patches into column format for convolution lowering

### Extract/Insert (TEXTRACT, TINSERT, TEXTRACT_FP, TINSERT_FP)

Extract a sub-tile from a tile, or insert a sub-tile into a tile at a specified position `(row_offset, col_offset)`. FP variants fill padding regions with a fill value.

```
TEXTRACT: dst = src[row_offset : row_offset + dst.Rv, col_offset : col_offset + dst.Cv]
TINSERT:  dst[row_offset : row_offset + src.Rv, col_offset : col_offset + src.Cv] = src
```

### Fill (TFILLPAD, TFILLPAD_INPLACE, TFILLPAD_EXPAND)

Fill the padding region (declared tile area outside the valid region) with a specified fill value. The INPLACE variant modifies the source tile directly. The EXPAND variant additionally expands the valid region.

## Type Support by Target Profile

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| f32 (float) | Yes | Yes | Yes |
| f16 (half) | Yes | Yes | Yes |
| bf16 (bfloat16_t) | Yes | Yes | Yes |
| i8 / u8 | Yes | Yes | Yes |
| i16 / u16 | Yes | Yes | Yes |
| i32 / u32 | Yes | Yes | Yes |
| i64 / u64 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |

## Constraints

- `TRESHAPE` requires the total element count to remain unchanged: `src.Rv × src.Cv == dst.Rv × dst.Cv`.
- `TTRANS` requires square shape (`Rv == Cv`) or produces a transposed declared shape.
- `TEXTRACT` requires the sub-tile shape to divide evenly into the source tile declared shape.
- `TINSERT` requires the inserted tile to fit within the destination's declared shape.
- FP variants (`*_fp`) require a valid fill value (`fp`) compatible with the tile element type.
- `TIMG2COL` requires specific kernel/padding/stride configuration; profile-dependent behavior.

## Cases That Are Not Allowed

- **MUST NOT** `TRESHAPE` to a shape with a different total element count.
- **MUST NOT** `TEXTRACT` with offsets outside the source tile's declared shape.
- **MUST NOT** `TINSERT` such that the inserted tile extends beyond the destination's declared shape.
- **MUST NOT** use FP8 types with `TIMG2COL` on CPU simulator or A2/A3.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Basic move/copy
template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TMOV(TileDst& dst, TileSrc& src);

// Move with fill/pad
template <typename TileDst, typename TileSrc, typename FillT>
PTO_INST RecordEvent TMOV_FP(TileDst& dst, TileSrc& src, FillT fp);

// Reshape tile shape
template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TRESHAPE(TileDst& dst, TileSrc& src, ShapeRef newShape);

// Transpose
template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TTRANS(TileDst& dst, TileSrc& src);

// Extract/Insert at offset
template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TEXTRACT(TileDst& dst, TileSrc& src, int rowOffset, int colOffset);

template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TINSERT(TileDst& dst, TileSrc& src, int rowOffset, int colOffset);

// Fill padding region
template <typename TileT, typename FillT>
PTO_INST RecordEvent TFILLPAD(TileT& dst, FillT fp);

// Image to column (convolution lowering)
template <typename TileDst, typename TileSrc, typename Cfg>
PTO_INST RecordEvent TIMG2COL(TileDst& dst, TileSrc& src, Cfg cfg);
```

## See Also

- [Tile instruction set](../instruction-families/tile-families.md) — Instruction set overview
- [Tile instruction set](../instruction-families/tile-families.md) — Instruction Set description
