# pto.subview

`pto.subview` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Reinterpret a tile as a subtile of another tile. This establishes a view relationship without copying data.

## Mechanism

`rowIdx` and `colIdx` are zero-based offsets into the valid region of `src` for the top-left corner of `dst`.

For each element `(i, j)` in the valid region of `dst`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{rowIdx} + i,\mathrm{colIdx} + j} $$

On A2/A3 and A5, this operation is purely a metadata operation that establishes a view relationship between `dst` and `src` within the tile register file; no data movement occurs. On the CPU simulator, the operation similarly records the offset in the tile descriptor.

No synchronization edges are created.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = pto.subview %src, %row_idx, %col_idx : !pto.tile<...>, i16, i16
```

### AS Level 1 (SSA)

```text
%dst = pto.subview %src, %row_idx, %col_idx : (!pto.tile<...>, i16, i16) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.subview ins(%src, %row_idx, %col_idx : !pto.tile_buf<...>, i16, i16) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent SUBVIEW(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `src` | Source tile |
| `rowIdx` | Zero-based row offset into the valid region of `src` for `dst`'s top-left corner |
| `colIdx` | Zero-based column offset into the valid region of `src` for `dst`'s top-left corner |
| `dst` | Destination tile that views a sub-rectangle of `src` |

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Constraints

Enforced by `SUBVIEW_IMPL`:

- **Tile type must match**: `TileDataSrc::Loc == TileDataDst::Loc`
- **Both tiles must have the same static capacity**: `TileDataSrc::Rows == TileDataDst::Rows` and `TileDataSrc::Cols == TileDataDst::Cols`
- **Both tiles must have the same BLayout**: `TileDataSrc::BFractal == TileDataDst::BFractal`
- **Source valid region must contain destination**: The source tile's `validRow` (`validCol`) must be at least as big as the destination tile's `validRow` (`validCol`)

## Cases That Are Not Allowed

- Creating a subview where `rowIdx + dst.validRow > src.validRow` or `colIdx + dst.validCol > src.validCol`.
- Mismatched tile types or layouts between `src` and `dst`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64>;
  using Dst = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 2, 32>;

  Src src;
  Dst dst0, dst1, dst2, dst3;

  // Split into four 2x32 subtiles
  SUBVIEW(dst0, src, 0, 0);
  SUBVIEW(dst1, src, 0, 32);
  SUBVIEW(dst2, src, 2, 0);
  SUBVIEW(dst3, src, 2, 32);
}
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Previous op: [pto.set_img2col_padding](./set-img2col-padding.md)
- Next op: [pto.get_scale_addr](./get-scale-addr.md)
