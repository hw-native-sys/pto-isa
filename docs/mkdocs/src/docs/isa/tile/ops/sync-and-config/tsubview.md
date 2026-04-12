<!-- Generated from `docs/isa/tile/ops/sync-and-config/tsubview.md` -->

# pto.tsubview

Standalone reference page for `pto.tsubview`. This page belongs to the [Sync And Config](../../sync-and-config.md) family in the PTO ISA manual.

## Summary

Reinterpret a tile as a subtile of another tile.

## Mechanism

Reinterpret a tile as a subtile of another tile. It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

- `rowIdx`: in the valid region of `src`, the starting row index of the `dst` subtile.
- `colIdx`: in the valid region of `src`, the starting column index of the `dst` subtile.

For each element `(i, j)` in the valid region of `dst`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{rowIdx} + i,\mathrm{colIdx} + j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tsubview %src, %row_idx, %col_idx : !pto.tile<...>, i16, i16
```

### AS Level 1 (SSA)

```text
%dst = pto.tsubview %src, %row_idx, %col_idx : (!pto.tile<...>, i16, i16) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsubview ins(%src, %row_idx, %col_idx : !pto.tile_buf<...>, i16, i16) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tsubview %src, %row_idx, %col_idx : (!pto.tile<...>, i16, i16) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tsubview ins(%src, %row_idx, %col_idx : !pto.tile_buf<...>, i16, i16) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBVIEW(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx, WaitEvents&... events);
```

## Inputs

- `src` provides the source tile.
- `rowIdx` and `colIdx` are zero-based offsets into the valid region of `src` for the top-left corner of `dst`.
- `dst` names the destination tile that views a sub-rectangle of `src`.

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Side Effects

This operation may establish a synchronization edge, bind or configure architectural tile state, or update implementation-defined configuration that later tile instructions consume.

## Constraints

Enforced by `TSUBVIEW_IMPL`:

- **Tile type must match**: `TileDataSrc::Loc == TileDataDst::Loc`.

- **Both tiles must have the same static capacity**: `TileDataSrc::Rows == TileDataDst::Rows` and `TileDataSrc::Cols == TileDataDst::Cols`.

- **Both tiles must have the same BLayout**: `TileDataSrc::BFractal == TileDataDst::BFractal`.

- **The source tile's validRow (validCol) is at least as big as the destination tile's validRow (validCol)**

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tsubview` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64>;
  using Dst = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 2, 32>;

  Src src;
  Dst dst0;
  Dst dst1;
  Dst dst2;
  Dst dst3;

  // e.g. split into four 2x32 subtiles
  TSUBVIEW(dst0, src, 0, 0);
  TSUBVIEW(dst1, src, 0, 32);
  TSUBVIEW(dst2, src, 2, 0);
  TSUBVIEW(dst3, src, 2, 32);
}
```

### Auto And Manual Kernels

The C++ intrinsic is the same whether the surrounding kernel uses Auto scheduling or Manual pipeline control. What changes is how `TLOAD`, `TSTORE`, `TSYNC`, and related edges are scheduled around the view; see [Auto Vs Manual](../../../programming-model/auto-vs-manual.md).

### PTO-AS Form

Concrete mnemonic spelling, attribute order, and register-like operand syntax live in the PTO-AS specification (`docs/assembly/PTO-AS.md` and `docs/assembly/PTO-AS.bnf`). This ISA page names the operation as `pto.tsubview` and the logical operands above.

## Related Ops / Family Links

- Family overview: [Sync And Config](../../sync-and-config.md)
- Previous op in family: [pto.tset_img2col_padding](./tset-img2col-padding.md)
- Next op in family: [pto.tget_scale_addr](./tget-scale-addr.md)
