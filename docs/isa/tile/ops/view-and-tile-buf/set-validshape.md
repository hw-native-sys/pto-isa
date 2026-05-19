# pto.set_validshape

`pto.set_validshape` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Update the runtime `v_row` / `v_col` metadata on an existing **dynamic** rank-2 tile buffer (allocated with `valid=?x?`).

## Mechanism

A tile buffer can be allocated with dynamic valid shape (`valid=?x?`). At allocation time the valid region is unspecified; `pto.set_validshape` writes the runtime `valid_row` / `valid_col` values into the descriptor so subsequent tile-level ops (loads, stores, compute) honor a well-defined valid window.

The op updates metadata only — it does NOT move data and does NOT change the physical storage layout. The static `R x C` shape of the tile is unchanged; only the valid sub-region inside that shape is updated.

## Syntax

```mlir
pto.set_validshape %src, %valid_row, %valid_col : !pto.tile_buf<vec, RxCxT, valid=?x?>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src` | `!pto.tile_buf<vec, RxCxT, valid=?x?>` | Dynamic rank-2 tile buffer (both valid dims dynamic). |
| `%valid_row` | `index` | Runtime valid row count. |
| `%valid_col` | `index` | Runtime valid column count. |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form has no SSA result; it updates the tile descriptor in place. |

## Constraints

!!! warning "Constraints"
    - `%src` MUST be rank-2 and use `v_row = ?` and `v_col = ?` on both dimensions.
    - Tile programs use `pto.tile_buf`; memref forms are a lowering artifact and are not part of this surface.
    - Constant `valid_row` / `valid_col` MUST be non-negative and `<=` the tile's static shape bounds.

## Examples

```mlir
%src = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
pto.set_validshape %src, %vr, %vc : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Allocate the dynamic tile: [pto.alloc_tile](./alloc-tile.md)
- Carve a sub-region with known size: [pto.subset](./subset.md)
