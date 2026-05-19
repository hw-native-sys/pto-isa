# pto.subset

`pto.subset` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Create a strided view of a parent tile buffer at runtime offsets with static sizes — `result = source[offsets] sizes [rows, cols]`. No data movement.

## Mechanism

`pto.subset` produces a child `!pto.tile_buf<...>` that aliases a sub-region of the parent. The runtime `%i`, `%j` give the top-left corner of the sub-region; the static `sizes` attribute fixes the result extents.

Boxed-layout tile buffers (e.g., fractal NZ tiles on the cube path) carry extra alignment constraints derived from their inner box shape; for them, the subset must align with the box. Non-boxed layouts apply no additional structural checks beyond the parent's element type and address space.

The op is pure: no allocation, no memory movement.

## Syntax

```mlir
%sub = pto.subset %src[%i, %j] sizes [rows, cols] : !pto.tile_buf<...>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src` | `!pto.tile_buf<...>` | Parent tile buffer. |
| `offsets` | `Variadic<index>` | Runtime offsets `[i, j]`. |
| `sizes` (attr) | `I64ArrayAttr` | Static shape `[rows, cols]`. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%sub` | `!pto.tile_buf<...>` | Strided sub-view of the parent. Element type, address space, and tile config are inherited from `%src`; `valid_shape` is derived from the parent valid shape and constant offsets when possible. |

## Constraints

!!! warning "Constraints"
    - Boxed-vs-non-boxed behavior is derived from the source's tile config (`blayout`, `slayout`, `fractal`) and element type.
    - For non-boxed layouts (`slayout=none_box`), no additional subset-specific structural checks are enforced.
    - For boxed layouts:
        - `sizes` MUST have length 2 and both subset sizes MUST be positive.
        - Subset sizes MUST be multiples of the inferred inner boxed shape.
        - `offsets` MUST have length 2; constant offsets MUST be non-negative and multiples of the inferred inner boxed shape.
        - Source tile shape MUST be statically known.
        - For boxed row-major tiles: subset MUST keep the full source column extent, and the column offset MUST be the constant `0`.
        - For boxed col-major tiles: subset MUST keep the full source row extent, and the row offset MUST be the constant `0`.
    - The inferred result reuses the source's element type, address space, and tile config. `valid_shape` is derived from the parent valid shape and constant offsets, or dynamic when offsets are dynamic.

## Examples

```mlir
%sub = pto.subset %src[%i, %j] sizes [32, 32]
     : !pto.tile_buf<vec, 64x64xf16>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Allocate the parent tile: [pto.alloc_tile](./alloc-tile.md)
- Tile-level reinterpretation (different op): [pto.subview](../sync-and-config/subview.md)
- Extract a pointer (vector scope): [pto.tile_buf_addr](./tile-buf-addr.md)
