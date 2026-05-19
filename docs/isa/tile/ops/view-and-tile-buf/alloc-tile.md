# pto.alloc_tile

`pto.alloc_tile` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Declare the lifetime of a `!pto.tile_buf<...>`. Each call produces an **independent** tile-buffer instance.

## Mechanism

A tile buffer is a bounded, rectangular 2-D region of on-chip memory (UB / L1 / L0A / L0B / L0C / BT / scaling buffer) with an explicit lifetime. `pto.alloc_tile` introduces a fresh SSA value standing for one such instance and lets the implementation decide on the concrete address — or accepts an explicit address via the optional `addr` clause.

When the result tile type has dynamic `v_row` / `v_col` (`valid=?x?`), the corresponding `valid_row` / `valid_col` operands must be supplied at allocation time so downstream ops see a well-defined valid region.

The op is pure: no data movement, no synchronization.

## Syntax

```mlir
%tb  = pto.alloc_tile : !pto.tile_buf<...>
%tb2 = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<vec, RxCxT, valid=?x?>
%tb3 = pto.alloc_tile addr = %ad : !pto.tile_buf<...>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `addr` | `Optional<i64>` | Optional explicit start address. If omitted, assigned by the implementation. |
| `valid_row` | `Optional<index>` | Dynamic valid-row count. Required when the result type has `v_row = ?`. |
| `valid_col` | `Optional<index>` | Dynamic valid-col count. Required when the result type has `v_col = ?`. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%tb` | `!pto.tile_buf<...>` | Newly allocated tile buffer instance. |

## Constraints

!!! warning "Constraints"
    - If result `v_row` / `v_col` are dynamic (`?`), the corresponding operands MUST be present.
    - If result `v_row` / `v_col` are static, the corresponding operands MUST be absent.
    - Each call produces an **independent** tile-buffer instance, even when called repeatedly with the same arguments.

## Examples

```mlir
%tb = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
```

```mlir
// Dynamic valid shape: must pass valid_row / valid_col.
%tb = pto.alloc_tile valid_row = %vr valid_col = %vc
    : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Update dynamic valid shape: [pto.set_validshape](./set-validshape.md)
- Carve a sub-region: [pto.subset](./subset.md)
- Bridge to vector load/store: [pto.tile_buf_addr](./tile-buf-addr.md)
