# pto.make_tensor_view

`pto.make_tensor_view` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Construct a global tensor view from a base pointer, runtime shape, and runtime strides. No allocation, no data movement — purely descriptor construction.

## Mechanism

A `!pto.tensor_view<...>` is a logical descriptor that carries a base pointer, per-dimension extents, per-dimension element strides, and an optional layout hint. Tile-level ops (`pto.tload`, `pto.tstore`, `pto.partition_view`, …) consume these views as their source-of-truth for global memory addressing.

`pto.make_tensor_view` packages those four pieces of information into a single SSA value. The op is pure: it materializes the descriptor only and does not touch memory.

## Syntax

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xT>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%ptr` | `!pto.ptr<T, gm>` (or matching pointer) | Source pointer; element type must match the result. |
| `shape` | `Variadic<index>` | Dynamic shape dimensions, one entry per result rank. |
| `strides` | `Variadic<index>` | Dynamic strides, counted in **elements** (not bytes), one entry per result rank. |
| `layout` (attr, optional) | `LayoutAttr` | `nd` / `dn` / `nz` hint. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` | Logical view descriptor. |

## Constraints

!!! warning "Constraints"
    - `%ptr` element type must match the result element type.
    - `shape` and `strides` operand counts must match the tensor_view rank.
    - If `layout` is provided with static shapes/strides, it must be consistent with the inferred layout.

## Examples

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Query view dim: [pto.get_tensor_view_dim](./get-tensor-view-dim.md)
- Query view stride: [pto.get_tensor_view_stride](./get-tensor-view-stride.md)
- Extract address: [pto.tensor_view_addr](./tensor-view-addr.md)
- Partition a view: [pto.partition_view](./partition-view.md)
