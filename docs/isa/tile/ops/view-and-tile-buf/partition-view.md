# pto.partition_view

`pto.partition_view` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Carve a `!pto.partition_tensor_view<...>` out of a parent `!pto.tensor_view<...>` by specifying per-dimension offsets and sizes. Logical sub-window only — no allocation, no data movement.

## Mechanism

`result = source[offsets, sizes]`. The operation captures both static and dynamic shape information into the result partition descriptor. Downstream tile-level ops (e.g., `pto.tload`, `pto.tstore`) can consume the partition view directly without re-deriving offsets at every call site.

The op is pure: it does not touch memory and does not change the parent view.

## Syntax

```mlir
%pv = pto.partition_view %tv, offsets = [%o0, %o1], sizes = [%s0, %s1]
    : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` | Input tensor view. |
| `offsets` | `Variadic<index>` | Dynamic offsets along each dimension. |
| `sizes` | `Variadic<index>` | Dynamic sizes (extents) of the partition. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%pv` | `!pto.partition_tensor_view<...>` | Logical partition descriptor. |

## Constraints

!!! warning "Constraints"
    - `offsets` and `sizes` operand counts MUST each match the rank of `%tv`.
    - The op is pure; it does not allocate memory or move data.
    - Out-of-bounds combinations of `offsets + sizes` against the parent shape are target-defined.

## Examples

```mlir
// 16x16 tile starting at (%off0, %off1) inside a 1024x512 view.
%pv = pto.partition_view %tv, offsets = [%off0, %off1], sizes = [%s0, %s1]
    : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Construct the parent view: [pto.make_tensor_view](./make-tensor-view.md)
- Extract the underlying address: [pto.tensor_view_addr](./tensor-view-addr.md)
