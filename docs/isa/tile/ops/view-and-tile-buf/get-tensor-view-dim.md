# pto.get_tensor_view_dim

`pto.get_tensor_view_dim` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Return the runtime size of a specific dimension from a `!pto.tensor_view<...>`.

## Mechanism

The op reads the extent of dimension `%idx` from the descriptor produced by [`pto.make_tensor_view`](./make-tensor-view.md). It is pure: no memory access, no synchronization, no architectural side effects.

## Syntax

```mlir
%dim = pto.get_tensor_view_dim %tv, %idx : !pto.tensor_view<...> -> index
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` | Logical tensor view. |
| `%idx` | `index` | Dimension index (0-based). |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dim` | `index` | Runtime size of dimension `%idx`. |

## Constraints

!!! warning "Constraints"
    - `%idx` MUST be in `[0, rank(%tv))`.
    - The op is pure; it does not modify the view or any underlying memory.

## Examples

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
%w = pto.get_tensor_view_dim %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Construct a view: [pto.make_tensor_view](./make-tensor-view.md)
- Query stride: [pto.get_tensor_view_stride](./get-tensor-view-stride.md)
