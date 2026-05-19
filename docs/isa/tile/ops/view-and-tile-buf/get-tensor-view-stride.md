# pto.get_tensor_view_stride

`pto.get_tensor_view_stride` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Return the logical stride of a specific dimension, measured in **elements** (not bytes), from a `!pto.tensor_view<...>` (or its lowered memref form).

## Mechanism

The op reads the per-dimension stride from the descriptor produced by [`pto.make_tensor_view`](./make-tensor-view.md). It is pure: no memory access, no synchronization.

Because the stride is reported in **elements**, downstream pointer arithmetic computed via [`pto.addptr`](../../../scalar/ops/micro-instruction/pointer-operations.md) (which is also element-based) composes cleanly without an additional `sizeof(T)` multiply.

## Syntax

```mlir
%stride = pto.get_tensor_view_stride %tv, %idx : !pto.tensor_view<...> -> index
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` or memref form | Tensor view or its lowered memory-reference form. |
| `%idx` | `index` | Dimension index (0-based). |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%stride` | `index` | Element-stride of dimension `%idx`. |

## Constraints

!!! warning "Constraints"
    - `%idx` MUST be in `[0, rank(%tv))`.
    - The returned stride is counted in **elements**, not bytes. Mixing element-stride and byte-offset values without an explicit `sizeof(T)` conversion is a bug.
    - The op is pure; it does not modify the view or any underlying memory.

## Examples

```mlir
// Stride of the leading dim (rows) for a row-major view.
%s0 = pto.get_tensor_view_stride %tv, %c0 : !pto.tensor_view<?x?xf32> -> index

// Stride of the inner dim is 1 (one element).
%s1 = pto.get_tensor_view_stride %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Construct a view: [pto.make_tensor_view](./make-tensor-view.md)
- Query dim size: [pto.get_tensor_view_dim](./get-tensor-view-dim.md)
- Element-offset pointer arithmetic: [pto.addptr](../../../scalar/ops/micro-instruction/pointer-operations.md)
