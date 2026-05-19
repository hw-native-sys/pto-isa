# pto.tensor_view_addr

`pto.tensor_view_addr` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Extract the underlying address (as a typed PTO pointer or a memref view) from a `!pto.tensor_view<...>` or `!pto.partition_tensor_view<...>` descriptor. Pure op — does not move data.

## Mechanism

A tensor view carries both addressing metadata (shape, strides, base pointer) and a logical descriptor. `pto.tensor_view_addr` projects out the address side: it returns the same underlying storage exposed as either a typed GM pointer (`!pto.ptr<T, gm>`) or as a memref view.

The op is pure. During compiler-internal lowering, the operand may already be rewritten to a memref form; in that case this op is folded away or rewritten to an equivalent memref-to-ptr cast.

## Syntax

```mlir
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> memref<...>
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> !pto.ptr<T, gm>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src` | `!pto.tensor_view<...>` or `!pto.partition_tensor_view<...>` | Source view descriptor. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `memref<...>` or `!pto.ptr<T, gm>` | Underlying address in the requested form. |

## Constraints

!!! warning "Constraints"
    - The result type MUST be either the lowered memref view or a GM pointer `!pto.ptr<T, gm>` to the same underlying storage. Other result types are rejected.
    - The op is pure and does not move data.

## Examples

```mlir
// Extract a GM pointer from a tensor view, for use in DMA copy ops.
%base = pto.tensor_view_addr %tv : !pto.tensor_view<?x?xf32> -> !pto.ptr<f32, gm>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Construct a view: [pto.make_tensor_view](./make-tensor-view.md)
- Partition a view: [pto.partition_view](./partition-view.md)
- Sister op for tile-buffer addresses: [pto.tile_buf_addr](./tile-buf-addr.md)
