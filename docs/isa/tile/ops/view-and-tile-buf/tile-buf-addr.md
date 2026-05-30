# pto.tile_buf_addr

`pto.tile_buf_addr` is part of the [View and Tile Buffer](../../view-and-tile-buf.md) instruction set.

## Summary

Extract the data-region address of a `!pto.tile_buf<...>` as either a typed PTO pointer (`!pto.ptr<T, space>`) or a memref view. **This op is the boundary between tile-buffer instructions and pointer-based vector instructions.**

## Mechanism

Inside a `pto.vecscope` / `pto.strict_vecscope` body, vector load/store ops (`pto.vlds`, `pto.vsts`, etc.) consume typed pointers, not tile handles. `pto.tile_buf_addr` materializes a `vec`-space pointer (or memref) from a tile handle allocated outside the scope so vector-scope code can read and write the same on-chip data the tile-level code prepared.

The op is pure: it does not move data, does not allocate, and does not participate in pipeline synchronization. During lowering it typically becomes a no-op or an attribute-driven address constant.

## Syntax

```mlir
%ub_ptr = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> !pto.ptr<T, vec>
%ub_ref = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> memref<...>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%tile` | `!pto.tile_buf<...>` (or tile-bound memref form) | Tile handle whose data-region address is taken. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%ub_ptr` / `%ub_ref` | `!pto.ptr<T, space>` or `memref<...>` | Typed pointer (e.g., `!pto.ptr<f32, vec>`) or memref view of the tile's data region. Memref results use the tile's static shape and address space; pointer results use the tile's element type and memory space. |

## Constraints

!!! warning "Constraints"
    - Result MUST be either a typed PTO pointer or a memref view; no other result types are accepted.
    - When a memref result is requested, the lowered form uses the tile's static shape and address space.
    - `pto.tile_buf_addr` is **only legal inside `pto.vecscope` / `pto.strict_vecscope`**.
    - Outside a vector scope, tile handles MUST be consumed by tile-level ops (`pto.tload`, `pto.tstore`, `pto.tadd`, …) rather than by address extraction.
    - Conversely, tile-level ops MUST NOT appear inside `pto.vecscope`.

## Examples

```mlir
%tile = pto.alloc_tile addr = %c0_i64 valid_row = %r
  : !pto.tile_buf<vec, 8x128xf32, valid=?x?>

pto.vecscope {
  %ub = pto.tile_buf_addr %tile
    : !pto.tile_buf<vec, 8x128xf32, valid=?x?> -> !pto.ptr<f32, vec>
  // ... vector-scope loads/stores on %ub ...
}
```

## Relationship to Tile vs Micro Surfaces

| Surface | Consumes | Bridge |
|---|---|---|
| **Tile** (`pto.t*`) | `!pto.tile_buf<...>` | — |
| **Micro / vector** (`pto.v*`, `pto.vlds`, `pto.vsts`) | `!pto.ptr<T, space>` | `pto.tile_buf_addr` |

The micro side is fenced by `pto.vecscope`. Inside that scope, `pto.tile_buf_addr` is the only legal way to obtain a pointer from a tile handle. Outside the scope, vector ops are illegal and tile ops own the tile handle exclusively.

## Related Ops / Instruction Set Links

- Instruction set overview: [View and Tile Buffer](../../view-and-tile-buf.md)
- Allocate the tile: [pto.alloc_tile](./alloc-tile.md)
- Vector execution scope: [pto.vecscope](../../../scalar/ops/micro-instruction/vecscope.md)
- Vector loads/stores that consume the resulting pointer: [Vector Load Store](../../../vector/vector-load-store.md)
- Sister op for tensor views: [pto.tensor_view_addr](./tensor-view-addr.md)
