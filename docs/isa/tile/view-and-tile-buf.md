# View and Tile Buffer

The view-and-tile-buffer operations are the foundation of the PTO tile programming model. They cover four concerns:

1. **Build descriptors** for global tensors (shape + strides + base pointer) — `pto.make_tensor_view`
2. **Query descriptors** for runtime shape/stride information — `pto.get_tensor_view_dim`, `pto.get_tensor_view_stride`
3. **Partition** a global descriptor into logical sub-windows — `pto.partition_view`
4. **Manage on-chip tile buffers** (allocate, sub-set, set valid shape, extract pointer) — `pto.alloc_tile`, `pto.subset`, `pto.set_validshape`, `pto.tile_buf_addr`, `pto.tensor_view_addr`

All these ops are pure descriptor/handle manipulation: none moves data, allocates memory at runtime, or participates in pipeline synchronization. They establish the addressing and lifetime contract that the tile compute ops (`pto.tload`, `pto.tstore`, `pto.tadd`, `pto.tmatmul`, …) and the vector micro ops (`pto.vlds`, `pto.vsts`, …) consume.

## Per-Op Pages

### Tensor View — Global Memory Descriptors

- [pto.make_tensor_view](./ops/view-and-tile-buf/make-tensor-view.md) — Build a tensor view from a pointer, shape, and strides
- [pto.get_tensor_view_dim](./ops/view-and-tile-buf/get-tensor-view-dim.md) — Read a dimension extent
- [pto.get_tensor_view_stride](./ops/view-and-tile-buf/get-tensor-view-stride.md) — Read an element-stride
- [pto.tensor_view_addr](./ops/view-and-tile-buf/tensor-view-addr.md) — Project the underlying address (memref or `!pto.ptr<T, gm>`)
- [pto.partition_view](./ops/view-and-tile-buf/partition-view.md) — Carve a partition window from a tensor view

### Tile Buffer — On-Chip Storage

- [pto.alloc_tile](./ops/view-and-tile-buf/alloc-tile.md) — Declare a new tile buffer lifetime
- [pto.subset](./ops/view-and-tile-buf/subset.md) — Strided sub-region of a parent tile
- [pto.set_validshape](./ops/view-and-tile-buf/set-validshape.md) — Update runtime valid shape on a dynamic tile
- [pto.tile_buf_addr](./ops/view-and-tile-buf/tile-buf-addr.md) — **Tile↔vector bridge:** extract a typed pointer inside `pto.vecscope`

## Tile ↔ Vector Bridge

The `!pto.tile_buf<...>` type belongs to the tile surface. Vector micro instructions consume typed pointers `!pto.ptr<T, space>`. The only legal bridge between the two surfaces is [`pto.tile_buf_addr`](./ops/view-and-tile-buf/tile-buf-addr.md), and it is only valid **inside** a [`pto.vecscope`](../scalar/ops/micro-instruction/vecscope.md) region. Outside `pto.vecscope`, tile handles can only be passed to tile-level ops; inside `pto.vecscope`, tile-level ops are illegal and the vector-scope code must work through the pointer obtained from `pto.tile_buf_addr`.

This split is what makes the two surfaces composable without ambiguity.

## Related Material

- [Tile ISA Reference](./README.md) — Tile instruction inventory
- [Memory and Data Movement](./memory-and-data-movement.md) — Tile-level GM ↔ tile DMA
- [Vector Execution Scope (`pto.vecscope`)](../scalar/ops/micro-instruction/vecscope.md) — Where `pto.tile_buf_addr` is legal
- [Pointer Operations](../scalar/ops/micro-instruction/pointer-operations.md) — `pto.addptr` / `pto.castptr` / `pto.load_scalar` / `pto.store_scalar`
