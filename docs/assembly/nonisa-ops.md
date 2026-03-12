# Non-ISA Operations

## 1. Scope

This page specifies PTO AS operations from `~/pto-isa.txt` that are not represented as ISA instruction pages.

- Level-1: SSA form, compiler-managed allocation/synchronization.
- Level-2: DPS form, explicit buffer reuse and synchronization primitives.

## 2. View Operations

### 2.1 `make_tensor_view`

```text
// L1
%dst = pto.make_tensor_view %ptr, shape = [sh1,sh2,sh3,sh4,sh5] strides = [st1,st2,st3,st4,st5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>
```

### 2.2 `partition_view`

```text
// L1
%dst = pto.partition_view %src, offsets = [of1,of2,of3,of4,of5], sizes = [sh1,sh2,sh3,sh4,sh5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype> -> !pto.partition_tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>
```

## 3. Tile Allocation

### 3.1 `alloc_tile` (static parameters)

```text
// L2
%dst = pto.alloc_tile : !pto.tile_buf<loc, dtype, rows, cols, v_row, v_col, blayout, slayou, fractal, pad>
```

### 3.2 `alloc_tile` (dynamic valid region)

```text
// L2
%dst = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<loc, dtype, rows, cols, v_row=?, v_col=?, blayout, slayou, fractal, pad>
```

## 4. Kernel Parameter Queries

### 4.1 `get_block_idx`

```text
// L1 / L2
%idx = pto.get_block_idx
```

### 4.2 `get_subblock_idx`

```text
// L1 / L2
%idx = pto.get_subblock_idx
```

### 4.3 `get_block_num`

```text
// L1 / L2
%num = pto.get_block_num
```

### 4.4 `get_subblock_num`

```text
// L1 / L2
%num = pto.get_subblock_num
```

## 5. Pointer and Scalar Access

### 5.1 `addptr`

```text
// L2
%ptr_new = pto.addptr %ptr, %offset
```

### 5.2 `tgetval`

```text
// L2
pto.tgetval ins(%src, %index : !pto.tile_buf<...>, dtype) outs(%val : dtype)
```

### 5.3 `tsetval`

```text
// L2
pto.tsetval ins(%index, %val : dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## 6. Synchronization Primitives (Level-2)

### 6.1 `record_event`

```text
pto.record_event[src_op, dst_op, eventID]
```

Note: this is a low-level synchronization primitive. For the current PTO-DSL
front-end flow, prefer sync-free source and let `ptoas --enable-insert-sync`
insert the corresponding synchronization.

Supported ops in current table: `TLOAD`, `TSTORE_ACC`, `TSTORE_VEC`, `TMOV_M2L`, `TMOV_M2S`, `TMOV_M2B`, `TMOV_M2V`, `TMOV_V2M`, `TMATMUL`, `TVEC`.

### 6.2 `wait_event`

```text
pto.wait_event[src_op, dst_op, eventID]
```

Note: this is a low-level synchronization primitive. For the current PTO-DSL
front-end flow, prefer sync-free source and let `ptoas --enable-insert-sync`
insert the corresponding synchronization.

Supported ops in current table: `TLOAD`, `TSTORE_ACC`, `TSTORE_VEC`, `TMOV_M2L`, `TMOV_M2S`, `TMOV_M2B`, `TMOV_M2V`, `TMOV_V2M`, `TMATMUL`, `TVEC`.

### 6.3 `barrier`

```text
pto.barrier(op)
```

Supported ops in current table: `TVEC`, `TMATMUL`.

## 7. Consistency Notes

- Non-ISA PTO AS operations are documented in this section and are intentionally separate from `docs/isa/` manifest-driven instruction entries.
- `TSYNC` instruction pages in `docs/isa/` remain the canonical ISA-level synchronization semantics.
- When table content in `~/pto-isa.txt` changes, this page should be updated in the same change set.
