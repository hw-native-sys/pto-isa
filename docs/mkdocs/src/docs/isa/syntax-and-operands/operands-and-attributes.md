<!-- Generated from `docs/isa/syntax-and-operands/operands-and-attributes.md` -->

# Operands And Attributes

PTO VISA operations work over a small set of operand kinds: tiles, global-memory views, scalars, predicates, and synchronization values. Attributes and modifiers refine operation behavior, but they do not replace operand legality.

## Operand Kinds

PTO defines seven operand kinds. Each kind maps to a specific SSA type and has distinct legality rules:

| Kind | SSA Type | C++ API | Description |
|------|----------|---------|-------------|
| **Tile** | `!pto.tile<...>` / `!pto.tile_buf<...>` | `Tile<TileType, DType, Rows, Cols, ...>` | Tile operand with shape, layout, valid-region metadata |
| **GlobalTensor** | `!pto.partition_tensor_view<...>` / `!pto.memref<...>` | `GlobalTensor<DType, Shape, Stride, Layout>` | GM-facing view; the source or destination of data movement |
| **Scalar** | `i8`–`i64`, `u8`–`u64`, `f16`, `bf16`, `f32` | Built-in C++ types | Immediate values or runtime-computed scalars |
| **Predicate** | `!pto.mask` | (IR-level) | Per-lane mask controlling which lanes participate in vector ops |
| **Event** | `!pto.event` | `RecordEvent` (return type) | Synchronization token; carries ordering information between operations |
| **UB Pointer** | `!pto.ptr<T, ub>` | (IR-level) | Pointer into Unified Buffer; used by vector load/store and DMA copy ops |
| **GM Pointer** | `!pto.ptr<T, gm>` | `__gm__ T*` | Pointer into Global Memory; used by scalar load/store and DMA copy ops |

## Operand Kind Details

### Tile Operands

Tile operands carry shape, layout, valid-region, and location-intent metadata. They are the primary payload type for `pto.t*` operations.

**SSA tile type signature**:
```
!pto.tile<loc=LOC, DTYPE, ROWS, COLS, BLAYOUT, SLAYOUT, FRACTAL, PAD>
```

**Components**:
| Component | Values | Description |
|-----------|--------|-------------|
| `loc` | `vec`, `mat`, `acc`, `scalar`, `left`, `right` | Location intent / pipeline destination |
| `DTYPE` | `f16`, `bf16`, `f32`, `i8`, `u8`, ... | Element type |
| `ROWS` | positive integer | Physical row count |
| `COLS` | positive integer | Physical column count |
| `BLAYOUT` | `RowMajor`, `ColMajor` | Block storage layout |
| `SLAYOUT` | `NoneBox`, `RowMajor`, `ColMajor` | Stripe layout |
| `FRACTAL` | `None`, `NZ`, `ZN`, `FR`, `RN` | Fractal encoding |
| `PAD` | `Zero`, `Null`, `Invalid` | Padding value |

### GlobalTensor Operands

`GlobalTensor` operands describe a view of GM storage. They pair a pointer with shape and stride metadata.

**SSA partition_tensor_view type**:
```
!pto.partition_tensor_view<BxHxWxRxCxdtype>
```
This is always 5D: batch, height, width, tile rows, tile columns.

### Scalar Operands

Scalar operands are immediate values encoded directly in the instruction or computed at runtime. They appear as:

- 32-bit integer or float immediates in assembly
- `i32`, `i64`, `f32` in SSA form
- Standard C++ types in C++ intrinsics

### Predicate Operands

Predicate operands (`!pto.mask`) control which lanes participate in vector operations. They are produced by predicate-generation operations (`pset_b8`, `pge_b32`, `plt_b16`, etc.) and consumed by vector operations.

A predicate with all bits set means "all lanes active". A predicate with some bits cleared means "only those lanes participate".

### UB Pointer Operands

UB pointer operands (`!pto.ptr<T, ub>`) specify addresses within the on-chip Unified Buffer. They are used by:

- Vector load/store (`vlds`, `vsld`, `vgather2`, `vsts`, `vsst`, `vscatter`)
- DMA copy operations (`copy_gm_to_ubuf`, `copy_ubuf_to_gm`)

### GM Pointer Operands

GM pointer operands (`!pto.ptr<T, gm>`) specify addresses in off-chip Global Memory. They are used by:

- Scalar load/store (`load_scalar`, `store_scalar`)
- DMA copy operations

## Attributes

Attributes modify the behavior of an operation without changing its operand types. Every attribute MUST have a documented value domain, and invalid attribute values MUST produce deterministic diagnostics.

### Compare Attributes

Used by `pto.tcmp`, `pto.vcmp`, and related comparison operations:

| Attribute | Values | Description |
|-----------|--------|-------------|
| `cmp` | `"eq"`, `"ne"`, `"lt"`, `"le"`, `"gt"`, `"ge"` | Comparison predicate mode |
| `cmpS` | (same) | Scalar compare variant: compares each element against an immediate |

### Rounding Mode Attributes

Used by conversion and narrowing operations:

| Attribute | Values | Description |
|-----------|--------|-------------|
| `rnd` | `"rne"`, `"rz"`, `"rp"`, `"rm"` | Rounding mode: nearest-even, zero, positive-infinity, negative-infinity |

### Atomic Mode Attributes

Used by `pto.tstore`:

| Attribute | Values | Description |
|-----------|--------|-------------|
| `atomic` | `"none"`, `"add"`, `"max"`, `"min"` | Atomic store mode |

### Transform Mode Attributes

Used by `pto.timg2col`, `pto.textract`, `pto.tinsert`:

| Attribute | Values | Description |
|-----------|--------|-------------|
| `mode` | `"hw"`, `"wh"`, `"cubic"`, ... | Transform mode; domain depends on operation |

### Matmul Phase Attributes

Used by `pto.tmatmul`:

| Attribute | Values | Description |
|-----------|--------|-------------|
| `phase` | `"relu"`, `"none"` | Post-matmul activation phase |

### Distribution Mode Attributes

Used by vector load/store (`pto.vlds`, `pto.vsts`):

| Attribute | Values | Description |
|-----------|--------|-------------|
| `dist` | `"NORM"`, `"BRC_B8/B16/B32"`, `"US_B8/B16"`, `"DS_B8/B16"`, `"UNPK_B8/B16/B32"`, `"DINTLV_B32"`, `"SPLT2CHN_B8/B16"`, `"SPLT4CHN_B8"` | Distribution mode |

### Mask Attributes

Used by vector load with alignment-state update:

| Attribute | Values | Description |
|-----------|--------|-------------|
| `mask` | `"POST_UPDATE"`, `"NO_POST_UPDATE"` | Whether to update alignment state after masked store |

## Operand Constraint Rules

### Tile Operand Constraints

For a binary tile operation `optile(dst, src0, src1)`:

1. **Type compatibility**: `dtype(src0) == dtype(src1) == dtype(dst)` (unless `TCVT` which explicitly changes dtype)
2. **Shape compatibility**: `shape(src0) == shape(src1) == shape(dst)` (no implicit broadcasting)
3. **Layout compatibility**: The combination of `BLayout`, `SLayout`, and `Fractal` MUST match the instruction family's requirements
4. **Location intent**: Source and destination location intents MUST be compatible with the instruction (e.g., matmul requires `Left` + `Right` → `Acc`)

### GlobalTensor Operand Constraints

For `TLOAD(tile, tensor)`:

1. **Dtype size**: `sizeof(tile.dtype) == sizeof(tensor.dtype)`
2. **Layout compatibility**: The `tensor.Layout` (ND/DN/NZ) MUST be compatible with `tile.TileType` and `tile.SLayout`
3. **Positive dimensions**: All shape dimensions MUST be > 0

### Predicate Operand Constraints

For a masked vector operation `opvec(result, src, mask)`:

1. **Mask width**: The mask width MUST match the vector width of the operation
2. **Mask production/consumption**: A predicate MUST be produced by a predicate-generation op before being consumed

### Immediate/Scalar Constraints

1. **Range**: Immediate values MUST be within the representable range of their declared type
2. **Shift amounts**: Shift amounts MUST be non-negative and less than the element bit-width
3. **Broadcast**: Scalar operands may be broadcast to match tile/vector shape; this is explicit in the operation (e.g., `tadds`)

## Rule Example

If an instruction accepts a tile plus a scalar mode attribute, legality still depends on both:

- whether the tile tuple is legal
- whether the attribute value is in the documented domain

A legal tile does not make an illegal modifier acceptable, and a valid modifier does not repair an illegal tile tuple.

## Contract Notes

- Every required attribute MUST define an allowed value domain.
- Invalid attribute values MUST produce deterministic diagnostics.
- Operand roles and attribute meaning MUST stay aligned across intrinsics, PTO-AS, and per-op reference pages.
- There is no implicit type promotion; a type mismatch between operands is always illegal unless an explicit conversion operation (`TCVT`, `vcvt`) is present.
- Broadcasting is explicit: `tadds` broadcasts the scalar operand to match the tile shape; `tadd` does not broadcast.

## See Also

- [Assembly Model](./assembly-model.md)
- [Type System](../state-and-types/type-system.md)
- [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md)
- [GlobalTensor And Data Movement](../programming-model/globaltensor-and-data-movement.md)
- [Instruction Contract Template](../reference/format-of-instruction-descriptions.md)
