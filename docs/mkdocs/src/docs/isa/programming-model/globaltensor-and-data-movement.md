<!-- Generated from `docs/isa/programming-model/globaltensor-and-data-movement.md` -->

# GlobalTensor And Data Movement

PTO does not hide movement between global memory and local execution state. `GlobalTensor` is the architecture-visible GM-facing object, and movement operations define when data enters or leaves the local payload surfaces. This page describes the GM-facing types and the complete data paths to tile buffers and vector registers.

## GlobalTensor

### GlobalTensor Template Signature

```
GlobalTensor<DType, Shape, Stride, Layout>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `DType` | C++ type | Element type matching the target tile |
| `Shape` | `Shape<ND>()` | N-dimensional shape: `Shape<B, H, W, R, C>` |
| `Stride` | `Stride<ND>()` | Per-dimension strides in elements |
| `Layout` | enum | Memory layout: `ND` (row-major), `DN` (col-major), `NZ` (row-major fractal) |

`GlobalTensor` represents a view of `__gm__` (off-chip device) memory. It is not the storage itself — it is a descriptor that pairs a pointer with shape and stride metadata.

### GlobalTensor vs PartitionTensorView

Two GM-facing types appear in PTO programs:

| Type | Description | Usage |
|------|-------------|-------|
| `GlobalTensor` | C++ API type; wraps a `__gm__ T*` with shape/stride | C++ kernel code |
| `!pto.partition_tensor_view<MxNxdtype>` | SSA/IR type; GM partition descriptor | PTO-AS and MLIR IR |
| `!pto.memref<dtype, Nd>` | MLIR standard memref | Lowered form |

The `partition_tensor_view` describes a sub-partition of GM visible to a specific block or sub-block. Its shape is always 5D: `(B, H, W, R, C)` — batch, height, width, tile rows, tile columns.

### Supported Layouts

| Layout | Stride Pattern | Description |
|--------|---------------|-------------|
| `ND` (default) | `stride[R] = C, stride[W] = R*C, stride[H] = W*H*C, ...` | Row-major, C-contiguous |
| `DN` | `stride[C] = B, stride[R] = B*C, stride[W] = B*C*R, ...` | Column-major, Fortran-contiguous |
| `NZ` | Row-major fractal stride | Used with fractal tile layouts |

## Tile Surface Data Path

The tile surface (`pto.t*`) moves data between GM and tile buffers through MTE2/MTE3:

```
GM
  │
  │  copy via DMA engine
  ▼
UB (Unified Buffer, 256 KB on-chip)
  │
  │  implicit tile buffer fill
  ▼
Tile Buffer  ──►  Tile Compute  ──►  Tile Buffer
                                     │
                                     │  copy via DMA engine
                                     ▼
                                   GM
```

### TLOAD

`TLOAD` moves data from a `GlobalTensor` into a tile buffer:

```
dst[i, j] = src[ r0 + i, c0 + j ]
```

Where `r0` and `c0` are the base offsets derived from the `GlobalTensor` shape/stride and the tile's declared valid region `(Rv, Cv)`.

**Transfer size**: `TLOAD` transfers exactly `dst.GetValidRow() × dst.GetValidCol()` elements.

**Constraints**:
- Source dtype size MUST equal destination dtype size.
- Layout compatibility MUST be satisfied:
  - `TileType::Vec`: ND→ND, DN→DN, NZ→NZ
  - `TileType::Mat`: ND→ND, DN→DN, NZ→NZ, ND→NZ, DN→ZN

### TSTORE

`TSTORE` moves data from a tile buffer to a `GlobalTensor`:

```
dst[ r0 + i, c0 + j ] = src[i, j]
```

Where `i ∈ [0, src.GetValidRow())`, `j ∈ [0, src.GetValidCol())`.

**Transfer size**: `TSTORE` transfers exactly `src.GetValidRow() × src.GetValidCol()` elements.

### Atomic Store Variants

`TSTORE` supports atomic store modes via the `AtomicType` attribute:

| AtomicType | Behavior |
|------------|----------|
| `AtomicNone` | Normal store (overwrite) |
| `AtomicAdd` | Atomic add to GM location |
| `AtomicMax` | Atomic max |
| `AtomicMin` | Atomic min |

## Vector Surface Data Path

The vector surface (`pto.v*`) requires an explicit GM↔UB DMA step before vector loads and after vector stores:

```
GM
  │
  │  copy_ubuf_to_gm / copy_gm_to_ubuf (DMA, MTE2/MTE3)
  ▼
UB (Unified Buffer, 256 KB on-chip)
  │
  │  vlds / vsld / vgather2 (vector load, from UB to vreg)
  ▼
Vector Registers  ──►  Vector Compute  ──►  Vector Registers
                                                │
                                                │  vsts / vsst / vscatter (vector store)
                                                ▼
                                            UB ──► GM
```

### DMA Copy Operations

The following scalar/control operations configure and execute GM↔UB DMA:

| Operation | Direction | Description |
|-----------|-----------|-------------|
| `copy_gm_to_ubuf` | GM → UB | Move data from GM to UB staging area |
| `copy_ubuf_to_gm` | UB → GM | Move data from UB to GM |
| `copy_ubuf_to_ubuf` | UB → UB | Copy within UB (e.g., double-buffering) |

These are `pto.*` control-surface operations. They do NOT implicitly synchronize — a `set_flag`/`wait_flag` sequence or explicit `TSYNC` is required before the data is consumed by subsequent vector compute.

### Vector Load/Store (pto.v*)

After DMA staging, `vlds`/`vsld` bring data from UB into vector registers, and `vsts`/`vsst` write data from vector registers back to UB:

| Operation | Path | Description |
|-----------|------|-------------|
| `vlds` | UB → vreg | Standard vector load with distribution mode |
| `vsld` | vreg → UB | Standard vector store |
| `vgather2` | UB → vreg | Strided/gather load from UB |
| `vscatter` | vreg → UB | Strided/scatter store to UB |

**Distribution modes** (for `vlds`):

| Mode | Meaning |
|------|---------|
| `NORM` | Contiguous 256-byte load |
| `BRC_B8/B16/B32` | Broadcast: all lanes read the same address |
| `US_B8/B16` | Upsample: duplicate every Nth element |
| `DS_B8/B16` | Downsample: keep every Nth element |
| `UNPK_B8/B16/B32` | Unpack: zero-extend to wider type |
| `DINTLV_B32` | Deinterleave: extract even/odd lanes |
| `SPLT2CHN_B8/B16` | Split 2-channel |
| `SPLT4CHN_B8` | Split 4-channel (RGBA→R) |

## MTE Pipeline

The DMA engine uses three sub-units that operate in a pipeline:

| MTE | Direction | Role in Tile Surface | Role in Vector Surface |
|-----|-----------|---------------------|----------------------|
| `MTE1` | GM → UB | Optional: explicit prefetch | Pre-stage data before vector load |
| `MTE2` | GM → UB | Load staging: GM→UB→tile buffer (via TLOAD) | DMA copy: GM→UB (via `copy_gm_to_ubuf`) |
| `MTE3` | Tile → GM | Store: tile→UB→GM (via TSTORE) | DMA copy: UB→GM (via `copy_ubuf_to_gm`) |

MTE1, MTE2, and MTE3 can operate in parallel with the Vector Pipeline and Matrix Multiply Unit when proper `set_flag`/`wait_flag` synchronization is used.

## Constraints

- Movement legality depends on source surface, destination surface, layout, and target profile.
- Movement ops do not erase valid-region rules; they carry or define them.
- Vector-surface loads and stores obey their own buffer/register rules and are NOT interchangeable with tile movement.
- DMA copy operations require explicit synchronization before their data is consumed by vector compute.
- `TLOAD`/`TSTORE` carry valid-region information implicitly; the transfer size is determined by the destination/source tile's valid region.

## Cases That Are Not Allowed

- Documenting data movement as though it were implicit when the ISA requires an explicit move.
- Assuming vector-buffer traffic and tile-buffer traffic share the same legality contract.
- Silently relying on target-specific movement shortcuts as if they were architecture-wide.
- Issuing a `vlds` before the corresponding `copy_gm_to_ubuf` has completed without an intervening `set_flag`/`wait_flag`.

## Examples

### Tile Surface: Elementwise Add

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add(Tile<float, 16, 16>& c, const GlobalTensor<float>& ga,
             const GlobalTensor<float>& gb) {
    Tile<float, 16, 16> a, b;
    TLOAD(a, ga);           // GM → UB → Tile Buffer A
    TLOAD(b, gb);           // GM → UB → Tile Buffer B
    TADD(c, a, b);          // c = a + b, iterated over c's valid region
    TSTORE(gc, c);          // Tile Buffer C → UB → GM
}
```

### Vector Surface: Fine-Grained Vector Load/Store

```c
// 1. DMA copy from GM to UB staging area
copy_gm_to_ubuf(%ub_ptr, %gm_ptr, %sid, %n_burst, %len_burst, %stride_dst, %stride_src);

// 2. Signal Vector pipe that data is ready
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

// 3. Wait for data, then vector load
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
%vreg = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>;

// 4. Vector compute
%result = pto.vadd %vreg, %vreg : !pto.vreg<64xf32> -> !pto.vreg<64xf32>;

// 5. Vector store
pto.vsts %result, %ub_out[%offset] : !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> ();

// 6. DMA copy from UB back to GM
copy_ubuf_to_gm(%ub_out, %gm_out, %sid, %n_burst, %len_burst, %reserved, %stride_dst, %stride_src);
```

## See Also

- [Tiles And Valid Regions](./tiles-and-valid-regions.md)
- [Vector Instruction Surface](../instruction-surfaces/vector-instructions.md)
- [Tile Memory And Data Movement Families](../tile/memory-and-data-movement.md)
- [Vector Load/Store Reference](../vector/vector-load-store.md)
- [Scalar DMA Copy Reference](../scalar/dma-copy.md)
