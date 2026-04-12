<!-- Generated from `docs/isa/vector/vector-load-store.md` -->

# Vector Families: Vector Load/Store

This page documents UB-to-vector-register data movement inside PTO ISA. The detailed forms below describe how `pto.v*` kernels move payloads between vector-visible UB storage and vector registers without crossing back into the tile surface.

> **Category:** UB ↔ Vector Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Vector loads move data from Unified Buffer (UB) to vector registers (`vreg`). Vector stores move data from `vreg` back to UB. All vector compute operates only on `vreg` — UB is the staging area between DMA and compute.

## Common Operand Model

- `%source` / `%dest` is the base address operand in SSA form. The base pointer
  MUST address the Vector tile buffer / UB space.
- `%offset` is the displacement operand in SSA form. The exact encoding is
  instruction-specific, but the effective address and any post-update behavior
  MUST match the selected instruction form.
- `%mask` is the predicate operand for predicated memory families. For memory
  families,
  inactive lanes or inactive blocks MUST NOT issue memory requests unless the
  instruction explicitly documents a different behavior.
- `%result` is the destination vector register value in SSA form.
- `!pto.align` is the SSA carrier for alignment-buffer state used by unaligned
  load/store families. The PTO ISA vector surface representation makes that state explicit rather than implicit.

---

## Contiguous Loads

### `pto.vlds`

- **syntax:** `%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Vector load with distribution mode.
- **inputs:**
  `%source` is the UB base address, `%offset` is the load displacement, and
  `DIST` selects the distribution mode.
- **outputs:**
  `%result` is the loaded vector register value.
- **constraints and limitations:**
  The effective address MUST satisfy the alignment rule of the selected
  distribution mode. `NORM` reads one full vector footprint. Broadcast,
  upsample, downsample, unpack, split-channel, and deinterleave modes change
  how memory bytes are mapped into destination lanes, but they do not change the
  fact that the source is UB memory.

**Distribution modes:**

| Mode | Description | C Semantics |
|------|-------------|-------------|
| `NORM` | Contiguous 256B load | `dst[i] = UB[base + i * sizeof(T)]` |
| `BRC_B8/B16/B32` | Broadcast single element | `dst[i] = UB[base]` for all i |
| `US_B8/B16` | Upsample (duplicate each element) | `dst[2*i] = dst[2*i+1] = UB[base + i]` |
| `DS_B8/B16` | Downsample (every 2nd element) | `dst[i] = UB[base + 2*i]` |
| `UNPK_B8/B16/B32` | Unpack (zero-extend to wider type) | `dst_i32[i] = (uint32_t)UB_i16[base + 2*i]` |
| `SPLT4CHN_B8` | Split 4-channel (RGBA → R plane) | Extract every 4th byte |
| `SPLT2CHN_B8/B16` | Split 2-channel | Extract every 2nd element |
| `DINTLV_B32` | Deinterleave 32-bit | Even elements only |
| `BLK` | Block load | Blocked access pattern |

**Example — Contiguous load:**
```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Example — Broadcast scalar to all lanes:**
```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

---

### `pto.vldas`

- **syntax:** `%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Prime alignment buffer for subsequent unaligned load.
- **inputs:**
  `%source` is the UB address whose surrounding aligned block seeds the load
  alignment state.
- **outputs:**
  `%result` is the initialized load-alignment state.
- **constraints and limitations:**
  This op is the required leading operation for a `pto.vldus` stream using the
  same alignment state. The source address itself need not be 32-byte aligned;
  hardware truncates it to the aligned block boundary for the priming load.

---

### `pto.vldus`

- **syntax:** `%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned load using primed align state.
- **inputs:**
  `%source` is the current UB address and `%align` is the incoming load
  alignment state primed by `pto.vldas` or a prior `pto.vldus`.
- **outputs:**
  `%result` is the assembled vector value, `%align_out` is the updated alignment
  state, and `%base_out` is the post-update base pointer state exposed in SSA
  form.
- **constraints and limitations:**
  A matching `pto.vldas` MUST appear before the first dependent `pto.vldus`
  stream in the same vector loop. Both the alignment state and the base address
  advance across the stream, and the PTO ISA vector surface representation exposes those updates as SSA results.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

---

## Dual Loads (Deinterleave)

### `pto.vldx2`

- **syntax:** `%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the displacement, and `DIST`
  selects a dual-load/deinterleave layout.
- **outputs:**
  `%low` and `%high` are the two destination vectors.
- **constraints and limitations:**
  This family is only legal for interleave/deinterleave style distributions.
  The two outputs form an ordered pair, and that pairing MUST be preserved.

**Distribution modes:** `DINTLV_B8`, `DINTLV_B16`, `DINTLV_B32`, `BDINTLV`

```c
// DINTLV_B32: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

---

## Strided Loads

### `pto.vsld`

- **syntax:** `%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Strided load with fixed stride pattern.
- **inputs:**
  `%source` is the UB base pointer and `%offset` is the displacement encoded
  with the selected fixed stride mode.
- **outputs:**
  `%result` is the loaded vector.
- **constraints and limitations:**
  This is a deprecated compatibility family. The selected stride token
  determines which sub-elements are read from each source block.

**Stride modes:** `STRIDE_S3_B16`, `STRIDE_S4_B64`, `STRIDE_S8_B32`, `STRIDE_S2_B64`

---

### `pto.vsldb`

- **syntax:** `%result = pto.vsldb %source, %offset, %mask : !pto.ptr<T, ub>, i32, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Block-strided load for 2D tile access.
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the packed stride/control word,
  and `%mask` controls which blocks participate.
- **outputs:**
  `%result` is the loaded vector.
- **constraints and limitations:**
  `%offset` is not a plain byte displacement; it encodes the block stride and
  repeat pattern. If a block is masked off, the corresponding destination block
  is zeroed and MUST NOT raise an address overflow exception for that block.

---

## Gather (Indexed) Loads

### `pto.vgather2`

- **syntax:** `%result = pto.vgather2 %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Indexed gather from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` provides per-lane element
  offsets, and `%active_lanes` bounds how many lanes participate.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  Only the first `%active_lanes` indices participate. The index element width
  and interpretation MUST match the selected gather form, and each effective
  address must satisfy that form's alignment rules.

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

---

### `pto.vgatherb`

- **syntax:** `%result = pto.vgatherb %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Byte-granularity indexed gather from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` contains per-block byte offsets,
  and `%active_lanes` bounds the number of active gathered blocks.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a block gather, not a byte-per-lane gather. `%source` MUST be 32-byte
  aligned, each participating offset MUST describe a 32-byte-aligned block, and
  inactive blocks are zero-filled.

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i]];  // byte-addressed
```

---

### `pto.vgather2_bc`

- **syntax:** `%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Gather with broadcast, conditioned by mask.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` contains gather indices, and
  `%mask` gates which lanes participate.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a backward-compatible family. Masked-off lanes do not participate in
  address coalescing and do not trigger address overflow exceptions; their
  destination lanes are zero-filled.

---

## Contiguous Stores

### `pto.vsts`

- **syntax:** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask`
- **semantics:** Vector store with distribution mode.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the displacement, `%mask` selects the active lanes or sub-elements, and
  `DIST` selects the store distribution.
- **outputs:**
  This op has no SSA result; it writes to UB memory.
- **constraints and limitations:**
  The effective destination address MUST satisfy the alignment rule of the
  selected store mode. Narrowing/packing modes may only preserve a subset of the
  source bits. Merge-channel modes reinterpret the source vector as channel
  planes and interleave them on store.

**Distribution modes:**

| Mode | Description | C Semantics |
|------|-------------|-------------|
| `NORM_B8/B16/B32` | Contiguous store | `UB[base + i] = src[i]` |
| `PK_B16/B32` | Pack/narrowing store | `UB_i16[base + 2*i] = truncate_16(src_i32[i])` |
| `MRG4CHN_B8` | Merge 4 channels (R,G,B,A → RGBA) | Interleave 4 planes |
| `MRG2CHN_B8/B16` | Merge 2 channels | Interleave 2 planes |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
```

---

## Dual Stores (Interleave)

### `pto.vstx2`

- **syntax:** `pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask`
- **semantics:** Dual interleaved store (SoA → AoS conversion).
- **inputs:**
  `%low` and `%high` are the two source vectors, `%dest` is the UB base pointer,
  `%offset` is the displacement, `DIST` selects the interleave layout, and
  `%mask` gates the participating elements.
- **outputs:**
  This op has no SSA result; it writes an interleaved stream to UB.
- **constraints and limitations:**
  This family is only legal for interleave distributions. The two source
  vectors form an ordered pair, and the interleave semantics of that pair MUST
  be preserved.

**Distribution modes:** `INTLV_B8`, `INTLV_B16`, `INTLV_B32`

```c
// INTLV_B32:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

---

## Strided Stores

### `pto.vsst`

- **syntax:** `pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>`
- **semantics:** Strided store with fixed stride pattern.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, and `%offset`
  / `STRIDE` select the fixed strided layout.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  This is a deprecated compatibility family. The stride token, not the vector
  lane number alone, determines which destination elements are written.

---

### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %offset, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i32, !pto.mask`
- **semantics:** Block-strided store for 2D tile access.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the packed stride/control word, and `%mask` controls block participation.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  `%offset` is a control word, not a plain byte displacement. This is a
  deprecated compatibility family kept for surface coverage.

---

## Scatter (Indexed) Stores

### `pto.vscatter`

- **syntax:** `pto.vscatter %value, %dest, %offsets, %active_lanes : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, index`
- **semantics:** Indexed scatter to UB.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offsets`
  provides per-lane or per-block indices, and `%active_lanes` bounds the active
  requests.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  Only `b8`, `b16`, and `b32` element sizes are supported. The index vector
  must use a supported integer element type and layout for this family.
  Each computed address MUST be element-aligned. If two or more indices alias,
  only one write is guaranteed and the winning lane is implementation-defined.

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

---

## Alignment State Stores

### `pto.vsta`

- **syntax:** `pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index`
- **semantics:** Flush alignment state to memory.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base pointer,
  and `%offset` is the flush displacement.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  The flush address MUST match the post-updated address expected by the
  preceding unaligned-store stream. After the flush, the corresponding store
  alignment state is consumed.

---

### `pto.vstas`
- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Scalar-register-offset form of alignment-state flush.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base
  pointer, and `%offset` is the scalar-register style displacement.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  This family uses the same buffered-tail semantics as `pto.vsta` but keeps the
  scalar-offset form explicit.

---

### `pto.vstar`
- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush alignment state using the register-update form.
- **inputs:**
  `%value` is the pending store-alignment state and `%dest` is the UB base
  pointer.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  The implicit update state consumed by this flush MUST correspond to the same
  store stream that produced `%value`.

---

### `pto.vstu`
- **syntax:** `%align_out, %base_out = pto.vstu %align_in, %base_in, %value, %dest, %mode : !pto.align, !pto.ptr<T, ub>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned store with explicit threaded alignment/base state.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%base_in` is the current
  stream base, `%value` is the vector to store, `%dest` is the UB base pointer,
  and `%mode` selects the post-update behavior.
- **outputs:**
  `%align_out` is the updated buffered-tail state and `%base_out` is the
  post-update base pointer state.
- **constraints and limitations:**
  This op models a stateful unaligned-store sequence in SSA form. A final
  `pto.vsta` / `pto.vstas` / `pto.vstar` is still required to flush the trailing
  buffered bytes.

---

### `pto.vstus`
- **syntax:** `%align_out, %base_out = pto.vstus %align_in, %base_in, %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, !pto.vreg<NxT>, !pto.ptr<T, ub>, i32 -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Scalar-offset unaligned store with threaded state.
- **inputs:**
  Same roles as `pto.vstu`, but `%offset` is provided explicitly as the scalar
  displacement.
- **outputs:**
  Updated alignment state and base state.
- **constraints and limitations:**
  The same final flush requirement and state-threading constraints as
  `pto.vstu` apply.

---

### `pto.vstur`
- **syntax:** `%align_out = pto.vstur %align_in, %value, %dest : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Register-update unaligned store form.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%value` is the vector to
  store, and `%dest` is the UB base pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state.
- **constraints and limitations:**
  This op updates only the residual alignment state. A matching flush op is
  still required to emit the trailing bytes.

- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Flush alignment state with scalar offset.

---

### `pto.vstar`

- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush remaining alignment state.
- **inputs:**
  `%value` is the pending alignment/buffer state that still needs to be emitted,
  and `%dest` is the UB destination base pointer.
- **outputs:**
  No SSA result. The effect is a memory-side flush that writes the remaining
  buffered bytes to memory.
- **constraints and limitations:**
  This op terminates an unaligned-store sequence. It MUST be paired with a
  compatible prior state-producing store sequence so that the pending tail state
  is well-defined.

---

## Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

### `pto.vstu`

- **syntax:** `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index`
- **semantics:** Unaligned store with align + offset state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset_in` is the current
  logical byte/element displacement, `%value` is the vector being stored, and
  `%base` is the UB base pointer.
- **outputs:**
  `%align_out` is the updated alignment/tail state and `%offset_out` is the
  next offset after applying the selected post-update rule.
- **constraints and limitations:**
  The alignment state MUST be threaded in program order. A terminating flush
  form such as `pto.vstar`/`pto.vstas` is still required to commit the buffered
  tail bytes.

**Mode tokens:** `POST_UPDATE`, `NO_POST_UPDATE`

---

### `pto.vstus`

- **syntax:** `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned store with scalar offset and state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset` is the scalar
  displacement, `%value` is the vector being stored, and `%base` is the UB base
  pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state and `%base_out` is the next
  base pointer when the lowering chooses a post-update form.
- **constraints and limitations:**
  This is the scalar-offset stateful form of the unaligned store family. The
  scalar offset width and update mode MUST match the selected form, and a later
  flush op is still required.

---

### `pto.vstur`

- **syntax:** `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Unaligned store with residual flush and state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%value` is the vector to
  store, and `%base` is the UB base pointer.
- **outputs:**
  `%align_out` is the updated residual state after the current partial store.
- **constraints and limitations:**
  This form exposes only the evolving state; it does not by itself guarantee
  that all buffered bytes have reached memory. A compatible final flush is still
  required unless the surrounding sequence is known to be complete.
