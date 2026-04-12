<!-- Generated from `docs/isa/vector/predicate-and-materialization.md` -->

# Vector Families: Predicate And Materialization

This page documents the predicate-register and materialization families used by `pto.v*` code. Predicate load/store, mask generation, and predicate algebra are architecture-visible because they control which lanes participate in later vector operations.

> **Category:** UB ↔ Predicate Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Predicate registers (`!pto.mask`) are 256-bit registers that enable per-lane conditional execution. These ops move predicate values between UB and predicate registers.

---

## Predicate Loads

### `pto.plds`

- **syntax:** `%result = pto.plds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.mask`
- **semantics:** Load predicate register with scalar offset.

**Distribution modes:** `NORM`, `US`, `DS`

**Example:**
```mlir
%mask = pto.plds %ub[%c0] {dist = "NORM"} : !pto.ptr<T, ub> -> !pto.mask
```

---

### `pto.pld`

- **syntax:** `%result = pto.pld %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask`
- **semantics:** Load predicate register with areg offset.

---

### `pto.pldi`

- **syntax:** `%result = pto.pldi %source, %offset, "DIST" : !pto.ptr<T, ub>, i32 -> !pto.mask`
- **semantics:** Load predicate register with immediate offset.

---

## Predicate Stores

### `pto.psts`

- **syntax:** `pto.psts %value, %dest[%offset] : !pto.mask, !pto.ptr<T, ub>`
- **semantics:** Store predicate register with scalar offset.

**Example:**
```mlir
pto.psts %mask, %ub[%c0] : !pto.mask, !pto.ptr<T, ub>
```

---

### `pto.pst`

- **syntax:** `pto.pst %value, %dest[%offset], "DIST" : !pto.mask, !pto.ptr<T, ub>, index`
- **semantics:** Store predicate register with areg offset.

**Distribution modes:** `NORM`, `PK`

---

### `pto.psti`

- **syntax:** `pto.psti %value, %dest, %offset, "DIST" : !pto.mask, !pto.ptr<T, ub>, i32`
- **semantics:** Store predicate register with immediate offset.

---

### `pto.pstu`

- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Predicate unaligned store with align state update.

---

## Typical Usage Pattern

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Store mask to UB for later use
pto.psts %mask, %ub_mask[%c0] : !pto.mask, !pto.ptr<T, ub>

// ... later in another kernel ...

// Load mask from UB
%saved_mask = pto.plds %ub_mask[%c0] {dist = "NORM"} : !pto.ptr<T, ub> -> !pto.mask

// Use for predicated select
%result = pto.vsel %v_true, %v_false, %saved_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

> **Category:** Scalar broadcast, predicate generation and manipulation
> **Pipeline:** PIPE_V (Vector Core)

These ops create vectors from scalar values and manipulate predicate registers.

## Common Operand Model

- `%value` is the scalar source value in SSA form.
- `%input` is either a source scalar or a source vector depending on the op.
- `%result` is the destination vector register value.
- For 32-bit scalar inputs, the scalar source MUST satisfy the backend's legal
  scalar-source constraints for this family.

---

## Scalar Materialization

### `pto.vbr`

- **syntax:** `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- **semantics:** Broadcast scalar to all vector lanes.
- **inputs:**
  `%value` is the scalar source.
- **outputs:**
  `%result` is a vector whose active lanes all carry `%value`.
- **constraints and limitations:**
  Supported forms are `b8`, `b16`, and `b32`. For `b8`, only the low 8 bits of
  the scalar source are consumed.

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

**Example:**
```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

---

### `pto.vdup`

- **syntax:** `%result = pto.vdup %input {position = "POSITION"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Duplicate scalar or vector element to all lanes.
- **inputs:**
  `%input` supplies the scalar or source-lane value selected by `position`.
- **outputs:**
  `%result` is the duplicated vector.
- **constraints and limitations:**
  `position` selects which source element or scalar position is duplicated. The
  current PTO ISA vector surface representation models that selector as an attribute rather than a
  separate operand.

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

---

## Predicate Generation

### `pto.pset_b8` / `pto.pset_b16` / `pto.pset_b32`

- **syntax:** `%result = pto.pset_b32 "PAT_*" : !pto.mask`
- **semantics:** Generate predicate from pattern.

**Patterns:**

| Pattern | Description |
|---------|-------------|
| `PAT_ALL` | All lanes active |
| `PAT_ALLF` | All lanes inactive |
| `PAT_H` | High half active |
| `PAT_Q` | Upper quarter active |
| `PAT_VL1`...`PAT_VL128` | First N lanes active |
| `PAT_M3`, `PAT_M4` | Modular patterns |

**Example — All 64 f32 lanes active:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask
```

**Example — First 16 lanes active:**
```mlir
%first_16 = pto.pset_b32 "PAT_VL16" : !pto.mask
```

---

### `pto.pge_b8` / `pto.pge_b16` / `pto.pge_b32`

- **syntax:** `%result = pto.pge_b32 "PAT_*" : !pto.mask`
- **semantics:** Generate tail mask — first N lanes active.

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

**Example — Tail mask for remainder loop:**
```mlir
%tail_mask = pto.pge_b32 "PAT_VL8" : !pto.mask

---

### `pto.plt_b8` / `pto.plt_b16` / `pto.plt_b32`

- **syntax:** `%mask, %scalar_out = pto.plt_b32 %scalar : i32 -> !pto.mask, i32`
- **semantics:** Generate predicate state together with updated scalar state.
```

---

## Predicate Pack/Unpack

### `pto.ppack`

- **syntax:** `%result = pto.ppack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Narrowing pack of predicate register.

**Part tokens:** `LOWER`, `HIGHER`

---

### `pto.punpack`

- **syntax:** `%result = pto.punpack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Widening unpack of predicate register.

---

## Predicate Logical Ops

### `pto.pand`

- **syntax:** `%result = pto.pand %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise AND.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] & src1[i];
```

---

### `pto.por`

- **syntax:** `%result = pto.por %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise OR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] | src1[i];
```

---

### `pto.pxor`

- **syntax:** `%result = pto.pxor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise XOR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] ^ src1[i];
```

---

### `pto.pnot`

- **syntax:** `%result = pto.pnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise NOT.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = ~src[i];
```

---

### `pto.psel`

- **syntax:** `%result = pto.psel %src0, %src1, %sel : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate select (mux).

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

---

## Predicate Movement

### `pto.ppack`

- **syntax:** `%result = pto.ppack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Narrowing pack of predicate register.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src[i];
```

---

### `pto.punpack`

- **syntax:** `%result = pto.punpack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Widening unpack of predicate register.

---

### `pto.pdintlv_b8`

- **syntax:** `%low, %high = pto.pdintlv_b8 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **semantics:** Predicate deinterleave.

---

### `pto.pintlv_b16`

- **syntax:** `%low, %high = pto.pintlv_b16 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **semantics:** Predicate interleave.

---

## Typical Usage

```mlir
// Generate all-active mask for f32 (64 lanes)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask

// Generate tail mask for remainder (last 12 elements)
%tail = pto.pge_b32 "PAT_VL12" : !pto.mask

// Compare and generate mask
%cmp_mask = pto.vcmp %a, %b, %all, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Combine masks: only process tail elements that passed comparison
%combined = pto.pand %cmp_mask, %tail, %all : !pto.mask, !pto.mask, !pto.mask -> !pto.mask

// Use for predicated operation
%result = pto.vsel %true_vals, %false_vals, %combined : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```
