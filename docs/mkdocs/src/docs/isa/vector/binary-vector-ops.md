<!-- Generated from `docs/isa/vector/binary-vector-ops.md` -->

# Vector Families: Binary Vector Ops

This page documents two-input `pto.v*` compute families. The detailed per-op sections below are imported into the PTO ISA manual because vector micro-instruction legality and operand discipline are part of the PTO architecture contract, not external notes.

> **Category:** Two-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take two vector inputs and produce one vector output.

## Common Operand Model

- `%lhs` and `%rhs` are the two source vector register values.
- `%mask` is the predicate operand `Pg` that gates which lanes participate.
- `%result` is the destination vector register value. Unless explicitly noted,
  it has the same lane count and element type as the inputs.
- Unless explicitly documented otherwise, `%lhs`, `%rhs`, and `%result` MUST
  have matching vector shapes and element types.

---

## Arithmetic

### `pto.vadd`

- **syntax:** `%result = pto.vadd %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise; `%mask` selects active
  lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Input and result types MUST match.

---

### `pto.vsub`

- **syntax:** `%result = pto.vsub %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] - src1[i];
```

- **inputs:** `%lhs` is the minuend, `%rhs` is the subtrahend, and `%mask`
  selects active lanes.
- **outputs:** `%result` is the lane-wise difference.
- **constraints and limitations:** Input and result types MUST match.

---

### `pto.vmul`

- **syntax:** `%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, bf16, f32 (**NOT** i8/u8)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

- **inputs:** `%lhs` and `%rhs` are multiplied lane-wise; `%mask` selects
  active lanes.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** The current A5 profile excludes `i8/u8`
  forms from this surface.

---

### `pto.vdiv`

- **syntax:** `%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32 only (no integer division)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

- **inputs:** `%lhs` is the numerator, `%rhs` is the denominator, and `%mask`
  selects active lanes.
- **outputs:** `%result` is the lane-wise quotient.
- **constraints and limitations:** Floating-point element types only. Active
  denominators containing `+0` or `-0` follow the target's exceptional
  behavior.

---

### `pto.vmax`

- **syntax:** `%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] > src1[i]) ? src0[i] : src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` holds the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match.

---

### `pto.vmin`

- **syntax:** `%result = pto.vmin %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] < src1[i]) ? src0[i] : src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` holds the lane-wise minimum.
- **constraints and limitations:** Input and result types MUST match.

---

## Bitwise

### `pto.vand`

- **syntax:** `%result = pto.vand %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] & src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise AND.
- **constraints and limitations:** Integer element types only.

---

### `pto.vor`

- **syntax:** `%result = pto.vor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] | src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise OR.
- **constraints and limitations:** Integer element types only.

---

### `pto.vxor`

- **syntax:** `%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise XOR.
- **constraints and limitations:** Integer element types only.

---

## Shift

### `pto.vshl`

- **syntax:** `%result = pto.vshl %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] << src1[i];
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the per-lane
  shift amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Shift counts
  SHOULD stay within `[0, bitwidth(T) - 1]`; out-of-range behavior is target-
  defined unless the verifier narrows it further.

---

### `pto.vshr`

- **syntax:** `%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the per-lane
  shift amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Signedness of the
  element type determines arithmetic vs logical behavior.

---

## Carry Operations

### `pto.vaddc`

- **syntax:** `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Add with carry output.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise and `%mask` selects active
  lanes.
- **outputs:** `%result` is the truncated arithmetic result and `%carry` is the
  carry/overflow predicate per lane.
- **constraints and limitations:** This is a carry-chain integer add family. On
  the current A5 surface, it SHOULD be treated as an unsigned integer
  operation.

---

### `pto.vsubc`

- **syntax:** `%result, %borrow = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Subtract with borrow output.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

- **inputs:** `%lhs` and `%rhs` are subtracted lane-wise and `%mask` selects
  active lanes.
- **outputs:** `%result` is the arithmetic difference and `%borrow` marks lanes
  that borrowed.
- **constraints and limitations:** This operation SHOULD be treated as an
  unsigned 32-bit carry-chain family unless and until the verifier states
  otherwise.

---

## Typical Usage

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask -> !pto.vreg<64xi32>
```
