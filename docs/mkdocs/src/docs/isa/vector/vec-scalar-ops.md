<!-- Generated from `docs/isa/vector/vec-scalar-ops.md` -->

# Vector Families: Vector-Scalar Ops

This page documents the `pto.v*` families that combine one vector register with one scalar operand. Scalar broadcasting, carry-chain rules, and active-lane behavior are architecture-visible and therefore documented here.

> **Category:** Vector-scalar operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that combine a vector with a scalar value, applying the scalar to every lane.

## Common Operand Model

- `%input` is the source vector register value.
- `%scalar` is the scalar operand in SSA form.
- `%mask` is the predicate operand.
- `%result` is the destination vector register value.
- For 32-bit scalar forms, the scalar source MUST satisfy the backend's legal
  scalar-source constraints for this family.

---

## Arithmetic

### `pto.vadds`

- **syntax:** `%result = pto.vadds %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] + scalar;
```

- **inputs:** `%input` is the source vector, `%scalar` is broadcast logically to
  each active lane, and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Inactive lanes follow the predication
  behavior defined for this family. On the current surface, inactive lanes are
  treated as zeroing lanes.

---

### `pto.vsubs`

- **syntax:** `%result = pto.vsubs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] - scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise difference.
- **constraints and limitations:** Integer or floating-point legality depends on
  the selected type family in lowering.

---

### `pto.vmuls`

- **syntax:** `%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** Supported element types are hardware-family
  specific; the current PTO ISA vector surface documentation covers the common numeric cases.

---

### `pto.vmaxs`

- **syntax:** `%result = pto.vmaxs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match.

---

### `pto.vmins`

- **syntax:** `%result = pto.vmins %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise minimum.
- **constraints and limitations:** Input and result types MUST match.

---

## Bitwise

### `pto.vands`

- **syntax:** `%result = pto.vands %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] & scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise AND.
- **constraints and limitations:** Integer element types only.

---

### `pto.vors`

- **syntax:** `%result = pto.vors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] | scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise OR.
- **constraints and limitations:** Integer element types only.

---

### `pto.vxors`

- **syntax:** `%result = pto.vxors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] ^ scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise XOR.
- **constraints and limitations:** Integer element types only.

---

## Shift

### `pto.vshls`

- **syntax:** `%result = pto.vshls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] << scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. The shift amount
  SHOULD stay within the source element width.

---

### `pto.vshrs`

- **syntax:** `%result = pto.vshrs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] >> scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only.

---

### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : scalar * src[i];
```

- **inputs:** `%input` is the activation vector, `%scalar` is the leaky slope,
  and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise leaky-ReLU result.
- **constraints and limitations:** Only `f16` and `f32` forms are currently
  documented for `pto.vlrelu`.

---

## Carry Operations

### `pto.vaddcs`

- **syntax:** `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Add with carry-in and carry-out.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

- **inputs:** `%lhs` and `%rhs` are the value vectors, `%carry_in` is the
  incoming carry predicate, and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic result and `%carry` is the carry-out
  predicate.
- **constraints and limitations:** This is the scalar-extended carry-chain
  family. Treat it as an unsigned integer operation unless the verifier states a
  wider legal domain.

---

### `pto.vsubcs`

- **syntax:** `%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Subtract with borrow-in and borrow-out.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - borrow_in[i];
    borrow_out[i] = (src0[i] < src1[i] + borrow_in[i]);
}
```

- **inputs:** `%lhs` and `%rhs` are the value vectors, `%borrow_in` is the
  incoming borrow predicate, and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic result and `%borrow` is the
  borrow-out predicate.
- **constraints and limitations:** This is the scalar-extended borrow-chain
  family and SHOULD be treated as an unsigned integer operation.

---

## Typical Usage

```mlir
// Add bias to all elements
%biased = pto.vadds %activation, %bias_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Scale by constant
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Clamp to [0, 255] for uint8 quantization
%clamped_low = pto.vmaxs %input, %c0, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmins %clamped_low, %c255, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Shift right by fixed amount
%shifted = pto.vshrs %data, %c4, %mask : !pto.vreg<64xi32>, i32, !pto.mask -> !pto.vreg<64xi32>
```
