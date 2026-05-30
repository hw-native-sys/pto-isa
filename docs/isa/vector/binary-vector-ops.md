# Vector Instruction Set: Binary Vector Instructions

Two-input `pto.v*` compute instruction sets are defined here. The detailed per-op sections below are part of the PTO ISA manual because vector micro-instruction legality and operand discipline belong to the PTO architecture contract rather than to external notes.

> **Category:** Two-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take two vector inputs and produce one vector output.

---

## Common Operand Model

- `%lhs` and `%rhs` are the two source vector register values.
- `%mask` is the predicate operand `Pg` that gates which lanes participate.
- `%result` is the destination vector register value. Unless explicitly noted,
  it has the same lane count and element type as the inputs.
- Unless explicitly documented otherwise, `%lhs`, `%rhs`, and `%result` MUST
  have matching vector shapes and element types.

---

## Execution Model: vecscope

Binary vector operations execute inside a `pto.vecscope { ... }` region, which establishes the Vector Core's execution context. All vector instructions inside the region are issued to `PIPE_V`.

**Producer-consumer pipeline pattern (A2/A3 double-buffering):**

```mlir
// Stage 1: MTE2 loads tile from GM to UB
pto.get_buf "PIPE_MTE2", %bufid, %c0 : i64, i64
pto.copy_gm_to_ubuf %gm_ptr, %ub_tile, ... : ...
pto.rls_buf "PIPE_MTE2", %bufid, %c0 : i64, i64

// Stage 2: Vector compute
pto.get_buf "PIPE_V", %bufid, %c0 : i64, i64
pto.vecscope {
  scf.for %offset = %c0 to %N step %c64 iter_args(%remaining = %N_i32) -> (i32) {
    %mask, %next = pto.plt_b32 %remaining : i32 -> !pto.mask<G>, i32
    %lhs = pto.vlds %ub_a[%offset] : !pto.ptr -> !pto.vreg<64xf32>
    %rhs = pto.vlds %ub_b[%offset] : !pto.ptr -> !pto.vreg<64xf32>
    %out = pto.vadd %lhs, %rhs, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %out, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr, !pto.mask<b32>
    scf.yield %next : i32
  }
}
pto.rls_buf "PIPE_V", %bufid, %c0 : i64, i64
```

**Key mechanism:** `pto.get_buf` / `pto.rls_buf` resolve cross-pipeline RAW/WAR dependencies automatically via buffer acquire/release — no explicit event IDs or loop peeling required.

---

## A5 Latency and Throughput (Ascend910_9599)

> All values are **popped→retire** cycle counts on the cycle-accurate simulator.

### Latency Summary Table

| PTO op | A5 RV (CA) | f32 | f16 | bf16 | i32 | i16 | i8 |
|--------|-------------|-----|------|------|-----|------|-----|
| `pto.vadd` | `RV_VADD` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vsub` | `RV_VSUB` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vmul` | `RV_VMUL` | 8 | 8 | — | 8 | 8 | — |
| `pto.vdiv` | `RV_VDIV` | 17 | 22 | — | — | — | — |
| `pto.vmax`/`vmin` | `RV_VMAX` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vand`/`vor`/`vxor` | `RV_VAND` | 7 | 7 | — | 7 | 7 | 7 |
| `pto.vshl`/`vshr` | `RV_VSHL` | — | — | — | 7 | 7 | 7 |
| `pto.vaddc` | `RV_VADDC` | — | — | — | 7 | — | — |
| `pto.vsubc` | `RV_VSUBC` | — | — | — | 7 | — | — |

### A2/A3 Latency and Throughput

| Metric | Constant | Value (cycles) | Applies To |
|--------|-----------|---------------|------------|
| Startup latency (arith) | `A2A3_STARTUP_BINARY` | 14 | all arithmetic binary ops |
| Completion: FP binary ops | `A2A3_COMPL_FP_BINOP` | 19 | `vadd`/`vsub` (f32) |
| Completion: FP transcendental | `A2A3_COMPL_FP32_EXP` | 26 | `vexp` (f32), `vsqrt` (f32) |
| Completion: FP transcendental | `A2A3_COMPL_FP16_EXP` | 28 | `vexp` (f16) |
| Completion: FP transcendental | `A2A3_COMPL_FP16_SQRT` | 29 | `vsqrt` (f16) |
| Completion: INT binary ops | `A2A3_COMPL_INT_BINOP` | 17 | `vadd`/`vsub` (int16/i32) |
| Completion: INT mul | `A2A3_COMPL_INT_MUL` | 18 | `vmul` (int) |
| Per-repeat throughput | `A2A3_RPT_1` | 1 | scalar/simple unary |
| Per-repeat throughput | `A2A3_RPT_2` | 2 | binary ops (`vadd`, `vmul`, `vmax`, `vmin`) |
| Per-repeat throughput | `A2A3_RPT_4` | 4 | transcendental ops (f16 exp/sqrt) |
| Pipeline interval | `A2A3_INTERVAL` | 18 | all vector ops |
| Pipeline interval (vmov) | `A2A3_INTERVAL_VCOPY` | 13 | `vmov` |

**Cycle model (A2/A3):**

```
total_cycles = startup + completion + repeats × per_repeat + (repeats - 1) × interval
```

---

## Arithmetic

### `pto.vadd`

- **syntax:** `%result = pto.vadd %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VADD`; **Latency:** 7 (f32/f16), 7 (i32/i16/i8)
- **A2/A3 throughput:** 2 cycles/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise; `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Input and result types MUST match.

---

### `pto.vsub`

- **syntax:** `%result = pto.vsub %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VSUB`; **Latency:** 7 (f32/f16), 7 (i32/i16/i8)
- **A2/A3 throughput:** 2 cycles/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] - src1[i];
```

- **inputs:** `%lhs` is the minuend, `%rhs` is the subtrahend, and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise difference.
- **constraints and limitations:** Input and result types MUST match.

---

### `pto.vmul`

- **syntax:** `%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VMUL`; **Latency:** 8 (f32/f16), 8 (i32/i16)
- **A2/A3 throughput:** 2 cycles/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

- **inputs:** `%lhs` and `%rhs` are multiplied lane-wise; `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** The current A5 profile excludes `i8/u8` forms from this instruction set. Integer overflow follows target-defined behavior.

---

### `pto.vdiv`

- **syntax:** `%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VDIV`; **Latency:** 17 (f32), 22 (f16)
- **A2/A3 throughput:** 2 cycles/repeat (f32), 4 cycles/repeat (f16); **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

- **inputs:** `%lhs` is the numerator, `%rhs` is the denominator, and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise quotient.
- **constraints and limitations:** Floating-point element types only. Active denominators containing `+0` or `-0` follow the target's exceptional behavior.
- **Performance note:** Division is significantly more expensive than multiplication (17–22 cycles vs 8 cycles). Prefer multiplying by the reciprocal (`vmuls`) when accuracy permits.

---

### `pto.vmax`

- **syntax:** `%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VMAX`; **Latency:** 7 (f32/f16), 7 (i32/i16/i8)
- **A2/A3 throughput:** 2 cycles/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] > src1[i]) ? src0[i] : src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` holds the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match.

---

### `pto.vmin`

- **syntax:** `%result = pto.vmin %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VMAX`; **Latency:** 7 (f32/f16), 7 (i32/i16/i8)
- **A2/A3 throughput:** 2 cycles/repeat; **interval:** 18 cycles

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

- **syntax:** `%result = pto.vand %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VAND`; **Latency:** 7 (integer types)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] & src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise AND.
- **constraints and limitations:** Integer element types only.

---

### `pto.vor`

- **syntax:** `%result = pto.vor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VOR`; **Latency:** 7 (integer types)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] | src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise OR.
- **constraints and limitations:** Integer element types only.

---

### `pto.vxor`

- **syntax:** `%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VXOR`; **Latency:** 7 (integer types)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

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

- **syntax:** `%result = pto.vshl %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VSHL`; **Latency:** 7 (integer types)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] << src1[i];   // per-lane shift: each lane's shift amount varies
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the **per-lane** shift amount (from a second vector register), and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Shift counts SHOULD stay within `[0, bitwidth(T) - 1]`; out-of-range behavior is target-defined unless the verifier narrows it further.

---

### `pto.vshr`

- **syntax:** `%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VSHR`; **Latency:** 7 (integer types)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the **per-lane** shift amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Signedness of the element type determines arithmetic vs logical behavior.

---

## Carry Operations

### `pto.vaddc`

- **syntax:** `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 RV:** `RV_VADDC`; **Latency:** 7 (i32, unsigned carry semantics)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);   // carry predicate: 1 if overflow occurred
}
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise and `%mask` selects active lanes.
- **outputs:** `%result` is the truncated arithmetic result and `%carry` is the carry/overflow predicate per lane (1 = carry generated, 0 = no carry).
- **constraints and limitations:** This is a carry-chain integer add instruction set. On the current A5 instruction set, it SHOULD be treated as an unsigned integer operation. The carry flag is per-lane and fits in a 1-bit predicate register.
- **Use case:** Arbitrary-precision integer arithmetic (multi-precision addition), flag propagation in numerical kernels.

---

### `pto.vsubc`

- **syntax:** `%result, %borrow = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 RV:** `RV_VSUBC`; **Latency:** 7 (i32, unsigned borrow semantics)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);  // borrow predicate: 1 if borrow occurred
}
```

- **inputs:** `%lhs` and `%rhs` are subtracted lane-wise and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic difference and `%borrow` marks lanes that borrowed (1 = borrow generated, 0 = no borrow).
- **constraints and limitations:** This operation SHOULD be treated as an unsigned 32-bit carry-chain instruction set unless and until the verifier states otherwise.
- **Use case:** Arbitrary-precision integer arithmetic (multi-precision subtraction), borrow propagation.

---

## Typical Usage

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```
