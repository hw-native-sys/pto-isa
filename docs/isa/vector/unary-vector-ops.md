# Vector Instruction Set: Unary Vector Instructions

Single-input `pto.v*` compute instruction sets are defined here. Unless a form states otherwise, the vector-register shape, active-lane mask semantics, and target-profile restrictions below define the portable contract.

> **Category:** Single-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take one vector input and produce one vector output.

---

## Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand. For this instruction set, inactive lanes follow the
  predication behavior of the selected instruction form: zeroing forms
  zero-fill inactive lanes, while merging forms preserve the destination value.
- `%result` is the destination vector register value. Unless stated otherwise,
  `%result` has the same lane count and element type as `%input`.

---

## Execution Model: vecscope

Unary vector operations execute inside a `pto.vecscope { ... }` region, which establishes the Vector Core's execution context. The `pto.vecscope` region is implicitly scoped to `PIPE_V`; all vector instructions inside it are issued to the Vector pipeline.

**Typical loop structure:**

```mlir
pto.vecscope {
  %remaining_init = arith.constant 1024 : i32
  %_:1 = scf.for %offset = %c0 to %total step %c64
      iter_args(%remaining = %remaining_init) -> (i32) {
    %mask, %next_remaining = pto.plt_b32 %remaining : i32 -> !pto.mask<G>, i32
    %vec = pto.vlds %ub_in[%offset] : !pto.ptr -> !pto.vreg<64xf32>
    %out = pto.vabs %vec, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %out, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr, !pto.mask<b32>
    scf.yield %next_remaining : i32
  }
}
```

**Predicate generation:** `%mask = pto.pset_b32 "PAT_ALL"` creates a full-active mask; `pto.plt_b32 %remaining` generates a tail mask based on the number of remaining elements.

---

## A5 Latency and Throughput (Ascend910_9599)

> All values are **popped→retire** cycle counts on the cycle-accurate simulator. Float16 types use `aclFloat16` tracing.

### Latency Summary Table

| PTO op | A5 RV (CA) | f32 | f16 | bf16 | i32 | i16 | i8 |
|--------|-------------|-----|------|------|-----|------|-----|
| `pto.vabs` | `RV_VABS_FP` | 5 | 5 | — | 5 | 5 | 5 |
| `pto.vneg` | `RV_VMULS` | 8 | 8 | — | 8 | 8 | 8 |
| `pto.vexp` | `RV_VEXP` | 16 | 21 | — | — | — | — |
| `pto.vln` | `RV_VLN` | 18 | 23 | — | — | — | — |
| `pto.vsqrt` | `RV_VSQRT` | 17 | 22 | — | — | — | — |
| `pto.vrelu` | `RV_VRELU` | 5 | 5 | — | — | — | — |
| `pto.vrec` | `RV_VREC` | (see note) | (see note) | — | — | — | — |
| `pto.vrsqrt` | `RV_VRSQRT` | (see note) | (see note) | — | — | — | — |
| `pto.vnot` | `RV_VNOT` | — | — | — | 5 | 5 | 5 |
| `pto.vbcnt` | — | — | — | — | (per-lane) | (per-lane) | (per-lane) |
| `pto.vcls` | — | — | — | — | (per-lane) | (per-lane) | (per-lane) |
| `pto.vmov` | `RV_VLD` (proxy) | 9 | 9 | — | 9 | 9 | 9 |

!!! note "Note on reciprocals:"
    `vrec` and `vrsqrt` are synthesized from `vdiv` and `vsqrt` respectively; their latency matches the corresponding divide instruction throughput.

### A2/A3 Latency and Throughput

| Metric | Constant | Value (cycles) | Applies To |
|--------|-----------|---------------|------------|
| Startup latency (reduce/transcendental) | `A2A3_STARTUP_REDUCE` | 13 | `vexp`, `vsqrt`, `vln` |
| Startup latency (binary/arith) | `A2A3_STARTUP_BINARY` | 14 | `vabs`, `vneg`, `vadd`, `vmul` |
| Completion: FP binary ops | `A2A3_COMPL_FP_BINOP` | 19 | `vabs`, `vneg`, `vadd` (f32), `vsub` (f32) |
| Completion: INT binary ops | `A2A3_COMPL_INT_BINOP` | 17 | `vabs`/`vadd`/`vsub` (int16/i32) |
| Completion: FP transcendental | `A2A3_COMPL_FP32_EXP` | 26 | `vexp` (f32) |
| Completion: FP transcendental | `A2A3_COMPL_FP16_EXP` | 28 | `vexp` (f16) |
| Completion: FP transcendental | `A2A3_COMPL_FP32_SQRT` | 27 | `vsqrt` (f32) |
| Completion: FP transcendental | `A2A3_COMPL_FP16_SQRT` | 29 | `vsqrt` (f16) |
| Per-repeat throughput | `A2A3_RPT_1` | 1 | scalar/unary/simple ops |
| Per-repeat throughput | `A2A3_RPT_2` | 2 | binary ops (`vadd`, `vmul`) |
| Per-repeat throughput | `A2A3_RPT_4` | 4 | transcendental ops (`vexp`, `vsqrt` f16) |
| Pipeline interval | `A2A3_INTERVAL` | 18 | all vector ops |
| Pipeline interval (vmov/vcopy) | `A2A3_INTERVAL_VCOPY` | 13 | `vmov` |

**Cycle model (A2/A3):**

```
total_cycles = startup + completion + repeats × per_repeat + (repeats - 1) × interval
```

---

## Arithmetic

### `pto.vabs`

- **syntax:** `%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VABS_FP`; **Latency:** 5 (f32/f16), 5 (i32/i16/i8)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

- **inputs:** `%input` supplies the source lanes and `%mask` selects which lanes participate.
- **outputs:** `%result` receives the lane-wise absolute values.
- **constraints and limitations:** Source and result types MUST match. Integer overflow on the most-negative signed value follows the target-defined behavior.

---

### `pto.vneg`

- **syntax:** `%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VMULS` (uses scalar-multiply hardware); **Latency:** 8 (f32/f16), 8 (i32/i16/i8)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise arithmetic negation.
- **constraints and limitations:** Source and result types MUST match.

---

## Transcendental

### `pto.vexp`

- **syntax:** `%result = pto.vexp %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VEXP`; **Latency:** 16 (f32), 21 (f16)
- **A2/A3 throughput:** 2 cycles/repeat (f32), 4 cycles/repeat (f16); **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `exp(input[i])` per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
- **Performance note:** f32 is significantly faster than f16 on A5 (16 vs 21 cycles). For f16, prefer `vexpdif` (fused exp-diff) for numerical stability in softmax.

---

### `pto.vln`

- **syntax:** `%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VLN`; **Latency:** 18 (f32), 23 (f16)
- **A2/A3 throughput:** 2 cycles/repeat (f32), 4 cycles/repeat (f16); **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the natural logarithm per active lane.
- **constraints and limitations:** Only floating-point element types are legal. For real-number semantics, active inputs SHOULD be strictly positive; non-positive inputs follow the target's exception/NaN rules.

---

### `pto.vsqrt`

- **syntax:** `%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VSQRT`; **Latency:** 17 (f32), 22 (f16)
- **A2/A3 throughput:** 2 cycles/repeat (f32), 4 cycles/repeat (f16); **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the square root per active lane.
- **constraints and limitations:** Only floating-point element types are legal. Negative active inputs follow the target's exception/NaN rules.
- **Performance note:** `vrsqrt` (reciprocal square root) uses the same hardware as `vsqrt` and costs equivalent cycles.

---

### `pto.vrsqrt`

- **syntax:** `%result = pto.vrsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **Latency:** equivalent to `vsqrt`; uses `RV_VRSQRT` hardware path
- **A2/A3 throughput:** 2 cycles/repeat (f32), 4 cycles/repeat (f16); **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / sqrtf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds reciprocal-square-root values per active lane.
- **constraints and limitations:** Only floating-point element types are legal. Active inputs containing `+0` or `-0` follow the target's divide-style exceptional behavior.

---

### `pto.vrec`

- **syntax:** `%result = pto.vrec %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **Latency:** synthesized via `vdiv`; throughput matches `vdiv`
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the reciprocal per active lane.
- **constraints and limitations:** Only floating-point element types are legal. Active inputs containing `+0` or `-0` follow the target's divide-style exceptional behavior.

---

## Activation

### `pto.vrelu`

- **syntax:** `%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VRELU`; **Latency:** 5 (f32/f16)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `max(input[i], 0)` per active lane.
- **constraints and limitations:** Only floating-point element types are legal on the current A5 instruction set described here.
- **Performance note:** `vrelu` is the lowest-latency unary operation (5 cycles). Use `vlrelu` for leaky-ReLU (adds one scalar multiply).

---

## Bitwise

### `pto.vnot`

- **syntax:** `%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VNOT`; **Latency:** 5 (integer types only)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 18 cycles

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the lane-wise bitwise inversion.
- **constraints and limitations:** Integer element types only.

---

### `pto.vbcnt`

- **syntax:** `%result = pto.vbcnt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Population count — counts the number of set bits in each lane's element.

```c
for (int i = 0; i < N; i++)
    dst[i] = __builtin_popcount(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the population count for each active lane.
- **constraints and limitations:** Integer element types only. The count is over the source element width, not over the full vector register.

---

### `pto.vcls`

- **syntax:** `%result = pto.vcls %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Count leading sign bits — for a signed integer, counts how many bits from the MSB are equal to the sign bit.

```c
for (int i = 0; i < N; i++)
    dst[i] = count_leading_sign_bits(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the leading-sign-bit count per active lane.
- **constraints and limitations:** Integer element types only. This operation is sign-aware, so signed interpretation matters.

---

## Movement

### `pto.vmov`

- **syntax:** `%result = pto.vmov %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 RV:** `RV_VLD` (proxy); **Latency:** 9 (f32/f16), 9 (integer)
- **A2/A3 throughput:** 1 cycle/repeat; **interval:** 13 cycles (`A2A3_INTERVAL_VCOPY`)

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` is a copy of the source vector.
- **constraints and limitations:** Predicated `pto.vmov` behaves like a masked copy, while the unpredicated form behaves like a full-register copy.

---

## Typical Usage

```mlir
// Softmax numerator: exp(x - max) using vexp
%sub = pto.vsub %x, %max_broadcast, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Reciprocal for division
%sum_rcp = pto.vrec %sum, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// ReLU activation (lowest latency unary on A5: 5 cycles)
%activated = pto.vrelu %linear_out, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```
