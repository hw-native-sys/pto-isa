# Vector Instruction Set: Conversion Ops

`pto.v*` conversion and index-generation instruction sets are defined here. Width changes, rounding rules, saturation, and part selection are target-visible constraints and must stay aligned with the PTO ISA verifier and lowering contracts.

> **Category:** Type conversion operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that convert between data types (float/int, narrowing/widening).

## Common Operand Model

- `%input` is the source vector register value.
- `%result` is the destination vector register value.
- `round_mode`, `sat`, and `part` control rounding, saturation, and lane-part
  selection in attribute form.
- The single `pto.vcvt` instruction set covers float-int, float-float, int-float, and
  int-int conversion instruction sets.

---

## `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- **semantics:** Generate a lane-index vector from a scalar seed/index value.
- **inputs:**
  `%index` is the scalar seed or base index.
- **outputs:**
  `%result` is the generated index vector.
- **constraints and limitations:**
  This is an index-generation instruction set, not a numeric conversion. `ORDER` and the
  result element type together determine how indices are generated.

---

## `pto.vcvt`

- **syntax:** `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Type conversion between float/int types with rounding control.

```c
for (int i = 0; i < min(N, M); i++)
    dst[i] = convert(src[i], T0, T1, round_mode);
```

- **inputs:**
  `%input` is the source vector; attributes select rounding, saturation, and
  even/odd placement when the conversion changes width.
- **outputs:**
  `%result` is the converted vector.
- **constraints and limitations:**
  Only documented source/destination type pairs are legal. `PART_EVEN` /
  `PART_ODD` is only meaningful for width-changing forms that pack two source
  streams into one destination register.

---

### Rounding Modes

| Mode | Description |
|------|-------------|
| `ROUND_R` | Round to nearest, ties to even (default) |
| `ROUND_A` | Round away from zero |
| `ROUND_F` | Round toward negative infinity (floor) |
| `ROUND_C` | Round toward positive infinity (ceil) |
| `ROUND_Z` | Round toward zero (truncate) |
| `ROUND_O` | Round to odd |

---

### Saturation Modes

| Mode | Description |
|------|-------------|
| `RS_ENABLE` | Saturate on overflow |
| `RS_DISABLE` | No saturation (wrap/undefined on overflow) |

---

### Part Modes (for width-changing conversions)

| Mode | Description |
|------|-------------|
| `PART_EVEN` | Output to even-indexed lanes |
| `PART_ODD` | Output to odd-indexed lanes |

---

### A5 Supported Conversions

**Float-Float (vcvtff):**
- f32 ↔ f16
- f32 ↔ bf16
- f16 ↔ bf16

**Float-Int (vcvtfi):**
- f16 → i16, f16 → i32
- f32 → i16, f32 → i32
- bf16 → i32

**Int-Float (vcvtif):**
- i16 → f16
- i32 → f32

---

### Width-Changing Conversion Pattern

For conversions that change width (e.g., f32→f16), use even/odd parts and combine:

```mlir
// Convert two f32 vectors to one f16 vector
%even = pto.vcvt %in0 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_EVEN"}
    : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_ODD"}
    : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
```

---

## `pto.vtrc`

- **syntax:** `%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Truncate/round float to integer-valued float (stays in float type).

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], round_mode);
```

- **inputs:**
  `%input` is the floating-point source vector and `ROUND_MODE` selects the
  truncation/rounding rule.
- **outputs:**
  `%result` is still a floating-point vector, but each active lane now carries
  an integer-valued floating-point result.
- **constraints and limitations:**
  This op does not change the element type. `ROUND_O` is supported for avoiding
  double-rounding errors during staged conversions.

**Example:**
```mlir
// Round to nearest integer, keep as float
%rounded = pto.vtrc %input, "ROUND_R" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
// input:  [1.4, 2.6, -1.5, 3.0]
// output: [1.0, 3.0, -2.0, 3.0]
```

---

## `pto.vbitcast`

- **syntax:** `%result = pto.vbitcast %input : !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Bitwise reinterpretation of a vreg vector without changing the underlying bit pattern. Performs a pure type cast that preserves the exact bits of each element, changing only their interpretation (for example, from floating-point to integer).
- **inputs:** `%input` is the source vector register value.
- **outputs:** `%result` is the reinterpreted vector register value.
- **constraints and limitations:**
  - Both source and result must be `!pto.vreg<...>` types.
  - Source and result vectors must have the same total bit width (currently 2048 bits): `N * bitwidth(T0) = M * bitwidth(T1) = 2048`.
  - Only integer and floating-point element types are supported.

See [`pto.vbitcast`](./ops/conversion-ops/vbitcast.md) for full details, type-pair examples, and the comparison with `pto.vcvt`.

---

## `pto.pbitcast`

- **syntax:** `%result = pto.pbitcast %input : !pto.mask<G0> -> !pto.mask<G1>`
- **semantics:** Bitwise reinterpretation of a predicate register without changing the underlying predicate-register image. Makes mask-family granularity reinterpretation explicit in VPTO IR when a producer and consumer expect different `!pto.mask<...>` views of the same hardware predicate state.
- **inputs:** `%input` is the source predicate register value.
- **outputs:** `%result` is the reinterpreted predicate register value.
- **constraints and limitations:**
  - Both source and result must be `!pto.mask<...>` types.
  - `pto.pbitcast` does not materialize or normalize predicate contents; it only changes which mask granularity the surrounding VPTO IR uses to interpret the same predicate bits.

See [`pto.pbitcast`](./ops/conversion-ops/pbitcast.md) for full details and examples.

---

## Typical Usage

```mlir
// Quantization: f32 → i8 with saturation
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%quantized = pto.vcvt %scaled {round_mode = "ROUND_R", sat = "RS_ENABLE"}
    : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
// Then narrow i32 → i8 via pack ops

// Mixed precision: bf16 → f32 for accumulation
%f32_vec = pto.vcvt %bf16_input {round_mode = "ROUND_R"}
    : !pto.vreg<128xbf16> -> !pto.vreg<64xf32>

// Floor for integer division
%floored = pto.vtrc %ratio, "ROUND_F" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%int_div = pto.vcvt %floored {round_mode = "ROUND_Z"}
    : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
```
