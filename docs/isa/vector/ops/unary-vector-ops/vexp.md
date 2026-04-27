# pto.vexp

`pto.vexp` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

Lane-wise exponential: computes `exp(src[i])` for each active lane. Active lanes compute `e^src[i]`; inactive lanes leave the destination register element unchanged.

## Mechanism

`pto.vexp` is a `pto.v*` compute operation. It applies the exponential function to each active lane independently:

For each lane `i` where the predicate mask bit is 1:

$$ \mathrm{dst}_i = \exp(\mathrm{src}_i) $$

Active lanes (`mask[i] == 1`): the exponential is computed and written to `dst[i]`.

Inactive lanes (`mask[i] == 0`): `dst[i]` is **unmodified** (preserves the prior value in the destination register).

## Syntax

### PTO Assembly Form

```mlir
%result = pto.vexp %input, %mask : (!pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vexp %input, %mask : (!pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vexp ins(%input, %mask : !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

```cpp
vector_f32 dst;
vector_f32 src;
vector_bool mask;
vexp(dst, src, mask);
```

## C Semantics

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = expf(src[i]);
    // else: dst[i] unchanged
```

Where `N` is the vector lane count determined by the element type:

| Element Type | Lanes (N) | Notes |
|-------------|:-----------:|-------|
| `f32` | 64 | |
| `f16`, `bf16` | 128 | |
| `i8`, `u8` | 256 | |

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%input` | `!pto.vreg<NxT>` | Source vector register; holds the input values |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Destination vector register; holds `exp(input[i])` on active lanes; inactive lanes are unmodified |

## Side Effects

None. This operation has no architectural side effects beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type match**: Source and destination registers MUST have identical element types.
- **Width match**: Source and destination registers MUST have the same vector width `N`.
- **Mask width**: Mask width MUST equal `N` (logical lane count).
- **Active lanes**: Only lanes where `mask[i] == 1` participate in the computation.
- **Inactive lanes**: Destination elements at inactive lanes are **unmodified** — do not assume they are zeroed.
- **Domain**: Input values follow the floating-point exception rules of the target profile for overflow, underflow, and NaN.

## Exceptions

- The verifier rejects illegal operand type mismatches, width mismatches, or mask width mismatches.
- Overflow and NaN behavior is target-defined; code MUST NOT rely on specific exceptional values unless explicitly documented.
- Any additional illegality stated in the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set overview page is part of the contract.

## Target-Profile Restrictions

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| `f32` | Simulated | Simulated | Supported |
| `f16` | Simulated | Simulated | Supported |

A5 is the primary concrete profile for vector transcendental operations. CPU simulation and A2/A3-class targets emulate `pto.v*` transcendental operations using scalar loops while preserving the visible PTO contract.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | A5 RV |
|---|---|---|
| `f32` | 16 | `RV_VEXP` |
| `f16` | 21 | `RV_VEXP` |

### A2/A3 Throughput

| Metric | Value (f32) | Value (f16) |
|--------|-------------|-------------|
| Startup latency | 13 (`A2A3_STARTUP_REDUCE`) | 13 |
| Completion latency | 26 (`A2A3_COMPL_FP32_EXP`) | 28 (`A2A3_COMPL_FP16_EXP`) |
| Per-repeat throughput | 2 | 4 |
| Pipeline interval | 18 | 18 |

**Example**: 1024 f32 elements with 16 iterations:

```
A5 (pipelined): 16 + 15×2 = 46 cycles
A2/A3: 13 + 26 + 32 + 270 = 341 cycles
```

**Performance note**: For numerically stable softmax, prefer `vexpdiff` (fused exp-diff) over `vexp` + `vsub` since it avoids a separate max-subtraction kernel and has better combined throughput.

---

## Examples

### Softmax numerator (numerically stable)

```mlir
// Softmax: exp(x - max) for numerical stability
%max_bc = pto.vlds %ub_max[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%sub = pto.vsub %x, %max_bc, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

### C++ usage

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void exp_vector(VReg<64, float>& dst, const VReg<64, float>& src, Mask<64>& mask) {
    VEXP(dst, src, mask);
}
```

## Detailed Notes

The exponential function `exp(x)` computes `e^x` where `e ≈ 2.71828`. On floating-point targets:

- `exp(+INF) = +INF`
- `exp(-INF) = +0`
- `exp(NaN) = NaN`
- Very large positive inputs may overflow to `+INF`
- Very large negative inputs may underflow to `+0`

For numerically stable softmax, prefer computing `exp(x - max(x))` rather than `exp(x)` directly to avoid overflow.

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vneg](./vneg.md)
- Next op in instruction set: [pto.vln](./vln.md)
- Vector instruction overview: [Vector Instructions](../../../instruction-families/vector-families.md)
- Type system: [Type System](../../../state-and-types/type-system.md)
