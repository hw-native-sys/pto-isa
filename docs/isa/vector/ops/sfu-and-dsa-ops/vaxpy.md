# pto.vaxpy

`pto.vaxpy` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

AXPY — scalar-vector multiply-add: computes `a * x + y` in a single fused operation.

## Mechanism

Fused multiply-add: `dst[i] = a * x[i] + y[i]`. Computes element-wise scaled addition in a single fused operation. This is equivalent to `vmula` with a broadcast scalar. The scalar multiplier `a` is applied to each element of vector `x`, then added to the corresponding element of vector `y`.

## Syntax

### PTO Assembly Form

```text
vaxpy %dst, %x, %y, %alpha, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vaxpy %x, %y, %alpha, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, T, !pto.mask<G>) -> !pto.vreg<NxT>
```

Documented A5 types: `f16, f32`.

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `%x` | `!pto.vreg<NxT>` | Scaled vector operand |
|| `%y` | `!pto.vreg<NxT>` | Addend vector operand |
|| `%alpha` | `T` (scalar) | Scalar multiplier |
|| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 are active |

Both source vectors MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `%result` | `!pto.vreg<NxT>` | Fused AXPY result: `dst[i] = alpha * x[i] + y[i]` |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - **Type match**: `%x`, `%y`, and `%result` MUST have identical element types.
    - **Width match**: Both vector registers MUST have the same vector width `N`.
    - **Mask width**: `%mask` MUST have width equal to `N`.
    - **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the computation.
    - **Floating-point only**: AXPY is defined for floating-point element types on the current documented instruction set.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `f16, f32`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

SFU operations have higher latency than standard arithmetic ops. Consult the target profile's performance model for cycle-accurate estimates.

### A2/A3 Throughput

|| Metric | Value | Constant |
||--------|-------|----------|
|| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
|| Completion latency | 26 | `A2A3_COMPL_FP32_EXP` |
|| Per-repeat throughput | 2 | `A2A3_RPT_2` |
|| Pipeline interval | 18 | `A2A3_INTERVAL` |

---

## Examples

### Scalar-vector multiply-add (AXPY)

```c
for (int i = 0; i < N; i++)
    dst[i] = alpha * src0[i] + src1[i];
```

### MLIR form

```mlir
%result = pto.vaxpy %x, %y, %alpha, %mask : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, f32, !pto.mask<b32>) -> !pto.vreg<64xf32>
```

### C++ intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

Mask<64> mask;
mask.set_all(true);
float alpha = 2.5f;

VAXPY(vdst, vx, vy, alpha, mask);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vsubrelu](./vsubrelu.md)
- Next op in instruction set: [pto.vaddreluconv](./vaddreluconv.md)
