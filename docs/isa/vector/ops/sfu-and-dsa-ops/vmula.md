# pto.vmula

`pto.vmula` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Multiply-accumulate: computes `lhs * rhs + add` in a single fused operation.

## Mechanism

Fused multiply-add: `dst[i] = lhs[i] * rhs[i] + add[i]`. Computes per-lane multiplication of two vectors, then adds a third vector, in a single fused operation. This combines multiplication and addition into one hardware instruction, eliminating an intermediate register write and improving throughput and numerical precision.

## Syntax

### PTO Assembly Form

```text
vmula %dst, %add, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vmula %add, %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>) -> !pto.vreg<NxT>
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `%add` | `!pto.vreg<NxT>` | Accumulator input vector |
|| `%lhs` | `!pto.vreg<NxT>` | Left-hand multiplicand vector |
|| `%rhs` | `!pto.vreg<NxT>` | Right-hand multiplicand vector |
|| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 are active |

All four operands MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `%result` | `!pto.vreg<NxT>` | Multiply-accumulate result: `dst[i] = add[i] + lhs[i] * rhs[i]` |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - **Type match**: All four registers MUST have identical element types.
    - **Width match**: All four registers MUST have the same vector width `N`.
    - **Mask width**: `%mask` MUST have width equal to `N`.
    - **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the computation.
    - **Fused semantics**: `pto.vmula` is a fused multiply-accumulate operation and is not always interchangeable with separate `vmul` plus `vadd`. The fused form provides better numerical precision and performance.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
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

### Fused multiply-accumulate

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = acc[i] + lhs[i] * rhs[i];
```

### MLIR form

```mlir
%result = pto.vmula %acc, %lhs, %rhs, %mask : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>) -> !pto.vreg<64xf32>
```

### C++ intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

Mask<64> mask;
mask.set_all(true);

VMULA(vdst, vacc, vlhs, vrhs, mask);
```

## Index Generation

This operation does not generate indices.

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vmull](./vmull.md)
- Next op in instruction set: [pto.vtranspose](./vtranspose.md)
