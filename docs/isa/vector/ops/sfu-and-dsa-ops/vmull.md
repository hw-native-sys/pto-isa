# pto.vmull

`pto.vmull` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Multiply-subtract: computes `lhs * rhs - sub` in a single fused operation.

## Mechanism

Fused multiply-sub: `dst[i] = lhs[i] * rhs[i] - sub[i]`. Computes per-lane multiplication of two vectors, then subtracts a third vector, in a single fused operation. This combines multiplication and subtraction into one hardware instruction, eliminating an intermediate register write and improving throughput and numerical precision.

## Syntax

### PTO Assembly Form

```text
vmull %dst, %sub, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vmull %sub, %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>) -> !pto.vreg<NxT>
```

Documented A5 types: `i32/u32 (native 32×32→64 widening multiply)`.

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `%sub` | `!pto.vreg<NxT>` | Subtrahend input vector |
|| `%lhs` | `!pto.vreg<NxT>` | Left-hand multiplicand vector |
|| `%rhs` | `!pto.vreg<NxT>` | Right-hand multiplicand vector |
|| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 are active |

All four operands MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `%result` | `!pto.vreg<NxT>` | Multiply-subtract result: `dst[i] = sub[i] - lhs[i] * rhs[i]` |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - **Type match**: All four registers MUST have identical element types.
    - **Width match**: All four registers MUST have the same vector width `N`.
    - **Mask width**: `%mask` MUST have width equal to `N`.
    - **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the computation.
    - **Fused semantics**: `pto.vmull` is a fused multiply-subtract operation and is not always interchangeable with separate `vmul` plus `vsub`. The fused form provides better numerical precision and performance.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `i32/u32 (native 32×32→64 widening multiply)`.
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

### Fused multiply-subtract

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = sub[i] - lhs[i] * rhs[i];
```

### MLIR form

```mlir
%result = pto.vmull %sub, %lhs, %rhs, %mask : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32>) -> !pto.vreg<64xi32>
```

### C++ intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

Mask<64> mask;
mask.set_all(true);

VMULL(vdst, vsub, vlhs, vrhs, mask);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vmulconv](./vmulconv.md)
- Next op in instruction set: [pto.vmula](./vmula.md)
