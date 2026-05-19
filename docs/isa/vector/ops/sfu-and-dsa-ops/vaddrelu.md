# pto.vaddrelu

`pto.vaddrelu` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Fused add + ReLU: computes addition followed by ReLU in a single fused operation.

## Mechanism

Fused add + ReLU: `dst[i] = max(0, lhs[i] + rhs[i])`. Computes addition followed by ReLU in a single fused operation. This combines two separate ops into one hardware instruction, eliminating an intermediate register write and improving throughput.

## Syntax

### PTO Assembly Form

```text
vaddrelu %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vaddrelu %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>) -> !pto.vreg<NxT>
```

Documented A5 types: `f16, f32`.

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `%lhs` | `!pto.vreg<NxT>` | Left-hand source vector register |
|| `%rhs` | `!pto.vreg<NxT>` | Right-hand source vector register |
|| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `%result` | `!pto.vreg<NxT>` | Fused add-then-ReLU result on active lanes |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - **Type match**: `%lhs`, `%rhs`, and `%result` MUST have identical element types.
    - **Width match**: All three registers MUST have the same vector width `N`.
    - **Mask width**: `%mask` MUST have width equal to `N`.
    - **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the computation.
    - **Floating-point only**: Fused add-ReLU is defined for floating-point element types on the current documented instruction set.

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

### Fused add + ReLU

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] + src1[i], 0);
```

### MLIR form

```mlir
%result = pto.vaddrelu %lhs, %rhs, %mask : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>) -> !pto.vreg<64xf32>
```

### C++ intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

Mask<64> mask;
mask.set_all(true);

VADDRELU(vdst, va, vb, mask);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vexpdif](./vexpdif.md)
- Next op in instruction set: [pto.vsubrelu](./vsubrelu.md)
