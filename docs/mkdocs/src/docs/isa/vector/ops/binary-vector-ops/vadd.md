<!-- Generated from `docs/isa/vector/ops/binary-vector-ops/vadd.md` -->

# pto.vadd

Standalone reference page for `pto.vadd`. This page belongs to the [Binary Vector Ops](../../binary-vector-ops.md) family in the PTO ISA manual.

## Summary

Lane-wise addition of two vector registers, producing a result vector register. Only lanes selected by the predicate mask are active; inactive lanes do not participate in the computation and their destination elements are left unchanged.

## Mechanism

`pto.vadd` is a `pto.v*` compute operation. It reads two source vector registers lane-by-lane, adds the corresponding elements, and writes the result to the destination vector register. The iteration domain covers all N lanes; the predicate mask determines which lanes are active.

For each lane `i` where the predicate is true:

$$ \mathrm{dst}_i = \mathrm{lhs}_i + \mathrm{rhs}_i $$

Lanes where the predicate is false are **inactive**: the destination register element at that lane is unmodified.

## Syntax

### PTO Assembly Form

```text
vadd %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vadd %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vadd ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename VecDst, typename VecLhs, typename VecRhs, typename MaskT, typename... WaitEvents>
PTO_INST RecordEvent VADD(VecDst& dst, const VecLhs& lhs, const VecRhs& rhs,
                          const MaskT& mask, WaitEvents&... events);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | Left-hand source vector register |
| `%rhs` | `!pto.vreg<NxT>` | Right-hand source vector register |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.vreg<NxT>` | Lane-wise sum on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type match**: `%lhs`, `%rhs`, and `%dst` MUST have identical element types.
- **Width match**: All three registers MUST have the same vector width `N`.
- **Mask width**: `%mask` MUST have width equal to `N`.
- **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the addition.
- **Inactive lanes**: Destination elements at inactive lanes are unmodified.

## Exceptions

- The verifier rejects illegal operand type mismatches, width mismatches, or mask width mismatches.
- Any additional illegality stated in the [Binary Vector Ops](../../binary-vector-ops.md) family page is also part of the contract.

## Target-Profile Restrictions

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| `f32` | Simulated | Simulated | Supported |
| `f16` / `bf16` | Simulated | Simulated | Supported |
| `i8`–`i64`, `u8`–`u64` | Simulated | Simulated | Supported |

A5 is the primary concrete profile for the vector surface. CPU simulation and A2/A3-class targets emulate `pto.v*` operations using scalar loops while preserving the visible PTO contract. Code that depends on specific performance characteristics or latency should treat those dependencies as target-profile-specific.

## Examples

### Full-vector addition (all lanes active)

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// All lanes active: mask set to all-ones
Mask<64> mask;
mask.set_all(true);  // predicate all-true

VADD(vdst, va, vb, mask);
```

### Partial predication

```mlir
// Only lanes where %cond is true participate in addition
%result = pto.vadd %va, %vb, %cond : (!pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask) -> !pto.vreg<128xf16>
```

### Complete vector-load / compute / vector-store pipeline

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vector_add(Ptr<ub_space_t, ub_t> ub_a, Ptr<ub_space_t, ub_t> ub_b,
                Ptr<ub_space_t, ub_t> ub_out, size_t count) {
    VReg<64, float> va, vb, vdst;
    Mask<64> mask;
    mask.set_all(true);

    VLDS(va, ub_a, "NORM");
    VLDS(vb, ub_b, "NORM");

    VADD(vdst, va, vb, mask);

    VSTS(vdst, ub_out);
}
```

## Related Ops / Family Links

- Family overview: [Binary Vector Ops](../../binary-vector-ops.md)
- Next op in family: [pto.vsub](./vsub.md)
- Vector surface overview: [Vector Instructions](../../instruction-surfaces/vector-instructions.md)
- Type system: [Type System](../../state-and-types/type-system.md)
