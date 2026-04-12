<!-- Generated from `docs/isa/vector/ops/binary-vector-ops/vsubc.md` -->

# pto.vsubc

Standalone reference page for `pto.vsubc`. This page belongs to the [Binary Vector Ops](../../binary-vector-ops.md) family in the PTO ISA manual.

## Summary

Subtract with borrow output.

## Mechanism

`pto.vsubc` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result, %borrow = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask
```

## Inputs

`%lhs` and `%rhs` are subtracted lane-wise and `%mask` selects
  active lanes.

## Expected Outputs

`%result` is the arithmetic difference and `%borrow` marks lanes
  that borrowed.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This operation SHOULD be treated as an
  unsigned 32-bit carry-chain family unless and until the verifier states
  otherwise.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask -> !pto.vreg<64xi32>
```

## Detailed Notes

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

## Typical Usage

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask -> !pto.vreg<64xi32>
```

## Related Ops / Family Links

- Family overview: [Binary Vector Ops](../../binary-vector-ops.md)
- Previous op in family: [pto.vaddc](./vaddc.md)
