<!-- Generated from `docs/isa/vector/ops/vec-scalar-ops/vmuls.md` -->

# pto.vmuls

Standalone reference page for `pto.vmuls`. This page belongs to the [Vec Scalar Ops](../../vec-scalar-ops.md) family in the PTO ISA manual.

## Summary

`%result` is the lane-wise product.

## Mechanism

`pto.vmuls` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

`%input`, `%scalar`, and `%mask` as above.

## Expected Outputs

`%result` is the lane-wise product.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Supported element types are hardware-family
  specific; the current PTO ISA vector surface documentation covers the common numeric cases.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

## Related Ops / Family Links

- Family overview: [Vec Scalar Ops](../../vec-scalar-ops.md)
- Previous op in family: [pto.vsubs](./vsubs.md)
- Next op in family: [pto.vmaxs](./vmaxs.md)
