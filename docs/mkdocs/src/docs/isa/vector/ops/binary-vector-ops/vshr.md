<!-- Generated from `docs/isa/vector/ops/binary-vector-ops/vshr.md` -->

# pto.vshr

Standalone reference page for `pto.vshr`. This page belongs to the [Binary Vector Ops](../../binary-vector-ops.md) family in the PTO ISA manual.

## Summary

`%result` is the shifted vector.

## Mechanism

`pto.vshr` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `all integer types`.

## Inputs

`%lhs` supplies the shifted value, `%rhs` supplies the per-lane
  shift amount, and `%mask` selects active lanes.

## Expected Outputs

`%result` is the shifted vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Integer element types only. Signedness of the
  element type determines arithmetic vs logical behavior.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `all integer types`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

## Carry Operations

## Related Ops / Family Links

- Family overview: [Binary Vector Ops](../../binary-vector-ops.md)
- Previous op in family: [pto.vshl](./vshl.md)
- Next op in family: [pto.vaddc](./vaddc.md)
