<!-- Generated from `docs/isa/vector/ops/binary-vector-ops/vdiv.md` -->

# pto.vdiv

Standalone reference page for `pto.vdiv`. This page belongs to the [Binary Vector Ops](../../binary-vector-ops.md) family in the PTO ISA manual.

## Summary

`%result` is the lane-wise quotient.

## Mechanism

`pto.vdiv` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32 only (no integer division)`.

## Inputs

`%lhs` is the numerator, `%rhs` is the denominator, and `%mask`
  selects active lanes.

## Expected Outputs

`%result` is the lane-wise quotient.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Floating-point element types only. Active
  denominators containing `+0` or `-0` follow the target's exceptional
  behavior.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Target-defined numeric exceptional behavior, such as divide-by-zero or out-of-domain inputs, remains subject to the selected backend profile unless this page narrows it further.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32 only (no integer division)`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

## Related Ops / Family Links

- Family overview: [Binary Vector Ops](../../binary-vector-ops.md)
- Previous op in family: [pto.vmul](./vmul.md)
- Next op in family: [pto.vmax](./vmax.md)
