<!-- Generated from `docs/isa/vector/ops/binary-vector-ops/vmul.md` -->

# pto.vmul

Standalone reference page for `pto.vmul`. This page belongs to the [Binary Vector Ops](../../binary-vector-ops.md) family in the PTO ISA manual.

## Summary

`%result` is the lane-wise product.

## Mechanism

`pto.vmul` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `i16-i32, f16, bf16, f32 (**NOT** i8/u8)`.

## Inputs

`%lhs` and `%rhs` are multiplied lane-wise; `%mask` selects
  active lanes.

## Expected Outputs

`%result` is the lane-wise product.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

The current A5 profile excludes `i8/u8`
  forms from this surface.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i16-i32, f16, bf16, f32 (**NOT** i8/u8)`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

## Related Ops / Family Links

- Family overview: [Binary Vector Ops](../../binary-vector-ops.md)
- Previous op in family: [pto.vsub](./vsub.md)
- Next op in family: [pto.vdiv](./vdiv.md)
