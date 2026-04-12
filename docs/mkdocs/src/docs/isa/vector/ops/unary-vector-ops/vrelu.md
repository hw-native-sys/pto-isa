<!-- Generated from `docs/isa/vector/ops/unary-vector-ops/vrelu.md` -->

# pto.vrelu

Standalone reference page for `pto.vrelu`. This page belongs to the [Unary Vector Ops](../../unary-vector-ops.md) family in the PTO ISA manual.

## Summary

`%result` holds `max(input[i], 0)` per active lane.

## Mechanism

`pto.vrelu` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects active lanes.

## Expected Outputs

`%result` holds `max(input[i], 0)` per active lane.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only floating-point element types are legal
  on the current A5 surface described here.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

## Bitwise

## Related Ops / Family Links

- Family overview: [Unary Vector Ops](../../unary-vector-ops.md)
- Previous op in family: [pto.vrec](./vrec.md)
- Next op in family: [pto.vnot](./vnot.md)
