<!-- Generated from `docs/isa/vector/ops/unary-vector-ops/vbcnt.md` -->

# pto.vbcnt

Standalone reference page for `pto.vbcnt`. This page belongs to the [Unary Vector Ops](../../unary-vector-ops.md) family in the PTO ISA manual.

## Summary

`%result` holds the population count for each active lane.

## Mechanism

`pto.vbcnt` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vbcnt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `all integer types`.

## Inputs

`%input` is the source vector and `%mask` selects active lanes.

## Expected Outputs

`%result` holds the population count for each active lane.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Integer element types only. The count is
  over the source element width, not over the full vector register.

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
    dst[i] = __builtin_popcount(src[i]);
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = __builtin_popcount(src[i]);
```

## Related Ops / Family Links

- Family overview: [Unary Vector Ops](../../unary-vector-ops.md)
- Previous op in family: [pto.vnot](./vnot.md)
- Next op in family: [pto.vcls](./vcls.md)
