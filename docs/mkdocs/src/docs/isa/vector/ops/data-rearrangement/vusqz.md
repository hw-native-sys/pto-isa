<!-- Generated from `docs/isa/vector/ops/data-rearrangement/vusqz.md` -->

# pto.vusqz

Standalone reference page for `pto.vusqz`. This page belongs to the [Data Rearrangement](../../data-rearrangement.md) family in the PTO ISA manual.

## Summary

Expand — scatter front elements to active positions.

## Mechanism

`pto.vusqz` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vusqz %mask : !pto.mask -> !pto.vreg<NxT>
```

## Inputs

`%mask` is the expansion/placement predicate.

## Expected Outputs

`%result` is the expanded vector image.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

The source-front stream is implicit in the
  current surface. Lane placement for active and inactive positions MUST be
  preserved exactly.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

## Detailed Notes

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

## Permutation

## Related Ops / Family Links

- Family overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in family: [pto.vsqz](./vsqz.md)
- Next op in family: [pto.vperm](./vperm.md)
