<!-- Generated from `docs/isa/vector/ops/data-rearrangement/vperm.md` -->

# pto.vperm

Standalone reference page for `pto.vperm`. This page belongs to the [Data Rearrangement](../../data-rearrangement.md) family in the PTO ISA manual.

## Summary

In-register permute (table lookup). **Not** memory gather.

## Mechanism

`pto.vperm` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vperm %src, %index : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>
```

## Inputs

`%src` is the source vector and `%index` supplies per-lane source
  indices.

## Expected Outputs

`%result` is the permuted vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is an in-register permutation family.
  `%index` values outside the legal range follow the wrap/clamp behavior of the
  selected form.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

**Note:** This operates on register contents, unlike `pto.vgather2` which reads from UB memory.

## Related Ops / Family Links

- Family overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in family: [pto.vusqz](./vusqz.md)
- Next op in family: [pto.vpack](./vpack.md)
