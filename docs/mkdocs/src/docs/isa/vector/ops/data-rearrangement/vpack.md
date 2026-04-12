<!-- Generated from `docs/isa/vector/ops/data-rearrangement/vpack.md` -->

# pto.vpack

Standalone reference page for `pto.vpack`. This page belongs to the [Data Rearrangement](../../data-rearrangement.md) family in the PTO ISA manual.

## Summary

Narrowing pack — two wide vectors to one narrow vector.

## Mechanism

`pto.vpack` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>
```

## Inputs

`%src0` and `%src1` are wide source vectors and `%part` selects
  the packing submode.

## Expected Outputs

`%result` is the packed narrow vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Packing is a narrowing conversion. Source
  values that do not fit the destination width follow the truncation semantics
  of the selected packing mode.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
// e.g., two vreg<64xi32> → one vreg<128xi16>
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

## Detailed Notes

```c
// e.g., two vreg<64xi32> → one vreg<128xi16>
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

## Related Ops / Family Links

- Family overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in family: [pto.vperm](./vperm.md)
- Next op in family: [pto.vsunpack](./vsunpack.md)
