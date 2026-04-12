<!-- Generated from `docs/isa/vector/ops/data-rearrangement/vsunpack.md` -->

# pto.vsunpack

Standalone reference page for `pto.vsunpack`. This page belongs to the [Data Rearrangement](../../data-rearrangement.md) family in the PTO ISA manual.

## Summary

Sign-extending unpack — narrow to wide (half).

## Mechanism

`pto.vsunpack` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>
```

## Inputs

`%src` is the packed narrow vector and `%part` selects which half
  is unpacked.

## Expected Outputs

`%result` is the widened vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is the sign-extending unpack family.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

## Detailed Notes

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

## Related Ops / Family Links

- Family overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in family: [pto.vpack](./vpack.md)
- Next op in family: [pto.vzunpack](./vzunpack.md)
