<!-- Generated from `docs/isa/vector/ops/data-rearrangement/vslide.md` -->

# pto.vslide

Standalone reference page for `pto.vslide`. This page belongs to the [Data Rearrangement](../../data-rearrangement.md) family in the PTO ISA manual.

## Summary

Concatenate two vectors and extract N-element window at offset.

## Mechanism

`pto.vslide` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vslide %src0, %src1, %amt : !pto.vreg<NxT>, !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>
```

## Inputs

`%src0` and `%src1` provide the concatenated source window and
  `%amt` selects the extraction offset.

## Expected Outputs

`%result` is the extracted destination window.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

`pto.vslide` operates on the logical
  concatenation of `%src1` and `%src0`. The source order and extraction offset
  MUST be preserved exactly.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
// Conceptually: tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
if (amt >= 0)
    for (int i = 0; i < N; i++)
        dst[i] = (i >= amt) ? src0[i - amt] : src1[N - amt + i];
```

## Detailed Notes

```c
// Conceptually: tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
if (amt >= 0)
    for (int i = 0; i < N; i++)
        dst[i] = (i >= amt) ? src0[i - amt] : src1[N - amt + i];
```

**Use case:** Sliding window operations, shift register patterns.

## Related Ops / Family Links

- Family overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in family: [pto.vdintlv](./vdintlv.md)
- Next op in family: [pto.vshift](./vshift.md)
