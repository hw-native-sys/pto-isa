<!-- Generated from `docs/isa/vector/ops/data-rearrangement/vdintlv.md` -->

# pto.vdintlv

Standalone reference page for `pto.vdintlv`. This page belongs to the [Data Rearrangement](../../data-rearrangement.md) family in the PTO ISA manual.

## Summary

Deinterleave elements into even/odd.

## Mechanism

`pto.vdintlv` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## Inputs

`%lhs` and `%rhs` represent the interleaved source stream in the
  current PTO ISA vector surface representation.

## Expected Outputs

`%low` and `%high` are the separated destination vectors.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

The two outputs form the even/odd
  deinterleave result pair, and their ordering MUST be preserved.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

## Detailed Notes

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

## Slide / Shift

## Related Ops / Family Links

- Family overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in family: [pto.vintlv](./vintlv.md)
- Next op in family: [pto.vslide](./vslide.md)
