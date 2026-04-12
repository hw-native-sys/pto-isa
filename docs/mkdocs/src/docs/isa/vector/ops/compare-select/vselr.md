<!-- Generated from `docs/isa/vector/ops/compare-select/vselr.md` -->

# pto.vselr

Standalone reference page for `pto.vselr`. This page belongs to the [Compare And Select](../../compare-select.md) family in the PTO ISA manual.

## Summary

Select with reversed mask semantics.

## Mechanism

`pto.vselr` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## Inputs

`%src0` and `%src1` are the source vectors.

## Expected Outputs

`%result` is the selected vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This family preserves reversed-select
  semantics. If the concrete lowering uses an implicit predicate source, that
  predicate source MUST be documented by the surrounding IR pattern.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src1[i] : src0[i];  // reversed from vsel
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src1[i] : src0[i];  // reversed from vsel
```

## Related Ops / Family Links

- Family overview: [Compare And Select](../../compare-select.md)
- Previous op in family: [pto.vsel](./vsel.md)
- Next op in family: [pto.vselrv2](./vselrv2.md)
