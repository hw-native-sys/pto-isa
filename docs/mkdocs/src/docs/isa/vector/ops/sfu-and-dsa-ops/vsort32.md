<!-- Generated from `docs/isa/vector/ops/sfu-and-dsa-ops/vsort32.md` -->

# pto.vsort32

Standalone reference page for `pto.vsort32`. This page belongs to the [SFU And DSA Ops](../../sfu-and-dsa-ops.md) family in the PTO ISA manual.

## Summary

Sort 32 elements in UB.

## Mechanism

`pto.vsort32` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the family can be reasoned about at the ISA level.

## Syntax

```mlir
pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## Inputs

`%dest` and `%src` are UB pointers and `%config` is the ISA
  control/config word.

## Expected Outputs

This op writes UB memory and returns no SSA value.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is a UB-to-UB accelerator helper, not a
  pure vector-register op.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## Detailed Notes

The family overview carries the remaining shared rules for this operation.

## Related Ops / Family Links

- Family overview: [SFU And DSA Ops](../../sfu-and-dsa-ops.md)
- Previous op in family: [pto.vtranspose](./vtranspose.md)
- Next op in family: [pto.vmrgsort](./vmrgsort.md)
