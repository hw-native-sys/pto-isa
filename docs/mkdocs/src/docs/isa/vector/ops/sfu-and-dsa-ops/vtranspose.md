<!-- Generated from `docs/isa/vector/ops/sfu-and-dsa-ops/vtranspose.md` -->

# pto.vtranspose

Standalone reference page for `pto.vtranspose`. This page belongs to the [SFU And DSA Ops](../../sfu-and-dsa-ops.md) family in the PTO ISA manual.

## Summary

UB-to-UB transpose operation (not vreg-to-vreg).

## Mechanism

`pto.vtranspose` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the family can be reasoned about at the ISA level.

## Syntax

```mlir
pto.vtranspose %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## Inputs

`%dest` and `%src` are UB pointers and `%config` is the ISA
  control/config word.

## Expected Outputs

This op writes UB memory and returns no SSA value.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is not a `vreg -> vreg` op even though
  it lives in the `pto.v*` namespace. Its correctness depends on the control
  word and UB layout contract.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
pto.vtranspose %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## Detailed Notes

**Note:** This operates on UB memory directly, not on vector registers.

## Sorting Operations

## Related Ops / Family Links

- Family overview: [SFU And DSA Ops](../../sfu-and-dsa-ops.md)
- Previous op in family: [pto.vmula](./vmula.md)
- Next op in family: [pto.vsort32](./vsort32.md)
