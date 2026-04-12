<!-- Generated from `docs/isa/vector/ops/conversion-ops/vci.md` -->

# pto.vci

Standalone reference page for `pto.vci`. This page belongs to the [Conversion Ops](../../conversion-ops.md) family in the PTO ISA manual.

## Summary

Standalone contract page for `pto.vci`.

## Mechanism

`pto.vci` belongs to the `pto.v*` conversion surface. It changes vector element interpretation, width, rounding, saturation, or index-generation state without leaving the vector-register model.

## Syntax


## Inputs

This operation follows the operand model of the [Conversion Ops](../../conversion-ops.md) family: SSA vector values carry payloads, masks gate active lanes when present, and family-specific attributes select rounding, selection, distribution, or fused-mode behavior.

## Expected Outputs

This form is primarily defined by the side effect it has on control state, predicate state, or memory. It does not publish a new payload SSA result beyond any explicit state outputs shown in the syntax.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This operation inherits the legality and operand-shape rules of its family overview. Any target-specific narrowing of element types, distributions, pipe/event spaces, or configuration tuples must be stated by the selected target profile.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
pto.vci
```

## Detailed Notes

The family overview carries the remaining shared rules for this operation.

## Related Ops / Family Links

- Family overview: [Conversion Ops](../../conversion-ops.md)
- Next op in family: [pto.vcvt](./vcvt.md)
