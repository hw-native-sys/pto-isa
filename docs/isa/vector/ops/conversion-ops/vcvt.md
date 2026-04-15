# pto.vcvt

`pto.vcvt` is part of the [Conversion Ops](../../conversion-ops.md) instruction set.

## Summary

Standalone contract page for `pto.vcvt`.

## Mechanism

`pto.vcvt` belongs to the `pto.v*` conversion instruction set. It changes vector element interpretation, width, rounding, saturation, or index-generation state without leaving the vector-register model.

## Inputs

This operation follows the operand model of the [Conversion Ops](../../conversion-ops.md) instruction set: SSA vector values carry payloads, masks gate active lanes when present, and instruction-set-specific attributes select rounding, selection, distribution, or fused-mode behavior.

## Expected Outputs

This form is primarily defined by the side effect it has on control state, predicate state, or memory. It does not publish a new payload SSA result beyond any explicit state outputs shown in the syntax.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This operation inherits the legality and operand-shape rules of its instruction set overview. Any target-specific narrowing of element types, distributions, pipe/event spaces, or configuration tuples must be stated by the selected target profile.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vcvt`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vcvt`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```mlir
pto.vcvt
```

## Detailed Notes

The instruction set overview carries the remaining shared rules for this operation.

## Related Ops / Instruction Set Links

- Instruction set overview: [Conversion Ops](../../conversion-ops.md)
- Previous op in instruction set: [pto.vci](./vci.md)
- Next op in instruction set: [pto.vtrc](./vtrc.md)
