# pto.vsort32

`pto.vsort32` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Sort 32 elements in UB.

## Mechanism

`pto.vsort32` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the instruction set can be reasoned about at the ISA level.

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

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vsort32`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vsort32`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```mlir
pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64
```

## Detailed Notes

The instruction set overview carries the remaining shared rules for this operation.

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vtranspose](./vtranspose.md)
- Next op in instruction set: [pto.vmrgsort](./vmrgsort.md)
