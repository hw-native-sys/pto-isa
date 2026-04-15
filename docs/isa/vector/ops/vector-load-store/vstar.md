# pto.vstar

`pto.vstar` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Flush remaining alignment state.

## Mechanism

`pto.vstar` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>
```

## Inputs

`%value` is the pending alignment/buffer state that still needs to be emitted,
  and `%dest` is the UB destination base pointer.

## Expected Outputs

No SSA result. The effect is a memory-side flush that writes the remaining
  buffered bytes to memory.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

This op terminates an unaligned-store sequence. It MUST be paired with a
  compatible prior state-producing store sequence so that the pending tail state
  is well-defined.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vstar`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vstar`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```mlir
pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>
```

## Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vstas](./vstas.md)
- Next op in instruction set: [pto.vstu](./vstu.md)
