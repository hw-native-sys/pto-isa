# pto.vstu

`pto.vstu` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Unaligned store with align + offset state update.

## Mechanism

`pto.vstu` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index
```

## Inputs

`%align_in` is the incoming store-alignment state, `%offset_in` is the current
  logical byte/element displacement, `%value` is the vector being stored, and
  `%base` is the UB base pointer.

## Expected Outputs

`%align_out` is the updated alignment/tail state and `%offset_out` is the
  next offset after applying the selected post-update rule.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

The alignment state MUST be threaded in program order. A terminating flush
  form such as `pto.vstar`/`pto.vstas` is still required to commit the buffered
  tail bytes.

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
For `pto.vstu`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vstu`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```mlir
%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index
```

## Detailed Notes

**Mode tokens:** `POST_UPDATE`, `NO_POST_UPDATE`

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vstar](./vstar.md)
- Next op in instruction set: [pto.vstus](./vstus.md)
