# pto.vstus

`pto.vstus` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Unaligned store with scalar offset and state update.

## Mechanism

`pto.vstus` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

## Inputs

`%align_in` is the incoming store-alignment state, `%offset` is the scalar
  displacement, `%value` is the vector being stored, and `%base` is the UB base
  pointer.

## Expected Outputs

`%align_out` is the updated buffered-tail state and `%base_out` is the next
  base pointer when the lowering chooses a post-update form.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

This is the scalar-offset stateful form of the unaligned store instruction set. The
  scalar offset width and update mode MUST match the selected form, and a later
  flush op is still required.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing sources for `pto.vstus` are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
Those sources define the buffered unaligned-store mechanism in detail, but they do **not** publish a numeric latency or steady-state throughput for `pto.vstus`.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | `visa.txt` §8.9 `VSTUS`, `PTOAS/docs/vpto-spec.md` |
| Steady-state throughput | Not publicly published | `visa.txt` §8.9 `VSTUS`, `PTOAS/docs/vpto-spec.md` |

Because `pto.vstus` participates in a stateful buffered-store stream, exact timing is backend-specific unless and until the public ISA source publishes a numeric contract.

## Examples

```mlir
%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

## Detailed Notes

The instruction set overview carries the remaining shared rules for this operation.

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vstu](./vstu.md)
- Next op in instruction set: [pto.vstur](./vstur.md)
