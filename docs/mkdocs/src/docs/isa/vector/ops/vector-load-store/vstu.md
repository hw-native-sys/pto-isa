<!-- Generated from `docs/isa/vector/ops/vector-load-store/vstu.md` -->

# pto.vstu

Standalone reference page for `pto.vstu`. This page belongs to the [Vector Load Store](../../vector-load-store.md) family in the PTO ISA manual.

## Summary

Unaligned store with align + offset state update.

## Mechanism

`pto.vstu` is part of the PTO vector memory/data-movement surface. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

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
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index
```

## Detailed Notes

**Mode tokens:** `POST_UPDATE`, `NO_POST_UPDATE`

## Related Ops / Family Links

- Family overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in family: [pto.vstar](./vstar.md)
- Next op in family: [pto.vstus](./vstus.md)
