<!-- Generated from `docs/isa/vector/ops/vector-load-store/vstar.md` -->

# pto.vstar

Standalone reference page for `pto.vstar`. This page belongs to the [Vector Load Store](../../vector-load-store.md) family in the PTO ISA manual.

## Summary

Flush remaining alignment state.

## Mechanism

`pto.vstar` is part of the PTO vector memory/data-movement surface. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

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
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>
```

## Detailed Notes

## Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

## Related Ops / Family Links

- Family overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in family: [pto.vstas](./vstas.md)
- Next op in family: [pto.vstu](./vstu.md)
