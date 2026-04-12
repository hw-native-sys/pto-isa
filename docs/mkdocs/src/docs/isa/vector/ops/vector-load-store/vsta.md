<!-- Generated from `docs/isa/vector/ops/vector-load-store/vsta.md` -->

# pto.vsta

Standalone reference page for `pto.vsta`. This page belongs to the [Vector Load Store](../../vector-load-store.md) family in the PTO ISA manual.

## Summary

Flush alignment state to memory.

## Mechanism

`pto.vsta` is part of the PTO vector memory/data-movement surface. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index
```

## Inputs

`%value` is the pending store-alignment state, `%dest` is the UB base pointer,
  and `%offset` is the flush displacement.

## Expected Outputs

This op writes buffered tail bytes to UB and returns no SSA value.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

The flush address MUST match the post-updated address expected by the
  preceding unaligned-store stream. After the flush, the corresponding store
  alignment state is consumed.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index
```

## Detailed Notes

The family overview carries the remaining shared rules for this operation.

## Related Ops / Family Links

- Family overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in family: [pto.vscatter](./vscatter.md)
- Next op in family: [pto.vstas](./vstas.md)
