<!-- Generated from `docs/isa/vector/ops/vector-load-store/vstur.md` -->

# pto.vstur

Standalone reference page for `pto.vstur`. This page belongs to the [Vector Load Store](../../vector-load-store.md) family in the PTO ISA manual.

## Summary

Unaligned store with residual flush and state update.

## Mechanism

`pto.vstur` is part of the PTO vector memory/data-movement surface. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align
```

## Inputs

`%align_in` is the incoming store-alignment state, `%value` is the vector to
  store, and `%base` is the UB base pointer.

## Expected Outputs

`%align_out` is the updated residual state after the current partial store.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

This form exposes only the evolving state; it does not by itself guarantee
  that all buffered bytes have reached memory. A compatible final flush is still
  required unless the surrounding sequence is known to be complete.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align
```

## Detailed Notes

The family overview carries the remaining shared rules for this operation.

## Related Ops / Family Links

- Family overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in family: [pto.vstus](./vstus.md)
