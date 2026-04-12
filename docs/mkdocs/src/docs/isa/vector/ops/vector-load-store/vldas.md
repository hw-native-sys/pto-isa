<!-- Generated from `docs/isa/vector/ops/vector-load-store/vldas.md` -->

# pto.vldas

Standalone reference page for `pto.vldas`. This page belongs to the [Vector Load Store](../../vector-load-store.md) family in the PTO ISA manual.

## Summary

Prime alignment buffer for subsequent unaligned load.

## Mechanism

`pto.vldas` is part of the PTO vector memory/data-movement surface. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## Inputs

`%source` is the UB address whose surrounding aligned block seeds the load
  alignment state.

## Expected Outputs

`%result` is the initialized load-alignment state.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

This op is the required leading operation for a `pto.vldus` stream using the
  same alignment state. The source address itself need not be 32-byte aligned;
  hardware truncates it to the aligned block boundary for the priming load.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## Detailed Notes

The family overview carries the remaining shared rules for this operation.

## Related Ops / Family Links

- Family overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in family: [pto.vlds](./vlds.md)
- Next op in family: [pto.vldus](./vldus.md)
