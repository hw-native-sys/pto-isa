<!-- Generated from `docs/isa/vector/ops/vector-load-store/vsldb.md` -->

# pto.vsldb

Standalone reference page for `pto.vsldb`. This page belongs to the [Vector Load Store](../../vector-load-store.md) family in the PTO ISA manual.

## Summary

Block-strided load for 2D tile access.

## Mechanism

`pto.vsldb` is part of the PTO vector memory/data-movement surface. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%result = pto.vsldb %source, %offset, %mask : !pto.ptr<T, ub>, i32, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

`%source` is the UB base pointer, `%offset` is the packed stride/control word,
  and `%mask` controls which blocks participate.

## Expected Outputs

`%result` is the loaded vector.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

`%offset` is not a plain byte displacement; it encodes the block stride and
  repeat pattern. If a block is masked off, the corresponding destination block
  is zeroed and MUST NOT raise an address overflow exception for that block.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```mlir
%result = pto.vsldb %source, %offset, %mask : !pto.ptr<T, ub>, i32, !pto.mask -> !pto.vreg<NxT>
```

## Detailed Notes

## Gather (Indexed) Loads

## Related Ops / Family Links

- Family overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in family: [pto.vsld](./vsld.md)
- Next op in family: [pto.vgather2](./vgather2.md)
