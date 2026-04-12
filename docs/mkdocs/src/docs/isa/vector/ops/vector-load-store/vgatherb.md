<!-- Generated from `docs/isa/vector/ops/vector-load-store/vgatherb.md` -->

# pto.vgatherb

Standalone reference page for `pto.vgatherb`. This page belongs to the [Vector Load Store](../../vector-load-store.md) family in the PTO ISA manual.

## Summary

Byte-granularity indexed gather from UB.

## Mechanism

`pto.vgatherb` is part of the PTO vector memory/data-movement surface. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%result = pto.vgatherb %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>
```

## Inputs

`%source` is the UB base pointer, `%offsets` contains per-block byte offsets,
  and `%active_lanes` bounds the number of active gathered blocks.

## Expected Outputs

`%result` is the gathered vector.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

This is a block gather, not a byte-per-lane gather. `%source` MUST be 32-byte
  aligned, each participating offset MUST describe a 32-byte-aligned block, and
  inactive blocks are zero-filled.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i]];  // byte-addressed
```

## Detailed Notes

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i]];  // byte-addressed
```

## Related Ops / Family Links

- Family overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in family: [pto.vgather2](./vgather2.md)
- Next op in family: [pto.vgather2_bc](./vgather2-bc.md)
