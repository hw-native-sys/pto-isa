<!-- Generated from `docs/isa/scalar/shared-scf.md` -->

# Scalar And Control Families: Shared Structured Control Flow

PTO source programs use shared MLIR `scf` operations to express loops, branches, and loop-carried state around PTO regions. These are part of the documented source surface, but they are not PTO mnemonic families.

## Summary

Shared structured control flow gives PTO a control shell that stays analyzable and explicit. It avoids inventing custom PTO branch syntax for logic that is already represented clearly by `scf.for`, `scf.if`, `scf.while`, `scf.condition`, and `scf.yield`.

## Mechanism

`scf` surrounds PTO regions rather than replacing them. It is used to:

- express counted loops around repeated tile or vector work
- carry scalar or tile state across iterations
- model structured conditional execution
- keep control flow visible to analyses and lowerings

This matters especially for the vector surface, where `__VEC_SCOPE__` is modeled using structured control rather than an opaque launch node.

## Inputs

Shared structured control flow consumes:

- scalar predicates
- loop bounds and step values
- region-carried SSA state
- yielded values from nested branches or loops

## Expected Outputs

It produces:

- well-structured control regions
- explicit loop-carried values
- branch-selected scalar or tile state

## Side Effects

`scf` itself does not create DMA, synchronization, or payload effects. Those effects come from the PTO instructions inside the structured regions.

## Constraints

- PTO control flow **SHOULD** stay in structured `scf` form unless a more specific architecture-visible mechanism is required.
- Region-carried values and branch results **MUST** be explicit through `scf.yield`.
- Predicate construction for `scf` control **SHOULD** come from the shared scalar surface, not from undocumented control side channels.

## Exceptions

The following are **ILLEGAL**:

- pretending `scf` is a PTO mnemonic family
- hiding loop-carried state that later affects PTO legality
- collapsing structured control into vague prose instead of documenting the carried values and branch conditions

## Target-Profile Restrictions

The `scf` surface is largely target-neutral. Restrictions appear when a region contains target-profile-specific PTO instructions or when a backend imposes extra structure on a vector-execution scope.

## Examples

### Counted Loop Around Vector Work

```mlir
scf.for %i = %c0 to %tile_count step %c1 {
  %offset = arith.muli %i, %tile_stride : index
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %v = pto.vlds %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
}
```

### Structured Conditional Around Tile Update

```mlir
%need_tail = arith.cmpi slt, %valid_cols, %tile_cols : index
scf.if %need_tail {
  pto.tsubs ins(%tile, %bias : !pto.tile_buf<...>, f32) outs(%tile : !pto.tile_buf<...>)
} else {
  pto.tadds ins(%tile, %bias : !pto.tile_buf<...>, f32) outs(%tile : !pto.tile_buf<...>)
}
```

## Related Ops And Family Links

- [Shared Scalar Arithmetic](./shared-arith.md)
- [Scalar And Control Families: Control And Configuration](./control-and-configuration.md)
- [Machine Model: Ordering And Synchronization](../machine-model/ordering-and-synchronization.md)
