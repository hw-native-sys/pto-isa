<!-- Generated from `docs/isa/scalar/control-and-configuration.md` -->

# Scalar And Control Families: Control And Configuration

This page is the control-shell overview for the `pto.*` surface. It explains how PTO programs establish ordering, configure DMA, and manipulate predicate-visible state around tile and vector payload work.

## Summary

Scalar and control operations do not carry tile payload semantics themselves. They set up the execution environment in which `pto.t*` and `pto.v*` work becomes legal and well ordered.

## Main Subfamilies

- [Pipeline sync](./pipeline-sync.md): explicit producer-consumer edges, buffer-token protocols, and vector-scope memory barriers.
- [DMA copy](./dma-copy.md): loop-size and stride configuration plus GM↔UB and UB↔UB copy operations.
- [Predicate load store](./predicate-load-store.md): moving `!pto.mask` state through UB and handling unaligned predicate-store streams.
- [Predicate generation and algebra](./predicate-generation-and-algebra.md): mask creation, tail masks, boolean combination, and predicate rearrangement.

## Architectural Role

The `pto.*` surface is where PTO exposes stateful setup and synchronization explicitly. These forms are still part of the virtual ISA contract, but their visible outputs are control, mask, or configuration state rather than tile or vector payload results.

## Related Material

- [Scalar and control instruction surface](../instruction-surfaces/scalar-and-control-instructions.md)
- [Scalar and control family overview](../instruction-families/scalar-and-control-families.md)
- [Vector ISA reference](../vector/README.md)
