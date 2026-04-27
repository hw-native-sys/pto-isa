# Scalar And Control Instruction Set: Control And Configuration

The control-shell overview for the `pto.*` instruction set explains how PTO programs establish ordering, configure DMA, and manipulate predicate-visible state around tile and vector payload work.

## Summary

Scalar and control operations do not carry tile payload semantics themselves. They set up the execution environment in which `pto.t*` and `pto.v*` work becomes legal and well ordered.

## Main Subfamilies

This overview groups all scalar/control operations by their architectural role:

- [Pipeline sync](./pipeline-sync.md): explicit producer-consumer edges, buffer-token protocols, and memory barriers.
- [DMA copy](./dma-copy.md): loop-size and stride configuration plus GM↔vector-tile-buffer and vector-tile-buffer↔vector-tile-buffer copy operations.
- [Predicate load store](./predicate-load-store.md): moving `!pto.mask` state through UB and handling unaligned predicate-store streams.
- [Predicate generation and algebra](./predicate-generation-and-algebra.md): mask creation, tail masks, boolean combination, and predicate rearrangement.
- [Micro-instruction reference](./ops/micro-instruction/README.md): scalar/vector boundary and runtime query operations.

## Architectural Role

The `pto.*` instruction set is where PTO exposes stateful setup and synchronization explicitly. These forms are still part of the virtual ISA contract, but their visible outputs are control, mask, or configuration state rather than tile or vector payload results.

The Tile ISA instruction set ([Sync and Config group](../tile/sync-and-config.md)) handles tile-mode configuration such as `pto.settf32mode`, `pto.setfmatrix`, `pto.set_img2col_rpt`, and `pto.set_img2col_padding`. These are Tile ISA instructions because they program tile-mode state; they are **not** in this scalar/control section.

This Control and Configuration section has no standalone operation list. Scalar/control operations are covered by the Pipeline Sync, DMA Copy, Predicate Load Store, Predicate Generation, and Micro-Instruction subfamilies.

## Related Material

- [Scalar and control instruction set](../instruction-families/scalar-and-control-families.md)
- [Scalar and control instruction set overview](../instruction-families/scalar-and-control-families.md)
- [Vector ISA reference](../vector/README.md)
- [Tile ISA reference: Sync and Config](../tile/sync-and-config.md)
