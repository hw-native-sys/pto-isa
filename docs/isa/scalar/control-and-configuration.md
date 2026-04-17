# Scalar And Control Instruction Set: Control And Configuration

The control-shell overview for the `pto.*` instruction set explains how PTO programs establish ordering, configure DMA, and manipulate predicate-visible state around tile and vector payload work.

## Summary

Scalar and control operations do not carry tile payload semantics themselves. They set up the execution environment in which `pto.t*` and `pto.v*` work becomes legal and well ordered.

## Main Subfamilies

- Legacy tile-prefixed control/configuration ops: [`pto.tsethf32mode`](./ops/control-and-configuration/tsethf32mode.md) and [`pto.tsetfmatrix`](./ops/control-and-configuration/tsetfmatrix.md). These keep historic `t`-prefixed API names but are documented here because they configure scalar-visible mode state rather than tile payload state.
- [Pipeline sync](./pipeline-sync.md): explicit producer-consumer edges, buffer-token protocols, and vector-scope memory barriers.
- [DMA copy](./dma-copy.md): loop-size and stride configuration plus GM↔vector-tile-buffer and vector-tile-buffer↔vector-tile-buffer copy operations.
- [Predicate load store](./predicate-load-store.md): moving `!pto.mask` state through UB and handling unaligned predicate-store streams.
- [Predicate generation and algebra](./predicate-generation-and-algebra.md): mask creation, tail masks, boolean combination, and predicate rearrangement.

## Architectural Role

The `pto.*` instruction set is where PTO exposes stateful setup and synchronization explicitly. These forms are still part of the virtual ISA contract, but their visible outputs are control, mask, or configuration state rather than tile or vector payload results.

The manual also places a small number of historic `pto.t*` configuration ops here when their architectural job is scalar/control setup rather than tile-buffer mutation. `tsethf32mode` and `tsetfmatrix` fall into that category.

## Related Material

- [Scalar and control instruction set](../instruction-surfaces/scalar-and-control-instructions.md)
- [Scalar and control instruction set overview](../instruction-families/scalar-and-control-families.md)
- [Vector ISA reference](../vector/README.md)
