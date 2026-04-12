<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO ISA Guide

This directory contains the canonical PTO ISA manual and the supporting instruction references. The preferred reading path is the merged `docs/isa/` tree, which treats PTO as one coherent multi-target virtual ISA rather than as a split between conceptual and reference folders.

## Start With The Manual

- [Manual landing page](PTO-Virtual-ISA-Manual.md)
- [Introduction](isa/introduction/what-is-pto-visa.md)
- [Document structure](isa/introduction/document-structure.md)
- [Goals Of PTO](isa/introduction/goals-of-pto.md)
- [PTO ISA Version 1.0](isa/introduction/pto-isa-version-1-0.md)
- [Scope And Boundaries](isa/introduction/design-goals-and-boundaries.md)
- [Programming model](isa/programming-model/tiles-and-valid-regions.md)
- [Machine model](isa/machine-model/execution-agents.md)
- [Syntax and operands](isa/syntax-and-operands/assembly-model.md)
- [Common conventions](isa/conventions.md)
- [Type system](isa/state-and-types/type-system.md)
- [Location intent and legality](isa/state-and-types/location-intent-and-legality.md)
- [Memory model](isa/memory-model/consistency-baseline.md)
- [Instruction overview](isa/instruction-surfaces/README.md)
- [Instruction set contracts](isa/instruction-families/README.md)
- [Format of instruction descriptions](isa/reference/format-of-instruction-descriptions.md)

## Use The Reference Trees For Exact Instruction Coverage

- [Tile ISA reference](isa/tile/README.md)
- [Vector ISA reference](isa/vector/README.md)
- [Scalar and control reference](isa/scalar/README.md)
- [Other and communication reference](isa/other/README.md)
- [Common conventions](isa/conventions.md)

For the full chapter map, see [Document structure](isa/introduction/document-structure.md).

## Documentation Layout

- `docs/isa/`: canonical PTO ISA manual and instruction overview tree
- `docs/isa/tile/`: tile-instruction reference and instruction set grouping
- `docs/isa/vector/`: vector-instruction reference derived from PTOAS VPTO structure
- `docs/isa/scalar/`: scalar/control/configuration reference
- `docs/isa/other/`: communication and residual supporting instruction sets
- `docs/reference/`: maintainer-facing reference and doc process material
