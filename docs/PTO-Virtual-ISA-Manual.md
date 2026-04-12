# PTO Virtual ISA Manual

This page is the stable landing page for the PTO ISA manual. The canonical manual now lives under `docs/isa/` as one merged tree. It presents PTO as a multi-target virtual ISA with a clear split between programming model, machine model, memory model, instruction sets, and instruction set contracts.

Use the top-right language icon to move between the English and Chinese tracks. Switching language lands on a real counterpart when one exists; otherwise it falls back to the matching language landing page.

## Start Here

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

## Manual Structure

- [Instruction overview](isa/instruction-surfaces/README.md)
- [Instruction set contracts](isa/instruction-families/README.md)
- [Format of instruction descriptions](isa/reference/format-of-instruction-descriptions.md)
- [Reference notes](isa/reference/README.md)

## Instruction Reference

- [Tile ISA reference](isa/tile/README.md)
- [Vector ISA reference](isa/vector/README.md)
- [Scalar and control reference](isa/scalar/README.md)
- [Other and communication reference](isa/other/README.md)
- [Syntax and operands](isa/syntax-and-operands/assembly-model.md)
- [Common conventions](isa/conventions.md)

## PTO ISA At A Glance

PTO is a virtual ISA that spans multiple targets, including CPU simulation, A2/A3-class targets, and A5-class targets. The visible ISA is not one flat pool of operations:

- `pto.t*` covers tile-oriented compute and data movement.
- `pto.v*` covers vector micro-instruction behavior and its buffer/register/predicate model.
- `pto.*` covers scalar, control, configuration, and shared supporting operations.
- communication and other supporting operations complete the instruction set where needed.

The manual explains what is guaranteed by PTO itself and what is only a target-profile restriction.

## Canonical Hub

The merged manual index is [PTO ISA manual and reference](isa/README.md).
