<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA Manual And Reference

This directory is the canonical PTO ISA tree. It combines the architecture manual, the instruction set guides, the instruction set contracts, and the exact instruction-reference groupings in one place.

## Textual Assembly Inside PTO ISA

This tree is the canonical PTO ISA manual. Textual assembly spelling belongs to the PTO ISA syntax instruction set, not to a second parallel architecture manual.

- PTO ISA defines architecture-visible semantics, legality, state, ordering, target-profile boundaries, and the visible behavior of `pto.t*`, `pto.v*`, `pto.*`, and other operations.
- PTO-AS is the assembler-facing spelling used to write those operations and operands. It is part of how PTO ISA is expressed, not a separate ISA with different semantics.

If the question is "what does this legal PTO program mean across CPU, A2/A3, and A5?", stay in this tree. If the question is "what is the operand shape or textual spelling of this operation?", use the syntax-and-operands pages in this same tree.

## Start Here

- [PTO ISA landing page](../PTO-Virtual-ISA-Manual.md)
- [Introduction](introduction/what-is-pto-visa.md)
- [Document structure](introduction/document-structure.md) (chapter map and reading order)
- [Goals Of PTO](introduction/goals-of-pto.md)
- [PTO ISA Version 1.0](introduction/pto-isa-version-1-0.md)
- [Scope And Boundaries](introduction/design-goals-and-boundaries.md)
- [Tiles And Valid Regions](programming-model/tiles-and-valid-regions.md)
- [Execution Agents And Target Profiles](machine-model/execution-agents.md)
- [Assembly spelling and operands](syntax-and-operands/assembly-model.md)
- [Operands and attributes](syntax-and-operands/operands-and-attributes.md)
- [Common conventions](conventions.md)
- [Type system](state-and-types/type-system.md)
- [Location intent and legality](state-and-types/location-intent-and-legality.md)
- [Consistency baseline](memory-model/consistency-baseline.md)

## Model Layers

Reading order matches the manual chapter map: programming and machine models, then syntax and state, then memory, then opcode reference.

- [Programming model](programming-model/tiles-and-valid-regions.md)
- [Machine model](machine-model/execution-agents.md)
- [Syntax and operands](syntax-and-operands/assembly-model.md)
- [Type system](state-and-types/type-system.md)
- [Location intent and legality](state-and-types/location-intent-and-legality.md)
- [Memory model](memory-model/consistency-baseline.md)

## Instruction Structure

- [Instruction overview](instruction-surfaces/README.md)
- [Instruction set contracts](instruction-families/README.md)
- [Format of instruction descriptions](reference/format-of-instruction-descriptions.md)
- [Tile instruction reference](tile/README.md)
- [Vector instruction reference](vector/README.md)
- [Scalar and control reference](scalar/README.md)
- [Other and communication reference](other/README.md)
- [Common conventions](conventions.md)

## Supporting Reference

- [Reference notes](reference/README.md) (glossary, diagnostics, portability, source of truth)

## Compatibility Wrappers

The grouped instruction set trees under `tile/`, `vector/`, `scalar/`, and `other/` are the canonical PTO ISA paths.

Some older root-level tile pages such as `TADD.md`, `TLOAD.md`, and `TMATMUL.md` now remain only as compatibility wrappers so existing links do not break immediately. New PTO ISA documentation should link to the grouped instruction set paths, especially the standalone per-op pages under:

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
