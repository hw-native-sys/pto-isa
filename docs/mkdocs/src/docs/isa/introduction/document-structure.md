<!-- Generated from `docs/isa/introduction/document-structure.md` -->

# Document Structure

This manual is organized as a layered architecture reference: establish the programming and machine models first, then syntax and types, then memory rules, then the instruction set. The chapter roles stay fixed so readers can locate model rules before opcode detail; the content remains specific to PTO's tile-first Ascend model.

## Chapter Map

| Chapter | PTO manual section | What it covers |
| --- | --- | --- |
| 1. Introduction | [Introduction](./what-is-pto-visa.md), [Goals Of PTO](./goals-of-pto.md), [PTO ISA Version 1.0](./pto-isa-version-1-0.md), [Scope And Boundaries](./design-goals-and-boundaries.md) | What PTO is, why it exists, version baseline, and specification boundaries versus PTO-AS and PTOBC. |
| 2. Programming model | [Tiles and valid regions](../programming-model/tiles-and-valid-regions.md), [GlobalTensor and data movement](../programming-model/globaltensor-and-data-movement.md), [Auto vs Manual](../programming-model/auto-vs-manual.md) | Tiles, valid regions, global tensor views, Auto versus Manual—the objects authors reason about. |
| 3. Machine model | [Machine model](../machine-model/execution-agents.md) | Execution agents, pipelines, target profiles, ordering vocabulary. |
| 4. Syntax | [Assembly model](../syntax-and-operands/assembly-model.md), [operands and attributes](../syntax-and-operands/operands-and-attributes.md), [common conventions](../conventions.md) | Textual spelling, operand shapes, attributes, and shared naming conventions. |
| 5. State, types, and location | [Type system](../state-and-types/type-system.md), [location intent](../state-and-types/location-intent-and-legality.md) | Types, tile roles, location intent, legality. |
| 6. Memory consistency | [Memory model](../memory-model/consistency-baseline.md) | Visibility and ordering rules for producers and consumers. |
| 7. Instruction set | [Instruction surfaces](../instruction-surfaces/README.md), [instruction families](../instruction-families/README.md), per-op reference under `tile/`, `vector/`, `scalar/`, `other/` | Surfaces and families first, then opcode-level pages. |
| 8. Supporting reference | [Reference notes](../reference/README.md) | Glossary, diagnostics, portability, source-of-truth order, [format of instruction descriptions](../reference/format-of-instruction-descriptions.md). |

Chapters 4–6 are the usual “read before you deep-dive opcodes” path. Chapter 7 is the bulk of the opcode reference.

## PTO-Specific Reading Notes

PTO is built around **tiles**, **valid regions**, and **explicit synchronization**. Read model chapters first, then syntax and state, then memory, then per-op pages. This keeps architecture guarantees separate from backend profile narrowing and avoids treating examples as standalone contracts.

## See Also

- [PTO ISA hub](../README.md)
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md)
