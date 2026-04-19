# docs/isa/

This directory is the top-level index for the PTO ISA manual. It provides a guided navigation through all instruction references and conceptual documentation.

## How This Manual Is Organized

The manual is organized into five logical layers:

| Layer | Contents | Audience |
|-------|----------|----------|
| **1. Foundations** | Introduction, programming model, machine model | Everyone — start here |
| **2. Syntax and Semantics** | Assembly model, operands, types, memory model | Compiler developers, kernel authors |
| **3. Instruction Surface** | Instruction-set overview and contracts | All users |
| **4. Reference Manual** | Tile, vector, scalar, and communication reference | Performance engineers, kernel authors |
| **5. Appendices** | Format guidelines, diagnostics, glossary, portability | Everyone |

## Key Entry Points

| Document | Content |
|----------|---------|
| [What is PTO ISA?](./introduction/what-is-pto-visa.md) | Core concepts and where PTO fits in the software stack |
| [Tiles and Valid Regions](./programming-model/tiles-and-valid-regions.md) | The tile abstraction — PTO's primary programming object |
| [Execution Agents and Profiles](./machine-model/execution-agents.md) | Execution hierarchy, pipelines, target profiles |
| [Instruction Surfaces Overview](./instruction-surfaces/README.md) | Map of all four instruction sets and when to use each |
| [Per-Instruction Reference](./tile/README.md) | Complete catalog organized by category |
| [Format of instruction descriptions](./reference/format-of-instruction-descriptions.md) | How to read each per-op page |

## Four Instruction Sets

| Instruction Set | Prefix | Operations | Reference |
|----------------|--------|------------|-----------|
| **Tile** | `pto.t*` | ~120 | [Tile reference](./tile/README.md) |
| **Vector** | `pto.v*` | ~99 | [Vector reference](./vector/README.md) |
| **Scalar & Control** | `pto.*` | ~60 | [Scalar reference](./scalar/README.md) |
| **Communication** | `pto.*` | ~24 | [Communication reference](./other/README.md) |

## By Task

| What you're doing | Start here |
|-------------------|------------|
| Writing a matrix multiplication kernel | [Tile → Matrix ops](./tile/matrix-and-matrix-vector.md) |
| Optimizing elementwise operations | [Tile → Elementwise ops](./tile/elementwise-tile-tile.md) |
| Setting up data movement (GM ↔ tile) | [Tile memory ops](./tile/memory-and-data-movement.md) |
| Hand-tuning vector kernels | [Vector instructions](./vector/README.md) |
| Using per-lane masking and predicates | [Vector → Predicate ops](./vector/predicate-and-materialization.md) |
| Implementing collective communication | [Communication instructions](./other/README.md) |
| Understanding Auto vs Manual mode | [Auto vs Manual](./programming-model/auto-vs-manual.md) |
| Checking target profile support | [Execution agents](./machine-model/execution-agents.md) |

## See Also

- [docs/README.md](../README.md) — Documentation hub
- [docs/coding/README.md](../coding/README.md) — Programming model docs
- [PTO-AS specification](../assembly/PTO-AS.md) — Assembly syntax and grammar
