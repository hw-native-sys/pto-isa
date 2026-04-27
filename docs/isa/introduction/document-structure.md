# PTO: Document Structure

The manual is organized as a layered architecture reference: establish the programming and machine models first, then syntax and types, then memory rules, then the instruction set. The chapter roles stay fixed so model rules appear before opcode detail, while the content remains specific to PTO's tile-first Ascend model.

## Chapter Map

This manual is organized into 12 navigation chapters. The sidebar mirrors the PTO ISA reading path: model first, instruction surfaces second, supporting reference last.

| Chapter | Sections | What It Covers |
| --- | --- | --- |
| **1. Introduction** | What Is PTO VISA, Document Structure, Goals, current scope, Scope | What PTO is, why it exists, current scope, and specification boundaries |
| **2. Programming Model** | Tiles & Valid Regions, GlobalTensor & Data Movement, Auto vs Manual | The primary programming objects in PTO programs |
| **3. Machine Model** | Execution Agents & Target Profiles, Ordering & Synchronization | Execution hierarchy, pipelines, target profiles, and sync vocabulary |
| **4. Syntax and Operands** | Assembly Spelling, Operands & Attributes, Common Conventions | Textual spelling, operand shapes, attributes, and naming conventions |
| **5. State and Types** | Type System, Layout Reference, Data Format Reference, Location Intent | Types, layouts, data formats, tile roles, and legality rules |
| **6. Memory Model** | Consistency Baseline, Producer-Consumer Ordering | Visibility and ordering rules across pipelines and cores |
| **7. Instruction Set Overview** | Tile, Vector, Scalar & Control, Communication instruction-set contracts | High-level maps and shared contracts for all instruction surfaces |
| **8. Tile ISA Reference** | Sync & Config, Elementwise Tile-Tile, Tile-Scalar, Reduce & Expand, Memory & Data Movement, Matrix & Matrix-Vector, Layout & Rearrangement, Irregular & Complex | All `pto.t*` operations |
| **9. Vector ISA Reference** | Vector Load/Store, Predicate & Materialization, Unary, Binary, Vec-Scalar, Conversion, Reduction, Compare & Select, Data Rearrangement, SFU & DSA | All `pto.v*` operations |
| **10. Scalar and Control Reference** | Pipeline Sync, DMA Copy, Predicate Load/Store, Predicate Generation & Algebra, Shared Arithmetic, Shared SCF | All `pto.*` scalar/control operations |
| **11. Communication ISA Reference** | Collective Communication, Runtime Synchronization | Inter-NPU communication operations such as `pto.tbroadcast`, `pto.tget`, `pto.tput`, `pto.treduce`, and `pto.twait` |
| **12. System Scheduling ISA Reference** | Scheduling-visible Runtime Control | TPipe/TMPipe scheduling and lifetime operations such as `pto.tpush`, `pto.tpop`, and `pto.tfree` |
| **12. Reference Notes** | Format of Descriptions, Diagnostics, Glossary, Portability, Source of Truth | Supporting reference material |

Chapters 1–6 are the "read before opcode detail" layer. Chapter 7 gives the high-level instruction-set map. Chapters 8–11 are the per-op reference grouped by PTO instruction surface. Chapter 12 is supporting material.

PTO-AS is not a separate architecture chapter. Textual assembly spelling is covered by [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md) inside the Syntax and Operands chapter, because PTO-AS is the textual form of the PTO ISA contract rather than a second ISA.

## PTO-Specific Reading Notes

PTO is built around **tiles**, **valid regions**, and **explicit synchronization**. Read model chapters first, then syntax and state, then memory, then per-op pages. This keeps architecture guarantees separate from backend profile narrowing and avoids treating examples as standalone contracts.

## See Also

- [PTO ISA hub](../scalar/ops/micro-instruction/README.md)
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md)
