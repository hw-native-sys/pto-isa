# PTO Virtual ISA Manual

The stable PTO ISA manual entry points to the merged tree under `docs/isa/`. That tree presents PTO as a multi-target virtual ISA with separate programming-model, machine-model, memory-model, instruction-set, and instruction-contract layers.

The top-right language icon switches between English and Chinese. When a counterpart page exists, the switch lands there; otherwise it returns to the matching language landing page.

---

## Quick Navigation

### I am new to PTO — where do I start?

Follow this path in order:

1. **[What is PTO VISA](isa/introduction/what-is-pto-visa.md)** — Overview, two compilation flows, key terms, and machine model
2. **[Programming Model](isa/programming-model/tiles-and-valid-regions.md)** — Tiles, valid regions, layouts, auto vs. manual mode
3. **[Instruction Overview](isa/instruction-surfaces/README.md)** — Map of all four instruction sets
4. **[Tile Instruction Reference](isa/tile/README.md)** — Primary programming instruction set
5. **[Format of Instruction Descriptions](isa/reference/format-of-instruction-descriptions.md)** — How to read each instruction page

### I am a kernel writer

- **[Tile ISA Reference](isa/tile/README.md)** — `TADD`, `TMATMUL`, `TLOAD`, `TSTORE`, and all tile operations
- **[Vector ISA Reference](isa/vector/README.md)** — `vadd`, `vmul`, `vlds`, `vsts`, and all vector operations
- **[GlobalTensor and Data Movement](isa/programming-model/globaltensor-and-data-movement.md)** — GM↔local-tile-buffer data flow, including the vector tile buffer (hardware UB)
- **[Ordering and Synchronization](isa/machine-model/ordering-and-synchronization.md)** — `set_flag`/`wait_flag`, `get_buf`/`rls_buf`, double-buffering

### I am a compiler backend developer

- **[Execution Agents and Target Profiles](isa/machine-model/execution-agents.md)** — CPU, A2A3 (Ascend 910B / 910C), and A5 (Ascend 950 PR / DT) profile details
- **[Instruction Set Contracts](isa/instruction-families/README.md)** — Legal types, layouts, and shapes per instruction family
- **[Location Intent and Legality](isa/state-and-types/location-intent-and-legality.md)** — TileType constraints
- **[Portability and Target Profiles](isa/reference/portability-and-target-profiles.md)** — What's guaranteed across profiles

### I want to find a specific instruction

Use the instruction set index pages:

| Instruction Set | Overview Page | Count |
|---------------|-------------|-------|
| Tile compute & data movement | [Tile Instructions](isa/tile/README.md) | ~70 operations |
| Vector micro-instructions | [Vector Instructions](isa/vector/README.md) | ~60 operations |
| Scalar, control & DMA | [Scalar and Control](isa/scalar/README.md) | ~50 operations |
| Collective communication | [Other and Communication](isa/other/README.md) | ~20 operations |

Or jump directly to the alphabetical index: [Instruction Families](isa/instruction-families/README.md)

---

## PTO ISA At A Glance

PTO is a virtual ISA that spans multiple targets — CPU simulation, A2A3 (Ascend 910B / 910C), and A5 (Ascend 950 PR / DT). The ISA is organized into four instruction sets:

```
pto.t*   Tile instructions         Tile-oriented compute and GM↔tile data movement
pto.v*   Vector instructions       Low-level vector-pipe ops (predication, masks, etc.)
pto.*    Scalar & control          Scalar setup, DMA config, pipeline sync
         Communication            TBROADCAST, TGET, TPUT, TREDUCE, TWAIT, etc.
```

The manual explains what is guaranteed by PTO itself and what is only a target-profile restriction.

---

## Manual Structure

The manual is organized in layers — read the earlier chapters before the later ones:

```
Layer 1:  Introduction & Overview
          ├── What Is PTO VISA
          ├── Document Structure
          ├── Goals Of PTO
          ├── PTO ISA Version 1.0
          └── Scope And Boundaries

Layer 2:  Programming Model
          ├── Tiles and Valid Regions
          ├── GlobalTensor and Data Movement
          └── Auto vs Manual

Layer 3:  Machine Model
          ├── Execution Agents and Target Profiles
          └── Ordering and Synchronization

Layer 4:  Memory Model
          ├── Consistency Baseline
          └── Producer-Consumer Ordering

Layer 5:  State and Types
          ├── Type System
          ├── Layout Reference
          ├── Data Format Reference
          └── Location Intent and Legality

Layer 6:  Syntax and Operands
          ├── Assembly Spelling and Operands
          └── Operands and Attributes

Layer 7:  Instruction Reference
          ├── Tile ISA Reference
          ├── Vector ISA Reference
          ├── Scalar and Control Reference
          └── Other and Communication Reference
```

---

## Start Here

### Core Reading

- [Introduction](isa/introduction/what-is-pto-visa.md)
- [Document structure](isa/introduction/document-structure.md)
- [Goals Of PTO](isa/introduction/goals-of-pto.md)
- [PTO ISA Version 1.0](isa/introduction/pto-isa-version-1-0.md)
- [Scope And Boundaries](isa/introduction/design-goals-and-boundaries.md)

### Programming and Machine Models

- [Tiles and Valid Regions](isa/programming-model/tiles-and-valid-regions.md)
- [GlobalTensor and Data Movement](isa/programming-model/globaltensor-and-data-movement.md)
- [Auto vs Manual](isa/programming-model/auto-vs-manual.md)
- [Execution Agents and Target Profiles](isa/machine-model/execution-agents.md)
- [Ordering and Synchronization](isa/machine-model/ordering-and-synchronization.md)

### Types, Memory, and Syntax

- [Type system](isa/state-and-types/type-system.md)
- [Layout reference](isa/state-and-types/layout.md)
- [Data format reference](isa/state-and-types/data-format.md)
- [Location intent and legality](isa/state-and-types/location-intent-and-legality.md)
- [Memory model](isa/memory-model/consistency-baseline.md)
- [Assembly spelling and operands](isa/syntax-and-operands/assembly-model.md)
- [Operands and attributes](isa/syntax-and-operands/operands-and-attributes.md)
- [Common conventions](isa/conventions.md)

### Instruction Reference

- [Instruction overview](isa/instruction-surfaces/README.md)
- [Instruction set contracts](isa/instruction-families/README.md)
- [Format of instruction descriptions](isa/reference/format-of-instruction-descriptions.md)
- [Glossary](isa/reference/glossary.md)
- [Diagnostics and illegal cases](isa/reference/diagnostics-and-illegal-cases.md)
- [Portability and target profiles](isa/reference/portability-and-target-profiles.md)
- [Source of truth](isa/reference/source-of-truth.md)

### Instruction Set References

- [Tile ISA reference](isa/tile/README.md)
- [Vector ISA reference](isa/vector/README.md)
- [Scalar and control reference](isa/scalar/README.md)
- [Other and communication reference](isa/other/README.md)

---

## Canonical Hub

The full manual index with all sections and per-op pages is at [PTO ISA manual and reference](isa/README.md).
