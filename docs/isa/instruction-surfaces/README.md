# Instruction Overview

PTO ISA is organized into four instruction sets, each representing a distinct mechanism, programming model, and operand domain. Understanding the instruction-set split is essential before reading the per-op reference pages.

## Overview

| Instruction Set | Prefix | Pipeline | Primary Role | Typical Operands |
|----------------|--------|----------|-------------|-----------------|
| [Tile Instructions](./tile-instructions.md) | `pto.t*` | All (via tile buffers) | Tile-oriented compute, data movement, layout transforms, synchronization | `!pto.tile<...>`, `!pto.tile_buf<...>`, `!pto.partition_tensor_view<...>` |
| [Vector Instructions](./vector-instructions.md) | `pto.v*` | Vector Pipe (V) | Vector micro-instructions: lane-level compute, masking, alignment state | `!pto.vreg<NxT>`, `!pto.mask`, `!pto.ptr<T, ub>` |
| [Scalar and Control](./scalar-and-control-instructions.md) | `pto.*` | Scalar Unit, DMA | Configuration, control flow, DMA setup, synchronization, predicates | Scalar regs, pipe ids, event ids, buffer ids |
| [Other Instructions](./other-instructions.md) | `pto.*` | Inter-NPU | Collective communication, runtime support, tile sequence operations | `!pto.group<N>`, tile sequences, allocation handles |

## Why Four Instruction Sets

PTO is not a flat list of opcodes — it layers by architecturally visible state. The reason is direct: tile, vector, scalar/control, and communication each expose different kinds of state. Mixing them into one flat layer would blur the ISA contract.

| Instruction Set | Core Abstraction | Primary Responsibilities |
|----------------|-----------------|-------------------------|
| Tile (`pto.t*`) | tile: architecturally visible objects with shape, layout, role, valid region | GM↔tile movement, elementwise/reduce/layout/matmul ops, sync edges |
| Vector (`pto.v*`) | vreg, predicates, vector-visible UB | Vector register ops, lane-level masking, UB↔vreg movement |
| Scalar/Control (`pto.*`) | Scalar regs, pipe/event ids, buffer ids | Sync edges, DMA config, predicate construction, control flow |
| Other (`pto.*`) | Collective groups, tile sequences, allocation handles | Collective comm, tile sequence ops, memory management |

## Instruction Data Flow

The four instruction sets form a layered execution model:

```
GM (off-chip device memory)
        │
        ├── Tile instructions: TLOAD / TSTORE
        └── Vector path: copy_gm_to_ubuf / copy_ubuf_to_gm
        ▼
Vector tile buffer (hardware implementation is UB)
        │
        ├── Tile instructions: direct read/write to tile buffer
        └── Vector instructions: vlds / vsts
        ▼
┌─────────────────┐              ┌─────────────────────────────┐
│  Tile Buffers   │              │  Vector Registers           │
│  (Vec/Mat/Acc/  │              │  !pto.vreg<NxT>            │
│   Left/Right)    │              │                             │
└────────┬─────────┘              └──────────────┬────────────┘
         │                                       │
         │  Tile instructions: pto.t*         │  Vector instructions: pto.v*
         │  (TMATMUL via Mat/Left/Right/Acc)  │  (vadd, vmul, vcmp, ...)
         │                                       │
         │  ◄── Matrix Multiply Unit            │  ◄── Vector Pipeline
         └─────────────────────────────────────┘
                       │
                       ▼
              [tile buffer → GM]
```

## Instruction Count Summary

| Instruction Set | Families | Operations | Notes |
|----------------|----------|------------|-------|
| Tile | 8 | ~120 | Full matmul, elementwise, reduce, layout, data movement |
| Vector | 9 | ~99 | Full vector compute, load/store, SFU |
| Scalar/Control | 6 | ~60 | Sync, DMA, predicates, control |
| Other/Communication | 2 | ~24 | Collective ops, supporting ops |

## Normative Language

Instruction set pages describe shared contracts for groups of operations — they do not repeat per-op details. Use **MUST / SHOULD / MAY** only for rules that a verifier, test, or review can check. Prefer plain language for explanation.

## See Also

- [Instruction set contracts](../instruction-families/README.md) — Group-level contracts for all four sets
- [Tile reference](../tile/README.md) — Tile instruction per-op reference
- [Vector reference](../vector/README.md) — Vector instruction per-op reference
- [Scalar reference](../scalar/README.md) — Scalar and control per-op reference
- [Other reference](../other/README.md) — Communication and supporting ops
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page format standard
