<!-- Generated from `docs/isa/instruction-surfaces/README.md` -->

# Instruction Surfaces

PTO ISA is organized into four instruction surfaces, each representing a distinct mechanism, programming model, and operand domain. Understanding the surface distinction is essential before reading the per-op reference pages.

## Overview

| Surface | Prefix | Pipeline | Primary Role | Operands |
|---------|--------|----------|-------------|----------|
| [Tile Instructions](./tile-instructions.md) | `pto.t*` | All (via tile buffers) | Tile-oriented compute, data movement, layout transforms, synchronization | `!pto.tile<...>`, `!pto.tile_buf<...>`, `!pto.partition_tensor_view<...>` |
| [Vector Instructions](./vector-instructions.md) | `pto.v*` | Vector Pipe (V) | Vector micro-instructions: lane-level compute, masking, alignment state | `!pto.vreg<NxT>`, `!pto.mask`, `!pto.ptr<T, ub>` |
| [Scalar And Control](./scalar-and-control-instructions.md) | `pto.*` | Scalar Unit, DMA | Configuration, control flow, DMA setup, synchronization, predicates | Scalar regs, pipe ids, event ids, buffer ids |
| [Other Instructions](./other-instructions.md) | `pto.*` | Inter-NPU | Collective communication, runtime support, tile sequence operations | `!pto.group<N>`, tile sequences, allocation handles |

## Why Surfaces Exist

PTO has four surfaces because different parts of the architecture expose different kinds of state. Mixing tile-level and vector-level state in one opcode space would blur the ISA contract.

### Tile Surface (`pto.t*`)

Tile instructions reason about tiles: bounded multi-dimensional arrays with architecturally visible shape, layout, role, and valid-region metadata. The primary operands are tile registers (`!pto.tile<T, R, C>` or `!pto.tile_buf<...>`). Tile instructions produce destination tiles, change valid-region interpretations, or establish synchronization edges.

```
Input:   Tile operands, scalar modifiers, GlobalTensor views
Output:  Tile payload, synchronization edges
Domain:  Valid regions, tile layouts, tile shapes, location intents
```

### Vector Surface (`pto.v*`)

Vector instructions expose the vector pipeline directly. Operands are vector registers (`!pto.vreg<NxT>`), scalar values, and predicate masks. Vector instructions are the fine-grained compute layer beneath tile instructions. The full register width is always meaningful — there is no valid-region abstraction at the vector level.

```
Input:   Vector registers, scalar registers, predicates, memory addresses
Output:  Vector registers, scalar registers, memory writes
Domain:  Vector length N, lane masks, alignment state, distribution modes
```

### Scalar And Control Surface (`pto.*`)

Scalar/control instructions handle configuration, control flow, synchronization, DMA setup, and predicate state. They set up the execution shell around tile and vector payload regions. Most do not produce tile or vector payloads; they produce control effects, event tokens, or predicate masks.

```
Input:   Scalar registers, pipe ids, event ids, buffer ids, DMA loop parameters
Output:  Control state, event tokens, predicate masks, configured DMA state
Domain:  Configuration tuples, pipe/event spaces, DMA loop sizes and strides
```

### Other Surface (`pto.*`)

Communication and supporting operations carry their own side effects and ordering rules that do not fit into the tile/vector/scalar model. Examples include collective broadcasts across NPUs and alias/concatenation operations on tile sequences.

```
Input:   Collective groups, tile sequences, allocation handles
Output:  Collective results, modified tile sequences, allocation state
Domain:  Parallel groups, tile sequences, memory allocation
```

## Surface Data Flow

The four surfaces form a layered execution model:

```
┌─────────────────────────────────────────────────────────────┐
│  GM (off-chip device memory)                                │
└──────────┬──────────────────────────────────────┬───────────┘
           │                                      │
           │  Tile Surface: TLOAD/TSTORE          │
           │  Vector Surface: copy_gm_to_ubuf / copy_ubuf_to_gm
           ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Unified Buffer (UB, 256 KB on-chip)                      │
│  !pto.ptr<T, ub> — shared staging area                    │
└──────┬──────────────────────────────────────────┬──────────┘
       │                                      │
       │  Tile Surface: implicit tile↔UB       │
       │  Vector Surface: vlds / vsts          │
       ▼                                      ▼
┌─────────────────┐              ┌─────────────────────────────┐
│  Tile Buffers   │              │  Vector Registers          │
│  !pto.tile_buf  │              │  !pto.vreg<NxT>           │
│  (Vec/Mat/Acc/  │              │  (N lanes)                │
│   Left/Right)   │              │                           │
└────────┬─────────┘              └──────────────┬────────────┘
         │                                     │
         │  Tile Surface: pto.t* ops           │  Vector Surface: pto.v* ops
         │  (TMATMUL via Mat/Acc slots)       │  (vadd, vmul, etc.)
         │                                     │
         │  ◄── Matrix Multiply Unit (M)       │  ◄── Vector Pipeline (V)
         └─────────────────────────────────────┘
                       │
                       │  Tile Surface: TSTORE
                       │  Vector Surface: vsts → copy_ubuf_to_gm
                       ▼
                  [UB → GM]
```

## Instruction Count Summary

| Surface | Families | Operations | Notes |
|---------|----------|------------|-------|
| Tile | 8 | ~120 | Full matmul, elementwise, reduce, layout |
| Vector | 9 | ~99 | Full vector compute, load/store, SFU |
| Scalar/Control | 6 | ~60 | Sync, DMA, predicates |
| Other/Communication | 2 | ~24 | Collective ops, supporting ops |

## Normative Language

Instruction text always means what happens in the declared valid region unless the page explicitly defines behavior outside it. PTO is **tile-first** and **valid-region-first**.

Use **MUST**, **SHOULD**, and **MAY** only for rules that a test, verifier, or review can check. Prefer plain language for explanation.

## See Also

- [Instruction families](./instruction-families/README.md) — Group-level contracts
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page format standard
- [Tile ISA reference](./tile/README.md) — Tile surface per-op reference
- [Vector ISA reference](./vector/README.md) — Vector surface per-op reference
- [Scalar ISA reference](./scalar/README.md) — Scalar surface per-op reference
- [Other ISA reference](./other/README.md) — Communication and supporting ops
