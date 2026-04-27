# Instruction Set Contracts

Instruction set pages describe shared contracts that apply across related PTO operations. They sit between the model chapters and the per-op reference pages. For how individual opcode pages are structured, see [format of instruction descriptions](../reference/format-of-instruction-descriptions.md).

## Overview

PTO ISA is organized into five instruction sets, each representing a distinct hardware domain, operand type, and programming model. Understanding the instruction-set split is essential before reading the per-op reference pages.

| | Instruction Set | Prefix | Domain | Primary Role | Operand Types |
|-|-----------------|--------|--------|-------------|--------------|
| | [Tile Instruction Set](./tile-families.md) | `pto.t*` | Tile Buffers (Vec/Mat/Acc/Left/Right) | Tile-oriented compute, data movement, layout transforms, synchronization | `!pto.tile<...>`, `!pto.tile_buf<...>`, `!pto.partition_tensor_view<...>` |
| | [Vector Instruction Set](../instruction-families/vector-families.md) | `pto.v*` | Vector Pipeline (V) | Lane-level compute, masking, vector load/store, SFU operations | `!pto.vreg<NxT>`, `!pto.mask`, `!pto.ptr<T, ub>` |
| | [Scalar/Control Instruction Set](./scalar-and-control-families.md) | `pto.*` | Scalar Unit, DMA Controller | Configuration, synchronization, DMA setup, predicate generation and load/store | Scalar registers, pipe IDs, event IDs, buffer IDs, predicate masks |
| | [Communication Instruction Set](./communication-families.md) | `pto.tbroadcast`, `pto.tget`, etc. | Inter-NPU Interconnect | Collective communication, point-to-point exchange, runtime synchronization | `!pto.group<N>`, tile operands, allocation handles |
| | [System Scheduling Instruction Set](../system/README.md) | `pto.tpush`, `pto.tpop`, `pto.tfree` | Runtime-visible scheduling state | TPipe/TMPipe producer-consumer flow and resource lifetime | Tile handles, stream state, resource handles |

## Why These Instruction Sets Exist

PTO has five instruction sets because different parts of the architecture expose fundamentally different kinds of state. Tile state carries shape, layout, and valid-region metadata. Vector state exposes a flat register file. Scalar state controls pipeline ordering and DMA. Communication state spans multiple NPUs. System scheduling state covers ISA-visible runtime protocols that are not payload computation, vector-lane execution, scalar setup, or inter-NPU transfer.

### Tile ISA (`pto.t*`)

Tile instructions reason about tiles: bounded multi-dimensional arrays with architecturally visible shape, layout, role, and valid-region metadata. The primary operands are tile registers. Tile instructions produce destination tiles, change valid-region interpretations, or establish synchronization edges.

```
Input:   Tile operands, scalar modifiers, GlobalTensor views
Output:  Tile payload, synchronization edges
Domain:  Valid regions, tile layouts, tile shapes, location intents
```

### Vector ISA (`pto.v*`)

Vector instructions expose the vector pipeline directly. Operands are vector registers, scalar values, and predicate masks. Vector instructions are the fine-grained compute layer beneath tile instructions. The full register width is always meaningful — there is no valid-region abstraction at the vector level.

```
Input:   Vector registers, scalar registers, predicates, memory addresses
Output:  Vector registers, scalar registers, memory writes
Domain:  Vector length N, lane masks, alignment state, distribution modes
```

### Scalar/Control ISA (`pto.*`)

Scalar and control instructions handle configuration, synchronization, DMA setup, and predicate state. They set up the execution shell around tile and vector payload regions. Most do not produce tile or vector payloads; they produce control effects, event tokens, or predicate masks.

```
Input:   Scalar registers, pipe IDs, event IDs, buffer IDs, predicate masks
Output:  Control state, event tokens, predicate masks, configured DMA state
Domain:  Pipe/event spaces, DMA loop parameters, predicate register state
```

### Communication ISA

Communication instructions span multiple NPUs in a parallel group. They require a `ParallelGroup` handle and involve interconnect traffic. These operations express collective broadcasts, point-to-point exchanges, and runtime notification primitives.

```
Input:   Collective groups, tile operands, scalar parameters
Output:  Collective results, modified tiles, runtime events
Domain:  Parallel groups, inter-NPU network, rank ordering
```

### System Scheduling ISA

System scheduling instructions expose runtime-visible TPipe/TMPipe producer-consumer flow and resource lifetime. These operations are PTO ISA instructions because programs can observe their effects through stream ordering and resource state.

```
Input:   Tile handles, stream handles, resource handles
Output:  Released resources, push/pop ordering effects
Domain:  Resource lifetime, producer-consumer stream state
```

## What An Instruction Set Contract Must State

Each instruction set page provides the following:

1. **Mechanism** — What the instruction set is for, explained in one short section.
2. **Shared operand model** — Common input/output roles and how they interact.
3. **Common side effects** — Synchronization, ordering, or configuration effects shared by all instructions in the set.
4. **Shared constraints** — Legality rules that apply across the set.
5. **Cases that are not allowed** — Conditions that are illegal for all instructions in the set.
6. **Target-profile narrowing** — Where A2/A3 and A5 differ in what the set accepts.
7. **Operation list** — Pointers to each per-op page under `ops/`.

Instruction set pages do not repeat per-op details; they set the contract for the group.

## Navigation Map

```
Instruction Sets
├── Tile Instruction Set
│   ├── Sync and Config            → pto.tsync, pto.tassign, pto.talias, pto.settf32mode,
│   │                                pto.setfmatrix, pto.set_img2col_rpt,
│   │                                pto.set_img2col_padding, pto.subview,
│   │                                pto.get_scale_addr
│   ├── Elementwise Tile-Tile      → pto.tadd, pto.tsub, pto.tneg, pto.tmul, pto.tdiv,
│   │                                  pto.tmax, pto.tmin, pto.tcmp, pto.tcvt, pto.tsel,
│   │                                  pto.tabs, pto.trelu, pto.tlog, pto.texp, pto.tsqrt, etc.
│   ├── Tile-Scalar and Immediate  → pto.tadds, pto.taxpy, pto.tsubs, pto.tmuls, pto.tdivs, pto.tmins,
│   │                                  pto.tmaxs, pto.tpows, pto.tcmps, pto.texpands, etc.
│   ├── Reduce and Expand          → pto.trowsum, pto.trowmax, pto.tcolsum, pto.tcolmax,
│   │                                  pto.trowexpand, pto.tcolexpand, pto.trowargmax, etc.
│   ├── Memory and Data Movement   → pto.tload, pto.tstore, pto.tprefetch,
│   │                                  pto.mgather, pto.mscatter
│   ├── Matrix and Matrix-Vector   → pto.tgemv, pto.tgemv_mx, pto.tmatmul, pto.tmatmul_acc,
│   │                                  pto.tmatmul_bias, pto.tmatmul_mx, etc.
│   ├── Layout and Rearrangement   → pto.tmov, pto.ttrans, pto.tconcat, pto.tpack,
│   │                                  pto.textract, pto.tinsert, pto.timg2col, etc.
│   └── Irregular and Complex      → pto.tmrgsort, pto.tsort32, pto.tgather, pto.tquant,
│                                      pto.tdequant, pto.trandom, pto.thistogram, etc.
│
├── Vector Instruction Set
│   ├── Vector Load Store          → pto.vlds, pto.vldas, pto.vgather2, pto.vsld,
│   │                                  pto.vsst, pto.vscatter, pto.vsta, pto.vstar, etc.
│   ├── Predicate and Materialization → pto.vbr, pto.vdup
│   ├── Unary Vector Instructions  → pto.vabs, pto.vneg, pto.vexp, pto.vln, pto.vsqrt,
│   │                                  pto.vrsqrt, pto.vrec, pto.vrelu, pto.vnot, etc.
│   ├── Binary Vector Instructions → pto.vadd, pto.vsub, pto.vmul, pto.vdiv, pto.vmax,
│   │                                  pto.vmin, pto.vand, pto.vor, pto.vxor, etc.
│   ├── Vector-Scalar Instructions → pto.vadds, pto.vmuls, pto.vshls, pto.vlrelu, etc.
│   ├── Conversion Ops             → pto.vci, pto.vcvt, pto.vtrc
│   ├── Reduction Instructions     → pto.vcadd, pto.vcmax, pto.vcmin, pto.vcgadd, etc.
│   ├── Compare and Select         → pto.vcmp, pto.vcmps, pto.vsel, pto.vselr, pto.vselrv2
│   ├── Data Rearrangement         → pto.vintlv, pto.vdintlv, pto.vslide, pto.vshift,
│   │                                  pto.vpack, pto.vzunpack, pto.vperm, etc.
│   └── SFU and DSA Instructions  → pto.vprelu, pto.vexpdiff, pto.vaxpy, pto.vtranspose,
│                                      pto.vsort32, pto.vmrgsort, etc.
│
├── Scalar And Control Instruction Set
│   ├── Control and Configuration  → (reserved for future control ops)
│   ├── Pipeline Sync              → pto.set_flag, pto.wait_flag, pto.pipe_barrier,
│   │                                pto.mem_bar, pto.set_cross_core, pto.get_buf, pto.rls_buf, etc.
│   ├── DMA Copy                   → pto.copy_gm_to_ubuf, pto.copy_ubuf_to_gm,
│   │                                pto.copy_ubuf_to_ubuf, pto.set_loop_size_*, etc.
│   ├── Predicate Load Store       → pto.pld, pto.plds, pto.pldi, pto.pst, pto.psts,
│   │                                pto.psti, pto.pstu
│   └── Predicate Generation       → pto.pset_b8, pto.pge_b8, pto.plt_b8, pto.pand,
│                                      pto.por, pto.pxor, pto.pnot, pto.psel, pto.ppack, etc.
│
├── Communication Instruction Set
    ├── Communication and Runtime  → pto.tbroadcast, pto.tget, pto.tget_async,
    │                               pto.tput, pto.tput_async, pto.treduce,
    │                               pto.tscatter, pto.tgather, pto.tnotify,
    │                               pto.ttest, pto.twait
│
└── System Scheduling Instruction Set
    └── System Scheduling           → pto.tpush, pto.tpop, pto.tfree
```

## Normative Language

Instruction set pages use **MUST**, **SHOULD**, and **MAY** only for rules that a test, verifier, or review can check. Prefer plain language for explanation.

## See Also

- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page format standard
- [Diagnostics and illegal cases](../reference/diagnostics-and-illegal-cases.md) — What makes a PTO program illegal
