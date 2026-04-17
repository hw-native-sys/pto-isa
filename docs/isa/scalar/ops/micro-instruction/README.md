# PTO Micro-Instruction Reference

This section documents the PTO micro-instruction surface for the A5 (Ascend 950) profile. These instructions operate at the vector-pipeline level, making DMA setup, vector registers, masks, synchronization, and `__VEC_SCOPE__` boundaries explicit.

> **Note:** This section is distinct from the Tile-level ISA. Tile instructions (`pto.t*`) operate on tiles with layout and valid-region metadata, while micro-instructions operate on vector registers (`vreg`), masks, and scalar state.

## Instruction Groups

| Group | Description | Operations |
|-------|-------------|-----------|
| [BlockDim and Runtime Query](./block-dim-query.md) | Block and subblock index/number queries | `pto.get_block_idx`, `pto.get_subblock_idx`, `pto.get_block_num`, `pto.get_subblock_num` |
| [Pointer Operations](./pointer-operations.md) | Typed pointer construction and arithmetic | `pto.castptr`, `pto.addptr`, `pto.load_scalar`, `pto.store_scalar` |
| [Vector Execution Scope](./vecscope.md) | Vector function launch and scope boundary | `pto.vecscope`, `pto.strict_vecscope` |
| [Alignment State Type](./align-type.md) | Unaligned load/store alignment management | `pto.init_align`, `pto.vldas`, `pto.vldus`, `pto.vstus` |

## Scope

PTO micro-instruction source programs are not restricted to `pto` operations alone. In practice they also use shared MLIR dialect ops:

- **`arith`**: Full scalar `arith` surface — scalar constants, arithmetic, comparisons, selects, casts, and shifts. See [Shared Scalar Arithmetic](../../shared-arith.md).
- **`scf`**: Structured control flow — counted loops (`scf.for`), conditionals (`scf.if`), while loops (`scf.while`). See [Shared SCF](../../shared-scf.md).

These shared-dialect ops are part of the supported PTO micro-instruction source surface and are regarded as part of PTO-ISA alongside `pto` dialect operations.

## Mechanism

This section of the manual explains the source-level micro-instruction model rather than one opcode. The key contract is that PTO micro-instruction code makes vector-pipeline state explicit: pointers are typed, masks are first-class SSA values, alignment carriers are explicit, and vector execution is fenced by `pto.vecscope`-style regions instead of being inferred from hidden backend state.

## Inputs

This landing page has no instruction operands of its own. Readers should treat the listed instruction groups as the entry points into the micro-instruction surface.

## Expected Outputs

This page defines the micro-instruction documentation map and the architectural concepts needed to read the per-group pages. It does not produce an SSA value or change execution state by itself.

## Constraints

- The PTO micro-instruction surface is profile-specific; this reference documents the A5-oriented surface used by the current manual.
- Micro-instruction code still shares scalar `arith` and `scf` constructs with the broader PTO source surface.
- Readers should not treat the micro-instruction surface as interchangeable with the tile instruction surface: the operand model, scheduling model, and state carriers are different.

## Relationship to PTO Tile ISA

| Aspect | PTO Tile ISA (`pto.t*`) | PTO Micro-ISA (`pto.v*`, `pto.*`) |
|--------|------------------------|----------------------------------|
| Abstraction level | Tiles (multi-dimensional buffers with layout and valid regions) | Vector registers, masks, scalar state |
| Operand model | `!pto.tile<shape x type x layout>` | `!pto.vreg<NxT>`, `!pto.mask<G>` |
| Data movement | GM ↔ Tile (with layout transform) | UB ↔ vreg, GM ↔ UB (DMA) |
| Scheduling model | Tile-level scheduling and fusion | Vector-pipeline scheduling, DAE |

## Key Architectural Concepts

### Vector Lane (VLane)

The vector register is organized as **8 VLanes** of 32 bytes each. A VLane is the atomic unit for group reduction operations.

```
vreg (256 bytes total):
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

Elements per VLane by data type:

| Data Type | Elements/VLane | Total Elements/vreg |
|-----------|---------------|-------------------|
| i8/si8/ui8 | 32 | 256 |
| i16/si16/ui16/f16/bf16 | 16 | 128 |
| i32/si32/ui32/f32 | 8 | 64 |
| i64/si64/ui64 | 4 | 32 |

### Mask Types

`mask<G>`: `!pto.mask<G>` Typed predicate-register view. `G` is one of `b8`, `b16`, `b32` and records the byte-granularity interpretation used by VPTO ops and verifiers.

| Mask Type | Bytes / Element Slot | Typical Element Family | Derived Logical Lanes |
|-----------|----------------------|------------------------|-----------------------|
| `!pto.mask<b32>` | 4 | `f32` / `i32` | 64 |
| `!pto.mask<b16>` | 2 | `f16` / `bf16` / `i16` | 128 |
| `!pto.mask<b8>` | 1 | 8-bit element family | 256 |

### Memory Hierarchy

```
┌─────────────────────────────────────────────┐
│                 Global Memory (GM)           │
│              (Off-chip HBM/DDR)             │
└─────────────────────┬───────────────────────┘
                      │ DMA (MTE2 inbound / MTE3 outbound)
┌─────────────────────▼───────────────────────┐
│   Vector Tile Buffer (hardware UB, 256KB)    │
└─────────────────────┬───────────────────────┘
                      │ Vector Load/Store (PIPE_V)
┌─────────────────────▼───────────────────────┐
│           Vector Register File (VRF)          │
│     vreg (256B each) + mask (256-bit each)  │
└─────────────────────────────────────────────┘
```

### Predication Behavior (Zero-Merge)

The native hardware predication mode is **ZEROING** — inactive lanes produce zero:

```c
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0    // ZEROING mode
```

## Related Sections

- [Vector ISA Reference](../../../vector/README.md) — Vector instruction reference at the PTO Tile ISA level
- [Scalar And Control Reference](../../README.md) — Control and configuration operations
- [Pipeline Synchronization](../../pipeline-sync.md) — Synchronization primitives
- [DMA Copy](../../dma-copy.md) — GM↔vector-tile-buffer data transfer
