# PTO Micro-Instruction: Alignment State Type (`!pto.align`)

This page documents the `!pto.align` type and its associated alignment-state operations. These are part of the PTO micro-instruction surface (A5 Ascend 950 profile).

## Overview

`!pto.align` models the A5 vector-align carrier state. It is not payload data — it is a state carrier that threads through unaligned load/store sequences to manage hardware alignment buffers.

## Mechanism

The `!pto.align` carrier makes hidden alignment-buffer state explicit in SSA form. A priming operation such as `pto.vldas` or `pto.init_align` creates the carrier, each unaligned load/store consumes one carrier value and produces the next, and the stream remains well-formed only when that state is threaded linearly through the sequence.

## Inputs

This page documents one architectural type and the operations that consume or produce it. The concrete inputs are the pointer, offset, vector, and alignment operands listed on each sub-operation below.

## Expected Outputs

The page defines the contract of `!pto.align` and the stream discipline around it. The documented operations either produce a new alignment carrier, consume one, or do both together with payload data.

## The `!pto.align` Type

`!pto.align` is the SSA carrier for alignment-buffer state used by unaligned load/store families. The PTO micro-instruction representation makes that state explicit rather than implicit.

### Key Properties

- `!pto.align` is **not** a payload type — it carries alignment state, not data.
- It must be threaded through a sequence of unaligned memory operations.
- A trailing flush form may still be required to complete the stream.
- Stateful unaligned forms expose their evolving state in SSA form.

## Alignment State Operations

### `pto.init_align`

**Syntax:** `%align = pto.init_align : -> !pto.align`

**Semantics:** Initialize a new alignment state carrier.

```c
align = init_align();
```

### `pto.vldas` — Prime Alignment for Unaligned Load

**Syntax:** `%align = pto.vldas %ub : !pto.ptr<T, ub> -> !pto.align`

**Semantics:** Prime the alignment buffer for a subsequent unaligned load. The source address's surrounding aligned block seeds the load alignment state.

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
```

### `pto.vldus` — Unaligned Load with Alignment State Update

**Syntax:** `%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align`

**Semantics:** Perform an unaligned load using the provided alignment state, and produce both the loaded vector and the updated alignment state.

```mlir
%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
```

### `pto.vstus` — Unaligned Store with Alignment State Update

**Syntax:** `%align_out = pto.vstus %align, %offset, %vec, %ub : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`

**Semantics:** Perform an unaligned store using the provided alignment state, and produce the updated alignment state.

```mlir
%store_align = pto.init_align : -> !pto.align
%next_align = pto.vstus %store_align, %offset, %vec, %ub
    : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

## Complete Alignment State Stream Pattern

The following example shows the complete unaligned load/store stream lifecycle:

```mlir
// ─── Load stream ───
// Prime alignment buffer
%align0 = pto.vldas %ub_in : !pto.ptr<f32, ub> -> !pto.align

// Stream through unaligned loads
%v0, %align1 = pto.vldus %ub_in, %align0 : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
%v1, %align2 = pto.vldus %ub_in, %align1 : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align

// ─── Compute ───
%result0 = pto.vabs %v0, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%result1 = pto.vabs %v1, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// ─── Store stream ───
%store_align0 = pto.init_align : -> !pto.align
%align_out1 = pto.vstus %store_align0, %c32, %result0, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
%align_out2 = pto.vstus %align_out1, %c32, %result1, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

## Constraints

- `pto.vldas` must be the leading operation of an unaligned load stream.
- `pto.vldus` must follow `pto.vldas` using the same alignment state.
- `pto.vstus` must be preceded by `pto.init_align` to start a new store alignment stream.
- The alignment state must be threaded through all operations in the stream without branching.
- For `pto.vstus`, the `%offset` parameter controls the per-operation stride within the stream.

## Why Explicit Alignment State?

On hardware that supports unaligned memory operations through internal alignment buffers, the state of those buffers must be managed explicitly. `!pto.align` makes this state visible in the SSA form, enabling:

1. **Correctness verification**: the compiler can verify that alignment state is properly threaded through a stream.
2. **Scheduling analysis**: operations that consume/produce alignment state can be correctly ordered.
3. **IR rewriting**: transformations can reason about alignment state without relying on hidden hardware state.

## Related Operations

- Vector load/store: [Vector Load Store](../../../vector/vector-load-store.md) — `pto.vlds`, `pto.vsts`
- Strict vecscope: [Vector Execution Scope](./vecscope.md) — `pto.vecscope`, `pto.strict_vecscope`
