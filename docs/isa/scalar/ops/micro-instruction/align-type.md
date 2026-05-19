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

### `pto.init_align` — Initialize Store-Side Align Carrier

**Syntax:** `%result = pto.init_align : !pto.align`

**Semantics:** Initialize store-side align carrier state.

**Outputs:** `%result` is a fresh zero-initialized align carrier for **store-side** unaligned streams such as `pto.vstus`, `pto.vstur`, `pto.vstar`, `pto.vstas`, and `pto.pstu`.

**Constraints:** This op is for store-family initialization only. Unaligned load streams still start from `pto.vldas`, not `pto.init_align`.

```c
align = init_align();
```

### `pto.vldas` — Prime Alignment for Unaligned Load

**Syntax:** `%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align`

**Semantics:** Prime alignment buffer for subsequent unaligned load.

**Inputs:** `%source` is the UB address whose surrounding aligned block seeds the load alignment state.

**Outputs:** `%result` is the initialized load-alignment state.

**Constraints:**

- This op is the required leading operation for a `pto.vldus` stream using the same alignment state.
- The source address itself need not be 32-byte aligned; hardware truncates it to the aligned block boundary for the priming load.

**Latency:** **9** cycles.

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
```

### `pto.vldus` — Unaligned Load with Alignment State Update

**Syntax:** `%result, %align_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align`

**Semantics:** Unaligned load using primed align state.

**Inputs:** `%source` is the current UB address; `%align` is the incoming load alignment state primed by `pto.vldas` or a prior `pto.vldus`.

**Outputs:** `%result` is the assembled vector value; `%align_out` is the updated alignment state.

**Constraints:**

- A matching `pto.vldas` MUST appear before the first dependent `pto.vldus` stream in the same vector loop.
- The installed no-post A5 interface keeps a struct-shaped internal return for lowering convenience, but its no-post `base` field is not meaningful user-visible state. VPTO therefore hides that value and only exposes the updated align carrier.
- Reusing the original `%source` starts a new explicit access point; if the caller wants another no-post access, it should compute the next source pointer explicitly and pair it with the required align setup.

**Latency:** **9** cycles.

```mlir
%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
```

### `pto.vstus` — No-Post Unaligned Store with Scalar Offset

**Syntax:** `%align_out = pto.vstus %align_in, %offset, %value, %base : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`

**Semantics:** No-post unaligned store with scalar offset.

**Inputs:** `%align_in` is the incoming store-alignment state, `%offset` is the scalar displacement, `%value` is the vector being stored, and `%base` is the UB base pointer.

**Outputs:** `%align_out` is the updated buffered-tail state.

**Constraints:**

- This is the scalar-offset stateful form of the unaligned store family. The first `%align_in` in the stream should come from `pto.init_align`.
- This op does **not** mean "store a full vector starting at `%base + %offset`". Instead, `%offset` describes how far the store stream advances at this step, and `%align_out` carries any residual tail that could not be committed yet.
- The no-post surface does not expose an updated base pointer. A later flush op (`pto.vstas` / `pto.vstar`) must therefore use an explicit destination/offset pair that identifies the same logical flush point as this `pto.vstus`.

**Latency:** **9** cycles.

```mlir
%store_align = pto.init_align : !pto.align
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
%store_align0 = pto.init_align : !pto.align
%align_out1 = pto.vstus %store_align0, %c32, %result0, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
%align_out2 = pto.vstus %align_out1, %c32, %result1, %ub_out : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

## Constraints

!!! warning "Constraints"
    - `pto.vldas` must be the leading operation of an unaligned load stream.
    - `pto.vldus` must follow `pto.vldas` using the same alignment state.
    - Store-side unaligned streams (`pto.vstus` and the related `pto.vstur`, `pto.vstar`, `pto.vstas`, `pto.pstu`) must be initialized by `pto.init_align`. `pto.init_align` is **store-side only** — it cannot be used to prime a load stream.
    - The alignment state must be threaded through all operations in the stream without branching.
    - For `pto.vstus`, `%offset` controls how far the store stream advances at each step, not the absolute store displacement from `%base`. A later flush op (`pto.vstas` / `pto.vstar`) must reuse the matching destination/offset pair.

## Why Explicit Alignment State?

On hardware that supports unaligned memory operations through internal alignment buffers, the state of those buffers must be managed explicitly. `!pto.align` makes this state visible in the SSA form, enabling:

1. **Correctness verification**: the compiler can verify that alignment state is properly threaded through a stream.
2. **Scheduling analysis**: operations that consume/produce alignment state can be correctly ordered.
3. **IR rewriting**: transformations can reason about alignment state without relying on hidden hardware state.

## Related Operations

- Vector load/store: [Vector Load Store](../../../vector/vector-load-store.md) — `pto.vlds`, `pto.vsts`
- Strict vecscope: [Vector Execution Scope](./vecscope.md) — `pto.vecscope`, `pto.strict_vecscope`
