# PTO Micro-Instruction: Vector Execution Scope (`pto.vecscope` / `pto.strict_vecscope`)

This page documents the PTO micro-instruction vector execution scope operations. These ops are part of the PTO micro-instruction surface (A5 Ascend 950 profile) and define the hardware interface between the Scalar Unit and the Vector Thread.

## Overview

`__VEC_SCOPE__` is the IR-level representation of a Vector Function (VF) launch. In the PTO architecture, it defines the hardware interface between the Scalar Unit and the Vector Thread.

In PTO micro-instruction source IR, vector execution scopes are modeled as dedicated region ops. The default form is `pto.vecscope`; when the scope body must reject implicit capture and require explicit region arguments, use `pto.strict_vecscope`.

## Mechanism

`pto.vecscope` and `pto.strict_vecscope` do not compute payload values on their own; they define the lifetime boundary of one vector interval. Inside that interval, vector registers, masks, and alignment carriers are legal, and outside it they are not. The strict form additionally makes the region interface explicit by requiring all external values to cross the boundary as operands and block arguments.

## Inputs

These scope operations take the region body as their primary input. `pto.strict_vecscope` additionally takes an explicit operand list that becomes the body block arguments.

## Expected Outputs

These scope operations delimit vector execution and validate how vector-visible state is used. They do not directly return payload values in the current manual examples; instead they define the region in which the enclosed vector operations execute.

## Execution Model

The PTO micro-instruction operates on the Ascend 950's **Decoupled Access-Execute** (DAE) architecture. The execution model follows **non-blocking fork semantics**:

- **Scalar invocation**: the scalar processor invokes a vector thread by calling a VF. Once the launch command is issued, the scalar unit does not stall and continues executing subsequent instructions in the pipeline.
- **Vector execution**: after invocation, the vector thread independently fetches and executes the instructions defined within the VF scope.
- **Parallelism**: this decoupled execution allows the scalar and vector units to run in parallel, so the scalar unit can prepare addresses or manage control flow while the vector unit performs heavy SIMD computation.

### Launch Mechanism And Constraints

- **Parameter buffering**: all arguments required by the VF must be staged in hardware-specific buffers.
- **Launch overhead**: launching a VF incurs a latency of a few cycles. Very small VFs should account for this overhead because launch cost can rival useful computation time.

---

## `pto.vecscope` — Default Vector Scope

### Syntax

```mlir
pto.vecscope {
  // region body
}
```

### Semantics

`pto.vecscope` allows the body to use surrounding SSA values directly (implicit capture). All operations that produce or consume `!pto.vreg`, `!pto.mask<...>`, or `!pto.align` must be enclosed by exactly one vector interval.

### Constraints

- Nested vector intervals are **not** part of the legal VPTO surface. Ordinary nested `scf.for` structure is fine, but one vector interval may not contain another vector interval.
- Regardless of whether the source form uses `pto.vecscope`, `pto.strict_vecscope`, or a lowered carrier loop with `llvm.loop.aivector_scope`, every op that produces or consumes `!pto.vreg`, `!pto.mask<...>`, or `!pto.align` must be enclosed by exactly one vector interval.

### Examples

```mlir
pto.set_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.copy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64,
    %false, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64

pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.vecscope {
  scf.for %lane = %c0 to %9 step %c64 {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %v = pto.vlds %2[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  }
}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.copy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64
```

---

## `pto.strict_vecscope` — Strict Vector Scope

### Syntax

```mlir
pto.strict_vecscope(%arg1, %arg2, ...) {
^bb0(%in1: <type>, %in2: <type>, ...):
  // region body — all external values must come through operands
}
: (<type1>, <type2>, ...) -> ()
```

### Semantics

`pto.strict_vecscope` requires every external value used by the body to be passed through the op operand list and received as a body block argument. It rejects implicit capture from the surrounding scope.

### Constraints

- `pto.strict_vecscope` rejects implicit capture from the surrounding scope.
- Both ops still represent one explicit VPTO vector interval.
- The scope op itself only defines the vector-interval boundary and region argument contract.

### Examples

```mlir
pto.strict_vecscope(%ub_in, %ub_out, %lane, %remaining) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index, %rem: i32):
  %mask, %next_remaining = pto.plt_b32 %rem : i32 -> !pto.mask<b32>, i32
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index, i32) -> ()
```

Use `pto.strict_vecscope` when the source form should make all vector-scope inputs explicit in the region signature instead of relying on surrounding SSA visibility.

---

## Comparison: `pto.vecscope` vs `pto.strict_vecscope`

| Aspect | `pto.vecscope` | `pto.strict_vecscope` |
|--------|----------------|----------------------|
| Implicit capture | Allowed | Rejected |
| Region arguments | Derived from surrounding SSA | Must be declared in operand list |
| Use case | Simple kernels, quick authoring | Formal verification, IR rewriting |
| SSA visibility | Body can reference outer SSA values | All inputs passed as block arguments |

---

## Relationship to Hardware Pipeline

Inside a vector scope, the Decoupled Access-Execute (DAE) architecture requires explicit synchronization between:

- **MTE2** (PIPE_MTE2): DMA copy-in from GM to UB
- **PIPE_V**: Vector ALU operations
- **MTE3** (PIPE_MTE3): DMA copy-out from UB to GM

Synchronization can be achieved through:
- `pto.set_flag` / `pto.wait_flag` (event-based)
- `pto.get_buf` / `pto.rls_buf` (buffer-based, recommended)

## Related Operations

- Pipeline sync: [Pipeline Synchronization](../../pipeline-sync.md) — `pto.set_flag`, `pto.wait_flag`, `pto.get_buf`, `pto.rls_buf`
- Memory barrier: [Pipeline Synchronization](../../pipeline-sync.md) — `pto.mem_bar`
- Scalar arithmetic: [Shared Scalar Arithmetic](../../shared-arith.md)
- Structured control: [Shared SCF](../../shared-scf.md)
- BlockDim queries: [BlockDim Query Operations](./block-dim-query.md)
