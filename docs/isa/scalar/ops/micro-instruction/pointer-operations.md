# PTO Micro-Instruction: Pointer Operations

This page documents the PTO micro-instruction pointer operations that construct typed pointers and perform pointer arithmetic. These ops are part of the PTO micro-instruction surface (A5 Ascend 950 profile).

## Overview

PTO micro-instruction memory operands use `!pto.ptr<element-type, space>` to distinguish between GM (`!pto.ptr<T, gm>`) and UB (`!pto.ptr<T, ub>`) address spaces. The following ops construct and manipulate typed PTO pointers.

## Mechanism

These operations make pointer typing and pointer arithmetic explicit in SSA form. `pto.castptr` materializes a typed PTO pointer from a scalar address, `pto.addptr` advances it in element units, and the scalar load/store helpers access one element at a time without switching into the vector load/store instruction families.

## Address Space Conventions

| Space | Interpretation |
|-------|----------------|
| `gm` | Global Memory (GM), off-chip HBM/DDR storage |
| `ub` | Vector tile buffer (implemented on current hardware by the Unified Buffer / UB) |

Typical pointer construction and pointer arithmetic follow the same `!pto.ptr<..., space>` form. Pointer arithmetic is element-based rather than byte-based.

---

## `pto.castptr`

**Syntax:** `%result = pto.castptr %addr : i64 -> !pto.ptr<T, space>`

**Semantics:** Reinterpret a scalar address value as a typed PTO pointer in the target memory space.

```c
result = (ptr<T, space>)addr;
```

### Inputs

| Operand | Type | Description |
|--------|------|-------------|
| `%addr` | `i64` | Scalar address value to cast |

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.ptr<T, space>` | Typed PTO pointer in the target memory space |

### Constraints

- `pto.castptr` is a pointer-construction operation. It does not perform data movement and does not by itself imply any load/store side effect.
- The element type `T` and memory space `space` must be consistent with the intended use.
- `T` is the element type associated with the pointed-to storage.
- `space` is the memory domain, typically `gm` or `ub`.

### Examples

```mlir
// Cast integer 0 to a f32 UB pointer
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>

// Cast GM base address to a bf16 GM pointer
%gm_ptr = pto.castptr %gm_base : i64 -> !pto.ptr<bf16, gm>
```

---

## `pto.addptr`

**Syntax:** `%result = pto.addptr %ptr, %offset : !pto.ptr<T, space> -> !pto.ptr<T, space>`

**Semantics:** Compute a new pointer by advancing the base pointer by an element offset.

```c
result = ptr + offset;  // offset counted in elements, not bytes
```

### Inputs

| Operand | Type | Description |
|--------|------|-------------|
| `%ptr` | `!pto.ptr<T, space>` | Base pointer |
| `%offset` | `index` or `i64` | Element offset to advance |

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.ptr<T, space>` | New pointer advanced by offset elements |

### Constraints

- `pto.addptr` preserves both the element type `T` and the memory-space tag `space`.
- The offset is counted in **elements**, not bytes. For example, advancing a `!pto.ptr<f32, ub>` by 1024 advances by 1024 × 4 = 4096 bytes.
- Pointer arithmetic that results in an out-of-bounds address is target-defined.

### Examples

```mlir
// Advance UB pointer by 1024 f32 elements (4096 bytes)
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>

// Compute row offset in a 2D tile
%row_ptr = pto.addptr %base_ptr, %row_offset : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
```

---

## `pto.load_scalar`

**Syntax:** `%value = pto.load_scalar %ptr[%offset] : !pto.ptr<T, space> -> T`

**Semantics:** Load one scalar element from a pointer-like operand.

```c
value = ptr[offset];
```

### Inputs

| Operand | Type | Description |
|--------|------|-------------|
| `%ptr` | `!pto.ptr<T, space>` | Typed PTO pointer |
| `%offset` | `index` | Element displacement counted in elements |

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%value` | `T` | Loaded scalar element |

### Constraints

- The result type MUST match the element type of `%ptr`.
- This op is a scalar memory helper; unlike `pto.vlds`, it does not produce a `vreg` result and does not participate in vector load `dist` families.
- UB address space is the primary valid space; GM scalar loads are target-defined.

### Examples

```mlir
// Load one f32 scalar from UB
%val = pto.load_scalar %ub_ptr[%c4] : !pto.ptr<f32, ub> -> f32
```

---

## `pto.store_scalar`

**Syntax:** `pto.store_scalar %value, %ptr[%offset] : !pto.ptr<T, space>, T`

**Semantics:** Store one scalar element to a pointer-like operand.

```c
ptr[offset] = value;
```

### Inputs

| Operand | Type | Description |
|--------|------|-------------|
| `%value` | `T` | Scalar value to store |
| `%ptr` | `!pto.ptr<T, space>` | Typed PTO pointer |
| `%offset` | `index` | Element displacement counted in elements |

### Constraints

- The stored value type MUST match the element type of `%ptr`.
- This op is a scalar memory helper; unlike `pto.vsts`, it does not consume a mask and does not target vector-store `dist` families.
- UB address space is the primary valid space; GM scalar stores are target-defined.

### Examples

```mlir
// Store one f32 scalar to UB
pto.store_scalar %val, %ub_ptr[%c8] : !pto.ptr<f32, ub>, f32
```

---

## Pointer-Based Vector Access Pattern

The following example shows how typed PTO pointers flow through pointer construction, pointer arithmetic, structured control flow, and PTO memory ops:

```mlir
// Materialize typed UB pointers
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>

pto.vecscope {
  %16 = scf.for %arg3 = %c0 to %11 step %c64 iter_args(%arg4 = %12) -> (i32) {
    // Generate tail mask
    %mask, %scalar_out = pto.plt_b32 %arg4 : i32 -> !pto.mask<b32>, i32

    // Scalar load from UB
    %s = pto.load_scalar %1[%c4] : !pto.ptr<f32, ub> -> f32

    // Scalar store to UB
    pto.store_scalar %s, %1[%c8] : !pto.ptr<f32, ub>, f32

    // Vector load from UB
    %17 = pto.vlds %1[%arg3] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

    // Vector operation
    %18 = pto.vabs %17, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

    // Vector store to UB
    pto.vsts %18, %10[%arg3], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

    scf.yield %scalar_out : i32
  }
}
```

## Related Operations

- BlockDim queries: [BlockDim Query Operations](./block-dim-query.md) — `pto.get_block_idx`, `pto.get_block_num`
- Vector load/store: [Vector Load Store](../../../vector/vector-load-store.md) — `pto.vlds`, `pto.vsts`
- Scalar arithmetic: [Shared Scalar Arithmetic](../../shared-arith.md) — `arith.constant`, `arith.index_cast`
