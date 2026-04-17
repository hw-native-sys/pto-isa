# PTO Micro-Instruction: BlockDim and Runtime Query Operations

This page documents the PTO micro-instruction runtime query operations that expose block-level execution coordinates to scalar code. These ops are part of the PTO micro-instruction surface (A5 Ascend 950 profile) and are distinct from the Tile-level ISA.

## Overview

These ops expose the current kernel instance's execution coordinates to scalar code. They are the PTO-level equivalent of runtime queries such as `GetBlockIdx()` and `GetBlockNum()` in GPU kernel programming models.

Use them when the same kernel body is launched across multiple blocks or subblocks and each execution instance must figure out which slice of the global workload it owns.

## Mechanism

The block-dimension query operations are pure scalar producers. They do not move data or synchronize pipelines; instead they expose launch-time execution coordinates so surrounding scalar arithmetic and pointer formation can derive the local GM or UB window owned by the current block or subblock.

## BlockDim Query Operations

### Common Pattern

A common pattern is:
- Split the full input/output tensor into `block_num` disjoint block-sized regions
- Let each block compute its own starting offset from `block_idx`
- Within one block, further tile the local region and drive the tile loop with ordinary scalar `arith` / `scf` ops

For example, if a tensor is split evenly across 8 blocks and each block handles `block_length = 2048` elements, then block `b` owns the global range `[b * block_length, (b + 1) * block_length)`. The per-block GM base pointer can be formed by adding `block_idx * block_length` elements to the original base pointer.

### Execution Model

At the PTO micro-instruction level, these runtime-query ops are **pure scalar producers**. They do not perform data movement, do not allocate memory, and do not by themselves create tiling or double buffering. Instead, they provide the scalar values used by surrounding address computation and structured control flow.

---

## `pto.get_block_idx`

**Syntax:** `%block = pto.get_block_idx`

**Result:** `i64`

**Semantics:** Return the current block ID in the range `[0, pto.get_block_num())`.

```c
block = block_idx();
```

### Inputs

None.

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%block` | `i64` | Current block ID in range `[0, block_num)` |

### Constraints

- The returned value is in the range `[0, get_block_num())`.
- These ops are valid only within a kernel launch context that defines block dimensions.

### Examples

```mlir
// Get current block index
%block = pto.get_block_idx
```

---

## `pto.get_subblock_idx`

**Syntax:** `%subblock = pto.get_subblock_idx`

**Result:** `i64`

**Semantics:** Return the current subblock ID in the range `[0, pto.get_subblock_num())`.

```c
subblock = subblock_idx();
```

### Inputs

None.

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%subblock` | `i64` | Current subblock ID in range `[0, subblock_num)` |

### Constraints

- The returned value is in the range `[0, get_subblock_num())`.

### Examples

```mlir
// Get current subblock index
%subblock = pto.get_subblock_idx
```

---

## `pto.get_block_num`

**Syntax:** `%block_num = pto.get_block_num`

**Result:** `i64`

**Semantics:** Return the total number of launched blocks visible to the current kernel instance.

```c
block_num = block_num();
```

### Inputs

None.

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%block_num` | `i64` | Total number of launched blocks |

### Constraints

- The returned value is a positive integer representing the total block count.

### Examples

```mlir
// Get total number of blocks
%block_num = pto.get_block_num
```

---

## `pto.get_subblock_num`

**Syntax:** `%subblock_num = pto.get_subblock_num`

**Result:** `i64`

**Semantics:** Return the total number of visible subblocks for the current execution instance.

```c
subblock_num = subblock_num();
```

### Inputs

None.

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%subblock_num` | `i64` | Total number of visible subblocks |

### Constraints

- The returned value is a positive integer representing the total subblock count per block.

---

## Typical Usage: Block-Level Data Partitioning

```mlir
// Get block-level coordinates
%block = pto.get_block_idx
%block_num = pto.get_block_num

// Compute per-block parameters
%block_len = arith.constant 2048 : index
%block_len_i64 = arith.index_cast %block_len : index to i64

// Compute block offset
%base = arith.index_cast %block : i64 to index
%offset = arith.muli %base, %block_len : index

// Adjust GM base pointers for this block
%block_in = pto.addptr %gm_in, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
%block_out = pto.addptr %gm_out, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

In this pattern, all blocks execute the same kernel body, but each block sees a different `%block` value and therefore computes a different GM window.

### Grid Design Considerations

When designing the block grid:

| Grid Dimension | Use Case |
|--------------|---------|
| `block_num` | Parallelism across disjoint data regions |
| `subblock_num` | Hierarchical tiling within each block |

## Related Operations

- Pointer arithmetic: [Pointer Operations](./pointer-operations.md) — `pto.addptr`, `pto.castptr`
- Scalar memory access: [Pointer Operations](./pointer-operations.md) — `pto.load_scalar`, `pto.store_scalar`
- Scalar arithmetic: [Shared Scalar Arithmetic](../../shared-arith.md) — `arith.constant`, `arith.index_cast`, `arith.muli`
