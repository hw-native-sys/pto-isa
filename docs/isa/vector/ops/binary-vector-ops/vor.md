# pto.vor

`pto.vor` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

Lane-wise bitwise OR: `dst[i] = lhs[i] | rhs[i]` for each active lane.

## Mechanism

Computes the lane-wise bitwise OR of two source vector registers. For each lane `i` where the predicate is true:

$$ \mathrm{dst}_i = \mathrm{lhs}_i \,|\, \mathrm{rhs}_i $$

Inactive lanes leave the destination unchanged. This is an integer-only operation.

## Syntax

### PTO Assembly Form

```text
vor %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vor %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vor ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

Supported element types: all integer types (`i8`–`i64`, `u8`–`u64`).

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | Left-hand source vector register |
| `%rhs` | `!pto.vreg<NxT>` | Right-hand source vector register |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same integer element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise bitwise OR: `dst[i] = lhs[i] | rhs[i]` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type**: Integer element types only (no floating-point).
- **Type match**: `%lhs`, `%rhs`, and `%result` MUST have identical element types.
- **Width match**: All three registers MUST have the same vector width `N`.
- **Mask width**: `%mask` MUST have width equal to `N`.
- **Active lanes**: Only lanes where the mask bit is 1 participate.
- **Inactive lanes**: Destination elements at inactive lanes are unmodified.

## Exceptions

- The verifier rejects non-integer element types, type mismatches, width mismatches, or mask width mismatches.
- Any additional illegality stated in the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set page is also part of the contract.

## Target-Profile Restrictions

|| Element Type | CPU Simulator | A2/A3 | A5 |
||------------|:-------------:|:------:|:--:|
|| Integer types | Simulated | Simulated | Supported |

A5 is the primary concrete profile for the vector instructions.

## Performance

### A5 Latency

|| Element Type | Latency (cycles) | A5 RV |
||---|---|---|
|| `i32` | 7 | `RV_VOR` |

### A2/A3 Throughput

|| Metric | Value | Constant |
||--------|-------|----------|
|| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
|| Completion latency | 17 | `A2A3_COMPL_INT_BINOP` |
|| Per-repeat throughput | 2 | `A2A3_RPT_2` |
|| Pipeline interval | 18 | `A2A3_INTERVAL` |

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] | src1[i];
```

### MLIR Usage

```mlir
// Bitwise OR of two integer vectors
%result = pto.vor %a, %b, %active : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>

// Set specific bits
%flags = pto.vbroadcast %c1 : i32 -> !pto.vreg<64xi32>
%set = pto.vor %data, %flags, %active : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vand](./vand.md)
- Next op in instruction set: [pto.vxor](./vxor.md)
