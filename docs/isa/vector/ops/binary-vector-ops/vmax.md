# pto.vmax

`pto.vmax` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

Lane-wise maximum: `dst[i] = (lhs[i] > rhs[i]) ? lhs[i] : rhs[i]` for each active lane.

## Mechanism

Computes the lane-wise maximum of two source vector registers. For each lane `i` where the predicate is true:

$$ \mathrm{dst}_i = \max(\mathrm{lhs}_i, \mathrm{rhs}_i) $$

The comparison follows the element type's ordering. Inactive lanes leave the destination unchanged.

## Syntax

### PTO Assembly Form

```text
vmax %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vmax %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vmax ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

Supported element types on A5: `i8-i32`, `f16`, `bf16`, `f32`.

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | First source vector register (first operand of the comparison) |
| `%rhs` | `!pto.vreg<NxT>` | Second source vector register (second operand of the comparison) |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise maximum: `dst[i] = max(lhs[i], rhs[i])` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type match**: `%lhs`, `%rhs`, and `%result` MUST have identical element types.
- **Width match**: All three registers MUST have the same vector width `N`.
- **Mask width**: `%mask` MUST have width equal to `N`.
- **Active lanes**: Only lanes where the mask bit is 1 participate in the comparison.
- **Inactive lanes**: Destination elements at inactive lanes are unmodified.
- **NaN behavior**: If either operand is NaN (floating-point), the result is NaN.

## Exceptions

- The verifier rejects illegal operand type mismatches, width mismatches, or mask width mismatches.
- Any additional illegality stated in the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set page is also part of the contract.

## Target-Profile Restrictions

|| Element Type | CPU Simulator | A2/A3 | A5 |
||------------|:-------------:|:------:|:--:|
|| `f32` | Simulated | Simulated | Supported |
|| `f16` / `bf16` | Simulated | Simulated | Supported |
|| `i8`–`i32` | Simulated | Simulated | Supported |

A5 is the primary concrete profile for the vector instructions.

## Performance

### A5 Latency

|| Element Type | Latency (cycles) | A5 RV |
||---|---|---|
|| `f32` | 7 | `RV_VMAX` |
|| `f16` | 7 | `RV_VMAX` |
|| `i32` | 7 | `RV_VMAX` |

### A2/A3 Throughput

|| Metric | Value | Constant |
||--------|-------|----------|
|| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
|| Completion: FP32 | 19 | `A2A3_COMPL_FP_BINOP` |
|| Completion: INT | 17 | `A2A3_COMPL_INT_BINOP` |
|| Per-repeat throughput | 2 | `A2A3_RPT_2` |
|| Pipeline interval | 18 | `A2A3_INTERVAL` |

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] > src1[i]) ? src0[i] : src1[i];
```

### MLIR Usage

```mlir
// Element-wise max of two vectors
%result = pto.vmax %a, %b, %active : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>

// Clamp to a minimum value: max(x, lower)
%clamped = pto.vmax %input, %lower, %active : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vdiv](./vdiv.md)
- Next op in instruction set: [pto.vmin](./vmin.md)
