# pto.vsub

`pto.vsub` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

Lane-wise subtraction: `dst[i] = lhs[i] - rhs[i]` for each active lane.

## Mechanism

Computes lane-wise difference of two source vector registers. For each lane `i` where the predicate is true:

$$ \mathrm{dst}_i = \mathrm{lhs}_i - \mathrm{rhs}_i $$

Inactive lanes leave the destination unchanged. The subtraction is type-specific: signed integer subtraction for signed types, unsigned for unsigned types.

## Syntax

### PTO Assembly Form

```text
vsub %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vsub %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vsub ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

Supported element types on A5: `i8-i64`, `f16`, `bf16`, `f32`.

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | Minuend: the value being subtracted from |
| `%rhs` | `!pto.vreg<NxT>` | Subtrahend: the value being subtracted |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise difference: `dst[i] = lhs[i] - rhs[i]` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type match**: `%lhs`, `%rhs`, and `%result` MUST have identical element types.
- **Width match**: All three registers MUST have the same vector width `N`.
- **Mask width**: `%mask` MUST have width equal to `N`.
- **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the subtraction.
- **Inactive lanes**: Destination elements at inactive lanes are unmodified.

## Exceptions

- The verifier rejects illegal operand type mismatches, width mismatches, or mask width mismatches.
- Any additional illegality stated in the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set page is also part of the contract.

## Target-Profile Restrictions

|| Element Type | CPU Simulator | A2/A3 | A5 |
||------------|:-------------:|:------:|:--:|
|| `f32` | Simulated | Simulated | Supported |
|| `f16` / `bf16` | Simulated | Simulated | Supported |
|| `i8`–`i64`, `u8`–`u64` | Simulated | Simulated | Supported |

A5 is the primary concrete profile for the vector instructions. CPU simulation and A2/A3-class targets emulate `pto.v*` operations using scalar loops while preserving the visible PTO contract.

## Performance

### A5 Latency

|| Element Type | Latency (cycles) | A5 RV |
||---|---|---|
|| `f32` | 7 | `RV_VSUB` |
|| `f16` | 7 | `RV_VSUB` |
|| `i32` | 7 | `RV_VSUB` |
|| `i16` | 7 | `RV_VSUB` |

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
    dst[i] = src0[i] - src1[i];
```

### MLIR Usage

```mlir
// Full-vector subtraction (all lanes active)
%result = pto.vsub %lhs, %rhs, %active : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>

// Partial predication: only subtract where %cond is true
%diff = pto.vsub %a, %b, %cond : (!pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask) -> !pto.vreg<128xf16>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vadd](./vadd.md)
- Next op in instruction set: [pto.vmul](./vmul.md)
