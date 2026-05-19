# pto.vmul

`pto.vmul` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

`%result` is the lane-wise product.

## Mechanism

`pto.vmul` is a `pto.v*` compute operation.

## Syntax

### PTO Assembly Form

```asm
vmul %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vmul ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>)
    outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

```cpp
PTO_VMUL_IMPL(result, lhs, rhs, mask);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | First source vector |
| `%rhs` | `!pto.vreg<NxT>` | Second source vector (multiplied with `%lhs` lane-wise) |
| `%mask` | `!pto.mask<G>` | Predication mask; inactive lanes produce zero |

Documented A5 types: `i16-i32`, `f16`, `bf16`, `f32` (**NOT** `i8`/`u8`).

## Expected Outputs

| Operand | Type | Description |
|---------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise product of `%lhs` and `%rhs` |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - Source and result element types MUST match.
    - The A5 profile excludes `i8`/`u8` forms from this instruction set.
    - `mask[i] == 0` lanes in the result are set to zero (zero-merge predication model).

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `i16-i32`, `f16`, `bf16`, `f32` (**NOT** `i8`/`u8`).
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | A5 RV |
|---|---|---|
| `f32` | 8 | `RV_VMUL` |
| `f16` | 8 | `RV_VMUL` |
| `i32` | 8 | `RV_VMUL` |
| `i16` | 8 | `RV_VMUL` |

### A2/A3 Throughput

| Metric | Value | Constant |
|--------|-------|----------|
| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
| Completion: FP | 20 | `A2A3_COMPL_FP_MUL` |
| Completion: INT | 18 | `A2A3_COMPL_INT_MUL` |
| Per-repeat throughput | 2 | `A2A3_RPT_2` |
| Pipeline interval | 18 | `A2A3_INTERVAL` |

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

Integer overflow follows the target-defined behavior. Predicated lanes (where `mask[i] == 0`) produce zero in the destination.

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vsub](./vsub.md)
- Next op in instruction set: [pto.vdiv](./vdiv.md)
