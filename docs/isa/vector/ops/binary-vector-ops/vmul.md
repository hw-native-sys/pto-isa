# pto.vmul

`pto.vmul` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

Lane-wise multiplication of two vector registers under a predicate mask.

## Mechanism

`pto.vmul` is a `pto.v*` compute operation. It multiplies `%lhs` and `%rhs` element-by-element on active lanes and returns the product vector. The PTO contract for this page uses zeroing predication: lanes masked off by `%mask` produce zero in `%result`.

## Syntax

### PTO Assembly Form

```mlir
%result = pto.vmul %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vmul %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2 (DPS)

```mlir
pto.vmul ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
         outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

The installed Bisheng public intrinsic is `vmul(...)`; use an explicit predicate mode when you want the PTO zeroing behavior described on this page.

```cpp
vector_f32 dst;
vector_f32 src0;
vector_f32 src1;
vector_bool mask;
vmul(dst, src0, src1, mask, __cce_simd::MODE_ZEROING);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | First source vector |
| `%rhs` | `!pto.vreg<NxT>` | Second source vector |
| `%mask` | `!pto.mask<G>` | Predicate mask; active lanes are multiplied, inactive lanes are zeroed |

Documented A5 types or forms: `i16-i32`, `f16`, `bf16`, `f32` (**not** `i8` / `u8`).

## Expected Outputs

| Operand | Type | Description |
|---------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise product of `%lhs` and `%rhs` |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- `%lhs`, `%rhs`, and `%result` MUST have the same vector shape and element type.
- `%mask` MUST be legal for the selected vector width.
- The documented A5 profile excludes `i8` / `u8` forms for this operation.
- Inactive lanes follow the zeroing semantics described by this PTO page.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the [Binary Vector Instructions](../../binary-vector-ops.md) page is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i16-i32`, `f16`, `bf16`, `f32`.
- CPU simulation and A2/A3-class targets may emulate the visible PTO contract in software.
- Code that depends on exact latency or element-type coverage should treat that dependency as target-profile-specific.

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

## Examples

### C-style semantics

```c
for (int i = 0; i < N; ++i)
  dst[i] = mask[i] ? lhs[i] * rhs[i] : 0;
```

### MLIR form

```mlir
%prod = pto.vmul %x, %y, %mask : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vsub](./vsub.md)
- Next op in instruction set: [pto.vdiv](./vdiv.md)
- Vector instruction overview: [Vector Instructions](../../../instruction-surfaces/vector-instructions.md)
