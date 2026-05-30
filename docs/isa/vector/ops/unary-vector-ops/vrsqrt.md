# pto.vrsqrt

`pto.vrsqrt` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` holds reciprocal-square-root values per active lane.

## Mechanism

`pto.vrsqrt` computes the lane-wise reciprocal square root: `dst[i] = 1 / sqrt(src[i])`. This is commonly used in normalizing vectors. Inactive lanes leave the destination unchanged.

## Syntax

### PTO Assembly Form

```text
vrsqrt %result, %input, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vrsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%input` | `!pto.vreg<NxT>` | Source vector register; read at each active lane `i` |
| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 (true) are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise reciprocal square root: `dst[i] = 1 / sqrt(src[i])` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

!!! warning "Constraints"
    Only floating-point element types are legal.
      Active inputs containing `+0` or `-0` follow the target's divide-style
      exceptional behavior.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Target-defined numeric exceptional behavior, such as divide-by-zero or out-of-domain inputs, remains subject to the selected backend profile unless this page narrows it further.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `f16, f32`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | A5 RV |
|---|---|---|
| `f32` | 13 | `RV_VMULS` (scalar-mul path) |
| `f16` | 13 | `RV_VMULS` (scalar-mul path) |

### A2/A3 Throughput

| Metric | Value | Constant |
|--------|-------|----------|
| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
| Completion latency | 20 (FP) | `A2A3_COMPL_FP_MUL` |
| Per-repeat throughput | 1 | `A2A3_RPT_1` |
| Pipeline interval | 18 | `A2A3_INTERVAL` |

---

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / sqrtf(src[i]);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vsqrt](./vsqrt.md)
- Next op in instruction set: [pto.vrec](./vrec.md)
