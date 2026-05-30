# pto.vsqrt

`pto.vsqrt` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` holds the square root per active lane.

## Mechanism

`pto.vsqrt` computes the lane-wise square root. For each lane `i` where the predicate is true, `dst[i] = sqrt(src[i])`. Domain behavior for negative inputs is target-defined. Inactive lanes leave the destination unchanged.

## Syntax

### PTO Assembly Form

```text
vsqrt %result, %input, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
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
| `%result` | `!pto.vreg<NxT>` | Lane-wise square root: `dst[i] = sqrt(src[i])` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

!!! warning "Constraints"
    Only floating-point element types are legal.
      Negative active inputs follow the target's exception/NaN rules.

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
| `f32` | 17 | `RV_VSQRT` |
| `f16` | 22 | `RV_VSQRT` |

### A2/A3 Throughput

| Metric | Value (f32) | Value (f16) |
|--------|-------------|-------------|
| Startup latency | 13 (`A2A3_STARTUP_REDUCE`) | 13 |
| Completion latency | 27 (`A2A3_COMPL_FP32_SQRT`) | 29 (`A2A3_COMPL_FP16_SQRT`) |
| Per-repeat throughput | 2 | 4 |
| Pipeline interval | 18 | 18 |

### Execution Note

`vsqrt` uses the SFU (Special Function Unit) pipeline. For a 1024-element vector with 16 iterations:

```
A5 (f32): 17 latency, pipelined throughput ≈ 2 cycles/iteration
A5 (f16): 22 latency, pipelined throughput ≈ 4 cycles/iteration
```

`vrsqrt` (reciprocal square root) shares the same SFU hardware and has equivalent latency.

---

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

`sqrtf` follows IEEE 754 square-root semantics. For floating-point types, negative active inputs produce NaN; out-of-domain inputs follow the target's exception/NaN rules.

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vln](./vln.md)
- Next op in instruction set: [pto.vrsqrt](./vrsqrt.md)
