# pto.vabs

`pto.vabs` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` receives the lane-wise absolute values.

## Mechanism

`pto.vabs` computes the lane-wise absolute value. For each lane `i` where the predicate is true, `dst[i] = abs(src[i])`. Inactive lanes leave the destination unchanged.

## Syntax

### PTO Assembly Form

```text
vabs %result, %input, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

Documented A5 types or forms: `i8-i32, f16, f32`.

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%input` | `!pto.vreg<NxT>` | Source vector register; read at each active lane `i` |
| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 (true) are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise absolute values: `dst[i] = abs(src[i])` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

!!! warning "Constraints"
    Source and result types MUST match. Integer
      overflow on the most-negative signed value follows the target-defined
      behavior.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `i8-i32, f16, f32`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | Notes |
|---|---|---|
| `f32` | 5 | |
| `f16` | 5 | |
| `i32` | 5 | |
| `i16` | 5 | |
| `i8` | 5 | |

### A2/A3 Throughput

| Metric | Value | Formula |
|--------|-------|---------|
| Startup latency | 14 (`A2A3_STARTUP_BINARY`) | |
| Completion latency | 19 (FP) / 17 (INT) | |
| Per-repeat throughput | 1 | |
| Pipeline interval | 18 | |
| Cycle model | `14 + C + 1×R + (R-1)×18` | C=completion, R=repeats |

### Execution Throughput

On A5, `vabs` has the **lowest latency** of all unary operations (5 cycles). For a 1024-element vector with 16 iterations (64 elements/iteration):

```
A5 cycles ≈ 5 (first) + 15 × 1 ≈ 20 cycles (pipelined)
A2/A3 cycles ≈ 14 + C + R + (R-1)×18  (depends on repeats)
```

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

This is mathematically equivalent to `abs(src[i])`. For integer inputs, overflow on the most-negative signed value (e.g., `-128` for `i8`) follows target-defined behavior.

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Next op in instruction set: [pto.vneg](./vneg.md)
