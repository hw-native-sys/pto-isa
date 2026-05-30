# pto.vln

`pto.vln` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` holds the natural logarithm per active lane.

## Mechanism

`pto.vln` computes the lane-wise natural logarithm: `dst[i] = ln(src[i])`. For real-number semantics, active inputs SHOULD be strictly positive; non-positive inputs follow the target's exception/NaN rules. Inactive lanes leave the destination unchanged.

## Syntax

### PTO Assembly Form

```text
vln %result, %input, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
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
| `%result` | `!pto.vreg<NxT>` | Lane-wise natural logarithm: `dst[i] = ln(src[i])` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

!!! warning "Constraints"
    Only floating-point element types are legal.

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
| `f32` | 18 | `RV_VLN` |
| `f16` | 23 | `RV_VLN` |

### A2/A3 Throughput

| Metric | Value (f32) | Value (f16) |
|--------|-------------|-------------|
| Startup latency | 13 (`A2A3_STARTUP_REDUCE`) | 13 |
| Completion latency | 26 (f32) / 28 (f16) | `A2A3_COMPL_FP32_EXP` / `A2A3_COMPL_FP16_EXP` |
| Per-repeat throughput | 2 | 4 |
| Pipeline interval | 18 | 18 |

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

**Input domain:** `logf(x)` is only defined for `x > 0`. Non-positive active inputs produce NaN; inactive lanes leave the destination unchanged.

### Numerical stability note

For softmax denominator (`log(sum(exp(x - max)))`), use `vexpdif` fused operation rather than separate `vsub` + `vln` for better combined throughput.

---

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vexp](./vexp.md)
- Next op in instruction set: [pto.vsqrt](./vsqrt.md)
