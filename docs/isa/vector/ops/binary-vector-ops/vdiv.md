# pto.vdiv

`pto.vdiv` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

`%result` is the lane-wise quotient.

## Mechanism

`pto.vdiv` is a `pto.v*` compute operation.

## Syntax

### PTO Assembly Form

```text
vdiv %result, %lhs, %rhs, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32 only (no integer division)`.

## Inputs

`%lhs` is the numerator, `%rhs` is the denominator, and `%mask`
  selects active lanes.

## Expected Outputs

`%result` is the lane-wise quotient.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

!!! warning "Constraints"
    Floating-point element types only. Active
      denominators containing `+0` or `-0` follow the target's exceptional
      behavior.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Target-defined numeric exceptional behavior, such as divide-by-zero or out-of-domain inputs, remains subject to the selected backend profile unless this page narrows it further.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `f16, f32 only (no integer division)`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | A5 RV |
|---|---|---|
| `f32` | 17 | `RV_VDIV` |
| `f16` | 22 | `RV_VDIV` |

### A2/A3 Throughput

| Metric | Value (f32) | Value (f16) |
|--------|-------------|-------------|
| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
| Completion latency | 20 | `A2A3_COMPL_FP_MUL` |
| Per-repeat throughput | 2 | 4 |
| Pipeline interval | 18 | 18 |

**Performance note:** Division is significantly more expensive than multiplication (17-22 cycles vs 8 cycles). When accuracy permits, prefer multiplying by the reciprocal (`vmuls %result, %dividend, %divisor_rcp`).

---

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

---

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vmul](./vmul.md)
- Next op in instruction set: [pto.vmax](./vmax.md)
