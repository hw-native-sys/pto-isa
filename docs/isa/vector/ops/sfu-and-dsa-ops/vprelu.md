# pto.vprelu

`pto.vprelu` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Parametric ReLU with per-element alpha vector.

## Mechanism

Computes parametric ReLU: `dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i]`. For each lane `i`, if the input is non-negative, pass it through; otherwise scale by the per-lane alpha coefficient.

## Syntax

### PTO Assembly Form

```text
vprelu %dst, %src, %alpha, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vprelu %input, %alpha, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

Documented A5 types: `f16, f32`.

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `%input` | `!pto.vreg<NxT>` | Activation input vector |
|| `%alpha` | `!pto.vreg<NxT>` | Per-element slope (alpha) vector |
|| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 are active |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `%result` | `!pto.vreg<NxT>` | Parametric ReLU output: `dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i]` |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

!!! warning "Constraints"
    - **Type match**: `%input`, `%alpha`, and `%result` MUST have identical element types.
    - **Width match**: All three registers MUST have the same vector width `N`.
    - **Mask width**: `%mask` MUST have width equal to `N`.
    - **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the computation.
    - **Floating-point only**: Parametric ReLU is defined for floating-point element types on the current A5 instruction set.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `f16, f32`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

SFU operations have higher latency than standard arithmetic ops. Consult the target profile's performance model for cycle-accurate estimates.

### A2/A3 Throughput

|| Metric | Value | Constant |
||--------|-------|----------|
|| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
|| Completion latency | 26 | `A2A3_COMPL_FP32_EXP` |
|| Per-repeat throughput | 2 | `A2A3_RPT_2` |
|| Pipeline interval | 18 | `A2A3_INTERVAL` |

---

## Examples

### Parametric ReLU with per-lane alpha

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

### MLIR form

```mlir
%result = pto.vprelu %input, %alpha, %mask : (!pto.vreg<64xf16>, !pto.vreg<64xf16>, !pto.mask<b16>) -> !pto.vreg<64xf16>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Next op in instruction set: [pto.vexpdif](./vexpdif.md)
