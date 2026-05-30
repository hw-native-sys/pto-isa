# pto.vaddreluconv

`pto.vaddreluconv` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Fused add + ReLU + element-type conversion (hardware fusion).

## Mechanism

Fused add + ReLU + element-type conversion: `dst[i] = max(0, lhs[i] + rhs[i])` with optional type conversion (e.g., f16 → f32). Combines addition, ReLU, and type conversion in a single fused operation. This eliminates intermediate register writes and improves both throughput and numerical precision.

## Syntax

### PTO Assembly Form

```text
vaddreluconv %dst, %lhs, %rhs, %mask : !pto.vreg<NxT0>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vaddreluconv %lhs, %rhs, %mask : (!pto.vreg<NxT0>, !pto.vreg<NxT0>, !pto.mask<G>) -> !pto.vreg<MxT1>
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `%lhs` | `!pto.vreg<NxT0>` | Left-hand source vector |
|| `%rhs` | `!pto.vreg<NxT0>` | Right-hand source vector |
|| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 are active |

Both source vectors MUST have the same element type `T0` and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `%result` | `!pto.vreg<MxT1>` | Fused add/ReLU/convert result |

The destination element type `T1` may differ from the source type `T0`. Only backend-supported source/destination type pairs are legal.

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - **Type match**: Both source vectors MUST have identical element types.
    - **Width match**: Both source vectors MUST have the same vector width `N`.
    - **Mask width**: `%mask` MUST have width equal to `N`.
    - **Active lanes**: Only lanes where the mask bit is 1 (true) participate in the computation.
    - **Type conversion**: Only backend-supported source/destination type pairs are legal.
    - **Rounding/saturation**: Rounding, saturation, and packing rules follow the semantics of this fused operation, not an arbitrary sequence of standalone ops.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
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

### Fused add + ReLU with type conversion

```c
// f32→f16 variant:
for (int i = 0; i < 64; i++)
    dst_f16[i] = f32_to_f16(max(src0_f32[i] + src1_f32[i], 0));

// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(max(src0_f16[i] + src1_f16[i], 0));
```

### MLIR form

```mlir
// Widening: f16 → f32
%result = pto.vaddreluconv %lhs, %rhs, %mask : (!pto.vreg<64xf16>, !pto.vreg<64xf16>, !pto.mask<G>) -> !pto.vreg<64xf32>

// Narrowing: f32 → f16 with saturation
%result = pto.vaddreluconv %lhs, %rhs, %mask : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G>) -> !pto.vreg<64xf16>
```

### Common use cases

This operation is particularly useful for:
- **Activation functions**: Computing ReLU after addition is common in neural network layers
- **Mixed-precision inference**: Converting from high-precision accumulation (f32) to low-precision storage (f16/i8)
- **Quantization workflows**: Performing the add+ReLU step while downcasting to quantized types

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vaxpy](./vaxpy.md)
- Next op in instruction set: [pto.vmulconv](./vmulconv.md)
