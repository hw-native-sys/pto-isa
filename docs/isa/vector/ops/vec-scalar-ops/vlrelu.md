# pto.vlrelu

`pto.vlrelu` is part of the [Vector-Scalar Instructions](../../vec-scalar-ops.md) instruction set.

## Summary

Lane-wise leaky ReLU with a broadcast slope scalar.

## Mechanism

For each active lane `i`, `dst[i] = (src[i] >= 0) ? src[i] : scalar * src[i]`. The scalar operand supplies the negative-path slope for every active lane. Inactive lanes do not participate in the computation.

## Syntax

### PTO Assembly Form

```text
vlrelu %dst, %src, %slope, %mask : !pto.vreg<NxT>, T
```

### AS Level 1 (SSA)

```mlir
%result = pto.vlrelu %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %input | `!pto.vreg<NxT>` | Source activation vector |
| %scalar | `T` | Negative-path slope broadcast to every active lane |
| %mask | `!pto.mask` | Predicate mask; only lanes with mask bit 1 participate |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Lane-wise leaky-ReLU result on the active lanes |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- The current manual documents floating-point forms for `f16` and `f32`.
- `%input` and `%result` MUST have the same vector width `N` and element type `T`.
- The mask width MUST match `N`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented floating-point forms: `f16`, `f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.

## Examples

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = (src[i] >= 0) ? src[i] : scalar * src[i];
```

```mlir
%result = pto.vlrelu %activations, %alpha, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector-Scalar Instructions](../../vec-scalar-ops.md)
- Previous op in instruction set: [pto.vshrs](./vshrs.md)
- Next op in instruction set: [pto.vaddcs](./vaddcs.md)
