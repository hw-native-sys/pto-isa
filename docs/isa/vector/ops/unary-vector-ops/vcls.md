# pto.vcls

`pto.vcls` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` holds the leading-sign-bit count per active lane.

## Mechanism

`pto.vcls` computes the lane-wise count of leading sign bits: `dst[i] = count_leading_sign_bits(src[i])`. This counts the number of identical bits (sign bit) starting from the most significant bit until the first opposite bit is found. Integer element types only. Inactive lanes leave the destination unchanged.

## Syntax

### PTO Assembly Form

```text
vcls %result, %input, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcls %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

Documented A5 types or forms: `all integer types`.

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%input` | `!pto.vreg<NxT>` | Source vector register; read at each active lane `i` |
| `%mask` | `!pto.mask<G>` | Predicate mask; lanes where mask bit is 1 (true) are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise leading-sign-bit count: `dst[i] = count_leading_sign_bits(src[i])` on active lanes; inactive lanes are unmodified |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

!!! warning "Constraints"
    Integer element types only. This operation is
      sign-aware, so signed interpretation matters.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `all integer types`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = count_leading_sign_bits(src[i]);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vbcnt](./vbcnt.md)
- Next op in instruction set: [pto.vmov](./vmov.md)
