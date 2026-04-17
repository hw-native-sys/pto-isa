# pto.vshls

`pto.vshls` is part of the [Vector-Scalar Instructions](../../vec-scalar-ops.md) instruction set.

## Summary

Lane-wise logical left shift by a broadcast scalar amount.

## Mechanism

For each active lane `i`, `dst[i] = src[i] << scalar`. The scalar shift count is applied uniformly to every active lane. Inactive lanes do not participate in the operation.

## Syntax

### PTO Assembly Form

```text
vshls %dst, %src, %shift, %mask : !pto.vreg<NxT>, T
```

### AS Level 1 (SSA)

```mlir
%result = pto.vshls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %input | `!pto.vreg<NxT>` | Source vector register |
| %scalar | `T` | Uniform shift amount broadcast to every active lane |
| %mask | `!pto.mask` | Predicate mask; only lanes with mask bit 1 participate |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Lane-wise left-shift result on the active lanes |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- Shift forms are defined for integer element types only.
- Portable code SHOULD keep the shift count below the element width of `T`.
- `%input` and `%result` MUST have the same vector width `N` and element type `T`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Integer element types only.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.

## Examples

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = src[i] << scalar;
```

```mlir
%result = pto.vshls %values, %shift, %mask : !pto.vreg<64xi32>, i32, !pto.mask -> !pto.vreg<64xi32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector-Scalar Instructions](../../vec-scalar-ops.md)
- Previous op in instruction set: [pto.vxors](./vxors.md)
- Next op in instruction set: [pto.vshrs](./vshrs.md)
