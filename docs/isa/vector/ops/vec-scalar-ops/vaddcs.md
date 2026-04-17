# pto.vaddcs

`pto.vaddcs` is part of the [Vector-Scalar Instructions](../../vec-scalar-ops.md) instruction set.

## Summary

Lane-wise add with explicit carry-in and carry-out masks.

## Mechanism

For each active lane `i`, `sum = lhs[i] + rhs[i] + carry_in[i]`, `result[i] = low_bits(sum)`, and `carry[i] = carry_out(sum)`. The carry chain is lane-local in the PTO surface: each lane consumes one incoming carry bit and produces one outgoing carry bit.

## Syntax

### PTO Assembly Form

```text
vaddcs %dst, %carry_out, %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %lhs | `!pto.vreg<NxT>` | Left-hand value vector |
| %rhs | `!pto.vreg<NxT>` | Right-hand value vector |
| %carry_in | `!pto.mask` | Incoming carry bit per lane |
| %mask | `!pto.mask` | Predicate mask; only lanes with mask bit 1 participate |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Lane-wise arithmetic result on the active lanes |
| %carry | `!pto.mask` | Carry-out bit produced for each active lane |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- Carry-chain forms are defined for integer element types.
- `%lhs`, `%rhs`, and `%result` MUST have the same vector width `N` and element type `T`.
- `%carry_in`, `%carry`, and `%mask` MUST all have width `N`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Treat this form as an unsigned integer carry-chain unless a target profile documents a wider legal domain.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.

## Examples

```c
for (int i = 0; i < N; i++) {
    if (!mask[i]) continue;
    uint64_t sum = (uint64_t)lhs[i] + rhs[i] + carry_in[i];
    result[i] = (T)sum;
    carry[i] = (sum >> BITWIDTH_T) & 0x1;
}
```

```mlir
%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask, !pto.mask -> !pto.vreg<64xi32>, !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector-Scalar Instructions](../../vec-scalar-ops.md)
- Previous op in instruction set: [pto.vlrelu](./vlrelu.md)
- Next op in instruction set: [pto.vsubcs](./vsubcs.md)
