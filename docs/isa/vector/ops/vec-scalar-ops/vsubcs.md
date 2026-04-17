# pto.vsubcs

`pto.vsubcs` is part of the [Vector-Scalar Instructions](../../vec-scalar-ops.md) instruction set.

## Summary

Lane-wise subtract with explicit borrow-in and borrow-out masks.

## Mechanism

For each active lane `i`, `diff = lhs[i] - rhs[i] - borrow_in[i]`, `result[i] = low_bits(diff)`, and `borrow[i] = borrow_out(diff)`. The borrow chain is lane-local in the PTO surface: each lane consumes one incoming borrow bit and produces one outgoing borrow bit.

## Syntax

### PTO Assembly Form

```text
vsubcs %dst, %borrow_out, %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %lhs | `!pto.vreg<NxT>` | Minuend vector |
| %rhs | `!pto.vreg<NxT>` | Subtrahend vector |
| %borrow_in | `!pto.mask` | Incoming borrow bit per lane |
| %mask | `!pto.mask` | Predicate mask; only lanes with mask bit 1 participate |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Lane-wise arithmetic result on the active lanes |
| %borrow | `!pto.mask` | Borrow-out bit produced for each active lane |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- Borrow-chain forms are defined for integer element types.
- `%lhs`, `%rhs`, and `%result` MUST have the same vector width `N` and element type `T`.
- `%borrow_in`, `%borrow`, and `%mask` MUST all have width `N`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Treat this form as an unsigned integer borrow-chain unless a target profile documents a wider legal domain.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.

## Examples

```c
for (int i = 0; i < N; i++) {
    if (!mask[i]) continue;
    uint64_t rhs_total = (uint64_t)rhs[i] + borrow_in[i];
    result[i] = lhs[i] - rhs_total;
    borrow[i] = lhs[i] < rhs_total;
}
```

```mlir
%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask, !pto.mask -> !pto.vreg<64xi32>, !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector-Scalar Instructions](../../vec-scalar-ops.md)
- Previous op in instruction set: [pto.vaddcs](./vaddcs.md)
- Next op in instruction set: (none)
