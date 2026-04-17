# pto.vsubc

`pto.vsubc` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

Lane-wise integer subtraction producing both a result vector and a borrow predicate mask vector.

## Mechanism

Computes lane-wise integer subtraction of two source vectors and produces two outputs: the result and a per-lane borrow/underflow predicate.

For each lane `i` where the predicate is true:

$$ \mathrm{dst}_i = \mathrm{lhs}_i - \mathrm{rhs}_i $$

$$ \mathrm{borrow}_i = (\mathrm{lhs}_i < \mathrm{rhs}_i) $$

On the current A5 instruction set, this should be treated as an unsigned 32-bit borrow-chain operation. The borrow output can be chained with another `vsubc` to implement multi-element arbitrary-precision subtraction.

Inactive lanes leave the destination and borrow registers unchanged.

## Syntax

### PTO Assembly Form

```text
vsubc %dst, %borrow, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result, %borrow = pto.vsubc %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>, !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.vsubc ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
            outs(%result, %borrow : !pto.vreg<NxT>, !pto.mask)
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | Minuend: the value being subtracted from |
| `%rhs` | `!pto.vreg<NxT>` | Subtrahend: the value being subtracted |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise arithmetic difference on active lanes; inactive lanes are unmodified |
| `%borrow` | `!pto.mask` | Per-lane borrow predicate: lane `i` is 1 if unsigned underflow occurred in lane `i` |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register and borrow predicate. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type**: Integer element types only. This is a borrow-chain integer subtraction instruction set.
- **Signedness**: On A5, treat as unsigned 32-bit borrow-chain operation unless and until the verifier states otherwise.
- **Type match**: `%lhs`, `%rhs`, and `%result` MUST have identical element types.
- **Width match**: All registers MUST have the same vector width `N`.
- **Mask width**: `%mask` MUST have width equal to `N`.
- **Active lanes**: Only lanes where the mask bit is 1 participate.
- **Inactive lanes**: Destination and borrow elements at inactive lanes are unmodified.

## Exceptions

- The verifier rejects non-integer element types, type mismatches, width mismatches, or mask width mismatches.
- Any additional illegality stated in the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set page is also part of the contract.

## Target-Profile Restrictions

A5 is the primary concrete profile for the vector instructions. CPU simulation and A2/A3-class targets emulate this operation while preserving the visible PTO contract.

## Performance

### A5 Latency

|| Element Type | Latency (cycles) | A5 RV |
||---|---|---|
|| `i32` | 7 | `RV_VSUBC` |

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

### MLIR Usage

```mlir
// Single-element subtraction with borrow
%result, %borrow = pto.vsubc %a, %b, %active : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>, !pto.mask

// Multi-word subtraction: chain borrows into next segment
%diff0, %borrow0 = pto.vsubc %a0, %b0, %active : ...  // low words
%diff1, %borrow1 = pto.vsubc %a1, %b1, %borrow0 : ...  // high words (borrow from low)
```

### Typical Usage: Multi-Word Integer Subtraction

```mlir
// Subtract two 128-bit integers represented as two 64-element i32 vectors:
//   A = [a_low, a_high], B = [b_low, b_high]
//   result = A - B
%diff_low, %borrow = pto.vsubc %a_low, %b_low, %active : ...
%diff_high, %borrow2 = pto.vsubc %a_high, %b_high, %borrow : ...
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vaddc](./vaddc.md)
