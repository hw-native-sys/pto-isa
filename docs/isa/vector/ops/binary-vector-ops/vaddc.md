# pto.vaddc

`pto.vaddc` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

Lane-wise integer addition producing both a result vector and a carry/overflow predicate mask vector.

## Mechanism

Computes lane-wise integer addition of two source vectors and produces two outputs: the truncated result and a per-lane carry/overflow predicate.

For each lane `i` where the predicate is true:

$$ \mathrm{dst}_i = \mathrm{lhs}_i + \mathrm{rhs}_i $$

$$ \mathrm{carry}_i = \text{carry out of lane } i \text{ (unsigned arithmetic) } $$

On the current A5 instruction set, this should be treated as an unsigned integer carry-chain operation. The carry output can be chained with another `vaddc` to implement multi-element arbitrary-precision addition.

Inactive lanes leave the destination and carry registers unchanged.

## Syntax

### PTO Assembly Form

```text
vaddc %dst, %carry, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result, %carry = pto.vaddc %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>, !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.vaddc ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
            outs(%result, %carry : !pto.vreg<NxT>, !pto.mask)
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%lhs` | `!pto.vreg<NxT>` | Minuend: the first addend |
| `%rhs` | `!pto.vreg<NxT>` | Subtrahend: the second addend |
| `%mask` | `!pto.mask` | Predicate mask; lanes where mask bit is 1 are active |

Both source registers MUST have the same element type and the same vector width `N`. The mask width MUST match `N`.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Lane-wise truncated sum on active lanes; inactive lanes are unmodified |
| `%carry` | `!pto.mask` | Per-lane carry/overflow predicate: lane `i` is 1 if unsigned overflow occurred in lane `i` |

## Side Effects

This operation has no architectural side effect beyond producing its destination vector register and carry predicate. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- **Type**: Integer element types only. This is a carry-chain integer addition instruction set.
- **Signedness**: On A5, treat as unsigned integer operation.
- **Type match**: `%lhs`, `%rhs`, and `%result` MUST have identical element types.
- **Width match**: All registers MUST have the same vector width `N`.
- **Mask width**: `%mask` MUST have width equal to `N`.
- **Active lanes**: Only lanes where the mask bit is 1 participate.
- **Inactive lanes**: Destination and carry elements at inactive lanes are unmodified.

## Exceptions

- The verifier rejects non-integer element types, type mismatches, width mismatches, or mask width mismatches.
- Any additional illegality stated in the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set page is also part of the contract.

## Target-Profile Restrictions

A5 is the primary concrete profile for the vector instructions. CPU simulation and A2/A3-class targets emulate this operation while preserving the visible PTO contract.

## Performance

### A5 Latency

|| Element Type | Latency (cycles) | A5 RV |
||---|---|---|
|| `i32` | 7 | `RV_VADDC` |

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

### MLIR Usage

```mlir
// Single-element addition with carry
%result, %carry = pto.vaddc %a, %b, %active : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>, !pto.mask

// Multi-word addition: chain carries into next segment
%sum0, %carry0 = pto.vaddc %a0, %b0, %active : ...  // low words
%sum1, %carry1 = pto.vaddc %a1, %b1, %carry0 : ...  // high words (carry from low)
```

### Typical Usage: Multi-Word Integer Addition

```mlir
// Add two 128-bit integers represented as two 64-element i32 vectors:
//   A = [a_low, a_high], B = [b_low, b_high]
//   result = A + B
%sum_low, %carry = pto.vaddc %a_low, %b_low, %active : ...
%sum_high, %borrow = pto.vaddc %a_high, %b_high, %carry : ...
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vshr](./vshr.md)
- Next op in instruction set: [pto.vsubc](./vsubc.md)
