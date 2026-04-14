# pto.por

`pto.por` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Bitwise OR of two predicates.

## Mechanism

`pto.por` computes the bitwise OR of two predicate registers, producing a new predicate where lane `i` is active iff at least one source lane `i` is active.

$$ \mathrm{dst}_i = \mathrm{src0}_i \lor \mathrm{src1}_i $$

## Syntax

### PTO Assembly Form

```mlir
%dst = pto.por %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.por %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.por ins(%src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool dst;
vector_bool src0;
vector_bool src1;
vector_bool mask;
por(dst, src0, src1, mask);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask` | First source predicate |
| `%src1` | `!pto.mask` | Second source predicate |
| `%mask` | `!pto.mask` | Optional masking predicate |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.mask` | Bitwise OR of src0 and src1 |

## Side Effects

None.

## Constraints

- **Operand widths**: All predicate operands MUST have the same width.

## Exceptions

- Illegal if predicate operand widths are not consistent.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Bitwise OR | Simulated | Supported | Supported |

## Examples

### Combine predicates from two conditions

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void union_masks(RegBuf<predicate_t>& dst,
                 const RegBuf<predicate_t>& mask_a,
                 const RegBuf<predicate_t>& mask_b) {
    POR(dst, mask_a, mask_b, mask_a);
}
```

### SSA form — union of two predicates

```mlir
// %mask_a: lanes where a[i] < threshold_a
// %mask_b: lanes where b[i] > threshold_b

// Union: lanes satisfying either condition
%combined = pto.por %mask_a, %mask_b, %mask_a : !pto.mask, !pto.mask, !pto.mask -> !pto.mask

// Reconstruct full-width predicate from two halves
%lo_combined = pto.por %mask_a_lo, %mask_b_lo, %mask_a_lo : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
%hi_combined = pto.por %mask_a_hi, %mask_b_hi, %mask_a_hi : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in instruction set: [pto.pand](./pand.md)
- Next op in instruction set: [pto.pxor](./pxor.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
