# pto.pand

`pto.pand` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Bitwise AND of two predicates.

## Mechanism

`pto.pand` computes the bitwise AND of two predicate registers, producing a new predicate where lane `i` is active iff both source lanes `i` are active.

$$ \mathrm{dst}_i = \mathrm{src0}_i \land \mathrm{src1}_i $$

The third operand (`%mask`) in the syntax is an optional masking predicate for the scalar and control instructions; the core boolean operation is `src0 AND src1`.

## Syntax

### PTO Assembly Form

```mlir
%dst = pto.pand %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.pand %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pand ins(%src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool dst;
vector_bool src0;
vector_bool src1;
vector_bool mask;
pand(dst, src0, src1, mask);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask` | First source predicate |
| `%src1` | `!pto.mask` | Second source predicate |
| `%mask` | `!pto.mask` | Optional masking predicate (scalar and control instructions context) |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.mask` | Bitwise AND of src0 and src1 |

## Side Effects

None.

## Constraints

- **Operand widths**: All predicate operands MUST have the same width. Mixing predicates of different widths without explicit pack/unpack is **illegal**.
- **No implicit masking**: The `mask` operand is for scalar and control instructions use; it does not affect the boolean AND operation itself.

## Exceptions

- Illegal if predicate operand widths are not consistent.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Bitwise AND | Simulated | Supported | Supported |

## Examples

### Combine comparison mask with tail mask

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void combine_masks(RegBuf<predicate_t>& dst,
                   const RegBuf<predicate_t>& cmp_mask,
                   const RegBuf<predicate_t>& tail_mask) {
    PAND(dst, cmp_mask, tail_mask, cmp_mask);
}
```

### SSA form — intersection of two predicates

```mlir
// %cmp_mask: lanes where a[i] < b[i]
%cmp = pto.vcmp %va, %vb, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// %tail_mask: lanes in the remainder region
%tail = pto.pge_b32 %rem : i32 -> !pto.mask

// Intersection: only process remainder lanes where comparison is true
%active = pto.pand %cmp, %tail, %cmp : !pto.mask, !pto.mask, !pto.mask -> !pto.mask

// Use in predicated operation
%result = pto.vsel %v_true, %v_false, %active : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in instruction set: [pto.punpack](./punpack.md)
- Next op in instruction set: [pto.por](./por.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
