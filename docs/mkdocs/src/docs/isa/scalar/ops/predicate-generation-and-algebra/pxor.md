<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pxor.md` -->

# pto.pxor

Standalone reference page for `pto.pxor`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Bitwise XOR of two predicates.

## Mechanism

`pto.pxor` computes the bitwise XOR of two predicate registers, producing a new predicate where lane `i` is active iff exactly one of the source lanes `i` is active (but not both).

$$ \mathrm{dst}_i = \mathrm{src0}_i \oplus \mathrm{src1}_i $$

XOR is commonly used to invert one predicate within a mask context: `pxor %p, %inv, %mask` produces `mask XOR inv`, effectively inverting `inv` where `mask` is 1.

## Syntax

### PTO Assembly Form

```text
pxor %dst, %src0, %src1 : !pto.mask, !pto.mask, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.pxor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pxor ins(%src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PXOR(RegBuf<predicate_t>& dst,
                   const RegBuf<predicate_t>& src0,
                   const RegBuf<predicate_t>& src1,
                   const RegBuf<predicate_t>& mask);
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
| `%dst` | `!pto.mask` | Bitwise XOR of src0 and src1 |

## Side Effects

None.

## Constraints

- **Operand widths**: All predicate operands MUST have the same width.

## Exceptions

- Illegal if predicate operand widths are not consistent.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Bitwise XOR | Simulated | Supported | Supported |

## Examples

### Conditional inversion via XOR

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void invert_with_mask(RegBuf<predicate_t>& dst,
                      const RegBuf<predicate_t>& to_invert,
                      const RegBuf<predicate_t>& mask) {
    // dst = mask XOR to_invert (inverts to_invert where mask is 1)
    PXOR(dst, mask, to_invert, mask);
}
```

### SSA form — XOR for predicate difference

```mlir
// %mask_a: lanes active in set A
// %mask_b: lanes active in set B

// Symmetric difference: lanes active in exactly one set
%diff = pto.pxor %mask_a, %mask_b, %mask_a : !pto.mask, !pto.mask, !pto.mask -> !pto.mask

// Intersection: lanes active in both sets (via De Morgan: A AND B = NOT(A XOR B))
%inv = pto.pnot %diff, %diff : !pto.mask, !pto.mask -> !pto.mask
%intersection = pto.pand %mask_a, %mask_b, %mask_a : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.por](./por.md)
- Next op in family: [pto.pnot](./pnot.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
