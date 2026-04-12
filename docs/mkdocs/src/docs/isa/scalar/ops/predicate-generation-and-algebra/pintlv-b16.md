<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pintlv-b16.md` -->

# pto.pintlv_b16

Standalone reference page for `pto.pintlv_b16`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Predicate interleave: merge two 16-bit predicate registers into one 32-bit predicate register by alternating bits.

## Mechanism

`pto.pintlv_b16` interleaves two 16-bit predicate registers into one 32-bit predicate by alternating bits from each source. This is the inverse of `pdintlv_b16` (which splits a 32-bit predicate into two 16-bit halves), but `pintlv_b16` specifically operates on 16-bit inputs.

For two 16-bit predicates `src0`, `src1` and 0 ≤ i < 16:

$$ \mathrm{dst}_i = \mathrm{src0}_i $$
$$ \mathrm{dst}_{i+16} = \mathrm{src1}_i $$

This operation concatenates two 16-bit predicates into a 32-bit predicate register, preserving the lane-to-bit correspondence.

## Syntax

### PTO Assembly Form

```text
pintlv_b16 %dst, %src0, %src1 : !pto.mask, !pto.mask, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.pintlv_b16 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pintlv_b16 ins(%src0, %src1 : !pto.mask, !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PINTLV_B16(RegBuf<predicate_t>& dst,
                          const RegBuf<predicate_t>& src0,
                          const RegBuf<predicate_t>& src1);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask` | Lower 16-bit source predicate |
| `%src1` | `!pto.mask` | Upper 16-bit source predicate |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.mask` | 32-bit concatenated predicate |

## Side Effects

None.

## Constraints

- **Source width**: Both source predicates MUST be 16 bits. Other widths are **illegal**.
- **Destination width**: The destination predicate is 32 bits.
- **Bit mapping**: `dst[0..15] = src0[0..15]`, `dst[16..31] = src1[0..15]`.

## Exceptions

- Illegal if source predicate widths are not 16 bits.
- Illegal if destination context does not expect a 32-bit predicate.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Predicate interleave (b16) | Simulated | Supported | Supported |

## Examples

### Concatenate two 16-bit predicates

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void concat_predicates(RegBuf<predicate_t>& dst,
                      const RegBuf<predicate_t>& lo,
                      const RegBuf<predicate_t>& hi) {
    PINTLV_B16(dst, lo, hi);
}
```

### SSA form — combine two halves for full 32-bit predicate

```mlir
// %cmp_lo: comparison result for lanes 0-15
// %cmp_hi: comparison result for lanes 16-31

// Combine into full 32-bit predicate
%full = pto.pintlv_b16 %cmp_lo, %cmp_hi : !pto.mask, !pto.mask -> !pto.mask

// Use for 32-lane predicated vector operation
%result = pto.vsel %v_true, %v_false, %full : !pto.vreg<32xf32>, !pto.vreg<32xf32>, !pto.mask -> !pto.vreg<32xf32>
```

### Round-trip deinterleave then interleave

```mlir
// %src32: 32-bit predicate

// Split into two 16-bit halves
%lo16, %hi16 = pto.pdintlv_b32 %src32 : !pto.mask -> !pto.mask, !pto.mask

// Modify %lo16 or %hi16 independently
%lo16_mod = pto.pnot %lo16, %lo16 : !pto.mask, !pto.mask -> !pto.mask

// Re-concatenate
%dst = pto.pintlv_b16 %lo16_mod, %hi16 : !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pdintlv_b8](./pdintlv-b8.md)
- Next op in family: (none — last in family)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
