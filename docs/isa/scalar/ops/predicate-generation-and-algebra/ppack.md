# pto.ppack

`pto.ppack` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Narrowing pack: concatenate two N-bit predicate segments into one 2N-bit predicate register, selecting one segment by a partition token.

## Mechanism

`pto.ppack` takes a source predicate register and a partition token, and writes a 2N-bit predicate register by filling the selected half with the source bits and zero-filling the other half. It is the inverse of `punpack`.

For source predicate `src` with N bits and partition token `P`:

$$ \mathrm{dst}_{2N} = \begin{cases} \mathrm{ZERO}(N) \Vert \mathrm{src}_N & \text{if } P = \text{LOWER} \\ \mathrm{src}_N \Vert \mathrm{ZERO}(N) & \text{if } P = \text{HIGHER} \end{cases} $$

## Syntax

### PTO Assembly Form

```mlir
%dst = pto.ppack %src, "PART" : !pto.mask -> !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.ppack %src, "PART" : !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.ppack ins(%src, "PART" : !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool dst;
vector_bool src;
ppack(dst, src, __cce_simd::LOWER);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src` | `!pto.mask` | Source N-bit predicate |
| `"PART"` | string attribute | Partition token: `"LOWER"` or `"HIGHER"` |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.mask` | 2N-bit predicate with the source in the selected half |

## Side Effects

None.

## Constraints

- **Partition token**: MUST be `"LOWER"` or `"HIGHER"`. Other tokens are **illegal**.
- **Destination width**: The destination predicate is always 2N bits. Programs MUST ensure the destination context expects a 2N-bit predicate. Attempting to use a 2N-bit result in an N-bit context without explicit extraction via `punpack` is **illegal**.
- **Source width**: The source predicate MUST be N bits (half the destination width). Mismatched widths are **illegal**.
- **Zero-fill behavior**: The non-selected half of the destination is always zero-filled, not sign-extended or replicated.

## Exceptions

- Illegal if the partition token is not `"LOWER"` or `"HIGHER"`.
- Illegal if source and destination predicate widths are not in a 1:2 ratio.
- Illegal if the operation is used in a context that does not expect a 2N-bit result.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Pack operation | Simulated | Supported | Supported |
| `LOWER` / `HIGHER` tokens | Supported | Supported | Supported |

## Examples

### Combine two b32 predicates for f32 (64 lanes)

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void pack_for_f32(RegBuf<predicate_t>& dst,
                  const RegBuf<predicate_t>& lo,
                  const RegBuf<predicate_t>& hi) {
    // dst = [ZERO(32) | lo] = hi concatenated with zero
    PPACK(dst, lo, "LOWER");
}
```

### SSA form

```mlir
// %rem = 47
// %lo: lanes 0-31 active (from plt_b32 iteration 1)
// %hi: lanes 0-14 active (from plt_b32 iteration 2, rem = 15)

// Pack %lo into lower half of 64-bit predicate
%full_lo = pto.ppack %lo, "LOWER" : !pto.mask -> !pto.mask

// Pack %hi into upper half of 64-bit predicate
%full_hi = pto.ppack %hi, "HIGHER" : !pto.mask -> !pto.mask

// OR them together to get full 64-lane tail mask
%tail = pto.por %full_lo, %full_hi, %full_lo : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

### Construct a full-width mask from two half-width masks

```mlir
// Pack lower half
%dst_lower = pto.ppack %src_lower, "LOWER" : !pto.mask -> !pto.mask

// Pack upper half
%dst_upper = pto.ppack %src_upper, "HIGHER" : !pto.mask -> !pto.mask

// Combine with OR to get full-width predicate
%combined = pto.por %dst_lower, %dst_upper, %dst_lower : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in instruction set: [pto.plt_b32](./plt-b32.md)
- Next op in instruction set: [pto.punpack](./punpack.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
