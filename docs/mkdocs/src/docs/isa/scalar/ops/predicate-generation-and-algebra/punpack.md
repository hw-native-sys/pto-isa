<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/punpack.md` -->

# pto.punpack

Standalone reference page for `pto.punpack`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Widening unpack: extract one N-bit segment from a 2N-bit predicate register, zero-filling the non-selected half of the source.

## Mechanism

`pto.punpack` takes a 2N-bit predicate register and a partition token, and produces an N-bit predicate by selecting one half and zero-filling the other. It is the inverse of `ppack`.

For source predicate `src` with 2N bits and partition token `P`:

$$ \mathrm{dst}_N = \begin{cases} \mathrm{LOWER}(\mathrm{src}_{2N}) & \text{if } P = \text{LOWER} \\ \mathrm{UPPER}(\mathrm{src}_{2N}) & \text{if } P = \text{HIGHER} \end{cases} $$

## Syntax

### PTO Assembly Form

```text
punpack %dst, %src, "PART" : !pto.mask, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.punpack %src, "PART" : !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.punpack ins(%src, "PART" : !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PUNPACK(RegBuf<predicate_t>& dst,
                      const RegBuf<predicate_t>& src,
                      const char* partition);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src` | `!pto.mask` | Source 2N-bit predicate |
| `"PART"` | string attribute | Partition token: `"LOWER"` or `"HIGHER"` |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.mask` | N-bit predicate extracted from the selected half |

## Side Effects

None.

## Constraints

- **Partition token**: MUST be `"LOWER"` or `"HIGHER"`. Other tokens are **illegal**.
- **Source width**: The source predicate MUST be 2N bits. Programs MUST ensure the source context provides a 2N-bit predicate.
- **Destination width**: The destination predicate is always N bits. Programs that need a 2N-bit result after extraction MUST use `ppack` to reconstruct it.
- **Zero-fill behavior**: The non-selected half of the source is ignored (zero-filled); the destination does NOT contain a concatenation or merge of both halves.

## Exceptions

- Illegal if the partition token is not `"LOWER"` or `"HIGHER"`.
- Illegal if source and destination predicate widths are not in a 2:1 ratio.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Unpack operation | Simulated | Supported | Supported |
| `LOWER` / `HIGHER` tokens | Supported | Supported | Supported |

## Examples

### Extract upper half of a 64-bit predicate

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void extract_upper(RegBuf<predicate_t>& dst,
                   const RegBuf<predicate_t>& src_64) {
    PUNPACK(dst, src_64, "HIGHER");
}
```

### Extract and re-pack with modification

```mlir
// %full_64: 64-bit predicate from a comparison

// Extract lower half
%lo = pto.punpack %full_64, "LOWER" : !pto.mask -> !pto.mask

// Extract upper half
%hi = pto.punpack %full_64, "HIGHER" : !pto.mask -> !pto.mask

// Modify lower half (e.g., invert)
%lo_inv = pto.pnot %lo, %lo : !pto.mask, !pto.mask -> !pto.mask

// Re-pack into 64-bit predicate
%new_lo = pto.ppack %lo_inv, "LOWER" : !pto.mask -> !pto.mask
%new_hi = pto.ppack %hi, "HIGHER" : !pto.mask -> !pto.mask
%new_full = pto.por %new_lo, %new_hi, %new_lo : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.ppack](./ppack.md)
- Next op in family: [pto.pand](./pand.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
