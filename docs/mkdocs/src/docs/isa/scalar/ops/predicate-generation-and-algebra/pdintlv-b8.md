<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pdintlv-b8.md` -->

# pto.pdintlv_b8

Standalone reference page for `pto.pdintlv_b8`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Predicate deinterleave: split one 16-bit predicate register into two 8-bit predicate registers by separating alternating bits.

## Mechanism

`pto.pdintlv_b8` deinterleaves a 16-bit predicate register into two 8-bit predicates by distributing alternating bits. Lane `i` from the lower half goes to the first output; lane `i` from the upper half goes to the second output.

For a 16-bit predicate `src` and 0 ≤ i < 8:

$$ \mathrm{dst0}_i = \mathrm{src}_i $$
$$ \mathrm{dst1}_i = \mathrm{src}_{i+8} $$

This operation is used when processing 16-bit-wide data with two independent 8-bit predicate contexts, or when separating even/odd lane groups for multi-step processing.

## Syntax

### PTO Assembly Form

```text
pdintlv_b8 %dst0, %dst1, %src : !pto.mask, !pto.mask, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst0, %dst1 = pto.pdintlv_b8 %src : !pto.mask -> !pto.mask, !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pdintlv_b8 ins(%src : !pto.mask) outs(%dst0, %dst1 : !pto.mask, !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PDINTLV_B8(RegBuf<predicate_t>& dst0,
                         RegBuf<predicate_t>& dst1,
                         const RegBuf<predicate_t>& src);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src` | `!pto.mask` | 16-bit source predicate register |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst0` | `!pto.mask` | Lower 8 bits: `src[0..7]` |
| `%dst1` | `!pto.mask` | Upper 8 bits: `src[8..15]` |

## Side Effects

None.

## Constraints

- **Source width**: The source predicate MUST be 16 bits. Sources of other widths are **illegal**.
- **Destination width**: Both destination predicates are 8 bits.
- **Relationship**: `pdintlv_b8` is the inverse of `pintlv_b16`.

## Exceptions

- Illegal if the source predicate width is not 16 bits.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Predicate deinterleave (b8) | Simulated | Supported | Supported |

## Examples

### Separate two 8-bit predicate contexts

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void split_predicate(RegBuf<predicate_t>& dst0,
                     RegBuf<predicate_t>& dst1,
                     const RegBuf<predicate_t>& src16) {
    PDINTLV_B8(dst0, dst1, src16);
}
```

### SSA form — two-phase predicate processing

```mlir
// %src16: 16-bit predicate from some comparison

// Phase 1: process lower 8 lanes
%lo, %hi = pto.pdintlv_b8 %src16 : !pto.mask -> !pto.mask, !pto.mask

// Use %lo for first phase of vector computation
%result_lo = pto.vsel %v_a_lo, %v_b_lo, %lo : !pto.vreg<8xf32>, !pto.vreg<8xf32>, !pto.mask -> !pto.vreg<8xf32>

// Use %hi for second phase of vector computation
%result_hi = pto.vsel %v_a_hi, %v_b_hi, %hi : !pto.vreg<8xf32>, !pto.vreg<8xf32>, !pto.mask -> !pto.vreg<8xf32>
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.psel](./psel.md)
- Next op in family: [pto.pintlv_b16](./pintlv-b16.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
