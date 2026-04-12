<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pnot.md` -->

# pto.pnot

Standalone reference page for `pto.pnot`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Bitwise NOT of a predicate.

## Mechanism

`pto.pnot` computes the bitwise NOT of a predicate register, producing a new predicate where lane `i` is active iff the source lane `i` is inactive.

$$ \mathrm{dst}_i = \neg \mathrm{src}_i $$

## Syntax

### PTO Assembly Form

```text
pnot %dst, %src : !pto.mask, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.pnot %src, %mask : !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pnot ins(%src, %mask : !pto.mask, !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PNOT(RegBuf<predicate_t>& dst,
                   const RegBuf<predicate_t>& src,
                   const RegBuf<predicate_t>& mask);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src` | `!pto.mask` | Source predicate to invert |
| `%mask` | `!pto.mask` | Optional masking predicate |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.mask` | Bitwise NOT of src |

## Side Effects

None.

## Constraints

- **Operand widths**: Both predicates MUST have the same width.
- **No implicit extension**: `pnot` operates on the full predicate width. For predicates of mixed widths, explicit pack/unpack must be used.

## Exceptions

- Illegal if predicate operand widths are not consistent.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Bitwise NOT | Simulated | Supported | Supported |

## Examples

### Invert a predicate

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void invert_mask(RegBuf<predicate_t>& dst,
                 const RegBuf<predicate_t>& src) {
    PNOT(dst, src, src);
}
```

### SSA form — complement of comparison result

```mlir
// %cmp: lanes where a[i] < b[i]
%cmp = pto.vcmp %va, %vb, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// %tail: lanes in remainder region
%tail = pto.pge_b32 %rem : i32 -> !pto.mask

// Complement: lanes NOT in remainder region
%not_tail = pto.pnot %tail, %tail : !pto.mask, !pto.mask -> !pto.mask

// Combine: lanes in remainder region AND NOT in comparison result
%active = pto.pand %tail, %not_tail, %tail : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pxor](./pxor.md)
- Next op in family: [pto.psel](./psel.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
