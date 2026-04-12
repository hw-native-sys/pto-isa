<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/psel.md` -->

# pto.psel

Standalone reference page for `pto.psel`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Predicate mux: select between two predicate sources based on a third predicate.

## Mechanism

`pto.psel` selects predicate bits from one of two source predicates based on a third predicate. For each lane `i`:

$$ \mathrm{dst}_i = \begin{cases} \mathrm{src0}_i & \text{if } \mathrm{sel}_i = 1 \\ \mathrm{src1}_i & \text{if } \mathrm{sel}_i = 0 \end{cases} $$

This is a predicate-level ternary select, analogous to vector `vsel` but operating on predicate values directly.

## Syntax

### PTO Assembly Form

```text
psel %dst, %src0, %src1, %sel : !pto.mask, !pto.mask, !pto.mask, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst = pto.psel %src0, %src1, %sel, %mask : !pto.mask, !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.psel ins(%src0, %src1, %sel, %mask : !pto.mask, !pto.mask, !pto.mask, !pto.mask) outs(%dst : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PSEL(RegBuf<predicate_t>& dst,
                   const RegBuf<predicate_t>& src0,
                   const RegBuf<predicate_t>& src1,
                   const RegBuf<predicate_t>& sel,
                   const RegBuf<predicate_t>& mask);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask` | Predicate selected when corresponding sel bit is 1 |
| `%src1` | `!pto.mask` | Predicate selected when corresponding sel bit is 0 |
| `%sel` | `!pto.mask` | Per-lane selection predicate |
| `%mask` | `!pto.mask` | Optional masking predicate |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.mask` | Per-lane selection between src0 and src1 |

## Side Effects

None.

## Constraints

- **Operand widths**: All four predicate operands MUST have the same width.
- **Select semantic**: `sel_i = 1` → select `src0_i`; `sel_i = 0` → select `src1_i`.

## Exceptions

- Illegal if predicate operand widths are not consistent.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Predicate select | Simulated | Supported | Supported |

## Examples

### Predicate mux

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void select_predicate(RegBuf<predicate_t>& dst,
                     const RegBuf<predicate_t>& src0,
                     const RegBuf<predicate_t>& src1,
                     const RegBuf<predicate_t>& sel) {
    PSEL(dst, src0, src1, sel, sel);
}
```

### SSA form — dynamic predicate routing

```mlir
// %active_a: predicate from comparison A
// %active_b: predicate from comparison B
// %condition: runtime condition determining which set to use

// If condition is true, use set A; otherwise use set B
%active = pto.psel %active_a, %active_b, %condition, %condition
    : !pto.mask, !pto.mask, !pto.mask, !pto.mask
    -> !pto.mask
```

### Equivalent to boolean expression

The `psel` operation is equivalent to the following boolean expression:

```mlir
// psel %dst, %src0, %src1, %sel
// = (src0 AND sel) OR (src1 AND NOT sel)

%sel_inv = pto.pnot %sel, %sel : !pto.mask, !pto.mask -> !pto.mask
%and0 = pto.pand %src0, %sel, %sel : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
%and1 = pto.pand %src1, %sel_inv, %sel : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
%dst = pto.por %and0, %and1, %and0 : !pto.mask, !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pnot](./pnot.md)
- Next op in family: [pto.pdintlv_b8](./pdintlv-b8.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
