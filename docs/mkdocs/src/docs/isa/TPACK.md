<!-- Generated from `docs/isa/TPACK.md` -->

# TPACK

## Tile Operation Diagram

![TPACK tile operation](../figures/isa/TPACK.svg)

## Introduction

Pack or convert tile elements into a narrower destination representation.

## Math Interpretation

Semantics are instruction-specific. Unless stated otherwise, behavior is defined over the destination valid region.

## Assembly Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%dst = pto.tpack ...
```

### IR Level 2 (DPS)

```text
pto.tpack ins(...) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`.

## Constraints

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## Examples

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
