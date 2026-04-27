# TCONCAT

## Tile Operation Diagram

![TCONCAT tile operation](../../../../figures/isa/TCONCAT.svg)

## Introduction

Concatenate two source tiles along the column dimension into a destination tile.

## Math Interpretation

Semantics are instruction-specific. Unless stated otherwise, behavior is defined over the destination valid region.

## Assembly Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%dst = pto.tconcat ...
```

### IR Level 2 (DPS)

```text
pto.tconcat ins(...) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`.

## Constraints

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## Examples

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
