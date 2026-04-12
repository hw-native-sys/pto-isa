<!-- Generated from `docs/isa/TAXPY.md` -->

# TAXPY

## Tile Operation Diagram

![TAXPY tile operation](../figures/isa/TAXPY.svg)

## Introduction

AXPY-style fused update: multiply a tile by a scalar and accumulate into the destination tile.

## Math Interpretation

Semantics are instruction-specific. Unless stated otherwise, behavior is defined over the destination valid region.

## Assembly Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%dst = pto.taxpy ...
```

### IR Level 2 (DPS)

```text
pto.taxpy ins(...) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`.

## Constraints

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## Examples

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
