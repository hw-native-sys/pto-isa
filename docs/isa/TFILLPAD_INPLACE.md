# TFILLPAD_INPLACE


## Tile Operation Diagram

![TFILLPAD_INPLACE tile operation](../figures/isa/TFILLPAD_INPLACE.svg)

## Introduction

In-place fill/pad variant of TFILLPAD (implementation-defined).

## See also

- TFILLPAD overview and constraints: `docs/isa/TFILLPAD.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData &dst, SrcTileData &src,
                            WaitEvents&... events);
```

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

### IR Level 1 (SSA)

```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## Constraints

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

