# TINSERT


## Tile Operation Diagram

![TINSERT tile operation](../figures/isa/TINSERT.svg)

## Introduction

Insert a source sub-tile into a destination tile at `(indexRow, indexCol)`. This is conceptually the inverse of `TEXTRACT` for many layouts.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. Conceptually, for `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}
$$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                            uint16_t indexRow, uint16_t indexCol, WaitEvents&... events);
```

## Constraints

- Runtime bounds must satisfy `indexRow + src.Rows <= dst.Rows` and `indexCol + src.Cols <= dst.Cols` (exact checks are target-dependent).

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

