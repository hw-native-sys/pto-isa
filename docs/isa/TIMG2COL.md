# TIMG2COL


## Tile Operation Diagram

![TIMG2COL tile operation](../figures/isa/TIMG2COL.svg)

## Introduction

Transform an input feature-map tile (e.g. NC1HWC0 layout) into an im2col-style matrix tile for convolution-like workloads. Parameters are provided via `Img2colTileConfig` and `(posM, posK)` offsets.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0,
                              WaitEvents&... events);
```

## Constraints

- This instruction is target/implementation-specific. See `include/pto/npu/*/TImg2col.hpp` for the supported tile types/layouts and config fields.

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

### IR Level 1 (SSA)

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

