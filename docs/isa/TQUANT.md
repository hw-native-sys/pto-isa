# TQUANT


## Tile Operation Diagram

![TQUANT tile operation](../figures/isa/TQUANT.svg)

## Introduction

Quantize an FP32 tile into a lower-precision format (e.g. FP8), producing auxiliary exponent/scaling/max tiles. The quantization mode is a compile-time template parameter (`mode`).

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST RecordEvent TQUANT(TileDataSrc &src, TileDataExp &exp, TileDataOut &dst,
                            TileDataMax &max, TileDataSrc &scaling, WaitEvents&... events);
```

## Constraints

- This instruction is currently implemented for specific targets (see `include/pto/npu/*/TQuant.hpp`).
- Input type requirements and output tile types are mode/target-dependent.

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

### IR Level 1 (SSA)

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

