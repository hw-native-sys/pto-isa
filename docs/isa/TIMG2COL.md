# TIMG2COL


## Tile Operation Diagram

![TIMG2COL tile operation](../figures/isa/TIMG2COL.svg)

## Introduction

Transform an input feature-map tile (e.g. NC1HWC0 layout) into an im2col-style matrix tile for convolution-like workloads. Parameters are provided via `Img2colTileConfig` and `(posM, posK)` offsets.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0,
                              WaitEvents &... events);
```

## Constraints

- This instruction is target/implementation-specific. See `include/pto/npu/*/TImg2col.hpp` for the supported tile types/layouts and config fields.

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
