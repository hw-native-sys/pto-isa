# TIMG2COL

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->



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
- **Implementation checks (A2A3)**: `TileData::DType` must be one of: `int8_t`, `half`, `bfloat16_t`, `float`.
- **Implementation checks (A5)**: `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.

### ConvTile conversion paths

The ConvTile path requires `ConvTileData::Loc == TileType::Mat` and source layout `NC1HWC0` or `NDC1HWC0`.
The destination paths and layout requirements are:

| Architecture | Destination | On-chip storage | Layout requirement |
| --- | --- | --- | --- |
| A2A3 | `TileType::Left` | L0A | `SFractal == SLayout::RowMajor` and `BLayout == RowMajor` |
| A5 | `TileType::Left` | L0A | `SFractal == SLayout::RowMajor` and `BLayout == ColMajor` |
| A5 | `TileType::Right` | L0B | `SFractal == SLayout::ColMajor` and `BLayout == RowMajor` |

Destination and source dtypes must be identical. Convolution configuration fields are still read from the source ConvTile.

The current A5 `Right` path does not use the source ConvTile `transpose` configuration; that field does not change
the output arrangement of a `Right` destination. A2A3 still accepts only `Left` destinations.

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
