# TEXTRACT_FP


## Tile Operation Diagram

![TEXTRACT_FP tile operation](../figures/isa/TEXTRACT_FP.svg)

## Introduction

Extract a sub-tile from a source tile, while also providing an `fp` (scaling) tile used for vector quantization parameters (target/implementation-defined).

## See also

- TEXTRACT base instruction: `docs/isa/TEXTRACT.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Constraints

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
