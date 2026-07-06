# TADDDEQRELU

## Tile Operation Diagram

![TADDDEQRELU tile operation](../figures/isa/TADDDEQRELU.svg)

## Introduction

Fused Add + Dequantize + ReLU: elementwise addition of two int32 tiles, followed by dequantization scaling and ReLU activation, outputting a half-precision tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \max(0, (\mathrm{src0}_{i,j} + \mathrm{src1}_{i,j}) \times \mathrm{deqScale}) $$

The dequantization uses precision-compensated scaling: `(x >> 17) * deqScale << 17` which is mathematically equivalent to `x * deqScale` but avoids precision loss for large int32 intermediate values.

## Assembly Syntax

Synchronous form:

```text
%dst = tadddeqrelu %src0, %src1, %deqScale : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tadddeqrelu %src0, %src1, %deqScale : (!pto.tile<...>, !pto.tile<...>, f32) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tadddeqrelu ins(%src0, %src1, %deqScale : !pto.tile_buf<...>, !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TADDDEQRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, float deqScale,
                                 TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

### General constraints / checks

- `src0` and `src1` must be `TileType::Vec` with element type `int32_t`.
- `dst` must be `TileType::Vec` with element type `half`.
- All tiles must use row-major layout (`TileData::isRowMajor`).
- Runtime valid-region checks:
    - `dst.GetValidRow() > 0` and `dst.GetValidCol() > 0`
    - `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`
    - `src1.GetValidRow() == dst.GetValidRow()` and `src1.GetValidCol() == dst.GetValidCol()`
- `deqScale` is a `float` scalar.

### A2A3 implementation checks

- `tmp` must be `TileType::Vec` with element type `int32_t`.
- `tmp` must be row-major.
- `tmp.GetValidRow() >= dst.GetValidRow()` and `tmp.GetValidCol() >= dst.GetValidCol()`.

### A5 implementation checks

- `tmp` is accepted by the interface but not validated or used on A5.
- All intermediate values stay in vector registers.

## Temporary Space

### A2A3

`tmp` **is used** as intermediate scratch storage. The A2A3 implementation performs the fused operation in multiple steps:

1. `tmp = src0 + src1` (vadd)
2. Convert `tmp` from int32 to float
3. Apply precision-compensated scaling: `floatBuf = (tmp / 131072.0) * deqScale * 131072.0`
4. `reluBuf = max(floatBuf, 0.0)`
5. Convert result to half and store to `dst`

- `tmp` element type must be `int32_t`.
- `tmp` must be row-major and `TileType::Vec`.
- `tmp.GetValidRow() >= dst.GetValidRow()` and `tmp.GetValidCol() >= dst.GetValidCol()`.

### A5

`tmp` is accepted by the interface but **not used** by the A5 implementation. The A5 backend uses the register-compute model (`__VEC_SCOPE__`) where all intermediate values stay in vector registers. No separate UB tmp buffer is needed. `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, int32_t, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
  SrcT src0, src1;
  DstT dst;
  TmpT tmp;
  float deqScale = 0.5f;
  TADDDEQRELU(dst, src0, src1, deqScale, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, int32_t, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
  SrcT src0, src1;
  DstT dst;
  TmpT tmp;
  float deqScale = 0.5f;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TASSIGN(tmp,  0x4000);
  TADDDEQRELU(dst, src0, src1, deqScale, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tadddeqrelu %src0, %src1, %deqScale : (!pto.tile<...>, !pto.tile<...>, f32) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tadddeqrelu %src0, %src1, %deqScale : (!pto.tile<...>, !pto.tile<...>, f32) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tadddeqrelu %src0, %src1, %deqScale : !pto.tile<...>
# AS Level 2 (DPS)
pto.tadddeqrelu ins(%src0, %src1, %deqScale : !pto.tile_buf<...>, !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```
