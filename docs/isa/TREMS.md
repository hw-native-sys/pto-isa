# TREMS


## Tile Operation Diagram

![TREMS tile operation](../figures/isa/TREMS.svg)

## Introduction

Elementwise remainder with a scalar: `%`.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \bmod \mathrm{scalar}$$

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1 (SSA)

```text
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, TileDataTmp &tmp,
                           WaitEvents &...events);
```

`PrecisionType` has the following values available:

* `RemSAlgorithm::DEFAULT`: Normal algorithm, faster but with lower precision.
* `RemSAlgorithm::HIGH_PRECISION`: High precision algorithm, but slower, only support `float` type.

## Constraints

- **Implementation Checks (A2A3)**:
    - `dst` and `src` must use the same element type.
    - Supported element types: `float` and `int32_t`.
    - `dst` and `src` must be vector tiles.
    - `dst` and `src` must be row-major.
    - Runtime: `dst.GetValidRow() == src.GetValidRow() > 0` and `dst.GetValidCol() == src.GetValidCol() > 0`.
    - **tmp Buffer Requirements**:
      - `tmp.GetValidCol() >= dst.GetValidCol()` (at least as many columns as dst)
      - `tmp.GetValidRow() >= 1` (at least 1 row)
      - Data type must match `TileDataDst::DType`.
- **Implementation Checks (A5)**:
    - `dst` and `src` must use the same element type.
    - Supported element types: `float`, `int32_t`, `uint32_t`, `half`, `int16_t`, and `uint16_t`.
    - `dst` and `src` must be vector tiles.
    - Static valid bounds: `ValidRow <= Rows` and `ValidCol <= Cols` for both tiles.
    - Runtime: `dst.GetValidRow() == src.GetValidRow()` and `dst.GetValidCol() == src.GetValidCol()`.
    - Note: tmp parameter is accepted but not validated or used on A5.
- **Division by Zero**:
    - Behavior is target-defined; the CPU simulator asserts in debug builds.
- **Valid Region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.
- **For `int32_t` Inputs (A2A3 Only)**: Both `src` elements and `scalar` must be in the range `[-2^24, 2^24]` (i.e., `[-16777216, 16777216]`) to ensure exact conversion to float32 during computation.
- **High Precision Algorithm**
    - Only available on A5, `PrecisionType` option is ignored on A3.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
# AS Level 2 (DPS)
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

