# TTRI


## Tile Operation Diagram

![TTRI tile operation](../figures/isa/TTRI.svg)

## Introduction

Generate a (lower/upper) triangular mask tile with ones and zeros. The triangular orientation is controlled by the compile-time template parameter `isUpperOrLower` (0 = lower, 1 = upper).

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `d = diagonal`.

Lower-triangular (`isUpperOrLower=0`) conceptually produces:

$$
\mathrm{dst}_{i,j} = \begin{cases}1 & j \le i + d \\\\ 0 & \text{otherwise}\end{cases}
$$

Upper-triangular (`isUpperOrLower=1`) conceptually produces:

$$
\mathrm{dst}_{i,j} = \begin{cases}0 & j < i + d \\\\ 1 & \text{otherwise}\end{cases}
$$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents &... events);
```

## Constraints

- `isUpperOrLower` must be `0` (lower) or `1` (upper).
- **Implementation checks (A2A3)**:
    - Destination tile must be row-major (`isRowMajor`), enforced by `static_assert`.
    - Supported element types: `int32_t`, `int`, `int16_t`, `uint32_t`, `uint16_t`, `half`, `float16_t`, `float`, `float32_t`.
- **Implementation checks (A5)**:
    - Supported element types: `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `uint16_t`, `uint8_t`, `half`, `float16_t`, `float32_t`, `bfloat16_t`.
    - Lower (`upperOrLower == 0`) and upper (`upperOrLower == 1`) are distinguished by `if constexpr` branches.
- Valid region is obtained via `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_lower() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TTRI<0>(dst, /*diagonal=*/0);   // lower triangular
}

void example_upper() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TTRI<1>(dst, /*diagonal=*/-1);  // upper triangular
}
```
