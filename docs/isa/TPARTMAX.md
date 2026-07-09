# TPARTMAX


## Tile Operation Diagram

![TPARTMAX tile operation](../figures/isa/TPARTMAX.svg)

## Introduction

Performs elementwise maximum selection over the destination valid region. When both `src0` and `src1` are valid at an element, the result is `max(src0, src1)`; when only one input is valid there, the result copies that input value. Handling of other mismatched-validity cases is implementation-defined.

## Math Interpretation

For each element `(i, j)` in the destination valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\max(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{if both inputs are defined at } (i,j) \\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j)
\end{cases}
$$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

### General constraints / checks

- `dst`, `src0`, and `src1` must use the same element type.
- The destination valid region defines the result domain.
- For each element in the destination valid region:
    - if both inputs are valid, the instruction applies the elementwise maximum;
    - if only one input is valid, the result copies that input value.
- If `dst` has a zero valid region, the instruction returns early.
- Supported partial-validity patterns require at least one source tile to have a valid region exactly equal to `dst`, while the other source tile's valid region must not exceed `dst` in either dimension.
- Handling of any validity pattern not explicitly listed above is implementation-defined.

### A2A3 implementation checks

- Supported element types: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
- `dst`, `src0`, and `src1` must all be row-major (`isRowMajor`).

### A5 implementation checks

- Supported element types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTMAX(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TPARTMAX(dst, src0, src1);
}
```
