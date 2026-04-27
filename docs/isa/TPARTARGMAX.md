# TPARTARGMAX


## Tile Operation Diagram

![TPARTARGMAX tile operation](../figures/isa/TPARTMAX.svg)

## Introduction

Performs elementwise maximum selection over the destination valid region and returns the corresponding index values. When both `src0Val` and `src1Val` are valid at an element, the result value is `max(src0Val, src1Val)` and the result index is the index of the maximum value; when only one input is valid there, the result copies that input's value and index. Handling of other mismatched-validity cases is implementation-defined.

## Math Interpretation

For each element `(i, j)` in the destination valid region:

$$
\begin{aligned}
(\mathrm{dstVal}_{i,j}, \mathrm{dstIdx}_{i,j}) =
\begin{cases}
(\mathrm{src0Val}_{i,j}, \mathrm{src0Idx}_{i,j}) & \text{if } \mathrm{src0Val}_{i,j} > \mathrm{src1Val}_{i,j} \text{ and both inputs are defined at } (i,j) \\
(\mathrm{src1Val}_{i,j}, \mathrm{src1Idx}_{i,j}) & \text{if } \mathrm{src1Val}_{i,j} \ge \mathrm{src0Val}_{i,j} \text{ and both inputs are defined at } (i,j) \\
(\mathrm{src0Val}_{i,j}, \mathrm{src0Idx}_{i,j}) & \text{if only src0 is defined at } (i,j) \\
(\mathrm{src1Val}_{i,j}, \mathrm{src1Idx}_{i,j}) & \text{if only src1 is defined at } (i,j)
\end{cases}
\end{aligned}
$$

## Assembly Syntax

PTO-AS form: see [Assembly model](syntax-and-operands/assembly-model.md).

Synchronous form:

```text
%dstVal, %dstIdx = tpartargmax %src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

### AS Level 1 (SSA)

```text
%dstVal, %dstIdx = pto.tpartargmax %src0Val, %src1Val, %src0Idx, %src1Idx : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

### AS Level 2 (DPS)

```text
pto.tpartargmax ins(%src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
          typename TileDataDstIdx, typename TileDataSrc0Idx, typename TileDataSrc1Idx,
          typename... WaitEvents>
PTO_INST RecordEvent TPARTARGMAX(TileDataDst &dstVal, TileDataSrc0 &src0Val, TileDataSrc1 &src1Val,
                                 TileDataDstIdx &dstIdx, TileDataSrc0Idx &src0Idx, TileDataSrc1Idx &src1Idx,
                                 WaitEvents &... events);
```

## Constraints

### General constraints / checks

- `dstVal`, `src0Val`, and `src1Val` must use the same element type.
- `dstIdx`, `src0Idx`, and `src1Idx` must use the same element type.
- Value type and index type combination constraints:
    - If the value type is `half`, the index type must be `int16_t` or `uint16_t`.
    - If the value type is `float`, the index type must be `int32_t` or `uint32_t`.
- Valid regions must match between value tiles and index tiles for each pair:
    - `src0Val` and `src0Idx` must have identical valid regions.
    - `src1Val` and `src1Idx` must have identical valid regions.
    - `dstVal` and `dstIdx` must have identical valid regions.
- The destination valid region must exactly match the valid region of either `src0Val` or `src1Val`.
- If `dstVal` has a zero valid region, the instruction returns early.
- For each element in the destination valid region:
    - if both inputs are valid, the instruction applies the elementwise maximum and returns the index of the larger value;
    - if only one input is valid, the result copies that input's value and index.
- Handling of any validity pattern not explicitly listed above is implementation-defined.

### A5 implementation checks

- Supported value types: `half`, `float`.
- Supported index types: `int16_t`, `uint16_t`, `int32_t`, `uint32_t`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using ValTileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxTileT = Tile<TileType::Vec, int32_t, 16, 16>;
  ValTileT src0Val, src1Val, dstVal;
  IdxTileT src0Idx, src1Idx, dstIdx;
  TPARTARGMAX(dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using ValTileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxTileT = Tile<TileType::Vec, int32_t, 16, 16>;
  ValTileT src0Val, src1Val, dstVal;
  IdxTileT src0Idx, src1Idx, dstIdx;
  TASSIGN(src0Val, 0x1000);
  TASSIGN(src1Val, 0x2000);
  TASSIGN(dstVal,  0x3000);
  TASSIGN(src0Idx, 0x4000);
  TASSIGN(src1Idx, 0x5000);
  TASSIGN(dstIdx,  0x6000);
  TPARTARGMAX(dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dstVal, %dstIdx = pto.tpartargmax %src0Val, %src1Val, %src0Idx, %src1Idx : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dstVal, %dstIdx = pto.tpartargmax %src0Val, %src1Val, %src0Idx, %src1Idx : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

### PTO Assembly Form

```text
%dstVal, %dstIdx = tpartargmax %src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
# AS Level 2 (DPS)
pto.tpartargmax ins(%src0Val, %src1Val, %src0Idx, %src1Idx : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```
