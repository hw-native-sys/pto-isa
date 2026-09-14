# TROWMAX


## Tile Operation Diagram

![TROWMAX tile operation](../figures/isa/TROWMAX.svg)

## Introduction

Reduce each row by taking the maximum across columns.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \max_{0 \le j < C} \mathrm{src}_{i,j} $$

## Assembly Syntax

Synchronous form:

```text
%dst = trowmax %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

### General constraints / checks

- `dst` and `src` must both be `TileType::Vec`.
- `src` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).
- `dst` must use one of the following non-fractal layouts:
    - ND layout (`BLayout::RowMajor`, `SLayout::NoneBox`), or
    - DN layout with exactly one column (`BLayout::ColMajor`, `SLayout::NoneBox`, `Cols == 1`).
- `dst` and `src` must use the same element type.
- Runtime valid-region checks:
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`
- The intrinsic signature requires an explicit `tmp` operand.

### A2A3 implementation checks

- Supported element types: `half`, `float`, `int32_t`, `int16_t`.

### A5 implementation checks

- Supported element types: `half`, `float`, `int32_t`, `int64_t`, `uint64_t`, `int16_t`, `int8_t`, `uint8_t`.
- For `int64_t` / `uint64_t`:
    - Set the output valid column count to 1. Only column 0 of valid rows is written; other physical padding is preserved.
    - ND output requires physical `Cols % 4 == 0`; DN output requires physical `Cols == 1` and `Rows % 4 == 0`. Valid rows need not be a multiple of 4.
    - Row strides follow physical shapes; see [shape and layout conventions](conventions.md).

## Temporary Space

### A2A3

`tmp` **is used** as scratch storage for row-wise max reduction.

- For **integer** types (`int32_t`, `int16_t`): `tmp` is used as a per-row accumulator buffer (1 block). For each row, `tmp` is initialized to the minimum representable value, then blocks of `src` are accumulated via `vmax`. The final max is read from `tmp` in scalar mode.
  - `tmp` size: at least 1 row and `BLOCK_BYTE_SIZE / sizeof(T)` columns (8 for `int32_t`, 16 for `int16_t`).
- For **floating-point** types (`float`, `half`): `tmp` is used for binary-tree reduction via `vcmax`/`vcgmax`.
  - A safe default: set `tmp` to the same shape as `src`.

### A5

`tmp` is accepted but not used. The 64-bit integer path performs exact integer reduction without floating-point conversion.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWMAX(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWMAX(dst, src, tmp);
}
```

### 64-bit ND output (A5)

The output has physical shape `[64,4]` and valid shape `[64,1]`, with one 8-byte result every 32 bytes.

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_int64() {
  using SrcT = Tile<TileType::Vec, int64_t, 64, 16>;
  using DstT = Tile<TileType::Vec, int64_t, 64, 4, BLayout::RowMajor, 64, 1>;
  SrcT src, tmp;
  DstT dst;
  TROWMAX(dst, src, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trowmax %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
