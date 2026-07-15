# TCOLSUM


## Tile Operation Diagram

![TCOLSUM tile operation](../figures/isa/TCOLSUM.svg)

## Introduction

Reduce each column by summing across rows.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dst}_{0,j} = \sum_{i=0}^{R-1} \mathrm{src}_{i,j} $$

`isBinary` selects the implementation path (binary-tree accumulation vs. sequential accumulation).

## Assembly Syntax

Synchronous form:

```text
%dst = tcolsum %src {isBinary = false} : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary, WaitEvents &... events);
```

## Constraints

### General constraints / checks

- `dst` and `src` must be `TileType::Vec`.
- `dst` and `src` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).
- `dst` and `src` must use the same element type.
- Runtime checks:
    - `src.GetValidCol() == dst.GetValidCol()`
    - `src.GetValidRow() != 0` (implementation silently returns early when zero, no computation performed)
    - `src.GetValidCol() != 0` (implementation silently returns early when zero, no computation performed)
    - `src.GetValidCol() <= tmp` row stride measured in `src` elements
- `isBinary` selects the checked backend path:
    - `true`: binary-tree accumulation using `tmp`
    - `false`: sequential accumulation into `dst`

### A2A3 implementation checks

- Supported element types: `half`, `float`, `int16_t`, `int32_t`.
- `tmp` must be `TileType::Vec` and use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).
- `tmp` must use the same element type as `src` and `dst`.
- If `src.GetValidRow() == 0` or `src.GetValidCol() == 0`, the implementation returns early.

### A5 implementation checks

- Shared A5 column-reduce checks allow `half`, `float`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `bfloat16_t`.
- The checked A5 `TCOLSUM` path still takes `tmp` only for the binary accumulation path; no extra compile-time `tmp` type/layout assertions are explicitly enforced in `TCOLSUM_IMPL`.

## Temporary Space

### Without `tmp` (2-argument overload: `TCOLSUM(dst, src)`)

No `tmp` is required. Both A2A3 and A5 use sequential accumulation directly into `dst`.

### With `tmp` and `isBinary` (4-argument overload: `TCOLSUM(dst, src, tmp, isBinary)`)

#### A2A3

- When `isBinary = true`: `tmp` **is used** for binary-tree accumulation. Adjacent row pairs from `src` are summed into `tmp`, then `tmp` is recursively halved until a single row remains.
  - `tmp` must have the same element type as `src`/`dst`.
  - `tmp` must be `TileType::Vec`, row-major, non-fractal.
  - `tmp.GetValidCol() >= src.GetValidCol()` (in element count, accounting for `tmp` stride).
  - `tmp` needs at least `ceil(src.GetValidRow() / 2)` rows.
- When `isBinary = false`: `tmp` is accepted but the implementation uses sequential accumulation into `dst`; `tmp` is not actively used.

#### A5

- When `isBinary = true`: `tmp` **is used** for binary-tree accumulation in vector registers with UB storage.
  - `tmp` must have the same element type as `src`/`dst`.
  - `tmp.GetValidCol() >= src.GetValidCol()` (in element count, accounting for `tmp` stride).
  - `tmp` needs at least `ceil(src.GetValidRow() / 2)` rows.
- When `isBinary = false`: `tmp` is not actively used; the implementation uses sequential reduction via `TColReduceInstr`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TCOLSUM(dst, src, tmp, /*isBinary=*/false);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TCOLSUM(dst, src, tmp, /*isBinary=*/false);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcolsum %src {isBinary = false} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
