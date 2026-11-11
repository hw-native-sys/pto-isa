# TROWPROD

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

![TROWPROD tile operation](../figures/isa/TROWPROD.svg)

## Introduction

Reduce each row by multiplying across columns.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \prod_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
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

- Supported element types: `half`, `float`, `int32_t`, `int16_t`.
- In the currently inspected implementation path, the enforced constraints are on `src` and `dst`.
- No extra shape/layout assertions on `tmp` are enforced in the current implementation path.

### CPU_SIM implementation checks

- `half` / `bfloat16_t` products are widened to `float` (fp32) and rounded back to the output type only
  once; other types accumulate in their own type. `tmp` is accepted for API compatibility but its storage
  is not accessed.

## Temporary Space

### A2A3

`tmp` **is used** as a per-row accumulator buffer. For each row, the implementation initializes `tmp` with `1.0` and then multiplies blocks of `src` data into `tmp` using `vmul`. After all blocks are accumulated, the scalar-mode pipeline reads `tmp` elements and computes the final product.

- `tmp` must have the same element type as `src`/`dst`.
- `tmp` size: at least 1 row and `CeilDivision(TileDataIn::Cols / 2, REPEAT_BYTE_SIZE / sizeof(T))` columns.
- A safe default: set `tmp` to the same shape as `src`.

### A5

`tmp` is accepted by the interface but **not used** by the A5 implementation. The A5 backend uses vector register-based reduction (`vmul` + `vintlv` for tree reduction) and does not require scratch tile storage. `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.

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
  TROWPROD(dst, src, tmp);
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
  TROWPROD(dst, src, tmp);
}
```
