# TCOLEXPANDDIV

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

![TCOLEXPANDDIV tile operation](../figures/isa/TCOLEXPANDDIV.svg)

## Introduction

Column-wise broadcast divide: divide each element of `src0` by a per-column scalar vector `src1`.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} / s_j
$$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- `TileDataDst::DType`, `TileDataSrc1::DType` must be one of: `half`, `float` for A2, A3 and A5, `int16`, `int32`, `uint16`, `uint32`, `int64`, `uint64` for A5.
- Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor`.
- `src1` is expected to provide **one scalar per column** (i.e., its valid shape must cover `C` values).
- Exact layout/fractal constraints are target-specific; see backend headers under `include/pto/npu/*/TColExpand*.hpp`.

### 64-bit element types (A5)

`int64` / `uint64` are supported on A5 only. A5 has no native 64-bit vector ALU, so the instruction is emulated on pairs of 32-bit registers that hold the low and the high word of every element; the per-column operand is read with the same de-interleaved layout as the full-sized operand.

Division truncates toward zero, matching the 32-bit integer behavior; the division-by-zero behavior is target-defined. Tile alignment follows the usual rule for 64-bit elements: a RowMajor tile needs `Cols % 4 == 0`.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
