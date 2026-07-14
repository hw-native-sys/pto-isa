# TAXPY

## Introduction

Perform an in-place scaled accumulation (AXPY, $a \cdot x + y$) on a tile: scale `src0` by `scalar` and accumulate it into `dst`.

$$ \mathrm{dst}_{i,j} \leftarrow \mathrm{scalar} \cdot \mathrm{src0}_{i,j} + \mathrm{dst}_{i,j} $$

`dst` is both the accumulation input ($y$) and the output, and must be initialized before the call; `src0` ($x$) is read-only; `scalar` ($a$) is a scalar.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j}^{\text{new}} = \mathrm{scalar} \cdot \mathrm{src0}_{i,j} + \mathrm{dst}_{i,j}^{\text{old}} $$

- `dst`: read-modify-write (RMW). Its old value is read as the accumulation base $y$, and $\mathrm{scalar} \cdot x + y$ is written back.
- `src0`: read-only, contributes element-wise ($x$).
- `scalar`: the scalar scale factor ($a$), of type `TileDataSrc::DType`.

> Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## C++ Intrinsics

Declared in `include/pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TAXPY(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events);
```

| Parameter | Direction | Meaning |
|-----------|-----------|---------|
| `dst` | input/output | Accumulation base and result tile ($y$), read-modify-write, `Vec` |
| `src0` | input | Scaled source tile ($x$), read-only, `Vec`, same valid shape as `dst` |
| `scalar` | input | Scalar scale factor ($a$), of type `TileDataSrc::DType` |
| `events...` | input | Wait events (`WaitEvents`); an implicit `TSYNC` precedes the op |

## Tile Sizes & Data Types

For a valid tile shape of $M \times N$:

| Tile | dtype | Valid shape | TileType | Notes |
|------|-------|-------------|----------|-------|
| `dst` | `half` or `float` | $M \times N$ | `Vec` (UB) | Accumulation base + result (RMW) |
| `src0` | `half` or `float` | $M \times N$ | `Vec` (UB) | Scaled source, element-wise |

> `dst` and `src0` must have identical valid row and column counts.

## Supported Input Dtypes

| `dst` dtype | `src0` dtype | `scalar` dtype | Notes |
|-------------|--------------|----------------|-------|
| `half` | `half` | `half` | Same-type path, direct `vaxpy` |
| `float` | `float` | `float` | Same-type path, direct `vaxpy` |
| `float` | `half` | `half` | Diff path: `src0` widened to FP32 before accumulation |

> `dst` and `src0` must share a dtype, or `dst` is `float` while `src0` is `half` (a half→float widening accumulation is allowed). A `half` `dst` with a `float` `src0` is illegal (rejected by an in-implementation `static_assert`).

## Implementation Notes

TAXPY runs on the vector pipeline (`PIPE_V`) using the `vaxpy` ($a \cdot x + y$) vector intrinsic:

1. **Same type (`dst` and `src0` share a dtype)**: load `src0` and `dst` per repeat, run `vaxpy(dst, src0, scalar)`, and store back to `dst`; tail columns shorter than a full repeat are masked by a predicate.
2. **Diff type (`dst`=`float`, `src0`=`half`)**: the half data of `src0` is widened to FP32 before accumulating (on A5 via `UNPK_B16` unpack followed by `vcvt`; on A2/A3 handled natively by `vaxpy` with 4-block src / 8-block dst).
3. On A2/A3, count mode vs. norm mode is selected based on whether the repeat-stride overflows and the relation between column count and row count, so any valid shape is covered.

## Constraints

| Constraint | Applies to | Reason |
|------------|------------|--------|
| `dst` and `src0` must be `TileType::Vec` | all targets | executes on UB (vector pipeline) |
| `dst` and `src0` share the valid shape ($M \times N$) | all targets | one-to-one element mapping |
| `dst` dtype ∈ {`half`, `float`} | all targets | floating-point widths supported by `vaxpy` |
| `dst`/`src0` dtype equal, or (`float`,`half`) | all targets | only half→float widening accumulation allowed |
| `dst` must be initialized before the call | all targets | `dst` is read in as the accumulation base $y$ |

## Examples

```cpp
// dst must be initialized first (as the accumulation base y); result: dst = scalar * src0 + dst
TAXPY(dstTile, srcTile, scalar);
```

See `tests/npu/a5/src/st/testcase/taxpy/` (A5), `tests/npu/a2a3/src/st/testcase/taxpy/` (A2/A3), `tests/npu/kirin9030/src/st/testcase/taxpy/` (Kirin9030), and `tests/cpu/st/testcase/taxpy/` (CPU reference) for complete ST examples.
