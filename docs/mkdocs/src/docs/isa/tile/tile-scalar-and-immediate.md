<!-- Generated from `docs/isa/tile/tile-scalar-and-immediate.md` -->

# Tile-Scalar And Immediate Family

Tile-scalar operations combine a tile operand with a scalar value or immediate operand. The scalar is broadcast to match the tile shape. Comparison variants produce a predicate tile.

## Operations

| Operation | Description | Category | C++ Intrinsic |
|-----------|-------------|----------|----------------|
| [pto.tadds](./ops/tile-scalar-and-immediate/tadds.md) | Elementwise addition with scalar | Binary | `TADDS(dst, src, scalar)` |
| [pto.tsubs](./ops/tile-scalar-and-immediate/tsubs.md) | Elementwise subtraction with scalar | Binary | `TSUBS(dst, src, scalar)` |
| [pto.tmuls](./ops/tile-scalar-and-immediate/tmuls.md) | Elementwise multiplication with scalar | Binary | `TMULS(dst, src, scalar)` |
| [pto.tdivs](./ops/tile-scalar-and-immediate/tdivs.md) | Elementwise division with scalar | Binary | `TDIVS(dst, src, scalar)` |
| [pto.tfmods](./ops/tile-scalar-and-immediate/tfmods.md) | Elementwise modulo with scalar | Binary | `TFMODS(dst, src, scalar)` |
| [pto.trems](./ops/tile-scalar-and-immediate/trems.md) | Elementwise remainder with scalar | Binary | `TREMS(dst, src, scalar)` |
| [pto.tmins](./ops/tile-scalar-and-immediate/tmins.md) | Elementwise minimum with scalar | Binary | `TMINS(dst, src, scalar)` |
| [pto.tmaxs](./ops/tile-scalar-and-immediate/tmaxs.md) | Elementwise maximum with scalar | Binary | `TMAXS(dst, src, scalar)` |
| [pto.tands](./ops/tile-scalar-and-immediate/tands.md) | Elementwise AND with scalar | Binary | `TANDS(dst, src, scalar)` |
| [pto.tors](./ops/tile-scalar-and-immediate/tors.md) | Elementwise OR with scalar | Binary | `TORS(dst, src, scalar)` |
| [pto.txors](./ops/tile-scalar-and-immediate/txors.md) | Elementwise XOR with scalar | Binary | `TXORS(dst, src, scalar)` |
| [pto.tshls](./ops/tile-scalar-and-immediate/tshls.md) | Shift left by scalar | Binary | `TSHLS(dst, src, shift)` |
| [pto.tshrs](./ops/tile-scalar-and-immediate/tshrs.md) | Shift right by scalar | Binary | `TSHRS(dst, src, shift)` |
| [pto.tlrelu](./ops/tile-scalar-and-immediate/tlrelu.md) | Leaky ReLU with scalar slope | Binary | `TLRELU(dst, src, slope)` |
| [pto.taddsc](./ops/tile-scalar-and-immediate/taddsc.md) | Saturating add with scalar | Binary | `TADDSC(dst, src, scalar)` |
| [pto.tsubsc](./ops/tile-scalar-and-immediate/tsubsc.md) | Saturating subtract with scalar | Binary | `TSUBSC(dst, src, scalar)` |
| [pto.texpands](./ops/tile-scalar-and-immediate/texpands.md) | Compare tile with scalar, produce predicate | Comparison | `TEXPMDS(dst, src, scalar)` |
| [pto.tcmps](./ops/tile-scalar-and-immediate/tcmps.md) | Compare tile with scalar, produce predicate | Comparison | `TCMPS(dst, src, scalar, cmp)` |
| [pto.tsels](./ops/tile-scalar-and-immediate/tsels.md) | Select from two tiles based on scalar predicate | Selection | `TSELS(dst, src0, src1, pred)` |

## Mechanism

For each lane `(r, c)` in the destination's valid region:

$$ \mathrm{dst}_{r,c} = f(\mathrm{src}_{r,c}, \mathrm{scalar}) $$

The scalar operand is broadcast to all lanes. Comparison operations produce a predicate tile: lane `(r, c)` is `1` where the condition holds, `0` otherwise.

## Scalar Operand

The scalar operand may be:

- A scalar register value (`!pto.scalar<T>`)
- A compile-time immediate constant
- A runtime scalar value passed as a parameter

The scalar type must be compatible with the tile element type. No implicit type conversion is performed.

## Saturating Variants

`TADDSC` and `TSUBSC` perform saturating arithmetic (clamp to type min/max on overflow/underflow), in contrast to `TADDS`/`TSUBS` which use wrapping semantics.

## Type Support by Target Profile

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| f32 (float) | Yes | Yes | Yes |
| f16 (half) | Yes | Yes | Yes |
| bf16 (bfloat16_t) | Yes | Yes | Yes |
| i8 / u8 | Yes | Yes | Yes |
| i16 / u16 | Yes | Yes | Yes |
| i32 / u32 | Yes | Yes | Yes |
| i64 / u64 | Yes | Yes | Yes |

## Constraints

- The scalar type MUST be compatible with the tile element type.
- Shift operations (`TSHLS`, `TSHRS`) interpret the scalar as an unsigned integer shift count.
- Saturating variants (`TADDSC`, `TSUBSC`) clamp results to type min/max on overflow/underflow.
- Comparison variants produce a predicate tile, not a numeric tile.

## Cases That Are Not Allowed

- **MUST NOT** use a scalar type that is not compatible with the tile element type.
- **MUST NOT** use a shift count `>=` element bit-width.
- **MUST NOT** rely on implicit type promotion between scalar and tile types.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Arithmetic with scalar
template <typename TileDst, typename TileSrc, typename ScalarT>
PTO_INST RecordEvent TADDS(TileDst& dst, TileSrc& src, ScalarT scalar);

template <typename TileDst, typename TileSrc, typename ScalarT>
PTO_INST RecordEvent TMULS(TileDst& dst, TileSrc& src, ScalarT scalar);

template <typename TileDst, typename TileSrc, typename ScalarT>
PTO_INST RecordEvent TMAXS(TileDst& dst, TileSrc& src, ScalarT scalar);

// Saturating arithmetic with scalar
template <typename TileDst, typename TileSrc, typename ScalarT>
PTO_INST RecordEvent TADDSC(TileDst& dst, TileSrc& src, ScalarT scalar);

// Shift by scalar (shift is unsigned integer)
template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TSHLS(TileDst& dst, TileSrc& src, uint32_t shift);

// Leaky ReLU: dst[i,j] = (src[i,j] > 0) ? src[i,j] : slope * src[i,j]
template <typename TileDst, typename TileSrc, typename ScalarT>
PTO_INST RecordEvent TLRELU(TileDst& dst, TileSrc& src, ScalarT slope);

// Comparison with scalar (produces predicate tile)
template <typename TileDst, typename TileSrc, typename ScalarT>
PTO_INST RecordEvent TCMPS(TileDst& dst, TileSrc& src, ScalarT scalar, CompareMode cmp);
```

## See Also

- [Tile families](../instruction-families/tile-families.md) — Family overview
- [Tile instruction surface](../instruction-surfaces/tile-instructions.md) — Surface description
