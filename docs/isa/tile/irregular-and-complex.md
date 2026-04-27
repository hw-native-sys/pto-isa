# Irregular And Complex Instruction Set

Irregular operations cover tile compute that does not fit the standard elementwise, reduce, or memory models. These include debugging, sorting, quantization, index-based data movement, triangular matrix operations, and partial reductions.

## Operations

| Operation | Description | Category | Target Profile |
|-----------|-------------|----------|:-------------:|
| [pto.tprint](./ops/irregular-and-complex/tprint.md) | Print tile data for debugging | Debug | All |
| [pto.tmrgsort](./ops/irregular-and-complex/tmrgsort.md) | Merging sort of tile rows | Sort | All |
| [pto.tsort32](./ops/irregular-and-complex/tsort32.md) | Sort 32-bit values | Sort | All |
| [pto.tgather](./ops/irregular-and-complex/tgather.md) | Gather tile elements by index | Gather | All |
| [pto.tgatherb](./ops/irregular-and-complex/tgatherb.md) | Batch gather | Gather | All |
| [pto.tscatter](./ops/irregular-and-complex/tscatter.md) | Scatter tile elements by index | Scatter | All |
| [pto.tci](./ops/irregular-and-complex/tci.md) | Complex index operation | Index | All |
| [pto.ttri](./ops/irregular-and-complex/ttri.md) | Triangular matrix extraction/operation | Matrix | All |
| [pto.tpartadd](./ops/irregular-and-complex/tpartadd.md) | Partial addition | Reduce | All |
| [pto.tpartmul](./ops/irregular-and-complex/tpartmul.md) | Partial multiplication | Reduce | All |
| [pto.tpartmax](./ops/irregular-and-complex/tpartmax.md) | Partial maximum | Reduce | All |
| [pto.tpartmin](./ops/irregular-and-complex/tpartmin.md) | Partial minimum | Reduce | All |
| [pto.tquant](./ops/irregular-and-complex/tquant.md) | Quantize tile values into a lower-precision representation | Quantize | A2/A3, A5 |
| [pto.tdequant](./ops/irregular-and-complex/tdequant.md) | Dequantize integer tile values back to floating-point values | Quantize | A2/A3, A5 |
| [pto.trandom](./ops/irregular-and-complex/trandom.md) | Generate random values into tile state | Generation | A5 |
| [pto.thistogram](./ops/irregular-and-complex/thistogram.md) | Accumulate histogram bins from tile values | Statistics | A5 |

## Mechanism

### Sort (TMREGSORT, TSORT32)

Sort elements within each row. The sort order (ascending/descending) is specified by an attribute or parameter. `TSORT32` sorts 32-bit values; `TMREGSORT` performs a merging sort across tile rows.

### Gather/Scatter (TGATHER, TGATHERB, TSCATTER)

Gather reads from non-contiguous GM locations based on an index tile. Scatter writes to non-contiguous GM locations. Unlike `MGATHER`/`MSCATTER` which operate on tile buffers, these operations work with tile registers directly in UB.

$$ \mathrm{dst}_i = \mathrm{src}_{\mathrm{index}_i} \quad \text{(gather)} $$

$$ \mathrm{dst}_{\mathrm{index}_i} = \mathrm{src}_i \quad \text{(scatter)} $$

### Partial Reductions (TPARTADD, TPARTMUL, TPARTMAX, TPARTMIN)

Partial reductions compute intermediate results that are later combined across tiles. Unlike full row/column reductions, partial reductions produce tiles with reduced but non-singular extent — they divide the reduction axis into segments.

### Quantization (TQUANT, TDEQUANT)

`TQUANT` converts floating-point tile data into quantized representations. `TDEQUANT` converts quantized integer tile data back into floating-point values using row-broadcast scale and offset tiles. Both are tile payload transforms, so they belong here rather than in system scheduling.

### Generated and Statistical State

`TRANDOM` and `THISTOGRAM` create tile-visible payload state with algorithm-specific behavior. They are irregular tile operations because their result is tile data, not a scheduling effect.

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
| Quantized formats (INT4/FP4/NF4) | No | Yes | Yes |

## Constraints

- Sort operations require compatible element types (bit-width appropriate for the sort variant).
- Quantization requires valid scale (non-zero) and zero-point values within representable range.
- Scatter requires a valid index tile with non-negative indices within the destination bounds.
- Partial reductions may have different behavior across profiles.

## Cases That Are Not Allowed

- **MUST NOT** use quantization with invalid scale (zero or NaN) or out-of-range zero-point.
- **MUST NOT** scatter to indices outside the destination tile's declared shape bounds.
- **MUST NOT** use sort operations with element types incompatible with the sort variant (e.g., `TSORT32` on i8).

## Performance Notes

Irregular operations may have different performance characteristics compared to regular elementwise operations. Some backends may fall back to a sequence of simpler operations. Quantization operations on CPU simulator are emulated and may be significantly slower than hardware paths.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Sort (sorting order attribute: Ascending/Descending)
template <typename TileT>
PTO_INST RecordEvent TMREGSORT(TileT& dst, SortOrder order = SortOrder::Ascending);

template <typename TileT>
PTO_INST RecordEvent TSORT32(TileT& dst, SortOrder order = SortOrder::Ascending);

// Gather/Scatter
template <typename TileDst, typename TileIdx, typename TileSrc>
PTO_INST RecordEvent TGATHER(TileDst& dst, TileIdx& indices, TileSrc& src);

template <typename TileDst, typename TileIdx, typename TileSrc>
PTO_INST RecordEvent TSCATTER(TileDst& dst, TileIdx& indices, TileSrc& src);

// Quantization
template <typename TileDst, typename TileSrc, typename TileScale, typename TileZp>
PTO_INST RecordEvent TQUANT(TileDst& dst, TileSrc& src, TileScale& scale, TileZp& zp);

```

## See Also

- [Tile instruction set](../instruction-families/tile-families.md) — Instruction set overview
- [Tile instruction set](../instruction-families/tile-families.md) — Instruction Set description
