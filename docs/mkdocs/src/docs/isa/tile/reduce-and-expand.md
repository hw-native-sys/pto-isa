<!-- Generated from `docs/isa/tile/reduce-and-expand.md` -->

# Reduce And Expand Family

Reduce operations collapse a 2D tile along one axis into a 1D result (or a tile with reduced extent along that axis). Expand operations broadcast a 1D tile along one axis to produce a 2D tile.

## Operations

### Reduce (Row)

| Operation | Description | C++ Intrinsic |
|-----------|-------------|----------------|
| [pto.trowsum](./ops/reduce-and-expand/trowsum.md) | Sum reduction along rows | `TROWSUM(dst, src, tmp)` |
| [pto.trowprod](./ops/reduce-and-expand/trowprod.md) | Product reduction along rows | `TROWPROD(dst, src, tmp)` |
| [pto.trowmax](./ops/reduce-and-expand/trowmax.md) | Maximum reduction along rows | `TROWMAX(dst, src, tmp)` |
| [pto.trowmin](./ops/reduce-and-expand/trowmin.md) | Minimum reduction along rows | `TROWMIN(dst, src, tmp)` |
| [pto.trowargmax](./ops/reduce-and-expand/trowargmax.md) | Index of maximum along rows | `TROWARGMAX(dst, src, tmp)` |
| [pto.trowargmin](./ops/reduce-and-expand/trowargmin.md) | Index of minimum along rows | `TROWARGMIN(dst, src, tmp)` |

### Reduce (Column)

| Operation | Description | C++ Intrinsic |
|-----------|-------------|----------------|
| [pto.tcolsum](./ops/reduce-and-expand/tcolsum.md) | Sum reduction along columns | `TCOLSUM(dst, src)` |
| [pto.tcolprod](./ops/reduce-and-expand/tcolprod.md) | Product reduction along columns | `TCOLPROD(dst, src)` |
| [pto.tcolmax](./ops/reduce-and-expand/tcolmax.md) | Maximum reduction along columns | `TCOLMAX(dst, src)` |
| [pto.tcolmin](./ops/reduce-and-expand/tcolmin.md) | Minimum reduction along columns | `TCOLMIN(dst, src)` |
| [pto.tcolargmax](./ops/reduce-and-expand/tcolargmax.md) | Index of maximum along columns | `TCOLARGMAX(dst, src, tmp)` |
| [pto.tcolargmin](./ops/reduce-and-expand/tcolargmin.md) | Index of minimum along columns | `TCOLARGMIN(dst, src, tmp)` |

### Expand (Row)

| Operation | Description | C++ Intrinsic |
|-----------|-------------|----------------|
| [pto.trowexpand](./ops/reduce-and-expand/trowexpand.md) | Expand row scalar to full tile | `TROWEXPAND(dst, src)` |
| [pto.trowexpandadd](./ops/reduce-and-expand/trowexpandadd.md) | Expand row and add | `TROWEXPANDADD(dst, src0, src1)` |
| [pto.trowexpandsub](./ops/reduce-and-expand/trowexpandsub.md) | Expand row and subtract | `TROWEXPSUB(dst, src0, src1)` |
| [pto.trowexpandmul](./ops/reduce-and-expand/trowexpandmul.md) | Expand row and multiply | `TROWEXPMUL(dst, src0, src1)` |
| [pto.trowexpanddiv](./ops/reduce-and-expand/trowexpanddiv.md) | Expand row and divide | `TROWEXPDIV(dst, src0, src1)` |
| [pto.trowexpandmax](./ops/reduce-and-expand/trowexpandmax.md) | Expand row and max | `TROWEXPANDMAX(dst, src0, src1)` |
| [pto.trowexpandmin](./ops/reduce-and-expand/trowexpandmin.md) | Expand row and min | `TROWEXPANDMIN(dst, src0, src1)` |
| [pto.trowexpandexpdif](./ops/reduce-and-expand/trowexpandexpdif.md) | Expand with exponential difference | `TROWEXPDIF(dst, src0, src1)` |

### Expand (Column)

| Operation | Description | C++ Intrinsic |
|-----------|-------------|----------------|
| [pto.tcolexpand](./ops/reduce-and-expand/tcolexpand.md) | Expand column scalar to full tile | `TCOLEXPAND(dst, src)` |
| [pto.tcolexpandadd](./ops/reduce-and-expand/tcolexpandadd.md) | Expand column and add | `TCOLEXPANDADD(dst, src0, src1)` |
| [pto.tcolexpandsub](./ops/reduce-and-expand/tcolexpandsub.md) | Expand column and subtract | `TCOLEXPSUB(dst, src0, src1)` |
| [pto.tcolexpandmul](./ops/reduce-and-expand/tcolexpandmul.md) | Expand column and multiply | `TCOLEXPMUL(dst, src0, src1)` |
| [pto.tcolexpanddiv](./ops/reduce-and-expand/tcolexpanddiv.md) | Expand column and divide | `TCOLEXPDIV(dst, src0, src1)` |
| [pto.tcolexpandmax](./ops/reduce-and-expand/tcolexpandmax.md) | Expand column and max | `TCOLEXPANDMAX(dst, src0, src1)` |
| [pto.tcolexpandmin](./ops/reduce-and-expand/tcolexpandmin.md) | Expand column and min | `TCOLEXPANDMIN(dst, src0, src1)` |
| [pto.tcolexpandexpdif](./ops/reduce-and-expand/tcolexpandexpdif.md) | Expand with exponential difference | `TCOLEXPDIF(dst, src0, src1)` |

## Mechanism

### Reduce

For each row `r`, reduce along the column axis:

$$ \mathrm{dst}_r = \bigoplus_{c=0}^{C-1} \mathrm{src}_{r,c} $$

For each column `c`, reduce along the row axis:

$$ \mathrm{dst}_c = \bigoplus_{r=0}^{R-1} \mathrm{src}_{r,c} $$

where $\bigoplus$ is the reduction operator (sum, max, min, prod).

### Expand

Expand takes a 1D tile of shape `(R)` or `(C)` and broadcasts it to a 2D tile of shape `(R, C)`:

$$ \mathrm{dst}_{r,c} = \mathrm{src}_r \quad \text{(row expand)} $$

$$ \mathrm{dst}_{r,c} = \mathrm{src}_c \quad \text{(column expand)} $$

Expand variants combine the broadcast with an elementwise operation using a second source tile:

$$ \mathrm{dst}_{r,c} = \mathrm{src0}_{r,c} \;\oplus\; \mathrm{src1}_r \quad \text{(row expand with op)} $$

## Output Shape

| Operation | Input Shape | Output Shape |
|-----------|-------------|-------------|
| Row reduce | `(R, C)` | `(R, 1)` |
| Column reduce | `(R, C)` | `(1, C)` |
| Row expand | `(R, 1)` | `(R, C)` |
| Column expand | `(1, C)` | `(R, C)` |

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

- The source tile's valid region determines the reduction domain.
- Arg variants (`*_argmax`, `*_argmin`) produce an **integer index tile**, not a numeric value tile.
- The destination tile for reduce operations has extent `1` along the reduced axis.
- Expand variants require a second source tile with shape `(R)` or `(C)` matching the expand axis.
- Exp-diff variants compute: `dst = exp(src0 - src1)` — used for softmax-style reductions.

## Cases That Are Not Allowed

- **MUST NOT** reduce along an axis with zero extent.
- **MUST NOT** use arg variants with non-numeric element types.
- **MUST NOT** use expand variants with mismatched expand-axis lengths.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Row reduce (requires temporary tile)
template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMAX(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMIN(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

// Column reduce
template <typename TileDst, typename TileSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDst& dst, TileSrc& src, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDst& dst, TileSrc& src, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDst& dst, TileSrc& src, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLMIN(TileDst& dst, TileSrc& src, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

template <typename TileDst, typename TileSrc, typename TileTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMIN(TileDst& dst, TileSrc& src, TileTmp& tmp, WaitEvents&... events);

// Row expand
template <typename TileDst, typename TileSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDst& dst, TileSrc& src, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPSUB(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPMUL(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPDIV(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPDIF(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

// Column expand
template <typename TileDst, typename TileSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDst& dst, TileSrc& src, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDADD(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPSUB(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPMUL(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPDIV(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMAX(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMIN(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPDIF(TileDst& dst, TileSrc0& src0, TileSrc1& src1, WaitEvents&... events);
```

## See Also

- [Tile families](../instruction-families/tile-families.md) — Family overview
- [Tile instruction surface](../instruction-surfaces/tile-instructions.md) — Surface description
