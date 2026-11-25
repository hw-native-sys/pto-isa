# GlobalTensor Programming Model

`pto::GlobalTensor` models a tensor stored in global memory (GM). It is a lightweight wrapper around:

- a `__gm__` pointer, and
- a **5-D** shape and stride description

This metadata is consumed by memory instructions such as `TLOAD`, `TSTORE`, `MGATHER`, and `MSCATTER`.

All identifiers in this document refer to definitions in `include/pto/common/pto_tile.hpp` unless noted otherwise.

## GlobalTensor type

```cpp
template <typename Element_, typename Shape_, typename Stride_, pto::Layout Layout_ = pto::Layout::ND>
struct GlobalTensor;
```

- `Element_`: scalar element type stored in GM.
- `Shape_`: a `pto::Shape<...>` (up to 5 dimensions).
- `Stride_`: a `pto::Stride<...>` (up to 5 strides, in **elements**).
- `Layout_`: a layout *hint* (`ND`, `DN`, `NZ`, ...), used to guide lowering and target-specific fast paths.

The GM pointer type is `GlobalTensor::DType`, which is `__gm__ Element_`.

## Shapes and strides (5-D)

PTO represents global-memory tensors as 5-D objects. Most 2-D uses set the leading dimensions to `1` and use the last two dimensions for `(rows, cols)`.

### `pto::Shape`

`pto::Shape<N1, N2, N3, N4, N5>` stores 5 integers. Each template parameter can be a compile-time constant or `pto::DYNAMIC` (`-1`).

- Static dimensions are carried in the type via `Shape::staticShape[dim]`.
- Dynamic dimensions are stored in the runtime `Shape::shape[dim]` and are populated by the `Shape(...)` constructors.

The constructors enforce “number of runtime parameters equals number of dynamic dimensions” via `static_assert`, so mismatched construction fails at compile time.

### `pto::Stride`

`pto::Stride<S1, S2, S3, S4, S5>` follows the same pattern as `Shape`, but stores strides.

- Strides are expressed in **elements**, not bytes.
- A stride describes how many elements you skip when you increment a given dimension index by 1.

### `GlobalTensor` construction and access

`GlobalTensor` stores a pointer plus runtime shape/stride values for dynamic dimensions:

```cpp
using GT = pto::GlobalTensor<float, pto::Shape<1,1,1,-1,-1>, pto::Stride<1,1,1,-1,1>, pto::Layout::ND>;
GT t(ptr, /*shape=*/{rows, cols}, /*stride=*/{ld});

auto* p = t.data();
int cols = t.GetShape(pto::GlobalTensorDim::DIM_4);
int ld   = t.GetStride(pto::GlobalTensorDim::DIM_3);
```

For fully-static tensors you can also query compile-time values:

```cpp
constexpr int cols = GT::GetShape<pto::GlobalTensorDim::DIM_4>();
```

## Layout hints (`pto::Layout`)

`GlobalTensor` includes a layout enum (`ND`, `DN`, `NZ`, `SCALE`,  `MX_A_ZZ`, `MX_A_ND`, `MX_ADN`, `MX_B_NN`, `MX_B_ND`, `MX_B_DN`...). This is a *hint* that can enable target-specific fast paths.

Why this is not identical to Tile layout:

- Tile layout (`BLayout`/`SLayout`) is a 2-D concept (outer and optional inner boxed layout).
- GlobalTensor is 5-D, so a single “outer+inner” pair cannot describe all cases.

Instead, `Layout` tags common storage patterns (for example, `ND` vs `DN` for the minor 2-D ordering, and `NZ` for a common cube-friendly packing).

## 2-D helpers used by examples

Two helper families are commonly used for 2-D tensors:

- `pto::TileShape2D<T, rows, cols, layout>`: produces a `pto::Shape<1,1,1,rows,cols>` (or an `NZ`-specific shape when `layout == Layout::NZ`).
- `pto::BaseShape2D<T, rows, cols, layout>`: produces a `pto::Stride<...>` suitable for a base 2-D view (or an `NZ`-specific stride when `layout == Layout::NZ`).
  - layout in `pto::TileShape2D`, `pto::BaseShape2D` also supports `MX_A_ZZ`, `MX_A_ND`, `MX_ADN`, `MX_B_NN`, `MX_B_ND`, `MX_B_DN`.

Despite its name, `BaseShape2D` is a **stride** helper (it derives from `pto::Stride`).

## Address binding (`TASSIGN`)

`TASSIGN(globalTensor, ptr)` sets the underlying GM pointer of a `GlobalTensor`. The pointer type must match `GlobalTensor::DType` (enforced by `static_assert` in `TASSIGN_IMPL`).

## Minimal example

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example(__gm__ float* in, __gm__ float* out) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin(in);
  GT gout(out);

  TileT t;
  TLOAD(t, gin);
  TSTORE(gout, t);
}
```
