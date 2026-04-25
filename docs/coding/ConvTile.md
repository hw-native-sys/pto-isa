# ConvTile Programming Model

PTO Lib programs operate on **ConvTile**: fixed-capacity 2-D to 6-D buffers that are the unit of computation and the unit of most data movement for PTO convolution operation.

Conceptually, a ConvTile lives in **on-chip tile storage** (a register-file-like or SRAM-like storage) and is moved to/from global memory (GM) via `TLOAD`/`TSTORE`.

This document describes the C++ tile types in `include/pto/common/pto_tile.hpp` and their layout/valid-region constraints.

## What a ConvTile represents

A ConvTile is defined by five families of attributes:

- **Location**: which logical tile storage class the tile belongs to (matrix/cube registers).
- **Element type**: scalar element type (`float`, `half`, `int8_t`, ...).
- **Buffer size**: the static space of convtile.
- **Layout**: a layout (`NCHW`, `NHWC`, `NC1HWC0`, ...), used to guide lowering and target-specific fast paths.
- **Shape**: a `pto::ConvTileShape<...>` (up to 6 dimensions).

## `pto::ConvTile` type

Tiles are declared as a C++ template type:

```cpp
pto::ConvTile<
  pto::TileType Loc_,
  Element_,
  BufferSize_,
  pto::Layout_ layout,
  pto::ConvTileShape Shape_
>;
```

### Location (`TileType`)

`TileType` encodes the logical/physical storage class of the tile and participates in overload selection and compile-time checks:

- `TileType::Vec`: vector tile storage (UB / vector pipeline).
- `TileType::Mat`: general matrix tile storage (Matrix L1).


Instruction pages in `docs/isa/` specify which locations are legal for each instruction.

### Capacity (`BufferSize_`)

`BufferSize_` define the **static capacity** of the tile object. Most instructions require static shapes so they can be specialized and optimized at compile time.

### Layout (`pto::Layout`)

`ConvTile` includes a layout enum (`NCHW`, `NHWC`, `NC1HWC0`, `FRACTAL_Z`,  `FRACTAL_Z_S16S8`...).


### Shape (`pto::Shape`)

`pto::ConvTileShape<...Shapes>` support 1-6 integers. it is a template parameter list, each template parameter can be a compile-time constant or `pto::DYNAMIC` (`-1`).

- Static dimensions are carried in the type via `ConvTileShape::staticShape[dim]`.
- Dynamic dimensions are stored in the runtime `ConvTileShape::shape[dim]` and are populated by the `ConvTileShape(...)` constructors.

The constructors enforce “number of runtime parameters equals number of dynamic dimensions” via `static_assert`, so mismatched construction fails at compile time.


## Address binding (`TASSIGN`)

In manual placement flows, `TASSIGN(tile, addr)` binds a convtile object to an implementation-defined address. In auto flows, `TASSIGN(tile, addr)` may be a no-op depending on build configuration.

See `docs/isa/TASSIGN.md` for details.

## Minimal example

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example(__gm__ half* in, __gm__ half* out) {
  using TileT = ConvTile<TileType::Mat, half, 4096, Layout::NC1HWC0, pto::ConvTileShape<1, 1, 16, 16, 16>>;
  using GShape = Shape<1, 1, 16, 16, 16>;
  using GStride = Stride<1 * 16* 16* 16, 16* 16* 16, 16 * 16, 16, 1>;
  using GT = GlobalTensor<half, GShape, GStride, Layout::NC1HWC0>;
  GT gin(in);

  TileT tile5d;
  TASSIGN(tile5d, 0x0);

  TLOAD(tile5d, gin);
}
```
