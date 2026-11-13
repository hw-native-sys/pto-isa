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

## Convolution configuration fields

A `ConvTile` object also carries runtime configuration fields consumed by convolution-related instructions.
By default, these fields describe ordinary convolution semantics: no padding, no dilation, stride 1, and no transpose.

| Field | Setter | Primary use |
| --- | --- | --- |
| `fmapH` / `fmapW` | `SetFmapH()` / `SetFmapW()` | FMATRIX source feature-map shape |
| `padList` | `SetPadListArray()` or per-element `SetPadList()` | left, right, top, and bottom padding |
| `filterH` / `filterW` | `SetFilterH()` / `SetFilterW()` | spatial filter size |
| `dilationH` / `dilationW` | `SetDilationH()` / `SetDilationW()` | filter dilation |
| `strideH` / `strideW` | `SetStrideH()` / `SetStrideW()` | IMG2COL convolution stride; the A5 `NCHW -> NC1HWC0` and `NCDHW -> NDC1HWC0` `TLOAD` paths also use these fields to express dx input expansion |
| `transpose` | `SetTranspose()` | IMG2COL ConvTile conversion; the A5 `TEXTRACT` / `TMOV` ConvTile paths also read this field. The A5 `TIMG2COL` `Right` path ignores it |
| `channelSize` | `SetChannelSize()` | IMG2COL channel configuration |
| `padValue` | `SetPadValue()` | padding value in automatic mode |
| `repeatStride` / `repeatTime` / `repeatMode` | corresponding `Set*` methods | IMG2COL repeat configuration |
| `dstStride` / `dstMposition` | corresponding `Set*` methods | A5 IMG2COL destination-placement configuration |

These fields do not change the physical shape declared by `ConvTileShape`. When expansion, padding, or transpose is required,
declare a shape and buffer size that still cover the actual physical storage.


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

### A5 backward-convolution dx input-expansion example

The function below loads a compact `NCHW` input into an expanded `NC1HWC0` ConvTile. With source H/W of 7 and
stride 2, the destination H/W is 13:

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

constexpr uint32_t H = 7, W = 7;
constexpr uint32_t strideH = 2, strideW = 2;
constexpr uint32_t expandedH = (H - 1) * strideH + 1;
constexpr uint32_t expandedW = (W - 1) * strideW + 1;

using DstTile = ConvTile<TileType::Mat, bfloat16_t, 1 * 2 * expandedH * expandedW * 16,
                         Layout::NC1HWC0, ConvTileShape<1, 2, expandedH, expandedW, 16>>;
using SrcTensor = GlobalTensor<
    bfloat16_t, Shape<1, 1, 32, H, W>,
    Stride<1 * 32 * H * W, 32 * H * W, H * W, W, 1>, Layout::NCHW>;

void example(__gm__ bfloat16_t* dyInput) {
  DstTile dy;
  SrcTensor src(dyInput);
  TASSIGN(dy, 0x0);

  // Initialize dy's backing storage first if the expansion holes must contain zero.
  dy.SetStrideH(strideH);
  dy.SetStrideW(strideW);
  TLOAD(dy, src);
}
