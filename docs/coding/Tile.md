# Tile Programming Model

PTO Tile Lib programs operate on **Tiles**: fixed-capacity 2-D buffers that are the unit of computation and the unit of most data movement for PTO instructions.

Conceptually, a Tile lives in **on-chip tile storage** (a register-file-like or SRAM-like storage) and is moved to/from global memory (GM) via `TLOAD`/`TSTORE`. On the CPU simulator backend, Tiles are stored in host memory, but the same shape/layout rules are preserved so code can be validated.

This document describes the C++ tile types in `include/pto/common/pto_tile.hpp` and their layout/valid-region constraints.

## What a Tile represents

A Tile is defined by five families of attributes:

- **Location**: which logical tile storage class the tile belongs to (vector vs matrix/cube registers).
- **Element type**: scalar element type (`float`, `half`, `int8_t`, ...).
- **Capacity shape**: compile-time `Rows × Cols` capacity.
- **Layout**: base layout (`BLayout`) and optional boxed/fractal layout (`SLayout`, `SFractalSize`).
- **Valid region**: how many rows/cols are meaningful for a specific operation (static or dynamic).

## `pto::Tile` type

Tiles are declared as a C++ template type:

```cpp
pto::Tile<
  pto::TileType Loc_,
  Element_,
  Rows_,
  Cols_,
  pto::BLayout BLayout_      = pto::BLayout::RowMajor,
  RowValid_                  = Rows_,
  ColValid_                  = Cols_,
  pto::SLayout SLayout_      = pto::SLayout::NoneBox,
  SFractalSize_              = pto::TileConfig::fractalABSize,
  pto::PadValue PadValue_    = pto::PadValue::Null
>;
```

### Location (`TileType`)

`TileType` encodes the logical/physical storage class of the tile and participates in overload selection and compile-time checks:

- `TileType::Vec`: vector tile storage (UB / vector pipeline).
- `TileType::Mat`: general matrix tile storage (Matrix L1).
- `TileType::Left`, `TileType::Right`: matrix-multiply operand tiles (Matrix L0A/L0B).
- `TileType::Acc`: matrix-multiply accumulator tiles.
- `TileType::Bias`, `TileType::Scaling`: auxiliary tiles for some matmul/move paths.

Instruction pages in `docs/isa/` specify which locations are legal for each instruction.

### Capacity shape (`Rows_`, `Cols_`)

`Rows_` and `Cols_` define the **static capacity** of the tile object. Most instructions require static shapes so they can be specialized and optimized at compile time.

### Valid region (`RowValid_`, `ColValid_`)

Tiles have a **valid region** `(valid_row, valid_col)` that defines which elements are meaningful.

- If `RowValid_ == Rows_` and `ColValid_ == Cols_`, the valid region is fully static.
- If either is `pto::DYNAMIC` (`-1`), the valid value is stored in the tile object and queried via `GetValidRow()` / `GetValidCol()`.

For a tile `t`, the valid region is always a *contiguous prefix*:

- Valid indices satisfy `0 <= i < t.GetValidRow()` and `0 <= j < t.GetValidCol()`.
- Elements outside the valid region are **unspecified** unless the instruction explicitly defines padding/behavior.

Instruction docs generally interpret semantics “for each element in the valid region”. The exact domain can be instruction-specific (some ops define the domain based on a source tile), so use `docs/isa/*` as the authoritative reference.

### Layout (`BLayout`, `SLayout`, `SFractalSize`)

PTO models layout with two knobs:

- **Base layout** `BLayout` (`RowMajor`/`ColMajor`): the outer (unboxed) matrix interpretation.
- **Boxed/fractal layout** `SLayout` (`NoneBox`, `RowMajor`, `ColMajor`): whether the tile is internally partitioned into fixed-size “base tiles” (also called *fractals*).
- **Base-tile size** `SFractalSize`: the byte size of a base tile. PTO Tile Lib currently uses:
  - `TileConfig::fractalABSize = 512` bytes (common for A/B operand tiles)
  - `TileConfig::fractalCSize = 1024` bytes (common for accumulator tiles)

#### Why boxed/fractal layout exists

Some matrix engines have preferred access patterns that operate on fixed-size base tiles. Boxed/fractal layout expresses this requirement explicitly so that:

- The compiler can choose legal layouts and shapes early.
- The runtime can avoid slow “fixup” paths.
- The same source code can map to different hardware generations with different micro-constraints.

#### Example: 512-byte base tiles (illustrative)

When `SFractalSize == 512` and the inner boxed layout is row-major, a common set of base tile shapes is:

- `fp32`: `16 × 8`  (16 * 8 * 4 bytes = 512 bytes)
- `fp16`: `16 × 16` (16 * 16 * 2 bytes = 512 bytes)
- `int8/fp8`: `16 × 32` (16 * 32 * 1 byte = 512 bytes)

For an inner col-major boxed layout, the same base tile can be viewed as the transpose (e.g., `8 × 16`, `16 × 16`, `32 × 16`).

Exact details are backend-dependent; the instruction constraints and compile-time checks in `pto::Tile` are the source of truth for what is legal for a given build target.

### Padding (`PadValue`)

`PadValue` is a compile-time policy used by some implementations when handling elements outside the valid region (for example, select/copy/pad paths). Its effect is instruction- and backend-dependent.

## Conceptual constraints (programmer model)

In addition to compile-time layout checks, PTO programs typically rely on these conceptual constraints:

- Tiles are **2-D** objects (matrix-shaped).
- Tiles are the smallest scheduling/data-movement unit: operations consume/produce whole tiles (not sub-tiles).
- Tile capacity shape (`Rows`, `Cols`) is intended to be **static**; the valid region may be static or dynamic.
- Real hardware commonly constrains tile sizes to a device-specific range (often on the order of **hundreds of bytes to tens of kilobytes**). CPU simulation is more permissive, but portable kernels should respect target limits.

## Compile-time constraints (alignment and boxing)

`pto::Tile` enforces a set of layout constraints with `static_assert`:

- For **unboxed row-major** tiles: `Cols * sizeof(Element)` must be a multiple of `TileConfig::alignedSize` (32 bytes).
- For **unboxed col-major** tiles: `Rows * sizeof(Element)` must be a multiple of 32 bytes.
- For **boxed** tiles: the shape must be compatible with the base-tile dimensions implied by `(SLayout, SFractalSize)` (with a small exception for some `Vec` tiles).

These constraints are intentional: they prevent generating programs that would be illegal or inefficient on real hardware.

## Common aliases

`include/pto/common/pto_tile.hpp` provides convenience aliases for matmul-related tiles:

- `pto::TileLeft<Element, Rows, Cols>`
- `pto::TileRight<Element, Rows, Cols>`
- `pto::TileAcc<Element, Rows, Cols>`

These aliases select target-appropriate boxed layouts and fractal sizes. For example, on the CPU simulator backend:

- `TileLeft`: outer col-major + inner row-major (often referred to as “Nz”)
- `TileRight`: outer row-major + inner col-major (often referred to as “Zn”)
- `TileAcc`: accumulator layout with `TileConfig::fractalCSize`

## Address binding (`TASSIGN`)

In manual placement flows, `TASSIGN(tile, addr)` binds a tile object to an implementation-defined address. In auto flows, `TASSIGN(tile, addr)` may be a no-op depending on build configuration.

See `docs/isa/TASSIGN.md` for details.

## Examples

### Basic vector tiles

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  TADD(c, a, b);
}
```

### Static valid region (mask)

```cpp
using TileT = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor,
                        127 /*row_valid*/, 127 /*col_valid*/,
                        pto::SLayout::NoneBox, pto::TileConfig::fractalABSize,
                        pto::PadValue::Zero>;
```

### Dynamic valid region (mask)

```cpp
using TileT = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor,
                        pto::DYNAMIC /*row_valid*/, 127 /*col_valid*/>;

TileT t(/*row_valid_runtime=*/m);
```
