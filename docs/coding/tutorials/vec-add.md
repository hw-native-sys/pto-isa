# Tutorial: Vector Add (Tiled) + Ping-Pong (Manual)

This tutorial expands the vector add example into:

1. A tiled kernel over a larger 2-D tensor.
2. A manual-mode ping-pong (“double buffer”) structure that can overlap `TLOAD`, compute, and `TSTORE`.

For background on tiles, valid regions, and events, see:

- `docs/coding/Tile.md`
- `docs/coding/GlobalTensor.md`
- `docs/coding/Event.md`

## 1. Tiled vector add (Auto-style structure)

Assume a 2-D matrix `A[GRows, GCols]` and tile shape `(TRows, TCols)` where `GCols % TCols == 0` and `GRows % TRows == 0`.

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, int GRows, int GCols, int TRows, int TCols>
__global__ AICORE void VecAddTiledAuto(__gm__ T* out, __gm__ T* a, __gm__ T* b) {
  constexpr int tiles_per_row = GCols / TCols;
  const int tile_row = static_cast<int>(block_idx) / tiles_per_row;
  const int tile_col = static_cast<int>(block_idx) % tiles_per_row;
  const int base = tile_row * (GCols * TRows) + tile_col * TCols;

  using GT = GT2D<T, TRows, TCols>;
  using TileT = Tile<TileType::Vec, T, TRows, TCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  GT ga(a + base), gb(b + base), gout(out + base);
  TileT ta(TRows, TCols), tb(TRows, TCols), tc(TRows, TCols);

  TLOAD(ta, ga);
  TLOAD(tb, gb);
  TADD(tc, ta, tb);
  TSTORE(gout, tc);
}
```

## 2. Edge tiles (dynamic valid region)

If `GRows` or `GCols` is not divisible by the tile shape, use dynamic valid regions for the edge tiles:

- Compute `valid_rows` and `valid_cols` for each tile.
- Construct tiles with runtime valid region.

The exact “how to interpret outside-valid elements” behavior depends on the instruction; see `docs/isa/conventions.md`.

## 3. Manual ping-pong structure (conceptual)

Manual performance tuning often uses two sets of tiles:

- While buffer 0 is being computed/stored, buffer 1 is loaded, then you swap.

Below is a **conceptual** structure. Exact address choices and event wiring are platform-dependent.

```cpp
template <typename T, int TRows, int TCols, int NumTiles>
__global__ AICORE void VecAddPingPong(__gm__ T* out, __gm__ T* a, __gm__ T* b) {
  using GT = GT2D<T, TRows, TCols>;
  using TileT = Tile<TileType::Vec, T, TRows, TCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  TileT ta[2] = {TileT(TRows, TCols), TileT(TRows, TCols)};
  TileT tb[2] = {TileT(TRows, TCols), TileT(TRows, TCols)};
  TileT tc[2] = {TileT(TRows, TCols), TileT(TRows, TCols)};

#ifndef __PTO_AUTO__
  constexpr uint32_t kBufStride = 0x8000;
  constexpr uint32_t kA0 = 0x0000, kB0 = 0x2000, kC0 = 0x4000;
  for (int p = 0; p < 2; ++p) {
    TASSIGN(ta[p], kA0 + p * kBufStride);
    TASSIGN(tb[p], kB0 + p * kBufStride);
    TASSIGN(tc[p], kC0 + p * kBufStride);
  }
#endif

  for (int i = 0; i < NumTiles; ++i) {
    const int p = i & 1;
    GT ga(a + i * TRows * TCols), gb(b + i * TRows * TCols), gout(out + i * TRows * TCols);

#ifdef __CCE_AICORE__
    Event<Op::TLOAD, Op::TADD> e0;
    Event<Op::TADD, Op::TSTORE_VEC> e1;

    TLOAD(ta[p], ga);
    e0 = TLOAD(tb[p], gb);
    e1 = TADD(tc[p], ta[p], tb[p], e0);
    TSTORE(gout, tc[p], e1);
#else
    TLOAD(ta[p], ga);
    TLOAD(tb[p], gb);
    TADD(tc[p], ta[p], tb[p]);
    TSTORE(gout, tc[p]);
#endif
  }
}
```

This version still appears sequential in source form; real overlap comes from the device’s pipeline concurrency plus correct event ordering and buffer reuse.
