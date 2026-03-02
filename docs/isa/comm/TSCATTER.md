# TSCATTER

## Introduction

Scatter operation: the calling NPU (root) distributes data to all ranks in the parallel group by splitting the local source tensor along **DIM_3** (row dimension). This is the inverse of `TGATHER`.


Only the root needs to execute `TSCATTER`. Non-root ranks only need to ensure their destination buffers are allocated and writable for the duration of the operation. Calling `TSCATTER` on non-root ranks is undefined behavior.

**Large Tile Support**: When the per-rank data exceeds the UB tile capacity in rows and/or columns, the transfer is automatically chunked via 2D sliding.

## Math Interpretation

The local source tensor has shape $(D_0, D_1, D_2, N \times H, W)$, where $N$ is the number of ranks and each rank receives $H$ rows. After the operation:

$$\mathrm{dst}^{(r)}_{d_0, d_1, d_2,\; i,\; j} = \mathrm{src}^{\mathrm{local}}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../../grammar/PTO-AS.md).

Synchronous form:

```text
tscatter %group, %src : (!pto.group<...>, !pto.memref<...>)
```
Lowering introduces UB staging tile(s) for the GM→UB→GM data path; the C++ intrinsic requires explicit `stagingTileData` (or `pingTile` / `pongTile`) operand(s).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic scatter (single staging tile)
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &stagingTileData, WaitEvents&... events);

// Ping-pong scatter (double buffering with two staging tiles)
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
  - `srcGlobalData` must point to local memory (current NPU) and be large enough to hold data for all ranks. Specifically, `srcGlobalData.GetShape(DIM_3)` must be $\geq N \times H$ where $H$ is each rank's `GetShape(DIM_3)`.
  - If `srcGlobalData.GetShape(DIM_3) > N × H`, only the first `N × H` rows are read; remaining rows are ignored.
  - `stagingTileData` (or `pingTile` / `pongTile`) must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - `parallelGroup.tensors[r]` must refer to rank `r`'s destination buffer (remote GM as seen by the root).
  - `parallelGroup.GetRootIdx()` identifies the calling NPU as the scatter root.
  - All destination tensors are assumed to have the same shape and strides; behavior is undefined if they differ.
- **Chunked mode constraints** (when per-rank data exceeds a single UB tile):
  - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` of each rank's destination must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` ValidRow for partial row support.
  - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` ValidCol for partial column support.

## Examples

### Basic Scatter (Single Staging Tile)

Root has `NRANKS * ROWS` rows of width `COLS`. Each rank receives `ROWS × COLS`, split along DIM_3.
The tile size (`TILE_ROWS × TILE_COLS`) can be smaller than the per-rank data — when it is, the implementation automatically chunks the transfer along both DIM_3 and DIM_4 via 2D sliding.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::TSCATTER(group, srcG, stagingTile);
}
```

### Ping-Pong Scatter (Double Buffering)

Uses two UB tiles to overlap TLOAD of the next chunk (MTE2) with TSTORE of the current chunk (MTE3).

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter_pingpong(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    // Tile can be smaller than the data in both dimensions
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Ping-pong: overlaps TLOAD and TSTORE for better throughput
    comm::TSCATTER(group, srcG, pingTile, pongTile);
}
```
