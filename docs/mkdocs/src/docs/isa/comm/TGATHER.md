<!-- Generated from `docs/isa/comm/TGATHER.md` -->

# TGATHER

## Introduction

Gather operation: the calling NPU (root) collects data from all ranks in the parallel group and concatenates the results along **DIM_3** (row dimension) into a local output buffer.


Only the root needs to execute `TGATHER`. Non-root ranks only need to ensure their source buffers are ready and remain valid for the duration of the operation. Calling `TGATHER` on non-root ranks is undefined behavior.

**Large Tile Support**: When the GlobalTensor exceeds the UB tile capacity in rows and/or columns, the transfer is automatically chunked via 2D sliding — the same mechanism used by other PTO-COMM instructions.

## Math Interpretation

Each rank $r$ has source data of shape $(D_0, D_1, D_2, H, W)$. The gather concatenates all $N$ ranks along DIM_3:

$$\mathrm{dst}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} = \mathrm{src}^{(r)}_{d_0, d_1, d_2,\; i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

The destination tensor has shape $(D_0, D_1, D_2, N \times H, W)$.

## Assembly Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
tgather %group, %dst : (!pto.group<...>, !pto.memref<...>)
```
Lowering introduces UB staging tile(s) for the GM→UB→GM data path; the C++ intrinsic requires explicit `stagingTileData` (or `pingTile` / `pongTile`) operand(s).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic gather (single staging tile)
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &stagingTileData, WaitEvents&... events);

// Ping-pong gather (double buffering with two staging tiles)
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
    - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
    - `dstGlobalData` must point to local memory (current NPU) and be large enough to hold the concatenated result from all ranks. Specifically, `dstGlobalData.GetShape(DIM_3)` must be $\geq N \times H$ where $H$ is each rank's `GetShape(DIM_3)`.
    - If `dstGlobalData.GetShape(DIM_3) > N × H`, only the first `N × H` rows are written; remaining rows are left unchanged.
    - `stagingTileData` (or `pingTile` / `pongTile`) must be pre-allocated in UB.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[r]` must refer to rank `r`'s source buffer (remote GM as seen by the root).
    - `parallelGroup.GetRootIdx()` identifies the calling NPU as the gather root.
    - All source tensors are assumed to have the same shape and strides; behavior is undefined if they differ.
- **Chunked mode constraints** (when source data exceeds a single UB tile):
    - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` of each rank's source must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` ValidRow for partial row support.
    - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` ValidCol for partial column support.

## Examples

### Basic Gather (Single Staging Tile)

Each rank contributes `ROWS × COLS` data. The root collects them into `NRANKS * ROWS` rows.
The tile size (`TILE_ROWS × TILE_COLS`) can be smaller than the per-rank data — when it is, the implementation automatically chunks the transfer along both DIM_3 and DIM_4 via 2D sliding.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void gather(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GResult = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::TGATHER(group, dstG, stagingTile);
}
```

### Ping-Pong Gather (Double Buffering)

Uses two UB tiles to overlap TLOAD of the next chunk (MTE2) with TSTORE of the current chunk (MTE3).

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void gather_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    // Tile can be smaller than the data in both dimensions
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GResult = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Ping-pong: overlaps TLOAD and TSTORE for better throughput
    comm::TGATHER(group, dstG, pingTile, pongTile);
}
```
