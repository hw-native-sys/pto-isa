# TBROADCAST

## Introduction

Broadcast data from current NPU to all ranks in the parallel group. The calling NPU is the root and its data is copied to all other NPUs.

Only the root needs to execute `TBROADCAST`. Non-root ranks only need to ensure their destination buffers are allocated and writable for the duration of the operation. Calling `TBROADCAST` on non-root ranks is undefined behavior.

**Large Tile Support**: When the GlobalTensor exceeds the UB (Unified Buffer) tile capacity in rows and/or columns, the transfer is automatically chunked via 2D sliding.

## Math Interpretation

After the operation:

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

where $N$ is the number of ranks and `root` is the calling NPU.

## Template Parameter

- `engine`:
    - `CollEngine::AIV` (default)
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only)

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic broadcast (single staging tile)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... Args>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData, Args&... args);

// Ping-pong broadcast (double buffering with two staging tiles)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... Args>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &pingTile, TileData &pongTile, Args&... args);
```

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext` containing the CKE slot VA and gate mask. The AIV kernel triggers the CKE gate; the actual broadcast data path runs on the CCU engine.

## Constraints

- **Type constraints**:
    - `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
    - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
    - `srcGlobalData` must point to local memory (current NPU).
    - `stagingTileData` (or `pingTile` / `pongTile`) must be pre-allocated in UB.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[k]` must refer to rank `k`'s destination buffer (remote GM as seen by the root).
    - `parallelGroup.GetRootIdx()` identifies the calling NPU as the broadcast root.
    - All destination tensors are assumed to have the same shape and strides; behavior is undefined if they differ.
- **Chunked mode constraints** (when data exceeds a single UB tile):
    - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` ValidRow for partial row support.
    - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` ValidCol for partial column support.

> **CCU path**: Unlike the AIV path where only root calls `TBROADCAST`, the CCU path requires all ranks to register and launch the CCU kernel via host-side `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. See `tests/npu/a5/comm/st/testcase/tbroadcast_ccu/` for a complete example.

## Examples

### Basic Broadcast

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    // Tile dimensions can differ from tensor dimensions.
    // The 2D sliding chunked path automatically tiles both row and column.
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                 BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    // Current NPU broadcasts its data to all others
    comm::TBROADCAST(group, srcG, stagingTile);
}
```

### Ping-pong Broadcast (Double Buffering)

Uses two UB tiles to overlap TLOAD of the next chunk with TSTORE of the current chunk.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {

    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GPerRank srcG(my_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Ping-pong: overlaps TLOAD and TSTORE for better throughput
    comm::TBROADCAST(group, srcG, pingTile, pongTile);
}
```
