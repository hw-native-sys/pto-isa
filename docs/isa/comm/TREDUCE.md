# TREDUCE

## Introduction

Reduce operation: gather data from multiple remote NPUs and perform element-wise reduction locally. 


Only the root needs to execute `TREDUCE`. Non-root ranks only need to ensure their source buffers are ready and remain valid for the duration of the operation. Calling `TREDUCE` on non-root ranks is undefined behavior.

**Large Tile Support**: When the GlobalTensor exceeds the UB tile capacity in rows and/or columns, the reduction is automatically chunked via 2D sliding.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j} $$

where $N$ is the number of ranks and $\oplus$ is the reduction operation (sum, max, min, etc.).

## Template Parameter

- `engine`:
    - `CollEngine::AIV` (default)
    - `CollEngine::CCU` (Ascend950, NPU_ARCH 3510 only)

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic reduce (accumulator + receive tile)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, 
                              TileData &accTileData, TileData &recvTileData, ReduceOp op, Args&... args);

// Ping-pong reduce (accumulator + ping + pong tiles for double buffering)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                              ReduceOp op, Args&... args);
```

When `engine == CollEngine::CCU`, the first variadic argument must be a `CcuTriggerContext` containing the CKE slot VA and gate mask. The AIV kernel triggers the CKE gate; the actual reduce data path runs on the CCU engine.

## Constraints

- **Type constraints**:
    - `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
    - `TileData::DType` must equal `GlobalDstData::RawDType`.
- **Memory constraints**:
    - `dstGlobalData` must point to local address (on current NPU).
    - `accTileData`, `recvTileData` (or `accTileData`, `pingTileData`, `pongTileData`) must be pre-allocated UB tiles.
- **ParallelGroup constraints**:
    - `parallelGroup.tensors[r]` must refer to rank `r`'s source buffer (remote GM as seen by the root).
    - `parallelGroup.GetRootIdx()` identifies the calling NPU as the reduce root.
    - All source tensors are assumed to have the same shape and strides; behavior is undefined if they differ.
- **Chunked mode constraints** (when data exceeds a single UB tile):
    - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` ValidRow for partial row support.
    - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` ValidCol for partial column support.

> **CCU path**: Unlike the AIV path where only root calls `TREDUCE`, the CCU path requires all ranks to register and launch the CCU kernel via host-side `HcclCcuKernelRegister` / `HcclCcuKernelLaunch`. See `tests/npu/a5/comm/st/testcase/treduce_ccu/` for a complete example.

## Examples

### Basic Reduce Sum

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, 
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    // Stack-allocated tensors
    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }
    
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### Max Reduce

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_max(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>, 
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }
    
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Max);
}
```

### Ping-pong Reduce (Double Buffering)

Uses an accumulator tile plus two receive tiles (ping/pong) to overlap the TLOAD of the next remote rank with the vector reduction of the current one.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, pingTile, pongTile;

    comm::TREDUCE(group, dstG, accTile, pingTile, pongTile, comm::ReduceOp::Sum);
}
```
