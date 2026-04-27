# pto.tbroadcast

`pto.tbroadcast` is part of the [Collective Communication](communication-runtime.md) instruction set.

## Summary

Broadcast data from one NPU (the root) to all NPUs in a parallel group. Each rank receives an identical copy of the root's data.

## Mechanism

The broadcasting NPU (root) distributes its local data to every other NPU in the parallel group. The operation uses the parallel group to identify the set of participating ranks.

After the operation, for all ranks `k` in the parallel group:

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} $$

where `root` is identified by `parallelGroup.GetRootIdx()`.

The data path uses UB as a staging area: GM → local UB (root) → interconnect → remote UB (all ranks) → GM. A staging tile (or ping/pong tile pair for double buffering) provides the local buffer for this transfer.

**Large tile support**: When the GlobalTensor exceeds the UB tile capacity, the transfer is automatically chunked via 2D sliding.

## Assembly Syntax

```text
pto.tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

Lowering introduces UB staging tile(s) for the GM → UB → interconnect → GM data path. The C++ intrinsic requires explicit `stagingTileData` (or `pingTile` / `pongTile`) operand(s).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic broadcast (single staging tile)
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent BROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &stagingTileData, WaitEvents&... events);

// Ping-pong broadcast (double buffering with two staging tiles)
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent BROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `parallelGroup` | `ParallelGroup` identifying participating ranks and their destination buffers |
| `srcGlobalData` | Source GlobalTensor on the root NPU |
| `stagingTileData` | UB staging tile for the transfer |
| `pingTile`, `pongTile` | Pair of UB staging tiles for ping-pong double buffering |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | `RecordEvent` | Token signaling completion |

After the operation, each non-root rank has the root's data written to the buffer described by `parallelGroup.tensors[k]`.

## Side Effects

Participates in collective communication over the interconnect. Only the root rank needs to execute the operation; non-root ranks must ensure their destination buffers are allocated and writable.

## Constraints

- **Type constraints**:
  - `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
- **Memory constraints**:
  - `srcGlobalData` must point to local memory (current NPU).
  - Staging tiles must be pre-allocated in UB.
- **ParallelGroup constraints**:
  - `parallelGroup.tensors[k]` must refer to rank `k`'s destination buffer.
  - `parallelGroup.GetRootIdx()` identifies the broadcast root.
  - All destination tensors are assumed to have the same shape and strides.
- **Chunked mode constraints** (when data exceeds a single UB tile):
  - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a tile with `DYNAMIC` ValidRow for partial row support.
  - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a tile with `DYNAMIC` ValidCol for partial column support.

## Exceptions

- Non-root ranks calling `pto.tbroadcast` produces undefined behavior.
- Using incompatible tensor types or ranks is rejected by the verifier.
- Accessing a rank's destination buffer outside its declared shape is undefined.

## Examples

### Basic Broadcast

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
  using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
  using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                               BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

  GTensor tensors[NRANKS];
  for (int i = 0; i < NRANKS; ++i)
    tensors[i] = GTensor(group_addrs[i]);

  comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
  GTensor srcG(my_data);
  TileT stagingTile(TILE_ROWS, TILE_COLS);

  comm::BROADCAST(group, srcG, stagingTile);
}
```

### Ping-Pong Broadcast (Double Buffering)

```cpp
template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
  using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
  using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

  GPerRank tensors[NRANKS];
  for (int i = 0; i < NRANKS; ++i)
    tensors[i] = GPerRank(group_addrs[i]);

  comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
  GPerRank srcG(my_data);
  TileT pingTile(TILE_ROWS, TILE_COLS);
  TileT pongTile(TILE_ROWS, TILE_COLS);

  comm::BROADCAST(group, srcG, pingTile, pongTile);
}
```

## See Also

- Instruction set overview: [Collective Communication](communication-runtime.md)
