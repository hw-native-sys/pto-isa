# TSCATTER

`TSCATTER` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Collective scatter: the root NPU distributes data to all ranks in a parallel group by splitting the local source tensor along DIM_3 (row dimension). This is the inverse of `TGATHER`. Only the root executes `TSCATTER`; non-root ranks only ensure their destination buffers are allocated. Executing `TSCATTER` on a non-root rank has undefined behavior.

When per-rank data exceeds the UB tile capacity, the transfer is automatically chunked via 2D sliding.

## Mechanism

The local source tensor has shape $(D_0, D_1, D_2, N \times H, W)$, where $N$ is the number of ranks and each rank receives $H$ rows. After the operation:

$$\mathrm{dst}^{(r)}_{d_0, d_1, d_2,\; i,\; j} = \mathrm{src}^{\mathrm{local}}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

## Syntax

### PTO Assembly Form

```text
tscatter %group, %src : (!pto.group<...>, !pto.memref<...>)
```

UB staging tiles are introduced during lowering. The C++ intrinsic exposes them explicitly.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic scatter — single staging tile
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &stagingTileData, WaitEvents&... events);

// Ping-pong scatter — two staging tiles for double buffering
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `parallelGroup` | `ParallelGroup` | Parallel group descriptor; `GetRootIdx()` identifies the scatter root |
|| `srcGlobalData` | `GlobalTensor` | Local source buffer on the root NPU; must contain data for all ranks |
|| `stagingTileData` | `Tile` | UB staging tile for the GM→UB→GM transfer path |
|| `pingTile` / `pongTile` | `Tile` | Two UB staging tiles for ping-pong double buffering |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the scatter |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `RecordEvent` | event | Token signaling scatter completion |

## Side Effects

This operation reads from the root's global memory and writes to all ranks' global memory. It establishes synchronization edges through the returned event token.

## Constraints

### Type constraints

- `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
- `TileData::DType` must equal `GlobalSrcData::RawDType`.

### Memory constraints

- `srcGlobalData` must point to local memory and be large enough to hold data for all ranks. Specifically, `srcGlobalData.GetShape(DIM_3)` must be $\geq N \times H$.
- If `srcGlobalData.GetShape(DIM_3) > N \times H`, only the first $N \times H$ rows are read; remaining rows are ignored.
- `stagingTileData` / `pingTile` / `pongTile` must be pre-allocated in UB.

### Parallel group constraints

- `parallelGroup.tensors[r]` must refer to rank `r`'s destination buffer (remote GM as seen from the root).
- `parallelGroup.GetRootIdx()` identifies the calling NPU as the scatter root.
- All destination tensors must have the same shape and strides.

### Chunked mode constraints

When per-rank data exceeds a single UB tile in rows or columns:

- If `TileData` has a static `ValidRow`, each rank's destination `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` `ValidRow` for partial row support.
- If `TileData` has a static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` `ValidCol` for partial column support.

## Target-Profile Restrictions

- Collective communication is supported on A2/A3 and A5 profiles. CPU simulation does not support collective operations.
- Use ping-pong double buffering for large transfers to overlap communication with computation.
- `TSCATTER` requires a properly initialized `ParallelGroup` covering all participating NPUs.

## Examples

### Basic scatter

Root has `NRANKS × ROWS` rows of width `COLS`. Each rank receives `ROWS × COLS`:

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
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::TSCATTER(group, srcG, stagingTile);
}
```

### Ping-pong scatter

```cpp
template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter_pingpong(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                     BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                     BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    comm::TSCATTER(group, srcG, pingTile, pongTile);
}
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Inverse operation: [TGATHER](./TGATHER.md)
- Collective operations: [TBROADCAST](./TBROADCAST.md), [TGATHER](./TGATHER.md), [TREDUCE](./TREDUCE.md)
- Point-to-point: [TGET](./TGET.md), [TPUT](./TPUT.md)
- Instruction set: [Other and Communication](../other/README.md)
