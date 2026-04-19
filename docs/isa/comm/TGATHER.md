# TGATHER

`TGATHER` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Collective gather: the root NPU collects data from all ranks in a parallel group and concatenates the results along DIM_3 (row dimension) into a local output buffer. Only the root executes `TGATHER`; non-root ranks only ensure their source buffers are ready. Executing `TGATHER` on a non-root rank has undefined behavior.

When the per-rank data exceeds the UB tile capacity, the transfer is automatically chunked via 2D sliding.

## Mechanism

Each rank $r$ contributes source data of shape $(D_0, D_1, D_2, H, W)$. The gather concatenates all $N$ ranks along DIM_3:

$$\mathrm{dst}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} = \mathrm{src}^{(r)}_{d_0, d_1, d_2,\; i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

The destination tensor has shape $(D_0, D_1, D_2, N \times H, W)$.

## Syntax

### PTO Assembly Form

```text
tgather %group, %dst : (!pto.group<...>, !pto.memref<...>)
```

UB staging tiles are introduced during lowering. The C++ intrinsic exposes them explicitly.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic gather — single staging tile
template <typename ParallelGroupType, typename GlobalDstData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &stagingTileData, WaitEvents&... events);

// Ping-pong gather — two staging tiles for double buffering
template <typename ParallelGroupType, typename GlobalDstData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `parallelGroup` | `ParallelGroup` | Parallel group descriptor; `GetRootIdx()` identifies the gather root |
|| `dstGlobalData` | `GlobalTensor` | Local destination buffer on the root NPU; must be large enough to hold concatenated data |
|| `stagingTileData` | `Tile` | UB staging tile for the GM→UB→GM transfer path |
|| `pingTile` / `pongTile` | `Tile` | Two UB staging tiles for ping-pong double buffering |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the gather |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `RecordEvent` | event | Token signaling gather completion |

## Side Effects

This operation reads from all ranks' global memory and writes to the root's global memory. It establishes synchronization edges through the returned event token.

## Constraints

### Type constraints

- `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
- `TileData::DType` must equal `GlobalDstData::RawDType`.

### Memory constraints

- `dstGlobalData` must point to local memory and be large enough to hold concatenated data from all ranks. Specifically, `dstGlobalData.GetShape(DIM_3)` must be $\geq N \times H$.
- If `dstGlobalData.GetShape(DIM_3) > N \times H`, only the first $N \times H$ rows are written; remaining rows are left unchanged.
- `stagingTileData` / `pingTile` / `pongTile` must be pre-allocated in UB.

### Parallel group constraints

- `parallelGroup.tensors[r]` must refer to rank `r`'s source buffer (remote GM as seen from the root).
- `parallelGroup.GetRootIdx()` identifies the calling NPU as the gather root.
- All source tensors must have the same shape and strides.

### Chunked mode constraints

When per-rank data exceeds a single UB tile in rows or columns:

- If `TileData` has a static `ValidRow`, each rank's source `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` `ValidRow` for partial row support.
- If `TileData` has a static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` `ValidCol` for partial column support.

## Target-Profile Restrictions

- Collective communication is supported on A2/A3 and A5 profiles. CPU simulation does not support collective operations.
- Use ping-pong double buffering for large transfers to overlap communication with computation.
- `TGATHER` requires a properly initialized `ParallelGroup` covering all participating NPUs.

## Examples

### Basic gather

Each rank contributes `ROWS × COLS` data. The root collects them into `NRANKS × ROWS` rows:

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
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::TGATHER(group, dstG, stagingTile);
}
```

### Ping-pong gather

```cpp
template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void gather_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                     BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GResult = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                     BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    comm::TGATHER(group, dstG, pingTile, pongTile);
}
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Inverse operation: [TSCATTLER](./TSCATTER.md)
- Collective operations: [TBROADCAST](./TBROADCAST.md), [TSCATTLER](./TSCATTER.md), [TREDUCE](./TREDUCE.md)
- Point-to-point: [TGET](./TGET.md), [TPUT](./TPUT.md)
- Instruction set: [Other and Communication](../other/README.md)
