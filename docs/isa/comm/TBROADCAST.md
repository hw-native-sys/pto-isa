# TBROADCAST

`TBROADCAST` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Broadcast data from the root NPU to all ranks in a parallel group. The calling NPU serves as the root; its data is replicated to every other NPU in the group.

Only the root executes the broadcast. Non-root ranks only ensure their destination buffers are allocated and writable. Executing `TBROADCAST` on a non-root rank has undefined behavior.

When the GlobalTensor exceeds the UB tile capacity, the transfer is automatically chunked via 2D sliding.

## Mechanism

`TBROADCAST` copies data from the root NPU's source buffer to the corresponding destination buffer on every other NPU in the parallel group. The data path uses UB as a staging area: GM → UB → GM.

For rank $k$ in a group of $N$ ranks, after the operation:

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

where `root` is the NPU that executes the broadcast.

## Syntax

### PTO Assembly Form

```text
tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

The assembly form takes a parallel group and a source memory reference. UB staging tiles are introduced during lowering; the C++ intrinsic exposes these explicitly.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic broadcast — single staging tile
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup,
                                GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData,
                                WaitEvents&... events);

// Ping-pong broadcast — two staging tiles for double buffering
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup,
                                GlobalSrcData &srcGlobalData,
                                TileData &pingTile,
                                TileData &pongTile,
                                WaitEvents&... events);
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `parallelGroup` | `ParallelGroup` | Parallel group descriptor; `GetRootIdx()` identifies the broadcast root |
|| `srcGlobalData` | `GlobalTensor` | Source data on the root NPU; must point to local GM |
|| `stagingTileData` | `Tile` | Staging tile in UB for the GM→UB→GM transfer path |
|| `pingTile` / `pongTile` | `Tile` | Two staging tiles for ping-pong double buffering |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the broadcast |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `RecordEvent` | event | Token signaling broadcast completion; depends on async variant |

## Side Effects

This operation reads from and writes to global memory across multiple NPUs. It establishes synchronization edges through the returned event token.

## Constraints

### Type constraints

- `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
- `TileData::DType` must equal `GlobalSrcData::RawDType`.

### Memory constraints

- `srcGlobalData` must point to local memory (the calling NPU's GM).
- `stagingTileData` (or `pingTile`/`pongTile`) must be pre-allocated in UB.

### Parallel group constraints

- `parallelGroup.tensors[k]` must refer to rank `k`'s destination buffer (remote GM as seen from the root).
- `parallelGroup.GetRootIdx()` identifies the calling NPU as the broadcast root.
- All destination tensors must have the same shape and strides.

### Chunked mode constraints

When the GlobalTensor exceeds a single UB tile in rows or columns:

- If `TileData` has a static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` `ValidRow` for partial row support.
- If `TileData` has a static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` `ValidCol` for partial column support.

## Target-Profile Restrictions

- Collective communication is supported on A2/A3 and A5 profiles. CPU simulation does not support collective operations.
- The ping-pong double-buffering form is recommended for large transfers to overlap communication with computation.
- `TBROADCAST` requires a properly initialized `ParallelGroup` covering all participating NPUs.

## Examples

### Basic broadcast

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

    // Root NPU broadcasts its data to all others
    comm::TBROADCAST(group, srcG, stagingTile);
}
```

### Ping-pong double buffering

Uses two UB staging tiles to overlap loading the next chunk with storing the current chunk:

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                     BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Overlaps TLOAD and TSTORE for better throughput
    comm::TBROADCAST(group, srcG, pingTile, pongTile);
}
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Collective operations: [TGET](./TGET.md), [TPUT](./TPUT.md), [TREDUCE](./TREDUCE.md), [TSCATTLER](./TSCATTER.md), [TGATHER](./TGATHER.md)
- Instruction set: [Other and Communication](../other/README.md)
- Machine model: [Ordering and Synchronization](../machine-model/ordering-and-synchronization.md)
