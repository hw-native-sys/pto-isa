# TREDUCE

`TREDUCE` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Collective reduction: gather data from all ranks in a parallel group and perform element-wise reduction locally. Only the root NPU executes `TREDUCE`; non-root ranks only ensure their source buffers are ready. Executing `TREDUCE` on a non-root rank has undefined behavior.

When the GlobalTensor exceeds the UB tile capacity, the reduction is automatically chunked via 2D sliding.

## Mechanism

`TREDUCE` gathers source data from all ranks in the parallel group and reduces it into the root NPU's destination buffer. For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j} $$

where $N$ is the number of ranks and $\oplus$ is the reduction operator.

### Supported reduction operators

|| Operator | Symbol | Notes |
||----------|--------|-------|
|| Sum | `Sum` | Additive reduction |
|| Max | `Max` | Maximum |
|| Min | `Min` | Minimum |

## Syntax

### PTO Assembly Form

```text
treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
treduce %group, %dst {op = #pto.reduce_op<Max>} : (!pto.group<...>, !pto.memref<...>)
```

Lowering introduces accumulator and receive tiles internally. The C++ intrinsic exposes these explicitly.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic reduce — accumulator + receive tile
template <typename ParallelGroupType, typename GlobalDstData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &recvTileData,
                              ReduceOp op, WaitEvents&... events);

// Ping-pong reduce — accumulator + ping + pong tiles for double buffering
template <typename ParallelGroupType, typename GlobalDstData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                              ReduceOp op, WaitEvents&... events);
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `parallelGroup` | `ParallelGroup` | Parallel group descriptor; `GetRootIdx()` identifies the reduce root |
|| `dstGlobalData` | `GlobalTensor` | Local destination buffer on the root NPU |
|| `accTileData` | `Tile` | Accumulator tile in UB for partial reduction |
|| `recvTileData` | `Tile` | Receive tile in UB for incoming remote data |
|| `pingTileData` / `pongTileData` | `Tile` | Two UB tiles for ping-pong double buffering |
|| `op` | `ReduceOp` | Reduction operator (`Sum`, `Max`, `Min`, etc.) |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the reduction |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `RecordEvent` | event | Token signaling reduction completion |

## Side Effects

This operation reads from all ranks' global memory and writes to the root's global memory. It establishes synchronization edges through the returned event token.

## Constraints

### Type constraints

- `ParallelGroup::value_type::RawDType` must equal `GlobalDstData::RawDType`.
- `TileData::DType` must equal `GlobalDstData::RawDType`.

### Memory constraints

- `dstGlobalData` must point to local address (on the root NPU).
- `accTileData` and `recvTileData` (or `accTileData`, `pingTileData`, `pongTileData`) must be pre-allocated in UB.

### Parallel group constraints

- `parallelGroup.tensors[r]` must refer to rank `r`'s source buffer (remote GM as seen from the root).
- `parallelGroup.GetRootIdx()` identifies the calling NPU as the reduce root.
- All source tensors must have the same shape and strides.

### Chunked mode constraints

When the GlobalTensor exceeds a single UB tile in rows or columns:

- If `TileData` has a static `ValidRow`, `GetShape(DIM_3)` must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` `ValidRow` for partial row support.
- If `TileData` has a static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` `ValidCol` for partial column support.

## Target-Profile Restrictions

- Collective communication is supported on A2/A3 and A5 profiles. CPU simulation does not support collective operations.
- Use ping-pong double buffering for large transfers to overlap communication with computation.
- `TREDUCE` requires a properly initialized `ParallelGroup` covering all participating NPUs.

## Examples

### Reduce sum

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                     BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### Reduce max

```cpp
template <typename T, int SIZE, int NRANKS>
void reduce_max(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                     BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Max);
}
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Collective operations: [TBROADCAST](./TBROADCAST.md), [TGATHER](./TGATHER.md), [TSCATTLER](./TSCATTER.md)
- Point-to-point: [TGET](./TGET.md), [TPUT](./TPUT.md)
- Instruction set: [Other and Communication](../other/README.md)
