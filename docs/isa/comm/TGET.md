# TGET

`TGET` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Remote read operation: copy data from a remote NPU's global memory to the local NPU's global memory. Data traverses a UB staging tile as an intermediate buffer. When the GlobalTensor exceeds the UB tile capacity, TGET automatically chunks the transfer via 2D sliding.

Only the local NPU executes TGET; the remote NPU is passive.

## Mechanism

`TGET` reads from a remote NPU's global memory and writes to the local NPU's global memory. The data path is: remote GM → staging tile (UB) → local GM.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

The data flow in full:

```
srcGlobalData (remote GM) → stagingTileData (UB) → dstGlobalData (local GM)
```

## Syntax

### PTO Assembly Form

```text
tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```

UB staging tiles are introduced during lowering. The C++ intrinsic exposes them explicitly.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Single-tile form — auto-chunking for large tensors
template <typename GlobalDstData, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);

// Ping-pong double buffering form
template <typename GlobalDstData, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `dstGlobalData` | `GlobalTensor` | Local destination; must point to local GM |
|| `srcGlobalData` | `GlobalTensor` | Remote source; must point to remote NPU's GM |
|| `stagingTileData` | `Tile` | UB staging tile for the GM→UB→GM transfer path |
|| `pingTile` / `pongTile` | `Tile` | Two UB staging tiles for ping-pong double buffering |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the get |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `RecordEvent` | event | Token signaling completion of the remote read |

## Side Effects

This operation reads from remote global memory and writes to local global memory. It establishes synchronization edges through the returned event token.

## Constraints

### Type constraints

- `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
- `TileData::DType` must equal `GlobalSrcData::RawDType`.
- `GlobalSrcData::layout` must equal `GlobalDstData::layout`.

### Memory constraints

- `srcGlobalData` must point to a remote address (on the source NPU).
- `dstGlobalData` must point to a local address (on the current NPU).
- `stagingTileData` / `pingTile` / `pongTile` must be pre-allocated in UB.

### Transfer constraints

- Transfer size is determined by the `GlobalTensor` shape; auto-chunking tiles data to fit the UB staging buffer.
- When auto-chunking, rows (DIM_3) and columns (DIM_4) are subdivided as needed.

### Ping-pong constraints

- `pingTile` and `pongTile` must have identical type and dimensions.
- They must reside at non-overlapping UB offsets.

## Target-Profile Restrictions

- Point-to-point communication is supported on A2/A3 and A5 profiles. CPU simulation does not support remote memory access.
- Use ping-pong double buffering when transferring large tensors to overlap consecutive transfers.
- `TGET` requires a valid remote GM address; the remote NPU must have the corresponding memory region allocated.

## Examples

### Basic remote read

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tget(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    using GShape = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(remote_addr);
    GTensor dstG(local_data);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    comm::TGET(dstG, srcG, stagingTile);
}
```

### Ping-pong double buffering

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // Non-overlapping UB offsets

// Overlaps TLOAD[i+1] with TSTORE[i] for better pipeline utilization
comm::TGET(dstG, srcG, pingTile, pongTile);
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Inverse operation: [TPUT](./TPUT.md)
- Collective operations: [TBROADCAST](./TBROADCAST.md), [TGATHER](./TGATHER.md), [TSCATTLER](./TSCATTER.md), [TREDUCE](./TREDUCE.md)
- Instruction set: [Other and Communication](../other/README.md)
