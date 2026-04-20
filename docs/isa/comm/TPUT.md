# TPUT

`TPUT` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Remote write operation: copy data from the local NPU's global memory to a remote NPU's global memory. Data traverses a UB staging tile as an intermediate buffer. When the GlobalTensor exceeds the UB tile capacity, TPUT automatically chunks the transfer via 2D sliding.

Only the local NPU executes TPUT; the remote NPU is passive.

## Mechanism

`TPUT` reads from local global memory and writes to remote global memory. The data path is: local GM → staging tile (UB) → remote GM.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{remote}}_{i,j} = \mathrm{src}^{\mathrm{local}}_{i,j} $$

The data flow in full:

```
srcGlobalData (local GM) → stagingTileData (UB) → dstGlobalData (remote GM)
```

## Syntax

### PTO Assembly Form

```text
tput %dst_remote, %src_local : (!pto.memref<...>, !pto.memref<...>)
```

UB staging tiles are introduced during lowering. The C++ intrinsic exposes them explicitly.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Single-tile form — auto-chunking for large tensors
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);

// Ping-pong double buffering form
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);

// Runtime atomic type selection
template <typename GlobalDstData, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, AtomicType atomicType, WaitEvents&... events);
```

### Atomic types

|| Value | Behavior |
||-------|----------|
|| `AtomicType::AtomicNone` | Direct write, no atomic semantics |
|| `AtomicType::AtomicAdd` | Atomically add source value to destination |

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `dstGlobalData` | `GlobalTensor` | Remote destination; must point to target NPU's GM |
|| `srcGlobalData` | `GlobalTensor` | Local source; must point to current NPU's GM |
|| `stagingTileData` | `Tile` | UB staging tile for the GM→UB→GM transfer path |
|| `pingTile` / `pongTile` | `Tile` | Two UB staging tiles for ping-pong double buffering |
|| `atomicType` | `AtomicType` | Atomic operation mode (optional; defaults to `AtomicNone`) |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the put |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `RecordEvent` | event | Token signaling completion of the remote write |

## Side Effects

This operation reads from local global memory and writes to remote global memory. It establishes synchronization edges through the returned event token.

## Constraints

### Type constraints

- `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
- `TileData::DType` must equal `GlobalSrcData::RawDType`.
- `GlobalSrcData::layout` must equal `GlobalDstData::layout`.

### Memory constraints

- `dstGlobalData` must point to a remote address (on the target NPU).
- `srcGlobalData` must point to a local address (on the current NPU).
- `stagingTileData` / `pingTile` / `pongTile` must be pre-allocated in UB.

### Transfer constraints

- Transfer size is determined by the `GlobalTensor` shape; auto-chunking tiles data to fit the UB staging buffer.
- When auto-chunking, rows (DIM_3) and columns (DIM_4) are subdivided as needed.

### Atomic constraints

- `atomicType` supports `AtomicNone` and `AtomicAdd`.

### Ping-pong constraints

- `pingTile` and `pongTile` must have identical type and dimensions.
- They must reside at non-overlapping UB offsets.

## Target-Profile Restrictions

- Point-to-point communication is supported on A2/A3 and A5 profiles. CPU simulation does not support remote memory access.
- Use ping-pong double buffering when transferring large tensors to overlap consecutive transfers.
- `TPUT` requires a valid remote GM address; the remote NPU must have the corresponding memory region allocated.
- `AtomicAdd` is useful for distributed accumulation patterns.

## Examples

### Basic remote write

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tput(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    using GShape = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(local_data);
    GTensor dstG(remote_addr);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    comm::TPUT(dstG, srcG, stagingTile);
}
```

### Remote write with atomic add

```cpp
comm::TPUT<AtomicType::AtomicAdd>(dstG, srcG, stagingTile);
```

### Ping-pong double buffering

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);

comm::TPUT(dstG, srcG, pingTile, pongTile);
```

### Runtime atomic type

```cpp
// Select atomic type at runtime instead of compile-time template parameter
comm::TPUT(dstG, srcG, stagingTile, AtomicType::AtomicAdd);
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Inverse operation: [TGET](./TGET.md)
- Collective operations: [TBROADCAST](./TBROADCAST.md), [TGATHER](./TGATHER.md), [TSCATTLER](./TSCATTER.md), [TREDUCE](./TREDUCE.md)
- Instruction set: [Other and Communication](../other/README.md)
