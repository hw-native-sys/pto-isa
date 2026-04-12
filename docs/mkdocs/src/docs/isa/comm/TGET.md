<!-- Generated from `docs/isa/comm/TGET.md` -->

# pto.tget / TGET

## Summary

Remote read operation: copies data from a remote NPU's global memory (GM) to local GM. `pto.tget` is the IR spelling; `TGET` is the C++ intrinsic spelling — both refer to the same operation.

Data is transferred via a staging tile in the Unified Buffer (UB) as an intermediate buffer. The complete data path is:

```
remote GM ──► staging tile (UB) ──► local GM
```

## Math Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

## Auto-Chunking (2D Sliding)

When the `GlobalTensor` exceeds the UB tile capacity in rows or columns, `TGET` automatically performs **2D sliding** — chunking along rows (DIM_3) and columns (DIM_4) to fit each chunk into the tile, iterating over all outer dimensions (DIM_0, DIM_1, DIM_2).

The author does not need to manually partition the transfer. The staging tile size determines the chunk granularity.

## Assembly Syntax

PTO-AS form (IR/assembly spelling):

```text
pto.tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```

The lowering introduces a UB staging tile for the GM→UB→GM data path. The C++ intrinsic requires an explicit `stagingTileData` (or `pingTile`/`pongTile`) operand.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

### Single-tile (auto-chunking)

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData,
                          GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData,
                          WaitEvents&... events);
```

### Ping-pong double buffering

Uses two staging tiles to overlap DMA transfers for adjacent chunks, hiding transfer latency behind computation.

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData,
                          GlobalSrcData &srcGlobalData,
                          TileData &pingTile,
                          TileData &pongTile,
                          WaitEvents&... events);
```

## Constraints

### Type constraints

- `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`
- `TileData::DType` must equal `GlobalSrcData::RawDType`
- `GlobalSrcData::layout` must equal `GlobalDstData::layout`

### Memory constraints

- `srcGlobalData` must point to a remote address (on the source NPU)
- `dstGlobalData` must point to a local address (on the current NPU)
- Staging tile(s) must be pre-allocated in UB

### Ping-pong constraints

- `pingTile` and `pongTile` must have identical type and dimensions
- They must reside at non-overlapping UB offsets

## Examples

### Basic usage

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void remote_read(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT   = Tile<TileType::Vec, T, 16, 16>;
    using GShape  = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(remote_addr);
    GTensor dstG(local_data);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    // Remote read: remote GM -> staging tile -> local GM
    comm::TGET(dstG, srcG, stagingTile);
}
```

### Large tensor with auto-chunking

```cpp
// GlobalTensor larger than UB tile: 2D sliding is automatic
using GShape  = Shape<1, 1, 1, 4096, 4096>;
using GStride = BaseShape2D<T, 4096, 4096, Layout::ND>;
using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

GTensor srcG(remote_addr);
GTensor dstG(local_data);
TileT stagingTile(64, 64);   // chunk size = 64x64
TASSIGN(stagingTile, 0);

// TGET automatically chunks the 4096x4096 transfer into 64x64 tiles
comm::TGET(dstG, srcG, stagingTile);
```

### Ping-pong double buffering

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // non-overlapping UB offsets

// Overlaps TGET[i+1] with TGET[i] for better pipeline utilization
comm::TGET(dstG, srcG, pingTile, pongTile);
```

## See Also

- [Communication And Runtime](../communication-and-runtime.md) — Family overview
- [TPUT](./TPUT.md) — Remote write (inverse operation)
- [TBROADCAST](./TBROADCAST.md) — Broadcast from one NPU to all
- [TGATHER](./TGATHER.md) — Gather from all NPUs to one
