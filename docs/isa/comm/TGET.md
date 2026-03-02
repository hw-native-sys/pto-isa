# TGET

## Introduction

Remote read operation: read remote NPU's data to local memory. Data is transferred via a UB tile as intermediate staging buffer.

When the GlobalTensor exceeds the UB tile capacity, TGET automatically performs **2D sliding** — chunking rows (DIM_3) and columns (DIM_4) to fit each chunk into the tile, iterating over all outer dimensions (DIM_0, DIM_1, DIM_2).

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

Data flow: `srcGlobalData (remote GM)` → `stagingTileData (UB)` → `dstGlobalData (local GM)`

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../../grammar/PTO-AS.md).

Synchronous form:

```text
tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```
Lowering introduces UB staging tile(s) for the GM→UB→GM data path; the C++ intrinsic requires explicit `stagingTileData` (or `pingTile` / `pongTile`) operand(s).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`

### Single-tile (auto-chunking)

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);
```

### Ping-pong double buffering

Uses two staging tiles to overlap TLOAD and TSTORE for adjacent chunks, hiding one DMA transfer behind the other.

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
  - `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
  - `TileData::DType` must equal `GlobalSrcData::RawDType`.
  - `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
- **Memory constraints**:
  - `srcGlobalData` must point to remote address (on source NPU).
  - `dstGlobalData` must point to local address (on current NPU).
  - `stagingTileData` / `pingTile` / `pongTile` must be pre-allocated in Unified Buffer.
- **Valid region**:
  - Transfer size is determined by `GlobalTensor` shape (auto-chunked to fit tile).
- **Ping-pong**:
  - `pingTile` and `pongTile` must have the same type and dimensions.
  - Must reside at non-overlapping UB offsets.

## Examples

### Basic Usage

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tget(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT = Tile<TileType::Vec, T, 16, 16>;
    using GShape = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    /* 
    If the globalTensor is larger than UB Tile, TGET will perform 2D sliding automatically. 
    using GShape = Shape<1, 1, 1, 4096, 4096>;
    using GStride = BaseShape2D<T, 4096, 4096, Layout::ND>;
    */
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(remote_addr);
    GTensor dstG(local_data);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    // Basic remote read
    comm::TGET(dstG, srcG, stagingTile);
}
```

### Ping-pong Double Buffering

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // Non-overlapping UB region

// Overlaps TLOAD[i+1] with TSTORE[i] for better pipeline utilization
comm::TGET(dstG, srcG, pingTile, pongTile);
```
