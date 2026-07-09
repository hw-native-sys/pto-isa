# TPREFETCH_ASYNC

## Introduction

Prefetch data from Global Memory (GM/HBM) into the NPU's L2 cache through SDMA CMO (Cache Maintenance Operation, opcode=6). Unlike `TPREFETCH`, which moves data from GM into UB, `TPREFETCH_ASYNC` only warms the data in L2 cache and does not consume UB space for the data itself. This allows subsequent `TLOAD` operations to hit in L2 cache instead of going to GM.

## Data Flow

```
GM / HBM  ──(SDMA CMO prefetch)──>  L2 Cache
                                       │
                                       └── subsequent TLOAD hits L2 (fast)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` under the `pto` namespace:

```cpp
namespace pto {

template <typename GlobalData, typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_ASYNC(GlobalData &src, PrefetchAsyncContext &ctx,
                                          WaitEvents &... events);

} // namespace pto
```

`PrefetchAsyncContext` contains the SDMA workspace pointer prepared on the host side by `SdmaWorkspaceManager::Init`:

```cpp
pto::PrefetchAsyncContext ctx(workspace);
auto evt = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
evt.Wait(ctx.session);
```

The instruction internally builds an `SdmaAsyncSession` with default parameters (`channelGroupIdx = get_block_idx()`, `syncId = 0`, `queue_num = 1`). When a downstream consumer depends on the prefetched data, use the returned `comm::AsyncEvent` together with `ctx.session` to wait for completion.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GlobalData&` | Source `GlobalTensor` region to prefetch into L2 |
| `ctx` | `PrefetchAsyncContext&` | Compute-side prefetch context containing the SDMA workspace base and internally built `comm::AsyncSession` |
| `events...` | `WaitEvents&...` | Optional wait events for synchronization |

### Return Value

`comm::AsyncEvent` - a handle for tracking asynchronous completion. Call `evt.Wait(ctx.session)` before a dependent `TLOAD` when the consumer must observe the prefetched data.

## Constraints

- Source data must be in Global Memory (GM/HBM address space).
- The source `GlobalTensor` must be flat contiguous (packed 1D layout); otherwise the instruction returns an invalid async event (`handle == 0`).
- The SDMA workspace must be initialized by host code before the kernel launch and passed into the kernel.
- In auto mode (`__PTO_AUTO__`), this instruction is a no-op stub for API compatibility and returns an invalid async event (`handle == 0`).
- On the CPU simulation backend, this instruction is a no-op and returns an invalid async event (`handle == 0`).

## Notes

- `PrefetchAsyncContext` contains a 256B UB scratch tile used only for SDMA control metadata, not for prefetched payload data. It also stores the `AsyncSession` used by the returned event.
- SDMA CMO operates at cache-line granularity; unaligned prefetch ranges are handled by hardware.

## Comparison with TPREFETCH

| Dimension | `pto::TPREFETCH` | `pto::TPREFETCH_ASYNC` |
|-----------|-----------|--------------|
| Data flow | GM → UB | GM → L2 Cache |
| Hardware path | MTE (`copy_gm_to_ubuf`) | SDMA CMO (opcode=6) |
| UB consumption | Yes (requires dst Tile) | No (only 256B scratch for SQE construction) |
| Synchronization | Synchronous (pipeline barrier) | Asynchronous (`AsyncEvent`) |
| Use case | Small data preload to UB | Large data or cross-stage L2 warm-up |

## Examples

### Basic Usage

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void my_kernel(__gm__ half *src, __gm__ half *dst,
                                 __gm__ uint8_t *workspace)
{
    using GShape = Shape<1, 1, 1, 1, 16384>;
    using GStride = Stride<16384, 16384, 16384, 16384, 1>;
    GlobalTensor<half, GShape, GStride> srcGlobal(src);

    PrefetchAsyncContext ctx(workspace);
    auto evt = TPREFETCH_ASYNC(srcGlobal, ctx);
    evt.Wait(ctx.session);

    using TileData = Tile<TileType::Vec, half, 128, 128, BLayout::RowMajor>;
    TileData tile;
    TLOAD(tile, srcGlobal);
}
```
