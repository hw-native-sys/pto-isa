# TPREFETCH_ASYNC

## Introduction

Prefetch data from Global Memory (GM/HBM) into the NPU's L2 Cache via SDMA CMO (Cache Maintenance Operation, opcode=6). Unlike `TPREFETCH`, which moves data from GM into UB, `TPREFETCH_ASYNC` keeps data in L2 cache only, consuming no UB space for the data itself. This allows subsequent `TLOAD` operations to hit L2 cache instead of going to GM.

`TPREFETCH_ASYNC` is logically a memory-access/cache-hint instruction. It uses the SDMA CMO path internally, but the public API lives in namespace `pto`, next to `pto::TPREFETCH`.

## Data Flow

```
GM / HBM  ──(SDMA CMO prefetch)──>  L2 Cache
                                       │
                                       └── subsequent TLOAD hits L2 (fast)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`, namespace `pto`:

```cpp
namespace pto {

template <typename GlobalData, typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_ASYNC(GlobalData &src, PrefetchAsyncContext &ctx,
                                          WaitEvents &... events);

} // namespace pto
```

`PrefetchAsyncContext` contains the SDMA workspace pointer prepared host-side via `SdmaWorkspaceManager::Init`:

```cpp
pto::PrefetchAsyncContext ctx(workspace);
auto evt = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
evt.Wait(ctx.session);
```

The instruction builds and stores an SDMA session inside `PrefetchAsyncContext` with default parameters (`channelGroupIdx = get_block_idx()`, `syncId = 0`, `queue_num = 1`). Use the returned `comm::AsyncEvent` with `ctx.session` to wait for completion when a dependent consumer must observe the prefetch.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GlobalData&` | Source GlobalTensor region to prefetch into L2 |
| `ctx` | `PrefetchAsyncContext&` | Compute-side prefetch context containing the SDMA workspace base and internally built `comm::AsyncSession` |
| `events...` | `WaitEvents&...` | Optional wait events for synchronization |

### Return Value

`comm::AsyncEvent` - handle for asynchronous completion tracking. Call `evt.Wait(ctx.session)` before a dependent `TLOAD` when the consumer must observe prefetched data.

## Constraints

- Source data must be in Global Memory (GM/HBM address space).
- For the GlobalTensor overload, the tensor must be flat contiguous (packed 1D layout).
- The SDMA workspace must be initialized by host code before the kernel launch and passed into the kernel.
- `PrefetchAsyncContext` owns a 256-byte UB scratch tile and an `AsyncSession` used while constructing and waiting on SDMA metadata.
- SDMA CMO operates at cache-line granularity; non-aligned prefetch ranges are rounded by hardware.
- On CPU simulation backend, this instruction is a no-op (returns an empty `AsyncEvent`).

## Comparison with TPREFETCH

| Dimension | `pto::TPREFETCH` | `pto::TPREFETCH_ASYNC` |
|-----------|-----------|--------------|
| Data flow | GM → UB | GM → L2 Cache |
| Hardware path | MTE (`copy_gm_to_ubuf`) | SDMA CMO (opcode=6) |
| UB consumption | Yes (requires dst Tile) | No (only 256Byte scratch for SQE construction) |
| Synchronization | Synchronous (pipeline barrier) | Asynchronous (`AsyncEvent`) |
| Use case | Small data preload to UB | Large data L2 warm-up |

## Examples

### Basic Usage

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void my_kernel(__gm__ half *src, __gm__ half *dst,
                                 __gm__ uint8_t *workspace)
{
    using GShape = Shape<1, 1, 1, 1, 16384>;
    using GStride = Stride<1, 1, 1, 1, 1>;
    GlobalTensor<half, GShape, GStride> srcGlobal(src);

    PrefetchAsyncContext ctx(workspace);
    auto evt = TPREFETCH_ASYNC(srcGlobal, ctx);
    evt.Wait(ctx.session);

    using TileData = Tile<TileType::Vec, half, 128, 128, BLayout::RowMajor>;
    TileData tile;
    TASSIGN(tile, 0x100);
    TLOAD(tile, srcGlobal);
}
```
