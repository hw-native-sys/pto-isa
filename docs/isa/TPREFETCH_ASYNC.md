# TPREFETCH_ASYNC

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->


## Tile Operation Diagram

> No `TPREFETCH_ASYNC.svg` is provided. `TPREFETCH_ASYNC` is a **cache hint** primitive; it issues an SDMA CMO prefetch from GM/HBM into L2 cache and does not move data into a Tile.

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

`PrefetchAsyncContext` contains the SDMA workspace pointer prepared host-side via `SdmaWorkspaceManager::Init`.
It owns an internal session unless constructed to reuse an external session.
Use the returned `comm::AsyncEvent` with `ctx.GetSession()` to wait for completion when a dependent consumer must
observe the prefetch.

When `TPREFETCH_ASYNC` shares a channel group with `TGET_ASYNC` or `TPUT_ASYNC`, reuse their session.
Waiting for the returned prefetch event also completes every earlier SDMA operation in that shared session.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GlobalData&` | Source GlobalTensor region to prefetch into L2 |
| `ctx` | `PrefetchAsyncContext&` | Compute-side context containing the workspace and either an internal or shared external session |
| `events...` | `WaitEvents&...` | Optional wait events for synchronization |

### Return Value

`comm::AsyncEvent` - handle for asynchronous completion tracking. Call `evt.Wait(ctx.GetSession())` before a dependent `TLOAD` when the consumer must observe prefetched data.

## Constraints

- Source data must be in Global Memory (GM/HBM address space).
- For the GlobalTensor overload, the tensor must be flat contiguous (packed 1D layout).
- The SDMA workspace must be initialized by host code before the kernel launch and passed into the kernel.
- `PrefetchAsyncContext` owns a 256-byte UB scratch tile and an `AsyncSession` used while constructing and waiting on SDMA metadata.
- When sharing a channel group with `TGET_ASYNC` or `TPUT_ASYNC`, reuse their session.
- An external session must remain alive while the context and related events are in use.
- Concurrent independent contexts must use separate workspaces or be serialized.
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
    evt.Wait(ctx.GetSession());

    using TileData = Tile<TileType::Vec, half, 128, 128, BLayout::RowMajor>;
    TileData tile;
    TASSIGN(tile, 0x100);
    TLOAD(tile, srcGlobal);
}
```

### Shared External Session

Given an externally built `sharedSession` for the selected channel group:

```cpp
PrefetchAsyncContext ctx(workspace, &sharedSession);
auto getEvt = TGET_ASYNC(dstGlobal, srcGlobal, sharedSession);
auto prefetchEvt = TPREFETCH_ASYNC(prefetchGlobal, ctx);
auto putEvt = TPUT_ASYNC(remoteGlobal, localGlobal, sharedSession);
(void)putEvt.Wait(ctx.GetSession());
```
