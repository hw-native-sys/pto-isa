# TGET_ASYNC

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


## Introduction

`TGET_ASYNC` is an asynchronous remote read primitive. It starts a transfer from remote GM to local GM and returns an `AsyncEvent` immediately.

Data flow:

`srcGlobalData (remote GM) -> DMA engine -> dstGlobalData (local GM)`

## Template Parameter

- `engine`:
    - `DmaEngine::SDMA` (default)
    - `DmaEngine::URMA` (Ascend950, NPU_ARCH 3510 only)
    - `DmaEngine::RDMA` (Ascend950, NPU_ARCH 3510 only; currently supports only the HNS1825 NIC platform)

> **Important (SDMA path)**
> `TGET_ASYNC` with `DmaEngine::SDMA` currently supports **only flat contiguous logical 1D tensors**.
> Non-1D or non-contiguous layouts are not supported by the current SDMA async implementation.

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

`AsyncSession` is an engine-agnostic session object. Build once with
`BuildAsyncSession<engine>()`, then pass to all async calls and event waits.
The template `engine` parameter selects the DMA backend at compile time, making the
code forward-compatible with future engines (CCU, etc.).

## AsyncSession Construction

Use `BuildAsyncSession` from `include/pto/comm/async_common/async_event_impl.hpp`.
There are two overloads — one for SDMA and one for URMA — with different parameter lists.

### SDMA Construction (default)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| Parameter | Default | Description |
|---|---|---|
| `scratchTile` | — | UB scratch tile for SDMA control metadata (see [scratchTile Role](#scratchtile-role)). |
| `workspace` | — | GM pointer allocated by the caller and initialized by host-side `SdmaWorkspaceManager`. |
| `session` | — | Output `AsyncSession` object. |
| `syncId` | `0` | MTE3/MTE2 pipe sync event id (0-7). Override if kernel uses other pipe barriers on the same id. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`. Suitable for most single-queue transfers. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index. Default uses `get_block_idx()` internally, mapping to current AI core. Override for multi-block, concurrent, or custom channel mapping scenarios. |

### URMA Construction (NPU_ARCH 3510 only)

> URMA (User-level RDMA Memory Access) is a hardware-accelerated RDMA transport available on Ascend950 (NPU_ARCH 3510).
> URMA requires CANN Toolkit **>= 9.1.0**.

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| Parameter | Description |
|---|---|
| `workspace` | GM pointer allocated by the caller and initialized by host-side `UrmaWorkspaceManager`. |
| `destRankId` | Remote PE rank id that this session communicates with. For `TGET_ASYNC` this is the source rank. |
| `session` | Output `AsyncSession` object. |

URMA does not require `scratchTile` — polling uses `ld_dev`/`st_dev` hardware intrinsics directly.

### RDMA Construction (NPU_ARCH 3510 only)

RDMA is intended for standard cross-node networking and currently supports only the HNS1825 NIC platform.

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    uint32_t qpIdx = 0);
#endif
```

| Parameter | Description |
|---|---|
| `scratchTile` | UB/Vec tile used for WQE/CQE control data; at least 64 bytes. |
| `workspace` | GM pointer returned by the host-side RDMA initialization flow. |
| `myPe` | Local rank id used to select the registered local memory region. |
| `session` | Output `AsyncSession` object. |
| `syncId` | MTE/scalar synchronization event id in the range 0-7. |
| `qpIdx` | QP index for each peer; defaults to 0 and must be less than the host-configured [`qpNum`](TPUT_ASYNC.md#rdma-construction-npu_arch-3510-only). |

Build one peer-independent session, then select the source rank with the explicit `peer` argument:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPe, session, syncId)) {
    auto event = comm::TGET_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, peer);
    (void)event.Wait(session);
}
```

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both SDMA and URMA paths require source tensor to be **flat contiguous logical 1D only**
- SDMA workspace must be a valid GM pointer allocated by the caller and initialized by host-side `SdmaWorkspaceManager`
- URMA workspace must be a valid GM pointer allocated by the caller and initialized by host-side `UrmaWorkspaceManager`
- Keep the session and its workspace alive until all associated events have completed
- URMA is only available on NPU_ARCH 3510 (Ascend950)
- URMA requires CANN Toolkit **>= 9.1.0**
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge-page memory (allocate with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires huge-page backing; `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4KB pages for small allocations, causing registration to fail
- RDMA is available only on Ascend950 / NPU_ARCH 3510 and currently supports only the HNS1825 NIC platform
- RDMA source and destination tensors must be flat contiguous logical 1D, and the complete local and remote ranges must lie within memory regions registered during host initialization
- One RDMA transfer must not exceed `0x7fffffff` bytes

If the 1D contiguous requirement is not met, current implementation returns an invalid async event (`handle == 0`).

## scratchTile Role

`scratchTile` is **not** used to hold transferred payload data.
It is converted to `TmpBuffer` and used as temporary UB workspace for:

- writing/reading SDMA control words (flag, sq_tail, channel_info)
- polling event completion flags
- committing queue tail during completion

The real payload path remains remote GM -> DMA engine -> local GM; `scratchTile` is only for control/synchronization metadata.

## scratchTile Type and Size Constraints

- must be a `pto::Tile` type
- must be UB/Vec tile (`ScratchTile::Loc == TileType::Vec`)
- available bytes must be at least `sizeof(uint64_t)` (8 bytes)

Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256Byte).

## Completion Semantics (Quiet Semantics)

The completion mechanism differs by engine, but user-facing quiet semantics are identical:

- **SDMA**: Each `TGET_ASYNC` submits data-transfer SQEs and flag SQEs that mark completion of that operation. `Wait` or `Test` on its returned event polls the corresponding flags to determine whether that `TGET_ASYNC` has completed; completion also guarantees that all earlier SDMA operations in the same session have completed.
- **URMA**: `TGET_ASYNC` submits an RDMA READ WQE and rings the doorbell immediately. `Wait` polls the Completion Queue (CQ) until all expected CQEs have been consumed.
- **RDMA**: `TGET_ASYNC` submits an RDMA READ WQE to the queue selected by `peer`. Completion is tracked independently for each peer/queue.

- `event.Wait(session)` — blocks until **all async operations issued since the last Wait** are complete

This means after multiple `TGET_ASYNC` calls, a single `Wait` on the last returned `AsyncEvent` drains all pending operations (similar to shmem's quiet semantics).

For RDMA operations targeting different peers, wait for the last event of each peer separately.

Up to 64 operations may be outstanding in one session before submission can apply backpressure.

After wait succeeds, all issued reads into `dstGlobalData` are complete.

## SDMA Concurrency and Session Ownership

- Do not use one session concurrently from multiple execution flows.
- Operations that share a channel group must also share the same session.
- Concurrent kernels, or multiple independent sessions within one kernel, must use isolated channel groups.
- Complete all outstanding events before rebuilding a session or reusing its channel group.

## Example

### Single Transfer

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGet(__gm__ T *localDst, __gm__ T *remoteSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(remoteSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### Batch Transfer (Quiet Semantics)

```cpp
template <typename T>
__global__ AICORE void BatchGet(__gm__ T *localDstBase, __gm__ T *remoteSrcBase,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(localDstBase + rank * 1024, shape, stride);
        GT srcG(remoteSrcBase + rank * 1024, shape, stride);
        lastEvent = comm::TGET_ASYNC(dstG, srcG, session);
    }
    (void)lastEvent.Wait(session);  // single Wait drains all pending ops
}
```

### URMA Example (NPU_ARCH 3510)

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGetUrma(__gm__ T *localDst, __gm__ T *remoteSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t srcRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(remoteSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, srcRankId, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```
