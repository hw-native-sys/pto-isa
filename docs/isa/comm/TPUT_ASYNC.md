# TPUT_ASYNC

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

`TPUT_ASYNC` is a remote write primitive. It issues a transfer from local GM to remote GM and returns an `AsyncEvent` to wait on.

Data flow:

`srcGlobalData (local GM) -> DMA engine -> dstGlobalData (remote GM)`


## Template Parameter

- `engine`:
    - `DmaEngine::SDMA` (default). Asynchronous on A2/A3; a synchronous implementation on Ascend 950, see [SDMA Path on A5](#sdma-path-on-a5)
    - `DmaEngine::URMA` (Ascend950, NPU_ARCH 3510 only)
    - `DmaEngine::RDMA` (Ascend950, NPU_ARCH 3510 only; currently supports only the HNS1825 NIC platform)

> **Important (layout constraint)**
> All engines currently support **only flat contiguous logical 1D tensors**; non-1D or non-contiguous layouts are not supported.


## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

### Contiguous Tensor Overload

```cpp
// Available on all architectures. URMA/RDMA use the destination rank bound in the session.
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);

// Available on Ascend 950 and CPU simulation only. URMA/RDMA select the destination rank with
// `peer`; SDMA ignores `peer`.
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               WaitEvents &... events);
```

### URMA Gather Overload

```cpp
template <DmaEngine engine = DmaEngine::URMA, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    __gm__ const GatherDestination *destinations,
    uint32_t destinationCount,
    __gm__ const GatherSegment *sourceSegments,
    uint32_t sourceSegmentCount,
    const AsyncSession &session,
    WaitEvents &... events);
```

This overload gathers non-contiguous local GM segments into one or more contiguous remote ranges and
supports only `DmaEngine::URMA`. Each destination descriptor carries its own `peer`, so one call may
target multiple ranks and still returns a single `AsyncEvent`. Validation or resource-reservation
failure returns a failed event; check `event.valid()` before `Wait` or `Test`.

`AsyncSession` is an engine-agnostic session object. Build once with
`BuildAsyncSession<engine>()`, then pass to all async calls and event waits.
The template `engine` parameter selects the DMA backend at compile time, making the
code forward-compatible with future engines (CCU, etc.).

`events` is a possibly empty variadic list; each event must expose a no-argument `Wait()`, and the
instruction waits on them in order before issuing the transfer. `AsyncEvent` uses `Wait(session)` and
cannot be passed as one of `events`.

## AsyncSession Construction

Use `BuildAsyncSession` from `include/pto/comm/async_common/async_event_impl.hpp`, picking the
overload for the engine. It returns `false` on failure; only a successfully built session may be used
in subsequent calls.

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
| `scratchTile` | — | UB scratch tile; see [scratchTile Role](#scratchtile-role) for its uses. |
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
// No destination rank bound; use with the instruction overload that takes `peer`,
// and with the Gather overload.
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    AsyncSession &session);

// Same, plus the UB scratch required by the Gather SIMT path.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    ScratchTile &scratchTile,
                                    AsyncSession &session);

// Binds the destination rank; use with the instruction overload without `peer`.
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| Parameter | Description |
|---|---|
| `scratchTile` | UB scratch tile allocated by the kernel. Required only by the Gather SIMT path. |
| `workspace` | GM pointer allocated by the caller and initialized by host-side `UrmaWorkspaceManager`. |
| `destRankId` | Remote PE rank id used by the contiguous-tensor overload. |
| `session` | Output `AsyncSession` object. |

The contiguous-tensor overload uses the builder that takes `destRankId`; the Gather overload uses the
rank-independent builder, where each `GatherDestination::peer` names its own target rank.

Contiguous transfers do not require `scratchTile`, because polling uses `ld_dev`/`st_dev` hardware
intrinsics directly. The Gather SIMT path needs at least `sizeof(urma::UrmaGatherSimtScratch)` bytes
of UB; without it Gather falls back to the scalar path.

Construction divides the URMA queues available to the current AI core. It returns `false` when
`workspace` is invalid, or when the current block count exceeds the queue capacity configured in the
workspace.

### RDMA Construction (NPU_ARCH 3510 only)

RDMA is intended for standard cross-node networking and currently supports only the HNS1825 NIC platform.

```cpp
#ifdef PTO_RDMA_SUPPORTED
// No peer bound; use with the instruction overload that takes `peer`, so one session can reach
// multiple remote ranks.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    uint32_t qpIdx = 0);

// Binds the destination rank; use with the instruction overload without `peer`.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t destRankId,
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
| `destRankId` | Remote rank bound to this session. |
| `myPe` | Local rank id used to select the registered local memory region. |
| `session` | Output `AsyncSession` object. |
| `syncId` | MTE/scalar synchronization event id in the range 0-7. |
| `qpIdx` | QP index for each peer; defaults to 0 and must be less than the host-configured `qpNum`. |

The host sets the QP count per peer with `RdmaTransportConfig::qpNum` (default: 1; identical on all ranks),
or `rdma::WorkspaceConfig::qpNum` when using the backend manager directly. Different QPs complete independently;
wait for the events on each QP.

When reaching multiple remote ranks, prefer building one peer-independent session and selecting the destination with the explicit `peer` argument:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPe, session, syncId)) {
    auto event = comm::TPUT_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, peer);
    (void)event.Wait(session);
}
```

## URMA Gather Descriptors and Data Mapping

The descriptor types are:

```cpp
struct GatherSegment {
    __gm__ const uint8_t *addr;  // start of a local GM segment
    uint32_t length;             // length in bytes
};

struct GatherDestination {
    __gm__ uint8_t *addr;        // start of a contiguous remote range
    uint32_t segCount;           // number of source segments consumed
    uint32_t peer;               // destination rank
};
```

Entries in `destinations` consume consecutive entries from `sourceSegments` in index order. If
the destination `segCount` values are 2 and 1:

```text
destinations[0] <- sourceSegments[0], sourceSegments[1]
destinations[1] <- sourceSegments[2]
```

The descriptors must therefore satisfy `sum(destinations[i].segCount) == sourceSegmentCount`.

## URMA Gather Completion Policies

Submission timing follows `submitMode` and `batchSize` as described in
[Batch Submission](#batch-submission), except that a Gather batch is not restricted to a single
destination rank. Completion-notification granularity is controlled separately by
`session.cfg.urma.cqePolicy`, which defaults to `OneCqe`.

| `UrmaCqePolicy` | Behavior |
|---|---|
| `OneCqe` | Keep full completion notifications for this operation; the returned event supports `Wait`/`Test`. The default. |
| `Final` | Reduce completion-notification overhead; the returned event supports `Wait`/`Test` and also covers preceding unconfirmed `NoCqe` operations. |
| `NoCqe` | Do not request a completion notification. An event returned by a non-empty call does not support `Wait`/`Test` and must be confirmed by a later `Final`. |

## Constraints

Contiguous tensor overload:

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Source and destination tensors must both be **flat contiguous logical 1D**, and the destination capacity must be no smaller than the source element count

URMA Gather overload:

- `destinations` and `sourceSegments` reside in GM; a nonzero count requires a non-null pointer
- Each source segment has a non-null `addr` and `length > 0`
- Every local source range and remote destination range lies within an MR registered during host initialization
- Every `peer` is less than the host-configured rank count
- `sum(destinations[i].segCount) == sourceSegmentCount`
- Do not mix an unfinished contiguous-tensor batch with a Gather batch in one session
- Do not use one session concurrently from multiple execution flows; each AIV must build its own session
- Keep descriptor arrays valid while the call reads them, and source/destination data alive until the associated event completes

General constraints:

- SDMA workspace must be a valid GM pointer allocated by the caller and initialized by host-side `SdmaWorkspaceManager`
- URMA workspace must be a valid GM pointer allocated by the caller and initialized by host-side `UrmaWorkspaceManager`
- Keep the session and its workspace alive until all associated events have completed
- URMA is only available on NPU_ARCH 3510 (Ascend950)
- URMA requires CANN Toolkit **>= 9.1.0**
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge-page memory (allocate with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires huge-page backing; `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4KB pages for small allocations, causing registration to fail
- RDMA is available only on Ascend950 / NPU_ARCH 3510 and currently supports only the HNS1825 NIC platform
- RDMA source and destination tensors must be flat contiguous logical 1D, and the complete local and remote ranges must lie within memory regions registered during host initialization
- One RDMA transfer must not exceed `0x7fffffff` bytes

Violations of the tensor constraints above are caught by `PTO_ASSERT`.

## SDMA Path on A5

On Ascend 950, `TPUT_ASYNC<DmaEngine::SDMA>` is a synchronous implementation that keeps the interface
unchanged: the payload is moved in chunks through UB, the transfer is already complete when the call
returns, and the returned event is already in the completed state, so `Wait` and `Test` return `true`
immediately. This path does not support [batch submission](#batch-submission).

`TGET_ASYNC<DmaEngine::SDMA>` on the same platform remains an asynchronous transfer.

## scratchTile Role

| Scenario | Purpose | Minimum usable bytes |
|---|---|---|
| A2/A3 SDMA | SDMA control metadata and completion flag polling; carries no payload | 8 |
| Ascend 950 SDMA | Payload chunk staging; the tile size is the chunk granularity | 8 |
| RDMA | WQE/CQE control data; carries no payload | 64 |

Type requirement: must be a `pto::Tile` located in UB/Vec (`ScratchTile::Loc == TileType::Vec`).

Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256Byte). On the Ascend 950
SDMA path, enlarge the tile to match the typical payload size when transferring large payloads, so
that fewer chunks are needed.

## Batch Submission

`AsyncSession::submitMode` and `AsyncSession::batchSize` control whether `TPUT_ASYNC` submits
immediately or stages operations into a batch. Both are assigned directly after `BuildAsyncSession`
succeeds; the instruction signature is unchanged. The defaults, `IMMEDIATE` and `UINT32_MAX`, submit
one operation at a time.

| `AsyncSubmitMode` | Meaning |
|---|---|
| `IMMEDIATE` | Submit this transfer immediately. A staged batch, if any, is submitted first. |
| `DEFER` | Stage this transfer; submit automatically once the staged count reaches `batchSize`. |
| `DEFER_AND_SUBMIT` | Stage this transfer and submit the whole batch immediately. |

- Batching is supported only by `TPUT_ASYNC<SDMA>` on A2/A3 and `TPUT_ASYNC<URMA>` on Ascend 950; other paths ignore `submitMode`.
- `batchSize` of 0 disables batching; using a non-`IMMEDIATE` mode then returns a failed event.
- All operations in one URMA batch must target the same rank, otherwise a failed event is returned.
- `TGET_ASYNC` and `TPUT_ASYNC_NOTIFY` do not join a batch, but they do submit any batch already staged on the same session.
- Wait on the event returned by the submitting call; an intermediate event returned during the `DEFER` stage does not mean the batch has been submitted.

```cpp
comm::AsyncSession session;
if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
    return;
}
session.batchSize = 2U;

session.submitMode = comm::AsyncSubmitMode::DEFER;
(void)comm::TPUT_ASYNC(dst0, src0, session);          // staged

session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
auto event = comm::TPUT_ASYNC(dst1, src1, session);   // staged, then the whole batch is submitted
(void)event.Wait(session);
```

## Completion Semantics (Quiet Semantics)

- **SDMA**: Each `TPUT_ASYNC` submits data-transfer SQEs and flag SQEs that mark completion of that operation. `Wait` or `Test` on its returned event polls the corresponding flags to determine whether that `TPUT_ASYNC` has completed; completion also guarantees that all earlier SDMA operations in the same session have completed.
- **URMA contiguous tensor overload**: `TPUT_ASYNC` submits an RDMA WRITE WQE and rings the doorbell immediately. `Wait` polls the Completion Queue (CQ) until all expected CQEs have been consumed.
- **URMA Gather overload**: `submitMode` controls submission, and `cqePolicy` controls completion semantics. Events returned with `OneCqe` or `Final` support waiting. A non-empty `NoCqe` event does not; use a later `Final` to confirm completion.
- **RDMA**: `TPUT_ASYNC` submits an RDMA WRITE WQE to the queue selected by `peer`. Completion is tracked independently for each peer/queue.

- `event.Wait(session)` — blocks until **all async operations issued since the last Wait** are complete

This means after multiple `TPUT_ASYNC` calls, a single `Wait` on the last returned `AsyncEvent` drains all pending operations (similar to shmem's quiet semantics).

For RDMA operations targeting different peers, wait for the last event of each peer separately.

The Gather overload follows its CQE policy instead: never wait on a non-empty `NoCqe` event, and use a later `Final` to confirm the preceding `NoCqe` operations in the session.

Up to 64 operations may be outstanding in one session before submission can apply backpressure.

After wait succeeds, all issued writes to `dstGlobalData` are complete.

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
__global__ AICORE void SimplePut(__gm__ T *remoteDst, __gm__ T *localSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### Repeated Transfers (Quiet Semantics)

```cpp
template <typename T>
__global__ AICORE void BatchPut(__gm__ T *remoteDstBase, __gm__ T *localSrc,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(remoteDstBase + rank * 1024, shape, stride);
        lastEvent = comm::TPUT_ASYNC(dstG, srcG, session);
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
__global__ AICORE void SimplePutUrma(__gm__ T *remoteDst, __gm__ T *localSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t destRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, destRankId, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### URMA Gather Example (NPU_ARCH 3510)

This example concatenates four non-contiguous local segments into two contiguous remote ranges.
The destinations may belong to different ranks:

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

__global__ AICORE void GatherPutUrma(
    __gm__ uint8_t *remote0, __gm__ uint8_t *remote1,
    __gm__ const uint8_t *local0, __gm__ const uint8_t *local1,
    __gm__ const uint8_t *local2, __gm__ const uint8_t *local3,
    __gm__ comm::GatherDestination *dsts, __gm__ comm::GatherSegment *segs,
    __gm__ uint8_t *urmaWorkspace, uint32_t peer0, uint32_t peer1)
{
    dsts[0] = {remote0, 2, peer0};
    dsts[1] = {remote1, 2, peer1};

    segs[0] = {local0, 4096};
    segs[1] = {local1, 1024};
    segs[2] = {local2, 2048};
    segs[3] = {local3, 3072};
    comm::urma::DcciCachelines(
        reinterpret_cast<__gm__ uint8_t *>(dsts), 2 * sizeof(comm::GatherDestination));
    comm::urma::DcciCachelines(
        reinterpret_cast<__gm__ uint8_t *>(segs), 4 * sizeof(comm::GatherSegment));
    dsb(DSB_DDR);

    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, sizeof(comm::urma::UrmaGatherSimtScratch)>;
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
        return;
    }
    session.cfg.urma.cqePolicy = comm::urma::UrmaCqePolicy::Final;

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dsts, 2, segs, 4, session);
    if (event.valid()) {
        (void)event.Wait(session);
    }
}
```

On completion, `remote0` contains `local0` followed by `local1`, while `remote1` contains `local2`
followed by `local3`. After writing descriptors, the execution flow that submits the session must
clean their cache lines and issue `dsb(DSB_DDR)` so they are visible before URMA reads them.

### URMA Gather Deferred Submission with a NoCqe Prefix

To reduce submission overhead, stage calls with `DEFER + NoCqe`, then submit and wait for the entire
prefix through an empty `Final` call:

```cpp
session.submitMode = comm::AsyncSubmitMode::DEFER;
session.batchSize = 4;
session.cfg.urma.cqePolicy = comm::urma::UrmaCqePolicy::NoCqe;

auto noCqeEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dsts, 2, segs, 4, session);
// Do not call Wait/Test on a non-empty NoCqe event.

session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
session.cfg.urma.cqePolicy = comm::urma::UrmaCqePolicy::Final;
auto finalEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    nullptr, 0, nullptr, 0, session);
if (finalEvent.valid()) {
    (void)finalEvent.Wait(session);
}
```

`DEFER` automatically submits accumulated operations according to `batchSize`. Set `batchSize`
to `UINT32_MAX` when submission should occur only through an explicit `DEFER_AND_SUBMIT` call.
