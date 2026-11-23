# TPUT_ASYNC_NOTIFY

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

`TPUT_ASYNC_NOTIFY` is an asynchronous remote write with notification. It transfers a non-empty payload from local
GM to remote GM and then updates one remote 32-bit signal. The returned `AsyncEvent` represents completion of both
the payload transfer and the signal update.

Data flow:

`srcGlobalData (local GM)` → DMA engine → `dstGlobalData (remote GM)` → update `dstSignalData (remote GM)`

The caller supplies remote addresses for both `dstGlobalData` and `dstSignalData`. Use `TNOTIFY` when only a signal
update is required and no payload is transferred.

## Template Parameter

`engine` selects the DMA backend at compile time:

| Engine | Platform restriction | Supported `NotifyOp` |
|---|---|---|
| `DmaEngine::SDMA` (default) | A2/A3 and A5 | `Set`, `AtomicAdd` |
| `DmaEngine::URMA` | Ascend950, NPU_ARCH 3510 only; requires CANN Toolkit >= 9.1.0 | `Set`, `AtomicAdd` |
| `DmaEngine::RDMA` | Ascend950, NPU_ARCH 3510 only; currently supports HNS1825 RoCE | `Set`, `AtomicAdd` |

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(GlobalDstData &dstGlobalData,
                                      GlobalSrcData &srcGlobalData,
                                      GlobalSignalData &dstSignalData,
                                      int32_t signalValue,
                                      NotifyOp notifyOp,
                                      const AsyncSession &session,
                                      uint32_t peer,
                                      WaitEvents &... events);
```

For URMA and RDMA, `peer` selects the communication queue and remote-memory information for the target rank. Both
`dstGlobalData` and `dstSignalData` must belong to that peer. SDMA does not use `peer`; its remote addresses come from
the global tensors.

`events` is a possibly empty parameter pack. Every event must provide a zero-argument `Wait()` method; the intrinsic
waits for the events before starting the payload transfer. `AsyncEvent` provides `Wait(session)` instead of a
zero-argument `Wait()`, so it cannot be passed as an `events` argument and must be waited explicitly with its session.

## Parameters

| Parameter | Description |
|---|---|
| `dstGlobalData` | Destination global tensor for the payload in remote GM. |
| `srcGlobalData` | Source global tensor for the payload in local GM. |
| `dstSignalData` | A 32-bit signal in remote GM. Its data type must be `int32_t`. |
| `signalValue` | Value assigned by `Set`, or increment used by `AtomicAdd`. |
| `notifyOp` | Signal update operation: `NotifyOp::Set` or `NotifyOp::AtomicAdd`. |
| `session` | `AsyncSession` built for the `engine` template parameter. |
| `peer` | Destination rank for URMA and RDMA; unused by SDMA. |
| `events` | Zero or more prerequisite PTO pipeline events. |

The return value is an `AsyncEvent`. Its completion covers both the payload transfer and the following signal update.

### Signal and `signalValue`

`comm::Signal` is the global-tensor alias for one `int32_t` signal:

```cpp
using Signal = GlobalTensor<int32_t,
                            Shape<1, 1, 1, 1, 1>,
                            Stride<1, 1, 1, 1, 1>,
                            Layout::ND>;
```

Constructing a `Signal` only wraps a caller-provided GM address; it does not allocate or initialize the underlying
memory. The caller allocates the signal and initializes it according to the communication protocol. `Set` assigns
`signalValue` to the signal. `AtomicAdd` uses `signalValue` as a signed increment.

## Operation Semantics

One invocation executes in this order:

1. Wait for all `events`.
2. Transfer the complete payload from `srcGlobalData` to `dstGlobalData`.
3. After the payload transfer completes, update `dstSignalData` according to `notifyOp`.

For `NotifyOp::Set`:

$$
\mathrm{signal}^{\mathrm{remote}} = \mathrm{signalValue}
$$

For `NotifyOp::AtomicAdd`:

$$
\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{signalValue} \quad (\text{atomic})
$$

`AtomicAdd` updates the signal atomically. Multiple producers may update the same signal concurrently, and the final
increment is the sum of their `signalValue` values. This atomicity does not apply to payload writes; concurrently
transferred payloads from multiple producers must have non-overlapping destination ranges.

The intrinsic does not guarantee a final value when multiple producers concurrently apply `Set` to one signal.
Without application-level synchronization, do not concurrently mix `Set`, `AtomicAdd`, or ordinary stores on the
same signal.

`DmaEngine::RDMA` implements `AtomicAdd` on HNS1825 with an INT32 SUM Write Reduce ordered after the payload WRITE.
The payload-before-signal order above applies only within one invocation and does not define ordering between
different sessions or independent execution flows.

## AsyncSession Construction

Use `BuildAsyncSession` from `include/pto/comm/async_common/async_event_impl.hpp`. It provides an engine-specific
construction interface for each backend. The function returns `false` when construction fails; only a successfully
built session can be used for asynchronous intrinsics and event waits.

### SDMA Construction (default)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    AsyncSession &session,
    uint32_t syncId = 0,
    const sdma::SdmaBaseConfig &baseConfig = {
        sdma::kDefaultSdmaBlockBytes, 0, 1},
    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| Parameter | Default | Description |
|---|---|---|
| `scratchTile` | — | UB scratch tile for SDMA control metadata. |
| `workspace` | — | GM pointer allocated by the caller and initialized by the host-side `SdmaWorkspaceManager`. |
| `session` | — | Output `AsyncSession`. |
| `syncId` | `0` | MTE3/MTE2 pipeline synchronization event ID in the range 0-7. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | SDMA block bytes, communication-block offset, and queue count. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel-group index; defaults to `get_block_idx()`. |

On A2/A3, concurrent AIVs must build separate sessions and use distinct Channel Groups. The default
`kAutoChannelGroupIdx` selects a group from `get_block_idx()`; callers that pass explicit groups must likewise give
each concurrent AIV exclusive ownership of one group. `queue_num` may be greater than one, but one
`TPUT_ASYNC_NOTIFY` call submits its payload and signal only through queue 0 of that group to preserve payload-before-
signal ordering. Increasing `queue_num` does not stripe that notify payload across queues.

### URMA Construction (NPU_ARCH 3510 only)

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    AsyncSession &session);
#endif
```

| Parameter | Description |
|---|---|
| `workspace` | GM pointer allocated by the caller and initialized by the host-side `UrmaWorkspaceManager`. |
| `session` | Output `AsyncSession`. |

URMA does not require `scratchTile` and requires CANN Toolkit >= 9.1.0. The session does not bind a destination rank;
the intrinsic selects it through `peer`.

### RDMA Construction (NPU_ARCH 3510 only)

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
| `scratchTile` | UB/Vec scratch tile used by RDMA; at least 64 bytes. |
| `workspace` | GM pointer returned by the host-side RDMA initialization flow. |
| `myPe` | Local rank ID. |
| `session` | Output `AsyncSession`. |
| `syncId` | MTE/scalar synchronization event ID in the range 0-7. |
| `qpIdx` | QP index for each peer; defaults to 0 and must be less than the host-configured [`qpNum`](TPUT_ASYNC.md#rdma-construction-npu_arch-3510-only). |

An RDMA session does not bind a destination peer. Select the target rank with the intrinsic's `peer` parameter:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
        scratchTile, rdmaWorkspace, myPe, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### scratchTile

`scratchTile` is temporary UB workspace used by SDMA and RDMA sessions; it does not contain the user payload. It must
be a `pto::Tile` in UB/Vec memory and remain valid until the associated events have completed.

- SDMA requires at least 8 available bytes. A common type is
  `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256 bytes).
- RDMA requires at least 64 available bytes.

## Constraints

- `GlobalSrcData::RawDType` must equal `GlobalDstData::RawDType`.
- `GlobalSrcData::layout` must equal `GlobalDstData::layout`.
- Source and destination payload tensors must be flat, contiguous logical 1D tensors.
- The destination element capacity must be at least the source element count.
- The payload size must be greater than zero. Use `TNOTIFY` for a signal-only operation.
- `dstSignalData` must contain exactly one `int32_t` in remote GM. Its address must be non-null and 4-byte aligned.
  The caller is responsible for allocation and initialization.
- The payload destination range must not overlap `dstSignalData`.
- For URMA and RDMA, the payload destination and signal must belong to the same target peer. The complete local
  payload, remote payload, and remote signal ranges must lie in memory regions registered during host initialization.
- A URMA payload has no per-call size cap: a payload larger than 256 MB (the single-WQE transfer limit) is automatically split into multiple WQEs by the implementation, transparently to the caller, and the signal is still guaranteed to complete after all payload chunks. One RDMA payload must not exceed `0x7fffffff` bytes.
- The SDMA workspace must be initialized by the host-side `SdmaWorkspaceManager`. The URMA workspace must be
  initialized by the host-side `UrmaWorkspaceManager`.
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be device memory that HCCL can register;
  its allocation method must satisfy the requirements of the active CANN/HCCL runtime.
- Keep the session, workspace, and scratch tile alive until all associated events have completed.

## Completion Semantics

- `event.Wait(session)` blocks until the event completes.
- `event.Test(session)` tests completion without blocking.
- Both methods must use the same session that issued the operation.
- Successful completion covers the complete payload transfer and the following signal update.

Asynchronous operations in one session and one backend queue complete in submission order. After several operations
have been submitted, waiting for the last event in that queue also covers earlier outstanding operations. URMA or
RDMA queues for different peers complete independently, so wait for the last event of each peer separately. `Wait`
and `Test` do not require the peer argument again.

When the receiver observes the signal update, the corresponding payload has been transferred to remote GM. The
intrinsic does not guarantee that an existing receiver-side payload cache entry has been updated. Before reading the
payload, the caller must ensure visibility according to the target platform and runtime memory-consistency rules.

## Concurrency and Session Ownership

- A2/A3 SDMA: AIVs using separate sessions and Channel Groups may concurrently access the same rank or different
  ranks.
- URMA: Different AIVs may concurrently access different peers; accesses to the same peer must be serialized.
- Concurrent payload ranges must not overlap. Use `AtomicAdd` for a shared signal and separate signals for `Set`.

- Do not use one session concurrently from multiple execution flows.
- For A2/A3 SDMA, every concurrent AIV must use a separate session and a distinct Channel Group. Multiple AIVs must
  not submit concurrently to one group. With `queue_num == N`, at most `kSdmaMaxChannelGroups / N` groups are valid.
- Concurrent `Set` producers should use separate remote signals. Use `AtomicAdd` when multiple AIVs share one signal
  as a completion counter. Payload destination ranges must not overlap in either mode.
- For URMA, submissions to the same peer/QP must be serialized even when callers constructed separate
  sessions over the same workspace. Different peers use independent queues.
- Complete all earlier events before rebuilding a session or reusing its backend queue.
- When using URMA, complete the last event for every active peer/QP and synchronize every host stream that uses the
  `CommContext` before releasing communication resources. If an external HCCL communicator is supplied through
  `existingComm`, also stop its other users and destroy the communicator to release its channels/MRs before calling
  `DestroyComm`, `Reset`, rebuilding with `BuildComm`, or destroying the `CommContext`.

## Examples

The following examples assume that the host communication runtime has translated the remote addresses and initialized
the workspace for the selected engine.

### SDMA Set

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void PutAndNotifySdma(__gm__ T *remoteDst,
                                       __gm__ T *localSrc,
                                       __gm__ int32_t *remoteSignalPtr,
                                       __gm__ uint8_t *sdmaWorkspace,
                                       uint32_t peer)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstGlobalData(remoteDst, shape, stride);
    GT srcGlobalData(localSrc, shape, stride);
    comm::Signal remoteSignal(remoteSignalPtr);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
            scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### SDMA AtomicAdd

The SDMA `AtomicAdd` example uses the same session and tensor construction as the preceding example:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
        scratchTile, sdmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::AtomicAdd, session, peer);
    (void)event.Wait(session);
}
```

### URMA Set

The URMA session does not bind a destination rank. The intrinsic selects the target through `peer`:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::URMA>(
        urmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### URMA AtomicAdd

URMA `AtomicAdd` uses the same session construction:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::URMA>(
        urmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::AtomicAdd, session, peer);
    (void)event.Wait(session);
}
```

### RDMA Set and AtomicAdd

RDMA uses `peer` to select the target rank. The example assigns the signal with `Set`; use `NotifyOp::AtomicAdd`
in the same call to add `signalValue` to the existing signal:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
        scratchTile, rdmaWorkspace, myPe, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### Receiver

```cpp
comm::Signal ready(localSignalPtr);
comm::TWAIT(ready, 1, comm::WaitCmp::EQ);

// Ensure payload visibility according to the target platform and runtime rules before reading it.
```
