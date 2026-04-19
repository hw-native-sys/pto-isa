# TGET_ASYNC

`TGET_ASYNC` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Asynchronous remote read: initiates a transfer from a remote NPU's global memory to local global memory and returns an `AsyncEvent` immediately without blocking. The event is used later to wait for transfer completion.

Two DMA engines are supported: SDMA (default, available on all targets) and URMA (hardware RDMA, available on Ascend950 / NPU_ARCH 3510 only).

## Mechanism

`TGET_ASYNC` starts a DMA transfer from remote GM to local GM and returns immediately:

```
srcGlobalData (remote GM) → DMA engine → dstGlobalData (local GM)
```

The `AsyncSession` manages the engine-agnostic async state. After issuing one or more async operations, call `event.Wait(session)` to block until all pending operations complete (quiet semantics — a single `Wait` drains all operations issued since the last `Wait`).

### Engine differences

- **SDMA**: Submits data transfer SQEs; flag SQE is deferred to `Wait`, which polls for completion.
- **URMA**: Submits an RDMA READ WQE and rings the doorbell immediately; `Wait` polls the Completion Queue.

## Syntax

### PTO Assembly Form

Engine selection is a template parameter at the C++ level. The assembly form does not expose the engine choice.

### Template Parameter

|| Value | Description |
||-------|-------------|
|| `DmaEngine::SDMA` | Default. System DMA — available on all targets. |
|| `DmaEngine::URMA` | User-level RDMA — Ascend950 (NPU_ARCH 3510) only. |

> **SDMA limitation**: Currently supports **only flat contiguous logical 1D tensors**. Non-1D or non-contiguous layouts are not supported. If this requirement is not met, the implementation returns an invalid async event (`handle == 0`).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

### AsyncSession construction (SDMA)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

|| Parameter | Default | Description |
||-----------|---------|-------------|
|| `scratchTile` | — | UB scratch tile for SDMA control metadata |
|| `workspace` | — | GM pointer from host-side `SdmaWorkspaceManager` |
|| `session` | — | Output `AsyncSession` object |
|| `syncId` | `0` | MTE3/MTE2 pipe sync event id (0–7) |
|| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}` |
|| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index; defaults to current AI core |

### AsyncSession construction (URMA, NPU_ARCH 3510 only)

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

|| Parameter | Description |
||-----------|-------------|
|| `workspace` | GM pointer from host-side `UrmaWorkspaceManager` |
|| `destRankId` | Source rank id (remote NPU for `TGET_ASYNC`) |
|| `session` | Output `AsyncSession` object |

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `dstGlobalData` | `GlobalTensor` | Local destination; must be flat contiguous 1D |
|| `srcGlobalData` | `GlobalTensor` | Remote source; must be flat contiguous 1D |
|| `session` | `AsyncSession` | Engine-agnostic session object |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the get |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `AsyncEvent` | event | Handle for later `Wait` call; drain via `event.Wait(session)` |

## Side Effects

This operation initiates a DMA transfer from remote global memory to local global memory. Completion is deferred to the `Wait` call.

## Constraints

### Type constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both SDMA and URMA require **flat contiguous logical 1D tensors** only.

### Memory constraints

- SDMA: `workspace` must be allocated by host-side `SdmaWorkspaceManager`.
- URMA: `workspace` must be allocated by host-side `UrmaWorkspaceManager`; the buffer must be backed by huge-page memory (`ACL_MEM_MALLOC_HUGE_ONLY`).

### Platform constraints

- URMA is available on NPU_ARCH 3510 (Ascend950) only.

## scratchTile Role (SDMA)

`scratchTile` does **not** hold payload data. It is converted to `TmpBuffer` and used as temporary UB workspace for SDMA control words (flag, sq_tail, channel_info), polling completion flags, and committing queue tail. The payload path is always remote GM → DMA engine → local GM.

Requirements: must be `pto::Tile` with `TileType::Vec`, at least 8 bytes. Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256B).

## Target-Profile Restrictions

- SDMA is available on all targets. URMA is Ascend950-only.
- CPU simulation does not support async communication operations.
- The `AsyncSession` is engine-agnostic; switching engines requires recompilation.

## Examples

### Single transfer (SDMA)

```cpp
template <typename T>
__global__ AICORE void SimpleGet(__gm__ T *localDst, __gm__ T *remoteSrc,
                                 __gm__ uint8_t *sdmaWorkspace) {
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
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session))
        return;

    auto event = comm::TGET_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### Batch transfer — quiet semantics

```cpp
comm::AsyncEvent lastEvent;
for (int rank = 0; rank < nranks; ++rank) {
    GT dstG(localDstBase + rank * 1024, shape, stride);
    GT srcG(remoteSrcBase + rank * 1024, shape, stride);
    lastEvent = comm::TGET_ASYNC(dstG, srcG, session);
}
(void)lastEvent.Wait(session);  // single Wait drains all pending ops
```

### URMA (Ascend950)

```cpp
comm::AsyncSession session;
if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, srcRankId, session))
    return;

auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
(void)event.Wait(session);
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Synchronous counterpart: [TGET](./TGET.md)
- Async write: [TPUT_ASYNC](./TPUT_ASYNC.md)
- Instruction set: [Other and Communication](../other/README.md)
