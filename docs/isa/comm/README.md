# PTO Communication ISA Reference

This directory contains the per-instruction reference for the PTO Communication ISA.

- Source of truth (C++ intrinsics): `include/pto/comm/pto_comm_inst.hpp`
- Type definitions: `include/pto/comm/comm_types.hpp`

## Point-to-Point Communication (Synchronous)
- [**TPUT**](TPUT.md): Remote write (GM → UB → GM)
- [**TGET**](TGET.md): Remote read (GM → UB → GM)

## Point-to-Point Communication (Asynchronous)
- [**TPUT_ASYNC**](TPUT_ASYNC.md): Asynchronous remote write (GM → DMA engine → GM)
- [**TPUT_ASYNC_NOTIFY**](TPUT_ASYNC_NOTIFY.md): Remote write followed by a remote `int32_t` signal update
- [**TGET_ASYNC**](TGET_ASYNC.md): Asynchronous remote read (GM → DMA engine → GM)

## Signal-Based Synchronization
- [**TNOTIFY**](TNOTIFY.md): Send notification to remote NPU
- [**TWAIT**](TWAIT.md): Blocking wait for signal condition
- [**TTEST**](TTEST.md): Non-blocking test signal condition

## Collective Communication

- [**TGATHER**](TGATHER.md): Gather data from all ranks
- [**TSCATTER**](TSCATTER.md): Scatter data to all ranks
- [**TREDUCE**](TREDUCE.md): Reduce data from all ranks to local
- [**TBROADCAST**](TBROADCAST.md): Broadcast from current NPU to all ranks

## Type Definitions

### NotifyOp

Operation type for `TNOTIFY`:

| Value | Description |
|-------|-------------|
| `NotifyOp::AtomicAdd` | Atomic add (`signal += value`) |
| `NotifyOp::Set` | Direct set (`signal = value`) |

### WaitCmp

Comparison operators for `TWAIT` and `TTEST`:

| Value | Description |
|-------|-------------|
| `WaitCmp::EQ` | Equal (`==`) |
| `WaitCmp::NE` | Not equal (`!=`) |
| `WaitCmp::GT` | Greater than (`>`) |
| `WaitCmp::GE` | Greater or equal (`>=`) |
| `WaitCmp::LT` | Less than (`<`) |
| `WaitCmp::LE` | Less or equal (`<=`) |

```cpp
// Usage (unified runtime parameter style):
comm::TNOTIFY(signal, 1, comm::NotifyOp::Set);
comm::TWAIT(signal, 1, comm::WaitCmp::EQ);
comm::TTEST(signal, 1, comm::WaitCmp::GE);
```

### ReduceOp

Reduction operators for `TREDUCE`:

| Value | Description |
|-------|-------------|
| `ReduceOp::Sum` | Element-wise sum |
| `ReduceOp::Max` | Element-wise maximum |
| `ReduceOp::Min` | Element-wise minimum |

### AtomicType

Atomic operation type for `TPUT` (defined in `include/pto/common/constants.hpp`):

| Value | Description |
|-------|-------------|
| `AtomicType::AtomicNone` | No atomic operation (default) |
| `AtomicType::AtomicAdd` | Atomic add operation |

### DmaEngine

DMA backend selection for `TPUT_ASYNC`, `TPUT_ASYNC_NOTIFY`, and `TGET_ASYNC`:

| Value | Description |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA engine (supports 1D transfer) |
| `DmaEngine::URMA` | URMA engine (supports 1D transfer, Ascend950 / NPU_ARCH 3510 only; requires CANN >= 9.1.0) |
| `DmaEngine::RDMA` | RDMA engine (supports 1D transfer, Ascend950 / NPU_ARCH 3510 only; currently supports only the HNS1825 NIC platform) |

### AsyncSubmitMode

Value of `AsyncSession::submitMode`, controlling whether `TPUT_ASYNC` submits immediately or stages operations into a batch:

| Value | Description |
|-------|-------------|
| `AsyncSubmitMode::IMMEDIATE` | Submit immediately (default). A staged batch, if any, is submitted first |
| `AsyncSubmitMode::DEFER` | Stage this operation; submit automatically once the staged count reaches `batchSize` |
| `AsyncSubmitMode::DEFER_AND_SUBMIT` | Stage this operation and submit the whole batch immediately |

Batching is supported only by `TPUT_ASYNC<SDMA>` on A2/A3 and `TPUT_ASYNC<URMA>` on Ascend 950. See [TPUT_ASYNC Batch Submission](TPUT_ASYNC.md#batch-submission).

### AsyncEvent

Returned by `TPUT_ASYNC` / `TPUT_ASYNC_NOTIFY` / `TGET_ASYNC`. Use to synchronize completion:

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                           // only means handle != 0
    bool Wait(const AsyncSession &session) const; // block until transfer completes
    bool Test(const AsyncSession &session) const; // non-blocking completion check

    // Completion records for consumers that cannot run device-side polling intrinsics
    uint32_t CompletionRecordCount(const AsyncSession &session) const;
    AsyncCompletionRecord CompletionRecordAt(const AsyncSession &session, uint32_t idx) const;
};
```

The return values of `Wait` and `Test` are the only way to judge the outcome: `true` means the operation completed, `false` means submission failed. `handle == 0` means there is nothing to wait for (an empty transfer, or the synchronous SDMA path on Ascend 950); `Wait` and `Test` then return `true` immediately. `valid()` only reflects `handle != 0` and must not be used to judge success.

`CompletionRecordCount` returns 0 when `handle == 0`. For a non-zero handle that cannot be resolved it returns `kCompletionRecordCountError` (`UINT32_MAX`), which must be treated as an error. `CompletionRecordAt` returns an empty record (`addr == nullptr`) when the record cannot be resolved.

### AsyncCompletionRecord

A completion record describes the completion predicate of one transfer, for consumers that cannot run device-side polling intrinsics. The consumer compares the value at `addr` against `expected` according to `kind`, without knowing the backend queue layout:

```cpp
enum class CompletionKind : uint32_t {
    SDMA_POST_DONE = 1,
    URMA_CQE_DW0 = 2,
    RDMA_HNS1825_CQE = 3,
};

struct AsyncCompletionRecord {
    __gm__ uint8_t *addr;
    uint64_t expected;
    CompletionKind kind;
};
```

### AsyncSession

Engine-agnostic session for async DMA operations. Build once, pass to all async calls:

```cpp
comm::AsyncSession session;
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, workspace, session);
session.submitMode = comm::AsyncSubmitMode::DEFER;  // optional; defaults to IMMEDIATE, one op per submit
session.batchSize = 8U;                             // optional; batch submission threshold
```

Defined in `include/pto/comm/async_common/async_types.hpp`. See [TPUT_ASYNC](TPUT_ASYNC.md) for construction parameters and batch submission.

### ParallelGroup

Wrapper for collective communication across multiple NPUs:

```cpp
template <typename GlobalData>
struct ParallelGroup {
    // Pointer to an array of `GlobalData` objects (each wraps a GM address).
    // The array itself is local metadata; the wrapped addresses may refer to local or remote GM,
    // depending on the collective instruction.
    GlobalData *tensors;
    int nranks;   // Number of ranks
    int rootIdx;  // Root NPU's rank index

    // Factory function (recommended): build from an existing tensor array.
    static ParallelGroup Create(GlobalData *tensorArray, int size, int rank_id);
};
```
