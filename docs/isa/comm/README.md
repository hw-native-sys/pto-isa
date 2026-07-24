# Communication ISA

Inter-NPU collective communication, point-to-point exchange, and runtime synchronization.

| | Instruction | PTO Name | Description |
|-|-----------|---------|-------------|
| | [TBROADCAST](./TBROADCAST.md) | `pto.tbroadcast` | Broadcast data from root NPU to all ranks |
| | [TGET](./TGET.md) | `pto.tget` | Get data from a remote NPU |
| | [TGET_ASYNC](./TGET_ASYNC.md) | `pto.tget_async` | Asynchronous variant of TGET |
| | [TNOTIFY](./TNOTIFY.md) | `pto.tnotify` | Notify other ranks of an event |
| | [TPUT](./TPUT.md) | `pto.tput` | Put data to a remote NPU |
| | [TPUT_ASYNC](./TPUT_ASYNC.md) | `pto.tput_async` | Asynchronous variant of TPUT |
| | [TREDUCE](./TREDUCE.md) | `pto.treduce` | Collective reduction across all ranks |
| | [TSCATTER](./TSCATTER.md) | `pto.tscatter` | Scatter data from root NPU to all ranks |
| | [TGATHER](./TGATHER.md) | `pto.tgather` | Gather data from all ranks to root NPU |
| | [TTEST](./TTEST.md) | `pto.ttest` | Test if a notification has been received |
| | [TWAIT](./TWAIT.md) | `pto.twait` | Wait for a notification |

See [Communication and Runtime](communication-runtime.md) for the instruction set contract.

## Point-to-Point Communication (Asynchronous)
- [**TPUT_ASYNC**](TPUT_ASYNC.md): Asynchronous remote write (GM → DMA engine → GM)
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

DMA backend selection for `TPUT_ASYNC` and `TGET_ASYNC`:

| Value | Description |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA engine (supports 1D transfer) |
| `DmaEngine::URMA` | URMA engine (supports 1D transfer, Ascend950 / NPU_ARCH 3510 only; requires CANN >= 9.1.0) |

### AsyncEvent

Returned by `TPUT_ASYNC` / `TGET_ASYNC`. Use to synchronize completion:

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                        // true if handle != 0
    bool Wait(const AsyncSession &session) const; // block until transfer completes
    bool Test(const AsyncSession &session) const; // non-blocking completion check
};
```

### AsyncSession

Engine-agnostic session for async DMA operations. Build once, pass to all async calls:

```cpp
comm::AsyncSession session;
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, workspace, session);
```

Defined in `include/pto/comm/async_common/async_types.hpp`. See [TPUT_ASYNC](TPUT_ASYNC.md) for construction details and parameters.

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
