<!-- Generated from `docs/isa/comm/README.md` -->

# PTO Communication ISA Reference

This directory contains the per-instruction reference for the PTO Communication ISA. Communication operations enable data movement and synchronization across execution agents and parallel ranks.

## Naming Convention

`pto.t*` is the IR/assembly spelling; the corresponding `T*` is the C++ intrinsic spelling. Both refer to the same operation. This manual documents both spellings on each page.

## Point-to-Point Communication (Synchronous)

- [**TGET / pto.tget**](./TGET.md): Remote read — copy data from remote NPU GM to local GM via UB staging tile
- [**TPUT / pto.tput**](./TPUT.md): Remote write — copy data from local GM to remote NPU GM via UB staging tile

## Point-to-Point Communication (Asynchronous)

- [**TGET_ASYNC / pto.tget_async**](./TGET_ASYNC.md): Asynchronous remote read
- [**TPUT_ASYNC / pto.tput_async**](./TPUT_ASYNC.md): Asynchronous remote write

## Signal-Based Synchronization

- [**TNOTIFY / pto.tnotify**](./TNOTIFY.md): Send notification to remote NPU
- [**TWAIT / pto.twait**](./TWAIT.md): Blocking wait for signal condition
- [**TTEST / pto.ttest**](./TTEST.md): Non-blocking test of signal condition

## Collective Communication

- [**TBROADCAST / pto.tbroadcast**](./TBROADCAST.md): Broadcast from root NPU to all ranks
- [**TGATHER / pto.tgather**](./TGATHER.md): Gather data from all ranks to root
- [**TSCATTER / pto.tscatter**](./TSCATTER.md): Scatter data from root to all ranks
- [**TREDUCE / pto.treduce**](./TREDUCE.md): Collective reduction across all ranks to root

## Type Definitions

These are normative specifications, not implementation declarations. Actual values are defined by each target profile.

### NotifyOp

Operation type for `TNOTIFY`:

| Value | Description |
|-------|-------------|
| `NotifyOp::Set` | Direct set (`signal = value`) |
| `NotifyOp::AtomicAdd` | Atomic add (`signal += value`) |

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

### ReduceOp

Reduction operators for `TREDUCE`:

| Value | Description |
|-------|-------------|
| `ReduceOp::Sum` | Element-wise sum |
| `ReduceOp::Max` | Element-wise maximum |
| `ReduceOp::Min` | Element-wise minimum |

### AtomicType

Atomic operation type for `TPUT`:

| Value | Description |
|-------|-------------|
| `AtomicType::AtomicNone` | No atomic operation (default) |
| `AtomicType::AtomicAdd` | Atomic add operation |

### DmaEngine

DMA backend selection for `TPUT_ASYNC` and `TGET_ASYNC`:

| Value | Description |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA engine (supports 2D transfer) |
| `DmaEngine::URMA` | URMA engine (supports 1D transfer; availability is **profile-specific** and **MUST** be verified against the target profile specification) |

### AsyncEvent

Returned by `TPUT_ASYNC` / `TGET_ASYNC`. Represents an outstanding asynchronous DMA transfer. Programs use `AsyncEvent` to poll or block until the transfer completes:

- A valid event **MUST** be tested with the corresponding `AsyncSession`
- An invalid event (e.g., handle value of zero) indicates the operation completed synchronously or failed

### AsyncSession

Engine-agnostic session for async DMA operations. Programs build a session once and pass it to all async calls. The session encapsulates the engine type, scratch buffer, and workspace needed for asynchronous progress.

### ParallelGroup

Wrapper for collective communication across multiple ranks. Encapsulates:

- An array of `GlobalData` objects (each wraps a GM address; addresses may be local or remote depending on the collective operation)
- The number of participating ranks
- The root rank index for root-based collectives

## Source Of Truth

The authoritative specification for communication operation behavior is the PTO ISA manual. Backend implementations in `include/pto/comm/` are **informative** and may reflect implementation details that are not part of the ISA guarantee.
