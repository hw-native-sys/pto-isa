# Communication Instruction Set

The Communication ISA covers inter-NPU collective operations and point-to-point exchange. These instructions span multiple NPUs in a parallel group and require a `ParallelGroup` handle. They are not available on the CPU simulator.

## Instruction Overview

| | Operation | PTO Name | Description |
|-|---------|---------|-------------|
| | Collective broadcast | `pto.tbroadcast` | Broadcast data from root NPU to all ranks in parallel group |
| | Point-to-point get | `pto.tget` / `pto.tget_async` | Get data from a remote NPU; async variant does not block |
| | Point-to-point put | `pto.tput` / `pto.tput_async` | Put data to a remote NPU; async variant does not block |
| | Collective gather | `pto.tgather` | Gather data from all ranks to root NPU |
| | Collective scatter | `pto.tscatter` | Scatter data from root NPU to all ranks |
| | Collective reduction | `pto.treduce` | Collective reduction across all ranks (sum, max, min, etc.) |
| | Notification | `pto.tnotify` | Notify other ranks of an event |
| | Notification test | `pto.ttest` | Test if a notification has been received |
| | Notification wait | `pto.twait` | Wait for a notification |

## Instruction Classes

### Collective Broadcast / Scatter / Gather

These operations replicate or distribute data across all ranks in a `ParallelGroup`. All participating NPUs must call the operation with matching group handles.

| | Operation | PTO Name | Direction | Root Role |
|-|---------|---------|---------|----------|
| | Broadcast | `pto.tbroadcast` | 1 → N | Source of broadcast data |
| | Scatter | `pto.tscatter` | 1 → N | Source of scattered slices |
| | Gather | `pto.tgather` | N → 1 | Destination of gathered data |

### Point-to-Point Exchange

Point-to-point operations transfer data between exactly two ranks. The async variants return immediately without waiting for the transfer to complete.

| | Operation | PTO Name | Blocking? |
|-|---------|---------|----------|
| | Get from remote | `pto.tget` | Yes (blocks until data arrives) |
| | Get from remote | `pto.tget_async` | No (returns event token) |
| | Put to remote | `pto.tput` | Yes (blocks until transfer completes) |
| | Put to remote | `pto.tput_async` | No (returns event token) |

### Collective Reduction

`pto.treduce` performs a collective reduction operation (sum, max, min, prod, etc.) across all ranks in a `ParallelGroup`, with the result available to all participating ranks.

### Runtime Synchronization

Notification primitives allow non-blocking signaling between ranks without data transfer:

| | Operation | PTO Name | Description |
|-|---------|---------|-------------|
| | Notify | `pto.tnotify` | Send a notification to specified ranks |
| | Test | `pto.ttest` | Non-blocking test for received notification |
| | Wait | `pto.twait` | Blocking wait for notification |

## Inputs

Communication instructions consume:

- `ParallelGroup` handles (`!pto.group<N>`)
- Tile operands for the data to be transferred
- Scalar parameters (rank indices, event IDs, reduction operator)
- Staging tiles for receive-side buffering

## Expected Outputs

- Modified tiles on receive-side ranks
- `RecordEvent` tokens from async operations
- Synchronization state changes

## Side Effects

| | Class | Side Effects |
|-|-------|-------------|
| | Collective broadcast/scatter/gather | Network/interconnect traffic; ordering across ranks |
| | Point-to-point | Network traffic between two specific ranks |
| | Collective reduction | Network traffic + arithmetic on interconnect hardware |
| | Runtime sync | Lightweight notification state in interconnect |

## Shared Constraints

- **Communication ops** require all participating NPUs to call the operation with matching `ParallelGroup` handles simultaneously.
- **Non-root ranks** for broadcast/scatter must have destination tiles allocated and writable.
- **CPU simulator** does not support communication ops.
- Async operations require subsequent `pto.twait` before the transferred data is safe to use.

## Constraints

- All participating NPUs **must** call collective operations with matching `ParallelGroup` handles simultaneously.
- Non-root ranks for broadcast/scatter must have destination tiles allocated and writable.
- Async operations require subsequent `pto.twait` before the transferred data is safe to use.
- **CPU simulator**: These instructions are not available. Programs using them on CPU produce a runtime error.

## Cases That Are Not Allowed

- Calling collective operations with mismatched `ParallelGroup` handles across ranks.
- Using uninitialized or improperly sized destination tiles for receive-side operations.
- Accessing data from an async operation before `pto.twait` returns.
- Using these instructions on the CPU simulator profile.

## Syntax

### Assembly Form (PTO-AS)

```asm
pto.tbroadcast %group, %src : (!pto.group<8>, !pto.tile<f32, 16, 16>)
pto.treduce %group, %src, %dst : (!pto.group<8>, !pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) {op = "sum"}
```

### SSA Form (AS Level 1)

```mlir
%result = pto.tbroadcast %group, %src
    : (!pto.group<8>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto::comm;

// Broadcast across a parallel group
template <typename GroupType, typename GlobalData, typename TileData>
PTO_INST RecordEvent BROADCAST(GroupType& group, GlobalData& src, TileData& stagingTile);

// Collective reduction
template <typename GroupType, typename GlobalData, typename TileData, ReduceOp Op>
PTO_INST RecordEvent REDUCE(GroupType& group, GlobalData& src, TileData& stagingTile);
```

## Navigation

See the [Communication ISA reference](../comm/README.md) for the full per-op reference.

## See Also

- [Instruction sets](./README.md) — All instruction sets
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
