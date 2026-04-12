<!-- Generated from `docs/isa/other/communication-and-runtime.md` -->

# Communication And Runtime

Communication operations span multiple NPUs in a parallel group. They express inter-NPU data exchange and collective reduction using a `ParallelGroup` handle.

> **Note on naming:** `pto.tget` (IR form) and `TGET` (C++ intrinsic form) refer to the same operation — remote read from a peer NPU's GM. Both spellings appear in this manual depending on context.

## Operations

| Operation | Description | Collective Type | IR Spelling | C++ Spelling |
|-----------|-------------|----------------|-------------|--------------|
| [TBROADCAST](./TBROADCAST.md) | Broadcast data from root NPU to all ranks | One-to-all | `pto.tbroadcast` | `TBROADCAST` |
| [TGET](./TGET.md) | Get data from a remote NPU | Point-to-point | `pto.tget` | `TGET` |
| [TGET_ASYNC](./TGET_ASYNC.md) | Asynchronous variant of TGET | Point-to-point | `pto.tget_async` | `TGET_ASYNC` |
| [TNOTIFY](./TNOTIFY.md) | Notify other ranks of an event | Synchronization | `pto.tnotify` | `TNOTIFY` |
| [TPUT](./TPUT.md) | Put data to a remote NPU | Point-to-point | `pto.tput` | `TPUT` |
| [TPUT_ASYNC](./TPUT_ASYNC.md) | Asynchronous variant of TPUT | Point-to-point | `pto.tput_async` | `TPUT_ASYNC` |
| [TREDUCE](./TREDUCE.md) | Collective reduction across all ranks | All-to-one | `pto.treduce` | `TREDUCE` |
| [TSCATTER](./TSCATTER.md) | Scatter data from root to all ranks | One-to-all | `pto.tscatter` | `TSCATTER` |
| [TGATHER](./TGATHER.md) | Gather data from all ranks to root | All-to-one | `pto.tgather` | `TGATHER` |
| [TTEST](./TTEST.md) | Test if a notification has been received | Synchronization | `pto.ttest` | `TTEST` |
| [TWAIT](./TWAIT.md) | Wait for a notification | Synchronization | `pto.twait` | `TWAIT` |

## Mechanism

Communication operations use a `ParallelGroup` handle (`!pto.group<N>`) to identify the set of participating NPUs. The group defines:

- **Size**: Number of ranks `N` in the parallel group
- **Root**: The designated NPU for broadcast/scatter operations (typically rank 0)
- **Tensors**: Per-rank destination/source buffers

### Data Flow

All collective communication operations share a common data flow pattern:

```
Local GM ──► UB (staging tile) ──► Inter-NPU interconnect ──► UB ──► Local GM
```

A **staging tile** in UB is always required as an intermediate buffer. For large tensors that exceed the UB tile capacity, the operation automatically performs **2D sliding** — chunking along rows and columns to fit each chunk into the tile, iterating over all outer dimensions.

### Broadcast

All non-root NPUs receive data from the root:

$$ \mathrm{dst}^{(k)} = \mathrm{src}^{(\text{root})} \quad \forall k \in [0, N) $$

Only the root calls `TBROADCAST`. Non-root ranks must ensure their destination buffers are allocated and writable for the duration of the operation.

### Reduce

All ranks contribute data to a reduction operation, with the result delivered to the root:

$$ \mathrm{result}^{(\text{root})} = \bigoplus_{k=0}^{N-1} \mathrm{src}^{(k)} $$

where $\bigoplus$ is the reduction operator (sum, max, min, etc.).

### Scatter/Gather

Scatter distributes slices of the root's data to each rank. Gather collects per-rank data back to the root.

### Point-to-Point (TGET/TPUT)

Point-to-point operations transfer data between two specific NPUs without involving the entire group:

- **`TGET`** (`pto.tget`): Read remote GM → local GM. Data flows from the source NPU to the current NPU.
- **`TPUT`** (`pto.tput`): Write local GM → remote GM. Data flows from the current NPU to the destination NPU.

Both use a staging tile in UB as the intermediate buffer. For `TGET`, the data path is: `remote GM → staging tile → local GM`. For `TPUT`, the data path is: `local GM → staging tile → remote GM`.

## ParallelGroup Handle

```mlir
// Define a parallel group of 8 NPUs
%tensors = "pto.make_group"(%addrs0, %addrs1, ..., %addrs7)
    : (!pto.memref<f32, 16x16>, ..., !pto.memref<f32, 16x16>) -> !pto.group<8>
```

In C++, the `ParallelGroup<GTensor>` template manages the group handle. See the per-op pages for C++ usage examples.

## Large Tile Support

When the GlobalTensor exceeds the UB tile capacity in rows and/or columns, transfers are automatically chunked via 2D sliding:

- If `ValidRow` is static, `GetShape(DIM_3)` must be divisible by `ValidRow`
- If `ValidCol` is static, `GetShape(DIM_4)` must be divisible by `ValidCol`
- To handle non-divisible cases, use tiles with `DYNAMIC` valid row/column

## Constraints

- All participating NPUs must call the collective operation with matching `ParallelGroup` handles
- Non-root ranks must not call broadcast/scatter operations
- Root rank is identified by `parallelGroup.GetRootIdx()`
- Destination/source tensors are assumed to have the same shape and strides across ranks
- The staging tile must be pre-allocated in UB at non-overlapping offsets for ping-pong variants

## Cases That Are Not Allowed

- Calling collective operations with mismatched `ParallelGroup` handles across ranks
- Calling broadcast/scatter on non-root ranks (undefined behavior)
- Using uninitialized or improperly sized destination buffers
- Using overlapping UB offsets for ping/pong staging tiles

## See Also

- [Other families](../instruction-families/other-families.md) — Family overview
- [Other instruction surface](../instruction-surfaces/other-instructions.md) — Surface description
- [Ordering and Synchronization](../machine-model/ordering-and-synchronization.md) — PTO synchronization model
