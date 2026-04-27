# Ordering And Synchronization

PTO does not assume that all execution resources are implicitly serialized. The machine model makes ordering visible where data or state moves across instruction sets, pipelines, or shared resources. The synchronization primitives, event model, and producer-consumer ordering contracts are described below.

## Synchronization Primitives

PTO defines four categories of synchronization primitives, one per instruction set:

### Tile Instructions Primitives

| Primitive | Syntax | Description |
|-----------|--------|-------------|
| `TSYNC` | `pto.tsync %events...` or `pto.tsync<Op>` | Wait on explicit `RecordEvent` tokens; or insert a pipeline barrier for a single op class |
| `set_flag` | `pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]` | Signal an event from one pipeline to another |
| `wait_flag` | `pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]` | Wait for a previously-signaled event |

`TSYNC` is the primary tile-instruction set synchronization. The event-wait form `TSYNC(events...)` establishes a **happens-before** edge on each `RecordEvent` token, ensuring all prior tile operations that produced those events are complete. The barrier form `TSYNC<Op>()` inserts a pipeline barrier for all operations of class `Op`.

> **Note:** `pipe_barrier` (`pto.pipe_barrier`) is a scalar and control instructions primitive, not a tile instructions primitive. It appears in the [Scalar Pipeline Sync](../scalar/pipeline-sync.md) instruction set.

### Vector Instructions Primitives

| Primitive | Syntax | Description |
|-----------|--------|-------------|
| `set_flag` / `wait_flag` | `pto.set_flag[...]` / `pto.wait_flag[...]` | Event-based handoff between DMA and vector compute pipelines |
| `mem_bar` | `pto.mem_bar` | Memory fence; ordering boundary for GM↔UB traffic |

On the vector instructions, `set_flag(PIPE_MTE2, PIPE_V, ID)` is issued by the DMA engine (MTE2) to signal the vector pipeline that data is ready. The vector pipeline issues `wait_flag(PIPE_MTE2, PIPE_V, ID)` before consuming the data.

### DMA Primitives

| Primitive | Syntax | Description |
|-----------|--------|-------------|
| `copy_gm_to_ubuf` | `pto.copy_gm_to_ubuf ...` | DMA: GM → UB |
| `copy_ubuf_to_gm` | `pto.copy_ubuf_to_gm ...` | DMA: UB → GM |
| `copy_ubuf_to_ubuf` | `pto.copy_ubuf_to_ubuf ...` | DMA: UB → UB (double-buffering) |

DMA operations do not implicitly synchronize with the compute pipeline. Explicit `set_flag`/`wait_flag` pairs (or equivalent `RecordEvent` chaining) are required wherever a DMA transfer and a compute operation share data.

### Communication Instructions Primitives

| Primitive | Description |
|-----------|-------------|
| `TBROADCAST` | Broadcast data to all participating blocks |
| `TGET` / `TPUT` | Point-to-point communication between blocks |
| `TWAIT` / `TTEST` | Barrier synchronization across blocks |
| `TNOTIFY` / `TREDUCE` | Notification and reduction operations |

## Event Model

PTO uses an **event-based** synchronization model. Events carry ordering information between pipelines.

### Event Lifecycle

```
Producer                                  Consumer
  │                                         │
  │  issue DMA / compute                    │
  │  ▼                                      │
  │  set_flag(SRC_PIPE, DST_PIPE, EVENT_ID)│
  │  (produces the event)                   │
  │                                         │
  │                              wait_flag(SRC_PIPE, DST_PIPE, EVENT_ID)
  │                              (consumes the event)
  │                                         │
  │  data/result available                  │
  ▼                                         ▼
```

An **event** is identified by a triple `(src_pipe, dst_pipe, event_id)`:

| Field | Values | Meaning |
|-------|--------|---------|
| `src_pipe` | `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_V`, `PIPE_M` | Source pipeline that produces the event |
| `dst_pipe` | `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_V`, `PIPE_M` | Destination pipeline that consumes the event |
| `event_id` | 0–15 (profile-specific) | Event slot identifier |

Events are **fire-and-forget** in the ISA contract: producing a flag makes it available to all subsequent waiters on the same `(src_pipe, dst_pipe, event_id)` triple.

### Events and RecordEvent

The C++ intrinsics for tile operations (e.g., `TLOAD`, `TSTORE`, `TMATMUL`) return a `RecordEvent` value. This event can be passed as a `WaitEvents...` argument to subsequent operations, establishing a **happens-before** edge:

```cpp
RecordEvent e0 = TLOAD(a, ga);     // produces event
RecordEvent e1 = TLOAD(b, gb);     // produces event
TMATMUL(c, a, b, e0, e1);          // waits for both e0 and e1 before executing
```

The `RecordEvent` return value is the **ISA-visible mechanism** for chaining tile-instruction set dependencies. This is equivalent to inserting explicit `set_flag`/`wait_flag` pairs but expressed at a higher level.

## Pipeline Dependency Graph

The AI Core contains multiple execution units that operate concurrently. The following diagram shows the dependency relationships:

```
               ┌──────────────────────────────────────────────┐
GM ──► MTE2 ──►│ Unified Buffer / Tile Buffer                 │──► MTE3 ──► GM
               │                                              │
               │  ┌───────────────┐     ┌──────────────────┐  │
GM ──► MTE1 ──►│  │ Tile Register │────►│ Vector Pipeline  │  │
               │  │ File          │     │ (pto.v* ops)     │  │
               │  │ Vec / Mat /   │     └────────┬─────────┘  │
               │  │ Acc locations │              │            │
               │  └───────┬───────┘              │            │
               │          │                      ▼            │
               │          │             ┌──────────────────┐  │
               │          └────────────►│ Matrix Pipeline  │  │
               │                        │ (pto.tmatmul*)   │  │
               │                        └────────┬─────────┘  │
               └─────────────────────────────────┼────────────┘
                                                 │
              Scalar Unit: control flow, address generation, system queries
```

### Dependency Types

| Producer | Consumer | Synchronization Required |
|----------|----------|------------------------|
| MTE2 (DMA GM→UB) | Vector pipeline (vlds) | `set_flag(PIPE_MTE2, PIPE_V, ID)` → `wait_flag` |
| Vector pipeline | MTE3 (store) | `set_flag(PIPE_V, PIPE_MTE3, ID)` → `wait_flag` |
| TLOAD | Tile compute | `RecordEvent` chaining or `TSYNC` |
| Tile compute | TSTORE | `RecordEvent` chaining or `TSYNC` |
| TLOAD | TMATMUL | `RecordEvent` chaining or `set_flag`/`wait_flag` |
| Tile compute (Mat) | Tile compute (Vec) | `set_flag`/`wait_flag` or `TSYNC` |

## Ordering Rules

### Tile Instructions Ordering

Tile-instruction set operations are ordered by **program order** within a single tile buffer, and by **event ordering** across tile buffers. The following rules apply:

1. **Tile-local order**: Within a single tile buffer, operations execute in program order. `TSYNC` establishes a barrier within that tile's ordering stream.
2. **Event ordering**: A `set_flag`/`wait_flag` pair establishes a **happens-before** edge between the producer pipeline and the consumer pipeline.
3. **RecordEvent chaining**: When an operation's `WaitEvents...` arguments include events from prior operations, those prior operations must complete before the current operation begins.

### Vector Instructions Ordering

Vector-instruction set ordering follows these rules:

1. **DMA ordering**: `copy_gm_to_ubuf` must complete (via `set_flag`) before any `vlds` that consumes the copied data.
2. **Compute ordering**: Vector operations within a `SimdVecScopeOp` execute in program order.
3. **Store ordering**: `vsts` must complete (via `set_flag` to MTE3) before `copy_ubuf_to_gm` begins copying the data back to GM.

### GM Visibility

Data written to GM by `TSTORE` or `copy_ubuf_to_gm` is guaranteed visible to subsequent GM reads by other blocks only after:

1. All prior store operations on that block have completed (program order).
2. Any required `mem_bar` or `pipe_barrier` has been issued.
3. The operation has been synchronized with the host runtime (event completion).

## Constraints

- Synchronization is required wherever the architecture does not already guarantee ordering.
- A target may add stronger internal ordering, but the manual must not rely on undocumented strength.
- Vector-pipe synchronization rules must be documented separately from tile-instruction set synchronization rules when the mechanisms differ.
- Events are fire-and-forget; the ISA does not provide a "test-and-clear" event flag.
- `TSYNC` is tile-buffer-scoped; it does not synchronize across tile buffers.

## Cases That Are Not Allowed

- Writing the manual as if synchronization were optional when the architecture requires it.
- Assuming vector-pipe hazards are covered by tile-instruction set rules without saying so.
- Documenting target-specific barriers as architecture-wide unless the PTO instruction set guarantees them.
- Issuing `vlds` before `copy_gm_to_ubuf` completes without an intervening `wait_flag`.
- Issuing `copy_ubuf_to_gm` before `vsts` completes without an intervening `wait_flag`.

## See Also

- [Consistency Baseline](../memory-model/consistency-baseline.md)
- [Producer-Consumer Ordering](../memory-model/producer-consumer-ordering.md)
- [Tile Instruction Set: Sync And Config](../tile/sync-and-config.md)
- [Vector Pipeline Sync](../vector/pipeline-sync.md)
- [Scalar Pipeline Sync](../scalar/pipeline-sync.md)
