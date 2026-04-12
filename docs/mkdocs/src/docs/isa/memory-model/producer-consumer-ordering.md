<!-- Generated from `docs/isa/memory-model/producer-consumer-ordering.md` -->

# Producer-Consumer Ordering

Producer-consumer ordering is the most useful way to explain PTO visibility rules. A program is legal when each consumer sees the writes or state changes its producer is required to make visible, using the synchronization and movement rules of the active surface.

## Producer-Consumer State Machine

Every data movement or compute operation participates in a producer-consumer chain. The state machine for each operation is:

```
┌─────────────────┐
│   IDLE          │  Operation not yet issued
└────────┬────────┘
         │ issue
         ▼
┌─────────────────┐
│  IN_PROGRESS    │  Operation executing (may be on different pipeline)
└────────┬────────┘
         │ completion (produces event)
         ▼
┌─────────────────┐
│   COMPLETE      │  Result visible to consumers who have
└────────┬────────┘    established the ordering edge
         │ (consumed by next operation)
         ▼
    [Consumer]
```

An operation is **consumed** by a subsequent operation when the consumer either:

1. Passes the producer's `RecordEvent` as a `WaitEvents` argument.
2. Issues a `wait_flag` for the same event that the producer issued.

## Tile Surface Ordering

For `pto.t*` programs, the common pattern is:

```
┌─────────────────────────────────────────────────────┐
│  TLOAD(tile, gtensor)                               │
│  (produces tile state)                              │
└─────────────────┬───────────────────────────────────┘
                  │ RecordEvent or implicit TSYNC
                  ▼
┌─────────────────────────────────────────────────────┐
│  Tile Compute (TADD, TMATMUL, etc.)                │
│  (consumes tile state; produces tile state)         │
└─────────────────┬───────────────────────────────────┘
                  │ RecordEvent or explicit TSYNC
                  ▼
┌─────────────────────────────────────────────────────┐
│  TSTORE(gtensor, tile)                              │
│  (consumes tile state; produces GM write)           │
└─────────────────────────────────────────────────────┘
```

### RecordEvent Chaining

The `RecordEvent` return value of each tile operation can be passed to the next operation as a `WaitEvents...` argument:

```cpp
RecordEvent e0 = TLOAD(a, ga);     // e0: TLOAD has completed
RecordEvent e1 = TLOAD(b, gb);     // e1: TLOAD has completed
TMATMUL(c, a, b, e0, e1);         // waits for e0 and e1 before starting
RecordEvent e2 = TMATMUL(...);
TSTORE(gc, c, e2);                  // waits for e2 before starting
```

When an operation has multiple `WaitEvents...` arguments, it waits for ALL of them before beginning execution.

### TSYNC

`TSYNC` provides a lightweight tile-buffer-scoped barrier when fine-grained event chaining is not needed:

```cpp
TLOAD(a, ga);
TLOAD(b, gb);
TSYNC();        // ensures both loads are complete before compute
TADD(c, a, b);
TSYNC();        // ensures compute is complete before store
TSTORE(gc, c);
```

`TSYNC` is equivalent to chaining all prior `RecordEvent` values for the same tile buffer.

## Vector Surface Ordering

For `pto.v*` programs, the ordering chain involves explicit DMA synchronization:

```
┌──────────────────────────────────────────────────────────┐
│  copy_gm_to_ubuf(%ub, %gm, ...)                         │
│  (DMA: GM → UB)                                          │
└────────────────┬─────────────────────────────────────────┘
                 │ set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)
                 ▼
┌──────────────────────────────────────────────────────────┐
│  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)                 │
│  (UB data now visible to Vector pipeline)                 │
└────────────────┬─────────────────────────────────────────┘
                 │ (implicit on vlds)
                 ▼
┌──────────────────────────────────────────────────────────┐
│  vlds %vreg, %ub[...] {dist = "NORM"}                   │
│  (UB → Vector Register)                                  │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  vadd %result, %vreg, %vreg                             │
│  (Vector Compute on Vector Register)                     │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  vsts %result, %ub[...]                                 │
│  (Vector Register → UB)                                   │
└────────────────┬─────────────────────────────────────────┘
                 │ set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1)
                 ▼
┌──────────────────────────────────────────────────────────┐
│  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1)                 │
│  (Vector result now staged for DMA)                      │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  copy_ubuf_to_gm(%gm, %ub, ...)                         │
│  (DMA: UB → GM)                                          │
└──────────────────────────────────────────────────────────┘
```

### Vector Surface vs Tile Surface Ordering

| Aspect | Tile Surface | Vector Surface |
|--------|-------------|---------------|
| Synchronization mechanism | `RecordEvent`, `TSYNC` | `set_flag`/`wait_flag` on pipe pairs |
| Data path | GM ↔ Tile Buffer (via MTE2/MTE3) | GM ↔ UB ↔ Vector Register (via DMA + vlds/vsts) |
| Visibility model | Producer-consumer chain via events | DMA signal → wait → vlds → compute → vsts → DMA signal |
| Implicit ordering | Within same tile buffer | None — explicit flag required between DMA and compute |
| Store path | Tile → MTE3 → GM | Vector Register → vsts → UB → MTE3 → GM |

## Cross-Surface Handoff

When a tile-surface result is consumed by a vector-surface operation (or vice versa), the handoff must go through UB:

```
Tile Surface                              Vector Surface
    │                                         ▲
    │  TLOAD/TSTORE handles GM ↔ Tile Buffer  │
    │                                              │
    └──── TSTORE → UB → copy_ubuf_to_gm ──────────┘
         (via copy_gm_to_ubuf on vector side)
```

The cross-surface handoff goes through GM or through an explicit UB double-buffering pattern:

```cpp
// Tile surface produces result in tile c
TSTORE(gc, c);

// Vector surface consumes from gc
copy_gm_to_ubuf(%ub, %gm_out, ...);
set_flag(PIPE_MTE2, PIPE_V, ID);
wait_flag(PIPE_MTE2, PIPE_V, ID);
%v = pto.vlds %ub[...] {dist = "NORM"};
```

## Constraints

- A consumer may only rely on visibility after the required producer-consumer edge is established.
- The exact synchronization mechanism may vary by surface or target profile.
- Family docs and per-op pages must state the relevant ordering expectations explicitly.
- An operation's `RecordEvent` return value is only valid for chaining to operations that execute AFTER the current operation in program order.

## Cases That Are Not Allowed

- Describing a consumer as legal without saying how producer visibility is established.
- Assuming a target's convenient scheduling behavior is the architecture contract.
- Leaving cross-surface handoff rules implicit.
- Issuing `vlds` before `copy_gm_to_ubuf` completes without an intervening `wait_flag`.
- Issuing `copy_ubuf_to_gm` before `vsts` completes without an intervening `wait_flag`.
- Passing a `RecordEvent` from a later operation to an earlier operation (wrong direction) — this is illegal and produces a verification error.

## See Also

- [Consistency Baseline](./consistency-baseline.md)
- [Ordering And Synchronization](../machine-model/ordering-and-synchronization.md)
- [Tile Instruction Surface](../instruction-surfaces/tile-instructions.md)
- [Vector Instruction Surface](../instruction-surfaces/vector-instructions.md)
