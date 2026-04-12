<!-- Generated from `docs/isa/scalar/pipeline-sync.md` -->

# Pipeline Sync

These `pto.*` forms establish explicit producer-consumer ordering across PTO execution stages. They belong to the scalar/control surface even when they coordinate vector-facing pipelines, because what they expose architecturally is dependency state rather than vector payload math.

## Synchronization Hierarchy

The four synchronization modes form a containment hierarchy:

```
Event-based synchronization  (set_flag / wait_flag)
        ↑
Buffer-token protocol  (get_buf / rls_buf)  — requires event-based under the hood
        ↑
Memory barrier  (mem_bar)  — may be used inside vector-visible scope
        ↑
Inter-core coordination  (set_cross_core / wait_flag_dev / set_intra_block / wait_intra_core)
```

- **Event-based** (`set_flag` / `wait_flag`): The foundational mode. Sets or waits on a named event signal between producer and consumer pipes. Used as the primitive for all higher-level modes.
- **Buffer-token** (`get_buf` / `rls_buf`): A protocol built on top of event-based synchronization for double-buffered execution. `get_buf` acquires a buffer token and implicitly sets an event; `rls_buf` releases the token and implicitly sets a dependent event.
- **Memory barrier** (`mem_bar`): Enforces visibility of memory operations within a vector-visible execution scope. Does not establish cross-stage ordering on its own.
- **Inter-core** (`set_cross_core` / `wait_flag_dev` / `set_intra_block` / `wait_intra_core`): Coordinate between execution units or cores. These are profile-restricted and **MAY NOT** be available on all targets.

Programs **MUST NOT** assume that a higher-level mode (e.g., buffer-token) replaces the need for event-based ordering; the protocol requires event-based synchronization underneath.

## What This Family Covers

- event-based synchronization between producer and consumer pipes
- buffer-token protocols for double-buffered execution
- explicit memory barriers inside vector-visible execution scope
- target-profile inter-core coordination forms

## Per-Op Pages

- [pto.set_flag](./ops/pipeline-sync/set-flag.md)
- [pto.wait_flag](./ops/pipeline-sync/wait-flag.md)
- [pto.pipe_barrier](./ops/pipeline-sync/pipe-barrier.md)
- [pto.get_buf](./ops/pipeline-sync/get-buf.md)
- [pto.rls_buf](./ops/pipeline-sync/rls-buf.md)
- [pto.mem_bar](./ops/pipeline-sync/mem-bar.md)
- [pto.set_cross_core](./ops/pipeline-sync/set-cross-core.md)
- [pto.wait_flag_dev](./ops/pipeline-sync/wait-flag-dev.md)
- [pto.set_intra_block](./ops/pipeline-sync/set-intra-block.md)
- [pto.wait_intra_core](./ops/pipeline-sync/wait-intra-core.md)

## Related Material

- [Control and configuration](./control-and-configuration.md)
- [Vector Families: Pipeline Sync](../vector/pipeline-sync.md)
- [Machine Model: Ordering And Synchronization](../machine-model/ordering-and-synchronization.md)
