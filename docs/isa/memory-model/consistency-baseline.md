# Consistency Baseline

PTO's memory model is built around **explicit movement** and **explicit ordering**. The baseline guarantee is intentionally narrower than "everything is globally ordered." PTO requires the program or the selected instruction set to express when data becomes visible across stages, instruction sets, and blocks.

## Memory Spaces

PTO defines three architecturally distinct memory spaces:

| Space | Address Qualifier | Scope | Visibility |
|-------|-----------------|-------|-----------|
| **Global Memory (GM)** | `__gm__` | All AI Cores | Shared |
| **Unified Buffer (UB)** | `!pto.ptr<T, ub>` | Single AI Core | Core-local |
| **Tile Buffer** | `!pto.tile_buf<...>` | Single AI Core | Core-local, pipeline-specific |

These are NOT interchangeable. Data must be explicitly moved between them, and each space has different visibility semantics.

## Ordering Levels

PTO defines three levels of ordering guarantee:

| Ordering Level | Description | Scope | How to Establish |
|---------------|-------------|-------|-----------------|
| **Program Order** | Operations within a single tile buffer or vector register file execute in program order | Single core | Implicit within a buffer |
| **Event Order** | Ordering between operations on different pipelines or buffers | Within a block | `set_flag`/`wait_flag` or `RecordEvent` chaining |
| **Barrier Order** | Ordering across multiple blocks | Grid-wide | `TBARRIER` / collective ops |

### Program Order

Within a single tile buffer or a single vector register, operations are ordered by program order:

```c
TLOAD(a, ga);  // 1. Load a
TLOAD(b, gb);  // 2. Load b (ordered after 1, same buffer)
TADD(c, a, b); // 3. Compute (ordered after 1 and 2, same buffer)
```

No explicit synchronization is needed between operations on the same tile buffer.

### Event Order

When data moves between different buffers or different pipelines, explicit event ordering is required:

```c
RecordEvent e0 = TLOAD(a, ga);     // produces event
RecordEvent e1 = TLOAD(b, gb);     // produces event
TMATMUL(c, a, b, e0, e1);          // waits for e0 and e1 before starting
```

The `RecordEvent` return value is a handle to the ordering guarantee. Passing it as a `WaitEvents` argument to a subsequent operation establishes a **happens-before** edge.

### Barrier Order

When multiple blocks must synchronize (e.g., after a collective operation), a grid-wide barrier is required:

```mlir
pto.tbroadcast %tensor, %src : !pto.tile<f32,16,16> -> ()
pto.twait // block until all blocks have received the broadcast
```

## What PTO Does NOT Guarantee Automatically

PTO does not automatically guarantee:

| Guarantee NOT Given | Reason |
|--------------------|--------|
| That every cross-pipeline write is immediately visible to every consumer | Requires explicit `set_flag`/`wait_flag` |
| That vector register, tile buffer, and GM traffic share one implicit fence model | Each space has distinct visibility rules |
| That target-specific stronger ordering is portable | Stronger ordering on A5 does not apply to CPU/A2/A3 |
| That UB writes are visible to GM reads without explicit `TSTORE` or `copy_ubuf_to_gm` | UB→GM requires explicit data movement |
| That GM writes are visible to UB reads without explicit `TLOAD` or `copy_gm_to_ubuf` | GM→UB requires explicit data movement |

## GM Visibility

Data written to GM by `TSTORE` or `copy_ubuf_to_gm` is guaranteed visible to subsequent GM reads by other blocks only after:

1. All prior store operations on that block have completed (program order within block).
2. Any required `mem_bar`, `pipe_barrier`, or collective synchronization has been issued.
3. The host runtime has confirmed completion (via event or runtime call).

The exact moment of GM visibility across blocks is **hardware-specific**: the ISA guarantees that the ordering contract is satisfied, but the exact timing of cross-block visibility depends on the target profile:
- **A2/A3 and A5**: Visibility is guaranteed after the block-level synchronization (e.g., `wait_flag`/`pipe_barrier` or collective op) completes; the hardware enforces visibility within a bounded time window after the sync operation.
- **CPU simulator**: Visibility is immediate within the same process address space, since there is no physical DMA or device memory hierarchy.

## UB Visibility

UB is **core-local**: only the AI Core that owns the UB can read or write it. UB data is NOT visible to other cores.

UB visibility within a core follows program order plus event order. The following are guaranteed by the ISA:

- UB reads within a core see all prior UB writes within that core (program order).
- UB reads see data from `copy_gm_to_ubuf` only after the corresponding `wait_flag` returns.

## Undefined, Unspecified, and Implementation-Defined

PTO uses these terms precisely:

| Term | Meaning | Example |
|------|---------|---------|
| **Undefined** | Behavior is intentionally unspecified; any outcome is permitted | Reading a tile's out-of-valid-region element |
| **Unspecified** | The ISA does not define the behavior; implementations may choose | Exact cycle count of an operation |
| **Implementation-Defined** | The behavior is defined by the implementation but documented | FTZ behavior for denormals on A5 |

## Target Refinement

CPU, A2/A3, and A5 may differ in implementation detail and support subsets, but the baseline manual must still say clearly which ordering facts are:

| Category | Portability |
|----------|------------|
| Program order within a tile buffer | Portable across all profiles |
| Event order via `set_flag`/`wait_flag` | Portable across all profiles (same semantics, different pipe spaces) |
| `RecordEvent` chaining | Portable across all profiles |
| Barrier order via collective ops | Not available on CPU; available on A2/A3 and A5 |
| UB→GM implicit visibility | Not portable; requires explicit `TSTORE`/`copy_ubuf_to_gm` |
| A5-specific stronger ordering | A5-specific, not portable to CPU/A2/A3 |

## Cases That Are Not Allowed

- Documenting implementation detail as though it were the portable memory model.
- Hiding visibility requirements inside vague words like "usually ordered."
- Mixing memory-model guarantees with scheduling heuristics.
- Claiming that data is visible across blocks without an explicit synchronization operation.
- Assuming that "the hardware does it automatically" without specifying which operation provides the guarantee.

## See Also

- [Producer-Consumer Ordering](./producer-consumer-ordering.md)
- [Ordering And Synchronization](../machine-model/ordering-and-synchronization.md)
- [Portability And Target Profiles](../reference/portability-and-target-profiles.md)
