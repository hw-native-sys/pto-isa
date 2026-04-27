# pto.wait_flag_dev

`pto.wait_flag_dev` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Block the entire SU (all pipelines) until a remote core signals an event (A2A3). Uses **mode2 reduce semantics**: one wait unblocks only when **all** subblocks in the cluster have signaled.

## Mechanism

`pto.wait_flag_dev` blocks the entire SU of the calling core until the named event is signaled by the remote core.

**Mode2 reduce semantics (A2A3):**

- The calling core (typically the Cube) waits on a single semaphore. The semaphore counter is decremented by each `set_cross_core` from remote subblocks.
- The SU is **fully blocked**: all pipelines (PIPE_MTE2, PIPE_V, PIPE_MTE3) are stalled.
- The wait unblocks when the semaphore counter reaches zero. In mode2 with 1:2 topology, this means both AIV0 and AIV1 must have called `set_cross_core` before the Cube unblocks.

This is the Cube's counterpart to `set_cross_core`. The pattern is:

```
Cube:         set_cross_core → (broadcast to AIV0+AIV1)
AIV0/AIV1:   [do work] → set_cross_core
Cube:         wait_flag_dev → unblocks when BOTH subblocks signaled
```

## Syntax

### PTO Assembly Form

```mlir
pto.wait_flag_dev %event_id : i64
```

### AS Level 1 (SSA)

```mlir
pto.wait_flag_dev %event_id : i64
```

## C++ Intrinsic

```cpp
pipe_t pipe = PIPE_MTE2;
int64_t flagId = 0;
wait_flag_dev(pipe, flagId);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%event_id` | `i64` | Semaphore/event identifier to wait on |

## Expected Outputs

None. This form is defined by its side effect (blocking) on the calling core.

## Side Effects

- **Blocks the entire SU**: All pipelines (MTE2, V, MTE3) on the calling core are stalled until the event fires.
- Decrements the semaphore counter when the event is signaled.

## Constraints

- **A2A3 only**: `wait_flag_dev` is only available on the A2A3 profile.
- **SU-level blocking**: Unlike `wait_flag` (intra-core) which only stalls the named destination pipeline, `wait_flag_dev` stalls **all** pipelines on the core. This is more restrictive than A5's `wait_intra_core`.
- **Semaphore pool**: The pool has 16 physical semaphore IDs per cluster with a 4-bit counter (0–15). The wait unblocks when the counter reaches zero. On A2/A3: if `wait_flag_dev` is called when the counter is already zero, the call returns immediately without error. On A5: this operation is not available; use `wait_intra_core` instead.
- **Event must be set**: Waiting on an event that was never set by a matching remote `set_cross_core` is **illegal**.

## Exceptions

- Illegal on non-A2A3 profiles.
- Illegal if `%event_id` is outside the valid range (0–15).
- Illegal if the event was never set by a remote core.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| `wait_flag_dev` | Not available | Supported | Use `wait_intra_core` |
| SU-level blocking | Not applicable | All pipelines blocked | Only named pipe blocked |
| Semaphore pool size | Not applicable | 16 IDs, 4-bit counter | 16 IDs, 32-ID address space |
| Reduce semantics | Not applicable | One wait unblocks on N signals | One wait per signal |

## Examples

### A2A3: Cube waits for both vector subblocks

```c
#include <pto/pto-inst.hpp>
using namespace pto;

// Cube: broadcast to both AIV subblocks
SET_CROSS_CORE(/* core_id */ 0, /* event_id */ 0);

// AIV0: do work, then signal
// AIV1: do work, then signal

// Cube: block until BOTH AIV0 and AIV1 have signaled (reduce)
WAIT_FLAG_DEV(/* event_id */ 0);
```

### SSA form — complete C↔V handshake

```mlir
// === Cube (Producer) ===
// Signal both AIV subblocks: data is ready
pto.set_cross_core %c0_i64, %c0_i64 : i64, i64

// === AIV0 (Consumer) ===
// Wait for Cube's signal
pto.wait_flag_dev %c0_i64 : i64
// [process data]
// Signal back to Cube: work on AIV0 done
pto.set_cross_core %c0_i64, %c1_i64 : i64, i64

// === AIV1 (Consumer) ===
// Wait for Cube's signal
pto.wait_flag_dev %c0_i64 : i64
// [process data]
// Signal back to Cube: work on AIV1 done
pto.set_cross_core %c1_i64, %c1_i64 : i64, i64

// === Cube (Producer) ===
// Block until BOTH AIV0 and AIV1 have signaled (reduce)
pto.wait_flag_dev %c1_i64 : i64
// Both signaled — Cube can proceed
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in instruction set: [pto.set_cross_core](./set-cross-core.md)
- Next op in instruction set: [pto.set_intra_block](./set-intra-block.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
