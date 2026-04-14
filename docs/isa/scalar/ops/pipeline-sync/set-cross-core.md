# pto.set_cross_core

`pto.set_cross_core` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Signal an event to another core in a cluster (A2A3). Uses **mode2 broadcast semantics**: one signal reaches both vector subblocks simultaneously; the Cube blocks until both subblocks have signaled back.

## Mechanism

`pto.set_cross_core` signals an event between execution units in a Core Cluster on the A2A3 platform.

**Mode2 semantics (A2A3 cluster, 1 Cube : 2 Vector subblocks):**

- **C→V broadcast**: One `set_cross_core` from the Cube (AIC) atomically increments the semaphore for **both** AIV0 and AIV1 subblocks simultaneously.
- **V→C reduce**: When the Cube calls `wait_flag_dev`, it blocks until **both** AIV0 and AIV1 have called `set_cross_core` on the same semaphore. Only then does the Cube unblock.

This is a hardware reduce operation: the Cube need only issue one `wait_flag_dev` to synchronize with both subblocks, rather than one per subblock.

The semaphore is a counter: incremented by `set_cross_core`, decremented by `wait_flag_dev`. A `wait_flag_dev` unblocks when the counter reaches zero.

## Syntax

### PTO Assembly Form

```mlir
pto.set_cross_core %core_id, %event_id : i64, i64
```

### AS Level 1 (SSA)

```mlir
pto.set_cross_core %core_id, %event_id : i64, i64
```

## C++ Intrinsic

The installed 3510 public CCE headers do not expose a same-name `set_cross_core(...)` intrinsic. The shipped sync implementation uses the internal cross-core helper shown below.

```cpp
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%core_id` | `i64` | Target core identifier (subblock selector: 0 = AIV0, 1 = AIV1) |
| `%event_id` | `i64` | Semaphore/event identifier |

## Expected Outputs

None. This form is defined by its side effect on inter-core synchronization state.

## Side Effects

- Atomically signals the named event to the target core.
- On mode2: increments semaphore for both subblocks simultaneously.

## Constraints

- **A2A3 only**: `set_cross_core` is only available on the A2A3 profile. Programs that use this operation MUST provide a fallback path for other profiles.
- **Semaphore pool**: The pool has 16 physical semaphore IDs per cluster. The hardware implements a 4-bit counter (0–15). `set_cross_core` increments the counter; `wait_flag_dev` decrements it. If the counter would overflow past 15, the behavior is **implementation-defined**.
- **Broadcast vs. per-subblock**: The broadcast behavior is specific to mode2. Other modes (if supported) may have different semantics.
- **core_id meaning**: `core_id = 0` targets AIV0 subblock; `core_id = 1` targets AIV1 subblock. Other values are **illegal**.

## Exceptions

- Illegal on non-A2A3 profiles.
- Illegal if `%event_id` is outside the valid range (0–15).
- Illegal if the hardware counter would overflow.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| `set_cross_core` | Not available | Supported (mode2) | Use `set_intra_block` |
| Mode2 broadcast semantics | Not applicable | Supported | Not applicable |
| Semaphore pool size | Not applicable | 16 IDs | Not applicable |
| Per-subblock signaling | Not applicable | 1 set reaches both | 1 set per subblock |

CPU simulator does not implement `set_cross_core`. Portable programs MUST guard this operation with profile checks or provide CPU-sim fallback.

## Examples

### A2A3: Cube broadcasts to both vector subblocks

```c
#include <pto/pto-inst.hpp>
using namespace pto;

// Cube signals both AIV subblocks simultaneously
SET_CROSS_CORE(/* core_id */ 0, /* event_id */ 0);  // broadcast to both AIV0 and AIV1
```

### SSA form — Cube→Vector broadcast

```mlir
// Cube: broadcast completion signal to both AIV0 and AIV1
pto.set_cross_core %c0_i64, %c0_i64 : i64, i64

// Both AIV subblocks receive the signal (atomic broadcast)

// AIV0: signals back when its work is done
pto.set_cross_core %c0_i64, %c1_i64 : i64, i64  // signals event 1

// AIV1: signals back when its work is done
pto.set_cross_core %c1_i64, %c1_i64 : i64, i64  // signals event 1

// Cube: waits for BOTH AIV0 and AIV1 (reduce)
pto.wait_flag_dev %c0_i64, %c1_i64 : i64, i64
// Unblocks only when both subblocks have signaled
```

### SSA form — Vector→Cube reduce

```mlir
// AIV0: signal Cube that vector work segment is done
pto.set_cross_core %c0_i64, %c2_i64 : i64, i64

// AIV1: signal Cube that vector work segment is done
pto.set_cross_core %c1_i64, %c2_i64 : i64, i64

// Cube: waits for both subblocks on one semaphore (reduce)
pto.wait_flag_dev %c2_i64 : i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in instruction set: [pto.mem_bar](./mem-bar.md)
- Next op in instruction set: [pto.wait_flag_dev](./wait-flag-dev.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
