<!-- Generated from `docs/isa/scalar/ops/pipeline-sync/wait-intra-core.md` -->

# pto.wait_intra_core

Standalone reference page for `pto.wait_intra_core`. This page belongs to the [Pipeline Sync](../../pipeline-sync.md) family in the PTO ISA manual.

## Summary

Block a specific pipeline within a cluster (A5) until a subblock signals an event. Only the named pipeline stalls; other pipelines on the same core continue executing.

## Mechanism

`pto.wait_intra_core` blocks a specific pipeline on the calling core until the named event is signaled by a remote subblock.

**Per-pipeline blocking (A5):**

- Unlike A2A3's `wait_flag_dev` which stalls the **entire SU** (all pipelines), `wait_intra_core` only stalls the **named pipeline**.
- Other pipelines on the same core continue executing while one pipeline is blocked.
- The semaphore pool uses 16 physical IDs with a 32-ID address space: IDs 0–15 target AIV0; IDs 16–31 target AIV1.

**Key advantage over A2A3**: A5's `wait_intra_core` enables finer-grained parallelism where multiple pipelines can be in different synchronization states simultaneously.

## Syntax

### PTO Assembly Form

```text
wait_intra_core %pipe, %sem_id : i64, i64
```

### AS Level 1 (SSA)

```mlir
pto.wait_intra_core %pipe, %sem_id : i64, i64
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void WAIT_INTRA_CORE(pipe_t wait_pipe, int64_t sem_id);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%pipe` | `pipe_t` | The pipeline that should wait (only this pipeline stalls) |
| `%sem_id` | `i64` | Semaphore ID: 0–15 for AIV0, base+15 for AIV1 |

## Expected Outputs

None. This form is defined by its side effect (blocking) on the named pipeline.

## Side Effects

- Blocks only the named pipeline. Other pipelines on the same core continue.
- Decrements the semaphore counter when the event is signaled.

## Constraints

- **A5 only**: `wait_intra_core` is only available on the A5 profile.
- **Per-pipeline blocking**: Only the named pipeline is blocked. All other pipelines continue. This differs fundamentally from A2A3's SU-level blocking.
- **Semaphore ID mapping**: IDs 0–15 target AIV0; IDs 16–31 target AIV1.
- **Event must be set**: Waiting on an event that was never set is **illegal**.
- **Semaphore pool**: 16 physical IDs, 32-ID address space. IDs outside 0–31 are **illegal**.

## Exceptions

- Illegal on non-A5 profiles.
- Illegal if `%sem_id` is outside the range 0–31.
- Illegal if the event was never set by a remote subblock.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| `wait_intra_core` | Not available | Use `wait_flag_dev` | Supported |
| Blocking scope | Not applicable | Entire SU blocked | Only named pipe blocked |
| Other pipes during wait | Not applicable | All stalled | Continue executing |
| Semaphore pool | Not applicable | 16 IDs | 16 IDs, 32-ID address space |

## Examples

### A5: per-pipeline blocking vs. A2A3 SU-level blocking

```c
#include <pto/pto-inst.hpp>
using namespace pto;

// A2A3 (wait_flag_dev): entire core stalls
// WAIT_FLAG_DEV(/* event_id */ 0);  // ALL pipelines blocked

// A5 (wait_intra_core): only PIPE_V stalls
WAIT_INTRA_CORE(PIPE_V, /* sem_id */ 0);  // Only Vector pipe stalls
// PIPE_MTE2 and PIPE_MTE3 continue executing
```

### SSA form — A5 C→V signaling with per-pipeline waits

```mlir
// === AIC signals both AIV subblocks (no broadcast — separate calls) ===
pto.set_intra_block "PIPE_MTE2", %c0_i64 : i64, i64   // → AIV0
pto.set_intra_block "PIPE_MTE2", %c16_i64 : i64, i64  // → AIV1

// === AIV0: Vector pipe waits (only Vector stalls; MTE2/MTE3 continue) ===
pto.wait_intra_core "PIPE_V", %c0_i64 : i64, i64
// [AIV0 Vector processes data while AIV0 MTE2/MTE3 continue]

// === AIV1: Vector pipe waits ===
pto.wait_intra_core "PIPE_V", %c16_i64 : i64, i64

// === AIV0 signals back to AIC ===
pto.set_intra_block "PIPE_V", %c1_i64 : i64, i64

// === AIV1 signals back to AIC ===
pto.set_intra_block "PIPE_V", %c1_i64 : i64, i64

// === AIC: waits for both AIV subblocks (separate waits) ===
pto.wait_intra_core "PIPE_MTE2", %c1_i64 : i64, i64    // wait for AIV0
pto.wait_intra_core "PIPE_MTE2", %c17_i64 : i64, i64   // wait for AIV1
```

## Related Ops / Family Links

- Family overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in family: [pto.set_intra_block](./set-intra-block.md)
- Next op in family: (none — last in family)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
