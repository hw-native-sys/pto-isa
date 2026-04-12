<!-- Generated from `docs/isa/scalar/ops/pipeline-sync/set-intra-block.md` -->

# pto.set_intra_block

Standalone reference page for `pto.set_intra_block`. This page belongs to the [Pipeline Sync](../../pipeline-sync.md) family in the PTO ISA manual.

## Summary

Signal an event within a cluster (A5). Uses **1:1 per-subblock semantics**: each call targets exactly one subblock. No broadcast; separate calls are required for each subblock.

## Mechanism

`pto.set_intra_block` signals an event within a Core Cluster on the A5 platform.

**1:1 semantics (A5 cluster, 1 Cube : 2 Vector subblocks):**

- Each `set_intra_block` call targets **exactly one** subblock (determined by the semaphore ID).
- IDs 0–15 target AIV0; IDs 16–31 (base + 15) target AIV1.
- There is **no broadcast**: to signal both subblocks, two separate `set_intra_block` calls are required.

This contrasts with A2A3's `set_cross_core` which broadcasts to both subblocks with one call.

## Syntax

### PTO Assembly Form

```text
set_intra_block %pipe, %sem_id : i64, i64
```

### AS Level 1 (SSA)

```mlir
pto.set_intra_block %pipe, %sem_id : i64, i64
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void SET_INTRA_BLOCK(pipe_t trigger_pipe, int64_t sem_id);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%pipe` | `pipe_t` | The triggering pipeline on the calling core |
| `%sem_id` | `i64` | Semaphore ID: 0–15 for AIV0, base+15 for AIV1 |

## Expected Outputs

None. This form is defined by its side effect on intra-block synchronization state.

## Side Effects

- Signals the named semaphore to the target subblock.
- The target subblock's `wait_intra_core` unblocks when the count reaches zero.

## Constraints

- **A5 only**: `set_intra_block` is only available on the A5 profile.
- **Semaphore ID mapping**: IDs 0–15 target AIV0; IDs 16–31 target AIV1. Programs MUST use the correct ID for the target subblock.
- **No broadcast**: Unlike A2A3's `set_cross_core`, one `set_intra_block` does NOT reach both subblocks. Separate calls are required for each subblock.
- **Semaphore pool**: 16 physical IDs with a 32-ID address space. IDs outside 0–31 are **illegal**.

## Exceptions

- Illegal on non-A5 profiles.
- Illegal if `%sem_id` is outside the range 0–31.
- Illegal if the target subblock is not reachable (invalid core ID encoding in sem_id).

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| `set_intra_block` | Not available | Use `set_cross_core` | Supported |
| Broadcast semantics | Not applicable | One set → both subblocks | One set → one subblock |
| Per-subblock control | Not applicable | Not available | Supported |
| Semaphore pool | Not applicable | 16 IDs, 4-bit counter | 16 IDs, 32-ID address space |

## Examples

### A5: C→V — separate signals per subblock

```c
#include <pto/pto-inst.hpp>
using namespace pto;

// AIC signals AIV0 on semaphore 0
SET_INTRA_BLOCK(PIPE_MTE2, /* sem_id */ 0);

// AIC signals AIV1 on semaphore 16 (0 + 15 offset)
SET_INTRA_BLOCK(PIPE_MTE2, /* sem_id */ 16);
```

### SSA form — C→V with 1:1 semantics

```mlir
// AIC: signal AIV0 that data is ready
pto.set_intra_block "PIPE_MTE2", %c0_i64 : i64, i64

// AIC: signal AIV1 that data is ready
pto.set_intra_block "PIPE_MTE2", %c16_i64 : i64, i64
```

### SSA form — V→C with 1:1 semantics

```mlir
// AIV0: signal AIC that segment 0 is done
pto.set_intra_block "PIPE_V", %c0_i64 : i64, i64

// AIV1: signal AIC that segment 1 is done
pto.set_intra_block "PIPE_V", %c0_i64 : i64, i64

// AIC: wait for AIV0 on sem 0
pto.wait_intra_core "PIPE_MTE2", %c0_i64 : i64, i64

// AIC: wait for AIV1 on sem 16
pto.wait_intra_core "PIPE_MTE2", %c16_i64 : i64, i64
```

## Related Ops / Family Links

- Family overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in family: [pto.wait_flag_dev](./wait-flag-dev.md)
- Next op in family: [pto.wait_intra_core](./wait-intra-core.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
