<!-- Generated from `docs/isa/scalar/ops/pipeline-sync/rls-buf.md` -->

# pto.rls_buf

Standalone reference page for `pto.rls_buf`. This page belongs to the [Pipeline Sync](../../pipeline-sync.md) family in the PTO ISA manual.

## Summary

Release a buffer slot, implicitly signaling the consuming pipeline to proceed.

## Mechanism

`pto.rls_buf` releases a previously acquired buffer slot for the calling (producer) pipeline. It is the releasing half of the `get_buf`/`rls_buf` double-buffering protocol.

The operation:

1. **Releases the slot**: Marks the buffer as free for the calling (producer) pipeline.
2. **Implicitly signals**: Issues `set_flag` from the producer pipeline to the consumer pipeline on the buffer's associated event ID, unblocking the consumer.

After `rls_buf`, the producer pipeline no longer holds the buffer and MUST NOT access it until it re-acquires it in a future iteration. The consumer pipeline is unblocked by the implicit `set_flag` and may proceed.

## Syntax

### PTO Assembly Form

```text
rls_buf %buf_id, "PIPE_*", %mode : i64, i64
```

### AS Level 1 (SSA)

```mlir
pto.rls_buf %buf_id, "PIPE_*", %mode : i64, i64
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void RLS_BUF(int64_t buf_id,
                      pipe_t producer_pipe,
                      int64_t mode);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%buf_id` | `i64` | Buffer slot identifier (must match a prior `get_buf`) |
| `"PIPE_*"` | pipe identifier | The producer pipeline that is releasing this slot |
| `%mode` | `i64` | Protocol mode; controls how the next stage is signaled |

## Expected Outputs

None. This form is defined by its side effect on buffer state and synchronization.

## Side Effects

- Marks the buffer slot as free for the calling pipeline.
- Implicitly issues `set_flag` from the producer pipeline to the consumer pipeline on the buffer's associated event ID.
- Does **not** block.

## Constraints

- **Must match prior acquire**: The calling pipeline MUST have previously acquired the named buffer ID via `get_buf`. Releasing a buffer that was never acquired is **illegal**.
- **Release-after-produce order**: `rls_buf` MUST be issued only after the producer has completed all work on the buffer. Releasing before the data is ready produces **implementation-defined** results.
- **One release per acquire**: Each `get_buf` MUST be matched by exactly one `rls_buf` before the next `get_buf` on the same pipeline and buffer ID. Extra releases or missing releases are **illegal**.
- **Producer-consumer pairing**: The pipeline named in `rls_buf` is the producer pipeline (the one that wrote to the buffer). The matching `get_buf` names the consumer pipeline.

## Exceptions

- Illegal if `%buf_id` was not previously acquired by the calling pipeline.
- Illegal if an extra `rls_buf` is issued without a matching prior `get_buf`.
- Illegal if `rls_buf` is issued before the producer has finished writing to the buffer (data hazard).

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Buffer release | Simulated | Supported | Supported |
| Implicit set_flag | Simulated | Supported | Supported |
| Maximum buffer IDs | Implementation-defined | 32 (global pool) | 32 (global pool) |

## Examples

### Release after DMA load

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void load_and_release(int64_t buf_id,
                      Ptr<ub_space_t, ub_t> gm_src,
                      Ptr<ub_space_t, ub_t> ub_dst) {
    // Acquire buffer slot (MTE2 acquires to write)
    GET_BUF(buf_id, PIPE_MTE2, 0);

    // MTE2 DMA load: GM → UB
    COPY_GM_TO_UBUF(gm_src, ub_dst, /* ... */);

    // Release: MTE2 signals Vector that data is ready
    RLS_BUF(buf_id, PIPE_MTE2, 0);
}
```

### SSA form — matching acquire/release

```mlir
// Producer (MTE2) acquires, loads, releases
pto.get_buf %bufid_in[%pp], "PIPE_MTE2", %c0 : i64, i64
pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_in[%pp], ...
pto.rls_buf %bufid_in[%pp], "PIPE_MTE2", %c0 : i64, i64

// Consumer (Vector) acquires, computes, releases
pto.get_buf %bufid_in[%pp], "PIPE_V", %c0 : i64, i64
// ... vector compute on ub_in[%pp] ...
pto.rls_buf %bufid_in[%pp], "PIPE_V", %c0 : i64, i64
```

## Related Ops / Family Links

- Family overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in family: [pto.get_buf](./get-buf.md)
- Next op in family: [pto.mem_bar](./mem-bar.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
