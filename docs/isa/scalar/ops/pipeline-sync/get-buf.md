# pto.get_buf

`pto.get_buf` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Acquire a buffer slot in a double-buffering protocol. Implicitly signals readiness to the consuming pipeline via the buffer-token event system.

## Mechanism

`pto.get_buf` acquires a named buffer slot for the calling (consumer) pipeline. It is the acquiring half of the `get_buf`/`rls_buf` double-buffering protocol.

The operation:

1. **Checks availability**: If the named buffer ID is held by another pipeline, the calling pipeline **blocks** until the holder releases it.
2. **Acquires the slot**: Marks the buffer as held by the calling pipeline.
3. **Implicitly signals**: Issues `set_flag` from the consumer pipeline to the producer pipeline on the buffer's associated event ID, allowing the producer to proceed.

The consumer pipeline holds the buffer until a matching `rls_buf` is issued. Buffer IDs use program order and the double-buffering protocol to implicitly resolve RAW and WAR dependencies — no explicit event IDs are required.

## Syntax

### PTO Assembly Form

```mlir
pto.get_buf %buf_id, "PIPE_*", %mode : i64, i64
```

### AS Level 1 (SSA)

```mlir
pto.get_buf %buf_id, "PIPE_*", %mode : i64, i64
```

## C++ Intrinsic

```cpp
pipe_t pipe = PIPE_S;
uint64_t bufId = 0;
bool mode = true;
get_buf(pipe, bufId, mode);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%buf_id` | `i64` | Buffer slot identifier (0–N, where N is profile-defined) |
| `"PIPE_*"` | pipe identifier | The consumer pipeline that is acquiring this slot |
| `%mode` | `i64` | Protocol mode; controls how the release signals the next stage |

## Expected Outputs

None. This form is defined by its side effect on buffer state and synchronization.

## Side Effects

- Marks the buffer slot as held by the calling pipeline.
- Implicitly issues `set_flag` from the consumer pipeline to the producer pipeline on the buffer's associated event ID.
- May block if the buffer is not currently available.

## Constraints

- **Buffer ID uniqueness per pipeline**: Each pipeline may hold at most one slot per buffer ID at a time. Acquiring the same buffer ID twice on the same pipeline without an intervening `rls_buf` is **illegal**.
- **Producer must release**: The producer pipeline must have issued `rls_buf` on the same buffer ID before this acquire can succeed. Acquiring a buffer that was never released (first iteration) succeeds immediately since all slots start free.
- **No explicit event IDs**: Unlike `set_flag`/`wait_flag`, buffer ID management requires no explicit event naming. The hardware maps buffer IDs to internal event IDs.
- **Buffer ID range**: Buffer IDs MUST be in the range `[0, B)` where `B` is the profile-defined maximum. Out-of-range IDs are **illegal**.

## Exceptions

- Illegal if `%buf_id` is not in the valid range for the target profile.
- Illegal if the same pipeline acquires the same buffer ID twice without an intervening `rls_buf`.
- Illegal if the buffer ID was never released and the producer has not yet issued `rls_buf` (the acquire will block indefinitely, which is treated as a protocol error).
- Illegal on CPU simulator if buffer state is inconsistent.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Buffer acquire | Simulated | Supported | Supported |
| Implicit set_flag | Simulated | Supported | Supported |
| Blocking on unavailable slot | Simulated | Supported | Supported |
| Maximum buffer IDs | Implementation-defined | 32 (global pool) | 32 (global pool) |

## Examples

### Acquire buffer for computation

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void compute_loop(int64_t buf_id,
                 Ptr<ub_space_t, ub_t> ub_in,
                 Ptr<ub_space_t, ub_t> ub_out) {
    // Consumer (Vector pipe) acquires input buffer
    GET_BUF(buf_id, PIPE_V, 0);

    // Vector load, compute, store ...
    RegBuf<predicate_t> mask;
    PSET_B32(mask, "PAT_ALL");

    // Release input buffer so MTE2 can reuse it
    RLS_BUF(buf_id, PIPE_V, 0);
}
```

### SSA form — acquire in loop

```mlir
scf.for %i = %c0 to %N step %c1 {
    // Acquire input buffer slot i%2
    pto.get_buf %bufid_in[%pp], "PIPE_V", %c0 : i64, i64

    // Acquire output buffer slot i%2
    pto.get_buf %bufid_out[%pp], "PIPE_V", %c0 : i64, i64

    // Compute (loads from ub_in[%pp], stores to ub_out[%pp])
    // ...

    // Release both slots
    pto.rls_buf %bufid_in[%pp], "PIPE_V", %c0 : i64, i64
    pto.rls_buf %bufid_out[%pp], "PIPE_V", %c0 : i64, i64
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in instruction set: [pto.pipe_barrier](./pipe-barrier.md)
- Next op in instruction set: [pto.rls_buf](./rls-buf.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
