<!-- Generated from `docs/isa/tile/ops/sync-and-config/tsync.md` -->

# pto.tsync

Standalone reference page for `pto.tsync`. This page belongs to the [Sync And Config](../../sync-and-config.md) family in the PTO ISA manual.

## Summary

Synchronize PTO execution (wait on events or insert a per-op pipeline barrier).

## Mechanism

`TSYNC(events...)` waits on a set of explicit event tokens. `TSYNC<Op>()` inserts a pipeline barrier for a single operation class.

Many intrinsics in `include/pto/common/pto_instr.hpp` call `TSYNC(events...)` internally before issuing the instruction. It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Event operand form:

```text
tsync %e0, %e1 : !pto.event<...>, !pto.event<...>
```

Single-op barrier form:

```text
tsync.op #pto.op<TADD>
```

### AS Level 1 (SSA)

The SSA form for `TSYNC` does not use explicit event SSA values. The ISA-level primitive is the C++ `RecordEvent` chaining via `WaitEvents...` on tile intrinsics. SSA-level representations of event ordering use `record_event` / `wait_event` (see below) which are PTO-DSL internal IR nodes, not the portable ISA surface.

### AS Level 2 (DPS)

The AS Level 2 form exposes explicit event ordering primitives:

```text
pto.record_event[src_op, dst_op, eventID]
// Supported ops: TLOAD, TSTORE_ACC, TSTORE_VEC, TMOV_M2L, TMOV_M2S,
//                 TMOV_M2B, TMOV_M2V, TMOV_V2M, TMATMUL, TVEC
pto.wait_event[src_op, dst_op, eventID]
// Supported ops: same as record_event
pto.barrier(op)
// Supported ops: TVEC, TMATMUL
```

In the current PTO-DSL front-end flow, `record_event` and `wait_event` should be treated as low-level TSYNC forms. Front-end kernels SHOULD normally stay free of explicit event wiring and rely on `ptoas --enable-insert-sync` instead.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <Op OpCode>
PTO_INST void TSYNC();

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &... events);
```

## Inputs

`TSYNC(events...)` takes one or more `RecordEvent` values as operands. Each `RecordEvent` is produced by a prior tile operation (`TLOAD`, `TADD`, `TMATMUL`, etc.). The call waits for all supplied events before proceeding.

`TSYNC<Op>()` takes a compile-time operation tag (`Op::TLOAD`, `Op::TADD`, `Op::TMATMUL`, etc.) and inserts a pipeline barrier for all operations of that class.

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile beyond any explicit state object named by the syntax.

## Side Effects

This operation may establish a synchronization edge, bind or configure architectural tile state, or update implementation-defined configuration that later tile instructions consume.

## Constraints

- **`TSYNC(events...)` semantics**:
    - `TSYNC(events...)` calls `WaitAllEvents(events...)`, which invokes `events.Wait()` on each event token. In auto mode, this is no-op.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (`TSYNC<Op>()`)**:
    - `TSYNC_IMPL<Op>()` only supports vector-pipeline ops (`static_assert(pipe == PIPE_V)` in `include/pto/common/event.hpp`).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto(__gm__ float* in) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin(in);
  TileT t;
  RecordEvent e = TLOAD(t, gin);  // TLOAD returns RecordEvent
  TSYNC(e);                       // wait for load to complete
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  RecordEvent e = TADD(c, a, b);  // TADD returns RecordEvent
  TSYNC<Op::TADD>();              // pipeline barrier for TADD
  TSYNC(e);                       // explicit wait
}
```

### PTO Assembly Form

Event-wait form (bare assembly):

```text
tsync %e0, %e1 : !pto.event<...>, !pto.event<...>
```

Barrier form:

```text
tsync.op #pto.op<TADD>
```

## Related Ops / Family Links

- Family overview: [Sync And Config](../../sync-and-config.md)
- Next op in family: [pto.tassign](./tassign.md)
