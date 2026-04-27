# pto.tsync

`pto.tsync` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Synchronize PTO execution. Two forms exist: the event-wait form blocks until specified event tokens are ready, and the barrier form inserts a pipeline drain for all operations of a given class.

## Mechanism

### Event-Wait Form: `TSYNC(events...)`

Waits on one or more `RecordEvent` tokens. Each token is produced by a prior tile operation (`TLOAD`, `TSTORE`, `TADD`, `TMATMUL`, etc.). The call does not return until all supplied events have been recorded. After this point, the tile data produced by the operations that created these events is guaranteed to be visible to subsequent operations.

The underlying mechanism calls `events.Wait()` on each event token. In Auto mode, this may be a no-op when the compiler/runtime proves ordering by construction.

### Barrier Form: `TSYNC<Op>()`

Inserts a pipeline barrier for all operations of the specified operation class. All operations of class `Op` that appear before the barrier complete before any operation of class `Op` that appears after the barrier begins. The barrier does not affect other operation classes.

`TSYNC<Op>()` only supports vector-pipeline operations. Attempting to use it for non-vector operations produces undefined results.

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

### AS Level 2 (DPS)

The AS Level 2 form exposes explicit event ordering primitives:

```text
pto.record_event[src_op, dst_op, eventID]
pto.wait_event[src_op, dst_op, eventID]
pto.barrier(op)
```

`record_event` / `wait_event` are low-level TSYNC forms. Front-end kernels should normally stay free of explicit event wiring and rely on `ptoas --enable-insert-sync` instead.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
// Barrier for a single operation class
template <Op OpCode>
PTO_INST void TSYNC();

// Wait on one or more RecordEvent tokens
template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &... events);
```

## Inputs

| Form | Operands | Description |
|------|----------|-------------|
| Event-wait | `events...` | One or more `RecordEvent` values produced by prior tile operations |
| Barrier | `Op` template parameter | Operation class tag (`Op::TLOAD`, `Op::TADD`, `Op::TMATMUL`, etc.) |

## Expected Outputs

The event-wait form produces no value; it blocks until events are ready.
The barrier form produces no value; it blocks until the barrier completes.

## Side Effects

- Event-wait: may block the scalar unit until all specified events are recorded.
- Barrier: may block the specified pipeline until all prior operations of that class complete.

Does not affect unrelated tile traffic or other operation classes (barrier form only).

## Constraints

- `TSYNC(events...)`: If no events are supplied, the call is a no-op.
- `TSYNC<Op>()`: Only valid for vector-pipeline operations. Attempting to use it for tile, matrix, or scalar operations is undefined.
- All events must be produced by operations that precede the `TSYNC` in program order.

## Exceptions

- Supplying events that are not produced by any preceding operation causes undefined behavior.
- Using `TSYNC<Op>()` with an unsupported operation class is rejected by the verifier.

## Target-Profile Restrictions

- CPU simulation: `TSYNC` emulates the ordering semantics but may not reflect hardware pipeline latency.
- A2/A3: `TSYNC<Op>()` only supports vector-pipeline ops.
- A5: same as A2/A3.

## Examples

### Event-wait

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example(__gm__ float* in) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin(in);
  TileT t;
  RecordEvent e = TLOAD(t, gin);  // TLOAD returns RecordEvent
  TSYNC(e);                        // wait for load to complete
}
```

### Pipeline barrier

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  RecordEvent e = TADD(c, a, b);     // returns RecordEvent
  TSYNC<Op::TADD>();                 // drain all TADD operations
  TSYNC(e);                           // wait for this specific TADD
}
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Next op in instruction set: [pto.settf32mode](./settf32mode.md)
- [Ordering and Synchronization](../../../machine-model/ordering-and-synchronization.md) for the full event model
