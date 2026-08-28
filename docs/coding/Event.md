# Events and Synchronization

PTO Tile Lib supports an explicit event model for expressing dependencies between operations without introducing a global barrier for every instruction.

This document describes the C++ event types used by `include/pto/common/pto_instr.hpp` and `include/pto/common/event.hpp`.

> Note: the concrete `pto::Event<SrcOp, DstOp>` type is defined only for device builds (`__CCE_AICORE__`). The CPU simulator backend treats event synchronization as a no-op and relies on ordinary program order within each CPU worker. Cross-worker ordering must use a synchronization or communication operation supported by CPU_SIM.

## Key types

### `pto::Op`

`pto::Op` is an opcode-like enumeration used to classify operations. Each `Op` maps to a hardware pipeline (`PIPE_V`, `PIPE_MTE2`, ...).

### `pto::RecordEvent`

Many intrinsics (e.g., `TADD`, `TLOAD`, `TSTORE`) return `pto::RecordEvent`. This is a marker value that can be assigned into an `Event<SrcOp, DstOp>` to record a token after the op finishes.

### `pto::Event<SrcOp, DstOp>` (device-only)

On device builds (`__CCE_AICORE__`), `include/pto/common/event.hpp` defines:

```cpp
template <Op SrcOp, Op DstOp>
struct Event {
  void Wait();
  void Record();
  Event& operator=(RecordEvent);
};
```

- `Wait()` blocks until the producer-side token is satisfied.
- `Record()` sets a token on the producer pipeline.
- `evt = OP(...)` (assignment from `RecordEvent`) records automatically.

The template parameters encode the producer/consumer opcodes and are used to select the correct pipeline pair.

### single-op synchronization (single-pipeline barrier)

Single-op synchronization is a single-op barrier implemented internally by the PTO runtime.

- On device, the current implementation restricts the single-op form to vector pipeline ops (`PIPE_V`).
- On the CPU simulator backend (`__CPU_SIM`), this internal barrier is a no-op.

## How `WaitEvents&...` works in intrinsics

Most intrinsics in `include/pto/common/pto_instr.hpp` have a trailing `WaitEvents&... events` pack.

Pattern:

- The intrinsic calls event waits.
- event waits calls `WaitAllEvents(events...)`, which invokes `events.Wait()` on each event.
- The instruction then executes, and the intrinsic returns a `RecordEvent`.

This enables a programming style where you:

1. Keep event tokens as SSA-like C++ variables.
2. Pass them into the next op to enforce ordering.
3. Record a new token by assigning the returned `RecordEvent`.

## Ordering guidelines (abstract)

Events are primarily used to express ordering **between pipeline classes** (for example, “a memory load must complete before a vector op consumes the tile”).

- Operations without explicit data or event dependencies may execute out of order on the device.
- Operations linked by an event dependency must observe the `Wait()`/`Record()` order implied by the program.

Instruction pages in `docs/isa/` indicate when ordering constraints matter for correctness.

## Minimal example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void pipeline(__gm__ float* in0, __gm__ float* in1, __gm__ float* out) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin0(in0), gin1(in1), gout(out);
  TileT a, b, c;

  Event<Op::TLOAD, Op::TADD> e0;
  Event<Op::TLOAD, Op::TADD> e1;
  Event<Op::TADD, Op::TSTORE_VEC> e2;

  e0 = TLOAD(a, gin0);
  e1 = TLOAD(b, gin1);
  e2 = TADD(c, a, b, e0, e1);
  TSTORE(gout, c, e2);
}
```
