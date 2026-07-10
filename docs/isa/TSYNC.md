# TSYNC


## Tile Operation Diagram

![TSYNC tile operation](../figures/isa/TSYNC.svg)

## Introduction

Synchronize PTO execution:

- `TSYNC(events...)` waits on a set of explicit event tokens.
- `TSYNC<Op>()` inserts a pipe barrier for the pipeline of the specified `Op`.

Many intrinsics in `include/pto/common/pto_instr.hpp` call `TSYNC(events...)` internally before issuing the instruction.

## Math Interpretation

Not applicable.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <Op OpCode>
PTO_INST void TSYNC();

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &... events);
```

## Constraints

- **Implementation checks (`TSYNC<Op>()`)**:
    - **A2A3**: `TSYNC_IMPL<Op>()` supports S / V / M / MTE1 / MTE2 / MTE3 / FIX / ALL pipelines (`static_assert` in `include/pto/npu/a2a3/TSync.hpp`).
    - **A5**: `TSYNC_IMPL<Op>()` only supports MTE2 / MTE3 / ALL pipelines (`static_assert` in `include/pto/npu/a5/TSync.hpp`).
- **`TSYNC(events...)` semantics**:
    - `TSYNC(events...)` calls `WaitAllEvents(events...)`, which invokes `events.Wait()` on each event token. In auto mode, this is no-op.

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
  Event<Op::TLOAD, Op::TADD> e;
  e = TLOAD(t, gin);
  TSYNC(e);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  Event<Op::TADD, Op::TSTORE_VEC> e;
  e = TADD(c, a, b);
  TSYNC<Op::TADD>();
  TSYNC(e);
}
```
