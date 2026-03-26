# TSYNC


## Tile Operation Diagram

![TSYNC tile operation](../figures/isa/TSYNC.svg)

## Introduction

Synchronize PTO execution:

- `TSYNC(events...)` waits on a set of explicit event tokens.
- `TSYNC<Op>()` inserts a pipeline barrier for a single vector op class.

Many intrinsics in `include/pto/common/pto_instr.hpp` call `TSYNC(events...)` internally before issuing the instruction.

## Math Interpretation

Not applicable.

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Event operand form:

```text
tsync %e0, %e1 : !pto.event<...>, !pto.event<...>
```

Single-op barrier form:

```text
tsync.op #pto.op<TADD>
```

### IR Level 1 (SSA)

```text
// Level 1 (SSA) does not support explicit synchronization primitives.
```

### IR Level 2 (DPS)

```text
pto.record_event[src_op, dst_op, eventID]
// 支持的op：TLOAD， TSTORE_ACC，TSTORE_VEC，TMOV_M2L，TMOV_M2S，TMOV_M2B，TMOV_M2V，TMOV_V2M，TMATMUL，TVEC
pto.wait_event[src_op, dst_op, eventID]
// 支持的op：TLOAD， TSTORE_ACC，TSTORE_VEC，TMOV_M2L，TMOV_M2S，TMOV_M2B，TMOV_M2V，TMOV_V2M，TMATMUL，TVEC
pto.barrier(op)
// 支持的op：TVEC,TMATMUL
```

In the current PTO-DSL front-end flow, `record_event` and `wait_event` should
be treated as low-level TSYNC forms. Front-end kernels SHOULD normally stay free
of explicit event wiring and rely on `ptoas --enable-insert-sync`.
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
  - `TSYNC_IMPL<Op>()` only supports vector-pipeline ops (`static_assert(pipe == PIPE_V)` in `include/pto/common/event.hpp`).
- **`TSYNC(events...)` semantics**:
  - `TSYNC(events...)` calls `WaitAllEvents(events...)`, which invokes `events.Wait()` on each event token.

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

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%result = pto.tsync ...
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%result = pto.tsync ...
```

### PTO Assembly Form

```text
tsync %e0, %e1 : !pto.event<...>, !pto.event<...>
# IR Level 2 (DPS)
pto.record_event[src_op, dst_op, eventID]
```
