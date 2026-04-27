# pto.twait

`pto.twait` is part of the [Collective Communication](communication-runtime.md) instruction set.

## Summary

Blocking wait until a signal (or all elements of a signal tensor) satisfies a comparison condition against a constant. Used with `pto.tnotify` for inter-NPU flag-based synchronization.

## Mechanism

`pto.twait` spins on a signal location until the comparison condition is satisfied. The operation halts the current NPU's scalar unit until the condition becomes true.

Single signal: the NPU waits until the scalar at the signal address satisfies `signal cmp cmpValue`.

Signal tensor: the NPU waits until **all** elements in the tensor satisfy the condition simultaneously.

The signal address must point to local (on-chip) memory on the current NPU.

## Assembly Syntax

```text
pto.twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
pto.twait %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void WAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `signalData` | Signal or signal tensor. Must be on local NPU memory. |
| `cmpValue` | Constant comparison value. |
| `cmp` | Comparison operator. |

### Comparison Operators

| Value | Condition |
|-------|-----------|
| `EQ` | `signal == cmpValue` |
| `NE` | `signal != cmpValue` |
| `GT` | `signal > cmpValue` |
| `GE` | `signal >= cmpValue` |
| `LT` | `signal < cmpValue` |
| `LE` | `signal <= cmpValue` |

## Expected Outputs

None. The operation blocks until the condition is satisfied.

## Side Effects

Halts the scalar unit. Does not affect other NPUs.

## Constraints

- `GlobalSignalData::DType` must be `int32_t`.
- `signalData` must point to local address on the current NPU.
- For signal tensors: all elements must satisfy the condition simultaneously.
- Up to 5-D tensor shapes are supported.

## Exceptions

- Using a non-local signal address is undefined behavior.
- The signal address must be accessible throughout the wait duration.

## Examples

### Wait for Single Signal

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void wait_ready(__gm__ int32_t* local_signal) {
  comm::Signal sig(local_signal);
  comm::WAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### Wait for Signal Matrix

```cpp
void wait_worker_grid(__gm__ int32_t* signal_matrix) {
  comm::Signal2D<4, 8> grid(signal_matrix);
  comm::WAIT(grid, 1, comm::WaitCmp::EQ);  // waits until all 32 signals == 1
}
```

### Producer-Consumer Pattern

```cpp
// Producer
void producer(__gm__ int32_t* remote_flag) {
  comm::Signal flag(remote_flag);
  comm::NOTIFY(flag, 1, comm::NotifyOp::Set);
}

// Consumer
void consumer(__gm__ int32_t* local_flag) {
  comm::Signal flag(local_flag);
  comm::WAIT(flag, 1, comm::WaitCmp::EQ);
}
```

## See Also

- [Collective Communication](communication-runtime.md) for related operations
- `pto.tnotify` for the signaling half of this protocol
