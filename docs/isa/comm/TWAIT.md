# pto.twait

`pto.twait` is part of the [Collective Communication](communication-runtime.md) instruction set.

Blocking wait until the signal (or all signals in a signal tensor) satisfies the given comparison condition. Used in conjunction with `TNOTIFY` for flag-based synchronization.

Blocking wait until a signal (or all elements of a signal tensor) satisfies a comparison condition against a constant. Used with `pto.tnotify` for inter-NPU flag-based synchronization.

## Mechanism

`pto.twait` spins on a signal location until the comparison condition is satisfied. The operation halts the current NPU's scalar unit until the condition becomes true.

Single signal: the NPU waits until the scalar at the signal address satisfies `signal cmp cmpValue`.

Signal tensor: the NPU waits until **all** elements in the tensor satisfy the condition simultaneously.

The signal address must point to local (on-chip) memory on the current NPU.

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

- **Type constraints**:
    - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
    - `signalData` must point to local address (on current NPU).
- **Shape semantics**:
    - For single signal: Shape is `<1,1,1,1,1>`.
    - For signal tensor: Shape determines the multi-dimensional region (up to 5-D) to wait on. All signals in the tensor must satisfy the condition.
**Comparison operators** (WaitCmp):

| Value | Condition |
|-------|-----------|
| `EQ` | `signal == cmpValue` |
| `NE` | `signal != cmpValue` |
| `GT` | `signal > cmpValue` |
| `GE` | `signal >= cmpValue` |
| `LT` | `signal < cmpValue` |
| `LE` | `signal <= cmpValue` |

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
