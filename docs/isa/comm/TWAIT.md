# TWAIT

`TWAIT` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Blocking wait for signal(s) to satisfy a comparison condition. Used in conjunction with `TNOTIFY` to implement flag-based producer-consumer synchronization. Supports single scalar signals and multi-dimensional signal tensors (up to 5-D).

`TWAIT` is a blocking call: it does not return until the condition is satisfied. For non-blocking polling, see `TTEST`.

## Mechanism

`TWAIT` spins until all checked signals satisfy the specified comparison. For single signals, it waits on one value. For signal tensors, all elements in the tensor must satisfy the condition simultaneously.

Single signal:

$$ \text{wait until}\ \mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

Signal tensor (all elements must satisfy):

$$ \forall d_0, d_1, d_2, d_3, d_4: \mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

where `cmp` is one of `EQ`, `NE`, `GT`, `GE`, `LT`, `LE`.

## Syntax

### PTO Assembly Form

```text
twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
twait %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

### Comparison operators

|| Value | Condition |
||-------|-----------|
|| `WaitCmp::EQ` | signal == cmpValue |
|| `WaitCmp::NE` | signal != cmpValue |
|| `WaitCmp::GT` | signal > cmpValue |
|| `WaitCmp::GE` | signal >= cmpValue |
|| `WaitCmp::LT` | signal < cmpValue |
|| `WaitCmp::LE` | signal <= cmpValue |

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `signalData` | `GlobalSignalData` | Signal or signal tensor; must be `int32_t` |
|| `cmpValue` | `int32_t` | Comparison threshold value |
|| `cmp` | `WaitCmp` | Comparison operator |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before entering the spin loop |

## Expected Outputs

None. This operation blocks until the condition is satisfied and then returns.

## Side Effects

This operation may block indefinitely if the signal never satisfies the condition. No architectural state is modified.

## Constraints

### Type constraints

- `GlobalSignalData::DType` must be `int32_t`.

### Memory constraints

- `signalData` must point to a local address.

### Shape semantics

- For a single signal, the effective shape is `<1,1,1,1,1>`.
- For a signal tensor, the shape determines the multi-dimensional region to wait on; all elements must satisfy the condition.

## Target-Profile Restrictions

- `TWAIT` is supported on A2/A3 and A5 profiles. CPU simulation may not implement blocking wait semantics.
- Use `TNOTIFY` on the producer side to signal when data is ready.

## Examples

### Wait for single signal

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### Wait for signal matrix

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);
    // Waits until all 32 signals == 1
    comm::TWAIT(grid, 1, comm::WaitCmp::EQ);
}
```

### Wait for counter threshold

```cpp
void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);
    comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
}
```

### Producer-consumer pattern

```cpp
// Producer: signals when data is ready
void producer(__gm__ int32_t* remote_flag) {
    // ... produce data ...
    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

// Consumer: blocks until data is signaled
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);
    // ... consume data ...
}
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Signal producer: [TNOTIFY](./TNOTIFY.md)
- Non-blocking counterpart: [TTEST](./TTEST.md)
- Instruction set: [Other and Communication](../other/README.md)
