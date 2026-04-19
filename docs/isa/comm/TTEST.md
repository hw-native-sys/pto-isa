# TTEST

`TTEST` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Non-blocking test whether signal(s) satisfy a comparison condition. Returns `true` if the condition is met, `false` otherwise. Use `TTEST` for polling-based synchronization when you need to interleave work with waiting, or to avoid blocking indefinitely. For blocking wait semantics, use `TWAIT` instead.

Supports single scalar signals and multi-dimensional signal tensors (up to 5-D). For tensors, returns `true` only if all signals in the tensor satisfy the condition.

## Mechanism

`TTEST` checks the signal condition and returns immediately. For single signals:

$$ \mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

For signal tensors (all must satisfy):

$$ \mathrm{result} = \bigwedge_{d_0, d_1, d_2, d_3, d_4} (\mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

where `cmp` is one of `EQ`, `NE`, `GT`, `GE`, `LT`, `LE`.

## Syntax

### PTO Assembly Form

```text
%result = ttest %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32) -> i1
%result = ttest %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32) -> i1
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `signalData` | `GlobalSignalData` | Signal or signal tensor; must be `int32_t` |
|| `cmpValue` | `int32_t` | Comparison threshold value |
|| `cmp` | `WaitCmp` | Comparison operator |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before testing |

## Expected Outputs

|| Result | Type | Description |
||--------|------|-------------|
|| `bool` | `true`/`false` | `true` if condition is satisfied, `false` otherwise |

## Side Effects

This operation reads signal state and returns immediately. No blocking or state modification.

## Constraints

### Type constraints

- `GlobalSignalData::DType` must be `int32_t`.

### Memory constraints

- `signalData` must point to a local address.

### Return value

- Returns `true` if the condition is satisfied, `false` otherwise.
- For signal tensors, returns `true` only if all signals satisfy the condition.

## Target-Profile Restrictions

- `TTEST` is supported on A2/A3 and A5 profiles. CPU simulation may implement simplified polling semantics.
- Use `TNOTIFY` on the producer side to update the signal.

## Examples

### Basic test

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool check_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    return comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```

### Test signal matrix

```cpp
bool check_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);
    // Returns true only if all 32 signals == 1
    return comm::TTEST(grid, 1, comm::WaitCmp::EQ);
}
```

### Polling with timeout

```cpp
bool poll_with_timeout(__gm__ int32_t* local_signal, int max_iterations) {
    comm::Signal sig(local_signal);

    for (int i = 0; i < max_iterations; ++i) {
        if (comm::TTEST(sig, 1, comm::WaitCmp::EQ))
            return true;
        // Do other useful work between polls
    }
    return false;
}
```

### Progress-based polling

```cpp
void process_with_progress(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);

    while (!comm::TTEST(counter, expected_count, comm::WaitCmp::GE)) {
        // Do useful work while waiting
    }
    // All expected signals received
}
```

### Compare TWAIT vs TTEST

```cpp
void compare_wait_test(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // Blocking: spins until signal == 1
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);

    // Non-blocking: returns immediately with result
    bool ready = comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```

## Related Ops / Instruction Set Links

- Communication overview: [Communication and Runtime](../other/communication-and-runtime.md)
- Blocking counterpart: [TWAIT](./TWAIT.md)
- Signal producer: [TNOTIFY](./TNOTIFY.md)
- Instruction set: [Other and Communication](../other/README.md)
