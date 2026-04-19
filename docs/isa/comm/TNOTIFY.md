# TNOTIFY

`TNOTIFY` is part of the [Communication and Runtime](../other/communication-and-runtime.md) instruction set.

## Summary

Send a flag notification to a remote NPU. Used for lightweight inter-NPU synchronization without bulk data transfer. The remote signal is updated atomically.

Used in conjunction with `TWAIT` (blocking) or `TTEST` (polling) on the receiver side to implement producer-consumer patterns.

## Mechanism

`TNOTIFY` writes to a remote signal location. The operation semantics depend on the selected operator:

For `NotifyOp::Set`:

$$ \mathrm{signal}^{\mathrm{remote}} = \mathrm{value} $$

For `NotifyOp::AtomicAdd`:

$$ \mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{hardware atomic}) $$

## Syntax

### PTO Assembly Form

```text
tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

### C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value,
                      NotifyOp op, WaitEvents&... events);
```

## Inputs

|| Operand | Type | Description |
||---------|------|-------------|
|| `dstSignalData` | `GlobalSignalData` | Remote signal location; must be `int32_t` |
|| `value` | `int32_t` | Value to set or add |
|| `op` | `NotifyOp` | Operator: `Set` (direct store) or `AtomicAdd` (hardware atomic) |
|| `WaitEvents...` | `RecordEvent...` | Events to wait on before issuing the notification |

## Expected Outputs

None. This is a fire-and-forget operation.

## Side Effects

This operation writes to remote global memory. It may establish synchronization edges through the returned event token.

## Constraints

### Type constraints

- `GlobalSignalData::DType` must be `int32_t`.

### Memory constraints

- `dstSignalData` must point to a remote address (on the target NPU).
- The signal location should be 4-byte aligned.

## Target-Profile Restrictions

- `TNOTIFY` is supported on A2/A3 and A5 profiles. CPU simulation does not implement remote signal operations.
- `AtomicAdd` uses hardware atomic store instructions; ensure the target NPU supports the atomic operation.

## Examples

### Basic set notification

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_set(__gm__ int32_t* remote_signal) {
    comm::Signal sig(remote_signal);
    comm::TNOTIFY(sig, 1, comm::NotifyOp::Set);
}
```

### Atomic counter increment

```cpp
void atomic_increment(__gm__ int32_t* remote_counter) {
    comm::Signal counter(remote_counter);
    comm::TNOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
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
- Blocking counterpart: [TWAIT](./TWAIT.md)
- Non-blocking counterpart: [TTEST](./TTEST.md)
- Instruction set: [Other and Communication](../other/README.md)
