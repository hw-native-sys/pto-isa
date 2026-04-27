# pto.tnotify

## Introduction

Send flag notification to remote NPU. Used for lightweight synchronization between NPUs without transferring bulk data.

## Math Interpretation

For `NotifyOp::Set`:

$$ \mathrm{signal}^{\mathrm{remote}} = \mathrm{value} $$

For `NotifyOp::AtomicAdd`:

$$ \mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{atomic}) $$

## Assembly Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

```text
pto.tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
pto.tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void NOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events);
```

## Constraints

- **Type constraints**:
    - `GlobalSignalData::DType` must be `int32_t` (32-bit signal).
- **Memory constraints**:
    - `dstSignalData` must point to remote address (on target NPU).
    - `dstSignalData` should be 4-byte aligned.
- **Operation semantics**:
    - `NotifyOp::Set`: Direct store to remote memory.
    - `NotifyOp::AtomicAdd`: Hardware atomic add using `st_atomic` instruction.

## Examples

### Basic Set Notification

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void notify_set(__gm__ int32_t* remote_signal) {
    comm::Signal sig(remote_signal);

    // Set remote signal to 1
    comm::NOTIFY(sig, 1, comm::NotifyOp::Set);
}
```

### Atomic Counter Increment

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void atomic_increment(__gm__ int32_t* remote_counter) {
    comm::Signal counter(remote_counter);

    // Atomically add 1 to remote counter
    comm::NOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
}
```

### Producer-Consumer Pattern

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// Producer: notify when data is ready
void producer(__gm__ int32_t* remote_flag) {
    // ... produce data ...

    comm::Signal flag(remote_flag);
    comm::NOTIFY(flag, 1, comm::NotifyOp::Set);
}

// Consumer: wait for data
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::WAIT(flag, 1, comm::WaitCmp::EQ);

    // ... consume data ...
}
```
