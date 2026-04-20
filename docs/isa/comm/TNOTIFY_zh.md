# TNOTIFY

`TNOTIFY` 是[通信与运行时](../other/communication-and-runtime_zh.md)指令集的一部分。

## 概述

向远端 NPU 发送标志通知，用于在不搬运大量数据的前提下建立轻量级同步。常与 `TWAIT`（阻塞等待）或 `TTEST`（轮询）配合使用，实现基于标志的生产者-消费者同步。

## 机制

`TNOTIFY` 向远端信号地址执行写操作。行为取决于选定的运算符：

`NotifyOp::Set`：

$$ \mathrm{signal}^{\mathrm{remote}} = \mathrm{value} $$

`NotifyOp::AtomicAdd`：

$$ \mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} \quad (\text{硬件原子}) $$

## 语法

### PTO 汇编形式

```text
tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

### C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value,
                     NotifyOp op, WaitEvents&... events);
```

## 输入

|| 操作数 | 类型 | 说明 |
||--------|------|------|
|| `dstSignalData` | `GlobalSignalData` | 远端信号地址；必须为 `int32_t` |
|| `value` | `int32_t` | 要写入或加上的值 |
|| `op` | `NotifyOp` | 运算符：`Set`（直接写入）或 `AtomicAdd`（硬件原子） |
|| `WaitEvents...` | `RecordEvent...` | 发指令前要等待的事件 |

## 预期输出

无。此操作是即发即忘型（fire-and-forget）。

## 副作用

此操作向远端全局内存写入数据。返回值（如果有）可用于建立同步边界。

## 约束

### 类型约束

- `GlobalSignalData::DType` 必须为 `int32_t`

### 内存约束

- `dstSignalData` 必须指向远端地址（目标 NPU）
- 信号地址建议满足 4 字节对齐

## 目标Profile限制

- `TNOTIFY` 在 A2/A3 和 A5 上支持。CPU 模拟器不支持远端信号操作。
- `AtomicAdd` 使用硬件原子存储指令；确保目标 NPU 支持该原子操作。

## 示例

### 基础 Set 通知

```cpp
void notify_set(__gm__ int32_t* remote_signal) {
    comm::Signal sig(remote_signal);
    comm::TNOTIFY(sig, 1, comm::NotifyOp::Set);
}
```

### 原子计数器自增

```cpp
void atomic_increment(__gm__ int32_t* remote_counter) {
    comm::Signal counter(remote_counter);
    comm::TNOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
}
```

### 生产者-消费者模式

```cpp
// 生产者：数据就绪后发信号
void producer(__gm__ int32_t* remote_flag) {
    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

// 消费者：阻塞等待数据
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);
}
```

## 相关页面

- 通信概述：[通信与运行时](../other/communication-and-runtime_zh.md)
- 阻塞等待：[TWAIT](./TWAIT_zh.md)
- 非阻塞轮询：[TTEST](./TTEST_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
