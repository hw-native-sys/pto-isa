# TWAIT

`TWAIT` 是[通信与运行时](../other/communication-and-runtime_zh.md)指令集的一部分。

## 概述

阻塞等待原语：在本地信号满足比较条件之前一直等待。常与 `TNOTIFY` 配合使用，实现基于标志的生产者-消费者同步。支持单个标量信号和最多 5 维的信号 tensor。对 tensor 形式，要求所有元素都满足比较条件后才结束等待。

`TWAIT` 是阻塞调用：不满足条件时会一直等待。如需非阻塞轮询，请使用 `TTEST`。

## 机制

`TWAIT` 轮询信号条件直到满足。对单个信号：

$$ \text{wait until}\ \mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

对信号 tensor（所有元素必须同时满足）：

$$ \forall d_0, d_1, d_2, d_3, d_4:\ \mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

其中 `cmp` 为 `EQ`、`NE`、`GT`、`GE`、`LT`、`LE` 之一。

## 语法

### PTO 汇编形式

```text
twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
twait %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## 输入

|| 操作数 | 类型 | 说明 |
||--------|------|------|
|| `signalData` | `GlobalSignalData` | 信号或信号 tensor；必须为 `int32_t` |
|| `cmpValue` | `int32_t` | 比较阈值 |
|| `cmp` | `WaitCmp` | 比较运算符 |
|| `WaitEvents...` | `RecordEvent...` | 进入轮询前要等待的事件 |

## 预期输出

无。此操作阻塞直到条件满足，然后返回。

## 副作用

此操作可能无限阻塞（如果信号永不满足条件）。不修改架构状态。

## 约束

### 类型约束

- `GlobalSignalData::DType` 必须为 `int32_t`

### 内存约束

- `signalData` 必须指向本地地址

### 比较运算符

|| 值 | 条件 |
||---|------|
|| `WaitCmp::EQ` | signal == cmpValue |
|| `WaitCmp::NE` | signal != cmpValue |
|| `WaitCmp::GT` | signal > cmpValue |
|| `WaitCmp::GE` | signal >= cmpValue |
|| `WaitCmp::LT` | signal < cmpValue |
|| `WaitCmp::LE` | signal <= cmpValue |

## 目标Profile限制

- `TWAIT` 在 A2/A3 和 A5 上支持。CPU 模拟器可能不实现阻塞等待语义。
- 生产者侧使用 `TNOTIFY` 更新信号。

## 示例

### 等待单个信号

```cpp
void wait_for_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### 等待信号矩阵

```cpp
void wait_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);
    // 等待所有 32 个信号 == 1
    comm::TWAIT(grid, 1, comm::WaitCmp::EQ);
}
```

### 等待计数器达到阈值

```cpp
void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);
    comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
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
    // ... 处理数据 ...
}
```

## 相关页面

- 通信概述：[通信与运行时](../other/communication-and-runtime_zh.md)
- 信号发送：[TNOTIFY](./TNOTIFY_zh.md)
- 非阻塞轮询：[TTEST](./TTEST_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
