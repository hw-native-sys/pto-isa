# TTEST

`TTEST` 是[通信与运行时](../other/communication-and-runtime_zh.md)指令集的一部分。

## 概述

非阻塞检测原语：检查一个或一组信号是否满足比较条件，满足时返回 `true`，否则立即返回 `false`。适合轮询式同步、带超时的等待，或在等待期间穿插其他工作。

支持单个标量信号和最多 5 维的信号 tensor。对 tensor 形式，只有当所有元素都满足条件时才返回 `true`。

如需阻塞等待，请使用 `TWAIT`。

## 机制

`TTEST` 检查信号条件后立即返回。对单个信号：

$$ \mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

对信号 tensor（所有元素必须满足）：

$$ \mathrm{result} = \bigwedge_{d_0, d_1, d_2, d_3, d_4} (\mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

其中 `cmp` 为 `EQ`、`NE`、`GT`、`GE`、`LT`、`LE` 之一。

## 语法

### PTO 汇编形式

```text
%result = ttest %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32) -> i1
%result = ttest %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32) -> i1
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## 输入

|| 操作数 | 类型 | 说明 |
||--------|------|------|
|| `signalData` | `GlobalSignalData` | 信号或信号 tensor；必须为 `int32_t` |
|| `cmpValue` | `int32_t` | 比较阈值 |
|| `cmp` | `WaitCmp` | 比较运算符 |
|| `WaitEvents...` | `RecordEvent...` | 检测前要等待的事件 |

## 预期输出

|| 结果 | 类型 | 说明 |
||------|------|------|
|| `bool` | `true`/`false` | 条件满足时为 `true`，否则为 `false` |

## 副作用

此操作读取信号状态后立即返回。不阻塞，不修改状态。

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

- `TTEST` 在 A2/A3 和 A5 上支持。CPU 模拟器可能实现简化的轮询语义。
- 生产者侧使用 `TNOTIFY` 更新信号。

## 示例

### 基础检测

```cpp
bool check_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    return comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```

### 检测信号矩阵

```cpp
bool check_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);
    // 所有 32 个信号都 == 1 时才返回 true
    return comm::TTEST(grid, 1, comm::WaitCmp::EQ);
}
```

### 带超时的轮询

```cpp
bool poll_with_timeout(__gm__ int32_t* local_signal, int max_iterations) {
    comm::Signal sig(local_signal);

    for (int i = 0; i < max_iterations; ++i) {
        if (comm::TTEST(sig, 1, comm::WaitCmp::EQ))
            return true;
        // 在轮询间隔内做其他有用的工作
    }
    return false;
}
```

### TWAIT vs TTEST

```cpp
// 阻塞：直到 signal == 1 才返回
comm::TWAIT(sig, 1, comm::WaitCmp::EQ);

// 非阻塞：立即返回当前检测结果
bool ready = comm::TTEST(sig, 1, comm::WaitCmp::EQ);
```

## 相关页面

- 通信概述：[通信与运行时](../other/communication-and-runtime_zh.md)
- 阻塞等待：[TWAIT](./TWAIT_zh.md)
- 信号发送：[TNOTIFY](./TNOTIFY_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
