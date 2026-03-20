# TWAIT

## 简介

阻塞等待，直到信号满足比较条件。与 `TNOTIFY` 配合使用，实现基于标志的同步。

支持单个信号或多维信号 tensor（最高 5 维，形状由 GlobalTensor 决定）。

## 数学语义

自旋等待，直到以下条件满足：

单个信号：

$$\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}$$

信号 tensor（所有元素均须满足）：

$$\forall d_0, d_1, d_2, d_3, d_4: \mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue}$$

其中 `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../assembly/PTO-AS_zh.md)。

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

## 约束

- **类型约束**：
    - `GlobalSignalData::DType` 必须为 `int32_t`（32 位信号）。
- **内存约束**：
    - `signalData` 必须指向本地地址（当前 NPU）。
- **形状语义**：
    - 单个信号：形状为 `<1,1,1,1,1>`。
    - 信号 tensor：形状决定要等待的多维区域（最高 5 维）。tensor 中所有信号必须满足条件。
- **比较运算符**（WaitCmp）：
  | 值 | 条件 |
  |-------|--------|
  | `EQ` | `signal == cmpValue` |
  | `NE` | `signal != cmpValue` |
  | `GT` | `signal > cmpValue` |
  | `GE` | `signal >= cmpValue` |
  | `LT` | `signal < cmpValue` |
  | `LE` | `signal <= cmpValue` |

## 示例

### 等待单个信号

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // 等待 signal == 1
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### 等待信号矩阵

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 等待 4x8 网格中所有 worker 的信号就绪
void wait_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);

    // 等待所有 32 个信号均为 1
    comm::TWAIT(grid, 1, comm::WaitCmp::EQ);
}
```

### 等待计数器阈值

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);

    // 等待 counter >= expected_count
    comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
}
```

### 生产者-消费者模式

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 生产者：数据就绪后发送通知
void producer(__gm__ int32_t* remote_flag) {
    // ... 生产数据 ...

    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

// 消费者：等待数据就绪
void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);

    // ... 消费数据 ...
}
```

