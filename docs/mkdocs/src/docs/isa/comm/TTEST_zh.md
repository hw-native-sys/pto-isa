<!-- Generated from `docs/isa/comm/TTEST_zh.md` -->

# TTEST

## 简介

非阻塞检测信号是否满足比较条件。满足则返回 `true`，否则返回 `false`。适用于基于轮询的同步（含超时）或与其他工作交错执行的场景。

支持单个信号或多维信号 tensor（最高 5 维，形状由 GlobalTensor 决定）。对于 tensor，仅当**所有**信号均满足条件时才返回 `true`。

## 数学语义

检测并返回结果：

单个信号：

$$\mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue})$$

信号 tensor（所有元素均须满足）：

$$\mathrm{result} = \bigwedge_{d_0, d_1, d_2, d_3, d_4} (\mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue})$$

其中 `cmp` ∈ {`EQ`, `NE`, `GT`, `GE`, `LT`, `LE`}

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../assembly/PTO-AS_zh.md)。

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

## 约束

- **类型约束**：
    - `GlobalSignalData::DType` 必须为 `int32_t`（32 位信号）。
- **内存约束**：
    - `signalData` 必须指向本地地址（当前 NPU）。
- **返回值**：
    - 条件满足时返回 `true`，否则返回 `false`。
    - 对于信号 tensor，仅当所有信号均满足条件时才返回 `true`。
- **形状语义**：
    - 单个信号：形状为 `<1,1,1,1,1>`。
    - 信号 tensor：形状决定要检测的多维区域（最高 5 维）。
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

### 基础检测

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool check_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // 检测 signal == 1
    return comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```

### 检测信号矩阵

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// 检测 4x8 网格中所有 worker 的信号是否就绪
bool check_worker_grid(__gm__ int32_t* signal_matrix) {
    comm::Signal2D<4, 8> grid(signal_matrix);

    // 仅当所有 32 个信号均为 1 时返回 true
    return comm::TTEST(grid, 1, comm::WaitCmp::EQ);
}
```

### 带超时的轮询

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

bool poll_with_timeout(__gm__ int32_t* local_signal, int max_iterations) {
    comm::Signal sig(local_signal);

    for (int i = 0; i < max_iterations; ++i) {
        if (comm::TTEST(sig, 1, comm::WaitCmp::EQ)) {
            return true;  // 收到信号
        }
        // 两次轮询之间可执行其他工作
    }
    return false;  // 超时
}
```

### 基于进度的轮询

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void process_with_progress(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);

    while (!comm::TTEST(counter, expected_count, comm::WaitCmp::GE)) {
        // 等待期间执行其他有用工作
        // ...
    }
    // 所有预期信号均已收到
}
```

### TWAIT 与 TTEST 对比

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

void compare_wait_test(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);

    // 阻塞：自旋直到 signal == 1
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);

    // 非阻塞：立即返回结果
    bool ready = comm::TTEST(sig, 1, comm::WaitCmp::EQ);
}
```
