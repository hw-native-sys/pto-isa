# pto.ttest

## 简介

`TTEST` 是非阻塞检测原语：检查一个或一组信号是否满足比较条件，满足时返回 `true`，否则立即返回 `false`。

它适合：

- 轮询式同步
- 带超时的等待
- 在等待期间穿插其他工作

既支持单个信号，也支持最多 5 维的信号 tensor。对 tensor 形式，只有当 **所有** 元素都满足条件时才返回 `true`。

## 数学语义

单个信号：

$$ \mathrm{result} = (\mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

信号 tensor：

$$ \mathrm{result} = \bigwedge_{d_0, d_1, d_2, d_3, d_4} (\mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue}) $$

其中 `cmp ∈ {EQ, NE, GT, GE, LT, LE}`。

## 汇编语法

PTO-AS 形式：

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

!!! warning "约束"
    - `GlobalSignalData::DType` 必须为 `int32_t`
    - `signalData` 必须指向本地地址（当前 NPU）
    - 单个信号的形状为 `<1,1,1,1,1>`
    - tensor 形式由其 shape 决定检测区域，并要求所有元素都满足条件

    ### 比较运算符

    | 值 | 条件 |
    | --- | --- |
    | `EQ` | `signal == cmpValue` |
    | `NE` | `signal != cmpValue` |
    | `GT` | `signal > cmpValue` |
    | `GE` | `signal >= cmpValue` |
    | `LT` | `signal < cmpValue` |
    | `LE` | `signal <= cmpValue` |

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
    return comm::TTEST(grid, 1, comm::WaitCmp::EQ);
}
```

### 与 TWAIT 的区别

`TWAIT` 会阻塞直到条件满足；`TTEST` 只返回当前检测结果，不会阻塞调用方。
