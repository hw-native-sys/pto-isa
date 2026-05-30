# pto.twait

## 简介

`TWAIT` 是阻塞等待原语：在本地信号满足比较条件之前一直等待。它通常与 `TNOTIFY` 配合使用，实现基于标志的同步。

既支持单个信号，也支持最多 5 维的信号 tensor。对 tensor 形式，要求所有元素都满足比较条件后才结束等待。

## 数学语义

单个信号：

$$ \mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

信号 tensor：

$$ \forall d_0, d_1, d_2, d_3, d_4:\ \mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

其中 `cmp ∈ {EQ, NE, GT, GE, LT, LE}`。

## 汇编语法

PTO-AS 形式：

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

!!! warning "约束"
    - `GlobalSignalData::DType` 必须为 `int32_t`
    - `signalData` 必须指向本地地址（当前 NPU）
    - 单个信号的形状为 `<1,1,1,1,1>`
    - tensor 形式由其 shape 决定等待区域，并要求所有元素满足条件

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

### 等待单个信号

```cpp
void wait_for_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### 等待计数器达到阈值

```cpp
void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);
    comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
}
```

### 与 TNOTIFY 配合

```cpp
void producer(__gm__ int32_t* remote_flag) {
    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);
}
```
