# pto.tnotify

## 简介

`TNOTIFY` 向远端 NPU 发送标志通知，用于在不搬运大量数据的前提下建立轻量级同步。

## 数学语义

`NotifyOp::Set`：

$$ \mathrm{signal}^{\mathrm{remote}} = \mathrm{value} $$

`NotifyOp::AtomicAdd`：

$$ \mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{value} $$

## 汇编语法

PTO-AS 形式：

```text
tnotify %signal_remote, %value {op = #pto.notify_op<Set>} : (!pto.memref<i32>, i32)
tnotify %signal_remote, %value {op = #pto.notify_op<AtomicAdd>} : (!pto.memref<i32>, i32)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events);
```

## 约束

!!! warning "约束"
    - `GlobalSignalData::DType` 必须为 `int32_t`
    - `dstSignalData` 必须指向远端地址
    - `dstSignalData` 建议满足 4 字节对齐
    - `NotifyOp::Set` 表示直接写入
    - `NotifyOp::AtomicAdd` 表示原子加

## 示例

### 基础通知

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
