# 信号同步指令详解（TNOTIFY / TWAIT / TTEST）

## TNOTIFY — 发送信号通知

向远端 NPU 发送标志通知，用于轻量级同步。

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
void TNOTIFY(GlobalSignalData &dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events);
```

### 约束

- `GlobalSignalData::DType` 必须为 `int32_t`
- `dstSignalData` 必须指向远端地址（目标 NPU）
- `dstSignalData` 应 4 字节对齐
- `NotifyOp::Set` 执行直接存储
- `NotifyOp::AtomicAdd` 使用硬件原子加指令

### 示例

```cpp
// 直接赋值通知
comm::Signal sig(remote_signal);
comm::TNOTIFY(sig, 1, comm::NotifyOp::Set);

// 原子计数器自增
comm::Signal counter(remote_counter);
comm::TNOTIFY(counter, 1, comm::NotifyOp::AtomicAdd);
```

**重要**：TNOTIFY 发送到**远端**地址，TWAIT/TTEST 检测**本地**地址。成对使用时需注意地址方向。

---

## TWAIT — 阻塞等待信号

阻塞等待，直到信号满足比较条件。

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

### 约束

- `GlobalSignalData::DType` 必须为 `int32_t`
- `signalData` 必须指向**本地**地址（当前 NPU）
- 支持单个信号和多维信号 tensor（最高 5 维）
- 对于 tensor，**所有**信号必须满足条件才会返回

### 示例

```cpp
// 等待单个信号
comm::Signal sig(local_signal);
comm::TWAIT(sig, 1, comm::WaitCmp::EQ);

// 等待信号矩阵（4×8 网格所有元素 >= 1）
comm::Signal2D<4, 8> grid(signal_matrix);
comm::TWAIT(grid, 1, comm::WaitCmp::GE);

// 等待计数器达到阈值
comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
```

---

## TTEST — 非阻塞信号检测

非阻塞检测信号条件，返回 `bool`。

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

### 约束

与 TWAIT 相同，但不阻塞。

### 示例

```cpp
// 非阻塞检测
bool ready = comm::TTEST(sig, 1, comm::WaitCmp::EQ);

// 带超时的轮询
for (int i = 0; i < max_iters; ++i) {
    if (comm::TTEST(sig, 1, comm::WaitCmp::EQ)) {
        break;
    }
}
```

---

## TWAIT vs TTEST 选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 确定必须等待（barrier） | TWAIT | 硬件自旋，更节能 |
| 等待期间需执行其他工作 | TTEST | 可交错执行 |
| 需要超时控制 | TTEST | 可设循环上限 |
| 就绪队列消费 | TTEST | 先检查再处理 |
