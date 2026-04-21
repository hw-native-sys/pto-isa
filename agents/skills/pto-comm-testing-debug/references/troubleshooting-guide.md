# 问题诊断手册

## 1. 死锁（程序挂起）

**症状**：程序无响应，`mpirun` 超时。

### 排查步骤

| 检查项 | 方法 | 典型原因 |
|--------|------|---------|
| TWAIT 未匹配 TNOTIFY | 检查每个 TWAIT 是否有对应的 TNOTIFY 发送 | 漏发通知或发送方向错误 |
| Barrier 不对称 | 确认所有 rank 都执行 barrier | 部分 rank 跳过 barrier 路径 |
| Signal 地址错误 | 打印 signal 地址确认远端/本地正确 | 远端地址计算错误 |
| Block 数不匹配 | 确认 intra-rank 计数器期望值 | `num_comm_blocks - 1` 与实际不符 |

### 超时保护模式

```cpp
int timeout = 1000000;
while (timeout-- > 0) {
    if (comm::TTEST(sig, expected, comm::WaitCmp::GE)) break;
}
if (timeout <= 0) {
    dcci((__gm__ void *)signal_ptr, SINGLE_CACHE_LINE);
    // 记录异常
}
```

---

## 2. 数据错误

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| 全零 | 传输未执行 / 地址错误 | 检查远端地址计算和 kernel 是否启动 |
| 随机值 | 读到未初始化内存 | 检查信号同步是否正确（先写后读） |
| 部分正确 | Tiling 边界问题 | 检查 AlignUp 和 Tile 边界处理 |
| NaN/Inf | FP16 溢出 | 检查 AtomicAdd 累积次数和数据范围 |
| 接近但不精确 | FP16 精度限制 | 放宽 atol/rtol 阈值 |

---

## 3. 信号残留

**症状**：第一次运行正确，第二次运行结果错误或提前通过 barrier。

**原因**：信号矩阵未在每次运行前清零。

**修复**：

```cpp
aclrtMemset(signal_matrix, signal_size, 0, signal_size);
aclrtSynchronizeStream(stream);
```

---

## 4. 编译错误

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `MEMORY_BASE` undefined | 编译选项缺少 `-DMEMORY_BASE` | CMakeLists 添加 target_compile_definitions |
| `comm::` 符号未找到 | 未包含 `pto_comm_inst.hpp` | 检查 include 路径 |
| `__gm__` 未定义 | CPU 编译时使用了 NPU 类型 | 检查 `#ifdef __CCE_AICORE__` 条件编译 |
| link error: runtime | 未链接 runtime 库 | CMakeLists 添加 `target_link_libraries(... runtime)` |

---

## 5. TPUT_ASYNC 返回无效 event

**症状**：`event.valid()` 返回 false（handle == 0）。

**原因**：
- 传入的 tensor 不是扁平连续一维
- BuildAsyncSession 失败（workspace 无效）
- A5 平台 MTE fallback 完成时返回 handle=0（正常行为）

**排查**：

```cpp
auto event = comm::TPUT_ASYNC(dstG, srcG, session);
if (!event.valid()) {
    // 检查 session.valid
    // 检查 tensor 是否一维连续
    // A5 平台下 MTE fallback 已完成，无需 Wait
}
```

---

## 6. dcci 缓存一致性问题

**症状**：Device 侧读到陈旧数据。

**原因**：AICore 读 GM 可能命中 L1 缓存，看不到其他核的写入。

**解决**：

```cpp
dcci((__gm__ void *)&shared_data, SINGLE_CACHE_LINE);
__asm__ __volatile__("");  // 编译器屏障
int32_t value = shared_data;
```
