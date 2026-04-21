# 同步与算法优化

## 同步开销优化

### Barrier 合并

将多个 phase barrier 合并为一个：

```
优化前：[RS] → [Barrier] → [local reduce] → [Barrier] → [AG]
优化后：[RS with AtomicAdd] → [Barrier] → [AG]
         ^^^^^^^^^^^^^^^^^^^^^
         RS 和 Reduce 融合为 TPUT<AtomicAdd>
```

通过使用 `AtomicAdd` 在 RS 阶段直接累加，消除了独立的 Reduce 阶段及其 barrier。

### 信号压缩

减少跨 rank 通知次数：

```
优化前：每个 tile 完成都通知远端 → N_tiles × nranks 次 TNOTIFY
优化后：所有 tile 完成后通知一次 → nranks 次 TNOTIFY
```

### Block 角色优化

只让 block 0 执行跨 rank 信号，其他 block 等待本地广播标志：

```
Block 0: TNOTIFY(remote) × (nranks-1) → TWAIT(local) × (nranks-1) → Set local flag
Block 1~N: TWAIT(local flag)  ← 一次 TWAIT 而非 nranks 次
```

### TTEST vs TWAIT 选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 确定必须等待（barrier） | TWAIT | 硬件自旋，更节能 |
| 轮询+做其他工作 | TTEST | 可交错执行 |
| 就绪队列消费 | TTEST | 先检查再处理 |

### 减少 dcci 调用

`dcci` 刷新缓存行是标量操作，频繁调用影响性能：

```cpp
// 优化前：每次读队列数据都 dcci
for (int i = 0; i < count; i++) {
    dcci(&queue->data[i], SINGLE_CACHE_LINE);
    process(queue->data[i]);
}

// 优化后：使用 TTEST 硬件指令代替 dcci + 软件比较
comm::Signal sig(&queue->count);
if (comm::TTEST(sig, expected, comm::WaitCmp::GE)) {
    dcci(&queue->data[head], SINGLE_CACHE_LINE);
    process(queue->data[head]);
}
```

---

## 算法选择

### AllReduce 分解策略

| 策略 | 通信量 | 延迟 | 适用场景 |
|------|--------|------|---------|
| ReduceScatter + AllGather | 2(N-1)/N × S | 2(N-1) steps | 中大数据量 |
| Ring AllReduce | 2(N-1)/N × S | 2(N-1) steps | 大数据量，带宽受限 |
| 内置 TREDUCE + TBROADCAST | N × S | 2 steps | 小数据量，root 带宽足够 |
| TPUT\<AtomicAdd\> RS + TPUT AG | 2(N-1)/N × S | 可重叠 | 通算融合场景 |

其中 S = 数据总大小，N = rank 数。

### RS 实现：AtomicAdd vs 独立 Reduce

**AtomicAdd 方式**（推荐用于融合）：
- RS 阶段使用 `TPUT<AtomicAdd>` 直接累加到 owner
- 无需独立 Reduce 阶段和额外 barrier
- FP16 下有累积精度损失

**独立 Reduce 方式**：
- RS 只做 scatter（无归约）
- owner 本地执行 TLOAD + TADD + TSTORE 归约
- 精度更好，但需要额外阶段和 barrier

### AG 实现

- **TPUT 直写**：owner rank 主动写到所有远端（推荐）
- **TGET 拉取**：各 rank 从 owner 拉取
- **TBROADCAST**：owner 使用内置集合通信广播

通常选择 **TPUT 直写**，因为 owner 知道数据就绪时机，无需额外通知。
