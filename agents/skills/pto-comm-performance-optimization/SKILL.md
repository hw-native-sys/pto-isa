---
name: PTO-COMM 通信算子性能优化指南
description: PTO-COMM 通信算子性能优化方法论与实战技巧。涵盖通算重叠（Overlap）、乒乓双缓冲、多 Block 负载均衡、同步开销优化、数据布局优化、Tile 大小选择、带宽利用率分析、性能建模与 Profiling 方法等。触发：需要优化 PTO-COMM 通信算子性能、分析通信瓶颈、提升带宽利用率、设计通算重叠策略时。
license: CANN Open Software License Agreement Version 2.0
---

# PTO-COMM 通信算子性能优化指南

## 定位

本 Skill 是**知识库型 + 方法论型** Skill，提供 PTO-COMM 通信算子性能优化的系统方法、关键技巧和分析工具。

---

## 性能优化哲学

### 核心原则

1. **隐藏延迟**：将通信延迟隐藏在计算中（通算重叠）
2. **最大化带宽**：让硬件通信通路保持满载（流水线、大块传输）
3. **消除瓶颈**：识别并消除负载不均衡和同步等待

### 性能上界

```
总时间 ≥ max(计算时间, 通信时间, 同步开销)

通信时间 = 数据量 / 链路带宽
计算时间 = 计算量 / 计算峰值
同步开销 = 信号轮次 × 单次同步延迟
```

**理想状态**：通信完全隐藏在计算中，总时间 ≈ 计算时间。

### 优化优先级

```
1. 算法选择（O(N) 级影响）
   └── Ring vs Mesh vs 内置集合通信
2. 通算重叠策略（2x+ 级影响）
   └── 双 stream、队列调度、分块粒度
3. 数据搬运优化（1.5x 级影响）
   └── 乒乓双缓冲、Tile 大小、对齐
4. 同步优化（1.1~1.3x 级影响）
   └── barrier 合并、信号压缩、block 角色优化
```

---

## 各优化维度详解

| 维度 | 详细文档 |
|------|---------|
| 通算重叠（Overlap）设计 | [通算重叠详解](references/overlap-design.md) |
| 乒乓双缓冲与 Tile 选择 | [数据搬运优化](references/data-transfer-optimization.md) |
| 同步开销优化与算法选择 | [同步与算法优化](references/sync-algorithm-optimization.md) |
| 性能建模、Profiling 与迭代 | [性能分析方法](references/profiling-methodology.md) |

---

## 优化决策快速参考

### 场景决策表

| 场景 | 首要优化 | 参考 |
|------|---------|------|
| 通信时间 > 计算时间 | 增大计算粒度 / 减少通信量 | overlap-design.md |
| 计算时间 > 通信时间 | 通信已被隐藏，优化计算本身 | — |
| Overlap 效率 < 80% | 检查分块粒度、队列空转、Block 均衡 | overlap-design.md |
| 带宽利用率 < 60% | 增大 Tile、使用乒乓、检查对齐 | data-transfer-optimization.md |
| 多次 Barrier 开销高 | 合并 RS+Reduce 为 AtomicAdd | sync-algorithm-optimization.md |
| 首次运行慢 | Warmup 排除首次开销 | profiling-methodology.md |

### 重叠效率度量

```
重叠效率 = 1 - (实际总时间 - max(T_comp, T_comm)) / min(T_comp, T_comm)

100% = 完美重叠，总时间 = max(T_comp, T_comm)
  0% = 无重叠，总时间 = T_comp + T_comm
```

---

## 优化检查清单

### 通算重叠

- [ ] 计算和通信使用不同 Stream
- [ ] 计算完成即入队（不等待全部完成）
- [ ] 通信 kernel 使用 TTEST 轮询（非 TWAIT 阻塞）
- [ ] 分块粒度适当（计算第一个 Tile 前通信有事做）
- [ ] 测量并报告 overlap 效率

### 数据搬运

- [ ] 大量小块使用乒乓双缓冲
- [ ] Tile 大小充分利用 UB 空间
- [ ] GM 数据 32B 对齐
- [ ] 异步传输使用一维连续布局

### 同步优化

- [ ] RS + Reduce 融合为 AtomicAdd（减少一次 barrier）
- [ ] 跨 rank 信号仅 block 0 执行，其他 block 等本地标志
- [ ] 避免不必要的 pipe_barrier
- [ ] 使用 TTEST 替代 dcci + 软件轮询

### 负载均衡

- [ ] AG 阶段使用 row-level 均分
- [ ] RS 阶段队列分配均匀
- [ ] Block 数量与硬件资源匹配

### 算法层面

- [ ] AllReduce 分解为 RS + AG（消除 root 瓶颈）
- [ ] 数据量大时使用 P2P 组合而非内置集合通信
- [ ] FP16 AtomicAdd 精度可接受，否则改为独立 Reduce

---

## 相关 Skills

| Skill | 用途 |
|-------|------|
| `pto-comm-isa-reference` | 指令签名与约束速查 |
| `pto-comm-operator-develop` | 通信算子开发流程 |
| `pto-comm-testing-debug` | 测试与调试方法 |
| `vector-fusion-operator-generate` | 向量融合算子开发 |
