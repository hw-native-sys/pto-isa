---
name: PTO-COMM 通信算子测试与调试指南
description: PTO-COMM 通信算子的测试方法和调试技巧。涵盖 CPU 仿真测试、NPU 硬件测试、Golden 数据生成、正确性验证、多 rank 测试框架、常见运行时错误诊断、信号死锁排查、精度问题分析、mssanitizer 内存检测等。触发：需要测试 PTO-COMM 通信算子、生成 Golden 数据、排查通信死锁/数据错误/信号异常、编写测试用例时。
license: CANN Open Software License Agreement Version 2.0
---

# PTO-COMM 通信算子测试与调试指南

## 定位

本 Skill 是**流程型 + 诊断型** Skill，覆盖 PTO-COMM 通信算子的完整测试流程和常见问题调试方法。

---

## 测试体系概述

### 测试层次

| 层次 | 目的 | 环境 | 速度 |
|------|------|------|------|
| CPU 仿真 | 功能验证、快速迭代 | x86_64 / AArch64 | 秒级 |
| 单指令 ST | 验证单条通信指令 | NPU 硬件 | 分钟级 |
| 算子级 ST | 验证完整通信算子 | 多 NPU 硬件 | 分钟级 |
| 性能测试 | 带宽/延迟/吞吐 | 多 NPU 硬件 | 分钟级 |

### 测试目录结构

```
tests/
├── cpu/comm/st/testcase/       # CPU 仿真测试
│   ├── common.hpp
│   ├── CMakeLists.txt
│   ├── tgather/
│   └── tscatter/
├── npu/a2a3/comm/st/testcase/  # NPU A2A3 测试
│   ├── common.hpp
│   ├── hccl_context.h
│   ├── comm_mpi.h
│   ├── CMakeLists.txt
│   ├── tput/ tget/ tnotify/ twait/ ttest/
│   ├── tgather/ tscatter/ tbroadcast/ treduce/
│   └── tput_async/ tget_async/
└── npu/a5/comm/st/testcase/    # NPU A5 测试
```

---

## 各测试维度详解

| 维度 | 详细文档 |
|------|---------|
| CPU 仿真与 NPU 硬件测试 | [测试环境与运行](references/test-environments.md) |
| Golden 数据生成与正确性验证 | [正确性验证方法](references/correctness-verification.md) |
| 常见问题诊断与调试工具 | [问题诊断手册](references/troubleshooting-guide.md) |
| mssanitizer 内存检测与高级调试 | [高级调试工具](references/advanced-debug-tools.md) |

---

## 快速诊断表

| 症状 | 可能原因 | 首查项 |
|------|---------|--------|
| 程序挂起/超时 | TWAIT/TNOTIFY 不匹配、barrier 不对称 | 信号方向、block 数一致性 |
| 全零输出 | 传输未执行、地址错误 | 远端地址计算、kernel 是否启动 |
| 随机值输出 | 读到未初始化内存 | 信号同步先写后读顺序 |
| 部分正确 | Tiling 边界问题 | AlignUp、Tile 边界处理 |
| NaN/Inf | FP16 溢出 | AtomicAdd 累积次数 |
| 接近但不精确 | FP16 精度限制 | 放宽 atol/rtol |
| 第二次运行异常 | 信号残留 | 每次运行前清零信号矩阵 |
| 读到陈旧数据 | 缓存一致性 | `dcci` + 编译器屏障 |

---

## 逐步排查法

```
1. CPU 仿真验证数据逻辑 → 通过
2. 单 rank 单 block 验证 → 通过
3. 单 rank 多 block 验证 → 排查 intra-rank 同步
4. 2 rank 验证 → 排查跨 rank 同步
5. 8 rank 验证 → 排查规模化问题
```

---

## 精度标准速查

| 数据类型 | 推荐 atol | 推荐 rtol | 说明 |
|---------|----------|----------|------|
| float (FP32) | 1e-5 | 1e-4 | 高精度 |
| half (FP16) | 1.0 | 0.01 | AtomicAdd 累积误差较大 |
| int32 / int16 | 0 | 0 | 精确匹配 |

---

## 测试检查清单

### 功能测试

- [ ] CPU 仿真通过（验证数据逻辑）
- [ ] 单 rank / 单 block 通过
- [ ] 单 rank / 多 block 通过（intra-rank 同步）
- [ ] 2 rank 通过（基本跨 rank 通信）
- [ ] 8 rank（或目标规模）通过
- [ ] 信号矩阵每次运行前清零
- [ ] FP16 精度在 atol/rtol 阈值内

### 边界测试

- [ ] 数据维度非对齐情况（需要 padding）
- [ ] 单个 Tile 即可容纳的小数据量
- [ ] 超大数据量（自动分块）
- [ ] nranks = 2（最小多 rank）
- [ ] nranks = 8（典型规模）

### 鲁棒性测试

- [ ] 连续多次运行均通过（排除信号残留）
- [ ] 不同数据模式（全零、全一、随机、极端值）
- [ ] 不同 Block 配置（1/4/8/24 blocks）

### 性能测试

- [ ] Warmup 后稳定性能数据
- [ ] Compute-only baseline 对比
- [ ] 流水线重叠效率（pipelined / (compute + comm)）
- [ ] 带宽利用率（实际 / 理论峰值）

---

## 相关 Skills

| Skill | 用途 |
|-------|------|
| `pto-comm-isa-reference` | 指令签名与约束速查 |
| `pto-comm-operator-develop` | 通信算子开发流程 |
| `pto-comm-performance-optimization` | 性能优化方法 |
| `vector-fusion-operator-generate` | 向量融合算子开发和测试 |
