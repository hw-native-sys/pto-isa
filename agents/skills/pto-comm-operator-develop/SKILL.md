---
name: PTO-COMM 通信算子开发指南
description: 基于 PTO-COMM ISA 开发通信算子的完整指南。涵盖 Host-Device 架构、文件结构、通信模式（P2P/集合通信/通算融合）、同步策略、信号矩阵设计、多 Block 调度、远端地址管理、构建系统配置等。触发：需要使用 PTO-COMM 开发通信算子、设计通信 kernel、编写 Host 侧代码、配置 CMakeLists 时。
license: CANN Open Software License Agreement Version 2.0
---

# PTO-COMM 通信算子开发指南

## 定位

本 Skill 是**流程型 Skill**，指导从零开发一个基于 PTO-COMM ISA 的通信算子。

---

## 架构概述

### Host-Device 分离

```
Host 侧                          Device 侧
┌─────────────────┐              ┌─────────────────────────┐
│ main.cpp        │              │ comm_kernel.cpp         │
│ - MPI 初始化    │   启动       │ - __global__ AICORE     │
│ - ACL 初始化    │──kernel──→   │ - TPUT/TGET/TNOTIFY/... │
│ - HCCL 通信域   │              │ - 信号同步逻辑          │
│ - 内存分配      │              ├─────────────────────────┤
│ - Kernel 启动   │              │ compute_kernel.cpp      │
│ - 结果验证      │   启动       │ - __global__ AICORE     │
│                 │──kernel──→   │ - TMATMUL/TADD/...      │
└─────────────────┘              │ - 计算逻辑              │
                                 └─────────────────────────┘
```

**关键原则**：
- **Host 侧**：负责 MPI/HCCL 通信域初始化、内存分配、远端地址获取、kernel 启动和结果验证
- **Device 侧**：使用 PTO-COMM 指令执行实际的数据传输和同步
- 计算和通信可以分别编译为独立的 `.so` 文件

---

## 编程模型选择

```
需要 NPU 间通信？
├── 仅需基本 P2P 传输
│   └── 使用 TPUT/TGET → 参考 "开发模式" 之 P2P 模式
│
├── 需要集合通信（AllReduce/AllGather/ReduceScatter 等）
│   ├── 可用内置集合指令完成？
│   │   └── 使用 TGATHER/TSCATTER/TBROADCAST/TREDUCE → 参考 "开发模式" 之集合通信模式
│   └── 需要自定义算法（如 RS+AG 组合 AllReduce）？
│       └── 使用 TPUT<AtomicAdd> + TNOTIFY/TWAIT 组合 → 参考 "开发模式" 之自定义集合通信
│
├── 需要通算融合（计算+通信重叠）
│   └── 使用双 kernel（cube + vec）+ 队列/信号同步 → 参考 "开发模式" 之通算融合模式
│
└── 需要异步大块传输
    └── 使用 TPUT_ASYNC/TGET_ASYNC → 参考 pto-comm-isa-reference
```

---

## 文件结构与命名规范

```
kernels/manual/<platform>/<operator_name>/
├── comm_kernel.cpp           # 通信 kernel（Vec 架构）
├── compute_kernel.cpp        # 计算 kernel（Cube 架构，如需融合）
├── config.h                  # Tiling 配置、Block 数量、常量定义
├── kernel_launchers.h        # Host 侧 kernel 启动函数声明
├── common.hpp                # 远端地址计算等共享工具
├── main.cpp                  # Host 侧：初始化、启动、验证
├── CMakeLists.txt            # 构建配置
├── run.sh                    # 运行脚本
└── README_zh.md              # 算子文档
```

---

## 核心开发模式

四种开发模式的完整代码示例和同步策略详见：

**详细指南**：[开发模式详解](references/development-patterns.md)

| 模式 | 指令组合 | 适用场景 |
|------|---------|---------|
| P2P | TPUT/TGET | 两 NPU 间数据传输 |
| 集合通信 | TGATHER/TSCATTER/TBROADCAST/TREDUCE | 标准多 rank 操作 |
| 自定义集合通信 | TPUT\<AtomicAdd\> + TNOTIFY/TWAIT | RS+AG 组合实现 AllReduce |
| 通算融合 | 双 kernel + 队列 + 信号矩阵 | 计算与通信重叠 |

---

## 同步策略与信号设计

信号矩阵布局、DeviceBarrier 实现、流水线同步详见：

**详细指南**：[信号与同步设计](references/signal-design.md)

### 快速参考

| 同步需求 | 推荐方式 |
|---------|---------|
| 跨 rank barrier | DeviceBarrier（Intra-rank + Cross-rank + 本地广播） |
| 阶段间分隔 | `pipe_barrier(PIPE_ALL)` |
| 计算→通信通知 | SPSC 就绪队列 + TTEST 轮询 |
| 手动流水线 | `set_flag`/`wait_flag`（仅 TLOAD/TSTORE_IMPL 时需要） |
| 多方通知一方 | `NotifyOp::AtomicAdd` |
| 一方通知多方 | `NotifyOp::Set` |

---

## 远端地址管理与多 Block 调度

远端地址获取方式、地址对齐要求、Block 分配策略、工作均分方法详见：

**详细指南**：[多 Block 调度与地址管理](references/multi-block-scheduling.md)

### 地址对齐要求

- 所有 GM 地址必须满足 32 字节对齐
- Signal 地址必须 4 字节对齐
- TPUT_ASYNC/TGET_ASYNC 的 workspace 由专用 Manager 管理

### 多核切分策略

| 切分维度 | 适用场景 | 方法 |
|---------|---------|------|
| Tile 维度 | 通信量大，Tile 数多 | 均分 Tile 到各 block |
| Row 维度 | 需要精确负载均衡 | 展平为 row-level 分配（推荐） |
| Rank 维度 | 不同 rank 独立传输 | 按 rank 分配给不同 block |

---

## Host 侧与构建系统

Host 侧标准初始化流程、CMakeLists 模板、SOC_VERSION 映射、kernel 启动模式详见：

**详细指南**：[Host 侧与构建系统](references/host-build-system.md)

### SOC_VERSION 与架构映射

| SOC_VERSION | 架构 | Cube Arch | Vec Arch |
|-------------|------|-----------|----------|
| Ascend910B | A2A3 | dav-c220-cube | dav-c220-vec |
| Ascend910C | A2A3 | dav-c220-cube | dav-c220-vec |
| Ascend950 | A5 | dav-c350-cube | dav-c350-vec |

---

## 开发检查清单

### 开发前

- [ ] 确认目标平台（A2A3/A5）和对应的架构编译选项
- [ ] 确认通信拓扑（节点内/跨节点）和链路类型
- [ ] 确定通信模式（P2P/集合/融合）
- [ ] 规划信号矩阵布局

### 实现中

- [ ] TNOTIFY 目标地址为远端，TWAIT/TTEST 监听地址为本地
- [ ] 乒乓 Tile 的 UB 偏移不重叠
- [ ] 使用 `pipe_barrier(PIPE_ALL)` 分隔不同阶段
- [ ] 手动 TLOAD/TSTORE_IMPL 之间有正确的 set_flag/wait_flag
- [ ] 所有 rank 使用相同的 rootIdx 构建 ParallelGroup
- [ ] 非 root rank 不调用集合通信指令
- [ ] 远端地址计算正确（基于通信窗口偏移）

### 测试前

- [ ] 信号矩阵每次运行前清零
- [ ] Host 侧 `aclrtSynchronizeStream` 确保 kernel 执行完成
- [ ] 内存大小与 Tile 配置一致
- [ ] CMakeLists 中 Vec/Cube 架构选择正确

---

## 相关 Skills

| Skill | 用途 |
|-------|------|
| `pto-comm-isa-reference` | PTO-COMM 指令签名、参数、约束速查 |
| `pto-comm-testing-debug` | 通信算子测试与调试指南 |
| `pto-comm-performance-optimization` | 通信算子性能优化 |
| `vector-fusion-operator-generate` | PTO 向量融合算子开发指南 |
