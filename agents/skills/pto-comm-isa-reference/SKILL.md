---
name: PTO-COMM ISA 指令速查手册
description: PTO 通信指令集完整参考手册。覆盖全部 12 条通信指令（TPUT/TGET/TNOTIFY/TWAIT/TTEST/TGATHER/TSCATTER/TBROADCAST/TREDUCE/TPUT_ASYNC/TGET_ASYNC 及 BuildAsyncSession）的签名、参数、约束、数据流和使用示例。触发：查询 PTO 通信指令用法、参数含义、约束条件、信号类型、ParallelGroup 用法、AsyncSession 构建方式时。
license: CANN Open Software License Agreement Version 2.0
---

# PTO-COMM ISA 指令速查手册

## 定位

本 Skill 是**知识库型**速查手册，提供 PTO-COMM 全部通信指令的快速索引。各指令按类别组织，包含签名、参数说明和关键约束。

---

## 头文件与命名空间

```cpp
#include <pto/comm/pto_comm_inst.hpp>  // 统一公共 API，只需此一个头文件
#include <pto/pto-inst.hpp>            // PTO 核心指令（TLOAD/TSTORE 等）

using namespace pto;
using namespace pto::comm;
```

`pto_comm_inst.hpp` 会根据编译宏自动分发到 NPU 原生实现或 CPU 仿真后端。

---

## 指令分类总览

| 类别 | 指令 | 说明 |
|------|------|------|
| 点对点（同步） | `TPUT`、`TGET` | 通过 UB 暂存 Tile 的远程写/读，支持单缓冲和 ping-pong 双缓冲 |
| 信号同步 | `TNOTIFY`、`TWAIT`、`TTEST` | 基于标志的跨 NPU 同步，信号为 `int32_t` 标量或二维网格 |
| 集合通信 | `TGATHER`、`TSCATTER`、`TBROADCAST`、`TREDUCE` | 基于 `ParallelGroup` 的多 rank 操作，由 root 发起 |
| 异步通信 | `TPUT_ASYNC`、`TGET_ASYNC` | 通过 SDMA/URMA 引擎的 GM→GM DMA 传输，返回 `AsyncEvent` |

### 数据流模型

```
同步点对点（TPUT/TGET）：
  本地 GM → UB Tile（暂存） → 远端 GM

异步点对点（TPUT_ASYNC/TGET_ASYNC）：
  本地 GM → DMA 引擎（SDMA/URMA） → 远端 GM（不经过 UB）

集合通信（TGATHER/TSCATTER/TBROADCAST/TREDUCE）：
  多 rank GM → UB Tile（暂存） → 本地 GM（自动二维分块滑动）
```

---

## 指令选择决策树

```
需要在 NPU 间传输数据？
├── 一对一传输
│   ├── 需要 UB 中间暂存（Tile 级操作） → TPUT / TGET
│   │   ├── 写到远端 → TPUT（支持 AtomicAdd）
│   │   └── 从远端读 → TGET（不支持原子操作）
│   └── 大块 GM→GM 直传（不经 UB） → TPUT_ASYNC / TGET_ASYNC
│       ├── SDMA 引擎（通用） → DmaEngine::SDMA
│       └── URMA 引擎（仅 A5） → DmaEngine::URMA
│
├── 多 rank 操作
│   ├── 收集 → TGATHER
│   ├── 分发 → TSCATTER
│   ├── 广播 → TBROADCAST
│   └── 归约 → TREDUCE
│
└── 仅需同步（无数据传输）
    ├── 阻塞等待 → TWAIT
    ├── 非阻塞检测 → TTEST
    └── 发送通知 → TNOTIFY
```

---

## 核心类型速查

| 类型 | 用途 | 关键约束 |
|------|------|---------|
| `Signal` | 标量同步信号 | `int32_t`，4 字节对齐 |
| `Signal2D<R,C>` | 二维信号网格 | 编译期形状，支持子区域视图 |
| `ParallelGroup<G>` | 集合通信分组 | 外部数组视图，所有 rank 必须传相同 `rootIdx` |
| `NotifyOp` | 通知操作类型 | `AtomicAdd`（原子加）/ `Set`（直接赋值） |
| `WaitCmp` | 比较运算符 | EQ / NE / GT / GE / LT / LE |
| `ReduceOp` | 归约运算符 | Sum / Max / Min |
| `AtomicType` | 原子操作类型 | `AtomicNone`（默认）/ `AtomicAdd` |
| `DmaEngine` | DMA 引擎选择 | `SDMA`（通用）/ `URMA`（仅 A5） |
| `AsyncEvent` | 异步事件句柄 | `Wait` 使用 Quiet 语义（等待所有 pending） |
| `AsyncSession` | 异步会话 | 通过 `BuildAsyncSession` 构建 |

**详细说明**：[核心类型详解](references/core-types.md)

---

## 指令约束速查表

| 指令 | 源地址 | 目标地址 | 需要 UB Tile | 原子操作 | 支持 Ping-Pong | 返回类型 |
|------|--------|---------|-------------|---------|---------------|---------|
| TPUT | 本地 | 远端 | 是 | AtomicNone/AtomicAdd | 是 | RecordEvent |
| TGET | 远端 | 本地 | 是 | 不支持 | 是 | RecordEvent |
| TNOTIFY | — | 远端 | 否 | Set/AtomicAdd | 否 | void |
| TWAIT | 本地 | — | 否 | — | 否 | void |
| TTEST | 本地 | — | 否 | — | 否 | bool |
| TGATHER | 远端(多) | 本地 | 是 | — | 是 | RecordEvent |
| TSCATTER | 本地 | 远端(多) | 是 | — | 是 | RecordEvent |
| TBROADCAST | 本地 | 远端(多) | 是 | — | 是 | RecordEvent |
| TREDUCE | 远端(多) | 本地 | 是(2~3) | — | 是 | RecordEvent |
| TPUT_ASYNC | 本地 | 远端 | 否(需scratch) | — | 否 | AsyncEvent |
| TGET_ASYNC | 远端 | 本地 | 否(需scratch) | — | 否 | AsyncEvent |

---

## 各类指令详解

| 类别 | 详细文档 |
|------|---------|
| TPUT / TGET（同步 P2P） | [P2P 指令详解](references/p2p-instructions.md) |
| TNOTIFY / TWAIT / TTEST（信号同步） | [信号同步指令详解](references/signal-instructions.md) |
| TGATHER / TSCATTER / TBROADCAST / TREDUCE（集合通信） | [集合通信指令详解](references/collective-instructions.md) |
| TPUT_ASYNC / TGET_ASYNC / BuildAsyncSession（异步通信） | [异步通信指令详解](references/async-instructions.md) |

---

## 常见错误速查

| # | 错误 | 规则 |
|---|------|------|
| 1 | TNOTIFY 发到本地 / TWAIT 等远端 | TNOTIFY → 远端，TWAIT/TTEST → 本地 |
| 2 | 非 root 调用集合通信 | 仅 root 执行，非 root 不得调用 |
| 3 | 乒乓 Tile UB 地址重叠 | pingTile 和 pongTile 使用不同 TASSIGN 偏移 |
| 4 | 异步传输使用非一维 tensor | TPUT_ASYNC/TGET_ASYNC 仅支持扁平连续一维 |
| 5 | Signal 类型不是 int32_t | Signal/Signal2D 元素类型必须为 `int32_t` |
| 6 | ParallelGroup tensors 未初始化 | 远端地址必须正确设置 |

---

## 相关 Skills

| Skill | 用途 |
|-------|------|
| `pto-comm-operator-develop` | 基于 PTO-COMM 开发通信算子的完整流程 |
| `pto-comm-testing-debug` | 通信算子测试与调试指南 |
| `pto-comm-performance-optimization` | 通信算子性能优化 |
| `vector-fusion-operator-generate` | PTO 向量融合算子开发指南 |
