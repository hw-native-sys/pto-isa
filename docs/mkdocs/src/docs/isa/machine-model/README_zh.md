<!-- Generated from `docs/isa/machine-model/README_zh.md` -->

# 机器模型

本章描述 PTO 的执行模型：执行代理（execution agent）、流水线（pipeline）、目标 Profile（A2/A3 vs A5）以及排序与同步的词汇表。

## 本章内容

- [执行代理](machine-model/execution-agents.md) — Host-Device-Core 三层执行架构、各 Profile 差异表、执行单元规格
- [排序与同步](machine-model/ordering-and-synchronization.md) — Tile/Vector/DMA/Communication 四类同步原语、事件模型、流水线依赖图

## 阅读建议

建议按以下顺序阅读：

1. 先读 [执行代理](machine-model/execution-agents.md)，理解 PTO 的三层执行层次（Host / Device / Core）和 Target Profile 差异
2. 再读 [排序与同步](machine-model/ordering-and-synchronization.md)，理解同步原语、事件链和 Producer-Consumer 依赖图

## 章节定位

本章属于手册的第 3 章。理解机器模型是理解 PTO 程序在目标硬件上如何执行的前提。在进入指令集章节之前，应先理解同步和排序规则，因为许多指令的行为直接依赖于机器模型的执行语义。
