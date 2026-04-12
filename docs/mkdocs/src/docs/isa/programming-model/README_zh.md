<!-- Generated from `docs/isa/programming-model/README_zh.md` -->

# 编程模型

本章描述程序员在 PTO 中推理和操作的对象：Tile、有效区域（valid region）、GlobalTensor 以及 Auto 模式与 Manual 模式的执行模型。

## 本章内容

- [Tile 与有效区域](programming-model/tiles-and-valid-regions.md) — Tile 的类型、角色、有效区域的概念与约束
- [GlobalTensor 与数据移动](programming-model/globaltensor-and-data-movement.md) — GlobalTensor 视图以及 GM 与 Tile 之间的数据移动
- [Auto 模式 vs Manual 模式](programming-model/auto-vs-manual.md) — 两种执行模式的对比与适用场景

## 阅读建议

建议按以下顺序阅读：

1. 先读 [Tile 与有效区域](programming-model/tiles-and-valid-regions.md)，理解 PTO 的核心抽象
2. 再读 [GlobalTensor 与数据移动](programming-model/globaltensor-and-data-movement.md)，理解数据如何流入和流出 Tile
3. 最后读 [Auto 模式 vs Manual 模式](programming-model/auto-vs-manual.md)，选择合适的执行模式

## 章节定位

本章属于手册的第 2 章。在进入指令细节之前，应先理解编程模型，因为 PTO 的所有指令都围绕 Tile 和有效区域展开。
