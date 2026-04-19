# docs/coding/

本目录从 **C++ 编程视角**描述 PTO Tile Library 的编程模型（Tile、GlobalTensor、事件、标量参数），并提供扩展库的指导。

如果你在寻找 **ISA 参考**，请从 [docs/isa/README_zh.md](../isa/README_zh.md) 开始。

## 按任务选择

| 你的目标 | 从这里开始 |
|----------|----------|
| 学习 PTO 编程（第一次） | [上手教程](tutorial_zh.md) |
| 理解 Tile 抽象与有效区域 | [Tile 编程模型](Tile_zh.md) |
| 理解全局内存张量 | [GlobalTensor](GlobalTensor_zh.md) |
| 理解事件与同步 | [事件与同步](Event_zh.md) |
| 理解 Auto Mode | [Auto Mode 概述](../auto_mode/Auto_Mode_Overview_zh.md) |
| 理解编译流程 | [编译流程](compilation-process_zh.md) |
| 定位性能瓶颈 | [性能优化](opt_zh.md) |
| 理解张量融合 | [算子融合](operator-fusion_zh.md) |
| 调试与错误处理 | [调试指南](debug_zh.md)、[错误码](error-codes_zh.md) |
| 多核编程 | [多核编程](multi-core-programming_zh.md) |
| 内存优化 | [内存优化](memory-optimization_zh.md) |
| PTO 与其他框架对比 | [PTO 对比](pto-comparison_zh.md) |

## 文档索引

### 基础

- [上手教程（编写第一个 kernel）](tutorial_zh.md) — 一步一步写出你的第一个 PTO kernel
- [更多教程示例](tutorials/README_zh.md) — 更多上手示例
- [Tile 编程模型](Tile_zh.md) — Tile 抽象与布局/有效区域规则
- [GlobalTensor](GlobalTensor_zh.md) — 全局内存张量（shape/stride/layout）
- [事件与同步](Event_zh.md) — 事件与同步模型
- [标量值、类型与枚举](Scalar_zh.md) — 标量参数、类型助记符与枚举
- [Auto Mode 概述](../auto_mode/Auto_Mode_Overview_zh.md) — 编译器自动管理资源与同步

### 编译与构建

- [编译流程](compilation-process_zh.md) — PTO 程序从源码到二进制的完整流程
- [CPU Simulator](cpu_sim.md) — 如何在 CPU 上运行 PTO 代码

### 调试与错误处理

- [调试指南](debug_zh.md) — 调试与断言查找
- [错误码](error-codes_zh.md) — 错误码说明

### 高级话题

- [性能优化](opt_zh.md) — 性能分析与调优建议
- [性能最佳实践](performance-best-practices_zh.md) — 最佳实践与性能要点
- [算子融合](operator-fusion_zh.md) — 张量融合技术
- [内存优化](memory-optimization_zh.md) — 内存优化策略
- [流水线并行](pipeline-parallel_zh.md) — 流水线并行编程
- [多核编程](multi-core-programming_zh.md) — 多核编程模型
- [版本兼容性](version-compatibility_zh.md) — 版本兼容性与迁移
- [框架集成](framework-integration_zh.md) — 与 PyTorch 等框架集成

### 参考

- [PTO 对比其他框架](pto-comparison_zh.md) — PTO 与 TVM、CUTLASS 等的对比
- [参考资料](references_zh.md) — 参考资料汇总
- [ConvTile](ConvTile.md) — Conv2D tile 优化

## 相关文档

| 文档 | 内容 |
|------|------|
| [PTO 抽象机器模型](../machine/README_zh.md) | 抽象执行模型 |
| [docs/README_zh.md](../README_zh.md) | 文档总入口 |
| [docs/isa/README_zh.md](../isa/README_zh.md) | ISA 指令参考 |
