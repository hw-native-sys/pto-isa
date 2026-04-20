# PTO AUTO 文档

本目录包含 PTO AUTO 模式的详细文档，帮助开发者理解并使用 Auto Mode 进行 PTO 编程。

## 按任务选择

| 你的需求 | 从这里开始 |
|----------|----------|
| 什么是 Auto Mode | [Auto Mode 概述](Auto_Mode_Overview_zh.md) |
| Kernel 开发规范与限制 | [Kernel Developer Rules](Kernel_Developer_Rules_And_Limitations_zh.md) |
| Library 开发规范与限制 | [Library Developer Rules](Library_Developer_Rules_And_Limitations_zh.md) |
| 代码示例 | [Examples](Examples_zh.md) |

## Auto Mode 是什么

PTO AUTO 是一种新的编程模式，主要提供以下两点优势：

1. **降低开发难度**的同时使开发者实现必要的优化。
2. **确保跨代兼容**不同的昇腾硬件架构。

在 AUTO 模式下，kernel 开发者**无需手动**为 tile 分配内存（`TASSIGN`），也**无需手动**管理不同 pipe 间的同步（`set_flag`/`wait_flag`）。编译器自动完成这些工作，同时保持良好的性能。

## Auto vs Manual 模式对比

| 方面 | Auto Mode | Manual Mode |
|------|-----------|-------------|
| Tile 地址分配 | 编译器自动 | 作者显式 `TASSIGN` |
| 同步管理 | 编译器自动 | 作者显式 `set_flag`/`wait_flag` |
| 数据搬运 | 编译器自动 `TLOAD`/`TSTORE` | 作者显式 `TLOAD`/`TSTORE` |
| 性能 | 接近手工调优 | 最高性能 |
| 开发难度 | 低 | 高 |
| 跨代兼容性 | 最好 | 需要针对不同代际调整 |

> 注意：auto 模式目前仅支持编译器 `-O2` 选项。

## 文档列表

| 文档 | 内容 |
|------|------|
| [Auto Mode 概述](Auto_Mode_Overview_zh.md) | Auto Mode 核心概念、编译器特性、与 Manual 模式对比 |
| [Kernel Developer Rules](Kernel_Developer_Rules_And_Limitations_zh.md) | kernel 开发者在 Auto Mode 下的编程规范与限制 |
| [Library Developer Rules](Library_Developer_Rules_And_Limitations_zh.md) | 库开发者在 Auto Mode 下的编程规范与限制 |
| [Examples](Examples_zh.md) | Auto Mode 代码示例 |

## 相关文档

| 文档 | 内容 |
|------|------|
| [docs/README_zh.md](../README_zh.md) | 文档总入口 |
| [docs/coding/README_zh.md](../coding/README_zh.md) | 编程模型文档入口 |
| [docs/isa/README_zh.md](../isa/README_zh.md) | ISA 指令参考 |
