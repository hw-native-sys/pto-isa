<!-- Generated from `docs/isa/README_zh.md` -->

<p align="center">
  <img src="../../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 手册与参考

本文档目录是 PTO ISA 的权威文档树。它将架构手册、表面指南、家族契约和精确的指令参考分组整合在同一个位置。

## PTO ISA 中的文本汇编

本树是权威的 PTO ISA 手册。文本汇编拼写属于 PTO ISA 语法表面，而非第二份并行的架构手册。

- PTO ISA 定义了架构可见的语义、合法性、状态、排序、目标 profile 边界，以及 `pto.t*`、`pto.v*`、`pto.*` 及其他操作的可见行为
- PTO-AS 是用于编写这些操作和操作数的汇编拼写。它是 PTO ISA 的表达方式的一部分，而非具有不同语义的分立 ISA

如果问题是"PTO 程序在 CPU、A2/A3 和 A5 上的含义是什么？"，请留在本树中。如果问题是"这个操作的操作数形状或文本拼写是什么？"，请使用本树中语法与操作数相关的页面。

## 从这里开始

- [PTO ISA 入口页](../PTO-Virtual-ISA-Manual_zh.md)
- [引言](introduction/what-is-pto-visa.md)
- [文档结构](introduction/document-structure.md)（章节地图与阅读顺序）
- [PTO 的设计目标](introduction/goals-of-pto.md)
- [PTO ISA 版本 1.0](introduction/pto-isa-version-1-0.md)
- [范围与边界](introduction/design-goals-and-boundaries.md)
- [Tile 与有效区域](programming-model/tiles-and-valid-regions.md)
- [执行代理与目标 Profile](machine-model/execution-agents.md)
- [汇编拼写与操作数](syntax-and-operands/assembly-model.md)
- [操作数与属性](syntax-and-operands/operands-and-attributes.md)
- [通用约定](conventions.md)
- [类型系统](state-and-types/type-system.md)
- [位置意图与合法性](state-and-types/location-intent-and-legality.md)
- [一致性基线](memory-model/consistency-baseline.md)

## 模型层次

阅读顺序与手册章节地图一致：先编程模型与机器模型，再语法与状态，再内存，最后是操作码参考。

- [编程模型](programming-model/tiles-and-valid-regions.md)
- [机器模型](machine-model/execution-agents.md)
- [语法与操作数](syntax-and-operands/assembly-model.md)
- [类型系统](state-and-types/type-system.md)
- [位置意图与合法性](state-and-types/location-intent-and-legality.md)
- [内存模型](memory-model/consistency-baseline.md)

## 指令结构

- [指令表面](instruction-surfaces/README.md)
- [指令族](instruction-families/README.md)
- [指令描述格式](reference/format-of-instruction-descriptions.md)
- [Tile 表面参考](tile/README.md)
- [Vector 表面参考](vector/README.md)
- [标量与控制参考](scalar/README.md)
- [其他与通信参考](other/README.md)
- [通用约定](conventions.md)

## 支持性参考

- [参考注释](reference/README.md)（术语表、诊断、可移植性、规范来源）

## 兼容性重定向

`tile/`、`vector/`、`scalar/` 和 `other/` 下的分组表面树是权威的 PTO ISA 路径。

部分旧的根级 tile 页面（如 `TADD_zh.md`、`TLOAD_zh.md`、`TMATMUL_zh.md` 等）现仅作为兼容性重定向保留，以避免现有链接立即失效。新 PTO ISA 文档应链接到分组表面路径，尤其是以下位置的独立 per-op 页面：

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
