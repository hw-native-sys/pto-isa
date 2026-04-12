<!-- Generated from `docs/isa/syntax-and-operands/README_zh.md` -->

# 语法与操作数

本章描述 PTO ISA 的文本拼写、操作数形状、属性以及共享的命名约定。这是理解指令语法格式的前置章节。

## 本章内容

- [汇编模型](syntax-and-operands/assembly-model.md) — PTO-AS 三层语法（Assembly / SSA / DPS）BNF 定义、操作数修饰符、立即数编码规则
- [操作数与属性](syntax-and-operands/operands-and-attributes.md) — 七类操作数（Tile / GlobalTensor / Scalar / Predicate / Event / UB Pointer / GM Pointer）的 SSA 类型表、属性完整列表（Compare / Rounding / Atomic / Transform / Distribution / Mask）、操作数约束规则

## 阅读建议

建议在深入具体指令之前，先阅读本章以理解语法格式和操作数约定。本章定义的 BNF 语法和操作数分类适用于手册中所有指令页。

## 章节定位

本章属于手册的第 4 章。与第 5 章（状态与类型）和第 6 章（内存模型）一起，构成进入指令集章节之前的"前置知识带"。
