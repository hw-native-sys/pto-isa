<!-- Generated from `docs/isa/instruction-surfaces/README_zh.md` -->

# 指令表面

本章描述 PTO ISA 的指令表面（Instruction Surface）——按功能角色组织的指令分类。表面是对应不同执行路径的操作集合。

## 本章内容

- [指令表面总览](instruction-surfaces/README.md) — 四层表面的整体说明、数据流图、操作数对照表
- [Tile 指令表面](instruction-surfaces/tile-instructions.md) — `pto.t*` 逐 tile 操作表面
- [Vector 指令表面](instruction-surfaces/vector-instructions.md) — `pto.v*` 向量微指令表面
- [标量与控制指令表面](instruction-surfaces/scalar-and-control-instructions.md) — 标量、控制和配置操作表面
- [其他指令表面](instruction-surfaces/other-instructions.md) — 通信、调试和其他支持操作

## 阅读建议

建议按以下顺序阅读：

1. 先读 [指令表面总览](instruction-surfaces/README.md)，理解 Tile / Vector / Scalar&Control / Other 四层表面的整体结构和数据流关系
2. 再根据需要深入具体表面页，了解该表面的操作数类型、约束和规范语言

## 章节定位

本章属于手册第 7 章（指令集）的一部分。表面是介于编程模型与具体指令之间的中层抽象，帮助读者按功能角色定位指令。
