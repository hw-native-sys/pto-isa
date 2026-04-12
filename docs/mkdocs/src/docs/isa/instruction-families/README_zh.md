<!-- Generated from `docs/isa/instruction-families/README_zh.md` -->

# 指令族

本章描述 PTO ISA 的指令族（Instruction Family）——共享约束和行为的指令分组。每个族定义了该族所有指令共同遵循的规则。

## 本章内容

- [指令族总览](instruction-families/README.md) — 完整导航地图和族规范模板
- [Tile 指令族](instruction-families/tile-families.md) — Tile 表面的 8 个指令族（逐元素、归约、布局等）
- [Vector 指令族](instruction-families/vector-families.md) — Vector 表面的 9 个指令族
- [标量与控制指令族](instruction-families/scalar-and-control-families.md) — 标量、控制和配置的 6 个指令族
- [其他指令族](instruction-families/other-families.md) — 通信和其他支持指令族

## 族与表面的关系

- **表面（Surface）** 按功能角色分类指令（Tile / Vector / Scalar&Control / Other）
- **族（Family）** 共享约束、行为模式和规范语言；同一族的指令共享家族概览页中的共同约束

## 每个族必须定义的内容

1. **Mechanism** — 族的用途说明
2. **Shared Operand Model** — 共同的操作数模型和交互方式
3. **Common Side Effects** — 所有族内操作共享的副作用
4. **Shared Constraints** — 适用于全族的合法性规则
5. **Cases That Are Not Allowed** — 全族禁止的条件
6. **Target-Profile Narrowing** — A2/A3 和 A5 的差异
7. **Operation List** — 指向各 per-op 页面的链接

## 章节定位

本章属于手册第 7 章（指令集）的一部分。族文档是 per-op 页面的上一层抽象，同一族的指令共享家族概览页中的共同约束。
