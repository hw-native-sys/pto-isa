<!-- Generated from `docs/isa/state-and-types/README_zh.md` -->

# 状态、类型与位置

本章描述 PTO 的类型系统和位置意图：数据类型、Ttile 角色、location intent 以及操作的合法性约束。

## 本章内容

- [类型系统](state-and-types/type-system.md) — 完整数据类型表（FP8/F16/BF16/F32 + 整型）、Vector Width 表、NaN/Inf 行为、类型转换规则
- [位置意图与合法性](state-and-types/location-intent-and-legality.md) — Location Intent 分类（Vec / Mat / Acc / Left / Right / Scalar）、四阶段合法性检查流程（Type Check → Shape Check → Layout Check → Target Profile Check）

## 阅读建议

建议按以下顺序阅读：

1. 先读 [类型系统](state-and-types/type-system.md)，理解 PTO 支持的元素类型和向量宽度
2. 再读 [位置意图与合法性](state-and-types/location-intent-and-legality.md)，理解 Tile Type、Layout 和 Target Profile 对操作合法性的影响

## 章节定位

本章属于手册的第 5 章。在进入指令集章节之前，应理解类型系统，因为每条指令的操作数都有严格的类型约束。Location Intent 和合法性检查流程是理解 Tile Type 组合（Left/Right/Acc/Mat/Vec）限制的关键。
