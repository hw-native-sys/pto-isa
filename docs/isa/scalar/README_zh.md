# 标量与控制参考

`pto.*` 中的标量与控制部分负责同步、DMA、谓词、控制外壳和共享标量支持逻辑。它们为 tile 与 vector 有效载荷提供执行外壳。

## 组织方式

标量与控制参考按指令族组织，具体 per-op 页面位于 `scalar/ops/` 下。

## 指令族

- 控制与配置
- PTO 微指令参考
- 流水线同步
- DMA 拷贝
- 谓词加载存储
- 谓词生成与代数
- 共享算术
- 共享 SCF

## 共享约束

- pipe / event 空间受目标 profile 约束
- DMA 参数必须自洽
- 谓词宽度和控制参数必须与目标操作匹配
- 顺序边必须与后续 tile / vector 有效载荷对齐

## 相关页面

- [标量与控制指令集](../instruction-surfaces/scalar-and-control-instructions_zh.md)
- [标量与控制指令族](../instruction-families/scalar-and-control-families_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
