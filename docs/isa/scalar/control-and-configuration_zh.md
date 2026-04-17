# 控制与配置

`pto.*` 中的控制与配置类操作建立执行外壳：顺序、DMA 配置、谓词相关可见状态以及围绕 tile / vector 有效载荷的控制结构。

## 概要

标量与控制操作本身不承载 tile 有效载荷。它们把 `pto.t*` 和 `pto.v*` 真正执行所需的执行环境显式建出来。

## 主要子类

- 历史上保留 `t` 前缀、但在手册中归入控制/配置路径的模式寄存器操作：
  [pto.tsethf32mode](./ops/control-and-configuration/tsethf32mode_zh.md)、[pto.tsetfmatrix](./ops/control-and-configuration/tsetfmatrix_zh.md)
- [流水线同步](./pipeline-sync_zh.md)
- [DMA 拷贝](./dma-copy_zh.md)
- [谓词加载存储](./predicate-load-store_zh.md)
- [谓词生成与代数](./predicate-generation-and-algebra_zh.md)

## 架构角色

这类操作在 PTO 中暴露：

- 控制状态
- 谓词状态
- DMA 配置
- 顺序边和同步状态

它们仍然是虚拟 ISA 契约的一部分，只是输出的不是 tile / vector payload。

手册也把少量保留 `pto.t*` 历史命名的配置操作放在这里，因为它们的架构角色属于控制/配置而不是 tile payload 变换。

## 相关页面

- [标量与控制指令集](../instruction-surfaces/scalar-and-control-instructions_zh.md)
- [标量与控制指令族](../instruction-families/scalar-and-control-families_zh.md)
