<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO 虚拟 ISA 手册

PTO 虚拟 ISA 手册的稳定入口位于 `docs/isa/` 这棵合并文档树。该文档树把 PTO 组织为一个多目标虚拟 ISA，并清楚区分编程模型、机器模型、内存模型、指令集和家族契约。

右上角的语言图标用于在英文和中文版本之间切换。有对应页面时会直接跳转；没有对应页面时会回到当前语言的着陆页。

## 从这里开始

- [PTO ISA 是什么](isa/introduction/what-is-pto-visa_zh.md)
- [文档结构](isa/introduction/document-structure_zh.md)
- [PTO 的设计目标](isa/introduction/goals-of-pto_zh.md)
- [当前 PTO ISA 范围](isa/introduction/current-isa-scope_zh.md)
- [范围与边界](isa/introduction/design-goals-and-boundaries_zh.md)
- [编程模型](isa/programming-model/tiles-and-valid-regions_zh.md)
- [机器模型](isa/machine-model/execution-agents_zh.md)
- [语法与操作数](isa/syntax-and-operands/assembly-model_zh.md)
- [通用约定](isa/conventions_zh.md)
- [类型系统](isa/state-and-types/type-system_zh.md)
- [位置意图与合法性](isa/state-and-types/location-intent-and-legality_zh.md)
- [内存模型](isa/memory-model/consistency-baseline_zh.md)

## 手册结构

- [指令集总览](isa/instruction-families/README_zh.md)
- [指令族](isa/instruction-families/README_zh.md)
- [指令描述格式](isa/reference/format-of-instruction-descriptions_zh.md)
- [参考注释](isa/reference/README_zh.md)

## 指令集参考

- [Tile ISA 参考](isa/tile/README_zh.md)
- [Vector ISA 参考](isa/vector/README_zh.md)
- [标量与控制参考](isa/scalar/README_zh.md)
- [通信指令集参考](isa/comm/README_zh.md)
- [系统调度指令集参考](isa/system/README_zh.md)
- [语法与操作数](isa/syntax-and-operands/assembly-model_zh.md)
- [通用约定](isa/conventions_zh.md)

## PTO ISA 一览

PTO 是一个跨越多目标的虚拟 ISA，涵盖 CPU 仿真、A2/A3 类目标和 A5 类目标。PTO 的可见 ISA 并不是一个扁平的指令池：

- `pto.t*` 覆盖以 tile 为导向的计算与数据移动
- `pto.v*` 覆盖向量微指令行为及其 buffer/register/predicate 模型
- `pto.*` 覆盖标量、控制、配置和共享支持操作
- 通信指令覆盖跨 rank 行为；系统调度指令覆盖 TPipe/TMPipe 协议和资源生命周期

手册阐明了 PTO 自身保证的内容与仅作为目标 profile 限制的内容之间的区别。

## 权威入口

合并后的手册索引位于 [PTO ISA 手册与参考](isa/scalar/ops/micro-instruction/README_zh.md)。
