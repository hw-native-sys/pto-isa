<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO 虚拟 ISA 手册

PTO 虚拟 ISA 手册的稳定入口位于 `docs/isa/` 这棵合并文档树。该文档树把 PTO 组织为一个多目标虚拟 ISA，并清楚区分编程模型、机器模型、内存模型、指令集和家族契约。

右上角的语言图标用于在英文和中文版本之间切换。有对应页面时会直接跳转；没有对应页面时会回到当前语言的着陆页。

- [前言与阅读顺序](mkdocs/src/manual/index_zh.md)
- [手册总览章节](mkdocs/src/manual/01-overview_zh.md)
- [虚拟 ISA 与 AS 契约](mkdocs/src/manual/08-virtual-isa-and-ir_zh.md)
- [字节码与工具链契约](mkdocs/src/manual/09-bytecode-and-toolchain_zh.md)
- [内存顺序与一致性](mkdocs/src/manual/10-memory-ordering-and-consistency_zh.md)
- [后端画像与一致性](mkdocs/src/manual/11-backend-profiles-and-conformance_zh.md)

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

1. [概述](mkdocs/src/manual/01-overview_zh.md)
2. [执行模型](mkdocs/src/manual/02-machine-model_zh.md)
3. [状态与类型](mkdocs/src/manual/03-state-and-types_zh.md)
4. [Tile 与 GlobalTensor](mkdocs/src/manual/04-tiles-and-globaltensor_zh.md)
5. [同步](mkdocs/src/manual/05-synchronization_zh.md)
6. [指令集（概述）](mkdocs/src/manual/06-instructions_zh.md)
7. [编程指南](mkdocs/src/manual/07-programming_zh.md)
8. [虚拟 ISA 与 AS](mkdocs/src/manual/08-virtual-isa-and-ir_zh.md)
9. [字节码与工具链](mkdocs/src/manual/09-bytecode-and-toolchain_zh.md)
10. [内存顺序与一致性](mkdocs/src/manual/10-memory-ordering-and-consistency_zh.md)
11. [后端画像与一致性](mkdocs/src/manual/11-backend-profiles-and-conformance_zh.md)
12. [附录 A：术语表](mkdocs/src/manual/appendix-a-glossary_zh.md)
13. [附录 B：指令契约模板](mkdocs/src/manual/appendix-b-instruction-contract-template_zh.md)
14. [附录 C：诊断分类](mkdocs/src/manual/appendix-c-diagnostics-taxonomy_zh.md)
15. [附录 D：指令族矩阵](mkdocs/src/manual/appendix-d-instruction-family-matrix_zh.md)
