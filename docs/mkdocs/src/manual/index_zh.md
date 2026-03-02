# PTO 虚拟指令集架构手册

## 0.1 范围

本手册定义 PTO 虚拟指令集架构（Virtual ISA, VISA）的架构级契约。
它规定了前端、IR 流水线、后端与运行时在执行 PTO 程序时 MUST 保持的行为。

逐条指令页面 `docs/isa/*_zh.md` 仍是单条指令语义的权威来源。
本手册定义围绕这些语义的系统级契约。

## 0.2 受众

本手册面向：

- 实现 PTO 降层链路的编译器与 IR 工程师
- 实现目标合法化与代码生成的后端工程师
- 需要验证架构可见行为的内核开发者
- 仿真器与一致性测试开发者

## 0.3 文档约定

本手册借鉴 PTX/Tile-IR 的结构化写法，同时保持 PTO 的架构特性。
章节在适用时采用统一规范节奏：

- 范围
- 语法/形式
- 语义
- 约束
- 诊断
- 兼容

## 0.4 规范性术语

本手册中的 `MUST`、`MUST NOT`、`SHOULD`、`MAY` 为规范性术语。

- `MUST` / `MUST NOT`：强制架构要求。
- `SHOULD`：推荐要求，偏离时需要明确理由。
- `MAY`：架构显式允许的可选行为。

## 0.5 权威来源优先级

文档冲突时按以下顺序对齐：

1. `docs/isa/*_zh.md`：逐条指令语义与约束。
2. `include/pto/common/pto_instr.hpp`：公共 API 形态与重载契约。
3. 本手册：分层模型、架构契约与一致性策略。

## 0.6 推荐阅读顺序

1. [概述](01-overview_zh.md)
2. [执行模型](02-machine-model_zh.md)
3. [状态与类型](03-state-and-types_zh.md)
4. [Tile 与 GlobalTensor](04-tiles-and-globaltensor_zh.md)
5. [同步](05-synchronization_zh.md)
6. [PTO 汇编 (PTO-AS)](06-assembly_zh.md)
7. [指令集（概述）](07-instructions_zh.md)
8. [编程指南](08-programming_zh.md)
9. [虚拟 ISA 与 IR](09-virtual-isa-and-ir_zh.md)
10. [字节码与工具链](10-bytecode-and-toolchain_zh.md)
11. [内存顺序与一致性](11-memory-ordering-and-consistency_zh.md)
12. [后端画像与一致性](12-backend-profiles-and-conformance_zh.md)
13. [附录 A：术语表](appendix-a-glossary_zh.md)
14. [附录 B：指令契约模板](appendix-b-instruction-contract-template_zh.md)
15. [附录 C：诊断分类](appendix-c-diagnostics-taxonomy_zh.md)
16. [附录 D：指令族矩阵](appendix-d-instruction-family-matrix_zh.md)
