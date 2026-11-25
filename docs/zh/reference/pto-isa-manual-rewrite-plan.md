# PTO ISA Manual 重写计划

这份文档是 PTO manual 升级的执行 backlog，目标是把它从模板化契约汇编，改造成一份可读、可教、同时对实现者仍然足够严格的架构手册。

## 重写目标

- 让 manual 解释 PTO 为什么是 tile-first，而不只是定义它是 tile-first
- 只在 verifier、backend 或测试能检查的地方使用规范性术语
- 每个重写章节至少给出一个 worked example
- 明确区分 architecture-defined 和 implementation-defined 行为
- 保持中英文章节结构、例子和术语同步

## Wave 计划

### Wave 1

规范来源文件：

- `docs/isa/README.md`
- `docs/isa/introduction/*.md`
- `docs/isa/machine-model/*.md`
- `docs/isa/programming-model/*.md`
- 对应中文镜像

目标：

- 把旧的前言模板改成真正的 introduction
- 解释 PTO 为什么存在，以及 Auto 和 Manual 为什么都重要
- 引入一个可被后续章节复用的 running example
- 把编程章节从 checklist 改成真正的 guide

### Wave 2

规范来源文件：

- `docs/isa/state-and-types/*.md`
- `docs/isa/programming-model/globaltensor-and-data-movement.md`
- `docs/isa/memory-model/*.md`
- 对应中文镜像

目标：

- 解释 PTO 里合法性在实践中到底如何判断
- 把 valid-region 行为讲具体
- 把同步规则和真实 producer/consumer 模式连接起来

### Wave 3

规范来源文件：

- `docs/isa/syntax-and-operands/*.md`
- `docs/isa/instruction-families/*.md`
- `docs/isa/tile/**`、`docs/isa/vector/**`、`docs/isa/scalar/**`、`docs/isa/comm/**`、`docs/isa/system/**`
- `docs/isa/reference/*.md`
- 对应中文镜像

目标：

- 在保持工具链精度的同时，把章节开头改成 explanation-first
- 降低 backend profile 和 conformance 章节的机械感
- 保持 verifier 和 interchange 契约的严格度，但不再写得像自动生成

## 章节清单

每个重写章节都必须包含：

- 一段说明“本章回答什么问题”
- 一段解释“为什么 PTO 这样设计，而不是更简单那样设计”
- 一个具体例子，优先来自 `demos/`、`tests/` 或 `docs/coding/tutorials/`
- 一个明确写出真实规范契约的部分
- 一个点名常见误区或非可移植假设的部分

## 应删除或下沉的内容

重写时要删除或下沉：

- 只有 “scope / audience / conventions” 但没有新增信息的段落
- 只是为了镜像模板而存在的编号小节
- 用 bullet 代替解释的段落
- 实际上只是写作建议或价值判断的 `MUST/SHOULD/MAY`

## 可复用的素材来源

优先复用的仓库内材料：

- `docs/coding/tutorials/vec-add.md`
- `docs/machine/abstract-machine.md`
- `include/pto/common/pto_instr.hpp`
- `tests/cpu/st/testcase/` 下有代表性的 CPU ST 用例

可借鉴的外部风格参考：

- Lua 5.4 参考手册
- SQLite 架构文档
- 更严格的 ISA 规范里关于“可测试规范语言”的写法
