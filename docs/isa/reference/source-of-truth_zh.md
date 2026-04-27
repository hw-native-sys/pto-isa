# 规范来源

重写或验证 PTO ISA 文档时，按下面的顺序判定事实来源：

1. `include/pto/common/pto_instr.hpp`
   公共 C++ 内建接口与可见 API 形态
2. 当前仓库中的 PTO ISA 文档
   架构可见语义与约束说明
3. PTO-AS 语法页面
   语法、汇编拼写与汇编层形式
4. 目标 profile 说明与实现说明
   用于判定 backend 专属缩窄

如果 prose 与代码可见的 PTO 指令集冲突，不应把未支持行为写成架构保证。

## 权威顺序

当规范边界不清楚时，按以下顺序裁决：

1. **PTO ISA manual 与 per-op 页**
   架构可见语义
2. **代码**
   合法指令集与接口形态
3. **PTO-AS 语法页面**
   语法和文本拼写
4. **目标 profile 说明**
   backend 专属缩窄

## 两条编译路径

PTO 程序进入工具链有两条路径，两条路径共享同一套 PTO ISA 语义：

```text
PTO program (.pto text)
        |
        +--> ptoas --> C++ --> bisheng --> binary
        |
        +--> ptoas ---------------------> binary
```

`ptoas` 是权威汇编器。文档讨论 “PTO 做什么” 时，指的是 PTO ISA 定义的语义，而不是某一条具体产物路径。

## 对文档维护的含义

- 如果 manual 说某个行为合法，而代码拒绝它，应把这视为缺陷并继续核对。
- 如果 manual 沉默而代码接受该行为，代码当前是事实来源，文档应补上。
- 如果 manual 与代码冲突，先以代码接口为准，再修正文档。
- 如果 manual 沉默而代码拒绝该行为，它通常属于 backend-specific 或尚未进入 PTO 契约。

## PTOAS 的角色

`ptoas` 定义：

- PTO-AS 语法与文法
- 解析与验证规则
- 从 PTO-AS 到 C++ 或二进制的 lowering 语义

当 PTO ISA manual 记录 SSA 或 DPS 形式时，应与 `ptoas` 实际接受的形式保持一致。

## 相关页面

- [指令描述格式](./format-of-instruction-descriptions_zh.md)
- [汇编写法与操作数](../syntax-and-operands/assembly-model_zh.md)
