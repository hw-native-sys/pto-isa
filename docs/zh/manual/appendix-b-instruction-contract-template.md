# 附录 B. 指令契约模板

## B.1 目的

本模板定义 PTO 单条指令文档的标准章节结构。
新增指令页面与现有页面重构应使用此模板。

## B.2 必需章节顺序

1. `# <INSTR>`
2. `## Scope`
3. `## Syntax`
4. `## Operands`
5. `## Semantics`
6. `## Constraints`
7. `## Diagnostics`
8. `## Implementation-defined behavior`
9. `## Compatibility`
10. `## Examples`

## B.3 各章节规范要求

### Scope

- MUST 标识指令族与设计意图。
- MUST 说明本页定义的是架构语义还是后端补充约束。

### Syntax

- MUST 给出 PTO-AS 形式与公共 API 签名参考。
- SHOULD 先给出一条规范形态，再给可选变体。

### Operands

每个操作数/结果 MUST 定义：

- 角色（`dst`、`src0`、`src1` 等）
- 类型类别
- 域/形状预期
- 位置/布局要求（如存在）

### Semantics

- MUST 定义有效域迭代模型。
- MUST 定义域内输出语义。
- MUST 明确域外行为（已定义或未指定）。

### Constraints

- MUST 列出合法性维度（`dtype`、`layout`、`location`、`shape`、模式属性）。
- MUST 区分架构层要求与后端画像限制。

### Diagnostics

- MUST 定义确定性拒绝条件。
- SHOULD 提供常见失败类别的期望/实际示例。

### Implementation-defined behavior

- MUST 枚举全部实现定义点。
- MUST 指向后端特定细节文档位置。

### Compatibility

- 发生行为变更时 MUST 提供版本/迁移说明。
- SHOULD 标注增量变更或破坏性变更类别。

## B.4 模板正文（可复制）

```markdown
# <INSTR>

## Scope

## Syntax

## Operands

## Semantics

## Constraints

## Diagnostics

## Implementation-defined behavior

## Compatibility

## Examples
```
