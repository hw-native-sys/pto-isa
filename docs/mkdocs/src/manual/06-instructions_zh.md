# 6. 指令族与契约

## 6.1 范围

本章定义“指令族级”的规范契约。
逐条指令的规范语义仍以 `docs/isa/*_zh.md` 为准。

## 6.2 指令族分类

PTO 指令族分为：

1. 同步与资源绑定
2. Tile-Tile 逐元素运算
3. Tile-标量与 Tile-立即数运算
4. 轴归约与扩展运算
5. 内存操作（`GM <-> Tile` 与索引变体）
6. 矩阵乘与 GEMV 运算
7. 数据搬运与布局变换
8. 不规则/复杂操作

源同步清单由 `docs/isa/manifest.yaml` 维护。

## 6.3 指令族通用契约

每个指令族 MUST 定义：

- 操作数/结果类别与位置规则
- 语义作用域（有效区域处理）
- 必需约束（dtype/layout/location/shape）
- 同步与顺序影响
- 非法使用的诊断行为
- 实现定义边界

## 6.4 有效区域优先规则

除非具体指令另有定义：

- 语义仅在操作有效域内定义
- 域外结果为未指定
- 多输入操作 MUST 定义域组合规则

## 6.5 各指令族摘要

### 6.5.1 同步与资源绑定

包含 `TSYNC`、`TASSIGN` 以及模式/配置类指令。
这类操作定义顺序或状态配置效果，MUST 保持架构顺序语义。

### 6.5.2 逐元素与标量变体

包含算术、位运算、比较、选择、一元数学以及标量融合形式。
操作 MUST 定义逐元素行为与模式相关约束。

### 6.5.3 归约/扩展族

包含按行/按列归约及广播扩展。
操作 MUST 定义轴语义与域兼容关系。

### 6.5.4 内存指令族

包含 load/store/prefetch 与索引 gather/scatter。
操作 MUST 定义 Tile 域与内存域的映射关系。

### 6.5.5 矩阵运算族

包含 `TMATMUL*` 与 `TGEMV*`。
契约 MUST 定义累加域、操作数角色合法性与精度模式交互。

### 6.5.6 搬运/布局族

包含 extract/insert/reshape/transpose/fillpad 等变换。
契约 MUST 定义索引映射与域保持规则。

### 6.5.7 复杂/不规则族

包含 sort/quant/partial/gather 变体及其他特种操作。
契约 MUST 显式标注实现定义部分。

## 6.6 单条指令文档契约

逐条指令页面 SHOULD 按附录 B 模板组织：

- Syntax
- Operands
- Semantics
- Constraints
- Diagnostics
- Implementation-defined behavior
- Compatibility notes

## 6.7 覆盖与同步策略

指令族与指令索引 MUST 与以下来源保持同步：

- `docs/isa/manifest.yaml`
- `include/pto/common/pto_instr.hpp`
- `docs/tools/` 下的索引/矩阵生成工具
