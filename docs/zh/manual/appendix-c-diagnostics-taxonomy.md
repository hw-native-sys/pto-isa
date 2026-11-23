# 附录 C. 诊断分类体系

## C.1 范围

本附录定义 PTO 虚拟 ISA 工具链的诊断分类与稳定性要求。

## C.2 诊断质量契约

所有诊断 SHOULD 满足：

- 确定性错误类别
- 确定性主消息形态
- 可执行上下文（期望 vs 实际）
- 在可用时提供源码定位

## C.3 主诊断类别

### C.3.1 解析诊断（`PARSE_*`）

用于 PTO-AS 文本错误：

- token 形态错误
- 文法违规
- 字面量/属性语法非法

### C.3.2 结构诊断（`STRUCT_*`）

用于 IR 结构违规：

- 操作数/结果元数错误
- 必需属性缺失
- 类型类别不兼容

### C.3.3 合法性诊断（`LEGAL_*`）

用于后端/画像合法性失败：

- 不支持的 dtype/layout/location/shape 组合
- 不支持的模式组合
- 选定画像中不支持的指令变体

### C.3.4 顺序诊断（`ORDER_*`）

用于同步/顺序错误：

- 缺失必需依赖边
- 非法同步形式
- 顺序契约违反

### C.3.5 字节码诊断（`BCODE_*`）

用于交换/序列化失败：

- 不支持的字节码版本
- section/record 畸形
- 未知必需字段/操作码

## C.4 推荐消息字段

诊断 SHOULD 包含：

- 错误类别（稳定标识）
- 操作名与操作数位置（如适用）
- 期望契约摘要
- 实际违规值/形状/类型/模式
- 定位信息或来源上下文

## C.5 稳定性策略

- 错误类别标识在补丁版本内 MUST 稳定。
- 消息文案在 CI 快照中 SHOULD 尽量稳定。
- 若文案发生实质变化，发布说明 SHOULD 记录该变化。

## C.6 示例格式

```text
LEGAL_UNSUPPORTED_TUPLE: tmatmul operand src1 has unsupported tuple
  expected: layout in {fractal_a, fractal_b}, dtype in {fp16, bf16}
  actual: layout=row_major, dtype=int8
  context: backend_profile=A3, op_loc=line 42
```
