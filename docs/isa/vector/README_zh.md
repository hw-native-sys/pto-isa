# 向量 ISA 参考

`pto.v*` 是 PTO ISA 的向量微指令集。它直接暴露向量流水线、向量寄存器、谓词和向量可见 UB 搬运。

## 组织方式

向量参考按指令族组织，具体 per-op 页面位于 `vector/ops/` 下。

## 指令族

- 向量加载存储
- 谓词与物化
- 一元向量操作
- 二元向量操作
- 向量-标量操作
- 转换操作
- 归约操作
- 比较与选择
- 数据重排
- SFU 与 DSA

## 共享约束

- 向量宽度由元素类型决定
- 谓词宽度必须匹配向量宽度
- 对齐、分布和部分高级形式依赖目标 profile
- 向量层没有 tile 级 valid region 语义

## 相关页面

- [向量指令集](../instruction-surfaces/vector-instructions_zh.md)
- [向量指令族](../instruction-families/vector-families_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)

## 来源与时序披露

当前向量微指令参考页以现有公开 VPTO 语义和时序材料为准，并据此统一生成各 per-op 页的时序披露。

各 per-op 页现在统一遵循下面的时序披露规则：

- 公开来源给出了数字时延或吞吐时，页面直接写出该数字。
- 公开来源只给出流级描述时，页面只写出该更窄的公开契约。
- 公开来源没有给出数字时，页面会明确写成“公开来源未给出”，而不是推测一个常数。
