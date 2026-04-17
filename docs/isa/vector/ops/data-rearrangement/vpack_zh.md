# pto.vpack

`pto.vpack` 属于[数据重排](../../data-rearrangement_zh.md)指令集。

## 概述

把两个宽向量收窄后拼成一个窄向量。

## 机制

`pto.vpack` 会先把两个宽向量按所选 `part` 子模式做收窄，再把两半结果拼成一个目标窄向量。它不是简单拼接，而是“收窄 + 组合”的一体化操作。

## 语法

### PTO 汇编形式

```text
vpack %dst, %src0, %src1, %part
```

### AS Level 1（SSA）

```mlir
%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%src0` | `!pto.vreg<NxT_wide>` | 第一个宽源向量 |
| `%src1` | `!pto.vreg<NxT_wide>` | 第二个宽源向量 |
| `%part` | `index` | 打包子模式选择器 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<2NxT_narrow>` | 由两个源向量收窄后组合而成的目标窄向量 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- 打包本质上是收窄转换；超出目标位宽的值按所选模式截断或饱和。
- lowering 必须保留第一个源半区与第二个源半区在目标里的先后次序。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vpack` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vpack` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的收窄重排路径。

## 示例

```c
for (int i = 0; i < N; i++) {
    dst[i] = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vpack`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vpack` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[数据重排](../../data-rearrangement_zh.md)
- 上一条指令：[pto.vperm](./vperm_zh.md)
- 下一条指令：[pto.vsunpack](./vsunpack_zh.md)
