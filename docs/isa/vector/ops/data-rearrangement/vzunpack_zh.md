# pto.vzunpack

`pto.vzunpack` 属于[数据重排](../../data-rearrangement_zh.md)指令集。

## 概述

把窄向量的一半解包并做零扩展。

## 机制

`pto.vzunpack` 从源向量里选中一半，再把每个窄元素做零扩展，写成更宽的目标向量。

## 语法

### PTO 汇编形式

```text
vzunpack %dst, %src, %part
```

### AS Level 1（SSA）

```mlir
%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%src` | `!pto.vreg<NxT_narrow>` | 打包后的窄源向量 |
| `%part` | `index` | 指定要解开的哪一半 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<N/2xT_wide>` | 经过零扩展的更宽结果向量 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- 选中的半区及扩宽模式必须是目标 profile 支持的合法形式。
- 它的扩宽行为是零扩展。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vzunpack` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vzunpack` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的扩宽重排路径。

## 示例

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vzunpack`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vzunpack` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[数据重排](../../data-rearrangement_zh.md)
- 上一条指令：[pto.vsunpack](./vsunpack_zh.md)
- 下一条指令：[pto.vintlvv2](./vintlvv2_zh.md)
