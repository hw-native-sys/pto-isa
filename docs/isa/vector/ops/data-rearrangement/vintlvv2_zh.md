# pto.vintlvv2

`pto.vintlvv2` 属于[数据重排](../../data-rearrangement_zh.md)指令集。

## 概述

`vintlv` 的变体，只返回交织结果的一半。

## 机制

`pto.vintlvv2` 保留了和 `vintlv` 相同的交织语义，但只把结果对中的一半通过 SSA 形式暴露出来。哪一半被返回，由 `PART` 选择器决定。

## 语法

### PTO 汇编形式

```text
vintlvv2 %dst, %lhs, %rhs, "PART"
```

### AS Level 1（SSA）

```mlir
%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 第一个源向量 |
| `%rhs` | `!pto.vreg<NxT>` | 第二个源向量 |
| `PART` | 枚举 | 指定返回交织结果的哪一半 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 选中的那一半交织结果 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- `PART` 决定返回成对交织结果中的哪一半。
- `%lhs`、`%rhs` 与 `%result` 必须有相同的元素类型和相同的向量宽度。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vintlvv2` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vintlvv2` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的寄存器重排路径。

## 示例

```mlir
%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vintlvv2`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vintlvv2` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[数据重排](../../data-rearrangement_zh.md)
- 上一条指令：[pto.vzunpack](./vzunpack_zh.md)
- 下一条指令：[pto.vdintlvv2](./vdintlvv2_zh.md)
