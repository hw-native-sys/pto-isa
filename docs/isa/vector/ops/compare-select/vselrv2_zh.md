# pto.vselrv2

`pto.vselrv2` 属于[比较与选择](../../compare-select_zh.md)指令集。

## 概述

`pto.vselrv2` 是 `vselr` 的变体形式。它保持与 `vselr` 相同的可见结果，但允许后端使用不同的目标专用编码。

## 机制

这条指令的可见契约仍然是 reverse-select：

```text
result[i] = pred[i] ? src1[i] : src0[i]
```

`pto.vselrv2` 存在的意义，是让后端在不改变 PTO 可见逐 lane 结果的前提下，保留一个独立的目标编码变体。

## 语法

### PTO 汇编形式

```text
vselrv2 %dst, %src0, %src1 : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%src0` | `!pto.vreg<NxT>` | 当控制谓词 lane 为假时使用的默认值 |
| `%src1` | `!pto.vreg<NxT>` | 当控制谓词 lane 为真时使用的值 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 采用 reverse-select 极性的结果向量 |

## 副作用

这条指令除了产生目标向量外，没有其他架构副作用。

## 约束

- `%src0`、`%src1` 与 `%result` 必须有相同的向量宽度 `N` 和相同的元素类型 `T`。
- lowering 必须精确保留所选 `vselrv2` 形式要求的目标专用编码与谓词来源。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vselrv2` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 如果代码依赖隐式谓词来源或某种目标专用编码变体，这类依赖都应视为 profile 相关。

## 示例

```c
for (int i = 0; i < N; i++)
    result[i] = pred[i] ? src1[i] : src0[i];
```

```mlir
%result = pto.vselrv2 %fallback, %preferred : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vselrv2`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vselrv2` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[比较与选择](../../compare-select_zh.md)
- 上一条指令：[pto.vselr](./vselr_zh.md)
