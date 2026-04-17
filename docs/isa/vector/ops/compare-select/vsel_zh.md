# pto.vsel

`pto.vsel` 属于[比较与选择](../../compare-select_zh.md)指令集。

## 概述

`pto.vsel` 根据显式谓词掩码，在两个向量操作数之间逐 lane 选择一个值。

## 机制

对每个 lane `i`：

```text
result[i] = mask[i] ? src0[i] : src1[i]
```

这条指令消费两个形状相同的向量操作数，并按显式谓词决定每个 lane 取真值还是假值分支。

## 语法

### PTO 汇编形式

```text
vsel %dst, %src_true, %src_false, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%src0` | `!pto.vreg<NxT>` | 当掩码位为 1 时被选中的值 |
| `%src1` | `!pto.vreg<NxT>` | 当掩码位为 0 时被选中的值 |
| `%mask` | `!pto.mask` | 逐 lane 选择 `%src0` 与 `%src1` 的谓词 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 逐 lane 选择后的结果向量 |

## 副作用

这条指令除了产生目标向量外，没有其他架构副作用。

## 约束

- `%src0`、`%src1` 与 `%result` 必须有相同的向量宽度 `N` 和相同的元素类型 `T`。
- `%mask` 的宽度必须与 `N` 一致。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vsel` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 如果代码依赖隐式谓词来源或某种目标专用编码变体，这类依赖都应视为 profile 相关。

## 示例

```c
for (int i = 0; i < N; i++)
    result[i] = mask[i] ? src0[i] : src1[i];
```

```mlir
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vsel`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vsel` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[比较与选择](../../compare-select_zh.md)
- 上一条指令：[pto.vcmps](./vcmps_zh.md)
- 下一条指令：[pto.vselr](./vselr_zh.md)
