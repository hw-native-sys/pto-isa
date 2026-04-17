# pto.vshift

`pto.vshift` 属于[数据重排](../../data-rearrangement_zh.md)指令集。

## 概述

单源位移，腾空出来的位置按零填充。

## 机制

`pto.vshift` 是 `vslide` 的单源版本。它把 `%src` 按 `%amt` 做位移，空出来的位置补零：

```c
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

## 语法

### PTO 汇编形式

```text
vshift %dst, %src, %amt
```

### AS Level 1（SSA）

```mlir
%result = pto.vshift %src, %amt : !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%src` | `!pto.vreg<NxT>` | 源向量 |
| `%amt` | `i16` | 以 lane 为单位的位移量 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 做完位移且空位补零的结果向量 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- `%src` 与 `%result` 必须有相同的元素类型和相同的向量宽度。
- 位移量必须在所选目标 profile 支持的范围内。
- 零填充而不是其他填充值，属于这条形式的语义。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vshift` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vshift` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的寄存器重排路径。

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vshift`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vshift` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[数据重排](../../data-rearrangement_zh.md)
- 上一条指令：[pto.vslide](./vslide_zh.md)
- 下一条指令：[pto.vsqz](./vsqz_zh.md)
