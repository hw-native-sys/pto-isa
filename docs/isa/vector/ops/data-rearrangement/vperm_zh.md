# pto.vperm

`pto.vperm` 属于[数据重排](../../data-rearrangement_zh.md)指令集。

## 概述

根据逐 lane 索引，在寄存器内部完成置换。

## 机制

`pto.vperm` 是纯寄存器内查表：`%index` 的每个 lane 指定 `%src` 中哪个 lane 被取到对应目标位置。它和 `vgather2` 的差别在于，数据源不是 UB 内存，而是另一个向量寄存器。

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

## 语法

### PTO 汇编形式

```text
vperm %dst, %src, %index
```

### AS Level 1（SSA）

```mlir
%result = pto.vperm %src, %index : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%src` | `!pto.vreg<NxT>` | 被置换的源向量 |
| `%index` | `!pto.vreg<NxI>` | 每个 lane 的源索引选择器 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 完成置换后的结果向量 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- 超出支持范围的 `%index` 值，其行为取决于所选形式的 wrap / clamp 规则。
- `%src` 与 `%result` 必须有相同的元素类型和相同的向量宽度。
- 这是寄存器内置换，不访问 UB。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vperm` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vperm` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的寄存器置换路径。

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vperm`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vperm` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[数据重排](../../data-rearrangement_zh.md)
- 上一条指令：[pto.vusqz](./vusqz_zh.md)
- 下一条指令：[pto.vpack](./vpack_zh.md)
