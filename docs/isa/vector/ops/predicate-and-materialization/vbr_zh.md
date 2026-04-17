# pto.vbr

`pto.vbr` 属于[谓词与物化](../../predicate-and-materialization_zh.md)指令集。

## 概述

把一个标量广播到整个向量的所有 lane。

## 机制

`pto.vbr` 会把一个标量值物化成向量寄存器。它虽然输入是标量，但结果是完整向量，因此仍然属于 `pto.v*` 指令面。

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

## 语法

### PTO 汇编形式

```text
vbr %result, %value
```

### AS Level 1（SSA）

```mlir
%result = pto.vbr %value : T -> !pto.vreg<NxT>
```

## 输入

- `%value`：标量源

## 预期输出

- `%result`：每个活跃 lane 都携带 `%value` 的结果向量

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 当前支持的常见形式是 `b8`、`b16`、`b32`。
- 对 `b8` 形式，只消费标量源的低 8 位。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vbr` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vbr` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的物化路径。

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vbr`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vbr` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[谓词与物化](../../predicate-and-materialization_zh.md)
- 下一条指令：[pto.vdup](./vdup_zh.md)
