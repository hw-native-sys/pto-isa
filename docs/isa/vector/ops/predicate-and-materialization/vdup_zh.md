# pto.vdup

`pto.vdup` 属于[谓词与物化](../../predicate-and-materialization_zh.md)指令集。

## 概述

把一个标量，或者一个选中的源 lane，复制到整个目标向量。

## 机制

`pto.vdup` 会把 `%input` 中选中的一个值复制到所有 lane。它既可以复制标量，也可以从源向量里抽出某个 lane 再广播到整向量。

## 语法

### PTO 汇编形式

```text
vdup %result, %input {position = "POSITION"}
```

### AS Level 1（SSA）

```mlir
%result = pto.vdup %input {position = "POSITION"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 输入

- `%input`：提供被复制值的标量或向量源
- `position`：指定复制哪个源元素或标量位置的选择器

## 预期输出

- `%result`：复制后的结果向量

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- `position` 决定复制哪个源元素。
- 当前 PTO 向量表示把 `position` 建模为属性，而不是单独的 SSA 操作数。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vdup` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

当前仓内没有为 `vdup` 单列周期表。若代码对具体延迟敏感，应把它视为目标 profile 相关的物化路径。

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vdup`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vdup` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[谓词与物化](../../predicate-and-materialization_zh.md)
- 上一条指令：[pto.vbr](./vbr_zh.md)
