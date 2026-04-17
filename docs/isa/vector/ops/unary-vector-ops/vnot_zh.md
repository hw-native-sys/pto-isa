# pto.vnot

`pto.vnot` 属于[一元向量操作](../../unary-vector-ops_zh.md)指令集。

## 概述

`%result` 在每个活跃 lane 上保存按位取反结果。

## 机制

`pto.vnot` 逐 lane 计算按位取反：`dst[i] = ~src[i]`。它会翻转每个元素中的全部比特。只支持整数元素类型。非活跃 lane 保持目标原值。

## 语法

### PTO 汇编形式

```text
vnot %result, %input, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 当前文档化支持的类型：全部整数类型。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器；在每个活跃 lane 上读取 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 为活跃 lane |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到 `~src[i]`；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 只支持整数元素类型。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vnot` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：全部整数类型。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

## 相关页面

- 指令集总览：[一元向量操作](../../unary-vector-ops_zh.md)
- 上一条指令：[pto.vrelu](./vrelu_zh.md)
- 下一条指令：[pto.vbcnt](./vbcnt_zh.md)
