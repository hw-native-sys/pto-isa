# pto.vtrc

`pto.vtrc` 属于[转换操作](../../conversion-ops_zh.md)指令集。

## 概述

对每个 lane 做舍入或截断，但保持元素类型不变。

## 机制

`pto.vtrc` 会对源向量中每个活跃 lane 按目标 profile 选定的舍入模式执行舍入。与 `pto.vcvt` 不同，它不会改变目标元素类型，而是在原有元素类型内部完成舍入，因此适用于 floor、向零截断、向最近值舍入等场景。

## 语法

### PTO 汇编形式

```text
vtrc %dst, %src, "ROUND_MODE"
```

### AS Level 1（SSA）

```mlir
%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器 |
| `ROUND_MODE` | 枚举 | 舍入模式，如 round-to-zero、floor、ceil、round-to-nearest 等 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 舍入或截断后的结果向量，元素类型与源保持一致 |

## 副作用

这条指令除了产生目标向量寄存器外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- `pto.vtrc` 会保持源向量的宽度和元素类型不变。
- 所选 `ROUND_MODE` 必须是目标 profile 支持的模式。
- lowering 必须精确保留逐 lane 的舍入语义。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不支持的舍入模式属性。
- 约束部分列出的额外非法情形，同样属于 `pto.vtrc` 的契约。

## 目标 Profile 限制

- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。
- 如果代码依赖某个特定舍入模式，应把这种依赖视为 profile 相关。

## 示例

```mlir
%rounded = pto.vtrc %input, "ROUND_R" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
```

## 相关页面

- 指令集总览：[转换操作](../../conversion-ops_zh.md)
- 上一条指令：[pto.vcvt](./vcvt_zh.md)
- 相关转换：[pto.vcvt](./vcvt_zh.md)
