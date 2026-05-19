# pto.pbitcast

`pto.pbitcast` 属于 [Conversion Ops](../../conversion-ops_zh.md) 指令集。

## 摘要

`pto.pbitcast` 对 `!pto.mask<...>` 值执行按位重新解释，不改变底层谓词寄存器映像。当生产者与消费者对同一个硬件谓词状态期待不同的粒度视图（`b8`、`b16`、`b32`）时，本操作把 mask 家族之间的重新解释显式化到 VPTO IR 里。

## 机制

本操作在 mask 寄存器层面是一次纯类型转换。VPTO 不会重新计算、规范化或物化任何谓词位，只会更新周围 IR 对同一段谓词位所采用的粒度视图。这样可以把 mask 生产者天然的粒度与消费者所需的粒度解耦，而不必插入额外的硬件操作。

## 语法

```mlir
%result = pto.pbitcast %input : !pto.mask<G0> -> !pto.mask<G1>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%input` | `!pto.mask<G0>` | 源谓词寄存器。 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%result` | `!pto.mask<G1>` | 同一份谓词位，在粒度 `G1` 下重新解释。 |

## 副作用

`pto.pbitcast` 除了产生 SSA 结果以外没有任何架构层面的副作用。它不会重新生成 mask 位，也不会改写硬件谓词状态。

## 约束

!!! warning "约束"
    - 源和目标都必须是 `!pto.mask<...>` 类型。
    - `pto.pbitcast` 不会物化或规范化谓词内容；它只更新周围 VPTO IR 对这同一段谓词位采用的粒度视图。
    - 仅在消费者需要不同 mask 粒度（`b8` / `b16` / `b32`）、但底层谓词映像可以原样复用时使用。若消费者需要的是「重新计算后的谓词」，应通过相应的谓词生成操作（而非 `pto.pbitcast`）来下沉或物化 mask。

## 示例

### 在消费者前把 b16 谓词重新解释为 b32

```mlir
%m16 = pto.pintlv_b16 %lhs, %rhs
    : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
%m32 = pto.pbitcast %m16#0 : !pto.mask<b16> -> !pto.mask<b32>
%result = pto.vsel %a, %b, %m32
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

## 相关页面

- 指令集总览：[Conversion Ops](../../conversion-ops_zh.md)
- 向量侧 bitcast：[pto.vbitcast](./vbitcast_zh.md)
- 谓词生成与代数：[谓词生成与代数](../../../scalar/ops/predicate-generation-and-algebra/)
