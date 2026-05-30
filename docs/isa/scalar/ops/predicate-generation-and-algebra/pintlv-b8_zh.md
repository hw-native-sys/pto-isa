# pto.pintlv_b8

`pto.pintlv_b8` 属于 [谓词生成与代数](../../predicate-generation-and-algebra_zh.md) 指令集。

## 摘要

按 `b8` 粒度对两个谓词源做按位交错，分别生成低半和高半两路谓词输出。

## 机制

`pto.pintlv_b8` 是谓词交错家族（`pto.pintlv_b8` / [`pto.pintlv_b16`](./pintlv-b16_zh.md) / [`pto.pintlv_b32`](./pintlv-b32_zh.md)）的 8 位元素粒度变体。它接收两个 `!pto.mask<b8>` 源，按相同的 `b8` 粒度产生两路交错后的谓词。底层硬件的谓词寄存器位模式保持不变；改变的只是把这些位按 8 位元素分组的方式。

## 语法

### AS Level 1（SSA）

```mlir
%low, %high = pto.pintlv_b8 %src0, %src1
    : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%src0` | `!pto.mask<b8>` | 第一路谓词源 |
| `%src1` | `!pto.mask<b8>` | 第二路谓词源 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%low` | `!pto.mask<b8>` | 从 `%src0` / `%src1` 交错产生的低半 |
| `%high` | `!pto.mask<b8>` | 从 `%src0` / `%src1` 交错产生的高半 |

## 副作用

无。`pto.pintlv_b8` 是纯粹的谓词变换：不会读写 UB、GM，也不会改变除两路 SSA 结果以外的任何架构状态。

## 约束

!!! warning "约束"
    - 所有操作数和结果都必须使用 `!pto.mask<b8>`。混合谓词粒度是非法的；如果生产者产生的是另一种粒度，先用 `pto.pbitcast` 重新解释。
    - 两路输出形成一个有序对 (`%low`, `%high`)，这种配对关系必须保持。

## 示例

```mlir
%lo, %hi = pto.pintlv_b8 %m0, %m1
    : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>
```

## 相关页面

- 指令集总览：[谓词生成与代数](../../predicate-generation-and-algebra_zh.md)
- 其它变体：[pto.pintlv_b16](./pintlv-b16_zh.md)、[pto.pintlv_b32](./pintlv-b32_zh.md)
- 反向操作：[pto.pdintlv_b8](./pdintlv-b8_zh.md)
- 谓词粒度重解释：[pto.pbitcast](../../../vector/ops/conversion-ops/pbitcast_zh.md)
