# pto.get_tensor_view_stride

`pto.get_tensor_view_stride` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

从 `!pto.tensor_view<...>`（或其下沉后的 memref 形式）读出指定维度的逻辑步长，**以元素为单位**（非字节）。

## 机制

读取由 [`pto.make_tensor_view`](./make-tensor-view_zh.md) 产生的描述符中 `%idx` 维度的步长。纯操作：不访问内存，不做同步。

因为步长以**元素**为单位返回，与 [`pto.addptr`](../../../scalar/ops/micro-instruction/pointer-operations_zh.md)（也是以元素为单位）做指针算术时可以直接组合，无需额外的 `sizeof(T)` 乘法。

## 语法

```mlir
%stride = pto.get_tensor_view_stride %tv, %idx : !pto.tensor_view<...> -> index
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%tv` | `!pto.tensor_view<...>` 或下沉后的 memref 形式 | 张量视图或其下沉形式 |
| `%idx` | `index` | 维度索引（0-based） |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%stride` | `index` | `%idx` 维度的元素步长 |

## 约束

!!! warning "约束"
    - `%idx` 必须在 `[0, rank(%tv))` 范围内。
    - 返回的步长以**元素**为单位（不是字节）。把元素步长和字节偏移混用而不做显式 `sizeof(T)` 换算是错误。
    - 纯操作：不改变视图或底层内存。

## 示例

```mlir
// 行优先视图的外层维度步长
%s0 = pto.get_tensor_view_stride %tv, %c0 : !pto.tensor_view<?x?xf32> -> index

// 内层维度步长通常为 1（一个元素）
%s1 = pto.get_tensor_view_stride %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 构造视图：[pto.make_tensor_view](./make-tensor-view_zh.md)
- 查询维度尺寸：[pto.get_tensor_view_dim](./get-tensor-view-dim_zh.md)
- 元素粒度的指针算术：[pto.addptr](../../../scalar/ops/micro-instruction/pointer-operations_zh.md)
