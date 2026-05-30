# pto.get_tensor_view_dim

`pto.get_tensor_view_dim` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

从 `!pto.tensor_view<...>` 读出指定维度的运行时大小。

## 机制

读取由 [`pto.make_tensor_view`](./make-tensor-view_zh.md) 产生的描述符中 `%idx` 维度的尺寸。纯操作：不访问内存，不做同步，无任何架构副作用。

## 语法

```mlir
%dim = pto.get_tensor_view_dim %tv, %idx : !pto.tensor_view<...> -> index
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%tv` | `!pto.tensor_view<...>` | 逻辑张量视图 |
| `%idx` | `index` | 维度索引（0-based） |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%dim` | `index` | `%idx` 维度的运行时尺寸 |

## 约束

!!! warning "约束"
    - `%idx` 必须在 `[0, rank(%tv))` 范围内。
    - 纯操作：不改变视图或底层内存。

## 示例

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
%w = pto.get_tensor_view_dim %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 构造视图：[pto.make_tensor_view](./make-tensor-view_zh.md)
- 查询步长：[pto.get_tensor_view_stride](./get-tensor-view-stride_zh.md)
