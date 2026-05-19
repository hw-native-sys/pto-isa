# pto.tensor_view_addr

`pto.tensor_view_addr` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

从 `!pto.tensor_view<...>` 或 `!pto.partition_tensor_view<...>` 描述符中提取底层地址，返回类型化的 PTO 指针或 memref 视图。**纯操作，不搬数据。**

## 机制

张量视图同时承载寻址元信息（形状、步长、基址）和逻辑描述。`pto.tensor_view_addr` 把其中的地址部分投影出来：以类型化 GM 指针（`!pto.ptr<T, gm>`）或 memref 视图的形式暴露同一底层存储。

本操作是纯操作。在编译器内部下沉过程中，操作数可能已被改写为 memref 形式；此时本 op 会被折叠或被改写成等价的 memref-to-ptr cast。

## 语法

```mlir
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> memref<...>
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> !pto.ptr<T, gm>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%src` | `!pto.tensor_view<...>` 或 `!pto.partition_tensor_view<...>` | 源视图描述符 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%result` | `memref<...>` 或 `!pto.ptr<T, gm>` | 按所请求形式返回的底层地址 |

## 约束

!!! warning "约束"
    - 结果类型必须是下沉后的 memref 视图或指向同一底层存储的 GM 指针 `!pto.ptr<T, gm>`，其它类型不接受。
    - 纯操作，不搬数据。

## 示例

```mlir
// 从 tensor view 提取 GM 指针，用于 DMA 拷贝
%base = pto.tensor_view_addr %tv : !pto.tensor_view<?x?xf32> -> !pto.ptr<f32, gm>
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 构造视图：[pto.make_tensor_view](./make-tensor-view_zh.md)
- 切分子视图：[pto.partition_view](./partition-view_zh.md)
- tile-buffer 取地址的姊妹操作：[pto.tile_buf_addr](./tile-buf-addr_zh.md)
