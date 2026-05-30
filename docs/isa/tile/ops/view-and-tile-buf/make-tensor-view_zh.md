# pto.make_tensor_view

`pto.make_tensor_view` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

由基地址指针、运行时形状和运行时步长构造一个全局张量视图。**不分配内存、不搬数据**——只是构造描述符。

## 机制

`!pto.tensor_view<...>` 是一个逻辑描述符，承载基址指针、每维度尺寸、每维度元素步长，以及可选的布局提示。tile 层操作（`pto.tload`、`pto.tstore`、`pto.partition_view` 等）以它作为全局内存寻址的唯一来源。

`pto.make_tensor_view` 把上述四块信息打包成一个 SSA 值。本操作是纯操作：只生成描述符，不访问内存。

## 语法

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xT>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%ptr` | `!pto.ptr<T, gm>`（或匹配的指针） | 源指针；元素类型必须与结果一致 |
| `shape` | `Variadic<index>` | 动态形状，按结果 rank 提供每维度一个 |
| `strides` | `Variadic<index>` | 动态步长，**以元素为单位**（非字节），按结果 rank 提供每维度一个 |
| `layout`（属性，可选） | `LayoutAttr` | `nd` / `dn` / `nz` 提示 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%tv` | `!pto.tensor_view<...>` | 逻辑视图描述符 |

## 约束

!!! warning "约束"
    - `%ptr` 的元素类型必须与结果元素类型一致。
    - `shape` 与 `strides` 的操作数数量必须等于 tensor_view 的 rank。
    - 若提供了 `layout` 而 shape/strides 是静态的，则推断出的布局必须与 `layout` 一致。

## 示例

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xf32>
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 查询维度尺寸：[pto.get_tensor_view_dim](./get-tensor-view-dim_zh.md)
- 查询步长：[pto.get_tensor_view_stride](./get-tensor-view-stride_zh.md)
- 取底层地址：[pto.tensor_view_addr](./tensor-view-addr_zh.md)
- 切分子视图：[pto.partition_view](./partition-view_zh.md)
