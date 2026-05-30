# pto.tile_buf_addr

`pto.tile_buf_addr` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

把 `!pto.tile_buf<...>` 的数据区地址提取为类型化的 PTO 指针（`!pto.ptr<T, space>`）或 memref 视图。**这条指令是 tile-buffer 指令与基于指针的向量指令之间的桥梁。**

## 机制

在 `pto.vecscope` / `pto.strict_vecscope` 体内，向量加载/存储操作（`pto.vlds`、`pto.vsts` 等）消费的是类型化指针，而不是 tile 句柄。`pto.tile_buf_addr` 从作用域外分配的 tile 句柄中物化出一个 `vec` 地址空间的指针（或 memref），让向量作用域代码可以读写 tile 层准备好的同一片片上数据。

纯操作：不搬数据、不分配、不参与流水线同步。下沉时通常变成 no-op 或一条由属性驱动的地址常量。

## 语法

```mlir
%ub_ptr = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> !pto.ptr<T, vec>
%ub_ref = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> memref<...>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%tile` | `!pto.tile_buf<...>`（或绑定到 tile 的 memref 形式） | 要取数据区地址的 tile 句柄 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%ub_ptr` / `%ub_ref` | `!pto.ptr<T, space>` 或 `memref<...>` | tile 数据区的类型化指针（如 `!pto.ptr<f32, vec>`）或 memref 视图。memref 结果使用 tile 的静态形状与地址空间；指针结果使用 tile 的元素类型与内存空间。 |

## 约束

!!! warning "约束"
    - 结果必须是类型化 PTO 指针或 memref 视图，其它结果类型不接受。
    - 当请求 memref 结果时，下沉形式使用 tile 的静态形状与地址空间。
    - `pto.tile_buf_addr` **只在 `pto.vecscope` / `pto.strict_vecscope` 内合法**。
    - 在向量作用域外，tile 句柄**必须**由 tile 层操作（`pto.tload`、`pto.tstore`、`pto.tadd` 等）消费，不能通过取地址来用。
    - 反之，tile 层操作**不得**出现在 `pto.vecscope` 内。

## 示例

```mlir
%tile = pto.alloc_tile addr = %c0_i64 valid_row = %r
  : !pto.tile_buf<vec, 8x128xf32, valid=?x?>

pto.vecscope {
  %ub = pto.tile_buf_addr %tile
    : !pto.tile_buf<vec, 8x128xf32, valid=?x?> -> !pto.ptr<f32, vec>
  // ... 向量作用域内的 load / store 在 %ub 上进行 ...
}
```

## 与 Tile / 微指令两个表面的关系

| 表面 | 消费什么 | 桥梁 |
|---|---|---|
| **Tile**（`pto.t*`） | `!pto.tile_buf<...>` | — |
| **微指令 / 向量**（`pto.v*`、`pto.vlds`、`pto.vsts`） | `!pto.ptr<T, space>` | `pto.tile_buf_addr` |

微指令侧由 `pto.vecscope` 包裹。在该作用域内，`pto.tile_buf_addr` 是从 tile 句柄拿到指针的**唯一**合法方式。作用域外，向量操作非法，tile 句柄完全由 tile 层操作所有。

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 分配 tile：[pto.alloc_tile](./alloc-tile_zh.md)
- 向量执行作用域：[pto.vecscope](../../../scalar/ops/micro-instruction/vecscope_zh.md)
- 消费该指针的向量加载/存储：[向量加载存储](../../../vector/vector-load-store_zh.md)
- tensor view 取地址姊妹操作：[pto.tensor_view_addr](./tensor-view-addr_zh.md)
