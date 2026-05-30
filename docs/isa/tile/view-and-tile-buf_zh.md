# 视图与 Tile Buffer

视图与 tile buffer 操作是 PTO tile 编程模型的底层基础，覆盖四类职责：

1. **构造描述符** —— 为全局张量构造形状 + 步长 + 基址描述符 `pto.make_tensor_view`
2. **查询描述符** —— 读取运行时形状/步长 `pto.get_tensor_view_dim`、`pto.get_tensor_view_stride`
3. **切分子视图** —— 把全局描述符切成逻辑子窗口 `pto.partition_view`
4. **管理片上 tile buffer** —— 分配、取子区、设置 valid 形状、取指针 `pto.alloc_tile`、`pto.subset`、`pto.set_validshape`、`pto.tile_buf_addr`、`pto.tensor_view_addr`

这些都是纯描述符/句柄操作：不搬数据、不在运行时分配内存、不参与流水线同步。它们建立了 tile 计算操作（`pto.tload`、`pto.tstore`、`pto.tadd`、`pto.tmatmul` 等）和向量微指令（`pto.vlds`、`pto.vsts` 等）共同依赖的寻址与生命周期契约。

## per-op 页面

### Tensor View —— 全局内存描述符

- [pto.make_tensor_view](./ops/view-and-tile-buf/make-tensor-view_zh.md)：由指针、形状、步长构造张量视图
- [pto.get_tensor_view_dim](./ops/view-and-tile-buf/get-tensor-view-dim_zh.md)：读维度尺寸
- [pto.get_tensor_view_stride](./ops/view-and-tile-buf/get-tensor-view-stride_zh.md)：读元素步长
- [pto.tensor_view_addr](./ops/view-and-tile-buf/tensor-view-addr_zh.md)：投影底层地址（memref 或 `!pto.ptr<T, gm>`）
- [pto.partition_view](./ops/view-and-tile-buf/partition-view_zh.md)：从张量视图切出分区窗口

### Tile Buffer —— 片上存储

- [pto.alloc_tile](./ops/view-and-tile-buf/alloc-tile_zh.md)：声明新的 tile buffer 生命周期
- [pto.subset](./ops/view-and-tile-buf/subset_zh.md)：从父 tile 切出 strided 子区
- [pto.set_validshape](./ops/view-and-tile-buf/set-validshape_zh.md)：为动态 tile 设置运行时 valid 形状
- [pto.tile_buf_addr](./ops/view-and-tile-buf/tile-buf-addr_zh.md)：**tile↔向量桥梁**——在 `pto.vecscope` 内取类型化指针

## Tile ↔ 向量桥梁

`!pto.tile_buf<...>` 类型属于 tile 表面，向量微指令消费的是类型化指针 `!pto.ptr<T, space>`。两个表面之间**唯一**合法的桥梁是 [`pto.tile_buf_addr`](./ops/view-and-tile-buf/tile-buf-addr_zh.md)，并且**只在** [`pto.vecscope`](../scalar/ops/micro-instruction/vecscope_zh.md) 区域内有效。在 `pto.vecscope` 外，tile 句柄只能传给 tile 层操作；在 `pto.vecscope` 内，tile 层操作非法，向量作用域代码必须通过 `pto.tile_buf_addr` 拿到的指针来工作。

这种拆分让两个表面可以无歧义地组合在一起。

## 相关页面

- [Tile ISA 参考](./README_zh.md)：tile 指令清单
- [内存与数据搬运](./memory-and-data-movement_zh.md)：tile 层 GM ↔ tile DMA
- [向量执行作用域 (`pto.vecscope`)](../scalar/ops/micro-instruction/vecscope_zh.md)：`pto.tile_buf_addr` 合法的作用域
- [指针操作](../scalar/ops/micro-instruction/pointer-operations_zh.md)：`pto.addptr` / `pto.castptr` / `pto.load_scalar` / `pto.store_scalar`
