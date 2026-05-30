# pto.set_validshape

`pto.set_validshape` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

为已存在的**动态** rank-2 tile buffer（用 `valid=?x?` 分配的）写入运行时 `v_row` / `v_col` 元数据。

## 机制

tile buffer 可以以动态 valid 形状分配（`valid=?x?`），此时 valid 区域在分配时未定。`pto.set_validshape` 把运行时 `valid_row` / `valid_col` 写入描述符，使后续的 tile 层操作（load、store、compute）按一个定义良好的 valid 子区域工作。

本操作只更新元数据——**不搬数据、不改变物理存储布局**。tile 的静态 `R x C` 形状不变，只更新其中的 valid 子区。

## 语法

```mlir
pto.set_validshape %src, %valid_row, %valid_col : !pto.tile_buf<vec, RxCxT, valid=?x?>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%src` | `!pto.tile_buf<vec, RxCxT, valid=?x?>` | 动态 rank-2 tile buffer（两个 valid 维度都是动态的） |
| `%valid_row` | `index` | 运行时 valid row 数 |
| `%valid_col` | `index` | 运行时 valid col 数 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 此形式无 SSA 结果，原地更新 tile 描述符 |

## 约束

!!! warning "约束"
    - `%src` 必须是 rank-2，并且两个维度都使用 `v_row = ?` / `v_col = ?`。
    - tile 程序使用 `pto.tile_buf`；memref 形式是 lowering 的产物，不属于本表面。
    - 常量 `valid_row` / `valid_col` 必须非负，且不超过 tile 静态形状的边界。

## 示例

```mlir
%src = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
pto.set_validshape %src, %vr, %vc : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 分配动态 tile：[pto.alloc_tile](./alloc-tile_zh.md)
- 在已知大小下切子区：[pto.subset](./subset_zh.md)
