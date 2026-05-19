# pto.alloc_tile

`pto.alloc_tile` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

声明一个 `!pto.tile_buf<...>` 的生命周期。每一次调用都会产生一个**相互独立**的 tile buffer 实例。

## 机制

tile buffer 是片上内存（UB / L1 / L0A / L0B / L0C / BT / scaling buffer）上一个有界的 2D 矩形区域，具有显式的生命周期。`pto.alloc_tile` 引入一个新鲜的 SSA 值来代表这样一个实例，并让实现自动分配具体地址，或通过可选的 `addr` 子句接受显式地址。

如果结果 tile 类型的 `v_row` / `v_col` 是动态的（`valid=?x?`），则必须在 alloc 时提供对应的 `valid_row` / `valid_col`，以便下游操作看到一个定义良好的 valid 区域。

纯操作：不搬数据、不做同步。

## 语法

```mlir
%tb  = pto.alloc_tile : !pto.tile_buf<...>
%tb2 = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<vec, RxCxT, valid=?x?>
%tb3 = pto.alloc_tile addr = %ad : !pto.tile_buf<...>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `addr` | `Optional<i64>` | 可选起始地址。省略时由实现分配。 |
| `valid_row` | `Optional<index>` | 动态 valid-row 数量。当结果类型的 `v_row = ?` 时必须提供。 |
| `valid_col` | `Optional<index>` | 动态 valid-col 数量。当结果类型的 `v_col = ?` 时必须提供。 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%tb` | `!pto.tile_buf<...>` | 新分配的 tile buffer 实例 |

## 约束

!!! warning "约束"
    - 若结果 `v_row` / `v_col` 是动态（`?`），对应操作数**必须**存在。
    - 若结果 `v_row` / `v_col` 是静态，对应操作数**必须**省略。
    - 每次调用都产生一个**独立**的 tile buffer 实例，即使重复用相同参数调用也不会复用。

## 示例

```mlir
%tb = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
```

```mlir
// 动态 valid 形状：必须传 valid_row / valid_col
%tb = pto.alloc_tile valid_row = %vr valid_col = %vc
    : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 更新动态 valid 形状：[pto.set_validshape](./set-validshape_zh.md)
- 切出子区：[pto.subset](./subset_zh.md)
- 跨向量加载/存储桥梁：[pto.tile_buf_addr](./tile-buf-addr_zh.md)
