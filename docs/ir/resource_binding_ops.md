# 资源绑定操作

本章介绍 PTO ISA 中的资源绑定操作，负责 Tile 缓冲区的逻辑分配。

---

## 目录

- [`pto.alloc_tile` — 逻辑 Tile 缓冲区分配](#ptoalloc_tile--逻辑-tile-缓冲区分配)

---

## 操作详解

### `pto.alloc_tile` — 逻辑 Tile 缓冲区分配

```mlir
pto.alloc_tile (valid_row = <vr>)? (valid_col = <vc>)?
               : <result_type>
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
result = allocate_tile_buffer(valid_row?, valid_col?)
// 创建一个逻辑 tile buffer handle，可选指定动态有效维度
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `valid_row` | `index`（可选） | 运行时有效行数，当结果类型的 `v_row` 为动态（`?`）时必须提供 |
| `valid_col` | `index`（可选） | 运行时有效列数，当结果类型的 `v_col` 为动态（`?`）时必须提供 |

**返回值：** `!pto.tile_buf<...>` — 分配得到的 tile buffer handle。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 结果 tile_buf 必须具有 rank-2 的 validShape。
  - 当结果类型的 `v_row` 为动态（`?`）时，必须提供 `valid_row` 操作数；当 `v_row` 为静态时，不得提供。
  - 当结果类型的 `v_col` 为动态（`?`）时，必须提供 `valid_col` 操作数；当 `v_col` 为静态时，不得提供。
  - 结果 tile_buf 的 layout 约束必须合法（通过 `verifyTileBufLayoutConstraints` 验证）。

**示例：**

```mlir
// 静态 valid shape，无需 valid_row/valid_col
%t0 = pto.alloc_tile
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>

// 动态 valid shape，需提供 valid_row 和 valid_col
%t1 = pto.alloc_tile
                     valid_row = %c16 valid_col = %c256
    : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                    v_row=?, v_col=?, blayout=col_major,
                    slayout=row_major, fractal=1024, pad=0>
```
