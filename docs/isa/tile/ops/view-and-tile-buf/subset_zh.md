# pto.subset

`pto.subset` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

在运行时偏移、静态大小下创建父 tile buffer 的 strided 子视图：`result = source[offsets] sizes [rows, cols]`。**不搬数据。**

## 机制

`pto.subset` 生成一个与父 tile 别名的子 `!pto.tile_buf<...>`。运行时 `%i`、`%j` 给出子区域的左上角，静态 `sizes` 属性固定结果尺寸。

带 box 布局的 tile buffer（例如 cube 路径上的 fractal NZ tile）从其内层 box 形状继承额外的对齐约束；对这类 tile，subset 必须与 box 对齐。非 box 布局除继承父 tile 的元素类型和地址空间外，不再施加额外结构性检查。

纯操作：不分配内存、不搬数据。

## 语法

```mlir
%sub = pto.subset %src[%i, %j] sizes [rows, cols] : !pto.tile_buf<...>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%src` | `!pto.tile_buf<...>` | 父 tile buffer |
| `offsets` | `Variadic<index>` | 运行时偏移 `[i, j]` |
| `sizes`（属性） | `I64ArrayAttr` | 静态形状 `[rows, cols]` |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%sub` | `!pto.tile_buf<...>` | 父 tile 的 strided 子视图。元素类型、地址空间、tile config 从 `%src` 继承；常量偏移下 `valid_shape` 由父 valid 形状与偏移推导，动态偏移下结果 valid 形状为动态。 |

## 约束

!!! warning "约束"
    - box / 非 box 的行为由源 tile config（`blayout`、`slayout`、`fractal`）和元素类型决定。
    - 非 box 布局（`slayout=none_box`）下不施加额外的结构性检查。
    - box 布局下：
        - `sizes` 长度必须为 2，且两个 subset 尺寸都必须为正。
        - subset 尺寸必须是推断出的内层 box 形状的整数倍。
        - `offsets` 长度必须为 2；常量偏移必须非负且为内层 box 形状的整数倍。
        - 源 tile 形状必须静态可知。
        - 对 box 行优先 tile：subset 必须保留源 tile 的完整列范围，且列偏移必须为常量 `0`。
        - 对 box 列优先 tile：subset 必须保留源 tile 的完整行范围，且行偏移必须为常量 `0`。
    - 结果继承源 tile 的元素类型、地址空间和 tile config。`valid_shape` 在常量偏移下由父 valid 形状与偏移推导，动态偏移下则保持动态。

## 示例

```mlir
%sub = pto.subset %src[%i, %j] sizes [32, 32]
     : !pto.tile_buf<vec, 64x64xf16>
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 分配父 tile：[pto.alloc_tile](./alloc-tile_zh.md)
- tile 层的重解释（不同操作）：[pto.subview](../sync-and-config/subview_zh.md)
- 取指针（向量作用域内）：[pto.tile_buf_addr](./tile-buf-addr_zh.md)
