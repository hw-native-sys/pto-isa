# pto.partition_view

`pto.partition_view` 属于 [视图与 Tile Buffer](../../view-and-tile-buf_zh.md) 指令集。

## 摘要

按每维度的偏移和大小从父 `!pto.tensor_view<...>` 中切出一个 `!pto.partition_tensor_view<...>`，即一个逻辑子窗口——**不分配内存、不搬数据**。

## 机制

`result = source[offsets, sizes]`。这条操作把静态与动态形状信息一并捕获到结果分区描述符中。下游 tile 层操作（如 `pto.tload`、`pto.tstore`）可以直接消费这个分区视图，无需在每个调用点重新计算偏移。

纯操作：不访问内存、不改变父视图。

## 语法

```mlir
%pv = pto.partition_view %tv, offsets = [%o0, %o1], sizes = [%s0, %s1]
    : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%tv` | `!pto.tensor_view<...>` | 输入张量视图 |
| `offsets` | `Variadic<index>` | 各维度的动态偏移 |
| `sizes` | `Variadic<index>` | 分区在各维度上的动态大小 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%pv` | `!pto.partition_tensor_view<...>` | 逻辑分区描述符 |

## 约束

!!! warning "约束"
    - `offsets` 与 `sizes` 的操作数数量必须各自等于 `%tv` 的 rank。
    - 纯操作，不分配内存、不搬数据。
    - `offsets + sizes` 超出父视图形状的行为是 target-defined。

## 示例

```mlir
// 在 1024x512 视图内取一个 16x16 分区，起点为 (%off0, %off1)
%pv = pto.partition_view %tv, offsets = [%off0, %off1], sizes = [%s0, %s1]
    : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

## 相关页面

- 指令集总览：[视图与 Tile Buffer](../../view-and-tile-buf_zh.md)
- 构造父视图：[pto.make_tensor_view](./make-tensor-view_zh.md)
- 取底层地址：[pto.tensor_view_addr](./tensor-view-addr_zh.md)
