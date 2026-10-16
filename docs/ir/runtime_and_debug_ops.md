# 运行时与调试操作

本节描述了 PTO ISA 中与运行时环境查询、调试输出和特殊控制相关的操作族。这类操作按功能分为以下几组：

- **核参数查询**：获取当前核或子核的索引、总核数或总子核数。
- **标量访问**：从 tile 缓冲区读取或写入单个元素。
- **调试输出**：格式化打印标量值或整个 tile 缓冲区内容，用于程序执行时的诊断。
- **中止执行**：无条件终止执行，用于断言失败或调试时的快速失败。

---

## 目录

[**核参数查询**](#核参数查询)

- [`pto.get_block_idx` — 获取当前核索引](#ptoget_block_idx--获取当前核索引)
- [`pto.get_subblock_idx` — 获取当前子核索引](#ptoget_subblock_idx--获取当前子核索引)
- [`pto.get_block_num` — 获取总核数](#ptoget_block_num--获取总核数)
- [`pto.get_subblock_num` — 获取总子核数](#ptoget_subblock_num--获取总子核数)

[**标量访问**](#标量访问)

- [`pto.tsetval` — 写入 Tile 单元素](#ptotsetval--写入-tile-单元素)
- [`pto.tgetval` — 读取 Tile 单元素](#ptotgetval--读取-tile-单元素)

[**调试输出**](#调试输出)

- [`pto.print` — 格式化标量打印](#ptoprint--格式化标量打印)
- [`pto.tprint` — Tile 打印](#ptotprint--tile-打印)

[**中止执行**](#中止执行)

- [`pto.trap` — 陷阱/中止执行](#ptotrap--陷阱中止执行)

---

## 操作详解

## 核参数查询

本组操作均为纯操作（Pure），无副作用，无额外约束。

### `pto.get_block_idx` — 获取当前核索引

```mlir
%idx = pto.get_block_idx
```

**语义：**

```text
result = block_idx()
```

**参数：** 无

**返回值：** `i64` 类型的值，表示当前执行核的索引，范围为 `[0, BlockNum-1]`。

**示例：**

```mlir
%idx = pto.get_block_idx
```

---

### `pto.get_subblock_idx` — 获取当前子核索引

```mlir
%idx = pto.get_subblock_idx
```

**语义：**

```text
result = subblock_idx()
```

**参数：** 无

**返回值：** `i64` 类型的值，表示当前执行子核（向量核）的索引。

**示例：**

```mlir
%idx = pto.get_subblock_idx
```

---

### `pto.get_block_num` — 获取总核数

```mlir
%num = pto.get_block_num
```

**语义：**

```text
result = block_num()
```

**参数：** 无

**返回值：** `i64` 类型的值，表示总的核（block）数量。

**示例：**

```mlir
%num = pto.get_block_num
```

---

### `pto.get_subblock_num` — 获取总子核数

```mlir
%num = pto.get_subblock_num
```

**语义：**

```text
result = subblock_num()
```

**参数：** 无

**返回值：** `i64` 类型的值，表示总的子核（向量核）数量。

**示例：**

```mlir
%num = pto.get_subblock_num
```

---

## 标量访问

### `pto.tsetval` — 写入 Tile 单元素

```mlir
pto.tsetval ins(<offset>, <val> : index, <scalar_type>)
            outs(<dst> : !pto.tile_buf<...>)
```

**语义：**

```text
dst[offset] = val  // 在线性偏移处写入单个标量值
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `offset` | `index` | tile 缓冲区中的线性偏移（从 0 开始） |
| `val` | `ScalarType` | 待写入的标量值 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **类型匹配** — `val` 的标量类型必须与 `dst` 的元素类型完全匹配（如 `f16` 与 `f16`）。
- **偏移范围** — `offset` 必须在 `[0, rows*cols)` 范围内。
- **有效性** — 操作只在 `dst` 的有效区域内生效。
- **硬件管道** — 该操作在标量管道（`PIPE_S`）上执行，因此与同管道及依赖管道上的操作之间存在顺序约束。

**示例：**

```mlir
%offset = ... : index
%val = ... : f16
pto.tsetval ins(%offset, %val : index, f16)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tgetval` — 读取 Tile 单元素

```mlir
%val = pto.tgetval ins(<src>, <offset> : !pto.tile_buf<...>, index) outs : <scalar_type>
```

**语义：**

```text
result = src[offset]  // 从线性偏移处读取单个标量值
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile 缓冲区 |
| `offset` | `index` | tile 缓冲区中的线性偏移（从 0 开始） |

**返回值：** 标量值，类型与 `src` 的元素类型相同。

**约束：**

- **位置要求** — `src` 必须使用 `loc=vec`。`loc=mat` 不被接受。
- **偏移范围** — `offset` 必须在 `[0, rows*cols)` 范围内。
- **元素类型** — `dst` 类型必须与 `src` 的元素类型一致。
- **硬件管道** — 该操作在标量管道（`PIPE_S`）上执行，因此与同管道及依赖管道上的操作之间存在顺序约束。

**示例：**

```mlir
%src = ... : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>
%offset = ... : index
%val = pto.tgetval ins(%src, %offset : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>, index) outs : f16
```

---

## 调试输出

**通用约束：**

- 此类操作具有内存写副作用（`MemWrite`），用于标记调试输出
- 输出在宿主（Host）端可见
- 是否生成有效输出由具体实现与目标平台决定

### `pto.print` — 格式化标量打印

```mlir
pto.print ins(<format_str>, <scalar> : <scalar_type>)
```

**语义：**

```text
printf(format_str, scalar)
```

其中 `format_str` 在编译时确定，`scalar` 为运行时值。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `format_str` | `StrAttr` | 编译时字符串属性，包含格式控制符（如 `%+08.3f`） |
| `scalar` | `ScalarType`（signless integer / float）| 运行时标量值，类型必须为整数或浮点数 |

**返回值：** 无。本操作为调试输出，不产生返回值。

**约束：**

- **格式要求** — `format_str` 必须为 MLIR 字符串属性（StrAttr）。
- **标量类型** — `scalar` 必须为以下类型之一：`i32`、`i64`、`f32`、`f64`、或其他数值类型。不支持 `tile_buf` 或复杂类型。

**示例：**

```mlir
%val = ... : f32
pto.print ins("Value: %+08.3f", %val : f32)
```

---

### `pto.tprint` — Tile 打印

```mlir
pto.tprint ins(<src> : !pto.tile_buf<...>)

// 通过临时全局缓冲区打印 Mat/Acc tile 的形式
pto.tprint ins(<src>, <tmp> : !pto.tile_buf<...>, !pto.partition_tensor_view<...>)
          {printFormat = <fmt>}
```

**语义：**

```text
print(src)  // 打印整个 tile 缓冲区的内容
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` / `pto.partition_tensor_view` | 待打印的 tile 缓冲区或全局视图；`tmp` 存在时必须为 `pto.tile_buf` |
| `tmp` | `pto.partition_tensor_view`（可选） | 打印 Mat/Acc tile 时使用的临时全局缓冲区视图，**不是** `pto.tile_buf` |

**返回值：** 无。本操作为调试输出，不产生返回值。

**属性：**

- `printFormat` — TPRINT 输出格式。可选，默认不指定。
  - `#pto<print_format width8_precision4>` — 宽度 8、精度 4
  - `#pto<print_format width8_precision2>` — 宽度 8、精度 2
  - `#pto<print_format width10_precision6>` — 宽度 10、精度 6

**约束：**

`src` 的两种形态走不同的校验路径，约束也不同。

- **src 为 tile（`pto.tile_buf`）**
  - **元素类型** — 必须为以下之一：`f16`、`f32`、`i8`、`i16`、`i32`。
  - **不带 `tmp`** — `src` 必须位于 `loc=vec`，其他位置会被拒绝。
  - **带 `tmp`** — `src` 可为 `vec`/`mat`/`acc`；Ascend 950PR&950DT系列产品 目标不允许打印 `mat` tile（仅 Atlas A2系列产品/Atlas A3系列产品 支持）。
- **src 为全局视图（`pto.partition_tensor_view`）**
  - **不带 `tmp`** — 直接打印，无 `loc` 要求（视图类型不携带局部位置）。
  - **不允许携带 `tmp`** — verifier 检查 `expects tmp only when src is a tile_buf`；`src` 不是 `tile_buf` 时必须不写 `tmp`。
- **`tmp` 类型** — 仅在 `src` 为 tile 时允许携带，且 `tmp` 必须是 `!pto.partition_tensor_view<...>`（verifier 检查 `expects mem to be !pto.partition_tensor_view`，写成 tile_buf 会被拒绝）；其元素类型必须与 `src` 的元素类型一致。
- **`tmp` 布局** — 当 `tmp` 的布局可推导时，必须使用 ND 逻辑布局。
- **缓冲区限制** — 打印缓冲区大小有限，超大 tile 打印可能被截断。
- **硬件管道** — 该操作在向量管道（`PIPE_V`）上执行，因此与同管道及依赖管道上的操作之间存在顺序约束。

**示例：**

```mlir
// src 为 tile，不带 tmp：必须 loc=vec
pto.tprint ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)

// src 为 tile，带 tmp：可打印 acc/mat，tmp 用 partition_tensor_view
pto.tprint ins(%src, %tmp :
        !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=col_major, slayout=row_major,
            fractal=1024, pad=0>,
        !pto.partition_tensor_view<16x16xf32>)
          {printFormat = #pto<print_format width10_precision6>}

// src 为全局视图：无 loc 要求，且不能携带 tmp
pto.tprint ins(%pv : !pto.partition_tensor_view<16x16xf32>)
```

---

## 中止执行

### `pto.trap` — 陷阱/中止执行

```mlir
pto.trap
```

**语义：**

```text
trap()  // 无条件终止当前核执行，不返回
```

**参数：** 无

**返回值：** 无。此操作无条件中止执行。

**约束：**

- **行为** — 无条件终止当前核的执行，不会返回到调用者。通常用于断言失败时的快速失败路径。
- **控制流** — 通常出现在条件分支中（如 `scf.if` 的某一分支），或作为不可达代码的标记。

**示例：**

```mlir
scf.if %cond {
  pto.trap  // 条件为真时中止
}
```
