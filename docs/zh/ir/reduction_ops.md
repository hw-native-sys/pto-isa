# 归约运算

本节描述了 PTO ISA 中沿行或沿列进行归约（reduction）的全部操作。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用“目标传递风格”（Destination-Passing Style, DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。全部操作执行在 **Vector 流水线**（`PIPE_V`）上。

这一类操作通常具有如下装配形式：

```mlir
pto.op ins(%src : !pto.tile_buf<...>)
       outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 所有 tile 必须位于 `loc=vec`（VEC/UB 存储空间）
- 输入 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
- 输入与输出元素类型一致（`argmax`/`argmin` 除外，其输出为整数索引类型）

---

## 目录

- [`pto.trowsum` — 行求和归约](#ptotrowsum--行求和归约)
- [`pto.trowprod` — 行乘积归约](#ptotrowprod--行乘积归约)
- [`pto.trowmax` — 行最大值归约](#ptotrowmax--行最大值归约)
- [`pto.trowmin` — 行最小值归约](#ptotrowmin--行最小值归约)
- [`pto.trowargmax` — 行最大值索引归约](#ptotrowargmax--行最大值索引归约)
- [`pto.trowargmin` — 行最小值索引归约](#ptotrowargmin--行最小值索引归约)
- [`pto.tcolsum` — 列求和归约](#ptotcolsum--列求和归约)
- [`pto.tcolprod` — 列乘积归约](#ptotcolprod--列乘积归约)
- [`pto.tcolmax` — 列最大值归约](#ptotcolmax--列最大值归约)
- [`pto.tcolmin` — 列最小值归约](#ptotcolmin--列最小值归约)
- [`pto.tcolargmax` — 列最大值索引归约](#ptotcolargmax--列最大值索引归约)
- [`pto.tcolargmin` — 列最小值索引归约](#ptotcolargmin--列最小值索引归约)

---

## 操作详解

### `pto.trowsum` — 行求和归约

```mlir
pto.trowsum ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    dst[i, 0] = sum over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的求和结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`，`blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例：**

```mlir
pto.trowsum ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowprod` — 行乘积归约

```mlir
pto.trowprod ins(<src>, <tmp> : <src_type>, <tmp_type>)
             outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    dst[i, 0] = product over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的乘积结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape 和元素类型
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`，`blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例：**

```mlir
pto.trowprod ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=1,
                 v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.trowmax` — 行最大值归约

```mlir
pto.trowmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    dst[i, 0] = max over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的最大值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`，`blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例：**

```mlir
pto.trowmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowmin` — 行最小值归约

```mlir
pto.trowmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    dst[i, 0] = min over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，用于中间计算 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的最小值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`，`blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例：**

```mlir
pto.trowmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowargmax` — 行最大值索引归约

```mlir
pto.trowargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    dst[i, 0] = argmax over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每行最大值的列索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `dst` 使用 `slayout=none_box`，且为 DN-style 列向量（`blayout=col_major`，`cols=1`）或 ND-style tile（`valid column == 1`）
  - `src` 元素类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 元素类型：`i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**示例：**

```mlir
pto.trowargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowargmin` — 行最小值索引归约

```mlir
pto.trowargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    dst[i, 0] = argmin over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每行最小值的列索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `dst` 使用 `slayout=none_box`，且为 DN-style 列向量（`blayout=col_major`，`cols=1`）或 ND-style tile（`valid column == 1`）
  - `src` 元素类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 元素类型：`i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**示例：**

```mlir
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolsum` — 列求和归约

```mlir
pto.tcolsum ins(<src>, <tmp> {isBinary = false} : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each column j:
    dst[0, j] = sum over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，用于中间计算 |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的求和结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `isBinary` — 是否使用二叉归约树。默认值为 `false`。
  - `true` — 使用二叉归约树
  - `false` — 使用默认归约方式

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == tmp_type == src_type`
  - `src valid column == dst valid column`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == tmp_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolprod` — 列乘积归约

```mlir
pto.tcolprod ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each column j:
    dst[0, j] = product over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的乘积结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`i16`、`ui16`、`i32`、`ui32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolprod ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                 v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.tcolmax` — 列最大值归约

```mlir
pto.tcolmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each column j:
    dst[0, j] = max over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的最大值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolmin` — 列最小值归约

```mlir
pto.tcolmin ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each column j:
    dst[0, j] = min over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的最小值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolmin ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolargmax` — 列最大值索引归约

```mlir
pto.tcolargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each column j:
    dst[0, j] = argmax over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每列最大值的行索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `src` 元素类型必须为 `f16` 或 `f32`
  - `dst` 元素类型必须为 `i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `dst valid row == 1`
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolargmin` — 列最小值索引归约

```mlir
pto.tcolargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each column j:
    dst[0, j] = argmin over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每列最小值的行索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `src` 元素类型必须为 `f16` 或 `f32`
  - `dst` 元素类型必须为 `i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `dst valid row == 1`
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```
