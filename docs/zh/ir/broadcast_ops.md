# 广播运算

本节描述了 PTO ISA 中沿行或沿列进行广播（broadcast）的全部操作。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用“目标传递风格”（Destination-Passing Style, DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。全部操作执行在 **Vector 流水线**（`PIPE_V`）上。

这一类操作通常具有如下装配形式：

```mlir
pto.op ins(%src : !pto.tile_buf<...>)
       outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 所有 tile 必须位于 `loc=vec`（VEC/UB 存储空间）
- 输入 tile 使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）

---

## 目录

- [`pto.tcolexpand` — 列广播](#ptotcolexpand--列广播)
- [`pto.trowexpand` — 行广播](#ptotrowexpand--行广播)
- [`pto.trowexpandsub` — 行广播减法](#ptotrowexpandsub--行广播减法)
- [`pto.trowexpandmul` — 行广播乘法](#ptotrowexpandmul--行广播乘法)
- [`pto.trowexpanddiv` — 行广播除法](#ptotrowexpanddiv--行广播除法)
- [`pto.tcolexpandmax` — 列广播取最大值](#ptotcolexpandmax--列广播取最大值)
- [`pto.tcolexpandmin` — 列广播取最小值](#ptotcolexpandmin--列广播取最小值)
- [`pto.tcolexpandmul` — 列广播乘法](#ptotcolexpandmul--列广播乘法)
- [`pto.tcolexpandadd` — 列广播加法](#ptotcolexpandadd--列广播加法)
- [`pto.tcolexpanddiv` — 列广播除法](#ptotcolexpanddiv--列广播除法)
- [`pto.tcolexpandexpdif` — 列广播指数差](#ptotcolexpandexpdif--列广播指数差)
- [`pto.tcolexpandsub` — 列广播减法](#ptotcolexpandsub--列广播减法)
- [`pto.trowexpandadd` — 行广播加法](#ptotrowexpandadd--行广播加法)
- [`pto.trowexpandexpdif` — 行广播指数差](#ptotrowexpandexpdif--行广播指数差)
- [`pto.trowexpandmax` — 行广播取最大值](#ptotrowexpandmax--行广播取最大值)
- [`pto.trowexpandmin` — 行广播取最小值](#ptotrowexpandmin--行广播取最小值)

---

## 操作详解

### `pto.tcolexpand` — 列广播

```mlir
pto.tcolexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile，行向量，每列携带一个逻辑标量 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 和 `dst` 必须使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 元素类型一致：`dst_type == src_type`
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                   v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowexpand` — 行广播

```mlir
pto.trowexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile，列向量，每行携带一个逻辑标量 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 必须使用 `slayout=none_box`
  - `dst` 必须使用 ND-style 布局（`blayout=row_major`，`slayout=none_box`）
  - 元素类型一致：`dst_type == src_type`
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `src valid row == dst valid row`
  - `src valid row != 0` 且 `src valid column != 0` 且 `dst valid row != 0` 且 `dst valid column != 0`

**示例：**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=8,
                   v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowexpandsub` — 行广播减法

```mlir
pto.trowexpandsub ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（减数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例：**

```mlir
pto.trowexpandsub ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmul` — 行广播乘法

```mlir
pto.trowexpandmul ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（乘数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例：**

```mlir
pto.trowexpandmul ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpanddiv` — 行广播除法

```mlir
// 默认精度
pto.trowexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)

// 高精度（需要 tmp）
pto.trowexpanddiv ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = #pto<div_precision high_precision>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile（被除数） |
| `src1` | `pto.tile_buf` | 每行标量载体（除数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 除法精度模式。默认值为 `#pto<div_precision default>`。
  - `#pto<div_precision default>` — 标准精度除法
  - `#pto<div_precision high_precision>` — 高精度除法，需要浮点元素类型和额外的 `tmp` 操作数

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 高精度模式下 `tmp` 操作数必须提供

**示例：**

```mlir
// 默认精度
pto.trowexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)

// 高精度
pto.trowexpanddiv ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  {precisionType = #pto<div_precision high_precision>}
```

---

### `pto.tcolexpandmax` — 列广播取最大值

```mlir
pto.tcolexpandmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[0, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmin` — 列广播取最小值

```mlir
pto.tcolexpandmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[0, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmul` — 列广播乘法

```mlir
pto.tcolexpandmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandadd` — 列广播加法

```mlir
pto.tcolexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpanddiv` — 列广播除法

```mlir
pto.tcolexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = <precision>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile（被除数） |
| `src1` | `pto.tile_buf` | 每列标量载体（除数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 除法精度模式。默认值为 `#pto<div_precision default>`。
  - `#pto<div_precision default>` — 标准精度除法
  - `#pto<div_precision high_precision>` — 高精度除法，仅当元素类型为 `f16` 或 `f32` 时合法

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandexpdif` — 列广播指数差

```mlir
pto.tcolexpandexpdif ins(<src0>, <src1> : <src0_type>, <src1_type>)
                     outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[0, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体（指数差中的减数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                         v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.tcolexpandsub` — 列广播减法

```mlir
pto.tcolexpandsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体（减数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandadd` — 行广播加法

```mlir
pto.trowexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - `src1` 为 row_major 时：`src1 valid_shape[1] == 32 / sizeof(dtype)`；否则：`src1 valid_shape[1] == 1`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - `src1` 为 row_major 时：`src1 valid_shape[1] == 32 / sizeof(dtype)`；否则：`src1 valid_shape[1] == 1`

**示例：**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandexpdif` — 行广播指数差

```mlir
pto.trowexpandexpdif ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                     outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[i, 0])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（指数差中的减数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例：**

```mlir
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                         v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.trowexpandmax` — 行广播取最大值

```mlir
pto.trowexpandmax ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, 0])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例：**

示例的逐行标量载体使用列优先的一列 Tile。若改用行优先载体，其有效列数必须为 `32/sizeof(dtype)`，f32 为 8。

```mlir
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmin` — 行广播取最小值

```mlir
pto.trowexpandmin ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, 0])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例：**

示例的逐行标量载体使用列优先的一列 Tile。若改用行优先载体，其有效列数必须为 `32/sizeof(dtype)`，f32 为 8。

```mlir
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```
