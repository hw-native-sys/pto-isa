# Tile-标量/Tile-立即数操作

本节描述了 PTO ISA 中 Tile 与标量（或立即数）之间的逐元素运算指令族。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用“目标传递风格”（Destination-Passing Style，DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。

这一类操作通常具有如下装配形式：

```mlir
 pto.op ins(%src, %scalar : !pto.tile_buf<...>, <scalar_type>)
        outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 输入 tile 的元素类型与标量类型兼容
- 输入 tile 和输出 tile 的 shape 和 valid-shape 兼容
- 所有 tile 使用 `loc=vec`

---

## 目录

- [`pto.tadds` — Tile-标量加法](#ptotadds--tile-标量加法)
- [`pto.taddsc` — Tile-标量融合加法](#ptotaddsc--tile-标量融合加法)
- [`pto.tands` — Tile-标量位与](#ptotands--tile-标量位与)
- [`pto.tcmps` — Tile-标量比较](#ptotcmps--tile-标量比较)
- [`pto.tdivs` — Tile-标量除法](#ptotdivs--tile-标量除法)
- [`pto.texpands` — 标量广播填充](#ptotexpands--标量广播填充)
- [`pto.tlrelu` — Leaky ReLU](#ptotlrelu--leaky-relu)
- [`pto.tmaxs` — Tile-标量取最大值](#ptotmaxs--tile-标量取最大值)
- [`pto.tmins` — Tile-标量取最小值](#ptotmins--tile-标量取最小值)
- [`pto.tmuls` — Tile-标量乘法](#ptotmuls--tile-标量乘法)
- [`pto.tors` — Tile-标量位或](#ptotors--tile-标量位或)
- [`pto.trems` — Tile-标量取余](#ptotrems--tile-标量取余)
- [`pto.tsels` — Tile-标量掩码选择](#ptotsels--tile-标量掩码选择)
- [`pto.tshls` — Tile-标量左移](#ptotshls--tile-标量左移)
- [`pto.tshrs` — Tile-标量右移](#ptotshrs--tile-标量右移)
- [`pto.tsubs` — Tile-标量减法](#ptotsubs--tile-标量减法)
- [`pto.tsubsc` — Tile-标量融合减法](#ptotsubsc--tile-标量融合减法)
- [`pto.txors` — Tile-标量异或](#ptotxors--tile-标量异或)
- [`pto.tfmods` — Tile-标量浮点取余](#ptotfmods--tile-标量浮点取余)
- [`pto.taxpy` — Tile-标量融合乘加](#ptotaxpy--tile-标量融合乘加)

---

## 操作详解

### `pto.tadds` — Tile-标量加法

```mlir
pto.tadds ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] + scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 加到每个元素上的标量值 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`f16`、`f32` 或 `bf16`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
pto.tadds ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.taddsc` — Tile-标量融合加法

```mlir
pto.taddsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + scalar + src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 标量值 |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1` 和 `dst` 必须为 PTO shaped-like 类型。
  - `src0`、`src1` 和 `dst` 必须具有相同的 shape。
  - 实现使用 `dst valid row` / `dst valid column` 作为迭代域。

**示例：**

```mlir
pto.taddsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tands` — Tile-标量位与

```mlir
pto.tands ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] & scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `AnySignlessInteger` | 整数标量操作数 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 8-bit 或 16-bit 无符号整数类型：`i8`、`i16`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。
  - 源 tile 和目标 tile 不可指向同一块内存。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 8-bit、16-bit 或 32-bit 无符号整数类型：`i8`、`i16`、`i32`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
pto.tands ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tcmps` — Tile-标量比较

```mlir
pto.tcmps ins(<src>, <scalar> {cmpMode = <mode>} : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = (src[i, j] <cmpMode> scalar) ? 1 : 0
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 用于比较的标量值 |
| `dst` | `pto.tile_buf` | 目标 tile buffer（打包谓词掩码） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `cmpMode` — 比较模式。默认值为 `EQ`。
  - `#pto<cmp eq>` — 等于（`==`）
  - `#pto<cmp ne>` — 不等于（`!=`）
  - `#pto<cmp lt>` — 小于（`<`）
  - `#pto<cmp le>` — 小于等于（`<=`）
  - `#pto<cmp gt>` — 大于（`>`）
  - `#pto<cmp ge>` — 大于等于（`>=`）

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 输入 tile 元素类型必须为以下之一：`i32`、`f16`、`f32`、`i16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域边界：`src valid row <= src.rows` 且 `src valid column <= src.cols`。
  - `src` 和 `dst` 必须具有相同的 valid row。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 输入 tile 元素类型必须为以下之一：`i32`、`f16`、`f32`、`i16`、`i8`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域边界：`src valid row <= src.rows`、`src valid column <= src.cols`、`dst valid row <= dst.rows`、`dst valid column <= dst.cols`。
  - `src` 和 `dst` 必须具有相同的 valid row。

**示例：**

```mlir
pto.tcmps ins(%a, %s {cmpMode = #pto<cmp lt>} :
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f16)
          outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tdivs` — Tile-标量除法

```mlir
// Tile / scalar
pto.tdivs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)

// Scalar / tile (reverse mode)
pto.tdivs ins(<scalar>, <src> : <scalar_type>, <src_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] / scalar    (default)
    dst[i, j] = scalar / src[i, j]    (reverse mode)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src/scalar` | `pto.tile_buf` / scalar | 源 tile buffer 或标量（取决于操作数顺序） |
| `scalar/src` | scalar / `pto.tile_buf` | 标量除数或源 tile（取决于操作数顺序） |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 除法精度模式。默认值为 `default`。
  - `#pto<div_precision default>` — 默认精度
  - `#pto<div_precision high_precision>` — 高精度模式（当前仅在 tile 元素类型为 `f16` 或 `f32` 时合法）

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **特殊行为**
  - 除零行为为目标定义；在 Ascend 950PR&950DT系列产品 上 tile/scalar 形式映射为乘倒数，`scalar == 0` 时结果为 `+inf`。

**示例：**

```mlir
// tile / scalar
pto.tdivs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// scalar / tile (reverse mode)
pto.tdivs ins(%s, %a : f32, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.texpands` — 标量广播填充

```mlir
pto.texpands ins(<scalar> : <scalar_type>)
              outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `scalar` | `ScalarType`（signless integer / float）| 要广播的标量值 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用 `loc=vec` 或 `loc=mat`。
  - 若 `loc=vec`：tile 必须使用行优先布局（`blayout=row_major`）；有效区域边界：`valid row <= rows` 且 `valid column <= cols`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32`。
  - tile 必须使用 `loc=vec` 或 `loc=mat`。
  - 若 `loc=vec`：有效区域边界：`valid row <= rows` 且 `valid column <= cols`。

**示例：**

```mlir
pto.texpands ins(%scalar : f32)
              outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.tlrelu` — Leaky ReLU

```mlir
pto.tlrelu ins(<src>, <slope> : <src_type>, <slope_type>)
           outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] > 0 ? src[i, j] : slope * src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `slope` | `F32` | 负半轴斜率系数 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`f16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`0 < valid row <= rows` 且 `0 < valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的 `validRow/validCol`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`f16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - `slope` 必须为 `f32` 类型。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的 `validRow/validCol`。

**示例：**

```mlir
pto.tlrelu ins(%a, %slope : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tmaxs` — Tile-标量取最大值

```mlir
pto.tmaxs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = max(src[i, j], scalar)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 标量值 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
pto.tmaxs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmins` — Tile-标量取最小值

```mlir
pto.tmins ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = min(src[i, j], scalar)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 标量值 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`、`bf16`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
pto.tmins ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmuls` — Tile-标量乘法

```mlir
pto.tmuls ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] * scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 标量乘数 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16`、`f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`、`bf16`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src valid column == dst valid column`。

**示例：**

```mlir
pto.tmuls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tors` — Tile-标量位或

```mlir
pto.tors ins(<src>, <scalar> : <src_type>, <scalar_type>)
         outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] | scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `AnySignlessInteger` | 整数标量操作数 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 8-bit 或 16-bit 无符号整数类型：`i8`、`i16`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。
  - 源 tile 和目标 tile 不可指向同一块内存。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 8-bit、16-bit 或 32-bit 无符号整数类型：`i8`、`i16`、`i32`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
// 仅 Ascend 950PR&950DT系列产品 支持：i32 元素类型
pto.tors ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)

// Atlas A3系列产品 和 Ascend 950PR&950DT系列产品 均支持：i16 元素类型
pto.tors ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>, i16)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
```

---

### `pto.trems` — Tile-标量取余

```mlir
pto.trems ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = fmod(src[i, j], scalar)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 标量除数 |
| `tmp` | `pto.tile_buf` | ISA API 所需的临时 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src`/`dst` 元素类型必须匹配，且必须为 `i32` 或 `f32`。
  - `scalar` 类型必须与 tile 元素类型匹配。
  - `tmp` 元素类型必须与 `dst` 匹配。
  - `src`/`tmp`/`dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的 `validRow/validCol`。
  - `tmp` 必须提供至少 `1` 个 valid row，且 `tmp.validCol >= dst.validCol`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src`/`dst` 元素类型必须匹配，且必须为以下之一：`i32`、`i16`、`f16`、`f32`。
  - `scalar` 类型必须与 tile 元素类型匹配。
  - `tmp` 元素类型必须与 `dst` 匹配。
  - `src`/`tmp`/`dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的 `validRow/validCol`。
  - `tmp` 必须提供至少 `1` 个 valid row，且 `tmp.validCol >= dst.validCol`。

- **特殊行为**
  - 除零行为为目标定义；CPU 模拟器在 debug 构建中会触发 assert。

**示例：**

```mlir
pto.trems ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32,
             !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32,
             v_row=1, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsels` — Tile-标量掩码选择

```mlir
pto.tsels ins(<mask>, <src>, <tmp>, <scalar> : <mask_type>, <src_type>, <tmp_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = mask[i, j] ? src[i, j] : scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `mask` | `pto.tile_buf` | 掩码 tile（选择谓词载体） |
| `src` | `pto.tile_buf` | 掩码位为 true 时选取的源 tile |
| `tmp` | `pto.tile_buf` | 当前 DPS 形式所需的临时 scratch tile |
| `scalar` | `ScalarType` | 掩码位为 false 时选取的标量值 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 16-bit 或 32-bit 类型：`i16`、`i32`、`f16` 或 `f32`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 8-bit、16-bit 或 32-bit 类型：`i8`、`i16`、`i32`、`f16` 或 `f32`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshls` — Tile-标量左移

```mlir
pto.tshls ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] << scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `AnySignlessInteger` | 移位计数 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型，且该元素类型必须为整型；verifier 不限制整数位宽。
  - `scalar` 必须为整型；当其取值在编译期已知时，必须非负。
  - `src` 和 `dst` 的 `valid_shape` 必须落在静态 shape 范围内。
- **目标支持边界**
  - 硬件与 TileLib 模板只支持 `loc=vec` 且 `blayout=row_major` 的 tile。
  - TileLib 模板的元素类型只覆盖 8/16/32 位整数；`i64` 等位宽在 verifier 层可通过，但落到模板选择时会因无合法模板而失败。
  - TileLib 模板要求 `src` 与 `dst` 具有相同的有效形状；仅 `valid_shape` 不同（例如有效列数为 64 与 32 的两个 tile）时同样会因无合法模板而失败。

**示例：**

```mlir
pto.tshls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshrs` — Tile-标量右移

```mlir
pto.tshrs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] >> scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `AnySignlessInteger` | 移位计数 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为以下之一：`i16`、`i32`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为以下之一：`i8`、`i16`、`i32`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
pto.tshrs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsubs` — Tile-标量减法

```mlir
pto.tsubs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] - scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 被减去的标量值 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`、`bf16`。
  - tile 必须使用 `loc=vec`。
  - 有效区域边界：`valid row <= rows` 且 `valid column <= cols`。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
pto.tsubs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsubsc` — Tile-标量融合减法

```mlir
pto.tsubsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - scalar + src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 被减去的标量值 |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer（加回的值） |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1` 和 `dst` 必须为 PTO shaped-like 类型。
  - `src0`、`src1` 和 `dst` 必须具有相同的 rank。
  - 实现使用 `dst valid row` / `dst valid column` 作为迭代域。

**示例：**

```mlir
pto.tsubsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.txors` — Tile-标量异或

```mlir
pto.txors ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j] ^ scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `AnySignlessInteger` | 整数标量操作数 |
| `tmp` | `pto.tile_buf` | 临时 scratch tile（Atlas A2系列产品/Atlas A3系列产品 上用于计算；Ascend 950PR&950DT系列产品 codegen 可能忽略，但 PTO ISA 操作数仍然必需） |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 8-bit 或 16-bit 无符号整数类型：`i8`、`i16`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。
  - DPS 形式接受一个 `tmp` scratch tile。在 Atlas A2系列产品/Atlas A3系列产品 上用于实际计算；在 Ascend 950PR&950DT系列产品 上 codegen 可能忽略，但 PTO ISA 操作数仍然必需。
  - 源 tile 和目标 tile 不可指向同一块内存。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 元素类型必须为 8-bit、16-bit 或 32-bit 无符号整数类型：`i8`、`i16`、`i32`。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src` 和 `dst` 必须具有相同的有效区域：`src valid row == dst valid row` 且 `src valid column == dst valid column`。

**示例：**

```mlir
// 仅 Ascend 950PR&950DT系列产品 支持：i32 元素类型
pto.txors ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Atlas A3系列产品 和 Ascend 950PR&950DT系列产品 均支持：i16 元素类型
pto.txors ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i16,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tfmods` — Tile-标量浮点取余

```mlir
pto.tfmods ins(<src>, <scalar> : <src_type>, <scalar_type>)
           outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = fmod(src[i, j], scalar)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | `ScalarType`（signless integer / float）| 浮点标量除数 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 必须具有相同的元素类型和相同的有效区域。
  - `src` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `scalar` 的类型必须与 tile 元素类型一致。
  - 元素类型仅支持 `i16`、`i32`、`f16`、`f32`。

- **NPU 约束**
  - `src` 和 `dst` 必须是形状兼容的 `loc=vec` tile buffer。
  - 除零行为为未定义行为。

**示例：**

```mlir
pto.tfmods ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
           outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.taxpy` — Tile-标量融合乘加

```mlir
pto.taxpy ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = dst[i, j] + src[i, j] * scalar
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `scalar` | 标量类型（`f16`/`bf16`/`f32`） | 乘法因子 |
| `dst` | `pto.tile_buf` | 目标 tile buffer（同时作为累加输入） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 的元素类型必须为 `f16` 或 `f32`。
  - `src` 和 `dst` 的元素类型必须一致，或使用 `src=f16`、`dst=f32` 的扩展组合。
  - 所有 tile 必须使用 `loc=vec`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 的元素类型必须为 `f16`、`bf16` 或 `f32`。
  - `src` 和 `dst` 的元素类型必须一致，或使用 `src=f16`、`dst=f32` 的扩展组合。
  - 所有 tile 必须使用 `loc=vec`。

**示例：**

```mlir
pto.taxpy
    ins(%src, %alpha :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        f32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                              v_row=16, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```
