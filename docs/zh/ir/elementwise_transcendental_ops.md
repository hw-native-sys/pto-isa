# 逐元素算术与超越函数

本节描述了 PTO ISA 逐元素算术与超越函数操作的指令名称、签名和语义。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用"目标传递风格"（Destination-Passing Style，DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。

这一类操作通常具有如下装配形式：

```mlir
 pto.op ins(%src : !pto.tile_buf<...>)
        outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 输入 tile 的元素类型兼容
- 输入 tile 的 shape 和 valid-shape 兼容
- 输出 tile 的类型与目标语义匹配

---

## 目录

- [`pto.tdiv` — 逐元素除法](#ptotdiv--逐元素除法)
- [`pto.tlog` — 逐元素自然对数](#ptotlog--逐元素自然对数)
- [`pto.trecip` — 逐元素倒数](#ptotrecip--逐元素倒数)
- [`pto.trsqrt` — 逐元素倒数平方根](#ptotrsqrt--逐元素倒数平方根)
- [`pto.tsqrt` — 逐元素平方根](#ptotsqrt--逐元素平方根)
- [`pto.texp` — 逐元素指数函数](#ptotexp--逐元素指数函数)
- [`pto.trem` — 逐元素取余（带临时 tile）](#ptotrem--逐元素取余带临时-tile)
- [`pto.tfmod` — 逐元素取余（无需临时 tile）](#ptotfmod--逐元素取余无需临时-tile)

---

## 操作详解

### `pto.tdiv` — 逐元素除法

```mlir
pto.tdiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
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
    dst[i, j] = src0[i, j] / src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被除数 tile buffer |
| `src1` | `pto.tile_buf` | 除数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 除法精度模式。默认值为 `default`。
  - `#pto<div_precision default>` — 默认精度
  - `#pto<div_precision high_precision>` — 高精度

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为 `f16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须元素类型一致，并使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内，且三个 tile 有效区域一致。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须元素类型一致，并使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内，且三个 tile 有效区域一致。

- **除零行为**
  - 除零行为由目标实现定义。

**示例：**

```mlir
pto.tdiv ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tlog` — 逐元素自然对数

```mlir
pto.tlog ins(<src> : <src_type>)
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
    dst[i, j] = ln(src[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 对数精度模式。默认值为 `default`。
  - `#pto<log_precision default>` — 默认精度
  - `#pto<log_precision high_precision>` — 高精度

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局（`blayout=row_major`）。

- **定义域行为**
  - 对 `src <= 0` 等输入的行为由目标实现定义。

**示例：**

```mlir
pto.tlog ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trecip` — 逐元素倒数

```mlir
pto.trecip ins(<src> : <src_type>)
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
    dst[i, j] = 1.0 / src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 倒数精度模式。默认值为 `default`。
  - `#pto<recip_precision default>` — 默认精度
  - `#pto<recip_precision high_precision>` — 高精度

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - tile 必须使用 `loc=vec` 和行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 必须具有相同的有效区域。
  - Atlas A3系列产品 的 `TRECIP` 指令不支持源 tile 与目标 tile 使用同一段内存。

- **除零行为**
  - 除零行为由目标实现定义；CPU simulator 的 debug 构建可能 assert。

**示例：**

```mlir
pto.trecip ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### `pto.trsqrt` — 逐元素倒数平方根

```mlir
// 默认精度（无 tmp）
pto.trsqrt ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)

// 高精度（需提供 tmp）
pto.trsqrt ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
           {precisionType = #pto<rsqrt_precision high_precision>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = 1.0 / sqrt(src[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `tmp` | `pto.tile_buf`（可选） | 临时 tile buffer；仅 `HighPrecision` 模式下必须提供，至少 32 字节 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 倒数平方根精度模式。默认值为 `default`。
  - `#pto<rsqrt_precision default>` — 默认精度
  - `#pto<rsqrt_precision high_precision>` — 高精度，需提供 `tmp` 操作数

**约束：**

- **可选 tmp 操作数**
  - `tmp` 是可选的临时 tile buffer（`Optional<PTODpsType>`），仅在 `precisionType = HighPrecision` 时必须提供。
  - 当 `precisionType` 为默认值 `Default` 时，`tmp` 可省略。
  - `tmp` 必须位于 `loc=vec`，且至少提供 32 字节的存储空间。

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局（`blayout=row_major`）。

- **定义域行为**
  - 对 `src == 0` 或负数等输入的行为由目标实现定义。

**示例：**

```mlir
// 默认精度，无 tmp
pto.trsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)

// 高精度，带 tmp
pto.trsqrt ins(%a, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
               v_row=1, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           {precisionType = #pto<rsqrt_precision high_precision>}
```

---

### `pto.tsqrt` — 逐元素平方根

```mlir
pto.tsqrt ins(<src> : <src_type>)
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
    dst[i, j] = sqrt(src[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 平方根精度模式。默认值为 `default`。所有可选值：`#pto<sqrt_precision default>`（默认精度）、`#pto<sqrt_precision high_precision>`（高精度）。

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局（`blayout=row_major`）。

- **定义域行为**
  - 对负数输入等情况的行为由目标实现定义。

**示例：**

```mlir
pto.tsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.texp` — 逐元素指数函数

```mlir
pto.texp ins(<src> : <src_type>)
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
    dst[i, j] = exp(src[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 指数精度模式。默认值为 `default`。所有可选值：`#pto<exp_precision default>`（默认精度）、`#pto<exp_precision high_precision>`（高精度）。

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局（`blayout=row_major`）。

**示例：**

```mlir
pto.texp ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trem` — 逐元素取余（带临时 tile）

```mlir
pto.trem ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
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
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被取余 tile buffer |
| `src1` | `pto.tile_buf` | 除数 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 取余精度模式。默认值为 `default`。所有可选值：`#pto<rem_precision default>`（默认精度）、`#pto<rem_precision high_precision>`（高精度）。

**约束：**

- **通用检查**
  - 实现使用 `dst` 的有效行/列作为迭代域。
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - `tmp` 元素类型必须与 `dst` 一致。
  - `src0`、`src1`、`tmp` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 和 `dst` 必须具有相同有效区域。
  - `tmp` 至少提供 1 个有效行，且 `tmp.validCol >= dst.validCol`。

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0/src1/dst` 元素类型必须为 `i32` 或 `f32`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0/src1/dst` 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。

**示例：**

```mlir
pto.trem ins(%a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tfmod` — 逐元素取余（无需临时 tile）

```mlir
pto.tfmod ins(<src0>, <src1> : <src0_type>, <src1_type>)
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
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被取余 tile buffer |
| `src1` | `pto.tile_buf` | 除数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — fmod 精度模式。默认值为 `default`。所有可选值：`#pto<fmod_precision default>`（默认精度）、`#pto<fmod_precision high_precision>`（高精度）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - 三个 tile 必须满足二元 tile 操作的形状/有效区域一致性检查。
  - tile 必须使用行优先布局（`blayout=row_major`）。

**示例：**

```mlir
pto.tfmod ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```
