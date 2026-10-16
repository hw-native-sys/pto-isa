# Union计算

本节描述了 PTO ISA 中Union计算操作的指令名称、签名和语义。所有操作均作用于本地 `tile_buf`。

通用汇编形式（generic assembly form）：

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## 目录

- [`pto.tpartadd` — 部分逐元素加法](#ptotpartadd--部分逐元素加法)
- [`pto.tpartmul` — 部分逐元素乘法](#ptotpartmul--部分逐元素乘法)
- [`pto.tpartmax` — 部分逐元素取最大值](#ptotpartmax--部分逐元素取最大值)
- [`pto.tpartmin` — 部分逐元素取最小值](#ptotpartmin--部分逐元素取最小值)
- [`pto.tpartargmax` — 部分逐元素取最大值及索引](#ptotpartargmax--部分逐元素取最大值及索引)
- [`pto.tpartargmin` — 部分逐元素取最小值及索引](#ptotpartargmin--部分逐元素取最小值及索引)

---

## 操作详解

### `pto.tpartadd` — 部分逐元素加法

```mlir
pto.tpartadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] + src1[i, j]

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 仅支持特定的部分有效区域模式（例如一个源等于 `dst`，另一个源在 valid-rows 或 valid-cols 上小于 `dst`）。

**示例：**

```mlir
pto.tpartadd ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmul` — 部分逐元素乘法

```mlir
pto.tpartmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] * src1[i, j]

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 要求 `src0` 和 `src1` 的有效区域在两个维度上均不超过 `dst` 的有效区域。

**示例：**

```mlir
pto.tpartmul ins(%src0, %src1 :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmax` — 部分逐元素取最大值

```mlir
pto.tpartmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j) in the valid region:
    dst[i, j] = max(src0[i, j], src1[i, j])

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求 `src0` 和 `src1` 的有效区域在两个维度上均不超过 `dst` 的有效区域。

**示例：**

```mlir
pto.tpartmax ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmin` — 部分逐元素取最小值

```mlir
pto.tpartmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j) in the valid region:
    dst[i, j] = min(src0[i, j], src1[i, j])

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求 `src0` 和 `src1` 的有效区域在两个维度上均不超过 `dst` 的有效区域。

**示例：**

```mlir
pto.tpartmin ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartargmax` — 部分逐元素取最大值及索引

```mlir
pto.tpartargmax ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                    : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**语义：**

```text
For each element (i, j):
    if src0[i, j] >= src1[i, j]:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一组值 tile |
| `src1` | `pto.tile_buf` | 第二组值 tile |
| `src0Idx` | `pto.tile_buf` | 第一组索引 tile（`ui32`） |
| `src1Idx` | `pto.tile_buf` | 第二组索引 tile（`ui32`） |
| `dst` | `pto.tile_buf` | 输出值 tile |
| `dstIdx` | `pto.tile_buf` | 输出索引 tile（`ui32`） |

**返回值：** 无。以 DPS 的形式写入 `dst` 和 `dstIdx`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 所有值 tile（src0、src1、dst）的元素类型必须一致，为 `f16` 或 `f32`。
  - 所有索引 tile（src0Idx、src1Idx、dstIdx）的元素类型必须为 `ui32`。
  - 所有 tile 必须使用 `loc=vec`。

**示例：**

```mlir
pto.tpartargmax
    ins(%src0, %src1, %src0_idx, %src1_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst, %dst_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tpartargmin` — 部分逐元素取最小值及索引

```mlir
pto.tpartargmin ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                    : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**语义：**

```text
For each element (i, j):
    if src0[i, j] <= src1[i, j]:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一组值 tile |
| `src1` | `pto.tile_buf` | 第二组值 tile |
| `src0Idx` | `pto.tile_buf` | 第一组索引 tile（`ui32`） |
| `src1Idx` | `pto.tile_buf` | 第二组索引 tile（`ui32`） |
| `dst` | `pto.tile_buf` | 输出值 tile |
| `dstIdx` | `pto.tile_buf` | 输出索引 tile（`ui32`） |

**返回值：** 无。以 DPS 的形式写入 `dst` 和 `dstIdx`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 所有值 tile（src0、src1、dst）的元素类型必须一致，为 `f16` 或 `f32`。
  - 所有索引 tile（src0Idx、src1Idx、dstIdx）的元素类型必须为 `ui32`。
  - 所有 tile 必须使用 `loc=vec`。

**示例：**

```mlir
pto.tpartargmin
    ins(%src0, %src1, %src0_idx, %src1_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst, %dst_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
```
