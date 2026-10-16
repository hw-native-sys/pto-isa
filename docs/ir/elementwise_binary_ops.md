# 逐元素双目运算

本节描述了 PTO ISA 逐元素双目操作的指令名称、签名和语义。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用"目标传递风格"（Destination-Passing Style，DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。

这一类操作通常具有如下装配形式：

```mlir
 pto.op ins(%lhs, %rhs : !pto.tile_buf<...>, !pto.tile_buf<...>)
        outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 输入 tile 的元素类型兼容
- 输入 tile 的 shape 和 valid-shape 兼容
- 输出 tile 的类型与目标语义匹配

---

## 目录

- [`pto.tadd` — 逐元素加法](#ptotadd--逐元素加法)
- [`pto.tand` — 逐元素按位与](#ptotand--逐元素按位与)
- [`pto.tor` — 逐元素按位或](#ptotor--逐元素按位或)
- [`pto.tsub` — 逐元素减法](#ptotsub--逐元素减法)
- [`pto.tmul` — 逐元素乘法](#ptotmul--逐元素乘法)
- [`pto.tmin` — 逐元素取最小值](#ptotmin--逐元素取最小值)
- [`pto.tmax` — 逐元素取最大值](#ptotmax--逐元素取最大值)
- [`pto.tcmp` — 逐元素比较](#ptotcmp--逐元素比较)
- [`pto.tshl` — 逐元素左移](#ptotshl--逐元素左移)
- [`pto.tshr` — 逐元素右移](#ptotshr--逐元素右移)
- [`pto.txor` — 逐元素按位异或](#ptotxor--逐元素按位异或)
- [`pto.tsel` — 掩码选择](#ptotsel--掩码选择)
- [`pto.tprelu` — 参数化 ReLU](#ptotprelu--参数化-relu)
- [`pto.taddc` — 三元逐元素加法](#ptotaddc--三元逐元素加法)
- [`pto.tsubc` — 三元逐元素减加](#ptotsubc--三元逐元素减加)

---

## 操作详解

### `pto.tadd` — 逐元素加法

```mlir
pto.tadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局（`blayout=row_major`）。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为以下之一：`i32`、`f32`、`i16`、`f16`、`bf16` 或 `i8`。
  - tile 必须使用行优先布局（`blayout=row_major`）。

**示例：**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tand` — 逐元素按位与

```mlir
pto.tand ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] & src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8` 或 `i16`。
  - 三个 tile 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - 三个 tile 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例：**

```mlir
pto.tand ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tor` — 逐元素按位或

```mlir
pto.tor ins(<src0>, <src1> : <src0_type>, <src1_type>)
        outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] | src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8` 或 `i16`。
  - 三个 tile 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - 三个 tile 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例：**

```mlir
pto.tor ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>,
            !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
```

---

### `pto.tsub` — 逐元素减法

```mlir
pto.tsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被减数 tile buffer |
| `src1` | `pto.tile_buf` | 减数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`i8`、`f32` 或 `f16`。
  - tile 必须使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例：**

```mlir
pto.tsub ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmul` — 逐元素乘法

```mlir
pto.tmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用 `loc=vec` 和行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `i32`、`f32`、`i16` 或 `f16`。
  - tile 必须使用 `loc=vec` 和行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例：**

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmin` — 逐元素取最小值

```mlir
pto.tmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`i8`、`f32`、`f16` 或 `bf16`。
  - tile 必须使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例：**

```mlir
pto.tmin ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
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

### `pto.tmax` — 逐元素取最大值

```mlir
pto.tmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`i8`、`f32` 或 `f16`。
  - tile 必须使用行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例：**

```mlir
pto.tmax ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
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

### `pto.tcmp` — 逐元素比较

```mlir
pto.tcmp ins(<src0>, <src1> {cmpMode = <mode>} : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = (src0[i, j] <cmpMode> src1[i, j]) ? 1 : 0
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个比较输入 |
| `src1` | `pto.tile_buf` | 第二个比较输入 |
| `dst` | `pto.tile_buf` | 目标 mask tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `cmpMode` — 比较模式。默认值为 `eq`。
  - `#pto<cmp eq>` — 等于
  - `#pto<cmp ne>` — 不等于
  - `#pto<cmp lt>` — 小于
  - `#pto<cmp le>` — 小于等于
  - `#pto<cmp gt>` — 大于
  - `#pto<cmp ge>` — 大于等于

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 输入元素类型必须为 `i32`、`f16` 或 `f32`。
  - 输出 mask 元素类型必须为 `i8`。
  - `src0`、`src1` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src0/src1/dst` 有效区域一致。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 输入元素类型必须为 `i32`、`i16`、`i8`、`f32`、`f16` 或 `bf16`。

**示例：**

```mlir
pto.tcmp ins(%a, %b {cmpMode = #pto<cmp lt>} :
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshl` — 逐元素左移

```mlir
pto.tshl ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] << src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 待左移的源 tile buffer |
| `src1` | `pto.tile_buf` | 左移位数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0` 和 `src1` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例：**

```mlir
pto.tshl ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshr` — 逐元素右移

```mlir
pto.tshr ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] >> src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 待右移的源 tile buffer |
| `src1` | `pto.tile_buf` | 右移位数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src0` 和 `src1` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例：**

```mlir
pto.tshr ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.txor` — 逐元素按位异或

```mlir
pto.txor ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] ^ src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 tile buffer；Ascend 950PR&950DT系列产品 中可复用占位 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1`、`tmp` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8` 或 `i16`。
  - 四个 tile 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1`、`tmp` 必须分别与 `dst` 具有相同有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。
  - `tmp` 在 Ascend 950PR&950DT系列产品 路径中仅作为占位参数。

**示例：**

```mlir
// Atlas A2系列产品/Atlas A3系列产品：需要独立的 tmp tile
pto.txor ins(%src0, %src1, %tmp :
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Ascend 950PR&950DT系列产品：tmp 可复用 dst 作为占位
pto.txor ins(%src0, %src1, %dst :
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsel` — 掩码选择

```mlir
pto.tsel ins(<mask>, <src0>, <src1>, <tmp> : <mask_type>, <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = mask[i, j] ? src0[i, j] : src1[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `mask` | `pto.tile_buf` | 谓词 mask tile buffer |
| `src0` | `pto.tile_buf` | mask 为真时选择的源 tile buffer |
| `src1` | `pto.tile_buf` | mask 为假时选择的源 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 scratch tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i16`、`i32`、`f16`、`bf16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局（`blayout=row_major`）。

- **临时 tile**
  - `tmp` 是当前 DPS/ISA 形式要求的临时 scratch tile。

**示例：**

```mlir
// 仅 Ascend 950PR&950DT系列产品 支持：tmp 使用 2 字节元素类型（f16）
pto.tsel ins(%mask, %a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Atlas A3系列产品 和 Ascend 950PR&950DT系列产品 均支持：tmp 使用 4 字节元素类型（f32）
pto.tsel ins(%mask, %a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tprelu` — 参数化 ReLU

```mlir
pto.tprelu ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] > 0 ? src0[i, j] : src1[i, j] * src0[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 输入激活 tile buffer |
| `src1` | `pto.tile_buf` | 逐元素负半轴斜率 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 tile buffer；Ascend 950PR&950DT系列产品 中可复用占位 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `dst/src0/src1` 元素类型必须一致，且必须为 `f16` 或 `f32`。
  - `tmp` 元素类型使用 `ui8`。令 R/C 为目标有效行列数：临时空间物理行数至少 R+1，有效列数至少 `ceil(C/8)`，总容量至少 `(R+1)*align_up(ceil(C/8),32)` 字节。目标有效区域必须为静态尺寸。
  - 所有相关 tile 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。
  - Atlas A3系列产品 中两个源 tile、目标 tile、临时空间不得内存重叠。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst/src0/src1` 元素类型必须一致，且必须为 `f16` 或 `f32`。
  - 所有相关 tile 必须使用行优先布局（`blayout=row_major`）。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。
  - `tmp` 在 Ascend 950PR&950DT系列产品 路径中可作为占位参数。

**示例：**

```mlir
// Atlas A2系列产品/Atlas A3系列产品：需要独立的 tmp tile（元素类型为 ui8）
pto.tprelu ins(%a, %slopes, %tmp :
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=ui8, rows=17, cols=32,
               v_row=17, v_col=32, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

Ascend 950PR&950DT系列产品 使用目标 Tile 作为占位参数：

```mlir
// Ascend 950PR&950DT系列产品：tmp 可复用 dst 作为占位
pto.tprelu ins(%a, %slopes, %c :
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### `pto.taddc` — 三元逐元素加法

```mlir
pto.taddc ins(<src0>, <src1>, <src2> : <src0_type>, <src1_type>, <src2_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j] + src2[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `src2` | `pto.tile_buf` | 第三个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现约束**
  - 实现以 `dst` 的有效行/列作为迭代域。
  - `src0`、`src1`、`src2` 与 `dst` 的具体类型和有效区域约束以 verifier 为准。

**示例：**

```mlir
pto.taddc ins(%a, %b, %c :
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tsubc` — 三元逐元素减加

```mlir
pto.tsubc ins(<src0>, <src1>, <src2> : <src0_type>, <src1_type>, <src2_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j] + src2[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 减数 tile buffer |
| `src2` | `pto.tile_buf` | 加数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现约束**
  - 实现以 `dst` 的有效行/列作为迭代域。
  - `src0`、`src1`、`src2` 与 `dst` 的具体类型和有效区域约束以 verifier 为准。

**示例：**

```mlir
pto.tsubc ins(%a, %b, %c :
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```
