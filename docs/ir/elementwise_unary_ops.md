# 逐元素单目运算

本节描述了 PTO ISA 逐元素单目操作的指令名称、签名和语义。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用"目标传递风格"（Destination-Passing Style，DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。

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

- [`pto.tabs` — 逐元素绝对值](#ptotabs--逐元素绝对值)
- [`pto.tnot` — 逐元素按位取反](#ptotnot--逐元素按位取反)
- [`pto.trelu` — ReLU 激活](#ptotrelu--relu-激活)
- [`pto.tneg` — 逐元素取负](#ptotneg--逐元素取负)

---

## 操作详解

### `pto.tabs` — 逐元素绝对值

```mlir
pto.tabs ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = abs(src[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 必须具有相同的有效区域。
  - tile 必须使用行优先布局（`blayout=row_major`）。

**示例：**

```mlir
pto.tabs ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tnot` — 逐元素按位取反

```mlir
pto.tnot ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = ~src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为 `i16`。
  - `src` 和 `dst` 元素类型必须一致。
  - tile 必须使用 `loc=vec` 和行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `i32`、`i16` 或 `i8`。
  - `src` 和 `dst` 元素类型必须一致。
  - tile 必须使用 `loc=vec` 和行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。

**示例：**

```mlir
// 仅 Ascend 950PR&950DT系列产品 支持：i32 元素类型
pto.tnot ins(%a : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Atlas A3系列产品 和 Ascend 950PR&950DT系列产品 均支持：i16 元素类型
pto.tnot ins(%a : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trelu` — ReLU 激活

```mlir
pto.trelu ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = max(0, src[i, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `f16`、`f32` 或 `i32`。
  - tile 必须使用 `loc=vec` 和行优先布局（`blayout=row_major`）。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例：**

```mlir
pto.trelu ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tneg` — 逐元素取负

```mlir
pto.tneg ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = -src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 必须具有相同有效区域。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - tile 元素类型必须为 `i8`、`i16`、`i32`、`f16`、`f32` 或 `bf16`。
  - tile 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 至少必须具有相同有效列，具体以 verifier 为准。

**示例：**

```mlir
pto.tneg ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```
