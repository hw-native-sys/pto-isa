# 数据类型转换

本节描述了 PTO ISA 中数据类型转换操作的指令名称、签名和语义。所有操作均作用于本地 `tile_buf`。

通用汇编形式（generic assembly form）：

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## 目录

- [`pto.tcvt` — 逐元素类型转换](#ptotcvt--逐元素类型转换)
- [`pto.tquant` — Tile 量化](#ptotquant--tile-量化)
- [`pto.tdequant` — Tile 反量化](#ptotdequant--tile-反量化)

---

## 操作详解

### `pto.tcvt` — 逐元素类型转换

```mlir
pto.tcvt ins(<src>[, <tmp>] {rmode = <round_mode>, satmode = <saturation_mode>}
            : <src_type>[, <tmp_type>])
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = saturate(cast(src[i, j], rmode), satmode)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `tmp` | `pto.tile_buf`（可选） | 临时 scratch tile；仅 Atlas A2系列产品/Atlas A3系列产品 上 `satmode=OFF` 且为 `f32 -> i16`、`f16 -> i16`、`f16 -> i8` 转换时必须提供 |
| `dst` | `pto.tile_buf` | 目标 tile buffer，元素类型可不同于 `src` |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `rmode` — 舍入模式。默认值为 `CAST_RINT`。
  - `#pto<round_mode NONE>` — 无舍入
  - `#pto<round_mode RINT>` — 四舍五入到最近偶数
  - `#pto<round_mode ROUND>` — 四舍五入
  - `#pto<round_mode FLOOR>` — 向下取整
  - `#pto<round_mode CEIL>` — 向上取整
  - `#pto<round_mode TRUNC>` — 截断
  - `#pto<round_mode ODD>` — 向最近奇数舍入
  - `#pto<round_mode CAST_RINT>` — 类型转换默认舍入
- `satmode` — 饱和模式，控制舍入后是否按目标类型范围 clamp。默认值为 `ON`。
  - `#pto<saturation_mode ON>` — 启用饱和
  - `#pto<saturation_mode OFF>` — 关闭饱和

**约束：**

- **通用检查**
  - `src` 和 `dst` 必须是兼容的 tile buffer。
  - `src` 与 `dst` 的逻辑范围和有效区域必须兼容。

- **Atlas A2系列产品/Atlas A3系列产品 与 Ascend 950PR&950DT系列产品 低精度限制**
  - Atlas A2系列产品/Atlas A3系列产品 不支持低精度 `tcvt` 操作数。
  - Ascend 950PR&950DT系列产品 仅接受实现中列出的低精度转换对，例如 `f32 -> f8E4M3*`、`f32 -> f8E5M2*`、`f32 -> !pto.hif8`、`f16 -> !pto.hif8`、`bf16 <-> !pto.f4E1M2x2`、`bf16 <-> !pto.f4E2M1x2`、`f8E4M3* -> f32`、`f8E5M2* -> f32`、`!pto.hif8 -> f32`。
  - 非低精度类型对沿用目标定义的转换行为。

- **可选 tmp 操作数**
  - `tmp` 为可选操作数；仅在 Atlas A2系列产品/Atlas A3系列产品 上、`satmode=OFF` 且转换对为 `f32 -> i16`、`f16 -> i16` 或 `f16 -> i8` 时才需要。
  - 默认的 `satmode=ON` 不需要 `tmp`；Ascend 950PR&950DT系列产品 上的窄化转换也不需要 `tmp`。
  - 省略 `tmp` 时，由实现自动补写临时空间。

**硬件：**

- PIPE_V

**示例：**

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>, satmode = #pto<saturation_mode ON>} :
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tquant` — Tile 量化

```mlir
pto.tquant ins(<src>, <fp> : !pto.tile_buf, !pto.tile_buf)
           outs(<dst> : !pto.tile_buf) {quant_type = <quant_type>}
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = Quantize(src[i, j]; fp, quant_type)
```

其中 `fp` 为缩放因子 tile（通常为单列或单行）。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（f32 类型） |
| `fp` | `pto.tile_buf` | 缩放因子 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer（整数类型） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `quant_type` — 量化类型。
  - `#pto<quant_type INT8_SYM>` — 对称量化，dst 为 `i8`
  - `#pto<quant_type INT8_ASYM>` — 非对称量化，dst 为 `ui8`

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - src 必须为 `f32` 类型
  - 可选 `offset` 的元素类型必须为 `f32`
  - src 与 dst 的有效 shape 必须一致
  - Atlas A2系列产品/Atlas A3系列产品: src 和 dst 必须使用 `blayout=row_major`

**示例：**

```mlir
pto.tquant
    ins(%src, %fp :
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
    {quant_type = #pto<quant_type INT8_SYM>}
```

---

### `pto.tdequant` — Tile 反量化

```mlir
pto.tdequant ins(<src>, <scale>, <offset> : !pto.tile_buf, !pto.tile_buf, !pto.tile_buf)
             outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each row i:
    For each column j:
        dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（整数类型，i8 或 i16） |
| `scale` | `pto.tile_buf` | 缩放因子 tile buffer（通常为单列） |
| `offset` | `pto.tile_buf` | 偏移 tile buffer（通常为单列） |
| `dst` | `pto.tile_buf` | 目标 tile buffer（f32 类型） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 元素类型必须为 8 位或 16 位整数
  - `scale`、`offset` 和 `dst` 的元素类型必须为 `f32`
  - `src`、`scale`、`offset` 和 `dst` 都必须是合法的 rank-2 `tile_buf`
  - Atlas A2系列产品/Atlas A3系列产品 额外要求 `src`、`dst` 使用 row-major 布局；Ascend 950PR&950DT系列产品 无此附加布局限制

**示例：**

```mlir
pto.tdequant
    ins(%src, %scale, %offset :
        !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```
