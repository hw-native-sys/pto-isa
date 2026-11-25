# 排序

本节描述了 PTO ISA 中排序操作的指令名称、签名和语义。所有操作均作用于本地 `tile_buf`。

通用汇编形式（generic assembly form）：

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## 目录

- [`pto.tsort32` — 32 元素块排序](#ptotsort32--32-元素块排序)
- [`pto.tmrgsort` — 归并排序](#ptotmrgsort--归并排序)
- [`pto.thistogram` — 逐行直方图累加](#ptothistogram--逐行直方图累加)

---

## 操作详解

### `pto.tsort32` — 32 元素块排序

```mlir
// 基本形式
pto.tsort32 ins(<src>, <idx> : <src_type>, <idx_type>)
            outs(<dst> : <dst_type>)

// 带临时 tile 的形式
pto.tsort32 ins(<src>, <idx>, <tmp> : <src_type>, <idx_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
dst = sort(src, idx)
// 对 src 中固定 32 元素块进行排序
// idx 为索引 tile，与 src 的值一起按排序结果进行排列
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 输入值 tile |
| `idx` | `pto.tile_buf` | 输入索引 tile，与 `src` 一起排列 |
| `tmp` | `pto.tile_buf` | 临时 scratch tile（可选） |
| `dst` | `pto.tile_buf` | 输出 tile，存储排序后的值-索引对 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 元素类型必须一致，且为 `f16` 或 `f32`。
  - `idx` 元素类型必须为 32 位无符号整数（MLIR 中表示为 `ui32` 或 `i32`）。
  - `src`、`dst` 和 `idx` 必须为 `loc=vec` 和 `blayout=row_major`。

**示例：**

```mlir
pto.tsort32 ins(%src, %idx :
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%dst0 : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=64,
                    v_row=1, v_col=64, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

### `pto.tmrgsort` — 归并排序

`pto.tmrgsort` 有两种格式：单列表归并排序（format1）和多列表归并排序（format2）。

```mlir
// format1：单列表归并排序
pto.tmrgsort ins(<src>, <blockLen> : <src_type>, <int_type>)
             outs(<dst> : <dst_type>)

// format2：多列表归并排序（2~4 路）
pto.tmrgsort ins(<src0>, <src1>, ... , <tmp> {exhausted = <bool>} :
                 <src_type>, <src_type>, ... , <tmp_type>)
             outs(<dst>, <executed> : <dst_type>, vector<4xi16>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
format1：
    dst = merge_sort(src, blockLen)
    // 对 src 中每 blockLen*4 个元素为一组，按 blockLen 长度的有序子块进行归并排序

format2：
    dst = merge(src0, src1, ...)
    // 将 2~4 个已排序的输入列表归并为单个有序输出
    executed = 每路消耗的元素计数
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` / `src0..src3` | `pto.tile_buf` | 输入 tile，format2 支持 2~4 个源 |
| `blockLen` | `AnyInteger` | format1 中的块长度 |
| `dst` | `pto.tile_buf` | 输出 tile |
| `tmp` | `pto.tile_buf` | format2 中的临时 tile（仅 format2） |
| `executed` | `vector<4xi16>` | format2 中输出的每路消耗计数向量（仅 format2） |

**返回值：** 无 SSA 返回。以 DPS 的形式写入 `dst`；format2 还会写入 `executed`。

**属性：**

- `exhausted` — 是否使用耗尽模式（format2 中使用）。默认值为 `false`。
  - `false` — 非耗尽模式
  - `true` — 耗尽模式，可接受额外的源操作数

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - format1：元素类型必须为 `f16` 或 `f32`，且 `src` 和 `dst` 元素类型必须一致。`src` 和 `dst` 必须为 rank-2，且 `rows == 1`（数据存储在单行中）。`src` 和 `dst` 的 `cols` 必须一致。`blockLen` 必须大于 0 且为 64 的整数倍。`src` 有效列数必须为 `blockLen * 4` 的整数倍。`repeatTimes = src 有效列数 / (blockLen * 4)` 必须在 `[1, 255]` 范围内。
  - format2：接受 2 路、3 路或 4 路归并。`dst` 和 `tmp` 元素类型和 shape 必须一致。所有 `src` 的元素类型必须与 `dst`/`tmp` 一致，且为 `f16` 或 `f32`。所有 tile 必须为 rank-2，且 `rows == 1`。`tmp.cols >= dst.cols`。`executed` 必须为 `vector<4xi16>` 类型。

**示例：**

```mlir
// format2：2 路归并排序
pto.tmrgsort ins(%src0, %src1, %tmp2 {exhausted = false} :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128,
                     v_row=1, v_col=128, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128,
                     v_row=1, v_col=128, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=256,
                     v_row=1, v_col=256, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst2, %ex :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=256,
                     v_row=1, v_col=256, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 vector<4xi16>)
```

---

### `pto.thistogram` — 逐行直方图累加

```mlir
pto.thistogram ins(<src>, <idx> : <src_type>, <idx_type>)
               outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持

**语义：**

```text
For each row i:
    bin = select_bin(src[i, :], idx[i, 0], isMSB)
    dst[i, bin] += 1
// 逐行对源 tile 按索引确定的位段进行 256-bin 直方图累加
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源数据 tile（`ui16`） |
| `idx` | `pto.tile_buf` | 索引 tile，指定位选择（`ui8`，单列） |
| `dst` | `pto.tile_buf` | 目标直方图 tile（`ui32`，列数为 256） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `isMSB` — 是否选择高位字节。默认值为 `true`。
  - `true` — 选择高 8 位
  - `false` — 选择低 8 位

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 不支持，仅 Ascend 950PR&950DT系列产品 可用。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 元素类型必须为 `ui16`。
  - `idx` 元素类型必须为 `ui8`，且为单列（cols=1）。
  - `dst` 元素类型必须为 `ui32`，且列数为 256。
  - 所有 tile 必须使用 `loc=vec`。

**示例：**

```mlir
pto.thistogram
    ins(%src, %idx :
        !pto.tile_buf<loc=vec, dtype=ui16, rows=32, cols=32,
                      v_row=8, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=1,
                      v_row=8, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=32, cols=256,
                              v_row=8, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```
