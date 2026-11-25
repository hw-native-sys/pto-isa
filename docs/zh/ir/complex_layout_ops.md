# 布局变换

本节描述了 PTO ISA 中布局变换操作的指令名称、签名和语义。所有操作均作用于本地 `tile_buf`。

通用汇编形式（generic assembly form）：

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## 目录

- [`pto.treshape` — Tile 形状重解释](#ptotreshape--tile-形状重解释)
- [`pto.tconcat` — 列方向 Tile 拼接](#ptotconcat--列方向-tile-拼接)
- [`pto.tconcatidx` — 索引控制列拼接](#ptotconcatidx--索引控制列拼接)
- [`pto.textract` — 子 Tile 提取](#ptotextract--子-tile-提取)
- [`pto.textract` 的 `fp` 形式](#ptotextract-的-fp-形式)
- [`pto.tinsert` — 子 Tile 插入](#ptotinsert--子-tile-插入)
- [`pto.tinsert` 的 `fp` 形式](#ptotinsert-的-fp-形式)
- [`pto.tgather` — 聚集/选择元素](#ptotgather--聚集选择元素)
- [`pto.tgatherb` — 按字节偏移聚集 32 字节块](#ptotgatherb--按字节偏移聚集-32-字节块)
- [`pto.tscatter` — 散射元素](#ptotscatter--散射元素)

---

## 操作详解

### `pto.treshape` — Tile 形状重解释

```mlir
%view = pto.treshape <src> : <src_type> -> <result_type>
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
view = reinterpret_view(src, result_type)
// view 与 src 共享同一块底层存储，不分配或复制元素。
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |

**返回值：** 指定结果类型的 `!pto.tile_buf` 视图。通过源或结果视图写入共享存储后，另一视图也会观察到该变化；它不是独立的数据副本。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 源和结果视图必须使用相同的存储位置：`src.loc == view.loc`
  - 源和结果必须具有静态物理尺寸，且总字节大小相等
  - 不支持有装箱（boxed）与无装箱（non-boxed）layout 之间的转换

**示例：**

```mlir
%view = pto.treshape %src
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                             v_row=16, v_col=32, blayout=row_major,
                             slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                              v_row=32, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>
```

---

### `pto.tconcat` — 列方向 Tile 拼接

```mlir
pto.tconcat ins(<src0>, <src1> : !pto.tile_buf, !pto.tile_buf)
            outs(<dst> : !pto.tile_buf)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    dst[i, 0:C0) = src0[i, 0:C0)
    dst[i, C0:C0+C1) = src1[i, 0:C1)
```

其中 C0 为 src0 的列数，C1 为 src1 的列数。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer（左侧） |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer（右侧） |
| `dst` | `pto.tile_buf` | 目标 tile buffer（拼接结果） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - src0、src1 和 dst 必须使用相同的元素类型，且为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`、`bf16`
  - 所有 tile 必须使用 `loc=vec`
  - 三个 tile 必须为 rank-2，且 src0、src1 的有效行数必须与 dst 的有效行数相同
  - src0 的有效列数 + src1 的有效列数 <= dst 的列数
  - 拼接会改变列方向长度，因此三者的物理 static shape 不要求完全相同

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 同 Atlas A2系列产品/Atlas A3系列产品 要求，额外要求所有 tile 必须使用 `blayout=row_major`

**示例：**

```mlir
pto.tconcat
    ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                               v_row=32, v_col=16, blayout=row_major,
                               slayout=none_box, fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                               v_row=32, v_col=16, blayout=row_major,
                               slayout=none_box, fractal=512, pad=0>)
    outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                            v_row=32, v_col=32, blayout=row_major,
                            slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tconcatidx` — 索引控制列拼接

```mlir
pto.tconcatidx ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                   : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
               outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each row i:
    idx0_num = src0Idx[i, 0]
    idx1_num = src1Idx[i, 0]
    copy from src0: min(idx0_num, src0_valid_col, dst_valid_col) columns
    copy from src1: min(idx1_num, src1_valid_col, dst_valid_col - copied_from_src0) columns
```

逐行按索引控制从两个源 tile 拼接到目标 tile 的列方向操作。与 `pto.tconcat` 不同，每行的拼接列数由索引 tile 动态控制。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `src0Idx` | `pto.tile_buf` | src0 的逐行索引 tile（每行指定拷贝列数） |
| `src1Idx` | `pto.tile_buf` | src1 的逐行索引 tile（每行指定拷贝列数） |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 所有操作数必须使用 `loc=vec`。
  - 数据 tile（src0、src1、dst）的元素类型必须一致，且为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`、`bf16`。
  - 索引 tile（src0Idx、src1Idx）的元素类型必须相同，允许 signless `i8`、`i16` 或 `i32`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 同 Atlas A2系列产品/Atlas A3系列产品 约束。
  - 额外要求所有操作数必须使用 `blayout=row_major`。

**示例：**

```mlir
pto.tconcatidx
    ins(%src0, %src1, %idx0, %idx1 :
        !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=8,
                      v_row=16, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=8,
                      v_row=16, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=64,
                              v_row=16, v_col=64, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.textract` — 子 Tile 提取

```mlir
pto.textract ins(<src>, <indexRow>, <indexCol> : !pto.tile_buf, index, index)
             outs(<dst> : !pto.tile_buf)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i + indexRow, j + indexCol]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `indexRow` | `index` | 以行为单位的提取起始偏移 |
| `indexCol` | `index` | 以元素为单位的提取起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer（子区域） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 基础 `vec` → `vec` 和 `mat` → `left`/`right` 形式使用相同元素类型，支持 `i8`、`f16`、`bf16`、`f32`。
  - 支持 `vec` → `vec`、`mat` → `left`/`right`，以及 `acc` → `mat`。累加器转换形式支持 `f32` → `f16`/`bf16`；附加量化参数见本章 `fp` 形式。
  - src 的 layout/fractal 必须与 dst 支持的组合兼容
  - 运行时约束：`indexRow + dst.rows <= src.rows` 且 `indexCol + dst.cols <= src.cols`
  - `mat` → `left`/`right` 的目标分别使用 `row_major/row_major` 和 `row_major/col_major` 布局；`acc` → `mat` 的源和目标使用 `col_major/row_major`，目标 `fractal=512`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 基础非累加器形式使用相同元素类型；累加器形式可进行支持的精度转换，附加量化参数见本章 `fp` 形式。
  - 支持 Mat->Left/Right/Scaling、Vec->Mat、Acc->Mat/Vec，以及
    ND 布局的 Vec->Vec；ND 指 `blayout=row_major, slayout=none_box`
  - 运行时约束：`indexRow + dst.rows <= src.rows` 且 `indexCol + dst.cols <= src.cols`

**示例：**

```mlir
pto.textract
    ins(%src, %row, %col :
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        index, index)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                              v_row=16, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.textract` 的 `fp` 形式

```mlir
pto.textract ins(<src>, <indexRow>, <indexCol> : <src_type>, index, index
                fp <fp> : <fp_type>)
             outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i, j] = dequant_extract(src[i + indexRow, j + indexCol], fp)
// 从累加器 tile 中提取子窗口，同时通过缩放因子进行反量化/类型转换
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer，必须为 `loc=acc` |
| `fp` | `pto.tile_buf` | 缩放因子 tile buffer，必须为 `loc=scaling` |
| `indexRow` | `index` | 提取起始行偏移 |
| `indexCol` | `index` | 提取起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 位置必须为 `src=acc`、`fp=scaling`、`dst=mat`。
  - 支持的类型对：`(src=f32, dst=i8)` 或 `(src=i32, dst=i8/f16/i16)`。
  - `dst` 的 fractal 必须为 512。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 位置必须为 `src=acc`、`fp=scaling`，`dst` 可为 `mat` 或 `vec`。
  - 支持的类型对：`(src=f32, dst=i8/fp8/f16/bf16/f32)` 或 `(src=i32, dst=i8/f16/bf16)`。
  - 无 fractal 512 限制。

**示例：**

```mlir
pto.textract ins(%src, %row, %col : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                 fractal=1024, pad=0>, index, index
                 fp %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=row_major,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                 fractal=512, pad=0>)
```

---

### `pto.tinsert` — 子 Tile 插入

```mlir
pto.tinsert ins(<src>, <indexRow>, <indexCol> : !pto.tile_buf, index, index)
            outs(<dst> : !pto.tile_buf)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i + indexRow, j + indexCol] = src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（待插入） |
| `indexRow` | `index` | 插入起始行偏移 |
| `indexCol` | `index` | 插入起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src`、`dst` 必须为 rank-2 `tile_buf`，`indexRow`、`indexCol` 必须非负，且插入区域不能越过 `dst` 静态边界
  - 支持 Vec->Vec（同元素类型，`i8`/`f16`/`bf16`/`f32`）和 Acc->Mat
  - Acc->Mat 要求两端为 NZ 布局（`col_major` + `row_major`），目标 `fractal=512`
  - `fp` 与 `preQuantScalar` 互斥，且仅适用于 `src.loc=acc`；`fp` 必须使用 `loc=scaling`

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 支持 Acc->Mat/Vec、Vec->Mat 和 Vec->Vec
  - Vec->Vec 两端布局必须同为 ND（`row_major` + `none_box`）或同为 NZ（`col_major` + `row_major`）
  - Vec->Mat 的目标必须为 NZ；源可为 ND 或 NZ，且源、目标元素类型相同
  - `accToVecMode` 仅适用于 Acc->Vec；`tinsertMode` 仅适用于 NZ 的 Vec->Mat
  - `fp`、`preQuantScalar` 和 ReLU 形式要求 `src.loc=acc`；`fp` 必须使用 `loc=scaling`

**示例：**

```mlir
pto.tinsert
    ins(%src, %row, %col :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        index, index)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tinsert` 的 `fp` 形式

```mlir
pto.tinsert ins(<src>, <indexRow>, <indexCol> : <src_type>, index, index
               fp <fp> : <fp_type>)
            outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
For each element (i, j):
    dst[i + indexRow, j + indexCol] = quant_insert(src[i, j], fp)
// 将 vector tile 通过缩放因子进行量化后插入到累加器 tile 的指定子窗口
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer，必须为 `loc=acc` |
| `fp` | `pto.tile_buf` | 缩放因子 tile buffer，必须为 `loc=scaling` |
| `indexRow` | `index` | 插入起始行偏移 |
| `indexCol` | `index` | 插入起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 位置必须为 `src=acc`、`fp=scaling`、`dst=mat`。
  - 支持的类型对：`(src=f32, dst=i8)` 或 `(src=i32, dst=i8/f16/i16)`。
  - `dst` 的 fractal 必须为 512。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 位置必须为 `src=acc`、`fp=scaling`，`dst` 可为 `mat` 或 `vec`。
  - 支持的类型对：`(src=f32, dst=i8/fp8/f16/bf16/f32)` 或 `(src=i32, dst=i8/f16/bf16)`。
  - 无 fractal 512 限制。

**示例：**

```mlir
pto.tinsert ins(%src, %row, %col : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                fractal=1024, pad=0>, index, index
                fp %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=32, cols=32,
                v_row=32, v_col=32, blayout=row_major, slayout=row_major,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=32,
                v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                fractal=512, pad=0>)
```

---

### `pto.tgather` — 聚集/选择元素

`pto.tgather` 有三种使用形式：索引形式、比较形式和掩码形式。

```mlir
// 索引形式
pto.tgather ins(<src>, <indices> : <src_type>, <indices_type>)
            outs(<dst> : <dst_type>)

// 比较形式
pto.tgather ins(<src>, <kValue> : <src_type>, <scalar_type>)
            outs(<dst>, <cdst> : <dst_type>, <cdst_type>)
            {cmpMode = #pto<cmp <mode>>, offset = <i32>}

// 掩码形式
pto.tgather ins(<src>, {maskPattern = #pto.mask_pattern<<pattern>>} : <src_type>, "row")
            outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
索引形式：
    For each element (i, j):
        dst[i, j] = src[indices[i, j]]

比较形式：
    dst 存储满足标量比较条件的索引
    cdst 存储每行选中的元素个数

掩码形式：
    将 src 的有效区域按行主序扫描，只保留列位置满足 maskPattern 的元素，
    并将这些元素按顺序连续写入 dst。

    设 src 的有效区域为 R x C，dst 从线性位置 0 开始顺序写入，
    若第 r 行第 c 列被 maskPattern 选中，则：
        dst[linear_out] = src[r, c]
        linear_out = linear_out + 1
```

对单行输入，或仅观察每一行内被选中的列位置时，默认掩码形式可写为：

```text
P1111:
    dst[r, k] = src[r, k]

P0101:
    dst[r, k] = src[r, 2*k]

P1010:
    dst[r, k] = src[r, 2*k + 1]

P0001:
    dst[r, k] = src[r, 4*k]

P0010:
    dst[r, k] = src[r, 4*k + 1]

P0100:
    dst[r, k] = src[r, 4*k + 2]

P1000:
    dst[r, k] = src[r, 4*k + 3]
```

其中 `k` 取到右侧索引仍小于 `src` 的有效列数为止。`dst` 的写入结果是连续压紧的，因此当 `src` 有多行时，上一行选出的尾部元素和下一行选出的首部元素在 `dst` 中也是连续存放的。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 主目标 tile 缓冲区 |
| `cdst` | `pto.tile_buf` | 比较形式中的辅助目标 tile（仅比较形式） |
| `indices` | `pto.tile_buf` | 索引形式中的索引 tile（仅索引形式） |
| `kValue` | 标量类型 | 比较形式中的标量比较值（仅比较形式） |

**返回值：** 无。以 DPS 的形式写入 `dst`（及比较形式下的 `cdst`）。

**属性：**

- `maskPattern` — 掩码模式，仅用于掩码形式。
  - `#pto.mask_pattern<P0101>` — 每 2 个元素取第 1 个，即选中列 `0, 2, 4, ...`
  - `#pto.mask_pattern<P1010>` — 每 2 个元素取第 2 个，即选中列 `1, 3, 5, ...`
  - `#pto.mask_pattern<P0001>` — 每 4 个元素取第 1 个，即选中列 `0, 4, 8, ...`
  - `#pto.mask_pattern<P0010>` — 每 4 个元素取第 2 个，即选中列 `1, 5, 9, ...`
  - `#pto.mask_pattern<P0100>` — 每 4 个元素取第 3 个，即选中列 `2, 6, 10, ...`
  - `#pto.mask_pattern<P1000>` — 每 4 个元素取第 4 个，即选中列 `3, 7, 11, ...`
  - `#pto.mask_pattern<P1111>` — 选中全部列（不筛选）

- `cmpMode` — 比较模式，仅用于比较形式。默认值为 `eq`。
  - `#pto<cmp eq>` — 相等比较
  - `#pto<cmp gt>` — 大于比较

- `offset` — 比较形式中的聚集基索引偏移。默认值为 `0`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 索引形式：`src` 和 `dst` 元素类型必须一致，且为 `i16`、`i32`、`f16` 或 `f32` 之一。`indices` 元素类型必须为 `i32`，物理形状和有效形状必须为静态值。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。
  - 比较形式：`dst` 和 `cdst` 元素类型必须为 `i32`。`src` 元素类型必须为 `f16`、`f32`，或当 `cmpMode=eq` 时可为 `i32`；`src` 的物理形状必须为静态值。`kValue` 类型必须与 `src` 元素类型一致。`cmpMode` 必须为 `eq` 或 `gt`。`src`、`dst`、`cdst` 必须为 `loc=vec`。
  - 掩码形式：`src` 元素大小必须为 2 或 4 字节。`src` 和 `dst` 必须使用 `loc=vec` 和 `blayout=row_major`。`src` 和 `dst` 元素大小必须一致。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 索引形式：`src` 和 `dst` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16` 或 `f32` 之一。`indices` 元素类型可为 `i16` 或 `i32`。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。
  - 比较形式：`dst` 和 `cdst` 元素类型必须为 `i32`。`src` 元素类型必须为 `i16`、`i32`、`f16` 或 `f32` 之一。`kValue` 类型必须与 `src` 元素类型一致。`cmpMode` 必须为 `eq` 或 `gt`。`src`、`dst`、`cdst` 必须为 `loc=vec`。
  - 掩码形式：`src` 元素大小必须为 1、2 或 4 字节。`src` 和 `dst` 必须使用 `loc=vec` 和 `blayout=row_major`。`src`/`dst` 元素类型必须为 `i8`、`i16`、`i32`、`f16`、`bf16`、`f32` 或 fp8 类支持类型之一。`src` 和 `dst` 元素大小必须一致。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。

**示例：**

```mlir
// 索引形式
pto.tgather ins(%src, %indices :
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%index_dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

### `pto.tgatherb` — 按字节偏移聚集 32 字节块

```mlir
pto.tgatherb ins(<src>, <offsets> : <src_type>, <offsets_type>)
             outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
offsets 的每个元素给出源侧一个 32 字节块的首字节地址：
    dst 第 i 行的第 b 个 32 字节块 = src 中起始于 offsets[i, b] 的 32 字节
其中每行 dst 的块个数 = ceil(dst 有效列数 / (32 / sizeof(dst 元素类型)))
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile 缓冲区 |
| `offsets` | `pto.tile_buf` | 32 字节块的字节地址 tile；有效行数须与 `dst` 一致，有效列数为 `align_up(ceil(dst 有效列数 / (32 / sizeof(dst 元素类型))), 8)` |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `dst` 和 `offsets` 必须使用行主序布局（`blayout=row_major`）。
  - `dst` 元素大小必须为 1、2 或 4 字节。
  - `offsets` 元素类型必须为 32 位整数（`i32`/`ui32`）。
  - `offsets` 的有效行数必须等于 `dst` 的有效行数。
  - `offsets` 的有效列数必须为紧凑的 32 字节块地址个数并按 8 对齐：`align_up(ceil(dst 有效列数 / (32 / sizeof(dst 元素类型))), 8)`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst` 元素大小必须为 1、2 或 4 字节。
  - 无行主序布局要求。

**示例：**

```mlir
// dst 为 8x32 的 f32 tile：每行 32*4=128 字节即 4 个 32 字节块，按 8 对齐后 offsets 每行 8 列
pto.tgatherb ins(%src, %offsets :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=32,
                     v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i32, rows=8, cols=8,
                     v_row=8, v_col=8, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=32,
                     v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tscatter` — 散射元素

`pto.tscatter` 有两种使用形式：索引形式和掩码形式。

```mlir
// 索引形式
pto.tscatter ins(<src>, <indexes> : <src_type>, <indexes_type>)
             outs(<dst> : <dst_type>)

// 掩码形式
pto.tscatter ins(<src>, {maskPattern = #pto.mask_pattern<<pattern>>} : <src_type>, <axis>)
             outs(<dst> : <dst_type>)
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
索引形式：
    For each element (i, j):
        flat_dst[indexes[i, j]] = src[i, j]

掩码形式：
    将 src 中按行主序连续存放的元素，按 maskPattern 指定的位置散射回 dst；
    未被 maskPattern 命中的位置补 0。
```

`axis="row"` 的掩码形式可视为 `pto.tgather` 掩码形式的反向展开。设 `src` 的有效区域为 `R x Csrc`，`dst` 的有效区域为 `R x Cdst`，并满足：

- `P0101` / `P1010`：`Cdst = 2 * Csrc`
- `P0001` / `P0010` / `P0100` / `P1000`：`Cdst = 4 * Csrc`
- `P1111`：`Cdst = Csrc`

则有：

```text
P1111:
    dst[r, k] = src[r, k]

P0101:
    dst[r, 2*k]     = src[r, k]
    dst[r, 2*k + 1] = 0

P1010:
    dst[r, 2*k]     = 0
    dst[r, 2*k + 1] = src[r, k]

P0001:
    dst[r, 4*k]     = src[r, k]
    dst[r, 4*k + 1] = 0
    dst[r, 4*k + 2] = 0
    dst[r, 4*k + 3] = 0

P0010:
    dst[r, 4*k]     = 0
    dst[r, 4*k + 1] = src[r, k]
    dst[r, 4*k + 2] = 0
    dst[r, 4*k + 3] = 0

P0100:
    dst[r, 4*k]     = 0
    dst[r, 4*k + 1] = 0
    dst[r, 4*k + 2] = src[r, k]
    dst[r, 4*k + 3] = 0

P1000:
    dst[r, 4*k]     = 0
    dst[r, 4*k + 1] = 0
    dst[r, 4*k + 2] = 0
    dst[r, 4*k + 3] = src[r, k]
```

例如当一行源数据为 `[a, b, c, d]` 时：

```text
P0101 -> [a, 0, b, 0, c, 0, d, 0]
P1010 -> [0, a, 0, b, 0, c, 0, d]
P0010 -> [0, a, 0, 0, 0, b, 0, 0, 0, c, 0, 0, 0, d, 0, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile 缓冲区 |
| `indexes` | `pto.tile_buf` | 与 src 有效区域相同的逐元素目标索引 Tile；索引单位为目标元素（仅索引形式） |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `axis` — 掩码形式必须指定 `"row"` 或 `"col"`，写在 ins 的源类型之后；没有默认值，索引形式不使用该属性。
  - `"row"`：保持行数，在每一行内沿列展开。
  - `"col"`：保持列数，沿行展开；将上述公式中的列坐标换为行坐标。

- `maskPattern` — 掩码模式，仅用于掩码形式。
  - `#pto.mask_pattern<P0101>` — 每 2 个位置中的第 1 个位置写入 `src`，其余位置补 0
  - `#pto.mask_pattern<P1010>` — 每 2 个位置中的第 2 个位置写入 `src`，其余位置补 0
  - `#pto.mask_pattern<P0001>` — 每 4 个位置中的第 1 个位置写入 `src`，其余位置补 0
  - `#pto.mask_pattern<P0010>` — 每 4 个位置中的第 2 个位置写入 `src`，其余位置补 0
  - `#pto.mask_pattern<P0100>` — 每 4 个位置中的第 3 个位置写入 `src`，其余位置补 0
  - `#pto.mask_pattern<P1000>` — 每 4 个位置中的第 4 个位置写入 `src`，其余位置补 0
  - `#pto.mask_pattern<P1111>` — 全量复制，不插入 0

**约束：**

- Atlas A3系列产品 和 Ascend 950PR&950DT系列产品 均支持索引形式及掩码形式；indexes 和 maskPattern 必须且只能选择其一。
- src/dst 均位于 `vec`，元素类型一致，支持 8/16/32 位整数及 `f16`、`bf16`、`f32`。
- 索引形式：indexes 位于 `vec`，与 src 有效形状相同；dst 每维有效尺寸不小于 src。4 字节数据使用 32 位索引，2 字节和 1 字节数据使用 16 位索引。
- 每个索引表示从 dst 存储起点计数的目标元素位置，不是行号或字节偏移。用户应提供范围内且互不冲突的索引；操作不进行逐值越界检查，不保证重复目标索引的覆盖顺序。
- Ascend 950PR&950DT系列产品 索引形式先将目标清零，再写入索引命中的位置；未命中位置为零。
- 掩码形式：src/dst 均使用 `blayout=row_major`。设扩展因子为 F：`P0101/P1010` 为 2，单个 1 的四位模式为 4，`P1111` 为 1。
- `axis="row"` 要求有效行数相同，`dst.valid_cols=F*src.valid_cols`；`axis="col"` 要求有效列数相同，`dst.valid_rows=F*src.valid_rows`。掩码未命中的位置补零。

**示例：**

索引形式中，若源有效数据为 `[11,22,33,44]`，对应索引为 `[2,0,3,1]`，目标前四个元素为 `[22,44,11,33]`。每个索引对应一个源元素。

```mlir
pto.tscatter ins(%values, %indexes :
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=8,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%result : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
```

```mlir
// Atlas A3系列产品/Ascend 950PR&950DT系列产品：沿每一行的列方向展开
pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<P0101>} :
                 !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                     v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>, "row")
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=64,
                     v_row=1, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

沿列的行方向展开由 `"col"` 指定。下例把两行有效数据展开成四行，第 2、4 行补零。

```mlir
pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<P0101>} :
  !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=8,
    v_row=2, v_col=8, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>, "col")
  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=8,
    v_row=4, v_col=8, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
```
