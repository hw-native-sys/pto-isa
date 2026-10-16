# 初始化

本节描述了 PTO ISA 中初始化操作的指令名称、签名和语义。所有操作均作用于本地 `tile_buf`。

通用汇编形式（generic assembly form）：

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## 目录

- [`pto.tci` — 连续整数序列生成](#ptotci--连续整数序列生成)
- [`pto.trandom` — 随机数生成](#ptotrandom--随机数生成)
- [`pto.ttri` — 三角掩码生成](#ptottri--三角掩码生成)
- [`pto.tfillpad` — 填充 Padding 区域](#ptotfillpad--填充-padding-区域)

---

## 操作详解

### `pto.tci` — 连续整数序列生成

```mlir
pto.tci ins(<S> : <int_type>)
         outs(<dst> : <dst_type>)
         {descending = <bool>}
```

**语义：**

```text
For each column j in the single valid row:
    if descending == false:
        dst[0, j] = S + j
    else:
        dst[0, j] = S - j
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `S` | `AnyInteger` | 起始整数值 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `descending` — 是否生成降序序列。默认值为 `false`。
  - `false` — 生成升序序列（S，S+1，S+2，...）
  - `true` — 从 S 开始递减（S，S-1，S-2，...）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `dst` 元素类型必须为整数类型，支持 16 位或 32 位整数，例如 `i16`、`ui16`、`i32`、`ui32`。
  - `S` 的类型必须与 `dst` 元素类型完全一致。
  - `dst` 使用 `loc=vec`，为 rank-2 的单行序列 Tile；有效行数为 1。
  - `dst.cols` 不能为 1。

**示例：**

```mlir
// 生成 i16 升序序列
pto.tci ins(%c0_i16 : i16)
        outs(%tile : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=16,
            v_row=1, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=1>)
```

当起始值为 10、有效列数为 4 时，升序结果为 `[10,11,12,13]`，降序结果为 `[10,9,8,7]`。

```mlir
%start = pto.constant 10 : i16
pto.tci ins(%start : i16)
  outs(%sequence : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=16,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>) {descending = true}
```

---

### `pto.trandom` — 随机数生成

```mlir
pto.trandom ins(<key0>, <key1>, <counter0>, <counter1>, <counter2>, <counter3>
                : i32, i32, i32, i32, i32, i32)
            outs(<dst> : <dst_type>)
```

**语义：**

```text
dst = philox_random(key0, key1, counter0..counter3, rounds)
// 使用 Philox 算法通过 key/counter 对生成伪随机数填充 dst tile
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `key0` | `i32` | 密钥字 0 |
| `key1` | `i32` | 密钥字 1 |
| `counter0` | `i32` | 计数器字 0 |
| `counter1` | `i32` | 计数器字 1 |
| `counter2` | `i32` | 计数器字 2 |
| `counter3` | `i32` | 计数器字 3 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区（`i32`/`ui32`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `rounds` — Philox 迭代轮次。默认值为 `10`。
  - `7` — 7 轮（较快，随机性稍弱）
  - `10` — 10 轮（标准）

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 不支持，仅 Ascend 950PR&950DT系列产品 可用。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 所有 key/counter 操作数必须为 `i32`/`ui32`。
  - `dst` 元素类型必须为 `i32` 或 `ui32`。
  - `dst` 必须使用 `blayout=row_major`。
  - `rounds` 必须为 7 或 10。

**示例：**

```mlir
pto.trandom
    ins(%k0, %k1, %c0, %c1, %c2, %c3 : i32, i32, i32, i32, i32, i32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=4, cols=256,
                              v_row=4, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)

// 使用 7 轮
pto.trandom
    ins(%k0, %k1, %c0, %c1, %c2, %c3 {rounds = 7 : i32}
        : i32, i32, i32, i32, i32, i32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=2, cols=256,
                              v_row=2, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.ttri` — 三角掩码生成

```mlir
pto.ttri ins(<diagonal> : <integer_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    if upperOrLower == 0:  // lower triangular
        dst[i, j] = (j <= i + diagonal) ? 1 : 0
    else:                  // upper triangular
        dst[i, j] = (j >= i + diagonal) ? 1 : 0
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `diagonal` | 整数类型 | 对角线偏移 |
| `dst` | `pto.tile_buf` | 目标掩码 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `upperOrLower` — 三角类型。默认值为 `0`。
  - `0` — 下三角掩码
  - `1` — 上三角掩码

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `dst` 元素类型必须为 `f16`、`f32`、`i16`、`i32`、`u16` 或 `u32`。
  - `dst` 必须使用 `loc=vec`。
  - `upperOrLower` 必须为 0 或 1。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst` 元素类型必须为 `f16`、`f32`、`bf16`、`i8`、`i16`、`i32`、`u8`、`u16` 或 `u32`。
  - `dst` 必须使用 `loc=vec`。
  - `upperOrLower` 必须为 0 或 1。

**示例：**

```mlir
// 下三角掩码
pto.ttri
    ins(%diag : i32)
    outs(%lower : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                                v_row=32, v_col=32, blayout=row_major,
                                slayout=none_box, fractal=512, pad=0>)

// 上三角掩码
pto.ttri
    ins(%diag {upperOrLower = 1 : i32} : i32)
    outs(%upper : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                v_row=16, v_col=16, blayout=row_major,
                                slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tfillpad` — 填充 Padding 区域

```mlir
pto.tfillpad ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：** 将源有效区域复制到目标中的相同坐标，并按目标类型的 pad 策略填充其余物理区域。该接口同时支持等容量填充、VEC 目标容量扩展及同一存储的原地填充。

```text
For each physical position (i,j) in dst:
    if i < src.valid_rows and j < src.valid_cols:
        dst[i,j] = src[i,j]
    else:
        dst[i,j] = padding_value(dst.pad, dst.dtype)
// src 与 dst 为同一 Tile 时，源有效区域保持原值。
```

**参数与返回值：** src 为源 Tile，dst 为预先分配的目标 Tile；没有 SSA 返回值。物理容量由 rows/cols 决定，参与复制的区域由源 v_row/v_col 决定。

**约束：**

- src/dst 为 rank-2 Tile；元素存储大小相同，均为 1、2 或 4 字节。
- dst 的 pad 不能为 `null`（0）；`zero`（1）、`max`（2）、`min`（3）分别表示零填充、元素类型最大值和最小值策略，整数编码不是任意填充值。
- 各维目标物理尺寸不小于源尺寸；物理尺寸扩展仅适用于 `loc=vec` 的源与目标。不允许用不匹配的动态物理尺寸表示扩展。
- `loc=mat` 时源与目标 Tile 类型相同，包括有效区域与 pad；可选 `padValue = #pto<pad_value zero|max|min>` 属性必须与目标类型的 pad 一致，省略时使用目标类型策略。该属性不能用于 VEC。
- 不使用单独的 mode 属性选择行为；同一 Tile 可同时作为 ins 与 outs，表示原地填充。

**示例：**

VEC 的 `16x16` 数据扩展到 `32x32` 容量，源区域外补零；随后演示对同一个 Tile 原地补零。

```mlir
pto.tfillpad
  ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=1>)
```

```mlir
pto.tfillpad
  ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=1>)
  outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=1>)
```
