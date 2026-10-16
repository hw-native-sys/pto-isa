# 矩阵计算操作

本节描述了 PTO ISA 中全部矩阵计算操作的指令名称、签名和语义。矩阵计算操作在 Cube（矩阵）流水线上执行，用于完成矩阵乘法（TMATMUL）和矩阵-向量乘法（TGEMV）。所有操作均采用“目标传递风格”（Destination-Passing Style，DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。

这一类操作通常具有如下汇编形式：

```mlir
 pto.op ins(%lhs, %rhs : !pto.tile_buf<...>, !pto.tile_buf<...>)
        outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 左矩阵 tile 位于 `loc=left`（L0A 缓冲区）
- 右矩阵 tile 位于 `loc=right`（L0B 缓冲区）
- 目标累加器 tile 位于 `loc=acc`（L0C 缓冲区）
- 形状约束：`lhs.rows == dst.rows`，`lhs.cols == rhs.rows`，`rhs.cols == dst.cols`
- 所有操作在矩阵流水线（`PIPE_M`）上执行
- **（Ascend 950PR&950DT系列产品）** 布局约束：`lhs.blayout=col_major, lhs.slayout=row_major`；`rhs.blayout=row_major, rhs.slayout=col_major`；`dst.blayout=col_major, dst.slayout=row_major`。Atlas A2系列产品/Atlas A3系列产品 不在 IR 层面强制校验 `blayout`/`slayout`

---

## 目录

- [`pto.tmatmul` — 矩阵乘法](#ptotmatmul--矩阵乘法)
- [`pto.tmatmul.acc` — 矩阵乘加（累加）](#ptotmatmulacc--矩阵乘加累加)
- [`pto.tmatmul.bias` — 矩阵乘法加偏置](#ptotmatmulbias--矩阵乘法加偏置)
- [`pto.tmatmul.mx` — 混合精度矩阵乘法](#ptotmatmulmx--混合精度矩阵乘法)
- [`pto.tmatmul.mx.acc` — 混合精度矩阵乘加](#ptotmatmulmxacc--混合精度矩阵乘加)
- [`pto.tmatmul.mx.bias` — 混合精度矩阵乘法加偏置](#ptotmatmulmxbias--混合精度矩阵乘法加偏置)
- [`pto.tgemv` — 矩阵-向量乘法](#ptotgemv--矩阵-向量乘法)
- [`pto.tgemv.acc` — 矩阵-向量乘加（累加）](#ptotgemvacc--矩阵-向量乘加累加)
- [`pto.tgemv.bias` — 矩阵-向量乘法加偏置](#ptotgemvbias--矩阵-向量乘法加偏置)
- [`pto.tgemv.mx` — 混合精度矩阵-向量乘法](#ptotgemvmx--混合精度矩阵-向量乘法)
- [`pto.tgemv.mx.acc` — 混合精度矩阵-向量乘加](#ptotgemvmxacc--混合精度矩阵-向量乘加)
- [`pto.tgemv.mx.bias` — 混合精度矩阵-向量乘法加偏置](#ptotgemvmxbias--混合精度矩阵-向量乘法加偏置)

---

## 操作详解

### `pto.tmatmul` — 矩阵乘法

```mlir
pto.tmatmul ins(<lhs>, <rhs> : <lhs_type>, <rhs_type>)
            outs(<dst> : <dst_type>)
            {accPhase = <phase>}
```

**语义：**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 左矩阵 tile buffer（`loc=left`，L0A） |
| `rhs` | `pto.tile_buf` | 右矩阵 tile buffer（`loc=right`，L0B） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`，L0C） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 支持的 `(dst 元素类型, lhs 元素类型, rhs 元素类型)` 组合：`(i32, i8, i8)`、`(f32, f16, f16)`、`(f32, f32, f32)`、`(f32, bf16, bf16)`。
  - 运行时约束：`m/k/n`（分别取自 `lhs valid row`、`lhs valid column`、`rhs valid column`）必须在 `[1, 4095]` 范围内。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst` 元素类型必须为 `i32` 或 `f32`。
  - 若 `dst` 为 `i32`，则 `lhs` 和 `rhs` 元素类型必须均为 `i8`。
  - 若 `dst` 为 `f32`，则 `lhs`/`rhs` 元素类型支持 `f16`、`bf16`、`f32`，以及部分 fp8 类型对（目标定义）。
  - 运行时约束：`m/k/n`（分别取自 `lhs valid row`、`lhs valid column`、`rhs valid column`）必须在 `[1, 4095]` 范围内。

**示例：**

```mlir
pto.tmatmul ins(%lhs, %rhs :
                !pto.tile_buf<loc=left, dtype=f16, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=col_major,
                    slayout=row_major, fractal=512, pad=0>,
                !pto.tile_buf<loc=right, dtype=f16, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=row_major,
                    slayout=col_major, fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=col_major,
                    slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tmatmul.acc` — 矩阵乘加（累加）

```mlir
pto.tmatmul.acc ins(<acc_in>, <lhs>, <rhs> : <acc_in_type>, <lhs_type>, <rhs_type>)
                outs(<dst> : <dst_type>)
                {accPhase = <phase>}
```

**语义：**

```text
For each (i, j):
    dst[i, j] = acc_in[i, j] + sum_k lhs[i, k] * rhs[k, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | 先前的累加器值（`loc=acc`） |
| `lhs` | `pto.tile_buf` | 左矩阵 tile buffer（`loc=left`） |
| `rhs` | `pto.tile_buf` | 右矩阵 tile buffer（`loc=right`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`dst` 的类型组合约束与 `pto.tmatmul` 相同。
  - `acc_in` 必须位于 `loc=acc`，其元素类型和形状与 `dst` 一致。
  - 运行时约束与 `pto.tmatmul` 相同。

**示例：**

```mlir
pto.tmatmul.acc ins(%acc_in, %lhs, %rhs :
                    !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=col_major,
                        slayout=row_major, fractal=1024, pad=0>,
                    !pto.tile_buf<loc=left, dtype=f16, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=col_major,
                        slayout=row_major, fractal=512, pad=0>,
                    !pto.tile_buf<loc=right, dtype=f16, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=row_major,
                        slayout=col_major, fractal=512, pad=0>)
                outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=col_major,
                        slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tmatmul.bias` — 矩阵乘法加偏置

```mlir
pto.tmatmul.bias ins(<lhs>, <rhs>, <bias> : <lhs_type>, <rhs_type>, <bias_type>)
                 outs(<dst> : <dst_type>)
                 {accPhase = <phase>}
```

**语义：**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j] + bias[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 左矩阵 tile buffer（`loc=left`） |
| `rhs` | `pto.tile_buf` | 右矩阵 tile buffer（`loc=right`） |
| `bias` | `pto.tile_buf` | 偏置 tile buffer（`loc=bias`，`rows=1`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `lhs`、`rhs`、`dst` 的类型组合约束与 `pto.tmatmul` 相同。
  - `bias` 元素类型必须与 `dst` 元素类型一致。
  - `bias` 必须使用 `loc=bias`，且 `rows=1`。
  - 运行时约束与 `pto.tmatmul` 相同。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`dst` 的类型组合约束与 `pto.tmatmul` 相同。
  - `bias` 元素类型必须与 `dst` 元素类型一致。
  - `bias` 必须使用 `loc=bias`、`rows=1` 和 `blayout=row_major`。
  - 运行时约束与 `pto.tmatmul` 相同。

**示例：**

```mlir
pto.tmatmul.bias ins(%lhs, %rhs, %bias :
                     !pto.tile_buf<loc=left, dtype=f16, rows=32, cols=32,
                         v_row=32, v_col=32, blayout=col_major,
                         slayout=row_major, fractal=512, pad=0>,
                     !pto.tile_buf<loc=right, dtype=f16, rows=32, cols=32,
                         v_row=32, v_col=32, blayout=row_major,
                         slayout=col_major, fractal=512, pad=0>,
                     !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=32,
                         v_row=1, v_col=32, blayout=row_major,
                         slayout=none_box, fractal=512, pad=0>)
                 outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                         v_row=32, v_col=32, blayout=col_major,
                         slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tmatmul.mx` — 混合精度矩阵乘法

```mlir
pto.tmatmul.mx ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                   <lhs_type>, <lhs_scale_type>, <rhs_type>, <rhs_scale_type>)
               outs(<dst> : <dst_type>)
               {accPhase = <phase>}
```

**语义：**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
```

`lhs_scale` 和 `rhs_scale` 是 MX 缩放因子 tile。K 维是矩阵乘法的归约维，即 `lhs` 的列维和 `rhs` 的行维；上式中的 `sum_k` 就沿这个维度累加。缩放沿 K 维分组，每 32 个连续 K 元素共享一个缩放因子：`lhs_scale[i, g]` 缩放 `lhs[i, 32*g .. 32*g+31]`，`rhs_scale[g, j]` 缩放 `rhs[32*g .. 32*g+31, j]`。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 左矩阵 tile buffer（`loc=left`） |
| `lhs_scale` | `pto.tile_buf` | 左矩阵缩放 tile buffer（`loc=scaling`） |
| `rhs` | `pto.tile_buf` | 右矩阵 tile buffer（`loc=right`） |
| `rhs_scale` | `pto.tile_buf` | 右矩阵缩放 tile buffer（`loc=scaling`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **Atlas A2系列产品/Atlas A3系列产品 不支持此操作。**
- **Ascend 950PR&950DT系列产品 约束**
  - `lhs`/`rhs` 必须位于 `left`/`right`，目标位于 `acc`，并满足本章的矩阵形状与布局约束；`dst` 元素类型为 `f32`。
  - 本节示例使用 `f8E4M3FN` 或 `f8E5M2` 的 FP8 类型对；以下尺寸关系分别使用数据类型中记录的物理与有效 extent。
  - 左数据物理列数与右数据物理行数必须为 64 的正整数倍；有效 `M/K/N` 必须在 `[1,4095]`。
  - 两个 scale 均位于 `scaling`。对物理尺寸和有效尺寸分别计算：左 scale 为 `[M,ceil(K/32)]`，右 scale 为 `[ceil(K/32),N]`。
  - 左 scale 使用 `blayout=row_major, slayout=row_major`；右 scale 使用 `blayout=col_major, slayout=col_major`；两者 `fractal=32`。

**示例：**

Ascend 950PR&950DT系列产品 示例：有效区域等于物理尺寸，M=16、K=64、N=16；左 scale 为 `16x2`，右 scale 为 `2x16`。输入数据与缩放因子已准备在对应 Tile 中。

```mlir
pto.tmatmul.mx ins(%a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=16, cols=64,
        v_row=16, v_col=64, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=2,
        v_row=16, v_col=2, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=64, cols=16,
        v_row=64, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=2, cols=16,
        v_row=2, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
      v_row=16, v_col=16, blayout=col_major, slayout=row_major,
      fractal=512, pad=0>)
```

---

### `pto.tmatmul.mx.acc` — 混合精度矩阵乘加

```mlir
pto.tmatmul.mx.acc ins(<acc_in>, <lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                       <acc_in_type>, <lhs_type>, <lhs_scale_type>,
                       <rhs_type>, <rhs_scale_type>)
                   outs(<dst> : <dst_type>)
                   {accPhase = <phase>}
```

**语义：**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] = acc_in[i, j] +
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | 先前的累加器值（`loc=acc`） |
| `lhs` | `pto.tile_buf` | 左矩阵 tile buffer（`loc=left`） |
| `lhs_scale` | `pto.tile_buf` | 左矩阵缩放 tile buffer（`loc=scaling`） |
| `rhs` | `pto.tile_buf` | 右矩阵 tile buffer（`loc=right`） |
| `rhs_scale` | `pto.tile_buf` | 右矩阵缩放 tile buffer（`loc=scaling`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`lhs_scale`、`rhs_scale`、`dst` 的约束与 `pto.tmatmul.mx` 相同。
  - `acc_in` 必须位于 `loc=acc`，其元素类型和形状与 `dst` 一致。
  - 运行时约束与 `pto.tmatmul.mx` 相同。

**示例：**

Ascend 950PR&950DT系列产品 示例：有效区域等于物理尺寸，M=16、K=64、N=32；左 scale 为 `16x2`，右 scale 为 `2x32`。输入数据与缩放因子已准备在对应 Tile 中。

```mlir
pto.tmatmul.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32,
        v_row=16, v_col=32, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=16, cols=64,
        v_row=16, v_col=64, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=2,
        v_row=16, v_col=2, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=64, cols=32,
        v_row=64, v_col=32, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=2, cols=32,
        v_row=2, v_col=32, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32,
      v_row=16, v_col=32, blayout=col_major, slayout=row_major,
      fractal=512, pad=0>)
```

---

### `pto.tmatmul.mx.bias` — 混合精度矩阵乘法加偏置

```mlir
pto.tmatmul.mx.bias ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale>, <bias> :
                        <lhs_type>, <lhs_scale_type>, <rhs_type>,
                        <rhs_scale_type>, <bias_type>)
                    outs(<dst> : <dst_type>)
```

**语义：**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j]) + bias[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 左矩阵 tile buffer（`loc=left`） |
| `lhs_scale` | `pto.tile_buf` | 左矩阵缩放 tile buffer（`loc=scaling`） |
| `rhs` | `pto.tile_buf` | 右矩阵 tile buffer（`loc=right`） |
| `rhs_scale` | `pto.tile_buf` | 右矩阵缩放 tile buffer（`loc=scaling`） |
| `bias` | `pto.tile_buf` | 偏置 tile buffer（`loc=bias`，`rows=1`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`lhs_scale`、`rhs_scale`、`dst` 的约束与 `pto.tmatmul.mx` 相同。
  - `bias` 元素类型必须为 `f32`。
  - `bias` 必须使用 `loc=bias`、`rows=1` 和 `blayout=row_major`。
  - 运行时约束与 `pto.tmatmul.mx` 相同。

**示例：**

Ascend 950PR&950DT系列产品 示例：有效区域等于物理尺寸，M=16、K=64、N=16；左 scale 为 `16x2`，右 scale 为 `2x16`。输入数据与缩放因子已准备在对应 Tile 中。

```mlir
pto.tmatmul.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=16, cols=64,
        v_row=16, v_col=64, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=2,
        v_row=16, v_col=2, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=64, cols=16,
        v_row=64, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=2, cols=16,
        v_row=2, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16,
        v_row=1, v_col=16, blayout=row_major, slayout=none_box,
        fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
      v_row=16, v_col=16, blayout=col_major, slayout=row_major,
      fractal=512, pad=0>)
```

---

### `pto.tgemv` — 矩阵-向量乘法

```mlir
pto.tgemv ins(<lhs>, <rhs> : <lhs_type>, <rhs_type>)
          outs(<dst> : <dst_type>)
          {accPhase = <phase>}
```

**语义：**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
// lhs 的 rows（即 m）必须为 1，退化为向量-矩阵乘法
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 矩阵 tile buffer（`loc=left`，`rows=1`） |
| `rhs` | `pto.tile_buf` | 向量 tile buffer（`loc=right`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 支持的 `(dst 元素类型, lhs 元素类型, rhs 元素类型)` 组合：`(i32, i8, i8)`、`(f32, f16, f16)`、`(f32, f32, f32)`、`(f32, bf16, bf16)`。
  - 运行时约束：`m` 必须为 `1`；`k/n`（分别取自 `rhs valid row`、`rhs valid column`）必须在 `[1, 4095]` 范围内。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `dst` 元素类型必须为 `i32` 或 `f32`。
  - 若 `dst` 为 `i32`，则 `lhs` 和 `rhs` 元素类型必须均为 `i8`。
  - 若 `dst` 为 `f32`，则 `lhs`/`rhs` 元素类型支持 `f16`、`bf16`、`f32`，以及部分 fp8 类型对（目标定义）。
  - 运行时约束：`m` 必须为 `1`；`k/n`（分别取自 `rhs valid row`、`rhs valid column`）必须在 `[1, 4095]` 范围内。

**示例：**

```mlir
pto.tgemv ins(%lhs, %rhs :
              !pto.tile_buf<loc=left, dtype=f16, rows=1, cols=128,
                  v_row=1, v_col=128, blayout=col_major,
                  slayout=row_major, fractal=512, pad=0>,
              !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=16,
                  v_row=128, v_col=16, blayout=row_major,
                  slayout=col_major, fractal=512, pad=0>)
          outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                  v_row=1, v_col=16, blayout=col_major,
                  slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tgemv.acc` — 矩阵-向量乘加（累加）

```mlir
pto.tgemv.acc ins(<acc_in>, <lhs>, <rhs> : <acc_in_type>, <lhs_type>, <rhs_type>)
              outs(<dst> : <dst_type>)
              {accPhase = <phase>}
```

**语义：**

```text
For each (i, j):
    dst[i, j] = acc_in[i, j] + sum_k lhs[i, k] * rhs[k, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | 先前的累加器值（`loc=acc`） |
| `lhs` | `pto.tile_buf` | 矩阵 tile buffer（`loc=left`，`rows=1`） |
| `rhs` | `pto.tile_buf` | 向量 tile buffer（`loc=right`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`dst` 的类型组合约束与 `pto.tgemv` 相同。
  - `acc_in` 必须位于 `loc=acc`，其元素类型和形状与 `dst` 一致。
  - 运行时约束与 `pto.tgemv` 相同。

**示例：**

```mlir
pto.tgemv.acc ins(%acc_in, %lhs, %rhs :
                  !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>,
                  !pto.tile_buf<loc=left, dtype=f16, rows=1, cols=128,
                      v_row=1, v_col=128, blayout=col_major,
                      slayout=row_major, fractal=512, pad=0>,
                  !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=16,
                      v_row=128, v_col=16, blayout=row_major,
                      slayout=col_major, fractal=512, pad=0>)
              outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tgemv.bias` — 矩阵-向量乘法加偏置

```mlir
pto.tgemv.bias ins(<lhs>, <rhs>, <bias> : <lhs_type>, <rhs_type>, <bias_type>)
               outs(<dst> : <dst_type>)
               {accPhase = <phase>}
```

**语义：**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j] + bias[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 矩阵 tile buffer（`loc=left`，`rows=1`） |
| `rhs` | `pto.tile_buf` | 向量 tile buffer（`loc=right`） |
| `bias` | `pto.tile_buf` | 偏置 tile buffer（`loc=bias`，`rows=1`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `lhs`、`rhs`、`dst` 的类型组合约束与 `pto.tgemv` 相同。
  - `bias` 元素类型必须与 `dst` 元素类型一致。
  - `bias` 必须使用 `loc=bias`，且 `rows=1`。
  - 运行时约束与 `pto.tgemv` 相同。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`dst` 的类型组合约束与 `pto.tgemv` 相同。
  - `bias` 元素类型必须与 `dst` 元素类型一致。
  - `bias` 必须使用 `loc=bias`、`rows=1` 和 `blayout=row_major`。
  - 运行时约束与 `pto.tgemv` 相同。

**示例：**

```mlir
pto.tgemv.bias ins(%lhs, %rhs, %bias :
                   !pto.tile_buf<loc=left, dtype=f16, rows=1, cols=128,
                       v_row=1, v_col=128, blayout=col_major,
                       slayout=row_major, fractal=512, pad=0>,
                   !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=16,
                       v_row=128, v_col=16, blayout=row_major,
                       slayout=col_major, fractal=512, pad=0>,
                   !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16,
                       v_row=1, v_col=16, blayout=row_major,
                       slayout=none_box, fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                       v_row=1, v_col=16, blayout=col_major,
                       slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tgemv.mx` — 混合精度矩阵-向量乘法

```mlir
pto.tgemv.mx ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                 <lhs_type>, <lhs_scale_type>, <rhs_type>, <rhs_scale_type>)
             outs(<dst> : <dst_type>)
             {accPhase = <phase>}
```

**语义：**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
// lhs 的 rows（即 m）必须为 1；缩放 tile 配置目标定义的量化行为
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 矩阵 tile buffer（`loc=left`，`rows=1`） |
| `lhs_scale` | `pto.tile_buf` | 左矩阵缩放 tile buffer（`loc=scaling`） |
| `rhs` | `pto.tile_buf` | 向量 tile buffer（`loc=right`） |
| `rhs_scale` | `pto.tile_buf` | 右矩阵缩放 tile buffer（`loc=scaling`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **Atlas A2系列产品/Atlas A3系列产品 不支持此操作。**
- **Ascend 950PR&950DT系列产品 约束**
  - 数据与目标分别位于 `left`、`right`、`acc`，使用本章 Ascend 950PR&950DT系列产品 矩阵布局；`dst` 元素类型为 `f32`。
  - 本节示例使用 `f8E4M3FN` 或 `f8E5M2` 的 FP8 类型对。
  - 有效 `M=1`，有效 `K/N` 在 `[1,4095]`；数据的物理尺寸满足矩阵乘法形状关系。
  - 两个 scale 均位于 `scaling`。令 `G=ceil(K/32)`，其中 K 为有效归约维：左 scale 的物理与有效尺寸均为 `[1,G]`；右 scale 的物理尺寸为 `[G,rhs.cols]`，有效尺寸为 `[G,N]`。
  - 右 scale 的物理列容量覆盖右数据的物理列容量，不能以有效 N 替代已对齐的物理 N。

**示例：**

Ascend 950PR&950DT系列产品 示例：有效区域等于物理尺寸，M=1、K=128、N=16；左 scale 为 `1x4`，右 scale 为 `4x16`。输入数据与缩放因子已准备在对应 Tile 中。

```mlir
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
        v_row=1, v_col=128, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=4,
        v_row=1, v_col=4, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=128, cols=16,
        v_row=128, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=4, cols=16,
        v_row=4, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
      v_row=1, v_col=16, blayout=col_major, slayout=row_major,
      fractal=1024, pad=0>)
```

---

### `pto.tgemv.mx.acc` — 混合精度矩阵-向量乘加

```mlir
pto.tgemv.mx.acc ins(<acc_in>, <lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                     <acc_in_type>, <lhs_type>, <lhs_scale_type>,
                     <rhs_type>, <rhs_scale_type>)
                 outs(<dst> : <dst_type>)
                 {accPhase = <phase>}
```

**语义：**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] = acc_in[i, j] +
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | 先前的累加器值（`loc=acc`） |
| `lhs` | `pto.tile_buf` | 矩阵 tile buffer（`loc=left`，`rows=1`） |
| `lhs_scale` | `pto.tile_buf` | 左矩阵缩放 tile buffer（`loc=scaling`） |
| `rhs` | `pto.tile_buf` | 向量 tile buffer（`loc=right`） |
| `rhs_scale` | `pto.tile_buf` | 右矩阵缩放 tile buffer（`loc=scaling`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accPhase` — 累加阶段控制。默认值为 `unspecified`。
  - `#pto<acc_phase unspecified>` — 默认行为，由实现决定累加策略
  - `#pto<acc_phase partial>` — 部分累加，结果为中间值，后续还会继续累加
  - `#pto<acc_phase final>` — 最终累加，标记本次为最后一次累加

**约束：**

- **Atlas A2系列产品/Atlas A3系列产品 不支持此操作。**

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`lhs_scale`、`rhs_scale`、`dst` 的约束与 `pto.tgemv.mx` 相同。
  - `acc_in` 必须位于 `loc=acc`，其元素类型和形状与 `dst` 一致。
  - 运行时约束与 `pto.tgemv.mx` 相同。

**示例：**

Ascend 950PR&950DT系列产品 示例：有效区域等于物理尺寸，M=1、K=128、N=16；左 scale 为 `1x4`，右 scale 为 `4x16`。输入数据与缩放因子已准备在对应 Tile 中。

```mlir
pto.tgemv.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
        v_row=1, v_col=16, blayout=col_major, slayout=row_major,
        fractal=1024, pad=0>,
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
        v_row=1, v_col=128, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=4,
        v_row=1, v_col=4, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=128, cols=16,
        v_row=128, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=4, cols=16,
        v_row=4, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
      v_row=1, v_col=16, blayout=col_major, slayout=row_major,
      fractal=1024, pad=0>)
```

---

### `pto.tgemv.mx.bias` — 混合精度矩阵-向量乘法加偏置

```mlir
pto.tgemv.mx.bias ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale>, <bias> :
                      <lhs_type>, <lhs_scale_type>, <rhs_type>,
                      <rhs_scale_type>, <bias_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j]) + bias[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | 矩阵 tile buffer（`loc=left`，`rows=1`） |
| `lhs_scale` | `pto.tile_buf` | 左矩阵缩放 tile buffer（`loc=scaling`） |
| `rhs` | `pto.tile_buf` | 向量 tile buffer（`loc=right`） |
| `rhs_scale` | `pto.tile_buf` | 右矩阵缩放 tile buffer（`loc=scaling`） |
| `bias` | `pto.tile_buf` | 偏置 tile buffer（`loc=bias`，`rows=1`） |
| `dst` | `pto.tile_buf` | 目标累加器 tile buffer（`loc=acc`） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **Atlas A2系列产品/Atlas A3系列产品 不支持此操作。**

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `lhs`、`rhs`、`lhs_scale`、`rhs_scale`、`dst` 的约束与 `pto.tgemv.mx` 相同。
  - `bias` 元素类型必须为 `f32`。
  - `bias` 必须使用 `loc=bias`、`rows=1` 和 `blayout=row_major`。
  - 运行时约束与 `pto.tgemv.mx` 相同。

**示例：**

Ascend 950PR&950DT系列产品 示例：有效区域等于物理尺寸，M=1、K=128、N=16；左 scale 为 `1x4`，右 scale 为 `4x16`。输入数据与缩放因子已准备在对应 Tile 中。

```mlir
pto.tgemv.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
        v_row=1, v_col=128, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=4,
        v_row=1, v_col=4, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=128, cols=16,
        v_row=128, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=4, cols=16,
        v_row=4, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16,
        v_row=1, v_col=16, blayout=row_major, slayout=none_box,
        fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
      v_row=1, v_col=16, blayout=col_major, slayout=row_major,
      fractal=1024, pad=0>)
```
