# 向量指令集：SFU 与 DSA 操作

这里定义 PTO ISA 中那些既不是普通二元算术、也不是单纯 load/store 的向量特化操作。它们覆盖特殊函数、融合算子、扩宽乘加、索引生成，以及少量直接作用在向量 tile buffer 上的专用加速器操作。

> **类别：** 特殊函数单元与领域专用加速器操作
> **流水线：** `PIPE_V` / SFU

这类操作通常比通用算术更窄，也更强依赖目标平台特性。因此页面会明确写出它们是否只在 A5（Ascend 950 PR / DT）上稳定存在，以及 lowering 时哪些语义不能被拆散。

---

## 通用操作数模型

- `%input`、`%lhs`、`%rhs`、`%acc`、`%alpha`：不同指令使用的源 SSA 值
- `%mask`：谓词操作数 `Pg`
- `%result`：目标 SSA 值

这一页混合了几种底层形态：

- 纯 `vreg -> vreg` 形式
- “计算 + 转换”融合形式
- 直接在向量 tile buffer 上工作的 UB-to-UB 帮助指令

因此每条指令的小节都会单独写明自己属于哪种存储模型。

---

## 融合激活

### `pto.vlrelu`

- **语法：** `%result = pto.vlrelu %input, %alpha, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **A5 类型：** f16、f32
- **语义：** Leaky ReLU，斜率由标量 `%alpha` 给出。

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha * src[i];
```

### `pto.vprelu`

- **语法：** `%result = pto.vprelu %input, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 类型：** f16、f32
- **语义：** Parametric ReLU，斜率是逐元素向量。

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

### `pto.vexpdiff`

- **语法：** `%result = pto.vexpdiff %input, %max : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 类型：** f16、f32
- **语义：** 计算 `exp(x - max)` 的融合形式，典型用途是数值稳定版 softmax。

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

这类融合形式的意义不只是“少一条指令”，还包括把数值路径固定成硬件期望的顺序。

---

## 融合计算 + 转换

### `pto.vaddrelu`

- **语法：** `%result = pto.vaddrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 类型：** f16、f32
- **语义：** 先相加，再过 ReLU。

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] + src1[i], 0);
```

### `pto.vsubrelu`

- **语法：** `%result = pto.vsubrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 类型：** f16、f32
- **语义：** 先相减，再过 ReLU。

### `pto.vaxpy`

- **语法：** `%result = pto.vaxpy %src0, %src1, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **A5 类型：** f16、f32
- **语义：** 执行 AXPY：`alpha * src0 + src1`。

### `pto.vaddreluconv`

- **语法：** `%result = pto.vaddreluconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **语义：** 先做加法和 ReLU，再执行类型转换。

合法性重点在于：

- 并不是任意源 / 目标类型对都支持
- 舍入、饱和、打包方式由这条融合指令定义，不能随便拆成普通 `vadd`、`vrelu`、`vcvt`

### `pto.vmulconv`

- **语法：** `%result = pto.vmulconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **语义：** 先乘法，再做硬件支持的类型转换。

这类形式的关键不是“乘法后再转型”这么简单，而是它把转换时机、舍入和打包路径锁定成一个不可随意改写的硬件语义。

---

## 扩展算术

### `pto.vmull`

- **语法：** `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **A5 类型：** i32 / u32 的原生 32×32→64 扩宽乘
- **语义：** 返回扩宽乘积的低半部分和高半部分。

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

### `pto.vmula`

- **语法：** `%result = pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 乘加融合。

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = acc[i] + lhs[i] * rhs[i];
```

`vmula` 不是总能用 `vmul + vadd` 替代。它是否保留更强的融合时序、舍入路径或吞吐特征，属于必须被后端保真的部分。

---

## 索引生成

### `pto.vci`

- **语法：** `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- **语义：** 生成一个逐 lane 索引向量。

```c
for (int i = 0; i < N; i++)
    dst[i] = base_index + i;
```

这类操作常用于 gather/scatter、排序、argsort 等场景。虽然 `vci` 也会出现在转换类页面里，但在这里更适合作为“索引物化原语”来理解。

---

## 直接作用于向量 tile buffer 的操作

### `pto.vtranspose`

- **语法：** `pto.vtranspose %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64`
- **语义：** 在向量 tile buffer 上直接做转置，不经过 `vreg -> vreg` 路径。

这里最重要的边界是：虽然名字在 `pto.v*` 命名空间里，但它不是普通向量寄存器运算，而是一个依赖控制字与布局约束的 UB-to-UB 加速器操作。

---

## 排序操作

### `pto.vsort32`

- **语法：** `pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64`
- **语义：** 对向量 tile buffer 中的 32 个元素排序。

### `pto.vbitsort`

- **语法：** `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, index`
- **语义：** 按分数降序排序 32 个 region proposal，并把排序后的记录写入 `%dest`。

要点：

- 每条输出记录 8 字节
- 高 4 字节是 index，低 4 字节是 score
- 相同 score 的 tie 保持稳定
- 所有指针都必须指向向量 tile buffer
- 这是 A5 特有的 `VBS32` 硬件单元能力

### `pto.vmrgsort`

- **语法：** `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub> x4, i64, i64`
- **语义：** 对四个已经排好序的输入流做 merge-sort。

约束：

- 四个输入必须已经按 `%config` 约定的方向排好序
- 页面正文常简写成 `pto.vmrgsort`，但当前实现摘要里仍保留 `pto.vmrgsort4`

---

## 当前实现摘要

当前文档覆盖的主要形式包括：

- `pto.vmull %lhs, %rhs, %mask`
- `pto.vmula %acc, %lhs, %rhs, %mask`
- `pto.vci %index {order = "ORDER"}`
- `pto.vbitsort %dest, %src, %indices, %repeat_times`
- `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config`

---

## 典型用法

```mlir
// Softmax：先减去最大值，再走 expdiff
%max_broadcast = pto.vlds %ub_max[%c0] {dist = "BRC_B32"}
    : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%exp_stable = pto.vexpdiff %logits, %max_broadcast
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Leaky ReLU
%activated = pto.vlrelu %linear_out, %alpha_scalar, %mask
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// 残差加法 + ReLU
%residual = pto.vaddrelu %conv_out, %skip_connection
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// 为 argsort 生成索引
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

---

## 相关页面

- [归约操作](./reduction-ops_zh.md)
- [向量指令面](../instruction-surfaces/vector-instructions_zh.md)
- [数据重排](./data-rearrangement_zh.md)
