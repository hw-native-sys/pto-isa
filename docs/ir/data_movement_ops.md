# 数据搬运操作

本节描述全局内存与本地缓冲区之间的数据转移、本地内存域之间的数据移动，以及标量读写。Tile 搬运形式采用“目标传递风格”（Destination-Passing Style，DPS），写入预先分配的目标；`pto.load` 返回读取值，`pto.tprefetch_async` 返回异步事件。

数据搬运操作通常涉及以下场景：

- **GM 到本地 Tile 缓冲区转移**：通过 `pto.tload` 和 `pto.tprefetch` 将全局内存分区视图加载到本地 tile buffer
- **异步预取**：使用 `pto.tprefetch_async` 启动 SDMA 驱动的异步预取，并返回同步事件
- **本地缓冲区存储回全局内存**：通过 `pto.tstore` 将 tile buffer 写回全局内存分区，支持原子操作和量化转换
- **带缩放因子的累加器存储**：通过 `pto.tstore` 的 `fp` 参数使用缩放 Tile 对累加器数据进行转换后存储到全局内存
- **聚集/散射操作**：通过 `pto.mgather` 和 `pto.mscatter` 基于索引在全局内存与本地 Tile 之间进行非连续数据搬运，基础形式支持 Atlas A3系列产品/Ascend 950PR&950DT系列产品
- **本地内存域间数据移动**：使用 `pto.tmov` 在 `mat`、`vec`、`acc`、`bias` 等本地存储域之间转移数据
- **Tile 转置**：通过 `pto.ttrans` 对 tile buffer 进行矩阵转置
- **标量读写**：通过 `pto.load` 和 `pto.store` 对指定指针地址空间中的元素进行读写

---

## 目录

- [`pto.tload` — 物理 DMA 加载](#ptotload--物理-dma-加载)
- [`pto.tprefetch` — 预取加载](#ptotprefetch--预取加载)
- [`pto.tprefetch_async` — 异步预取](#ptotprefetch_async--异步预取)
- [`pto.tstore` — Tile 缓冲区存储](#ptotstore--tile-缓冲区存储)
- [`pto.tstore` 的 `fp` 形式](#ptotstore-的-fp-形式)
- [`pto.mgather` — 聚集加载](#ptomgather--聚集加载)
- [`pto.mscatter` — 散射存储](#ptomscatter--散射存储)
- [`pto.load` — 标量加载](#ptoload--标量加载)
- [`pto.store` — 标量存储](#ptostore--标量存储)
- [`pto.tmov` — 本地内存域间数据移动](#ptotmov--本地内存域间数据移动)
- [`pto.ttrans` — Tile 转置](#ptottrans--tile-转置)
- [`pto.tmov` 的 `fp` 形式](#ptotmov-的-fp-形式)

---

## 操作详解

### `pto.tload` — 物理 DMA 加载

```mlir
pto.tload ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.partition_tensor_view<...>` | 源全局内存分区视图 |
| `dst` | `pto.tile_buf` | 目标本地 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - Tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`。
  - `dst` 必须使用 `loc=vec` 或 `loc=mat`。
  - `dst` 和 `src` 的元素类型必须具有相同的位宽。
  - 运行时约束：所有源分区的 extent 必须为正；`dst` 的有效区域必须非负。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 元素类型必须为以下之一：`i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`、`f8E4M3*`、`f8E5M2*`、`!pto.hif8`、`!pto.f4E1M2x2`、`!pto.f4E2M1x2`。
  - `dst` 元素大小必须为 1、2、4 或 8 字节，并与 `src` 匹配。
  - 对于 `i64` 类型，`dst` 的 padding 必须为 null 或零。

**硬件管道：** PIPE_MTE2（GM 至 UB 的 DMA 转移）

**示例：**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.tprefetch` — 预取加载

```mlir
pto.tprefetch ins(<src> : <src_type>)
              outs(<dst> : <dst_type>)
```

**语义：**

```text
TPREFETCH(dst, src)
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.partition_tensor_view<...>` | 源全局内存视图 |
| `dst` | `pto.tile_buf` | 目标本地缓冲区 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `dst` 必须使用 `loc=vec` 或 `loc=mat`。
  - 静态源 extent 必须为正；`dst` 的有效 extent 必须非负。
  - `src` 和 `dst` 的元素类型必须具有相同的元素大小。
  - 低精度类型（`f8*`、`!pto.hif8`、`!pto.f4*`）仅在 Ascend 950PR&950DT系列产品 上受支持。

**硬件管道：** PIPE_MTE2（GM 到本地 tile 的预取）

**示例：**

```mlir
pto.tprefetch ins(%pv : !pto.partition_tensor_view<16x16xf16>)
              outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tprefetch_async` — 异步预取

```mlir
%event = pto.tprefetch_async ins(<src>, <ctx> : <src_type>, !pto.prefetch_async_context)
                             -> !pto.async_event
```

**语义：**

```text
%event = pto.tprefetch_async(%src, %ctx)
启动基于 SDMA 的异步预取，从全局内存到缓存中，并返回异步事件。
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tensor_view` / `pto.partition_tensor_view` | 源全局内存视图，必须是平坦的连续逻辑 1D 视图 |
| `ctx` | `pto.prefetch_async_context` | 预取异步上下文，必须有效 |

**返回值：** `!pto.async_event` — 用于同步异步操作的事件句柄。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `src` 必须是扁平连续的逻辑 1D GM 视图。
  - `ctx` 必须是有效的 `prefetch_async_context`。

**执行方式：** 该操作生成 `TPREFETCH_ASYNC` 调用，由 CANN 的 SDMA 运行时完成异步
传输。`SDMA` 是这里的传输引擎，不是 PTO `PIPE_*` 调度枚举。

**示例：**

```mlir
%ctx = pto.make_prefetch_async_context(%workspace : !pto.ptr<i8>) -> !pto.prefetch_async_context
%event = pto.tprefetch_async(%src, %ctx : !pto.partition_tensor_view<128xf32>,
                             !pto.prefetch_async_context) -> !pto.async_event
```

---

### `pto.tstore` — Tile 缓冲区存储

```mlir
pto.tstore ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)
           {attributes}

// 支持可选的 preQuantScalar 参数：
pto.tstore ins(<src> : <src_type>, <preQuantScalar> : i64)
           outs(<dst> : <dst_type>)
           {attributes}
```

**语义：**

```text
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]

（支持可选的原子操作、ReLU 前置处理和量化转换）
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer，位置为 `vec`、`mat` 或 `acc` |
| `dst` | `pto.partition_tensor_view<...>` | 目标全局内存分区视图 |
| `preQuantScalar` | `i64`（可选）| 量化前的标量值，与 fp 互斥 |
| `fp` | `!pto.tile_buf<loc=scaling, ...>`（可选）| 逐列缩放/量化参数，见本节 fp 形式 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `stPhase` — `PTO_STPhaseAttr` 存储阶段属性，默认值为
  `#pto<st_phase unspecified>`。

  | 文本属性值 | 含义 |
  | --- | --- |
  | `#pto<st_phase unspecified>` | 未指定存储阶段 |
  | `#pto<st_phase partial>` | 部分存储（累加中间值） |
  | `#pto<st_phase final>` | 最终存储（完成累加） |

- `atomicType` — 原子操作类型。默认值为 `atomic_none`。
  - `#pto<atomic_type atomic_none>` — 无原子操作
  - `#pto<atomic_type atomic_add>` — 原子加法（dst += src）

- `reluPreMode` — ReLU 前置处理模式。默认值为 `no_relu`。
  - `#pto<relu_pre_mode no_relu>` — 不进行 ReLU 处理
  - `#pto<relu_pre_mode normal_relu>` — 应用标准 ReLU（max(0, x)）

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src.loc` 必须为 `vec`、`mat` 或 `acc`。
  - 对于 `loc=vec` 或 `loc=mat`：不允许 `preQuantScalar`；`src` 元素类型必须为 `i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`；位宽必须匹配。
  - 对于 `loc=acc`：`src` 必须为 `i32` 或 `f32`。
  - 不带 `preQuantScalar` 时：`dst` 为 `i32`、`f32`、`f16` 或 `bf16`。
  - 带 `preQuantScalar` 时：`src=i32` → `dst=i8` 或 `f16`；`src=f32` → `dst=i8`。
  - 静态列数 `1 <= cols <= 4095`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src.loc` 必须为 `vec` 或 `acc`。
  - 对于 `loc=vec`：不允许 `preQuantScalar`；`src` 元素类型必须为 `i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`、`f8*`、`!pto.hif8`、`!pto.f4*`。
  - 对于 `loc=acc`：`src` 必须为 `i32` 或 `f32`。
  - 不带 `preQuantScalar` 时：`dst` 为 `i32`、`f32`、`f16` 或 `bf16`。
  - 带 `preQuantScalar` 时：`src=i32` → `dst=i8`、`f16` 或 `bf16`；`src=f32` → `dst=i8`、`f16`、`bf16`、`f32` 或 `!pto.hif8` 或 `f8E4M3*`。

**硬件管道：**

- `loc=acc` 使用 PIPE_FIX（浮点修复）
- `loc=vec` / `loc=mat` 使用 PIPE_MTE3（MTE3 存储）

**示例：**

```mlir
// 基本存储
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)

// 带最终阶段标记的存储
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {stPhase = #pto<st_phase final>}

// 带原子加法的存储
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {atomicType = #pto<atomic_type atomic_add>}

// Accumulator 到 mat 的存储，带原子操作和 ReLU
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32,
                   v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                   fractal=1024, pad=0>)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}
```

---

### `pto.tstore` 的 `fp` 形式

```mlir
pto.tstore ins(<src> : <src_type> fp <fp> : <fp_type>)
  outs(<dst> : <dst_type>) {attributes}
```

**语义：** 使用逐列缩放/量化参数 fp 对累加器的有效区域进行转换，并存储到 GM 目标视图。fp 是独立的缩放 Tile，不是新的数据源或标量字节地址。

**参数与约束：** src 位于 `acc`，元素类型为 `f32` 或 32 位整数；fp 位于 `scaling`；dst 为 GM 分区视图。fp 与 preQuantScalar 互斥；源布局使用 `col_major/row_major`，有效列数在 `[1,4095]`；精度转换、ReLU、原子和阶段属性遵循 `tstore` 的相应规则。没有 SSA 返回值。

**示例：**

```mlir
pto.tstore
  ins(%acc : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32,
    v_row=16, v_col=32, blayout=col_major, slayout=row_major,
    fractal=1024, pad=0>
    fp %fp : !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.partition_tensor_view<16x32xf16>)
```

---

### `pto.mgather` — 聚集加载

```mlir
pto.mgather ins(<mem>, <idx> : <mem_type>, <idx_type>)
  outs(<dst> : <dst_type>) {coalesce = #pto<coalesce row|elem>}

// GM -> L1 的元素模式另需 GM 临时视图。
pto.mgather ins(<mem>, <idx>, <scratch> : <mem_type>, <idx_type>, <scratch_type>)
  outs(<dst> : <dst_type>) {coalesce = #pto<coalesce elem>}
```

**语义：** 从 GM 视图按行索引或逐元素索引读取数据，写入目标 Tile。

```text
coalesce=row:
    dst[r,j] = mem[row_index[r],j]
coalesce=elem:
    dst[r,j] = flat_mem[indexes[r,j]]
```

row 模式的索引单位为源行；elem 模式的索引单位为源元素，按 GM 表的连续元素序号寻址。`coalesce` 必须显式指定，不能省略为默认模式。

**参数与返回值：**

| 参数 | 类型与角色 |
| --- | --- |
| mem | GM `!pto.partition_tensor_view`，源数据表 |
| idx | 32 位整数索引；GM→VEC 使用 `vec` Tile，GM→L1 使用 GM 分区视图 |
| scratch | 仅 GM→L1 的 elem 模式使用，元素类型与目标相同的 GM 分区视图 |
| dst | 预先分配的 `vec` 或 `mat` Tile |

无 SSA 返回值。

**属性：**

- `coalesce = #pto<coalesce row>`：按行读取；`#pto<coalesce elem>`：逐元素读取。
- `gatherOob` 默认为 `#pto<gather_oob undefined>`，要求索引在有效范围内；Ascend 950PR&950DT系列产品 可显式选择 `clamp`（钳位）、`wrap`（环绕）或 `zero`（越界结果为零）。

**约束：**

- Atlas A3系列产品/Ascend 950PR&950DT系列产品 均支持基础聚集形式；非默认越界模式仅用于 Ascend 950PR&950DT系列产品。
- mem/dst 元素类型相同，支持 8/16/32 位整数、`f16`、`bf16`、`f32`；Ascend 950PR&950DT系列产品 额外支持 FP8 和 HiFloat8 类型。
- 源 GM 分区在可推断布局时使用 ND。
- GM→VEC：dst 为 `row_major/none_box`；idx 位于 `vec`，使用 `none_box`。row 模式的有效索引形状为 row_major 的 `[1,R]` 或 col_major 的 `[R,1]`，R 为目标有效行数；elem 模式与目标有效形状相同。不提供 scratch。
- GM→L1：dst 位于 `mat`，使用 `col_major/row_major`、`fractal=512`；物理行数为 16 的倍数，物理列数为 `32/sizeof(dtype)` 的倍数。idx 是 GM 分区视图，不能用 VEC Tile；row 模式不提供 scratch，elem 模式必须提供连续且容量足够的 GM scratch。
- 索引值和临时空间容量由调用者保证；默认 undefined 越界模式不提供越界保护。

**示例：**

```mlir
// Atlas A3系列产品/Ascend 950PR&950DT系列产品：从 1024 行表中聚集 32 行。
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<1024x32xi32>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  {coalesce = #pto<coalesce row>}
```

```mlir
// Ascend 950PR&950DT系列产品：相同的行聚集，越界行补零。
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<1024x32xi32>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  {coalesce = #pto<coalesce row>, gatherOob = #pto<gather_oob zero>}
```

---

### `pto.mscatter` — 散射存储

```mlir
pto.mscatter ins(<src>, <idx> : <src_type>, <idx_type>)
  outs(<mem> : <mem_type>) {coalesce = #pto<coalesce row|elem>}
```

**语义：** 从 VEC Tile 读取数据，按行索引或逐元素索引写入 GM 目标表。

```text
coalesce=row:
    mem[row_index[r],j] = src[r,j]
coalesce=elem:
    flat_mem[indexes[r,j]] = src[r,j]
// 原子模式将赋值替换为指定的 add/max/min 更新。
```

row 模式的索引单位为目标行，elem 模式为目标元素。未写入的 GM 位置保持原值。无原子模式时，用户应避免重复目标索引，不能依赖冲突写入的先后顺序。

**参数与返回值：** src 为 `vec` Tile，idx 为 `vec` 中的 32 位整数索引 Tile，mem 为 GM `!pto.partition_tensor_view`；没有 SSA 返回值。

**属性：**

- `coalesce`：显式 `row` 或 `elem`。省略时仅支持从索引形状区分的基础形式：row_major 的 `[R,1]` 行索引，或与 src 有效形状相同的元素索引。建议显式写出模式。
- `scatterAtomicOp`：默认 `#pto<scatter_atomic_op none>`，Ascend 950PR&950DT系列产品 可选择 `add`、`max`、`min`。add 支持 32 位整数、`f16`、`f32`；max/min 支持 32 位整数、`f32`。
- `scatterOob`：默认 `#pto<scatter_oob undefined>`；Ascend 950PR&950DT系列产品 可选择 `skip`（跳过越界）、`clamp`（钳位）、`wrap`（环绕）。
- 指定非默认 atomic/oob 或 scatterConflict 时，必须显式指定 coalesce；这些扩展不改变索引单位。

**约束：**

- Atlas A3系列产品/Ascend 950PR&950DT系列产品 均支持基础形式；非默认原子和越界模式仅用于 Ascend 950PR&950DT系列产品。
- src/mem 元素类型相同，支持 8/16/32 位整数、`f16`、`bf16`、`f32`；Ascend 950PR&950DT系列产品 额外支持 FP8 和 HiFloat8。
- src 使用 `row_major/none_box`；idx 使用 `none_box`；GM 视图在可推断时为 ND。
- 显式 row 模式的有效索引形状为 row_major 的 `[1,R]` 或 col_major 的 `[R,1]`，R 为源有效行数；显式 elem 模式的索引有效形状与 src 相同。
- 默认 undefined 模式不保护越界访问，调用者应提供有效范围内的索引。

**示例：**

```mlir
// Atlas A3系列产品/Ascend 950PR&950DT系列产品：将 32 行写到 GM 表中的指定行。
pto.mscatter ins(%src, %idx :
  !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%mem : !pto.partition_tensor_view<1024x32xi32>)
  {coalesce = #pto<coalesce row>}
```

```mlir
// Ascend 950PR&950DT系列产品：原子累加并跳过越界行。
pto.mscatter ins(%src, %idx :
  !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%mem : !pto.partition_tensor_view<1024x32xi32>)
  {coalesce = #pto<coalesce row>,
   scatterAtomicOp = #pto<scatter_atomic_op add>,
   scatterOob = #pto<scatter_oob skip>}
```

---

### `pto.load` — 标量加载

```mlir
%value = pto.load <ptr>[<offset>] : <ptr_type> -> <element_type>
```

**语义：** 返回 `ptr[offset]`，不改变存储。

**参数与返回值：** ptr 为 `!pto.ptr<T,space>` 或 `memref`；offset 为 `index`，以指针元素为单位，不能传入字节偏移；结果类型必须等于指针或 memref 的元素类型。访问地址须有效且满足元素对齐与容量要求。

**示例：**

```mlir
%offset = pto.constant 3 : index
%value = pto.load %ptr[%offset] : !pto.ptr<f32> -> f32
```

这里读取第 4 个 f32 元素，即相对基地址 12 字节的位置。

---

### `pto.store` — 标量存储

```mlir
pto.store <value>, <ptr>[<offset>] : <ptr_type>, <element_type>
```

**语义：** 将 value 写到 `ptr[offset]`；没有 SSA 返回值。

**参数与约束：** ptr 为 `!pto.ptr<T,space>` 或 `memref`；offset 为以元素计数的 `index`；value 类型必须等于指针或 memref 的元素类型。类型列表先写指针类型，再写值类型；地址有效性、对齐与容量由调用者保证。

**示例：**

```mlir
%offset = pto.constant 3 : index
pto.store %value, %ptr[%offset] : !pto.ptr<f32>, f32
```

---

### `pto.tmov` — 本地内存域间数据移动

```mlir
// 基本形式
pto.tmov ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)

// 带 fp（scaling 缓冲区）和属性的形式
pto.tmov ins(<src> : <src_type>, <fp> : <fp_type>)
         outs(<dst> : <dst_type>)
         {accToVecMode = ..., reluPreMode = ...}

// 带 preQuantScalar 的形式
pto.tmov ins(<src> : <src_type>, <preQuantScalar> : i64)
         outs(<dst> : <dst_type>)
         {attributes}
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, j]

（支持可选的精度转换、ReLU 前置处理和量化）
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer，位置为 `mat`、`vec` 或 `acc` |
| `dst` | `pto.tile_buf` | 目标 tile buffer，位置为 `left`、`right`、`bias`、`scaling` 等 |
| `fp` | `pto.tile_buf<loc=scaling>`（可选）| 浮点精度缓冲区，仅在特定转换中使用 |
| `preQuantScalar` | `i64`（可选）| 量化前的标量值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `accToVecMode` — Accumulator 到 vec 的转换模式。仅在 `src.loc=acc` 且 `dst.loc=vec` 时使用。
  - `#pto<acc_to_vec_mode single_mode_vec0>` — 单向量执行单元模式，选择 vec0
  - `#pto<acc_to_vec_mode single_mode_vec1>` — 单向量执行单元模式，选择 vec1
  - `#pto<acc_to_vec_mode dual_mode_split_m>` — 双向量执行单元模式，沿行维 M 拆分
  - `#pto<acc_to_vec_mode dual_mode_split_n>` — 双向量执行单元模式，沿列维 N 拆分

- `reluPreMode` — ReLU 前置处理模式。默认值为 `no_relu`。仅当 `src.loc=acc` 时支持。
  - `#pto<relu_pre_mode no_relu>` — 不进行 ReLU 处理
  - `#pto<relu_pre_mode normal_relu>` — 应用标准 ReLU（max(0, x)）

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 静态 shape 必须匹配。
  - 支持的位置对：`mat` → `left`/`right`/`bias`/`scaling`；`vec` → `vec`；`acc` → `mat`；`acc` → `vec`。
  - `accToVecMode` 仅用于 `acc` → `vec` 转换。
  - `reluPreMode`、`fp`、`preQuantScalar` 仅在 `src.loc=acc` 时支持；`fp` 与 `preQuantScalar` 互斥。
  - `fp` 必须使用 `loc=scaling`；带 `fp` 的源元素类型必须为 `f32` 或 32 位整数。
  - `src.loc=acc` 且使用 `fp` 或 ReLU 时，源布局必须为 `blayout=col_major, slayout=row_major`。
  - `acc` → `mat` 的目标 `fractal` 必须为 `512`；带 `fp` 时目标布局也必须为 `blayout=col_major, slayout=row_major`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 支持的位置对为：`mat` → `left`/`right`/`bias`/`scaling`；`vec` → `vec`/`mat`；`acc` → `vec`/`mat`。
  - `preQuantScalar`、`fp`、ReLU 与 `fp.loc=scaling` 的约束同上。
  - `src.loc=acc` 且使用 `fp` 或 ReLU 时，源布局必须为 `blayout=col_major, slayout=row_major`。
  - Ascend 950PR&950DT系列产品 不要求 `acc` → `mat` 的目标 `fractal=512`。

**硬件管道：**

- `vec` → `vec` 使用 PIPE_V（向量管道）
- `mat` → `left`/`right`/`bias`/`scaling` 使用 PIPE_MTE1（MTE1 转移）
- `acc` → `mat`/`vec` 使用 PIPE_FIX（浮点修复）

**示例：**

```mlir
// 基本 acc 到 vec 转换
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=col_major, slayout=row_major,
                  fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
```

```mlir
// 带 ReLU 的 acc 到 vec 转换
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                  v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                  fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                  v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         {reluPreMode = #pto<relu_pre_mode normal_relu>}
```

```mlir
// mat 到 scaling 的移动
pto.tmov ins(%src : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
```

---

### `pto.ttrans` — Tile 转置

```mlir
pto.ttrans ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[j, i]

使用临时缓冲区 tmp 完成转置操作。
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer，必须使用 `blayout=row_major` |
| `tmp` | `pto.tile_buf` | 临时工作缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - `src` 和 `dst` 的元素类型必须匹配。
  - `src` 必须使用 `blayout=row_major`。
  - 元素大小必须为 1、2 或 4 字节。
  - 支持的类型：`i32`/`f32`（4 字节）、`i16`/`f16`/`bf16`（2 字节）、`i8`（1 字节）。
  - 转置在 `src` 的有效区域上进行。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - `src` 和 `dst` 的元素大小必须匹配。
  - 主维度上需要 32 字节对齐。
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 转置在静态 tile shape 上操作。

**硬件管道：** PIPE_V（向量管道）

**示例：**

```mlir
pto.ttrans ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                       v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                       fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                       v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                       fractal=512, pad=0>)
           outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tmov` 的 `fp` 形式

```mlir
pto.tmov ins(<src> : <src_type>, <fp> : <fp_type>)
  outs(<dst> : <dst_type>) {attributes}
```

**语义：** 使用缩放 Tile fp 对累加器数据进行支持的精度转换，写入目标本地 Tile；不产生 SSA 返回值。

**约束：** src 位于 acc，元素类型为 f32 或 32 位整数；fp 位于 scaling；fp 与 preQuantScalar 互斥。源使用 col_major/row_major 布局，目标路径与布局、转换类型及 ReLU 遵循 `tmov` 的约束。Atlas A3系列产品 的 acc→mat 目标 fractal 为 512，源与目标物理形状匹配；Ascend 950PR&950DT系列产品 不要求该目标 fractal 值。

**示例：**

```mlir
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
    v_row=16, v_col=256, blayout=col_major, slayout=row_major,
    fractal=1024, pad=0>, %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=16, cols=256,
    v_row=16, v_col=256, blayout=row_major, slayout=row_major,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=16, cols=256,
    v_row=16, v_col=256, blayout=col_major, slayout=row_major,
    fractal=512, pad=0>)
```
