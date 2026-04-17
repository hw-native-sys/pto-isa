# 向量指令集：DMA 拷贝

PTO指令集架构中的向量 DMA 指令负责在 GM 与向量可见的 tile buffer 之间搬运数据。当前硬件用 UB 实现这个向量 tile buffer，因此这组指令既是数据搬运原语，也是 `pto.v*` 计算能否开始执行的前置条件。

> **类别：** DMA 传输配置与执行
> **流水线：** MTE2（GM→向量 tile buffer）、MTE3（向量 tile buffer→GM）

GM 与向量 tile buffer 之间的数据搬运由 MTE 引擎异步执行。MTE2、MTE3 与 `PIPE_V` 并不自动串行，因此 DMA 指令通常要和[向量流水线同步](./pipeline-sync_zh.md)页面里的事件或 buffer 协议配合使用。

MTE2 / MTE3 DMA 引擎的执行模型本质上是一个多层嵌套循环。拷贝指令描述最内层 burst 行为，外层循环次数与步长则通过一组独立的配置指令写入硬件循环寄存器。

---

## GM→向量 tile buffer 的循环与步长配置

这组指令配置 MTE2 的硬件循环，用于 GM→向量 tile buffer 的搬运。它们必须先于 `pto.copy_gm_to_ubuf` 执行。

### `pto.set_loop_size_outtoub`

- **语法：** `pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64`
- **语义：** 配置 GM→向量 tile buffer DMA 的两级硬件循环迭代次数。

| 参数 | 位宽 | 说明 |
|------|------|------|
| `%loop1_count` | 21 位 | 内层硬件循环次数 |
| `%loop2_count` | 21 位 | 外层硬件循环次数 |

如果这次搬运不需要多级循环，两个计数都应设为 `1`。

### `pto.set_loop2_stride_outtoub`

- **语法：** `pto.set_loop2_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **语义：** 配置外层循环 `loop2` 每轮结束后 GM 源地址与向量 tile buffer 目标地址的推进距离。

| 参数 | 位宽 | 说明 |
|------|------|------|
| `%src_stride` | 40 位 | 每次 `loop2` 结束后 GM 读指针前进的字节数 |
| `%dst_stride` | 21 位 | 每次 `loop2` 结束后向量 tile buffer 写指针前进的字节数 |

### `pto.set_loop1_stride_outtoub`

- **语法：** `pto.set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **语义：** 配置内层循环 `loop1` 每轮结束后的地址推进距离。

| 参数 | 位宽 | 说明 |
|------|------|------|
| `%src_stride` | 40 位 | 每次 `loop1` 结束后 GM 读指针前进的字节数 |
| `%dst_stride` | 21 位 | 每次 `loop1` 结束后向量 tile buffer 写指针前进的字节数 |

---

## 向量 tile buffer→GM 的循环与步长配置

这组指令配置 MTE3 的硬件循环，用于向量 tile buffer→GM 的回写。它们必须先于 `pto.copy_ubuf_to_gm` 执行。

向量 tile buffer 地址空间较小，因此相关 stride 字段只需要 21 位；GM 地址范围更大，因此 GM 侧 stride 使用 40 位。

### `pto.set_loop_size_ubtoout`

- **语法：** `pto.set_loop_size_ubtoout %loop1_count, %loop2_count : i64, i64`
- **语义：** 配置向量 tile buffer→GM DMA 的两级硬件循环迭代次数。

| 参数 | 位宽 | 说明 |
|------|------|------|
| `%loop1_count` | 21 位 | 内层硬件循环次数 |
| `%loop2_count` | 21 位 | 外层硬件循环次数 |

### `pto.set_loop2_stride_ubtoout`

- **语法：** `pto.set_loop2_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **语义：** 配置外层循环 `loop2` 的地址推进量。

| 参数 | 位宽 | 说明 |
|------|------|------|
| `%src_stride` | 21 位 | 每次 `loop2` 结束后向量 tile buffer 源地址前进的字节数 |
| `%dst_stride` | 40 位 | 每次 `loop2` 结束后 GM 目标地址前进的字节数 |

### `pto.set_loop1_stride_ubtoout`

- **语法：** `pto.set_loop1_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **语义：** 配置内层循环 `loop1` 的地址推进量。

| 参数 | 位宽 | 说明 |
|------|------|------|
| `%src_stride` | 21 位 | 每次 `loop1` 结束后向量 tile buffer 源地址前进的字节数 |
| `%dst_stride` | 40 位 | 每次 `loop1` 结束后 GM 目标地址前进的字节数 |

---

## DMA 传输执行指令

### `pto.copy_gm_to_ubuf`

- **语法：**
```mlir
pto.copy_gm_to_ubuf %gm_src, %ub_dst,
    %sid, %n_burst, %len_burst, %left_padding, %right_padding,
    %data_select_bit, %l2_cache_ctl, %src_stride, %dst_stride
    : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```
- **语义：** 从 GM 把一组 burst 行搬运到向量 tile buffer。

| 参数 | 说明 |
|------|------|
| `%gm_src` | GM 源指针（`!pto.ptr<T, gm>`） |
| `%ub_dst` | 向量 tile buffer 目标指针（`!pto.ptr<T, ub>`，要求 32B 对齐） |
| `%sid` | Stream ID，通常为 0 |
| `%n_burst` | burst 行数 |
| `%len_burst` | 每一行连续搬运的有效字节数 |
| `%left_padding` | 左侧 padding 字节数 |
| `%right_padding` | 右侧 padding 字节数 |
| `%data_select_bit` | padding / 数据选择控制位 |
| `%l2_cache_ctl` | L2 cache 分配控制位，当前文档仍保留为待细化字段 |
| `%src_stride` | GM 源行的起始地址到下一行起始地址的字节距离 |
| `%dst_stride` | 向量 tile buffer 目标行的起始地址到下一行起始地址的字节距离，要求 32B 对齐 |

### `pto.copy_ubuf_to_gm`

- **语法：**
```mlir
pto.copy_ubuf_to_gm %ub_src, %gm_dst,
    %sid, %n_burst, %len_burst, %reserved, %dst_stride, %src_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64, i64, i64
```
- **语义：** 从向量 tile buffer 回写到 GM。MTE3 每一行只读取 `len_burst` 指定的有效字节，不会把前面装载时补到 32B 边界的 padding 一并写回。

| 参数 | 说明 |
|------|------|
| `%ub_src` | 向量 tile buffer 源指针（`!pto.ptr<T, ub>`，要求 32B 对齐） |
| `%gm_dst` | GM 目标指针（`!pto.ptr<T, gm>`） |
| `%sid` | Stream ID，通常为 0 |
| `%n_burst` | burst 行数 |
| `%len_burst` | 每行写回的有效字节数 |
| `%reserved` | 保留字段，应设为 0 |
| `%dst_stride` | GM 目标行的起始地址跨度 |
| `%src_stride` | 向量 tile buffer 源行的起始地址跨度，要求 32B 对齐 |

### `pto.copy_ubuf_to_ubuf`

- **语法：**
```mlir
pto.copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64 x5
```
- **语义：** 在向量 tile buffer 内部执行 DMA 风格的块搬运。

| 参数 | 说明 |
|------|------|
| `%source` | 向量 tile buffer 源指针 |
| `%dest` | 向量 tile buffer 目标指针 |
| `%sid` | Stream ID |
| `%n_burst` | burst 次数 |
| `%len_burst` | 每次 burst 的长度 |
| `%src_stride` | 源行跨度 |
| `%dst_stride` | 目标行跨度 |

---

## burst / stride / padding 模型

A5 的 DMA 地址模型是纯 stride 模型。这里的 stride 指“这一行起始地址到下一行起始地址的距离”，不是“有效数据长度之外的空洞长度”。因此约束是 `stride >= len_burst`，而不是 `gap = stride - len_burst` 另外单独编码。

### 关键术语

```text
burst  = 每一行连续搬运的 lenBurst 字节
stride = row[r] 起始地址到 row[r+1] 起始地址的字节距离
pad    = ub_stride - lenBurst，用来把行尾补到 32B 对齐边界
```

### 对齐约束

- 向量 tile buffer 地址无论作为源还是目标，都必须 32 字节对齐。
- 在 GM→向量 tile buffer 的方向上，如果 `data_select_bit = true`，则每一行有效数据之后会用 `pad_val` 补到 `dst_stride` 指定的 32B 对齐边界。
- 在向量 tile buffer→GM 的方向上，MTE3 每行只读取 `len_burst` 字节，因此装载时补进去的 padding 会被自然丢弃，不会写回 GM。

### GM→向量 tile buffer 的二维示意

```text
GM（源，!pto.ptr<T, gm>）：

          |<--- src_stride（起始到起始） --->|
          |<- len_burst ->|                 |
第 0 行： [##DATA########]..................|
第 1 行： [##DATA########]..................|
第 2 行： [##DATA########]..................|
          ...
第 N-1 行：[##DATA########]

向量 tile buffer（目标，!pto.ptr<T, ub>，32B 对齐）：

          |<---------- dst_stride（32B 对齐） ---------->|
          |<- len_burst ->|<- 补到 32B 边界的 pad ->|   |
第 0 行： [##DATA########][000000 PAD 000000000000000]
第 1 行： [##DATA########][000000 PAD 000000000000000]
第 2 行： [##DATA########][000000 PAD 000000000000000]
          ...
第 N-1 行：[##DATA########][000000 PAD 000000000000000]

N      = n_burst
stride = row[r] 起始地址到 row[r+1] 起始地址的距离
pad    = 当 data_select_bit=true 时，用 pad_val 补到 32B 边界
```

### 向量 tile buffer→GM 的二维示意

```text
向量 tile buffer（源，!pto.ptr<T, ub>，32B 对齐）：

          |<---------- src_stride（32B 对齐） --------->|
          |<- len_burst ->|<-- pad（读取时忽略） -->| |
第 0 行： [##DATA########][000 pad 000000000000000000]
第 1 行： [##DATA########][000 pad 000000000000000000]
第 2 行： [##DATA########][000 pad 000000000000000000]
          ...
第 N-1 行：[##DATA########][000 pad 000000000000000000]

GM（目标，!pto.ptr<T, gm>）：

          |<--- dst_stride（起始到起始） --->|
          |<- len_burst ->|                 |
第 0 行： [##DATA########]..................|
第 1 行： [##DATA########]..................|
第 2 行： [##DATA########]..................|
          ...
第 N-1 行：[##DATA########]

N = n_burst
MTE3 每行只读 len_burst 字节，因此不会把 padding 写回 GM。
```

---

## 多级循环的精确定义

完整 DMA 搬运是一个三层嵌套循环：外层两级由 `set_loop_size_*` 和 `set_loop*_stride_*` 配置，最内层由拷贝指令的 burst 参数控制。

### GM→向量 tile buffer

```c
for (int j = 0; j < loop2_count; j++) {
    uint8_t *gm1 = gm_src + j * loop2_src_stride;
    uint8_t *ub1 = ub_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {
        uint8_t *gm2 = gm1 + k * loop1_src_stride;
        uint8_t *ub2 = ub1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {
            memcpy(ub2 + r * dst_stride,
                   gm2 + r * src_stride,
                   len_burst);
            if (data_select_bit)
                memset(ub2 + r * dst_stride + len_burst,
                       pad_val, dst_stride - len_burst);
        }
    }
}
```

### 向量 tile buffer→GM

```c
for (int j = 0; j < loop2_count; j++) {
    uint8_t *ub1 = ub_src + j * loop2_src_stride;
    uint8_t *gm1 = gm_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {
        uint8_t *ub2 = ub1 + k * loop1_src_stride;
        uint8_t *gm2 = gm1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {
            memcpy(gm2 + r * dst_stride,
                   ub2 + r * src_stride,
                   len_burst);
        }
    }
}
```

---

## 示例 1：把 32×32 的 f32 tile 从 GM 装入向量 tile buffer

这是最简单的二维搬运形态，对应 `abs_kernel_2d` 这样的测试。

```text
GM 布局（32 × 32 f32，连续）：

    |<- len_burst = 128B（32 × 4） ->|
    |<- src_stride = 128B ---------->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

向量 tile buffer 布局（32 × 32 f32，32B 对齐，连续）：

    |<- dst_stride = 128B（天然满足 32B 对齐） ->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31
```

```mlir
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64

pto.copy_gm_to_ubuf %arg0, %ub_in,
    %c0_i64,
    %c32_i64,
    %c128_i64,
    %c0_i64,
    %c0_i64,
    %false,
    %c0_i64,
    %c128_i64,
    %c128_i64
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

---

## 示例 2：从大矩阵中切出一个 64×128 的 f16 tile

从 1024×512 的 GM 矩阵里抽取一个 64×128 tile，装入向量 tile buffer。

```text
GM 中每一行总宽度 = 512 × 2 = 1024B
tile 每一行有效数据 = 128 × 2 = 256B
因此：
len_burst  = 256B
src_stride = 1024B
dst_stride = 256B
```

```mlir
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,
    %c64_i64,
    %c256_i64,
    %c0_i64,
    %c0_i64,
    %false,
    %c0_i64,
    %c1024_i64,
    %c256_i64
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

---

## 示例 3：带 padding 的 GM→向量 tile buffer 装载

把 100 列有效的 f16 数据装入一个逻辑宽度为 128 的向量 tile。剩余 28 列由 DMA 在向量 tile buffer 行尾补齐。

```text
len_burst  = 100 × 2 = 200B
src_stride = 200B
dst_stride = 128 × 2 = 256B
pad        = 256 - 200 = 56B
```

```mlir
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,
    %c64_i64,
    %c200_i64,
    %c0_i64,
    %c0_i64,
    %true,
    %c0_i64,
    %c200_i64,
    %c256_i64
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

这里真正起作用的是 `data_select_bit = true`。只要这个位被置位，DMA 就会把每一行有效数据后的空白区补成对齐后的目标宽度。

---

## 示例 4：把 32×32 的 f32 tile 从向量 tile buffer 回写到 GM

```mlir
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_out, %arg1,
    %c0_i64,
    %c32_i64,
    %c128_i64,
    %c0_i64,
    %c128_i64,
    %c128_i64
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
```

这个例子没有 padding：`len_burst == src_stride == dst_stride == 128B`。

---

## 示例 5：把 64×128 的 f16 tile 写回大矩阵

向量 tile buffer 中每行是连续的 256B，但 GM 里的目标矩阵每一行总跨度是 1024B，因此回写时只写入前 256B，有效行间距由 `dst_stride` 保证。

```mlir
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c0_i64, %c0_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_ptr, %gm_ptr,
    %c0_i64,
    %c64_i64,
    %c256_i64,
    %c0_i64,
    %c1024_i64,
    %c256_i64
    : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64
```

---

## 示例 6：利用多级循环搬运一批 tile

把一个 `[4, 8, 128]` 的 f16 张量按 batch 维度搬到向量 tile buffer。每个 batch 是 `8 × 128`，每行 256B，因此每个 batch 总大小是 `2048B`。

```mlir
pto.set_loop_size_outtoub %c4_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c2048_i64, %c2048_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,
    %c8_i64,
    %c256_i64,
    %c0_i64,
    %c0_i64,
    %false,
    %c0_i64,
    %c256_i64,
    %c256_i64
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

执行轨迹可以直接按 `loop1` 展开理解：

```text
loop1 第 0 轮：gm_ptr + 0×2048 → ub_ptr + 0×2048，搬 8 行 × 256B
loop1 第 1 轮：gm_ptr + 1×2048 → ub_ptr + 1×2048，搬 8 行 × 256B
loop1 第 2 轮：gm_ptr + 2×2048 → ub_ptr + 2×2048，搬 8 行 × 256B
loop1 第 3 轮：gm_ptr + 3×2048 → ub_ptr + 3×2048，搬 8 行 × 256B
```

---

## 相关页面

- [向量加载与存储](./vector-load-store_zh.md)
- [向量流水线同步](./pipeline-sync_zh.md)
- [标量 DMA 拷贝](../scalar/dma-copy_zh.md)
