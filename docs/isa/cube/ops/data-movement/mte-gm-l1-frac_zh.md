# pto.mte_gm_l1_frac

`pto.mte_gm_l1_frac` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

加载逻辑 2D GM 区域并向 L1 写入一个或多个 **NZ fractal** 矩阵组。`nd2nz` 读取一个逻辑 `src[n, d]` 矩阵；`dn2nz` 读取一个逻辑 `src[d, n]` 矩阵，并把相同的逻辑 `N x D` 结果按 NZ 布局写出。

这是把行优先 / 列优先 GM 操作数装入 cube 的 [NZ Fractal 布局](../../nz-fractal-layout_zh.md) 的标准入口。`pto.mte_gm_l1_frac` 之后，L1 tile 即可被 [`pto.mte_l1_l0a`](./mte-l1-l0a_zh.md) / [`pto.mte_l1_l0b`](./mte-l1-l0b_zh.md) 消费。

## 机制

参考地址：

```text
for g in 0 .. group_count-1:
  src_g = src + g * src_outer_stride
  dst_g = dst + g * dst_loop4_stride * 32

  for n in 0 .. n_value-1:
    for d in 0 .. d_value-1:
      if mode == nd2nz:
        value = load(src_g + n * src_inner_stride + d * sizeof(T))
      else:
        value = load(src_g + d * src_inner_stride + n * sizeof(T))
      store value into NZ position for logical [n, d] under dst_g

  最后一个 C0 组中的无效 lane 写 0
```

## 语法

```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz|dn2nz,
  shape(%n_value, %d_value),
  src_layout(%src_inner_stride[, %src_outer_stride]),
  dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride),
  ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, ...
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src` | ptr | GM 源基址 |
| `%dst` | ptr | L1 NZ 目标基址（`!pto.ptr<T, l1>`） |
| `nd2nz` / `dn2nz` | 关键字 | 源逻辑布局模式 |
| `shape(%n_value, %d_value)` | i64 对 | NZ 打包前的逻辑输出形状 |
| `src_layout(%src_inner_stride[, %src_outer_stride])` | i64 / 可选 i64 | 源行 / 矩阵字节步长 |
| `dst_group(...)` | i64 元组 | 目标组数与放置步长，单位是 C0（1 个 C0 = 32 字节） |
| `ctrl(%l2_cache_ctrl, %smallc0_en)` | i64, i1 | Cache 提示与 small-C0 打包使能 |

`src_layout(%src_inner_stride)` 描述一个逻辑源矩阵。对 `nd2nz` 而言，`%src_inner_stride` 是从 `src[n, 0]` 到 `src[n + 1, 0]` 的字节距离；对 `dn2nz` 而言，则是从 `src[d, 0]` 到 `src[d + 1, 0]` 的字节距离。`%src_outer_stride` 表示相邻源矩阵之间的字节距离；不写时为 0。

`dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride)` 写出 `%group_count` 个逻辑矩阵。目标步长以 C0 为单位。这些步长把生成的 NZ block 放置在相对 `%dst` 的位置，并不切换到独立的内存块。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把一个或多个 NZ 矩阵组写入 L1。 |

## 副作用

读 GM 可见存储，写 L1 可见存储。占用 AIC MTE2 流水线。

## 约束

!!! warning "约束"
    - 源步长是字节。对行优先 16×16 f16 输入，`src_layout(32)` 描述连续的行。
    - 目标步长是 C0，**不是**字节，**不是**元素。
    - `smallc0_en = true` 仅在目标支持的 small-C0 场景下合法；当前契约在 small-C0 模式拒绝 `d_value > 4`。
    - 普通 C0 模式下，每个目标 C0 burst 会被填充到 32 字节；small-C0 模式下，每个目标 burst 会被填充到 4 个逻辑通道，所生成的内层 N 与 C0 放置位置由 small-C0 打包规则固定。`%dst_loop4_stride` 仍然负责放置相邻矩阵组。
    - small-C0 模式下，缺失的逻辑 `N` 行与无效 `D` lane 写 0；生成的 NZ 矩阵尾部按 32 字节 C0 边界做填充。
    - `%dst` 与 `dst_group(...)` 选中的目标区域不得重叠；如果两次写命中相同字节，最终结果不是稳定的程序结果。

## 示例

```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz,
  shape(%c32_i64, %c16_i64),
  src_layout(%c32_i64, %c1024_i64),
  dst_group(%c2_i64, %c1_i64, %c16_i64, %c64_i64),
  ctrl(%c0_i64, %false)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz, shape i64, i64,
    src_layout(i64, i64), dst_group i64, i64, i64, i64, ctrl i64, i1
```

## 相关指令

- 直接 GM→L1 拷贝（不重排）：[pto.mte_gm_l1](./mte-gm-l1_zh.md)
- 消费该 NZ tile：[pto.mte_l1_l0a](./mte-l1-l0a_zh.md)、[pto.mte_l1_l0b](./mte-l1-l0b_zh.md)
- 布局参考：[NZ Fractal 布局](../../nz-fractal-layout_zh.md)
