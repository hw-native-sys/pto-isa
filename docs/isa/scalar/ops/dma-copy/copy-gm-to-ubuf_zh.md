# pto.mte_gm_ub

`pto.mte_gm_ub` 属于 [DMA Copy](../../dma-copy_zh.md) 指令集。

!!! note "PTO ISA v0.6 表面"
    v0.6 PTO 微指令表面已经把旧的 `pto.copy_gm_to_ubuf` 与独立的 `set_loop_size_*` / `set_loop_stride_*` 配置指令合并成一个分组指令 `pto.mte_gm_ub`，并通过内联的 `nburst(...)`、可选的 `loop(...)` 和可选的 `pad(...)` 子句承载所有重复结构。之前由独立循环/步长寄存器记录的信息，现在全部表达在传输指令本身上。

## 摘要

执行一次分组 GM→UB DMA 传输。`nburst(...)` 描述最内层的重复 burst，`loop(...)` 可选地追加外层重复，`pad(...)` 可选地控制 UB 行填充。

## 机制

MTE2 引擎从 `%gm_src` 读取 `%n_burst` 行，写入 `%ub_dst`。每行搬运 `%len_burst` 字节的连续数据，源/目标步长操作数给出相邻行起点的字节距离。可选的外层 `loop(...)` 子句把 `nburst(...)` 包裹起来表达多层重复，而无需外部的循环配置状态。当存在 `pad(...)` 时，UB 行会被填充到下一个 32 字节对齐边界，填充值来自 `pad_value`。

## 语法

```mlir
pto.mte_gm_ub %gm_src, %ub_dst, %l2_cache_ctl, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  [pad(%pad_value[, %left_padding_count, %right_padding_count])]
  : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64, i64, i64,
    [loop i64, i64, i64,]*
    [pad T[, i64, i64]]
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%gm_src` | ptr | GM 源指针（`!pto.ptr<T, gm>`） |
| `%ub_dst` | ptr | UB 目标指针（`!pto.ptr<T, ub>`，32B 对齐） |
| `%l2_cache_ctl` | 2 bits | L2 cache 分配控制 |
| `%len_burst` | 16 bits | 每行连续搬运的字节数 |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 / 40 / 21 bits | 必备最内层 burst 组：数量、GM 源步长、UB 目标步长 |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 / 40 / 21 bits | 可选外层重复组：数量、GM 源步长、UB 目标步长 |
| `pad(%pad_value[, %left_padding_count, %right_padding_count])` | 标量 / 8 / 8 bits | 可选填充：填充值，可选左右填充计数 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 此形式不返回 SSA 值；它把数据写入 Unified Buffer。 |

## 副作用

读取 GM 可见存储并写入 UB 可见存储。MTE2 流水线在传输期间被占用；下游消费者需要通过 `pto.set_flag` / `pto.wait_flag`（`PIPE_MTE2` → `PIPE_V`）做显式同步。

## 约束

!!! warning "约束"
    - `nburst(...)` 必须存在。
    - 每个 `loop(...)` 子句出现时必须给出完整三元组。
    - `nburst(...)` 是最内层。
    - `loop(...)` 子句按从内到外排序；第一个 `loop(...)` 包裹 `nburst(...)`，每多一个 `loop(...)` 又把前面所有层包裹起来。
    - `pad(...)` 可以只包含 `%pad_value`；省略时左右填充计数默认为 0。若提供左右填充任一者，必须两者都提供。
    - `pad(...)` 与可选的 `loop(...)` 互相独立。DMA load 可以只带 `nburst(...) pad(...)` 而没有任何 `loop(...)`。
    - `%ub_dst` 必须 32 字节对齐。当存在 `pad(...)` 时，每个 UB 行会被从 `%len_burst` 填充到 UB 目标步长的 32B 对齐边界，确保每行起点 32B 对齐。

## 异常

!!! danger "异常"
    - verifier 会拒绝非法的 operand 形状、错误的子句组以及目标 profile 不支持的属性组合。
    - 约束中列出的其它非法情形同样属于契约。

## 目标 Profile 限制

??? info "目标 Profile 限制"
    - CPU 仿真保留可见拷贝契约，但可能不暴露所有 DMA 重叠风险。
    - A2/A3 和 A5 可能收窄元素大小、行宽或 cache 控制语义。

## 示例

```mlir
// 单层传输，带 padding：每行 %len_burst 字节，UB 行被填充到 32B 对齐。
pto.mte_gm_ub %gm_in, %ub_out, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  pad(%pad)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,
    pad f16
```

```mlir
// 两层传输：rows × tiles，带 UB 行 padding。
pto.mte_gm_ub %gm_in, %ub_out, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  loop(%tiles, %gm_tile_stride, %ub_tile_stride)
  pad(%pad)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,
    loop i64, i64, i64, pad f16
```

## 相关页面

- 指令集总览：[DMA Copy](../../dma-copy_zh.md)
- 反向传输：[pto.mte_ub_gm](./copy-ubuf-to-gm_zh.md)
- UB 内拷贝：[pto.mte_ub_ub](./copy-ubuf-to-ubuf_zh.md)
- 流水线同步：[pto.set_flag](../pipeline-sync/set-flag_zh.md)、[pto.wait_flag](../pipeline-sync/wait-flag_zh.md)
- 控制壳总览：[Control and configuration](../../control-and-configuration_zh.md)
