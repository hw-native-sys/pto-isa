# pto.mte_ub_gm

`pto.mte_ub_gm` 属于 [DMA Copy](../../dma-copy_zh.md) 指令集。

!!! note "PTO ISA v0.6 表面"
    v0.6 PTO 微指令表面已把旧的 `pto.copy_ubuf_to_gm` 与独立的 `set_loop_size_*` / `set_loop_stride_*` 配置指令合并成一个分组指令 `pto.mte_ub_gm`，并通过内联的 `nburst(...)` 与可选的 `loop(...)` 子句承载所有重复结构。

## 摘要

执行一次分组 UB→GM DMA 传输。`nburst(...)` 描述最内层的重复 burst，`loop(...)` 可选地追加外层重复。

## 机制

MTE3 引擎从 `%ub_src` 读取 `%n_burst` 行，写入 `%gm_dst`。每行搬运 `%len_burst` 字节的连续数据，源/目标步长操作数给出相邻行起点的字节距离。可选的外层 `loop(...)` 子句把 `nburst(...)` 包裹起来表达多层重复。在 GM→UB 加载时增加的填充字节会被剥除：MTE3 仅从每个 32B 对齐的 UB 行读取 `%len_burst` 个字节，写入 GM 的只是有效数据。

## 语法

```mlir
pto.mte_ub_gm %ub_src, %gm_dst, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64,
    [loop i64, i64, i64,]*
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB 源指针（`!pto.ptr<T, ub>`，32B 对齐） |
| `%gm_dst` | ptr | GM 目标指针（`!pto.ptr<T, gm>`） |
| `%len_burst` | 16 bits | 每行连续搬运的字节数 |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 / 21 / 40 bits | 必备最内层 burst 组：数量、UB 源步长、GM 目标步长 |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 / 21 / 40 bits | 可选外层重复组：数量、UB 源步长、GM 目标步长 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 此形式不返回 SSA 值；它把数据写入 Global Memory。 |

## 副作用

读取 UB 可见存储并写入 GM 可见存储。MTE3 流水线在传输期间被占用；下游消费者（以及对同一 GM 地址的后续写入排序）需要通过 `pto.set_flag` / `pto.wait_flag`（`PIPE_V` → `PIPE_MTE3`），以及在背靠背写同一 GM 地址时通过 `pto.mem_bar` 做显式同步。

## 约束

!!! warning "约束"
    - `nburst(...)` 必须存在。
    - 每个 `loop(...)` 子句出现时必须给出完整三元组。
    - `nburst(...)` 是最内层。
    - `loop(...)` 子句按从内到外排序；第一个 `loop(...)` 包裹 `nburst(...)`，每多一个 `loop(...)` 又把前面所有层包裹起来。
    - `%ub_src` 必须 32 字节对齐。

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
// 两层 outbound 传输：rows × tiles。
pto.mte_ub_gm %ub_in, %gm_out, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride)
  loop(%tiles, %ub_tile_stride, %gm_tile_stride)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64,
    loop i64, i64, i64
```

```mlir
// 三层 outbound 传输：rows × tiles × batches。
pto.mte_ub_gm %ub_in, %gm_out, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride)
  loop(%tiles, %ub_tile_stride, %gm_tile_stride)
  loop(%batches, %ub_batch_stride, %gm_batch_stride)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64,
    loop i64, i64, i64, loop i64, i64, i64
```

## 相关页面

- 指令集总览：[DMA Copy](../../dma-copy_zh.md)
- 反向传输：[pto.mte_gm_ub](./copy-gm-to-ubuf_zh.md)
- UB 内拷贝：[pto.mte_ub_ub](./copy-ubuf-to-ubuf_zh.md)
- 流水线同步：[pto.set_flag](../pipeline-sync/set-flag_zh.md)、[pto.wait_flag](../pipeline-sync/wait-flag_zh.md)、[pto.mem_bar](../pipeline-sync/mem-bar_zh.md)
- 控制壳总览：[Control and configuration](../../control-and-configuration_zh.md)
