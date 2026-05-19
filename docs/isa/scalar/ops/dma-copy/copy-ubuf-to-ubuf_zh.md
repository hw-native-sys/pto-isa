# pto.mte_ub_ub

`pto.mte_ub_ub` 属于 [DMA Copy](../../dma-copy_zh.md) 指令集。

!!! note "PTO ISA v0.6 表面"
    v0.6 PTO 微指令表面用分组指令 `pto.mte_ub_ub` 取代了旧的 `pto.copy_ubuf_to_ubuf`，并通过内联的 `nburst(...)` 子句承载 burst 结构。`%len_burst`、`%src_gap`、`%dst_gap` 都以 32 字节为单位编码。

## 摘要

执行一次分组 UB→UB 拷贝。`nburst(...)` 描述两块 UB 区域之间的重复 burst 传输。

## 机制

MTE 引擎从 `%ub_src` 读取 `%n_burst` 个块（每块 `%len_burst * 32` 字节），写入 `%ub_dst`。两次 burst 之间，源前进 `(len_burst + src_gap) * 32` 字节，目标前进 `(len_burst + dst_gap) * 32` 字节。`src_gap` 和 `dst_gap` 是跨 burst 的间隔（以 32 字节为单位），用于推进到下一块的起点。

## 语法

```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB 源指针（`!pto.ptr<T, ub>`，32B 对齐） |
| `%ub_dst` | ptr | UB 目标指针（`!pto.ptr<T, ub>`，32B 对齐） |
| `%len_burst` | 16 bits | Burst 长度（以 32 字节为单位） |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 / 16 / 16 bits | 必备 burst 组：数量、源间隔、目标间隔 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 此形式不返回 SSA 值；它把数据写入 UB 目标区域。 |

## 副作用

读取 UB 可见存储并写入 UB 可见存储。MTE 流水线在传输期间被占用；下游消费者需要通过相应的流水线同步原语做显式同步。

## 约束

!!! warning "约束"
    - UB 源与目标地址都必须 32 字节对齐。
    - `%len_burst`、`%src_gap`、`%dst_gap` 都以 32 字节为单位编码。

## 异常

!!! danger "异常"
    - verifier 会拒绝非法的 operand 形状以及目标 profile 不支持的子句组合。
    - 约束中列出的其它非法情形同样属于契约。

## 目标 Profile 限制

??? info "目标 Profile 限制"
    - CPU 仿真保留可见拷贝契约，但可能不暴露所有 DMA 重叠风险。
    - A2/A3 和 A5 可能收窄元素大小、burst 长度或间隔编码。

## 示例

```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64
```

## 相关页面

- 指令集总览：[DMA Copy](../../dma-copy_zh.md)
- GM↔UB 传输：[pto.mte_gm_ub](./copy-gm-to-ubuf_zh.md)、[pto.mte_ub_gm](./copy-ubuf-to-gm_zh.md)
- UB→L1（cube 路径）：[pto.mte_ub_l1](./mte-ub-l1_zh.md)
- 控制壳总览：[Control and configuration](../../control-and-configuration_zh.md)
