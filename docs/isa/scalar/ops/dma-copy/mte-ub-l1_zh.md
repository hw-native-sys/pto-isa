# pto.mte_ub_l1

`pto.mte_ub_l1` 属于 [DMA Copy](../../dma-copy_zh.md) 指令集。

## 摘要

执行一次分组 UB→L1（CBUF）拷贝。`nburst(...)` 描述把 UB tile 暂存到 cube 侧 L1（CBUF）缓冲区的重复 burst。

## 机制

MTE 引擎从 `%ub_src` 读取 `%n_burst` 个块（每块 `%len_burst * 32` 字节），写入 `%l1_dst`。两次 burst 之间，源前进 `(len_burst + src_gap) * 32` 字节，目标前进 `(len_burst + dst_gap) * 32` 字节。`src_gap` 和 `dst_gap` 是跨 burst 的间隔（以 32 字节为单位）。

`pto.mte_ub_l1` 是把 Vector 产出 tile 回灌到 cube 侧 L1 暂存区的架构级回退路径。硬件会做 L1 fractal 布局所需的 ND→NZ 转换；详见（后续会补齐的）[NZ Fractal Layout](../../../cube/nz-fractal-layout_zh.md)。

## 语法

```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, l1>, i64, i64, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB 源指针（`!pto.ptr<T, ub>`，32B 对齐） |
| `%l1_dst` | ptr | L1 目标指针（`!pto.ptr<T, l1>`，32B 对齐） |
| `%len_burst` | 16 bits | Burst 长度（以 32 字节为单位） |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 / 16 / 16 bits | 必备 burst 组：数量、源间隔、目标间隔 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 此形式不返回 SSA 值；它把数据写入 L1 目标区域。 |

## 副作用

读取 UB 可见存储并写入 L1 可见存储。MTE 流水线在传输期间被占用。下游 cube 消费者（例如 `pto.mte_l1_l0a` / `pto.mte_l1_l0b`）必须等待相应的流水线同步事件，再去读 L1。

## 约束

!!! warning "约束"
    - UB 源和 L1 目标地址都必须 32 字节对齐。
    - `%len_burst`、`%src_gap`、`%dst_gap` 都以 32 字节为单位编码。

## 异常

!!! danger "异常"
    - verifier 会拒绝非法的 operand 形状以及目标 profile 不支持的子句组合。
    - 约束中列出的其它非法情形同样属于契约。

## 目标 Profile 限制

??? info "目标 Profile 限制"
    - UB↔L1 专用通路是带 cube 的目标 profile（A2/A3/A5）的架构特性。CPU 仿真建模了拷贝契约，但不会暴露底层的 NZ fractal 暂存细节。
    - 在 1:2 Cube/Vector 协同模式下，AIV0 和 AIV1 各自对同一个 L1 基址发起自己的 `pto.mte_ub_l1`；cube 把两个 sub-tile 组合成一个连续的 NZ Mat tile。

## 示例

```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, l1>, i64, i64, i64, i64
```

## 相关页面

- 指令集总览：[DMA Copy](../../dma-copy_zh.md)
- UB 内拷贝：[pto.mte_ub_ub](./copy-ubuf-to-ubuf_zh.md)
- GM↔UB 传输：[pto.mte_gm_ub](./copy-gm-to-ubuf_zh.md)、[pto.mte_ub_gm](./copy-ubuf-to-gm_zh.md)
- 控制壳总览：[Control and configuration](../../control-and-configuration_zh.md)
