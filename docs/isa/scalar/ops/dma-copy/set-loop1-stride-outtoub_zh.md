# pto.set_loop1_stride_outtoub

!!! warning "v0.6 已弃用"
    v0.6 PTO 微指令表面不再使用独立的循环/步长配置寄存器。`pto.set_loop1_stride_outtoub` 原本承载的步长信息，现在通过 [`pto.mte_gm_ub`](./copy-gm-to-ubuf_zh.md) 上最内层 `nburst(%n_burst, %src_stride, %dst_stride)` 子句直接表达。新代码请使用分组形式。本页保留作为历史参考与 pre-v0.6 移植用途。

配置 out-to-ub 方向 DMA 的第一层 stride。

## 语法

```mlir
pto.set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [DMA 拷贝](../../dma-copy_zh.md)
