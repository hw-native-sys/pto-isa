# pto.set_loop_size_outtoub

!!! warning "v0.6 已弃用"
    v0.6 PTO 微指令表面不再使用独立的循环/步长配置寄存器。`pto.set_loop_size_outtoub` 原本承载的循环计数信息，现在通过 [`pto.mte_gm_ub`](./copy-gm-to-ubuf_zh.md) 上的内联 `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` 子句直接表达。新代码请使用分组形式。本页保留作为历史参考与 pre-v0.6 移植用途。

配置 out-to-ub 方向 DMA 的循环大小。

## 语法

```mlir
pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [DMA 拷贝](../../dma-copy_zh.md)
