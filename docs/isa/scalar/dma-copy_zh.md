# DMA 拷贝

这些 `pto.*` 形式配置并执行 GM、UB、L1 之间的标量侧 DMA 搬运。它们属于标量与控制指令，因为它们定义的是 DMA 配置和搬运行为，而不是向量寄存器计算。

## 本指令集覆盖

- 使用内联 burst / loop / pad 子句的分组 GM↔UB 传输
- 分组 UB↔UB 与 UB→L1 拷贝
- （pre-v0.6 历史）独立的循环大小与循环 stride 配置寄存器

## v0.6 分组传输指令

以下是 PTO ISA v0.6 微指令表面中四条公开的分组 DMA 接口。每条指令都通过内联的 `nburst(...)` / `loop(...)` 子句表达自己的重复结构，不再需要外部独立的循环/步长配置寄存器。

- [pto.mte_gm_ub](./ops/dma-copy/copy-gm-to-ubuf_zh.md)：GM → UB，附带可选 `pad(...)` 做 32B 对齐行填充
- [pto.mte_ub_gm](./ops/dma-copy/copy-ubuf-to-gm_zh.md)：UB → GM，剥除 load 时增加的 padding
- [pto.mte_ub_ub](./ops/dma-copy/copy-ubuf-to-ubuf_zh.md)：UB 内拷贝，以 32B 为单位的 burst + gap 字段
- [pto.mte_ub_l1](./ops/dma-copy/mte-ub-l1_zh.md)：UB → L1（cube CBUF），以 32B 为单位的 burst + gap 字段

## Pre-v0.6 已弃用配置指令

下列指令对应旧的表面：循环计数与每层步长在独立的配置寄存器里编程，再由单独的拷贝指令消费。v0.6 把这些信息全部放进了分组传输指令本身的 `nburst(...)` 与外层 `loop(...)` 子句。下面这些页面保留作为历史参考与 pre-v0.6 移植用途。

- [pto.set_loop_size_outtoub](./ops/dma-copy/set-loop-size-outtoub_zh.md)
- [pto.set_loop2_stride_outtoub](./ops/dma-copy/set-loop2-stride-outtoub_zh.md)
- [pto.set_loop1_stride_outtoub](./ops/dma-copy/set-loop1-stride-outtoub_zh.md)
- [pto.set_loop_size_ubtoout](./ops/dma-copy/set-loop-size-ubtoout_zh.md)
- [pto.set_loop2_stride_ubtoout](./ops/dma-copy/set-loop2-stride-ubtoout_zh.md)
- [pto.set_loop1_stride_ubtoout](./ops/dma-copy/set-loop1-stride-ubtoout_zh.md)

旧的执行指令 `pto.copy_gm_to_ubuf` / `pto.copy_ubuf_to_gm` / `pto.copy_ubuf_to_ubuf` 已被 v0.6 的分组形式 `pto.mte_gm_ub` / `pto.mte_ub_gm` / `pto.mte_ub_ub` 取代（链接见上方）。它们对应的 per-op 页面（URL slug 保留不变）现在直接记录 v0.6 表面。

## 相关页面

- [控制与配置](./control-and-configuration_zh.md)
- [向量 DMA 路径](../vector/dma-copy_zh.md)
- [流水线同步](./ops/pipeline-sync/)
