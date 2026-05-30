# pto.mte_ub_l1（cube 侧视图）

`pto.mte_ub_l1` 是 [`pto.mte_l1_ub`](./mte-l1-ub_zh.md) 的反向通路。该指令的标准参考页面位于标量 DMA Copy 节，因为它由 AIV 侧的标量程序发起：

- **参考页面**：[pto.mte_ub_l1 — 标量 DMA Copy](../../../scalar/ops/dma-copy/mte-ub-l1_zh.md)

本占位页存在的目的，是让 cube 节的索引链接能稳定解析；完整的语法、参数表与约束请直接参考标量 DMA Copy 中的指令参考。

## 为什么挂在标量 DMA 下

UB→L1 传输由 Vector（AIV）侧的标量程序发起，因为源指针位于 Vector 块拥有的 UB 缓冲。Cube（AIC）侧在收到 L1 就绪通知后，通过 [`pto.mte_l1_l0a`](./mte-l1-l0a_zh.md) / [`pto.mte_l1_l0b`](./mte-l1-l0b_zh.md) 消费产出的 L1 tile。

## 相关指令

- 反向通路（L1 → UB）：[pto.mte_l1_ub](./mte-l1-ub_zh.md)
- Cube 操作数装载：[pto.mte_l1_l0a](./mte-l1-l0a_zh.md)、[pto.mte_l1_l0b](./mte-l1-l0b_zh.md)
- 跨块同步：[Cluster 编程模型](../../../machine-model/execution-agents_zh.md)
