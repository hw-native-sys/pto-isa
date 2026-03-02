# 手工调优 kernels（Manual kernels）

本目录包含**手工调优（手写、面向性能）**的 kernel 示例：需要显式管理 buffer、同步与流水线控制，以在支持的 NPU 上获得最佳性能。

如果你刚接触 PTO 编程，建议先从 ISA 与教程入手：

- 编程教程：[docs/coding/tutorial_zh.md](../../docs/coding/tutorial_zh.md)
- 优化笔记：[docs/coding/opt_zh.md](../../docs/coding/opt_zh.md)
- PTO ISA 参考：[docs/PTOISA_zh.md](../../docs/PTOISA_zh.md)

## 平台

- `a2a3/`：Ascend A2/A3 平台的手工调优 kernels。

## 如何运行

每个子目录都是一个独立示例，包含各自的构建/运行说明。请从这里开始：

- [a2a3/README_zh.md](a2a3/README_zh.md)
