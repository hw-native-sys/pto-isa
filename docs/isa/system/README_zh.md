# 系统调度指令集

系统调度指令描述 PTO 可见的运行时协议，用于协调 tile buffer 生命周期以及生产者-消费者流。这个集合故意保持窄边界：tile alias、tile-scalar 计算、布局打包、量化、随机生成和直方图都归入定义其 payload 语义的 tile 指令族。

## 操作

| 操作 | 说明 | 类别 |
| --- | --- | --- |
| [pto.tfree](./ops/TFREE.md) | 释放 tile 或 buffer 资源 | Resource lifetime |
| [pto.tpop](./ops/TPOP.md) | 从 TPipe/TMPipe 生产者-消费者流中弹出 | Scheduling protocol |
| [pto.tpush](./ops/TPUSH.md) | 向 TPipe/TMPipe 生产者-消费者流中推入 | Scheduling protocol |

## 契约

系统调度指令是 PTO ISA 指令。它们的效果通过 buffer ownership、资源生命周期以及 TPipe/TMPipe 排序体现。即使 backend 将某个操作 lowering 为 scalar 同步或 runtime 步骤，也必须保留文档定义的协议。

## 相关页面

- [指令集总览](../instruction-families/README_zh.md)
- [Tile ISA 参考](../tile/README_zh.md)
- [Communication ISA 参考](../comm/README_zh.md)
