# 系统调度指令集

系统调度指令暴露 PTO 可见的运行时协议，用于 tile buffer 生命周期以及生产者-消费者流。它们不覆盖 tile payload 变换；这类指令归入定义其数据语义的 tile 指令族。

## 指令总览

| 操作 | PTO 名称 | 角色 |
| --- | --- | --- |
| Resource release | `pto.tfree` | 结束 tile 或 buffer 资源生命周期 |
| Producer-consumer push | `pto.tpush` | 向 TPipe/TMPipe 流发布 tile work 或资源 |
| Producer-consumer pop | `pto.tpop` | 从 TPipe/TMPipe 流获取 tile work 或资源 |

## 契约

系统调度操作是 PTO ISA 指令。它们的可见效果通过资源生命周期、stream 状态和生产者-消费者顺序体现。backend 可以把它们 lowering 成 scalar 或 runtime 机制，但最终 PTO 可见状态必须匹配操作契约。

## 共享约束

- 资源生命周期操作不能让后续指令持有悬空 tile 或 buffer handle。
- Push/pop 操作必须定义 stream、匹配端点以及阻塞行为。
- lowering 到 scalar 同步或 runtime 调用时，不能削弱生产者-消费者顺序。

## 相关页面

- [系统调度参考](../system/README_zh.md)
- [指令集契约](./README_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
