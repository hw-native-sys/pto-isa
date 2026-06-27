# 通信 ISA

跨 NPU 集合通信、点对点交换和运行时同步。

| | 指令 | PTO 名称 | 说明 |
|-|------|----------|------|
| | [TBROADCAST](./TBROADCAST_zh.md) | `pto.tbroadcast` | 从根 NPU 向所有 rank 广播数据 |
| | [TGET](./TGET_zh.md) | `pto.tget` | 从远端 NPU 读取数据 |
| | [TGET_ASYNC](./TGET_ASYNC_zh.md) | `pto.tget_async` | TGET 的异步变体 |
| | [TNOTIFY](./TNOTIFY_zh.md) | `pto.tnotify` | 向其他 rank 发送事件通知 |
| | [TPUT](./TPUT_zh.md) | `pto.tput` | 向远端 NPU 写入数据 |
| | [TPUT_ASYNC](./TPUT_ASYNC_zh.md) | `pto.tput_async` | TPUT 的异步变体 |
| | [TREDUCE](./TREDUCE_zh.md) | `pto.treduce` | 跨所有 rank 执行集合归约 |
| | [TSCATTER](./TSCATTER_zh.md) | `pto.tscatter` | 从根 NPU 向所有 rank 分发数据 |
| | [TGATHER](./TGATHER_zh.md) | `pto.tgather` | 从所有 rank 收集数据到根 NPU |
| | [TTEST](./TTEST_zh.md) | `pto.ttest` | 测试是否已收到通知 |
| | [TWAIT](./TWAIT_zh.md) | `pto.twait` | 等待通知 |

请参阅[通信与运行时](communication-runtime_zh.md)了解该指令集的契约。

URMA 后端的异步 DMA 仅在 Ascend950 / NPU_ARCH 3510 上可用，并要求 CANN Toolkit >= 9.1.0。
