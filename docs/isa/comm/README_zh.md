# PTO Communication ISA Reference

本目录包含 PTO 通信 ISA 的逐指令参考文档。通信操作实现跨执行代理和并行 rank 的数据移动和同步。

## 权威来源

- C++ 内建接口：`include/pto/comm/pto_comm_inst.hpp`
- 类型定义：`include/pto/comm/comm_types.hpp`

## 按操作类型选择

| 类型 | 指令 | 说明 |
|------|------|------|
| **点对点同步写** | [TPUT](./TPUT_zh.md) | 远程写（GM → UB → GM） |
| **点对点同步读** | [TGET](./TGET_zh.md) | 远程读（GM → UB → GM） |
| **点对点异步写** | [TPUT_ASYNC](./TPUT_ASYNC_zh.md) | 异步远程写（GM → DMA 引擎 → GM） |
| **点对点异步读** | [TGET_ASYNC](./TGET_ASYNC_zh.md) | 异步远程读（GM → DMA 引擎 → GM） |
| **通知** | [TNOTIFY](./TNOTIFY_zh.md) | 向远端 NPU 发送通知 |
| **等待** | [TWAIT](./TWAIT_zh.md) | 阻塞等待信号条件满足 |
| **非阻塞检测** | [TTEST](./TTEST_zh.md) | 非阻塞检测信号条件 |
| **广播** | [TBROADCAST](./TBROADCAST_zh.md) | 从 root NPU 广播数据到所有 rank |
| **收集** | [TGATHER](./TGATHER_zh.md) | 从所有 rank 收集数据到 root |
| **散发** | [TSCATTER](./TSCATTER_zh.md) | 从 root 向所有 rank 分发数据 |
| **归约** | [TREDUCE](./TREDUCE_zh.md) | 从所有 rank 归约数据到本地 |

## 同步 vs 异步

| 类型 | 特点 | CANN 版本要求 |
|------|------|--------------|
| 同步（`TPUT`/`TGET`） | 操作阻塞直到完成，通过 UB 暂存 tile | CANN 8.x 及以上 |
| 异步（`TPUT_ASYNC`/`TGET_ASYNC`） | 非阻塞，通过 SDMA/URMA 引擎，支持轮询查询 | **CANN 9.0 及以上** |

## 相关页面

| 页面 | 内容 |
|------|------|
| [通信与运行时契约](../other/communication-and-runtime_zh.md) | 通信指令集的规范契约 |
| [其他与通信](../other/README_zh.md) | 其他与通信指令集总入口 |
| [tests/](../../../tests/README_zh.md) | 通信测试运行方式 |
