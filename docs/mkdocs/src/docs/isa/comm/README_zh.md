<!-- Generated from `docs/isa/comm/README_zh.md` -->

# PTO 通信 ISA 参考

本目录包含 PTO 通信 ISA 的逐指令参考。通信操作用于跨执行代理和并行 rank 的数据传输和同步。

## 命名约定

`pto.t*` 是 IR/汇编语法；对应的 `T*` 是 C++ 内建语法。两者指同一操作。本手册在每个页面同时记录两种拼写。

## 点对点通信（同步）

- [**TGET / pto.tget**](./TGET_zh.md): 远程读 — 通过 UB 暂存 tile 将远端 NPU GM 数据复制到本地 GM
- [**TPUT / pto.tput**](./TPUT_zh.md): 远程写 — 通过 UB 暂存 tile 将本地 GM 数据复制到远端 NPU GM

## 点对点通信（异步）

- [**TGET_ASYNC / pto.tget_async**](./TGET_ASYNC_zh.md): 异步远程读
- [**TPUT_ASYNC / pto.tput_async**](./TPUT_ASYNC_zh.md): 异步远程写

## 信号式同步

- [**TNOTIFY / pto.tnotify**](./TNOTIFY_zh.md): 向远端 NPU 发送通知
- [**TWAIT / pto.twait**](./TWAIT_zh.md): 阻塞等待信号条件
- [**TTEST / pto.ttest**](./TTEST_zh.md): 非阻塞测试信号条件

## 集合通信

- [**TBROADCAST / pto.tbroadcast**](./TBROADCAST_zh.md): 从根 NPU 广播到所有 rank
- [**TGATHER / pto.tgather**](./TGATHER_zh.md): 从所有 rank 聚集到根节点
- [**TSCATTER / pto.tscatter**](./TSCATTER_zh.md): 从根节点散射数据到所有 rank
- [**TREDUCE / pto.treduce**](./TREDUCE_zh.md): 跨所有 rank 的集合归约到根节点

## 类型定义

以下为规范性规范，非实现声明。实际值由各目标 profile 定义。

### NotifyOp

`TNOTIFY` 的操作类型：

| 值 | 描述 |
|-------|-------------|
| `NotifyOp::Set` | 直接设置 (`signal = value`) |
| `NotifyOp::AtomicAdd` | 原子加 (`signal += value`) |

### WaitCmp

`TWAIT` 和 `TTEST` 的比较操作符：

| 值 | 描述 |
|-------|-------------|
| `WaitCmp::EQ` | 等于 (`==`) |
| `WaitCmp::NE` | 不等于 (`!=`) |
| `WaitCmp::GT` | 大于 (`>`) |
| `WaitCmp::GE` | 大于或等于 (`>=`) |
| `WaitCmp::LT` | 小于 (`<`) |
| `WaitCmp::LE` | 小于或等于 (`<=`) |

### ReduceOp

`TREDUCE` 的归约操作符：

| 值 | 描述 |
|-------|-------------|
| `ReduceOp::Sum` | 按元素求和 |
| `ReduceOp::Max` | 按元素取最大值 |
| `ReduceOp::Min` | 按元素取最小值 |

### AtomicType

`TPUT` 的原子操作类型：

| 值 | 描述 |
|-------|-------------|
| `AtomicType::AtomicNone` | 无原子操作（默认） |
| `AtomicType::AtomicAdd` | 原子加操作 |

### DmaEngine

`TPUT_ASYNC` 和 `TGET_ASYNC` 的 DMA 后端选择：

| 值 | 描述 |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA 引擎（支持 2D 传输） |
| `DmaEngine::URMA` | URMA 引擎（支持 1D 传输；可用性**因 profile 而异**，必须对照目标 profile 规范验证） |

### AsyncEvent

`TPUT_ASYNC` / `TGET_ASYNC` 的返回值。表示一个待处理的异步 DMA 传输。程序使用 `AsyncEvent` 来轮询或阻塞直到传输完成：

- 有效事件**必须**与对应的 `AsyncSession` 一起使用
- 无效事件（例如句柄值为零）表示操作已同步完成或失败

### AsyncSession

异步 DMA 操作的引擎无关会话。程序构建一次会话并将其传递给所有异步调用。会话封装了引擎类型、暂存缓冲区和异步推进所需的工作区。

### ParallelGroup

跨多个 rank 的集合通信封装器。包含：

- `GlobalData` 对象数组（每个封装一个 GM 地址；根据集合操作类型，地址可能是本地或远端）
- 参与 rank 的数量
- 基于根节点的集合操作的根节点索引

## 规范来源

通信操作行为的权威规范是 PTO ISA 手册。`include/pto/comm/` 中的后端实现是**参考性**的，可能反映不属于 ISA 保证范围的实现细节。
