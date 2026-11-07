# include/pto/comm/

PTO 通信指令集，提供 NPU 间数据传输、信号同步与集合通信操作。

## 推荐的 include

- `pto_comm_inst.hpp`：统一公共 API 头文件。上层代码只需 include 该文件，它会引入所有必要的类型定义，并根据编译宏分发到正确的后端（NPU 原生实现或 CPU 仿真）。

## 目录结构

```
comm/
├── pto_comm_inst.hpp            # 公共 API：TPUT、TGET、TNOTIFY、TWAIT、TTEST、
│                                #   TGATHER、TSCATTER、TBROADCAST、TREDUCE、
│                                #   TPUT_ASYNC、TGET_ASYNC
├── pto_comm_instr_impl.hpp      # 后端分发 — 根据 __CCE_AICORE__ / __CPU_SIM / PTO_NPU_ARCH_A5
│                                #   宏选择对应架构实现
├── comm_types.hpp               # 公共类型：ParallelGroup、Signal、Signal2D、
│                                #   NotifyOp、WaitCmp、ReduceOp、DmaEngine、AsyncEvent
│
├── a2a3/                        # A2/A3（Ascend 910B/910C）架构实现
│   ├── TPut.hpp                 # TPUT_IMPL  — 远程写（本地 GM → UB → 远端 GM）
│   ├── TGet.hpp                 # TGET_IMPL  — 远程读（远端 GM → UB → 本地 GM）
│   ├── TNotify.hpp              # TNOTIFY_IMPL — 发送标志通知
│   ├── TWait.hpp                # TWAIT_IMPL — 阻塞等待信号满足条件
│   ├── TTest.hpp                # TTEST_IMPL — 非阻塞信号检测
│   ├── TGather.hpp              # TGATHER_IMPL  — root 从所有 rank 收集数据
│   ├── TScatter.hpp             # TSCATTER_IMPL — root 向所有 rank 分发数据
│   ├── TBroadCast.hpp           # TBROADCAST_IMPL — root 向所有 rank 广播数据
│   ├── TReduce.hpp              # TREDUCE_IMPL — root 收集并归约（Sum/Max/Min）
│   └── async/
│       ├── TPutAsync.hpp        # TPUT_ASYNC_IMPL（仅 SDMA）
│       └── TGetAsync.hpp        # TGET_ASYNC_IMPL（仅 SDMA）
│
├── a5/                          # A5（Ascend 950）架构实现
│   ├── T*.hpp                   # 同步指令（include a2a3/ 对应文件）
│   └── async/
│       ├── TPutAsync.hpp        # TPUT_ASYNC_IMPL（SDMA + MTE 回退 + URMA + RDMA）
│       └── TGetAsync.hpp        # TGET_ASYNC_IMPL（SDMA + URMA + RDMA）
│
└── async_common/                # 异步公共实现（a2a3/a5 共享）
    ├── async_types.hpp          # SDMA/URMA/RDMA 会话与上下文类型
    ├── async_event_impl.hpp     # AsyncEvent::Wait/Test、BuildAsyncSession
    ├── TPutAsyncCommonDetail.hpp # TPUT_ASYNC 公共辅助函数 + SDMA 实现
    └── TGetAsyncCommonDetail.hpp # TGET_ASYNC 公共辅助函数 + SDMA 实现
```

## 架构

```
 用户代码
    │
    ▼
 pto_comm_inst.hpp              ← 公共 API（模板封装 + 事件处理）
    │                              引入 async_common/async_event_impl.hpp
    ▼
 pto_comm_instr_impl.hpp        ← 编译期分发
    │
    ├── PTO_NPU_ARCH_A5    →  a5/T*.hpp / a5/async/T*Async.hpp
    ├── __CCE_AICORE__     →  a2a3/T*.hpp / a2a3/async/T*Async.hpp
    └── __CPU_SIM          →  pto/cpu/comm/T*.hpp（CPU 仿真 stubs）
```

## 指令分类

| 类别 | 指令 | 说明 |
|---|---|---|
| 点对点（同步） | `TPUT`、`TGET` | 通过 UB 暂存 Tile 的远程写/读。支持单缓冲和 ping-pong 双缓冲模式。 |
| 点对点（异步） | `TPUT_ASYNC`、`TGET_ASYNC` | 通过 SDMA、URMA 或 RDMA 引擎进行 GM-to-GM DMA。返回 `AsyncEvent` 用于后续 Wait/Test。 |
| 信号同步 | `TNOTIFY`、`TWAIT`、`TTEST` | 基于标志的跨 NPU 同步。信号为 `int32_t` 标量或二维网格。 |
| 集合通信 | `TGATHER`、`TSCATTER`、`TBROADCAST`、`TREDUCE` | 基于 `ParallelGroup` 的多 rank 操作。由 root 发起，支持 2D 分块滑动和 ping-pong 双缓冲。 |

## 核心类型（comm_types.hpp）

- **`ParallelGroup<GlobalData>`** — 轻量级视图，封装每个 rank 对应的 `GlobalTensor` 对象数组。
- **`Signal`** — 标量 `GlobalTensor<int32_t, Shape<1,1,1,1,1>>`，用于单标志同步。
- **`Signal2D<Rows, Cols>`** — 编译期形状的二维信号网格，支持密集布局和带步长的子区域视图。
- **`AsyncEvent`** — 异步指令返回的句柄，调用 `.Wait(session)` 或 `.Test(session)` 进行同步。
- **`AsyncSession`** — 引擎无关的会话，通过 `BuildAsyncSession<engine>()` 构建。

## 相关文档

- 指令语义与示例：`docs/isa/`
- CPU 仿真 stubs：`pto/cpu/comm/`
- NPU 异步后端：`pto/comm/async/`
