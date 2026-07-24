# PTO Tile Intrinsics 编程模型

PTO Tile Lib 提供以 **Tile 粒度**为核心的 C++ 内建接口（intrinsics），并可映射到 PTO ISA。该模型的设计目标是：

- **跨设备代际的可移植性**：硬件细节可能变化（指令细节、存储布局、调度约束等），但编程模型保持稳定。
- **接近硬件的性能表达能力**：Tile 与 GlobalTensor 足够底层，可表达高效的数据搬运与计算。
- **覆盖两类开发者**：偏“编译器做重活”的高效编程方式，以及偏“显式控制放置与同步”的专家调优方式。

抽象执行模型（core/device/host）参见：`docs/machine/abstract-machine_zh.md`。

## 核心概念

- **Tile**：固定容量的二维片上缓冲区（概念上类似 tile 寄存器 / SRAM 块），也是大多数 PTO 指令的主要计算单元。参见：`docs/coding/Tile_zh.md`。
- **GlobalTensor**：全局内存（GM）的轻量级视图，带 5 维 shape/stride/layout 元数据；被 `TLOAD`、`TSTORE` 等内存类指令消费。参见：`docs/coding/GlobalTensor_zh.md`。
- **Scalar**：用于参数化指令的立即数与枚举（舍入模式、比较模式、原子模式等）。参见：`docs/coding/Scalar_zh.md`。
- **Event**：显式的依赖 token，用于在不引入全局屏障的情况下表达流水线类之间的顺序约束。参见：`docs/coding/Event_zh.md`。

## 两种开发风格

### PTO-Auto

PTO-Auto 面向希望获得简单、可移植体验的开发者：

- 编译器/运行时选择内存放置与地址绑定策略。
- 编译器插入必需的同步。
- 编译器调度操作并在可能时做融合。

该模式适合作为正确性与可移植性的起点。

### PTO-Manual

PTO-Manual 面向需要完全控制以进行性能调优的开发者：

- 开发者控制内存放置与绑定（例如通过 `TASSIGN`）。
- 开发者显式表达顺序（events 和/或 `TSYNC`）。
- 开发者控制操作调度与流水线结构。

该模式使关键内核能够进行专家级优化，同时仍复用同一套 Tile/GlobalTensor 抽象。

## 执行模型：SPMD 与 MPMD

PTO 支持 **SPMD** 与 **MPMD** 两种执行模型。

这两种模型描述的是**工作如何映射到核心**；它们与 **Auto vs Manual** 开发风格是正交的（可以写 SPMD-Auto、SPMD-Manual、MPMD-Auto 或 MPMD-Manual）。

### SPMD（Single Program, Multiple Data）

在 SPMD 中，所有参与的核心运行同一入口函数，每个核心使用自身的运行时身份（例如 `block_idx`）选择其数据区域。

当存在 sub-block 分解时，可以构造稳定的“虚拟 id”：

```cpp
auto cid = get_block_idx();
auto vid = get_block_idx() * get_subblockdim() + get_subblockid();
```

SPMD 适合规则的张量 tiling（GEMM、按行 softmax、逐元素算子等）。

### MPMD（Multiple Program, Multiple Data）

在 MPMD 中，不同核心（或核心组）可以在同一 tile 图中执行**不同的 tile 程序**。概念上由 **Device Machine 调度器**决定某个核心运行哪段“程序”。

一种可移植写法是：调度器提供一个 **task id** 给内核入口函数，在内核中按 task 分派：

```cpp
__global__ __aicore__ void KernelMPMD(__gm__ float* out,
                                     __gm__ const float* in,
                                     uint32_t task_id) {
  switch (task_id) {
    case 0: return ProducerStage(out, in);
    case 1: return ConsumerStage(out, in);
    default: return;
  }
}
```

说明：

- `task_id` 的获取机制依赖平台/运行时；抽象模型只要求 Device Machine 能将不同 tile block 调度到可用核心。
- 也可以用**多个入口点**（多个 kernel）替代单个入口点 + `switch` 的形式。
