# 附录：集群 ID 映射与核心架构假设

## 概述

本附录描述了 A5 和 A2A3 平台的**集群 ID（CVID）映射**假设，这是 TPUSH/TPOP 环形缓冲通信设计的基础。

## 推荐方法：使用逻辑 Block ID 作为集群 ID

当 **`block_dim <= 核心数`** 时，最简单且推荐的方法是**直接使用逻辑 block ID 作为集群 ID**：

```cpp
// 推荐：使用逻辑 block_idx 作为 cluster_id
int cluster_id = get_block_idx();

// GM_SLOT_BUFFER 访问
my_gm_slot_buffer = GM_SLOT_BUFFER_BASE + cluster_id * PER_CLUSTER_SLOT_BUFFER_SIZE;
```

### 为什么这种方法可行

1. **硬件分配 block_idx**：FFTS 块调度器在启动任务时分配 `block_idx` 值。这是硬件提供的逻辑标识符。

2. **1:1 映射**：当 `block_dim <= num_cores` 时，每个逻辑块恰好映射到一个物理集群——不存在超额订阅。

3. **无需 GM 通信**：Cube 和 Vector 核心都可以直接使用 `get_block_idx()`，无需运行时协商。

4. **无需工作缓冲区预留**：不需要用于 CVID 交换的 12.5KB `cv_comm_buf` 区域。

### 内核标识

内核使用硬件提供的 ID 来识别其集群归属：

```cpp
// 在 Cube (AIC) 上
int my_cluster = get_block_idx();

// 在 Vector (AIV) 上
int my_cluster = get_block_idx();
int my_aiv_idx = get_subblockid();  // 0 或 1
```

## 平台架构对比

| 方面 | A5 | A2A3 |
|-----|-----|------|
| **架构** | 紧耦合 | 解耦 |
| **集群绑定** | 硬件固定 1:2 映射 | 任务调度器绑定 |
| **同步机制** | SET 块内同步 | 通过 FFTS 的 SET 跨核同步 |
| **本地数据通路** | L0C↔UB, UB↔L1 直连 | 通过 GM 中转 |

## 跨核同步机制

### FFTS 信号量 ID

每个集群有 **16 个信号量 ID** 可用于通过 `set_cross_core` 和 `wait_flag_dev` 进行跨核同步：

```
集群信号量资源：

    +---------------------------------------------------------------+
    |  每集群 16 个信号量 ID（ID 0-15）                              |
    |                                                               |
    |  每个 ID 有一个 4 位信号量值（0-15）                           |
    |  每个信号量可控制 0-15 个 FIFO 槽位                            |
    |                                                               |
    +---------------------------------------------------------------+
```

### TPUSH/TPOP 信号量分配

TPUSH/TPOP 使用 **4 个信号量 ID** 用于双向 Cube-Vector 通信：

| ID | 方向 | 用途 |
|----|------|------|
| 0 | C→V | Cube 通知 Vector 数据就绪 |
| 1 | C→V | Vector 通知 Cube 槽位空闲 |
| 2 | V→C | Vector 通知 Cube 数据就绪 |
| 3 | V→C | Cube 通知 Vector 槽位空闲 |

### 信号量操作

```cpp
// 生产者通知数据就绪（信号量加 1）
// pipe: VEC, MTE, CUBE, 或 FIX（避免 SU 屏障）
// 使用 mode2 用于 1:2 集群配置
set_cross_core(pipe, semaphore_id);

// 消费者等待数据（信号量减 1，若为 0 则阻塞）
wait_flag_dev(semaphore_id);
```

**约束**：
- 增量始终为 **1**（不可配置）
- 必须指定 **pipe**（VEC/MTE/CUBE/FIX）以避免 SU 屏障停顿
- 使用 **mode2** 用于 1:2 集群配置

### Mode2 语义（1:2 配置）

在 1:2 集群配置下，`set_cross_core` 和 `wait_flag_dev` 具有特殊的广播/归约语义：

| 方向 | 操作 | 语义 |
|------|------|------|
| **C→V** | `set_cross_core` | **广播**：Block 为两个 subblock（AIV0 + AIV1）设置信号量 |
| **C→V** | `wait_flag_dev` | 每个 Vector 核心独立等待 |
| **V→C** | `set_cross_core` | 每个 Vector 核心设置自己的信号量 |
| **V→C** | `wait_flag_dev` | **归约**：Cube 等待**两个** Vector subblock 都设置完成 |

```
C→V 广播（Cube 发出 set_cross_core）：

    AIC ──set──┬──> AIV0 信号量++
               └──> AIV1 信号量++

V→C 归约（Cube 上的 wait_flag_dev）：

    AIV0 ──set──┐
                ├──> AIC 等待两者完成
    AIV1 ──set──┘
```

这确保了 1:2 Cube-Vector 集群拓扑的正确同步，无需向每个 Vector 核心单独发信号。

### 4 位信号量范围

每个信号量 ID 有一个 **4 位计数器**（值 0-15），限制了最大未完成的 FIFO 槽位数：

```
信号量值范围：0-15

    - 值 0：无可用槽位（消费者在 wait_flag_dev 上阻塞）
    - 值 1-15：有 N 个槽位可用
    - 最大未完成槽位数：每个方向 15 个
```

这与环形缓冲设计相匹配，每个方向最多可有 8 个槽位（远在 15 槽位信号量限制之内）。

## 集群绑定流程

### 硬件 Block/Subblock 分配

`block_idx` 和 `subblock_id` 由**硬件**（FFTS 块调度器）分配，而非软件。当 FFTS 启动混合内核时，它创建具有 1:2 block-subblock 关系的逻辑集群：

```
FFTS 混合内核启动：

    +---------------------------------------------------------------------+
    |  FFTS 块调度器（硬件）                                               |
    |                                                                     |
    |  分配：每核心的 block_idx、subblock_id                               |
    |  创建：每集群 1 个 block + 2 个 subblock（1:2 比例）                  |
    |                                                                     |
    |  +-------------------+  +-------------------+                       |
    |  | 集群 0            |  | 集群 1            |  ...                  |
    |  |   block_idx=0     |  |   block_idx=1     |                       |
    |  |   AIC (block)     |  |   AIC (block)     |                       |
    |  |   AIV0 (subblk 0) |  |   AIV0 (subblk 0) |                       |
    |  |   AIV1 (subblk 1) |  |   AIV1 (subblk 1) |                       |
    |  +-------------------+  +-------------------+                       |
    |                                                                     |
    +---------------------------------------------------------------------+
```

### AICPU 握手进行核心映射

当 AICPU 需要在集群上启动运行时，它必须通过与硬件调度器的**握手来获取核心到 block/subblock 的映射**，而不是自行分配这些 ID：

```
AICPU 运行时启动：

    +---------------------------------------------------------------------+
    |  AICPU                                                              |
    |                                                                     |
    |  1. 向硬件调度器请求集群分配                                         |
    |  2. 接收映射：physical_core_id <-> (block_idx, subblock_id)         |
    |  3. 为跨核同步初始化 ffts_addr                                       |
    |  4. 在分配的核心上启动运行时，使用一致的 block_idx                    |
    |                                                                     |
    +---------------------------------------------------------------------+
                |
                | 握手
                v
    +---------------------------------------------------------------------+
    |  FFTS / 硬件调度器                                                   |
    |                                                                     |
    |  提供：物理核心的 block_idx、subblock_id 分配                         |
    |  确保：与内核启动相同的 1:2 集群结构                                  |
    |                                                                     |
    +---------------------------------------------------------------------+
```

这确保 TPUSH/TPOP 环形缓冲操作在直接通过 FFTS 启动或通过 AICPU 运行时启动时都能正确工作。

### A3 ffts_addr 初始化

在 A3 上，必须在 AICPU 握手过程中初始化 `ffts_addr`，以启用通过 `set_cross_core` 和 `wait_flag_dev` 的跨核同步：

- **ffts_addr**：FFTS 信号量寄存器的基地址
- **初始化时机**：必须在任何跨核同步操作之前完成
- **作用域**：每集群，由集群中所有核心（AIC + AIV0 + AIV1）共享

此初始化是 AICPU 握手（上述步骤 3）的一部分，确保信号量 ID（0-15）正确映射到分配集群的硬件寄存器。

## A3 FFTS 调度器与逻辑集群设置

### A3 上的当前 TPUSH/TPOP 实现

A3 TPUSH/TPOP 实现依赖于 **FFTS 跨核同步**功能。在混合内核启动期间，FFTS 硬件通过块调度器建立逻辑集群。

### 逻辑到物理核心映射

FFTS 硬件在任务启动时建立逻辑到物理核心的映射：

1. **Block ID → 物理 Cube 核心**：块调度器为每个逻辑块分配一个物理 AIC 核心。

2. **Subblock ID → 物理 Vector 核心**：每个 subblock（0, 1）映射到一个成为已分配 Cube 伙伴的物理 AIV 核心。

3. **集群内同步解析**：FFTS 硬件根据此映射解析所有集群内同步和通信路径。


---

## 附录 A：基于通用核心 ID 的 CVID 计算

> **注意**：本附录记录了用于 `block_dim > num_cores` SIMD 模式的通用实现。**不推荐用于带有 MPMD AICPU 运行时的 PyPTO**——请使用主文档中描述的逻辑 block_idx 方法。

### 常量参考

| 常量 | A5 值 | A2A3 值 | 描述 |
|------|-------|---------|------|
| `CORE_PER_DIE` | 18 | 25 | 每 die 的集群数 |
| `AIV_RATIO` | 2 | 2 | 每 Cube 的 Vector 核心数 |
| `AIC_AIV_PER_DIE` | 54 | 75 | 每 die 的总核心数（AIC + AIV） |
| `SEMAPHORE_IDS` | 16 | 16 | 每集群的信号量 ID 数 |
| `TPUSH_TPOP_SEMA_IDS` | 4 | 4 | 用于 CV 双向通信的 ID 数 |
| `SEMA_BITS` | 4 | 4 | 每信号量的位数（0-15 槽位） |
| `CV_MAX_CORES` | 36 | 25 | 支持的最大集群数 |


### 相关文档

- [源码：A5 TSyncCVID.hpp](https://gitcode.com/cann/pto-isa/blob/master/include/pto/npu/a5/custom/TSyncCVID.hpp)
- [源码：A2A3 TSyncCVID.hpp](https://gitcode.com/cann/pto-isa/blob/master/include/pto/npu/a2a3/custom/TSyncCVID.hpp)


本节记录了从物理核心 ID 计算集群 ID 的**通用实现**。这是为 **`block_dim > num_cores`**（SIMD 超额订阅）场景或直接 block_idx 映射不可用时提供的备选方案。

### A5：直接核心 ID 计算

在 A5 上，每个 die 包含 **18 个核心集群**，具有固定的 1:2 架构。集群 ID 直接从物理核心 ID 计算：

```cpp
// A5 TSYNC_CVID 实现（通用）
#ifdef __DAV_CUBE__
    int die_id = get_coreid() / AIC_AIV_PER_DIE;     // AIC_AIV_PER_DIE = 54
    comm_slot = die_id * CORE_PER_DIE + get_coreid() % AIC_AIV_PER_DIE;
#elif defined(__DAV_VEC__)
    int die_id = get_coreid() / AIC_AIV_PER_DIE;
    comm_slot = die_id * CORE_PER_DIE +
                (((get_coreid() % AIC_AIV_PER_DIE) - CORE_PER_DIE - get_subblockid()) / AIV_RATIO);
#endif
```

**关键特性**：
- **无需运行时通信**
- **确定性映射**：核心 ID → 集群 ID 是纯函数
- **硬件强制 1:2 关系**：每个集群内存在 L0C↔UB 和 UB↔L1 本地数据通路

### A2A3：通过 GM 交换核心 ID（通用）

在 A2A3 上，使用通用实现时，集群 ID 通过 GM 通信：

```cpp
// A2A3 TSYNC_CVID 实现（通用）
#ifdef __DAV_CUBE__
    // Cube 核心将其核心 ID 写入 GM 槽位
    comm_slot = static_cast<int>(get_coreid() & 0x7f);
    comm_slot %= CV_MAX_CORES;

    // 写入 GM 槽位并刷新缓存
    __gm__ volatile uint32_t *comm_slot_ptr = reinterpret_cast<__gm__ volatile uint32_t *>(
        cv_comm_buf + static_cast<std::size_t>(block_idx) * CV_COMM_SLOT_BYTES);
    comm_slot_ptr[0] = static_cast<uint32_t>(comm_slot);
    dcci(comm_slot_ptr, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);

    // 通过 FFTS 向 Vector 核心发信号
    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, CV_COMM_CTRL));

#elif defined(__DAV_VEC__)
    // Vector 核心等待 Cube 的信号，然后从 GM 读取集群 ID
    __gm__ volatile uint32_t *comm_slot_ptr = reinterpret_cast<__gm__ volatile uint32_t *>(
        cv_comm_buf + static_cast<std::size_t>(block_idx) * CV_COMM_SLOT_BYTES);
    dcci(comm_slot_ptr, SINGLE_CACHE_LINE);
    wait_flag_dev(CV_COMM_CTRL);
    comm_slot = static_cast<int>(comm_slot_ptr[0]);
#endif
```

---

## 附录 B：A2A3 工作缓冲区预留（通用实现）

使用**基于通用核心 ID 的 CVID 计算**（附录 A）时，A2A3 需要在工作缓冲区底部预留一个区域用于 `cv_comm_buf` 槽位。这是 **`block_dim > num_cores`**（SIMD 超额订阅）场景所需的。

### 预留空间计算

```
CV_COMM_SLOT_BYTES = 512 字节（每 block，512B 对齐）
CV_MAX_CORES       = 25（最大 block_dim）

预留空间 = CV_COMM_SLOT_BYTES * CV_MAX_CORES
         = 512 * 25
         = 12,800 字节
         = 12.5 KB（向上取整到 16KB 以对齐）
```

**注意**：使用推荐的 `block_idx` 作为集群 ID 方法（当 `block_dim <= num_cores`）时，**不需要**此预留。

### 内存布局（仅通用实现）

```
A2A3 工作缓冲区 (GM) - 仅通用实现：

    +------------------------------------------------------------------+
    |  底部 12.5KB：为 cv_comm_buf 预留（CVID 协商）                     |
    |                                                                  |
    |  +------------+------------+------------+-----+------------+     |
    |  | block_idx=0| block_idx=1| block_idx=2| ... | block_idx=24|    |
    |  |   512B     |   512B     |   512B     |     |   512B      |    |
    |  +------------+------------+------------+-----+------------+     |
    |                                                                  |
    +------------------------------------------------------------------+
    |  剩余空间：可用于 GM_SLOT_BUFFER、任务数据等                       |
    +------------------------------------------------------------------+
```

### A5：无需工作缓冲区预留

在 A5 上，CVID 直接从 `get_coreid()` 计算，无需任何 GM 通信。无论 `block_dim` 如何，都不需要工作缓冲区预留。

### 常量（仅通用实现）

| 常量 | A5 值 | A2A3 值 | 描述 |
|------|-------|---------|------|
| `CV_COMM_SLOT_BYTES` | 512 | 512 | 每 block 通信槽位字节数 |
| `CV_COMM_RESERVED` | 0 | 12.5KB | 工作缓冲区预留 |
