# 高性能 GEMM AllReduce 融合算子示例

## 概览

本示例演示如何使用 PTO 实现多卡 GEMM + AllReduce 融合算子，采用双流（Compute Stream + Comm Stream）计算通信重叠设计，通过 PTO 通信指令集在 HCCL RDMA 窗口上完成 AllReduce。

## 支持的 AI 处理器

- Ascend950PR

## 目录结构

```
kernels/manual/a5/gemm_ar/
├── CMakeLists.txt              # 构建配置（3 个 target：cube kernel, vec kernel, host exe）
├── run.sh                      # 一键构建+运行脚本（自动计算 HCCL_BUFFSIZE、发现 MPI 路径）
├── gemm_ar_config.h            # 全局参数配置（矩阵维度、tile 大小、block 数量）
├── main.cpp                    # 入口：MPI 初始化、数据生成、HCCL 初始化、窗口分配、性能测量、验证
├── gemm_compute_kernel.cpp     # GEMM 计算内核（Cube 架构，L0C FP32→GM FP16 自动 cast）
├── comm_kernel.cpp             # 通信内核（Vector 架构，单 kernel 内做 RS/AG overlap AllReduce）
├── kernel_launchers.h          # Host 侧 kernel launcher 声明
├── common.hpp                  # HcclRemotePtr 设备端包装（RDMA 窗口地址转换）
├── hccl_context.h              # HcclDeviceContext 结构体（每 rank 的 RDMA 窗口地址）
├── ready_queue.hpp             # 多 block 无锁 tile 队列（compute→comm 信号机制）
└── comm_mpi.h                  # MPI 动态加载包装（dlopen/dlsym，免硬链接依赖）
```

## 算子说明

### 计算功能

本示例实现多卡 GEMM + AllReduce：

$$
C_{final} = \sum_{i=0}^{nranks-1} A_i \times B
$$

其中：

- `A_i` 为 `M×K`（每 rank 独立）
- `B` 为 `K×N`（所有 rank 共享）
- `C_i` 为 `M×N`（每 rank 本地 GEMM 结果）
- `C_final` 为 `M×N`（AllReduce 归约后的最终输出）

`gemm_ar_config.h` 中默认的参考配置为 `M=5416, K=6144, N=1408`，2 卡运行。

### 规格


| 项目           | 值                                                                         |
| ------------ | ------------------------------------------------------------------------- |
| OpType       | `GEMM + AllReduce`                                                        |
| 输入           | `A_i`: `M×K`, `float16`, `ND`（每 rank 独立）; `B`: `K×N`, `float16`, `DN`（共享） |
| 输出           | `C_final`: `M×N`, `float16`, `ND`（AllReduce 归约结果）                         |
| 计算 Kernel 名称 | `GemmComputeKernel`（Cube 架构，`dav-c220-cube`）                              |
| 通信 Kernel 名称 | `GemmCommAllKernel`（Vector 架构，`dav-c220-vec`）                             |


## 优化说明

本示例以 2 卡 Ascend950PR 平台作为性能验证平台。Ascend950PR（DAV_3510 / arch35）采用分离模式架构：Cube（AIC）与 Vector（AIV）物理分立，可配合双流做计算通信重叠。

> **核数以 CANN** `platform_config` **为准（推荐），以**`950PR_958b为例`：
>
> - `cube_core_cnt=32`（Cube / AIC 侧并行度）
> - `vector_core_cnt=64`（Vector / AIV 侧并行度）

- **双流计算通信重叠**：计算 kernel 运行在 Compute Stream（Cube），通信 kernel 运行在 Comm Stream（Vector），通过逐 tile 信号机制实现计算与通信并行。
- **逻辑上仍是 RS + AG，执行上改成单循环 overlap**：RS 负责把数据归约到 owner rank，AG 负责把 owner-local 结果广播到其它 rank，两者通过 ready counter 在一个 subtile 粒度的混合循环里衔接。
- **Block Swizzle**：计算 kernel 采用 zigzag tile 遍历顺序（奇数行反向），改善相邻 tile 间 B 矩阵的 L1 缓存复用。
- **两级双缓冲流水线**：L1 缓存（stepK=4 批量 TLOAD）+ L0 双缓冲（ping/pong），让 DMA 搬运与 Cube 计算尽可能重叠。
- **无锁 Ready Queue**：每个 AIC 负责一条队列，AIV block 静态接管队列子集 `{block_idx, block_idx + num_comm_blocks, ...}`。无数据时先 `TTEST` 轮询，必要时再 `TWAIT`，避免空转。
- **RS 双缓冲流水线**：RS 生产路径按 subtile 粒度用 ping/pong tile 重叠当前 subtile 的 `TLOAD` 与上一个 subtile 的 `TSTORE<AtomicAdd>`。
- **owner-local subtile 执行器**：每个 owner-local tile 会沿 M 维切成固定高度的 subtile（默认 `G_COMM_SUB_M=64` 行），AG block 以 reversed-stripe 方式分配 subtile，尽量把 RS 重块和 AG 重块错开。
- **publish / consume fence**：queue producer 只有在 slot payload 可见后才发布 doorbell；RS 只有在 `pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)` 之后才发布 `subtile-ready` / `ag-summary` 门铃。

## Tiling 参数


| 参数                  | 值         |
| ------------------- | --------- |
| `M`（原始）             | 5416      |
| `K`                 | 6144      |
| `N`（原始）             | 1408      |
| `M`（对齐后）            | 5504      |
| `N`（对齐后）            | 1536      |
| `baseM`             | 128       |
| `baseK`             | 64        |
| `baseN`             | 256       |
| `stepKa`            | 4         |
| `stepKb`            | 4         |
| `commSubM`          | 64        |
| `subtilesPerTile`   | 2         |
| `tile 数`            | 258（43×6） |
| `COMPUTE_BLOCK_NUM` | 24        |
| `COMM_BLOCK_NUM`    | 24        |


## 整体架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Compute Stream (24 logical AIC)       Comm Stream (24 logical AIV)          │
│                                                                              │
│  GemmComputeKernel:                     GemmCommAllKernel:                   │
│  ┌─────────────────────────┐            ┌──────────────────────────────┐     │
│  │ for each tile:          │            │ RS/AG overlap loop           │     │
│  │   K-loop (L1→L0→Cube)  │            │   轮询 Ready Queue            │     │
│  │   TSTORE → gemm_output │──Ready──→ │   TLOAD tile from gemm_output│     │
│  │   pipe_barrier(ALL)     │  Queue    │   TSTORE<AtomicAdd> → owner  │     │
│  │   Enqueue tile_idx      │            │   subtile-ready / summary++  │     │
│  └─────────────────────────┘            │   消费 ready subtile 做 AG   │     │
│                                          │   TLOAD → TSTORE 到远端 rank │     │
│                                          │   ready 驱动的 AG 衔接        │     │
│                                          │   subtile 粒度 overlap        │     │
│                                          └──────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 计算内核详解

```
时间 →
L1 (MTE2):  [TLOAD A0,B0]                [TLOAD A1,B1]              ...
L0 (MTE1):       [TEXTRACT k0] [k1] [k2] [k3] [TEXTRACT k0'] ...
Cube (M):             [TMATMUL k0] [ACC k1] [ACC k2] [ACC k3] [TMATMUL k0'] ...
                      ↑ 三级流水线完全并行 ↑
```

每个 AIC 负责一组 tile（按 `block_idx × tiles_per_block` 分配），对每个 tile：

1. **Block Swizzle 映射**：将线性 tile 索引重映射为 zigzag 遍历顺序，奇数行反向，使连续 tile 共享 B 矩阵列，提升 L1 复用。
2. **K-loop**：每 `stepKa=4` 次迭代做一次 TLOAD（L1 缓存优化），每次迭代 TEXTRACT 提取单个 K-slice 到 L0，再 TMATMUL/TMATMUL_ACC 累积。
3. **TSTORE**：L0C FP32 经 FixPipe 自动 cast 为 FP16，写入 `gemm_output`。
4. **pipe_barrier(PIPE_ALL)**：确保 GM 写入完成。
5. **MultiBlockEnqueueFast**：入队 `tile_idx`，通知通信 kernel。

## 通信内核详解

当前真正被 launch 的通信路径是 `GemmCommAllImpl()` 里的 mixed subtile pipeline：RS 生产和 AG 消费在一个循环里交错执行，同步点是每个 subtile 的 ready counter，而不是整机级 barrier。

### RS 生产路径

每个通信 block 接管如下队列子集：

```
queues(block b) = { b, b + num_comm_blocks, b + 2*num_comm_blocks, ... }
```

当默认 `COMPUTE_BLOCK_NUM = COMM_BLOCK_NUM = 24` 时，会退化成最熟悉的 1:1；如果通信 block 更少，一个 block 会通过 `RsPollQueues()` / `RsWaitOnQueue()` 轮询多条队列。

对于每个出队 tile：

1. 先沿 `M` 维切成 `G_COMM_SUBTILES_PER_TILE = G_BASE_M / G_COMM_SUB_M` 个固定高度 subtile。
2. `RsPipelineStep()` 用 ping/pong UB tile 重叠当前 subtile 的 `TLOAD` 和上一个 subtile 的 `TSTORE<AtomicAdd>`。
3. RS 目标是 `owner = tile_idx % nranks` 对应 rank 的 `reduced_output`，归约直接在 owner rank 完成。

### RS/AG 重叠同步

当前协议在 owner rank 的 `signal_matrix` 上维护两类计数器：

1. `subtile-ready[local_subtile_id]`：记录这个 owner-local subtile 已经被多少个 rank 完成 RS。
2. `ag-summary[summary_block]`：给负责该 subtile 的 AG block 提供较粗粒度的唤醒门铃。

发布路径由 `RsPublishSubtileReady()` 完成：

1. 先执行 `pipe_barrier(PIPE_ALL)` 刷本地流水。
2. 再执行 `dsb(DSB_DDR)`，确保 `reduced_output` 中的数据对其它 block / rank 可见。
3. `RsNotifySubtileReady()` 递增 subtile-ready counter。
4. `RsNotifyAgSummary()` 递增 `AgSummaryBlockForSubtile()` 选中的 summary counter。

消费路径由 `AgDrainReadyAssignedSubtiles()` 完成：

1. 先用 `TTEST(..., nranks, GE)` 轮询自己负责的 `subtile-ready` counter。
2. 一次 drain pass 第一次命中时，执行一组 acquire fence（`pipe_barrier + dsb`），后续本轮 ready subtile 共享这次 acquire。
3. 对每个 ready subtile 执行广播。
4. 如果本轮没有任何进展，则由 `AgWaitAssignedSummary()` 在 `summary_ack_count + 1` 上阻塞，等待下一次属于本 block 的唤醒。

`AgSummaryBlockForSubtile()` 使用 reversed-stripe 映射，把 AG 更重的 block 尽量落在 RS 更轻的 block 上，从而拉平总工作量。

### AG 执行路径

AG 的分工是在 owner-local subtile 空间里完成的：

```
total_local_subtiles = my_tile_count * G_COMM_SUBTILES_PER_TILE
assigned_ids(block b) = { num_comm_blocks - 1 - b + k*num_comm_blocks }
```

对每个已 ready 的 assigned subtile：

1. `AgDecodeLocalSubtile()` 把 owner-local subtile id 还原成 `reduced_output` 中的全局行偏移。
2. `AgTransferSubtileToAll()` 把固定 `G_COMM_SUB_M` 行广播给所有远端 rank。
3. 第一个远端 peer 会按 `local_subtile_id % (nranks - 1)` 做轮转，避免所有 block 先打同一个目标 rank。

现在 AG 只要看到某个 owner-local subtile 已被所有 rank 完成 RS，就可以立刻启动这一段广播。

## Ready Queue 机制

```
┌─────────────┐         ┌─────────────┐
│  AIC 0      │         │  AIV 0      │
│  (Compute)  │──Queue──│  (Comm)     │
│  block_idx=0│   0     │  block_idx=0│
└─────────────┘         └─────────────┘
┌─────────────┐         ┌─────────────┐
│  AIC 1      │         │  AIV 1      │
│  (Compute)  │──Queue──│  (Comm)     │
│  block_idx=1│   1     │  block_idx=1│
└─────────────┘         └─────────────┘
      ...                     ...
┌─────────────┐         ┌─────────────┐
│  AIC 23     │         │  AIV 23     │
│  (Compute)  │──Queue──│  (Comm)     │
│  block_idx=23│  23    │  block_idx=23│
└─────────────┘         └─────────────┘
```

（示意：默认 `COMPUTE_BLOCK_NUM = COMM_BLOCK_NUM = 24` 时的逻辑 `block_idx` 0…23；与 `Ascend950PR_958b.ini` 给出的 `**cube_core_cnt=32` / `vector_core_cnt=64**` 不必一一等同。若使用 `Ascend950PR_9599` 等 SoC，请以对应 `.ini` 中的计数为准。）

- 每个队列由 64 字节对齐的 `PerBlockQueue` 元数据块和 cache-line 隔离的 `PerBlockQueueSlot` payload 组成。
- **生产者**（AIC）：`PerBlockQueueEnqueueFast` 通过 `GetQueueSlot()` 写目标 slot，先对该 slot 做 `dcci` 刷新，再做 release fence，最后才递增 `count`。
- **消费者**（AIV）：`PerBlockQueueTryDequeue` 先用 `TTEST` 检查 `count >= head+1`，再做 acquire fence，并通过 `GetQueueSlot()` 反复读取目标 slot，直到 payload 可见；无数据时返回 -1，长时间无数据时再走 `TWAIT`。
- 单生产者单消费者设计，无需原子操作。

## 内存布局与 HCCL 窗口

只有被远端 TPUT/TNOTIFY 写入的 buffer 需要放在 HCCL RDMA 窗口中，本地读写的 buffer 使用普通 `aclrtMalloc`。


| 缓冲区                    | 大小                        | 位置              | 原因                                 |
| ---------------------- | ------------------------- | --------------- | ---------------------------------- |
| `reduced_output`       | M × N × 2B                | **HCCL 窗口**     | RS AtomicAdd + AG 远端 TPUT 写入（FP16） |
| `signal_matrix`        | `G_SIGNAL_TOTAL_SLOTS` × 4B，对齐 64B | **HCCL 窗口**     | subtile-ready / AG summary 计数器（含保留 legacy barrier 槽位） |
| `gemm_output`          | M × N × 2B                | **aclrtMalloc** | 仅本地读写（FP16）                        |
| `src0_dev`, `src1_dev` | 输入矩阵（FP16）                | **aclrtMalloc** | 仅本地读写                              |


窗口大小由 `HCCL_BUFFSIZE` 环境变量控制。`run.sh` 会基于 pad 后的 `reduced_output` 大小估算窗口需求，并额外预留较大的安全边界：

```text
pad(M, G_BASE_M) × pad(N, G_BASE_N) × 2 / 1MB + 64MB
```

`signal_matrix` 也在同一个窗口里，但相对这 64MB margin 很小。

## 实测性能（参考）

以下数据在 2 卡 Ascend950PR 上测得，参数 `M=5416, K=6144, N=1408`（padded `5504×1536`），`258 tiles (43×6)`，`compute_blocks=32`，`comm_blocks=24`。每 rank 计算完整 GEMM `C_i = A_i × B`，AllReduce 对 2 个 `C_i` 求和。


| 指标           | 值                                               |
| ------------ | ----------------------------------------------- |
| Compute-only | `323.2 us`（`289926 GFLOPS`） |
| Sequential   | `856.7 us`（compute `325.8 us` + comm `530.8 us @ 29.7 GB/s`） |
| Pipelined    | **`580.2 us`**（compute done `360.4 us`，comm done `580.2 us @ 27.1 GB/s`） |
| Speedup      | `1.476x` |
| Time saved   | `276.5 us`（`32.3%`） |
| Overlap eff  | `84.8%` |
| Throughput   | `322996 GFLOPS`（total） |


### 这些数字意味着什么

- **Compute-only**：纯 GEMM 时间（无通信），反映单卡 Cube 利用率上限。当前纯算时间为 `323.2 us`，对应 `289926 GFLOPS`。
- **Sequential**：计算→通信串行执行，无重叠。当前串行总时间为 `856.7 us`，其中 compute `325.8 us`、comm `530.8 us`。
- **Pipelined**：计算与通信双流并行。当前 `Pipelined = 580.2 us`，相比 `Sequential = 856.7 us` 加速 `1.476x`；`compute done = 360.4 us`。
- **Speedup**：Sequential / Pipelined，越高说明计算通信重叠越有效。
- **Time saved**：相对串行路径节省的总时长。当前节省 `276.5 us`，约占 `32.3%`。
- **Overlap eff**：重叠带来的时间节省占较短阶段时间的百分比。`84.8%` 表明较短阶段的大部分时间已被成功隐藏。

### 优化历程

> 下表前几行是历史优化检查点；最后一行是当前 `subtile-ready / AG-summary overlap` 在 Ascend950PR 上的最新端到端结果。不要把前面的历史值误解成当前 live path 的逐项开关。


| 优化                        | Pipelined (us) | 增益                    | 结论             |
| ------------------------- | -------------- | --------------------- | -------------- |
| 基线                        | 808            | —                     | —              |
| Block Swizzle             | 793            | -1.8%                 | **保留**         |
| RS AtomicAdd 消除 Reduce 阶段 | 736            | -6.6%                 | **保留**         |
| AG 行级展平分解                 | 623            | -15.4%                | 历史检查点       |
| 48 AIV（RS skip + AG 参与）   | 639            | RS 仅 24 AIV、AG 48 AIV | **回退**（AIC 干扰） |
| 48 AIV 双队列（1 AIC : 2 AIV） | 667            | RS/AG 均 48 AIV        | **回退**（AIC 干扰） |
| 当前 subtile-ready / AG-summary overlap | **580.2** | 当前 2 卡 Ascend950PR 实测结果 | **当前结果** |


## 性能优化指南（如何调这个 kernel）

### 1) 优先做多核切分

每个 AIC 按 `block_idx × tiles_per_block` 分配 tile 子集，block 之间互不干涉。

检查清单：

- 调整 `COMPUTE_BLOCK_NUM` 使每个 block 承担接近相等的 tile 数。
- 对于不同形状的矩阵，重新计算 tile 总数 `G_NUM_TILES = (M_padded/128) × (N_padded/256)`。

### 2) 选择合适的 base tile

L0A 与 L0B 使用双缓冲（ping/pong），每个 buffer 上限 32 KiB。

对于 FP16 输入（2 bytes/elem）：

- L0A tile bytes ≈ `baseM × baseK × 2` = `128 × 64 × 2 = 16 KiB`
- L0B tile bytes ≈ `baseK × baseN × 2` = `64 × 256 × 2 = 32 KiB`

通信 tile 大小为 `baseM × baseN × sizeof(FP16) = 128 × 256 × 2 = 64 KB`。

### 3) 用 L1 的 "stepK" 缓存提升复用

`stepKa=stepKb=4`：一次 TLOAD 搬入 4 个 K-slice 到 L1，后续 TEXTRACT 逐个提取到 L0。

L1 使用量：`2×64KB(A) + 2×128KB(B) = 384KB ≤ 1024KB`（L1 总容量）。

增大 `stepK` 可减少 DMA 启动开销，但须确保不超过 L1 容量。

### 4) 保持流水线重叠

计算 kernel 内部的双缓冲（L1/L0A/L0B）+ 计算与通信之间的双流重叠是性能核心。

当你观察到：

- **通信时间远大于计算时间** → 计算侧已充分优化，重点优化通信效率或增大重叠。
- **计算时间远大于通信时间** → 通信已被完全隐藏，重点优化计算侧。

### 5) 调整通信 block 数

`COMM_BLOCK_NUM` 控制通信 kernel 的 AIV 并行度。通过 `--comm-blocks` 参数调整。

注意：在 **Ascend910B** 上实测发现，将 `COMM_BLOCK_NUM` 从 24 提升到 48（使用更多 AIV 参与通信）会导致 AIC 计算时间显著增加（约 +24%），原因是 HBM 带宽争用和 TSCH 调度开销；当时更稳妥的默认是 24。迁移到 **Ascend950PR** 后，应以对应 `.ini` 的 `**vector_core_cnt`** 为上限重新权衡（例如 **958b 为 64**、**9599 为 72**），是否仍适用「24 最优、48 回退」需结合目标 SoC 与 `platform_config` 重新 profiling，不宜照搬。

### 6) 约束条件

- K 必须能被 `G_BASE_K × G_STEP_KA`（默认 64×4=256）整除。
- M 自动 pad 到 128 对齐，N 自动 pad 到 256 对齐。
- 所有窗口内 buffer 必须在每个 rank 上分配在相同偏移处。
- `signal_matrix` 在每轮迭代开始前通过 `aclrtMemset` 清零。

## 构建与运行

1. 配置 Ascend CANN 环境：

```bash
export ASCEND_CANN_PATH=/usr/local/Ascend/cann-<version>/set_env.sh
source "${ASCEND_CANN_PATH}"
```

1. 激活 conda 环境（需含 Python + NumPy）：

```bash
conda activate <your-conda-env>
```

1. 运行示例（2 卡）：

```bash
cd ${git_clone_path}/kernels/manual/a5/gemm_ar
./run.sh --nranks 2 --soc-version Ascend950PR_958b
```

1. 指定起始设备编号：

```bash
FIRST_DEVICE=0 ./run.sh --nranks 2 --soc-version Ascend950PR_958b
```

1. 自定义 block 分配：

```bash
./run.sh --nranks 2 --soc-version Ascend950PR_958b --compute-blocks 20 --comm-blocks 4
```

成功时输出：

```text
GEMM AllReduce demo completed successfully.
```

### 环境变量说明


| 环境变量                 | 用途                         | 默认行为                                               |
| -------------------- | -------------------------- | -------------------------------------------------- |
| `ASCEND_CANN_PATH`   | CANN `set_env.sh` 的完整路径    | 自动 glob `/usr/local/Ascend/cann-*/set_env.sh` 取最新版 |
| `MPI_SEARCH_DIRS`    | MPI `bin/` 目录搜索路径（空格分隔）    | 搜索 `/usr/local/mpich/bin`、`/home/mpich/bin` 等常见路径  |
| `ASCEND_DRIVER_PATH` | Ascend driver 路径（CMake 使用） | 默认 `/usr/local/Ascend/driver`                      |
| `MPI_LIB_PATH`       | `libmpi.so` 绝对路径（运行时动态加载）  | 由 `run.sh` 根据找到的 MPI 自动设置                          |
| `HCCL_BUFFSIZE`      | HCCL RDMA 窗口大小（MB）         | 由 `run.sh` 根据 M/N 自动计算                             |
| `FIRST_DEVICE`       | 起始 NPU 设备编号                | 默认 0                                               |


## 修改矩阵维度

修改 `gemm_ar_config.h` 中的 `CONFIG_G_M`、`CONFIG_G_K`、`CONFIG_G_N` 即可，所有源文件通过 include 共享配置。也可通过 CMake 参数传入：

```bash
cmake -DCONFIG_G_M=8192 -DCONFIG_G_K=8192 -DCONFIG_G_N=2048 ..
```

约束：K 必须能被 `G_BASE_K × G_STEP_KA`（默认 64×4=256）整除。`HCCL_BUFFSIZE` 由 `run.sh` 自动计算。

## 常见问题


| 问题                             | 原因与解决                                                                |
| ------------------------------ | -------------------------------------------------------------------- |
| `HCCL window too small`        | 窗口需要覆盖 pad 后的 `reduced_output` 和 `signal_matrix`。先检查是否手工覆盖了 `HCCL_BUFFSIZE`；`run.sh` 默认会按 `pad(M) × pad(N) × 2 / 1MB + 64MB` 自动抬高 |
| `HcclGetRootInfo failed: 7`    | 上次运行残留脏状态。执行 `rm -rf /dev/shm/sem.hccl*; ipcrm -a` 或等待 ~30s 重试       |
| HCCL 初始化后挂死                    | rank 同步问题，检查所有 rank 是否到达 `CommMpiBarrier`                            |
| 通信 kernel 段错误                  | 通常是窗口地址无效，验证 `windowsIn[]` 值非零                                       |
| signal wait 卡死或 AG stall       | `signal_matrix` 未在迭代间清零，或 subtile-ready / AG summary 计数映射有误；先检查 `resetState` 是否 memset 了 signal_matrix |
| 验证失败 max_diff 较大               | FP16 精度有限，验证容差为 atol=1.0, rtol=0.01；若 diff 异常大，检查 subtile-ready / AG summary 同步与 owner 映射 |
| `aclInit repeat init` (100002) | 无害，代码已做保护，同一进程只调用一次 `aclInit`                                        |
| `--allow-run-as-root` 失败       | 本项目使用 MPICH，此选项是 OpenMPI 专用                                          |


## 构建系统

- **编译器**：bisheng（CANN 内置 clang 15.0.5）
- **Cube kernel**：`--cce-aicore-arch=dav-c220-cube -DMEMORY_BASE`
- **Vec kernel**：`--cce-aicore-arch=dav-c220-vec -DMEMORY_BASE`
- **Host 可执行文件**：`-xc++` 标准编译
- **链接库**：`runtime`、`ascendcl`、`hcomm`、`tiling_api`
- pto-comm-isa 的 include 路径**必须放在首位**，以覆盖 CANN 自带的 `pto_tile.hpp`

## 变更记录


| 日期         | 变更 |
| ---------- | ------------------------------------------------ |
| 2026-04-15 | 增加 A5 版本的 `gemm_ar` 适配 |
| 2026-04-21 | 通信模式从 `RS -> DeviceBarrier -> AG` 改成 `subtile-ready / AG-summary overlap` |
| 2026-04-24 | 修复 ready queue 的显式 slot 寻址与可见性发布链路，并移除过时的 `debug_state` 诊断链路 |
