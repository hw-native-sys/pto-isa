# AllGather + GEMM 通算融合算子示例

## 概览

本示例演示如何使用 PTO 实现多卡 AllGather + GEMM 融合算子，采用 **M 维切分** 与 **chunk 流式流水线** 设计。在多卡 LLM 推理场景中，每个 rank 持有矩阵 `A` 沿 M 维的一个分片。本实现不等 AllGather 完成就开始计算，而是在 chunk 粒度上重叠通信与计算——通信 kernel 每传输完一个 chunk 即通知计算 kernel 开始处理，从而将通信延迟隐藏在计算之后。

## 支持的 AI 处理器

- A2/A3

## 目录结构

```
kernels/manual/a2a3/allgather_gemm/
├── main.cpp                           # Host 入口：HCCL 初始化、双流调度、warmup、验证、性能统计
├── allgather_gemm_comm_kernel.cpp     # AIV 通信 kernel：通过 TPUT 实现 AllGather
├── allgather_gemm_compute_kernel.cpp  # AIC 计算 kernel：流式 GEMM，等待 chunk 就绪
├── kernel_launch.hpp                  # Host 侧 kernel launcher 声明
├── ready_queue.hpp                    # ChunkFlagMatrix / summary counter 元数据
├── run.sh                             # 构建与运行脚本（环境探测、shape/block override、多 rank 启动）
├── scripts/
│   └── gen_data.py                    # 输入数据生成（FP16 A 分片 + B + golden.bin）
├── prof_analysis/                     # 可选 profiling 知识与 overlap 解析脚本
└── CMakeLists.txt                     # 构建配置
```

## 算子说明

### 计算功能

本示例实现 AllGather + GEMM：

$$
C = A \times B
$$

其中：

- `n_ranks` 个 rank 各持有 `A` 的一个 M 分片：行 `[rank * M/n_ranks, (rank+1) * M/n_ranks)`。
- `B` 在所有 rank 上复制（`K × N`，FP16）。
- AllGather 收集完整 `A`（`M × K`，FP16）后，每个 rank 计算完整 `C`（`M × N`，FP32）。

AllGather 与 GEMM 被融合为流式流水线，计算在 AllGather 完成之前即开始。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `AllGather + GEMM`（通算融合） |
| 输入 | `A`: `M × K`, `float16`, `ND`（M 维跨 rank 切分）; `B`: `K × N`, `float16`, `ND`（复制） |
| 输出 | `C`: `M × N`, `float32`, `ND` |
| 通信 Kernel 名称 | `RingCommStreamingKernel`（AIV） |
| 计算 Kernel 名称 | `AllGatherGemmComputeStreamingKernel`（AIC） |

## 整体架构

### 双流并行

通信与计算 kernel 运行在两个独立的 AICPU stream 上，由 host 并发下发：

- **Comm stream** → `RingCommStreamingKernel` 运行在 **AIV**（Vector）核。
- **Compute stream** → `AllGatherGemmComputeStreamingKernel` 运行在 **AIC**（Cube）核。

Host 依次下发两个 kernel 后等待二者完成。

### AI Core 资源

| 单元 | 硬件引擎 | 本示例中的角色 |
| --- | --- | --- |
| **AIC（Cube）** | 矩阵引擎 | 计算 kernel：GEMM（`TMATMUL` / `TMATMUL_ACC`） |
| **AIV（Vector）** | 向量 / DMA | 通信 kernel：RDMA 数据传输（`TPUT`）+ 信号（`TNOTIFY`） |

### 流式流水线

```
串行执行：
  [ AllGather 全部完成 ] ──► [ GEMM 全部完成 ]

流式流水线执行：
  通信 (AIV):   [chunk0 TPUT][TNOTIFY] [chunk1 TPUT][TNOTIFY] [chunk2 TPUT][TNOTIFY] ...
                      │                      │                      │
                      ▼                      ▼                      ▼
  计算 (AIC):   [本地 GEMM]  [TWAIT chunk0][GEMM chunk0] [TWAIT chunk1][GEMM chunk1] ...
                 (无需等待)
```

计算 kernel 分两个阶段运行：

1. **阶段 1（本地）**：立即处理本 rank 的 row-group（数据已在共享内存中，无需等待）。
2. **阶段 2（远端）**：对每个远端 rank 的 row-group，使用 `TWAIT` 在 summary counter 上阻塞等待 chunk 到达，就绪后立即计算。

## 优化说明

- **Summary 单调计数器 + TWAIT**：通信 kernel 每完成一个 chunk 传输后原子递增 per-source summary counter（`TNOTIFY` AtomicAdd）。计算 kernel 使用硬件 `TWAIT`（compare-and-block）等待计数器到达期望值——零轮询开销，无忙等。
- **本地数据零等待优先**：计算 kernel 优先处理本 rank 的 row-group（阶段 1），无需 flag 检查，与远端 chunk 传输重叠执行。
- **发送顺序与消费顺序一致**：通信 kernel 按计算 kernel 消费的顺序发送 chunk，最小化等待时间。
- **连续 K 累积流水线**：每个 row-group 内，K-tile 通过 `TMATMUL`（首次迭代）+ `TMATMUL_ACC`（后续迭代）处理，保持连续累积，无需中间 store/reload。
- **L1/L0 两级双缓冲**：L1 中 `aMatTile[2]` / `bMatTile[2]`，L0A/L0B 中 `aTile[2]` / `bTile[2]`，实现 DMA（`TLOAD`）↔ 提取（`TEXTRACT`）↔ 计算（`TMATMUL`）重叠。
- **AIV 并行全连接通信**：在 full-mesh 模式下，每个 rank 的 AIV 核直接 `TPUT` 数据到所有其他 rank，多个 AIV block 分配到每个目的地以充分利用带宽。
- **动态 chunk 大小**：`ComputeOptimalChunkSize()` 自动选择 chunk 粒度，将每个 source 的 chunk 数保持在 64–128 范围内，平衡流水线深度与信号开销。
- **灵活的 block 分配**：通信 kernel 适配可用的 block 数量——当 block 多于目的地时均匀分配；否则通过 round-robin 调度。

## 实测性能（参考）

以下数据在 Ascend A3（910B1）上测得，fp16 输入 → fp32 输出，使用 `aclrtEvent` 计时（5 次 warmup + 10 次计时取平均）。TFLOPS 计算公式：`2 × M × K × N / time`。

### 2 卡

| M | K | N | 执行时间（ms） | TFLOPS |
| --- | --- | --- | --- | --- |
| 2048 | 2048 | 1024 | 0.297 | 28.96 |
| 4096 | 4096 | 2048 | 1.098 | 62.57 |
| 4096 | 4096 | 4096 | 1.231 | 111.62 |
| 8192 | 4096 | 4096 | 2.519 | 109.13 |

### 4 卡

| M | K | N | 执行时间（ms） | TFLOPS |
| --- | --- | --- | --- | --- |
| 4096 | 4096 | 4096 | 0.986 | 139.42 |
| 8192 | 4096 | 4096 | 1.648 | 166.75 |

### 8 卡

| M | K | N | 执行时间（ms） | TFLOPS |
| --- | --- | --- | --- | --- |
| 8192 | 4096 | 4096 | 1.439 | 191.03 |
| 16384 | 4096 | 4096 | 2.585 | 212.71 |

### 这些数字意味着什么

- **多卡扩展性**：对于相同的 GEMM 总规模（M=8192, K=4096, N=4096），吞吐从 109 TFLOPS（2 卡）提升到 167 TFLOPS（4 卡）再到 191 TFLOPS（8 卡）。这反映了有效的通算重叠——随着每个 rank 计算的本地 GEMM 变小，通信开销的相对占比增大，但流式流水线成功隐藏了其中的大部分。
- **更大的 M 提升吞吐**：8 卡场景下，M 从 8192 翻倍到 16384，吞吐从 191 提升到 213 TFLOPS，因为计算通信比增大，流水线有更多 chunk 可供重叠。
- **小规模场景受通信主导**：2048×2048×1024（2 卡）仅达到 29 TFLOPS——AllGather 数据量小但固定的通信开销（HCCL 建链、信号传递）未被充分摊销。

## 构建与运行

当前的 `run.sh` 会一次完成三件事：

1. 生成输入数据和 golden 输出到 `./out`
2. 重新创建 `build/` 并重编 `allgather_gemm`
3. 启动 `mpirun -n <n_ranks> ./allgather_gemm`

运行前，请先配置 Ascend CANN 环境，确保 `ASCEND_HOME_PATH` 可用：

```bash
source <cann-install>/set_env.sh
```

然后进入示例目录：

```bash
cd ${git_clone_path}/kernels/manual/a2a3/allgather_gemm
```

运行默认 2 卡 A3 示例：

```bash
bash run.sh -r npu -v Ascend910B1 -n 2
```

指定 rank 数和 GEMM shape：

```bash
bash run.sh -r npu -v Ascend910B1 -n 4 --gm 4096 --gk 2048 --gn 1536
```

指定 chunk tile 和 block 配置：

```bash
bash run.sh -r npu -v Ascend910B1 -n 2 --gm 2048 --gk 2048 --gn 1024 --base-m 128 --base-n 256 --compute-blocks 32 --comm-blocks 24
```

在模拟器模式下运行：

```bash
bash run.sh -r sim -v Ascend910B1 -n 2 --gm 2048 --gk 2048 --gn 1024
```

`run.sh` 会检查以下 shape 约束：

- `--base-n` 必须能被 4 整除
- `G_M % G_BASE_M == 0`
- `G_K % G_BASE_N == 0`
- `G_N % G_BASE_N == 0`

### 命令行参数

| 参数 | 说明 |
| --- | --- |
| `-r/--run-mode` | 运行模式：`npu` 或 `sim` |
| `-v/--soc-version` | SoC 版本字符串，例如 `Ascend910B1` |
| `-n/--n-ranks` | 传给 `mpirun` 的 MPI rank 数 |
| `--gm` | 数据生成和编译期配置使用的全局 M 维 |
| `--gk` | 数据生成和编译期配置使用的全局 K 维 |
| `--gn` | 数据生成和编译期配置使用的全局 N 维 |
| `--base-m` | chunk tile 的 M 维大小 |
| `--base-n` | chunk tile 的 N 维大小（要求能被 4 整除） |
| `--compute-blocks` | 覆盖计算 kernel 的 block 数配置 |
| `--comm-blocks` | 覆盖通信 kernel 的 block 数配置 |

## Benchmark 与输出说明

当前 host 程序会在最终功能校验前执行三类 benchmark：

1. **Compute-only**：由 host 直接把所有 chunk 标记为 ready，只测纯计算延迟
2. **Sequential**：先让通信完整结束，再启动计算
3. **Pipelined**：通信和计算在两个 stream 上并发启动，测量重叠效果

benchmark 结束后，会再执行一次最终 functional verification，并与 `golden.bin` 做结果比对。

成功运行时，输出类似：

```text
[INFO] Running warmup...
[INFO] Functional run completed. Verification PASSED.
[SUCCESS] AllGather GEMM (HCCL)
  Compute-only:   ...
  Sequential:     ...
  Pipelined:      ...
  Speedup:        ...
  Overlap eff:    ...
```

每个 rank 的输出张量也会写到：

```text
out/output_rank<rank_id>.bin
```

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2025-07-01 | 初始实现：基于 M 维切分和 chunk 流式流水线的 AllGather + GEMM 融合版本 |
| 2026-04-21 | 恢复 host 侧 benchmark/profiling 流程，以及 run.sh/CMake 中对 shape、base tile、block override 的参数透传 |
