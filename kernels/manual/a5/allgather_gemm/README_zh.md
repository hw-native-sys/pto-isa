# AllGather + GEMM 通算融合示例

## 概览

本示例演示如何在 Ascend AI Core 上实现 **AllGather + GEMM** 融合算子，采用 **M 维切分** 与 **chunk 流式流水线** 设计。在多卡 LLM 推理场景中，每个 rank 持有矩阵 `A` 在 M 维上的一段本地切片。与“先完成 AllGather，再启动 GEMM”的串行方式不同，本实现将通信与计算按 chunk 粒度重叠执行：通信 kernel 每搬完一个 chunk 并发出就绪信号后，计算 kernel 就可以立即开始处理该 chunk，从而尽可能将通信延迟隐藏在计算之后。

## 支持的 AI 处理器

- Ascend950PR（A5 系列）

## 目录结构

```text
kernels/manual/a5/allgather_gemm/
├── main.cpp                           # Host 侧入口: HCCL 初始化、双流调度、warmup、校验、性能统计
├── allgather_gemm_comm_kernel.cpp     # AIV 通信 kernel: 通过 TPUT 执行 AllGather
├── allgather_gemm_compute_kernel.cpp  # AIC 计算 kernel: 基于 chunk 就绪等待的流式 GEMM
├── kernel_launch.hpp                  # Host 侧 kernel launcher 声明
├── ready_queue.hpp                    # ChunkFlagMatrix / summary counter 元数据
├── run.sh                             # 构建与运行脚本（环境探测、shape/block override、多 rank 启动）
├── scripts/
│   └── gen_data.py                    # 输入数据生成（FP16 A 切片 + B + golden.bin）
└── CMakeLists.txt                     # 构建配置
```

## 算子说明

### 计算功能

本示例实现 AllGather 后接 GEMM：

$$
C = A \times B
$$

其中：

- `n_ranks` 个 rank 中，每个 rank 持有 `A` 在 M 维上的一个本地切片：行范围为 `[rank * M/n_ranks, (rank+1) * M/n_ranks)`。
- `B` 在所有 rank 上均复制一份，形状为 `K × N`，数据类型为 FP16。
- AllGather 收齐完整 `A`（`M × K`，FP16）后，每个 rank 计算完整输出 `C`（`M × N`，FP32）。

AllGather 与 GEMM 被融合为一个流式流水线，因此无需等全部 AllGather 完成后再启动 GEMM。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `AllGather + GEMM`（通算融合） |
| 输入 | `A`: `M × K`, `float16`, `ND`（按 M 维在各 rank 间切分）；`B`: `K × N`, `float16`, `ND`（各 rank 复制） |
| 输出 | `C`: `M × N`, `float32`, `ND` |
| 通信 kernel | `RingCommStreamingKernel`（AIV） |
| 计算 kernel | `AllGatherGemmComputeStreamingKernel`（AIC） |

## 整体架构

### 双流并发

通信与计算 kernel 由 Host 侧并发下发到两个独立的 AICPU stream：

- **Comm stream** -> `RingCommStreamingKernel` 运行在 **AIV**（Vector）侧
- **Compute stream** -> `AllGatherGemmComputeStreamingKernel` 运行在 **AIC**（Cube）侧

Host 侧连续下发两个 kernel，并在二者都完成后统一同步。

### AI Core 资源分工

| 单元 | 硬件引擎 | 本示例中的职责 |
| --- | --- | --- |
| **AIC (Cube)** | 矩阵引擎 | 计算 kernel：GEMM（`TMATMUL` / `TMATMUL_ACC`） |
| **AIV (Vector)** | Vector / DMA | 通信 kernel：RDMA 数据搬运（`TPUT`）+ 就绪通知（`TNOTIFY`） |

### 流式流水线

```text
串行执行:
  [ AllGather 全部完成 ] ──► [ GEMM 全部完成 ]

流式流水线执行:
  Comm (AIV):   [chunk0 TPUT][TNOTIFY] [chunk1 TPUT][TNOTIFY] [chunk2 TPUT][TNOTIFY] ...
                      │                      │                      │
                      ▼                      ▼                      ▼
  Compute (AIC): [local GEMM]  [TWAIT chunk0][GEMM chunk0] [TWAIT chunk1][GEMM chunk1] ...
                  (zero-wait)
```

计算 kernel 分为两个阶段：

1. **Phase 1（local）**：优先处理本 rank 的本地 row-group，这部分数据已经在共享内存中，不需要等待。
2. **Phase 2（remote）**：对每个远端 rank 的 row-group，使用 summary counter 上的 `TWAIT` 等待 chunk 到达；一旦满足条件，立即继续计算。

## 优化说明

- **summary 单调计数器 + `TWAIT`**：通信 kernel 每完成一个 chunk 传输后，通过 `TNOTIFY AtomicAdd` 原子递增对应 source 的 summary counter。计算 kernel 使用硬件 `TWAIT`（compare-and-block）等待计数器达到预期值，无需轮询，也不会 busy-spin。
- **本地数据零等待优先执行**：计算 kernel 先处理本 rank 的本地 row-group（Phase 1），无需检查标志位，可与远端 chunk 的传输自然重叠。
- **发送顺序与消费顺序对齐**：通信 kernel 按照计算 kernel 的消费顺序发送 chunk，尽量缩短等待时间。
- **连续 K 维累加流水线**：在每个 row-group 内，第一个 K-block 使用 `TMATMUL`，后续 K-block 使用 `TMATMUL_ACC`，保持连续累加，不做中间结果回写/回读。
- **L1/L0 两级双缓冲**：L1 中的 `aMatTile[2]` / `bMatTile[2]` 与 L0A/L0B 中的 `aTile[2]` / `bTile[2]` 组成双缓冲流水线，实现 DMA（`TLOAD`）-> 提取（`TEXTRACT`）-> 计算（`TMATMUL`）的重叠。
- **AIV 并行 full-mesh 通信**：在 full-mesh 模式下，每个 rank 的多个 AIV block 可同时向所有其他 rank 直接执行 `TPUT`，以提升带宽利用率。
- **动态 chunk 大小**：`ComputeOptimalChunkSize()` 自动选择 chunk 粒度，将每个 source 的 chunk 数控制在 64-128 之间，在流水线深度与信号开销之间做平衡。
- **弹性 block 分配**：通信 kernel 会根据可用 block 数自动适配；当 block 数多于目标数时，按 destination 均匀分配，否则采用 round-robin 方式调度 work item。

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
cd ${git_clone_path}/kernels/manual/a5/allgather_gemm
```

运行默认 2 卡 A5 示例：

```bash
bash run.sh -r npu -v Ascend950PR_958b
```

指定 rank 数和 GEMM shape：

```bash
bash run.sh -r npu -v Ascend950PR_958b -n 4 --gm 4096 --gk 2048 --gn 1536
```

指定 base tile 和 block 配置：

```bash
bash run.sh -r npu -v Ascend950PR_958b -n 2 --gm 2048 --gk 2048 --gn 1024 --base-m 128 --base-n 256 --compute-blocks 32 --comm-blocks 24
```

在模拟器模式下运行：

```bash
bash run.sh -r sim -v Ascend950PR_958b -n 2 --gm 2048 --gk 2048 --gn 1024
```

`run.sh` 会检查以下 shape 约束：

- `--base-n` 必须能被 4 整除
- `G_M % G_BASE_M == 0`
- `G_K % G_BASE_N == 0`
- `G_N % G_BASE_N == 0`

脚本还会：

- 在未提供 `ASCEND_CANN_PATH` 时自动探测并 `source` 最新的 CANN `set_env.sh`
- 搜索常见 MPICH 安装路径，并补全 `PATH` / `LD_LIBRARY_PATH`
- 在每次运行前清理残留的 HCCL 共享内存状态
- 在构建和启动前打印本次使用的 shape、base tile 和 block 配置

### 命令行参数

| 参数 | 说明 |
| --- | --- |
| `-r/--run-mode` | 运行模式：`npu` 或 `sim` |
| `-v/--soc-version` | SoC 版本字符串，例如 `Ascend950PR_958b` |
| `-n/--n-ranks` | 传给 `mpirun` 的 MPI rank 数 |
| `--gm` | 数据生成和编译期配置使用的全局 M 维 |
| `--gk` | 数据生成和编译期配置使用的全局 K 维 |
| `--gn` | 数据生成和编译期配置使用的全局 N 维 |
| `--base-m` | M 维 tile 大小 |
| `--base-n` | N 维 tile 大小（要求能被 4 整除） |
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
| 2025-07-01 | 初始实现：基于 M 维切分流式流水线的 AllGather + GEMM 融合版本 |
| 2026-04-21 | 将 A5 的运行/文档规范与 A2/A3 版本对齐：补充环境感知 `run.sh`、更清晰的启动输出，以及 benchmark/输出说明，保持 chunk 流式流水线语义不变 |
