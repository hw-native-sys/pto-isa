# PTO MegaMoE Dispatch + Combine 融合算子示例

## 概览

本示例演示如何使用 PTO Manual kernel 实现 MegaMoE 中 `dispatch -> GMM1 -> SwiGLU -> GMM2 -> combine -> unpermute` 的端到端融合流程。算子把传统 MoE 中多次重排、AlltoAllV 通信和 grouped FFN 计算合并到一个大 kernel 内，以 local expert 为流水粒度做 AIC/AIV 重叠。

## 支持的 AI 处理器

- Ascend910B1
- Ascend910B / Ascend910C
- Ascend910_93 / Ascend910_9391 / Ascend910_9381 / Ascend910_9372 / Ascend910_9392 / Ascend910_9382 / Ascend910_9362

> 当前目录位于 `a2a3` 手写 kernel 下，性能和运行验证主要面向 A2/A3 形态。`CMakeLists.txt` 中也保留了 `Ascend910_9599` 的 `dav-c310` 编译分支，但使用前需要按目标环境重新验证。

## 目录结构

```text
kernels/manual/a2a3/dispatch_mega_combine/
├── CMakeLists.txt                  # 构建配置，生成 host 可执行文件和 device kernel so
├── run.sh                          # 数据生成、构建、mpirun 执行的一键脚本
├── main.cpp                        # Host 入口：加载 case、初始化 ACL/HCCL/MPI、申请窗口、启动 kernel、校验和计时
├── kernel_launch.cpp               # Device kernel launch 包装
├── runtime_context.*               # 单 rank runtime、HCCL window、device/context 管理
├── tiling_builder.*                # Host 侧 tiling 构造和 workspace 规划
├── data_utils.*                    # case 数据文件读写、校验和性能统计辅助
├── comm_mpi.h                      # MPI 动态加载包装
├── scripts/
│   ├── gen_data.py                 # synthetic 输入、权重、golden 生成
│   └── tests/                      # 数据分布相关单测
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoe device 主流程入口
│   ├── front_reorder.h             # front reorder 公共后处理和量化 scatter
│   ├── front_fullload_sort.h       # FullLoad 小 route 排序路径
│   ├── front_vms_sort.h            # OneCore / MultiCore VMS 排序路径
│   ├── dispatch.h                  # destination rank 拉取 offsetA，生成 GMM1 输入
│   ├── gmm_common.h                # GMM1/GMM2 共享 tile 调度和 helper
│   ├── gmm1.h                      # 第一层 grouped matmul
│   ├── swiglu.h                    # SwiGLU + dynamic quant
│   ├── gmm2.h                      # 第二层 grouped matmul
│   ├── combine.h                   # GMM2 输出远端写回 offsetD
│   ├── unpermute.h                 # topK weighted reduce 和原 token 顺序还原
│   └── utils/                      # PTO vector、sync、HCCL window、GMM pipeline helper
├── overview.md                     # 总体设计、性能对比和阶段伪码
├── front_reorder.md                # front reorder / sort / count-as-flag 细节
├── dispatch.md                     # dispatch 阶段契约和搬运策略
├── gmm1.md / gmm2.md               # GMM tile 调度、swizzle、同步和 pipeline
├── swiglu.md                       # SwiGLU 分段和量化策略
├── combine.md                      # combine large/small path 和远端写回协议
├── unpermute.md                    # unpermute 还原与累加策略
└── glden.md                        # Python golden batch rewrite 设计
```

## 算子说明

### 计算功能

本示例实现多 rank MoE FFN 主体流程：

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> routed expert-major token rows
  -> grouped GMM1
  -> SwiGLU activation + dynamic quant
  -> grouped GMM2
  -> combine 回源 rank
  -> topK weighted reduce
  -> out[rank, M, K]
```

逻辑公式可以理解为：

```text
for each rank, token:
  out[token] = sum_{topK route} probs[token, route] * FFN_expert(x[token])
```

其中 `FFN_expert` 由 int8 GMM1、SwiGLU、int8 GMM2 组成；前后通信通过 HCCL RDMA window 和 PTO 通信/同步 helper 完成。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| 输入 | `x`: `[M, K]`, `half/bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`; `weight1/weight2`: per local expert int8 packed weights; `scale1/scale2`: fixpipe scale |
| 输出 | `out`: `[M, K]`, `half/bfloat16` |
| Kernel 名称 | `dispatch_mega_combine_kernel` |
| Host 可执行文件 | `dispatch_mega_combine` |
| 默认脚本 case | `worldSize=8, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## 优化说明

- **expert 级流水重叠**：AIC 侧 GMM1/GMM2 和 AIV 侧 dispatch/SwiGLU/combine 按 local expert group 轮转推进，通过 hard flag 串接阶段边界。
- **front reorder 三路径**：按 UB 工作集大小选择 FullLoad、OneCore、MultiCore。小 route 尽量留在 UB 内完成排序、count、反排和 quant，避免不必要的 GM 中间态。
- **count-as-flag**：front 发布 `tokenPerExpert` 时给 count row 加 marker，peer 通过数据值等待到达，减少 AlltoAll count 后的整机同步。
- **GMM PTO tile 优化**：GMM1/GMM2 使用 PTO tile 编程，包含 output tile swizzle、L1 -> L0 多级复用、双缓冲和 fixpipe quant/cast。
- **SwiGLU 分段 overlap**：SwiGLU 按 segment 切分，第一段尽量压到 GMM2 启动前完成，降低 GMM1 -> GMM2 中间等待。
- **combine 双路径**：large token path 按完整 row 写回，small token path 按 GMM2 tile 拆 subtile，提高小 M 场景 AIV 利用率。

## Tiling 参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `M` | 由 `run.sh --m` 或 case.json 指定 |
| `K` | GMM 输入 hidden size；要求满足 packed row 和 GMM tile 对齐 |
| `N` | FFN 中间维度；GMM1 输出，SwiGLU 后进入 `N/2` |
| `topK` | 每 token 路由专家数 |
| `expertPerRank` | 每 rank 本地 expert 数 |
| `worldSize` | MPI/HCCL rank 数 |
| `maxOutputSize` | 每 rank routed row workspace 上限；典型性能 case 显式使用固定 workspace 上限 |
| `aicNum` | AIC 逻辑核数，默认脚本为 24 |
| `aivNum` | AIV 逻辑核数，默认脚本为 48 |
| `GMM baseM/baseN` | 主要 tile 口径为 `128 x 256` output tile |
| `Front FullLoad` | 由 `routeElems`、`K`、`expertNum` 和 UB 192 KiB 预算共同决定 |
| `Combine small path` | `problemM * topK <= 4096` 时倾向使用 subtile path |

## 支持 Case

当前典型性能 case 固定除 `M` 外的其它主参数：

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=24
aivNum=48
```

典型 case 列表：

| M | maxOutputSize | 命令 |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## 整体架构

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ FrontReorder (AIV)                                                          │
│   sort expertId route -> offsetA + count/prefix metadata                    │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ D2C ready / count-as-flag
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Expert-level overlapped pipeline                                             │
│                                                                              │
│ AIV: Dispatch(group i) -> SwiGLU(segment/group i) -> Combine(group i)         │
│ AIC:                    GMM1(group i)      -> GMM2(group i)                  │
│                                                                              │
│ Stages communicate with hard flags: D2C, C2V, V2C, G2C/Combine ready          │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ final boundary
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Unpermute (AIV)                                                              │
│   offsetD + probs + expandedRowIdx -> topK weighted reduce -> out[M, K]       │
└──────────────────────────────────────────────────────────────────────────────┘
```

完整执行顺序在 `MegaMoe::Process()` 中串联，阶段顺序为：

```text
FrontReorder -> Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine -> Unpermute
```

## FrontReorder 阶段

FrontReorder 在 source rank 内把 `[token, topK]` route 按 global expert 排序，并把量化后的 token row 写入本 rank 的 remote window：

```text
x[M, K] + expertId[M, topK]
  -> offsetA[expert-major rows, K + 32]
  -> expandedRowIdx / localTokenPerExpert / preSumBeforeRank / cumsumMM
```

三种排序路径语义一致：

- **FullLoad**：route、count、quant 工作集都能放入 UB；active AIV 重复完整排序，再按 route 区间分担 quant/scatter。
- **OneCore**：单核能完成排序，但 FullLoad 全工作集放不下；core0 排序后写 `frontExpandedExpert/frontExpandDstToSrc`。
- **MultiCore**：route 较大；多 AIV 先生成 VBS sorted run，再用 VMS 多轮归并，最后 sort-out 写 GM 中间态。

FrontReorder 结束时发布：

```text
tokenPerExpert[srcRank, globalExpert]
preSumBeforeRank[srcRank, localExpert]
cumsumMM[srcRank, localExpert]
expertTokenNums[localExpert]
```

## Dispatch 阶段

Dispatch 在 destination rank 上拉取所有 source rank 发给本地 expert 的 packed rows：

```text
srcRank.remoteWindow.offsetA[srcRowBase : srcRowBase + rows]
  -> workspace.gmA[dstRowBase : dstRowBase + rows, 0:K]
  -> workspace.perTokenScale1[dstRowBase : dstRowBase + rows]
```

地址由 front 生成的三张表决定：

```text
rows       = tokenPerExpert[srcRank, globalExpert]
srcRowBase = preSumBeforeRank[srcRank, localExpert]
dstRowBase = groupBase + (srcRank == 0 ? 0 : cumsumMM[srcRank - 1, localExpert])
```

每个 local expert group 搬运完成后，dispatch 设置 GMM1 ready flag，允许 AIC 开始消费该 group。

## GMM1 / SwiGLU / GMM2 阶段

### GMM1

GMM1 在 AIC 上执行第一层 grouped matmul：

```text
gmA[int8] x weight1[int8]
  -> int32 accumulator
  -> fixpipe scale1
  -> gmC[half]
```

每个 local expert 的输出 tile 网格按 `128 x 256` output tile 切分。线性 tile id 会通过 swizzle 映射到 `(blockM, blockN)`，让相邻 tile 更容易复用 L1 中的 B 侧权重。

### SwiGLU

SwiGLU 在 AIV 上消费 GMM1 的 `gmC` 和 dispatch 生成的 `perTokenScale1`：

```text
gmC * perTokenScale1
  -> silu(up) * gate
  -> dynamic quant
  -> gmPermutedToken[int8] + perTokenScale2[float]
```

SwiGLU 按 segment 切分；core0 写 segment metadata，其它 AIV 读取后按 row 分担计算。

### GMM2

GMM2 在 AIC 上执行第二层 grouped matmul：

```text
gmPermutedToken[int8] x weight2[int8]
  -> int32 accumulator
  -> fixpipe scale2
  -> gmm2Output[half]
```

GMM2 完成每个 local expert group 后设置 combine ready flag，AIV combine 才能写回该 group。

## Combine / Unpermute 阶段

Combine 在 AIV 上把 GMM2 输出乘 `perTokenScale2` 后写回 source rank 的 `offsetD`：

```text
gmm2Output[srcRow, 0:K] half
  -> fp32
  -> * perTokenScale2[srcRow]
  -> OutputElement
  -> srcRank.remoteWindow.offsetD[dstRow, 0:K]
```

路径选择：

- **DirectLarge**：大 token 量场景，按完整 row 写回。
- **DirectSmall**：小 token 量场景，按 subtile 拆分，提升 AIV 并行度。


Unpermute 是最后的源 rank 还原阶段：

```text
offsetD + probs + expandedRowIdx
  -> 按原 token/topK 加权累加
  -> out[M, K]
```

## 内存布局与 HCCL 窗口

HCCL remote window 主要承载跨 rank 可见的数据：

| Buffer | 位置 | 用途 |
| --- | --- | --- |
| `offsetA` | HCCL window | FrontReorder 写入 packed int8 token row，Dispatch 从 peer 拉取 |
| `offsetD` | HCCL window | Combine 写回源 rank，Unpermute 在源 rank 消费 |
| `tokenPerExpert` | HCCL window | count-as-flag 的跨 rank count row |
| `gmA` | workspace GM | Dispatch 生成的 GMM1 输入 |
| `gmC` | workspace GM | GMM1 输出，SwiGLU 输入 |
| `gmPermutedToken` | workspace GM | SwiGLU dynamic quant 后的 GMM2 输入 |
| `gmm2Output` | workspace GM | GMM2 输出，Combine 输入 |
| `expandedRowIdx` | workspace GM | route -> expert-major row 的反排索引 |
| `cumsumMM / preSumBeforeRank` | workspace GM | dispatch/combine 地址计算 metadata |

`run.sh` 会根据 `maxOutputSize` 和 `K` 估算 HCCL window 需求，并在需要时自动提高 `HCCL_BUFFSIZE`。

## 实测性能（参考）

性能对比和 pipeline 图见 `overview.md`：

```text
overview.md
  - PTO-ISA 实测效果对比
  - PTO megamoe 2048 case overlap 实况
```

当前结论摘要：

- 小 M 场景下 PTO MegaMoE 与 AscendC 实测基本持平。
- 随着 M 增大，PTO GMM tile 编程和通信计算 overlap 的收益更明显。
- 2048 case 中 AIC 基本满载，进一步优化重点在减少 AIC/AIV 对 HBM 的竞争。

## 性能优化指南

### 1. 先确认 FrontReorder case

小 route 量优先进入 FullLoad，可以显著减少 GM 中间态写读。若落到 OneCore/MultiCore，需要重点观察排序和 `frontExpandedExpert/frontExpandDstToSrc` 的 GM 可见性同步。

### 2. 保持 expert 级阶段边界清晰

Dispatch、GMM1、SwiGLU、GMM2、Combine 之间依赖 hard flag。优化时优先确认每个 group 的 set/wait 是否一一匹配，避免为了减少 `SYNCALL` 破坏跨 AIC/AIV 的真实数据依赖。

### 3. 优先优化 GMM tile 效率

GMM1/GMM2 是主耗时阶段。重点关注：

- output tile swizzle 是否改善 L1 复用；
- L1 -> L0 ping/pong 是否形成稳定重叠；
- `currentM` 较小时 AIC 是否负载不均；
- AIV 通信/写回是否与 GMM HBM 访问冲突。

### 4. 小 token 场景使用 subtile combine

小 M 下 direct row path 容易让 AIV 并行度不足。DirectSmall 通过 subtile 切分提高核利用率，但需要保持 `gmm2Tiling.l1TileN` 与 small path subtile 列宽对齐。

### 5. Golden 生成使用 batch backend

large synthetic case 默认使用 `python-batch` golden backend，避免逐 token GEMV 导致数据生成成为瓶颈。需要对照旧路径时再切到 `python-naive`。

## 构建与运行

配置 Ascend CANN 环境：

```bash
export ASCEND_CANN_PATH=/usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann/cann
source /usr/local/Ascend/cann/cann/set_env.sh
```

运行默认 2048 case：

```bash
cd ${git_clone_path}/kernels/manual/a2a3/dispatch_mega_combine
./run.sh
```

默认 case 由 `run.sh` 内部参数决定，当前是 `worldSize=8, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940`。A2/A3 场景使用脚本默认配置即可，不需要在典型 case 命令里显式指定芯片参数。

切换其它典型 M 档位时，建议直接使用上面的典型 case 命令。例如：

```bash
bash run.sh --world-size 8 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### 环境变量说明

| 环境变量 | 用途 | 默认行为 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN 安装目录 | 必须提前设置 |
| `CMAKE_COMPILER` | CMake 使用的编译器 | `bisheng` |
| `MPI_ENV_BIN` | MPI/conda bin 路径 | `/home/ntlab/miniconda3/envs/ltr_pto/bin` |
| `MPI_ENV_LIB` | MPI/conda lib 路径 | `/home/ntlab/miniconda3/envs/ltr_pto/lib` |
| `MPI_LIB_PATH` | `libmpi.so` 绝对路径 | `${MPI_ENV_LIB}/libmpi.so` |
| `MPI_RUNNER` | MPI 启动命令 | `mpirun` |
| `HCCL_BUFFSIZE` | HCCL RDMA window 大小 | `run.sh` 按 case 自动抬高到安全值 |

## 修改 Case 参数

典型性能数据只比较不同 `M`，其它主参数保持一致。推荐从“支持 Case”表中复制完整命令运行。

```bash
bash run.sh --world-size 8 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

常用约束：

- `K` 需要满足 packed row、GMM1/GMM2 tile 和量化路径的对齐要求。
- `N` 是 GMM1 输出维度，SwiGLU 后进入 GMM2 的维度为 `N / 2`。
- `maxOutputSize` 必须覆盖单 rank 接收的 routed rows 上限；典型性能 case 按表中命令显式传入。
- synthetic `expert_idx` 默认使用 global token round-robin，使小 M case 也能覆盖全局 expert。

## 常见问题

| 问题 | 原因与解决 |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | 运行 `run.sh` 前需要 source CANN 环境并导出 `ASCEND_HOME_PATH` |
| HCCL window too small | 手动设置的 `HCCL_BUFFSIZE` 低于 case 需求；取消覆盖或调大该变量 |
| MPI 启动失败 | 检查 `MPI_ENV_BIN`、`MPI_ENV_LIB`、`MPI_LIB_PATH` 是否指向同一个 conda/MPI 环境 |
| golden 生成很慢 | 使用默认 `python-batch`，必要时调大 `--golden-chunk-rows`；只有调试对照才使用 `python-naive` |
| 小 M 性能不稳定 | 优先检查 FullLoad case 是否命中、combine 是否走 DirectSmall、AIV 并发是否过高影响 GMM |
| 结果 diff 异常 | 先检查 data cache 是否复用旧分布；改变 expert 分布或 case 关键参数后不要使用旧 `out/` |

## 构建系统

- **编译器**：`bisheng`
- **Device kernel flags**：`-xcce --cce-aicore-arch=${CCE_AICORE_ARCH}`
- **Host executable**：`-xc++ -std=c++17`
- **输出 target**：`dispatch_mega_combine_kernel`、`dispatch_mega_combine`
- **链接库**：`stdc++`、`ascendcl`、`hcomm`、`runtime`、`tiling_api`、`platform`、`nnopbase`、`pthread` 等
- **PTO include**：仓库根目录 `include/` 会被放入 include path，用于 PTO tile/comm helper

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-06-26 | 新增 `dispatch_mega_combine` README，整理 MegaMoE 算子说明、阶段流程、构建运行和 FAQ |
