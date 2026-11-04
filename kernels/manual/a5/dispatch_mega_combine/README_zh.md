# PTO MegaMoE Dispatch + Combine 融合算子示例

跨 server URMA 通过同一启动脚本启用，设置 `--rank-num-per-server N` 即可；计算调度保持不变。

## 概览

本示例使用 PTO Manual kernel 实现 MegaMoE 的端到端融合流程，把 token 量化、route mask 交换、
grouped FFN、Combine 和 Unpermute 合并到一个 mixed-core kernel。当前实现采用 MXFP8 数据、AIC 到 AIV
的 CV 直传、动态 GMM mailbox 和 token-ready RankStreaming Unpermute，重叠通信、元数据准备、AIC
计算和 AIV 处理。

## 支持的 AI 处理器

- Ascend950PR（A5，arch35 / DAV_3510 系列）
- 工具链兼容别名 `Ascend910_9599`

本项目固定使用 `dav-c310` 和 `PTO_NPU_ARCH_A5` 构建 mixed-core kernel。`Ascend910B` 属于 A3，不能作为本目录的 `--soc` 参数。

## 多 server URMA 通信

同一二进制同时支持原有 MTE 路径和多 server URMA。使用 `--rank-num-per-server N` 开启跨 server URMA；同 server 通信继续使用 MTE。

URMA 使用 HCCL SharedPool，通信 queue 由 AIV0 独立持有。启用 URMA 不改变原计算流水、GMM 调度、Combine 和 Unpermute 调度。

## 目录结构

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # 构建配置，生成 host 可执行文件和 device kernel so
├── run.sh                          # 数据生成、构建、mpirun 执行的一键脚本
├── main.cpp                        # Host 入口：加载 case、初始化 ACL/HCCL/MPI、启动、校验和计时
├── kernel_launch.cpp               # Device kernel launch 包装
├── runtime_context.*               # 单 rank runtime、HCCL window、device/context 管理
├── tiling_builder.*                # Host 侧 tiling 构造和 workspace 规划
├── data_utils.*                    # case 数据文件读写和校验辅助
├── comm_mpi.h                      # MPI 动态加载包装
├── scripts/
│   └── gen_data.py                 # synthetic 输入、权重、golden 生成
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoe device 主流程入口
│   ├── front_reorder.h             # token-order MXFP8 量化、route mask 和 partial count 发布
│   ├── front_metadata_sort.h       # deferred metadata 使用的 expert-major route 排序
│   ├── deferred_route_metadata.h   # task descriptor、preSum 和逆向 route metadata
│   ├── dispatch.h                  # mask 压缩、远端 token 拉取和 GMM1 输入构造
│   ├── gmm_common.h                # GMM1/GMM2 共享 tile 调度和 helper
│   ├── gmm_task_producer.h         # GMM1/GMM2 P/C ticket 调度
│   ├── gmm_expert_progress.h       # Combine completion 到 rank progress 的协调
│   ├── gmm1.h                      # 第一层 grouped matmul 和 GMM1 mailbox 消费
│   ├── swiglu.h                    # SwiGLU + dynamic quant
│   ├── gmm2.h                      # producer 派发的 GMM2 mailbox 消费
│   ├── combine.h                   # CV tile 消费和远端 route output 写回
│   ├── unpermute.h                 # token-ready topK reduce 和原 token 顺序还原
│   └── utils/                      # PTO vector、sync、HCCL window、GMM pipeline helper
```

## 算子说明

### 计算功能

本示例实现多 rank MoE FFN 主体流程：

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> route mask 拉取为目标侧 expert-major token rows
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

其中 `FFN_expert` 由 MXFP8 GMM1、SwiGLU、MXFP8 GMM2 组成；前后通信通过 HCCL RDMA window 和 PTO 通信/同步 helper 完成。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| 输入 | `x`: `[M, K]`, `bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`, `float32`; `weight1/weight2`: 每个本地 expert 的 E4M3 数据；`scale1/scale2`: E8M0 scale |
| 输出 | `out`: `[M, K]`, `bfloat16` |
| Kernel 名称 | `dispatch_mega_combine_kernel` |
| Host 可执行文件 | `dispatch_mega_combine` |
| 默认脚本 case | `worldSize=2, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## 优化说明

Front 为每个 expert 分配一个 mask owner，单独发布紧凑的 raw count；Dispatch 与 deferred metadata 构造
expert-major 全局前缀，不再使用旧的 mask 尾部 count 协议。

所有 M 的 GMM1 都只直通 wave0，后续由 producer 分配；GMM2 从第一个 expert 开始就消费 mailbox 任务。
GMM 分组根据发布进度动态选择：`M <= 128` 目标是 GMM2 落后 4～6 个 expert，大场景目标是 1～2 个。
GMM 内部 L1 panel 按 expert 行数自适应，同时保留生产版 Gate/Up 拼接权重加载和 CV 交接。

| AIC | AIV | 默认 Dispatch AIV0 池 | 初始 GMM1:GMM2 映射 | 大 M Unpermute phase1 AIV0 |
| ---: | ---: | ---: | --- | ---: |
| 28 | 56 | 18 | 20:8 | 18 |
| 32 | 64 | 21 | 21:11 | 21 |
| 36 | 72 | 24 | 24:12 | 24 |

初始映射不是最终运行时配比。`M <= 256` 的 Dispatch 改由全部配对 AIV1 在 SwiGLU 前执行；更大 M 使用
上表的 AIV0 池，按商和余数分配 rank，避免丢弃剩余 worker。Dispatch 分工不绑定 GMM 分组。

Combine 保留 CV 交接，使用 PTO `TSTORE`。`M < 512` 的 Unpermute 只有一阶段；`M >= 512` 时先由
Dispatch 释放的 AIV0 池消费冻结进度快照，最终由全部 AIV 参与 live 阶段。

`run.sh --aicore-num 0|28|32|36` 选择有效核数，零表示设备查询值，不能超过实际物理核数。
保留 当前 rank 上限 `min(32, aicNum - 2)`；更宽 EP 会相应扩大 Dispatch 池，保证每个 rank 至少有一个 worker。

## Tiling 参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `M` | 由 `run.sh --m` 或 case.json 指定 |
| `K` | GMM 输入 hidden size；要求满足 packed row 和 GMM tile 对齐 |
| `N` | FFN 中间维度；GMM1 输出，SwiGLU 后进入 `N/2` |
| `topK` | 每 token 路由专家数 |
| `expertPerRank` | runtime 本地 expert 数：`4`、`8`、`16` 或 `32`，共用同一 kernel |
| `worldSize` | MPI/HCCL rank 数，上限为 `min(32, aicNum - 2)` |
| `maxOutputSize` | 每 rank routed row workspace 上限 |
| `aicNum` | 有效 AICore launch 核数；已验证 28、32 和 36 |
| `aivNum` | 按 `2 * aicNum` 推导：56、64 或 72 |
| `GMM baseM/baseN` | 主要 tile 口径为 `256 x 256` output tile |
| `Front Mask Pull` | 唯一 Front 实现；host 拒绝 clipping、非法 expert、inactive token 和接收容量不足 |
| `固定角色 AIV UB` | Dispatch、SwiGLU、Combine、Unpermute 使用 216 KiB 主区；尾部 40 KiB 保留给同步快照 |
| `Dispatch / Unpermute tile` | Dispatch 使用自适应 2～6 槽 packed-token ring；Unpermute 尽量保留完整 K 行、单 tile 最多 8192 列，并使用自适应 2～6 槽输入 ring |
| `GMM / AIV CV tile` | GMM1/GMM2 的 AIC 结果都由 FIX 从 L0C 转为 BF16 后通过 CV 槽交给配对 AIV1；基础调度 tile 为 `256 x 256` |

## 支持 Case

八卡调优性能 case 固定除 `M` 外的其它主参数。同一组 shape 也可用于两卡冒烟和正确性验证；
generic tiling 还支持其它满足约束的 rank 数和 runtime 本地 expert 数。

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=runtime（28、32 或 36），或通过 `run.sh --aicore-num 28|32|36` 指定
aivNum=2 * aicNum
```

典型 case 列表：

| M | maxOutputSize | 命令 |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## 整体架构

```text
Front：packed token + route mask + raw expert count
  -> Dispatch 与 deferred metadata 并行
  -> GMM1：wave0 直通 + mailbox 任务
  -> SwiGLU：配对 AIV1 消费 CV 并做 MXFP8 量化
  -> GMM2：mailbox 任务，逐 expert 等待输入 ready
  -> Combine：TSTORE 写同 server 输出或跨 server URMA staging
  -> Unpermute：token-ready TopK 加权累加
```

AIC 计算、AIV 元数据处理与通信重叠。producer 控制运行时 GMM 配比；MTE 和 URMA 保持相同的任务与完成语义。

## Front Mask Pull 阶段

所有 AIV 量化自身 token 后，直接复用私有 UB 生成 route mask；每个全局 expert 只有一个 mask owner。
raw count 独立使用 `[localExpert][sourceRank]` 布局。

deferred metadata 构造 expert-major 全局 `cumsumMM`、route 顺序、逆索引和任务描述符。
GMM2 入口等待所需元数据与 preSum，不再等待整个 Dispatch 结束；计算输入由每个消费者自行获取 ready。
URMA 继续使用原有 metadata 交换，并把接收的 count 放入相同的本地布局。

## Dispatch 阶段

Dispatch 只拉取命中的源 token，把 E4M3 数据和 E8M0 scale 写入 GMM1 workspace。
小 M 使用 AIV1 rolling worker；大 M 使用 AIV0 池按商和余数分配 rank。2～6 槽 ring 重叠数据搬入与写回，
每个 expert 的 256-row M tile 在 payload store 完成后发布 ready count。

小 M 的每个 AIV1 为配对 AIC 发布一行 expert count。跨 server payload 地址继续来自 URMA relay。
route record 不参与主体计算。

## GMM1 / SwiGLU / GMM2 阶段

### GMM1

GMM1 在 AIC 上执行第一层 MXFP8 grouped matmul：

```text
gmA[E4M3] / gmAScale[E8M0] x weight1[E4M3] / scale1[E8M0]
  -> TMATMUL_MX（A/B data 与 scale 一起参与计算）
  -> L0C FP32 accumulator
  -> FIX 转 BF16 并送入成对的 x/gate CV tile
```

每个 local expert 按 `256 x 256` 半输出 tile 切分，Gate/Up 以拼接权重 panel 成对计算，CV tile 采用交错布局，通过单槽 CV FIFO 直接交给
配对 AIV1。所有 M 都执行 wave0 直通，随后消费 mailbox 任务。逻辑输出 tile 仍是 `256 x 256`，
内部 L1 panel 根据整个 expert 的行数自适应。

### SwiGLU

SwiGLU 在配对 AIV1 上消费 BF16 Gate/Up 交错 CV tile：

```text
x/gate BF16 CV tile
  -> silu(x) * gate
  -> MXFP8 dynamic quant
  -> gmSwigluA[E4M3] + gmSwigluScale[E8M0]
```

CV control stream 携带准确的 GMM task 顺序。SwiGLU 完成每个 E4M3/E8M0 写回后，再增加对应
expert 的 GMM2 dependency counter；达到该 expert 所需的 SwiGLU tile 数后，GMM2 才开始处理该 expert。

### GMM2

GMM2 在 AIC 上执行第二层 MXFP8 grouped matmul：

```text
gmSwigluA[E4M3] / gmSwigluScale[E8M0] x weight2[E4M3] / scale2[E8M0]
  -> TMATMUL_MX（A/B data 与 scale 一起参与计算）
  -> L0C FP32 accumulator
  -> FIX 转 BF16 并送入 GMM2/Combine CV tile
```

所有 GMM2 tile 都由 producer 分配。GMM1 释放的 AIC 根据运行时分组与发布进度加入，不经过全局
GMM1/GMM2 barrier。消费者先等待自身输入依赖，再计算并通过 CV FIFO 将 BF16 结果交给配对 Combine AIV1。

## Combine / Unpermute 阶段

Combine 在配对 AIV1 上把每个 BF16 GMM2 CV tile 直接写到源 rank 的 compact output row：

```text
GMM2 BF16 CV tile
  -> 与各 source-rank cumsum row 区间求交
  -> srcRank.combineOutputByRouteSlot[compactRow, 0:K]
```

Combine 预取 `cumsumMM` 和 `preSumBeforeRank`，然后对每个 source-rank 交集执行带 stride 的 PTO `TSTORE`。发布 expert completion count 前会先排空此前的远端 MTE3 写，因此 progress coordinator
向各源 rank 暴露的是有序 expert prefix。

Unpermute 是最后的源 rank 还原阶段：

```text
expandedRowIdx[token * topK + topk]
  -> combineOutputByRouteSlot[compactRow] * probs[token, topk]
  -> 按原 token/topK 累加
  -> out[M, K]
```

`M < 512` 时全部 AIV 直接执行单个 live 阶段；`M >= 512` 时先由 Dispatch 释放的 AIV0 池消费冻结 progress
快照允许的 token，再由全部 AIV 根据 live expert-prefix progress 消费剩余 token。RankStreaming 是唯一
Unpermute 实现，不支持的 shape 由 Host 校验直接拒绝，不回退到 barrier kernel。

## 内存布局与 HCCL 窗口

HCCL remote window 主要承载跨 rank 可见的数据：

| Buffer | 位置 | 用途 |
| --- | --- | --- |
| `sourceTokenRecords` | HCCL window | 每个源 token 一份 packed E4M3/E8M0 记录；Dispatch 按 `routeSlot / topK` 拉取 |
| `routeMaskSlots` | HCCL window | 每个 local expert/source rank 的 mask 和对齐 lane partial count |
| `combineOutputByRouteSlot` | HCCL window | Combine 写入的 expert-major compact BF16 row，由 `expandedRowIdx` 索引 |
| `preSumBeforeRank` | HCCL window | 每个 global expert 的 source-rank compact-row 基址 |
| signal tail | HCCL window | Front/preSum/data-ready epoch 和 rank-progress slot |
| `gmA` / `gmAScale` | workspace GM | Dispatch 生成的 E4M3/E8M0 GMM1 输入 |
| `gmSwigluA` / `gmSwigluScale` | workspace GM | SwiGLU 生成的 E4M3/E8M0 GMM2 输入 |
| `cumsumMM` | workspace GM | expert-major、按 source-rank 累积的全局 row 前缀，供 Dispatch/GMM/Combine 定位 |
| `sortedRouteSlot` / `expandedRowIdx` | workspace GM | expert-major route 顺序及其 route-slot 到 compact-row 的逆映射 |
| GMM queue/mailbox | workspace GM | runtime task descriptor、ticket、dependency counter 和 completion counter |

`HCCL_BUFFSIZE` 由外部运行环境设置。
布局与容量检查只由 C++ tiling 维护；容量不足会明确报错，大 case 需在外部调大该变量。

## 构建与运行

构建前 source CANN 安装目录中的 `set_env.sh`，由其提供 `ASCEND_HOME_PATH` 和工具链环境。
MPI、CANN 的动态库搜索路径、HCCL 参数及设备可见性均由外部环境配置，设备可见性需与 rank 到设备的映射一致。
`run.sh` 继承这些配置，不自行修改。数据生成需要 Python 3.10 或更高版本及 NumPy。

只编译 A5 mixed-core kernel 和 host：

```bash
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
bash run.sh --build-only
```

请使用连续且可用的 A5 设备运行。环境需要提供 MPICH，MPI 兼容包装不支持 OpenMPI。
将 MPICH 启动程序目录加入 `PATH`，或设置 `MPI_RUNNER`；通过系统动态库搜索配置（例如 `LD_LIBRARY_PATH`）
提供其共享库，或显式设置 `MPI_LIB_PATH`。启动程序和共享库必须来自同一套 MPICH，安装路径由部署环境配置。

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 2048 --k 7168 --n 4096 \
  --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

该命令将 rank 0、1 映射到物理卡 2、3。上面的 `Ascend910_9599` 是 A5 工具链兼容别名，不是 A3 的
`Ascend910B`。

切换其它 `M` 档位时可保持两卡首验配置，例如：

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### 环境变量说明

| 环境变量 | 用途 | 默认行为 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN 安装目录 | 必须在构建前从外部设置 |
| `CMAKE_COMPILER` | CMake 使用的编译器 | `bisheng` |
| `FIRST_DEVICE` | 连续 rank 映射中的第一张物理卡 | 默认 `0`，可由 `--first-device` 覆盖 |
| `MPI_LIB_PATH` | 可选的 MPICH 共享库路径 | 默认通过系统动态库搜索解析 `libmpi.so` 或 `libmpich.so` |
| `MPI_RUNNER` | MPICH 启动命令 | 使用 `PATH` 中的 `mpirun` |
| `HCCL_BUFFSIZE` | HCCL RDMA window 大小 | 由外部运行环境配置 |
| `DISPATCH_MEGA_COMBINE_AICORE_NUM` | 有效 AIC 数 | 默认 `0`，使用 runtime 查询值 |
| `DISPATCH_MEGA_COMBINE_REUSE_DATA` | 复用兼容的已生成 case 数据 | 默认关闭；非零值等价于 `--reuse-data` |
| `DISPATCH_MEGA_COMBINE_WARMUP_ITERS` | 计时前的 warmup 启动次数 | `3` |
| `DISPATCH_MEGA_COMBINE_MEASURE_ITERS` | 用于整体 kernel 统计的计时启动次数 | `5` |

### 整体 Kernel 性能统计

计时启动完成后，rank 0 输出一份 `[KERNEL_PERF]` 整体 kernel 统计。每个 AIC/AIV 只记录整体起止 syscnt，统计以
每轮各 rank 最大耗时为口径，输出平均值、最小值、最大值、标准差、token 吞吐、等效计算 TFLOPS 和等效通信带宽。

## 修改 Case 参数

修改 case 时，MXFP8 维度、rank 拓扑和接收容量都需要满足 Host 校验。

```bash
bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

常用约束：

- `K` 必须是 128 的倍数，并满足 Dispatch packed-row 容量。
- `N` 必须为偶数，且 `N / 2` 必须是 128 的倍数。
- `topK` 必须位于 `1..32`。
- `expertPerRank` 必须是 `4`、`8`、`16` 或 `32`，全部共用一份 runtime kernel，device 容量为
  32 个本地 expert。
- `worldSize` 不能超过所选 28/32/36-AIC 拓扑的 `min(32, aicNum - 2)`。
- `maxOutputSize` 必须覆盖单 rank 接收的 routed rows 上限。
- 每个 RankStreaming 阶段每 worker 最多处理 256 个 token；超出能力的 shape 会被 Host tiling 拒绝。
- synthetic `expert_idx` 默认使用 global token round-robin，使小 M case 也能覆盖全局 expert。

## 常见问题

| 问题 | 原因与解决 |
| --- | --- |
| 缺少 CANN 安装路径 | source CANN 安装目录中的 `set_env.sh`，由其设置 `ASCEND_HOME_PATH` |
| HCCL window too small | 在外部运行环境调大 `HCCL_BUFFSIZE`，使其满足 case 需求 |
| MPI 启动失败 | 检查 `MPI_RUNNER`/`PATH` 及配套 MPICH 共享库配置，启动程序版本应显示 MPICH/HYDRA；不支持 OpenMPI |
| golden 生成很慢 | 首次生成后使用 `--reuse-data` 复用文件；chunk 大小由程序内部固定 |
| shape 或 rank 拓扑被拒绝 | 检查 28/32/36-AIC 选择、rank 上限、MXFP8 对齐、UB 容量和 RankStreaming 每 worker token 上限 |
| 结果 diff 异常 | 先检查 data cache 是否复用旧分布；改变 expert 分布或 case 关键参数后不要使用旧 `out/` |

## 构建系统

- **编译器**：`bisheng`
- **Device kernel flags**：`-xcce --cce-aicore-arch=dav-c310`，并启用 A5 address transform 和显式 DCCI 策略
- **Host executable**：`-xc++ -std=c++17`
- **输出 target**：`dispatch_mega_combine_kernel`、`dispatch_mega_combine`
- **链接库**：`stdc++`、`ascendcl`、`hcomm`、`runtime`、`tiling_api`、`platform`、`nnopbase`、`pthread` 等
- **PTO include**：仓库根目录 `include/` 会被放入 include path，用于 PTO tile/comm helper

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-06-26 | 新增 `dispatch_mega_combine` README，整理 MegaMoE 算子说明、阶段流程、构建运行和 FAQ |
| 2026-07-27 | 将 A3 优化调度迁移到 A5 后端，并完成仅编译验证 |
| 2026-08-14 | 删除开发期观测代码，完成 A5 生产化整理 |
| 2026-08-20 | 同步 runtime expert、deferred metadata、hybrid GMM1、mailbox GMM2、CV 直传、纯 RankStreaming Unpermute 和 Combine progress 有序发布 |
