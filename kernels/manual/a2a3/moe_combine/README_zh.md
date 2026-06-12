# moe_combine - A2/A3 PTO MoE Combine Kernel

## 概览

本示例在 Ascend A2/A3 系列芯片上使用 PTO 实现 MoE combine 阶段。它对应
dispatch-compute-combine 流水中的 return 半段：本地 expert 完成计算后，combine kernel 将 expert
输出行按路由账本返还给原 token 所在 rank，并使用 gate 权重还原每个 token 的最终输出。

当前 kernel 是一个独立 combine kernel，入口消费的是显式低层路由账本 `routeMeta`。
`expert_ids`、`assist_info_for_combine`、`ep_send_counts` 等路由信息由上游或 host 侧按本算子的账本布局整理后，
显式写入 `routeMeta` 传入。

```text
expertOutput[local expert rows, K]
  -> 通过 HCCL peerWindow.ptrD 做变长 return
  -> 通过 TNOTIFY/TWAIT 做跨 rank 完成同步
  -> 加权还原: outputC[token, :] = sum(topK probs * returned rows)
```

## 支持的 AI 处理器

- A2/A3，已在 Atlas 910B1 验证

## 目录结构

```text
kernels/manual/a2a3/moe_combine/
├── CMakeLists.txt           # Bisheng CCE + host 构建配置
├── run.sh                   # 一键构建和运行脚本，发现 MPI，估算 HCCL_BUFFSIZE
├── common.h                 # 共享 ABI: shape, routeMeta layout, peerWindow layout, HCCL context
├── layout.h                 # Host 侧 layout 计算和 HCCL_BUFFSIZE 估算
├── kernel_launchers.h       # Host 侧 kernel launcher 声明
├── moe_combine_kernel.cpp   # PTO AIV kernel: return + wait + weighted restore
├── main.cpp                 # Host 编排: MPI, ACL, HCCL window, fixture, verify, profile
├── golden.h                 # CPU golden 数据结构和公开接口声明
├── golden.cpp               # CPU golden 路由构造和输出校验实现
├── hccl_context.h           # HCCL window 初始化和 peer-window 地址交换
├── comm_mpi.h               # MPI 动态加载封装
├── DESIGN.md                # 从 dispatch 拆分的设计说明
├── PHASE2_DESIGN.md         # PTO 风格重构计划
├── IMPLEMENTATION_PLAN.md   # 历史实现追踪
├── README.md                # 英文 README
└── README_zh.md             # 中文 README
```

## 算子说明

### 计算功能

对每个 rank，本算子消费已经按本地 expert 和来源 rank 排布好的 expert 输出。kernel 内部流程是：

1. 读取 `routeMeta`，得到每个 source rank 给每个 expert 的行数，以及这些行在 `expertOutput` 中的位置。
2. 使用 PTO `TPUT` 将每行 expert 输出返还到 token owner rank 的 HCCL peer window。
3. 使用 `TNOTIFY` / `TWAIT` 等待所有 peer 完成 return 写入。
4. 读取 `routeMeta.expandedRowIdx` 和 `probs`，还原 `outputC[M, K]`。

对本 rank 的第 `t` 个 token，dispatch 阶段会产生 `topK` 条 expert route。combine return 完成后，这些 route
对应的 expert 输出行已经写回本 rank 的 `peerWindow.ptrD`。`expandedRowIdx[t * topK + slot]` 记录第 `slot`
条 route 在 `ptrD` 中的行号，`probs[t * topK + slot]` 是这条 route 的 gate 权重。

因此对输出的每一列 `c`，还原逻辑是：

```text
outputC[t, c] = 0
for slot in 0..topK-1:
    row = expandedRowIdx[t * topK + slot]
    if row >= 0:
        outputC[t, c] += probs[t * topK + slot] * peerWindow.ptrD[row, c]
```

也就是把同一个 token 的 `topK` 路 expert 输出按 gate 权重加权求和，得到最终的 `outputC[t, :]`。

### 覆盖范围

| 包含 | 不包含 |
| --- | --- |
| EP 域内基于 HCCL window 的 combine return | Dispatch pack/gather kernel |
| 使用 `TPUT` 实现变长 all-to-all-like return | HCCL collective `AllToAllV` API |
| 使用 `probs` 做加权还原 | Expert FFN/GMM 计算 |
| 显式低层 `routeMeta` 契约 | 量化、TP ReduceScatterV、shared/copy/const expert |
| A2/A3 HCCL peer-window 路径 | 上层公共 ABI 适配层 |

## 入口契约

### Kernel Launcher ABI

```cpp
void LaunchMoeCombineKernel(MoeCombineShape shape, uint32_t myRank,
                            uint8_t *expertOutput,
                            uint8_t *probs,
                            uint8_t *outputC,
                            uint8_t *routeMeta,
                            uint8_t *peerWindow,
                            uint8_t *hcclCtx,
                            uint8_t *workspace,
                            void *stream,
                            uint32_t launchBlockCount);
```

### 运行时输入

| 参数 | 方向 | 存储 | 含义 |
| --- | --- | --- | --- |
| `shape` | 输入 | 值传递 | 静态 shape 和 AIV block 数，如 `ep`, `m`, `k`, `topK`, `expertPerRank`, `aivBlocks` |
| `myRank` | 输入 | 值传递 | EP 域内 rank id |
| `expertOutput` | 输入 | `aclrtMalloc` GM | 本地 expert 输出行，形状 `[maxOutputSize, K]`，fp16 |
| `probs` | 输入 | `aclrtMalloc` GM | gate 权重，形状 `[M, topK]`，fp32 |
| `outputC` | 输出 | `aclrtMalloc` GM | 还原后的 token 输出，形状 `[M, K]`，fp16 |
| `routeMeta` | 输入 | `aclrtMalloc` GM | 显式 combine 路由账本 |
| `peerWindow` | 输入/输出 | HCCL RDMA window | 远端可见的 `ptrD` return buffer 和 signal |
| `hcclCtx` | 输入 | `aclrtMalloc` GM | 设备侧所有 rank 的 HCCL window 地址 |
| `workspace` | 临时 | `aclrtMalloc` GM | 本地 AIV soft sync 区 |
| `stream` | 输入 | ACL stream | kernel launch stream |
| `launchBlockCount` | 输入 | 值传递 | kernel 使用的 AIV block 数 |

### `peerWindow` 内容

`localWindowBase` 是 HCCL window 的起始地址。A2/A3 上传给 kernel 的 `peerWindow` 指向
`localWindowBase` 处的 live payload；本 layout 不额外保留 A5 的 4096B head guard。

```text
A2/A3 localWindowBase
  peerWindow live payload:
    ptrD
    countReadySignal[ep]
    combineDoneSignal[ep]
```

| 字段 | 位置 | 内容 |
| --- | --- | --- |
| `ptrD` | HCCL window live payload | return 目标行，被远端 `TPUT` 写入 |
| `countReadySignal[ep]` | HCCL window live payload | per-rank ready 计数区 |
| `combineDoneSignal[ep]` | HCCL window live payload | per-rank 完成计数器；远端 rank 完成写入本 rank `ptrD` 后 `TNOTIFY` 对应槽位 |

### `MoeCombineShape`

| 字段 | 含义 |
| --- | --- |
| `ep` | EP rank 数 |
| `m` | 每 rank token 数 |
| `k` | hidden size |
| `topK` | 每 token 的 expert 路由数 |
| `expertPerRank` | 每 rank 本地 expert 数 |
| `expertNum` | 全局 expert 数，通常为 `ep * expertPerRank` |
| `maxOutputSize` | 每 rank expert 输出最大行容量 |
| `aivBlocks` | 逻辑 AIV block 数；A3 默认 `24`，可传参覆盖 |

### `routeMeta` 布局

`routeMeta` 是显式低层 combine 路由账本。它是本地 GM，不属于 HCCL window。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `peerTokenPerExpert` | `[ep, expertNumPadded]` int32 | 每个 source rank 到每个 global expert 的行数 |
| `expandedRowIdx` | `[M * topK]` int32 | token route 到 `peerWindow.ptrD` 的行映射；`-1` 表示无效 route |
| `cumsumPerExpert` | `[ep, expertNumPadded]` int32 | 每个 source rank 内按 global expert 的 inclusive prefix：`cumsum[src,e] = sum(peerTokenPerExpert[src,0..e])` |
| `dispatchOffset` | `[expertPerRank]` int32 | 每个本地 expert 在 `expertOutput` 中的基地址行 |
| `prevSumBeforeRank` | `[ep, expertPerRank]` int32 | 某 source rank 在本地 expert 行段中的前缀偏移 |

## 优化说明

该 kernel 是 AIV-only combine kernel。对于 `K=7168` 这类 hidden size，一行 fp16 数据是 14 KiB，整体主要受
GM/HCCL window 搬运带宽影响。优化目标是让数据搬运尽量流式化，同时降低控制面元数据开销。

- **显式 routeMeta**：路由元数据作为独立 GM buffer 传入。`peerWindow` 只保留远端可见 return 数据和信号，
  `workspace` 只保留本地 AIV soft sync 区。
- **chunk 化 return 分片**：return 阶段遍历 `src_rank x local_expert` segment，并按
  `chunkBase % blockNum` 把行 chunk 分给 AIV block。
- **PTO `TPUT` ping/pong 路径**：远端 return 使用 `TPUT(remoteDst, localSrc, ping, pong)`，通过 UB 双缓冲
  让 MTE2 load 和 MTE3 store 形成流水。
- **Restore route cache**：当 `topK <= 16` 时，每个 token 的 route row 和 prob 会缓存到标量数组，减少内层
  restore loop 对 route metadata 的重复读取。
- **DCCI 批量 acquire**：每个 token 在消费返回的 `ptrD` 行前先刷新对应 GM range，然后对本轮 cached routes
  做一次 `dsb(DSB_DDR)`。
- **TLOAD 到 TAXPY event chain**：restore 阶段通过 PTO `TLOAD -> TAXPY` event 依赖完成加载和计算衔接。
- **Soft AIV sync**：同一个 kernel 内用 `SoftSyncAiv` 分隔 return、wait、restore 阶段。

## Tiling 与默认参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `PES` / `ep` | `2` | EP rank 数 |
| `M` | `64` | 每 rank token 数 |
| `K` | `7168` | hidden size |
| `topK` | `8` | 每 token expert 路由数 |
| `expertPerPe` | `2` | 每 rank 本地 expert 数 |
| `expertNum` | `4` | `PES * expertPerPe` |
| `maxOutputSize` | `PES * M * topK` | 默认容量；默认 shape 下为 `1024` |
| `aivBlocks` | `24` | A3 默认逻辑 AIV block 数 |
| 内部 Vector tile 列宽 | `1024` | 示例实现固定值 |
| 内部 return chunk | `8 rows` | 固定的 return 阶段行 chunk |
| 内部 metadata pad | `16` | expert metadata 对齐粒度 |

使用 `PES=2, M=64, K=7168, topK=8, expertPerPe=2, aivBlocks=24` 时，各布局大小为：

| Layout | 字节数 |
| --- | --- |
| `workspace` | `2304` |
| `routeMeta` | `2432` |
| `peerWindow` | `7340160` |

## 整体架构

```text
Host:
  ParseArgs -> ComputeWorkspaceLayout / ComputeCombineRouteMetaLayout / ComputePeerWindowLayout
    -> PrepareHostData and CPU golden
    -> Init HCCL peer window
    -> AllocateLocalBuffers(routeMeta/workspace/expertOutput/probs/outputC)
    -> loop(warmup + measured):
         ClearDeviceState
         PrepareCombineFixture -> 写入 routeMeta + expertOutput
         LaunchMoeCombineKernel
         Verify outputC

Device:
  ReturnExpertRowsToOwners -> WaitCombinePhase -> RestoreOutputRows
```

```text
Return phase:
  routeMeta(peerToken/cumsum/offset) + expertOutput
    -> local or remote peerWindow.ptrD
    -> TNOTIFY peer combineDoneSignal[myRank]

Restore phase:
  routeMeta.expandedRowIdx + probs + peerWindow.ptrD
    -> outputC
```

## Kernel 细节

### 阶段 1: ReturnExpertRowsToOwners

kernel 遍历所有本地 expert segment：

```text
segment = src_rank * expertPerRank + localExpert
globalExpert = myRank * expertPerRank + localExpert
rows = routeMeta.peerTokenPerExpert[src_rank, globalExpert]
```

对每个非空 segment：

1. `srcStart` 由 `dispatchOffset[localExpert] + prevSumBeforeRank[src_rank, localExpert]` 计算。
2. `dstStart` 由 `cumsumPerExpert[src_rank, globalExpert - 1]` 计算；`globalExpert == 0` 时为 `0`。
3. 如果 `src_rank == myRank`，行被本地复制到本 rank 的 `peerWindow.ptrD`。
4. 否则，PTO `TPUT` 把行 chunk 写入 source rank 的远端 peer window。

### 阶段 2: WaitCombinePhase

return 写完后，每个 rank 通知所有 token-owner rank：

```text
TNOTIFY(remotePeer.combineDoneSignal[myRank], AtomicAdd)
TWAIT(localPeer.combineDoneSignal[peer] >= 1)
```

Host 会在每轮迭代前清零 `combineDoneSignal`，因此 kernel 固定等待每个 peer 的一次 notify。

### 阶段 3: RestoreOutputRows

每个 AIV block 负责一段连续 token。对每个 token 和每个列 tile：

1. 使用 `TEXPANDS` 把输出 tile 清零。
2. 对每个有效 route，加载 `ptrD[expandedRowIdx]`。
3. 使用 `TAXPY(outTile, ptrTile, prob)` 累加。
4. 将 fp16 tile 写回 `outputC`。

## 实测性能

本工程可直接在 A2/A3 机器（Atlas 910B1）上运行。脚本会输出如下 profile：

```text
[PROFILE] CombineTile
  M=64 K=7168 ranks=2 topK=8 expertPerPe=2 warmup=3 measured=5 samples=5
  prepare_fixture: avg=... us max=... us
  combine_e2e: avg=... us max=... us
  verify=PASS
```

关键指标含义：

| 指标 | 含义 |
| --- | --- |
| `combine_e2e` | combine kernel launch 到 stream sync；不包含 clear、fixture、verify，也不包含 kernel launch 窗口之外的 MPI barrier |
| `verify=PASS` | device `outputC` 与 CPU golden 一致 |

## 构建与运行

### 环境

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
```

执行 `run.sh` 前需要先在 shell 中加载 CANN 环境。如果 shell 中没有 `mpirun`，请先配置 MPI 环境。

### 仅编译

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh --skip-build 0 --clean-build 1
```

### 快速验证

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --aiv-blocks 2
```

### 默认 shape

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```

### 主要命令行参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-pes` | `2` | rank 数 |
| `-M` | `64` | 每 rank token 数 |
| `-K` | `7168` | hidden size |
| `-topK` | `8` | 每 token route 数 |
| `-expertPerPe` | `2` | 每 rank expert 数 |
| `--max-output-size` | `PES * M * topK` | expert output 行容量 |
| `--aiv-blocks` | `0 -> 24` | 逻辑 AIV block 数，用于匹配不同硬件资源规划 |
| `--device-base` | `0` | rank 到 device 映射使用的起始 device id |
| `--ndevices` | `PES` | 示例 launcher 使用的可见 device 数 |

## 验证

Host 会构造确定性的 CPU golden 路由账本，将其写入 `routeMeta`，拷贝 `expertOutput`，启动 kernel，
并将 `outputC` 与 CPU golden 输出对比。默认开启验证。

预期成功输出：

```text
verify=PASS
```
