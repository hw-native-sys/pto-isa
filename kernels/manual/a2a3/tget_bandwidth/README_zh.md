# TGET / TGET_ASYNC 带宽对比示例

## 概览

本示例对比 **TGET**（同步远程读）与 **TGET_ASYNC**（异步 SDMA 远程读）的点对点通信带宽，覆盖 4 KB ~ 4 MB 的传输规模，同时测量 host 侧带宽（GB/s）和 device 侧平均执行 cycle 数。

- **TGET** 通过 UB（Unified Buffer）暂存进行远程读：`Remote GM → UB → Local GM`，受 UB 带宽限制，大传输量时饱和在约 4 GB/s。
- **TGET_ASYNC** 通过 SDMA 引擎直传：`Remote GM → SDMA → Local GM`，绕过 UB 瓶颈，在 4 MB 时可达约 13–14 GB/s。

## 支持的 AI 处理器

- A2/A3

## 目录结构

```
kernels/manual/a2a3/tget_bandwidth/
├── scripts/
│   └── plot_bw_compare.py           # 绘制带宽对比图
├── CMakeLists.txt                   # 构建配置
├── tget_bandwidth_kernel.cpp        # Kernel 实现（AICORE + Host 编排）
├── tget_bandwidth_kernel.h          # Kernel 头文件
├── main.cpp                         # Host 侧入口（MPI 初始化）
├── run.sh                           # 便捷脚本
├── README_zh.md                     # 本文件
└── README.md                        # 英文版
```

## 算子说明

### 数据流

**TGET（同步）**：
```
Peer NPU GM ──TGET──▶ Local UB ──TSTORE──▶ Local GM
```

**TGET_ASYNC（异步）**：
```
Peer NPU GM ──SDMA──▶ Local GM   (直传，无 UB 中转)
```

当前SDMA异步完成协议使用kernel-local post ID。Session会累计记录使用过的Queue数，每次Post都在这些
Queue上追加8B flag SQE，因此等待最后一个Event也会覆盖该Session内此前所有Post。Wait/Test只轮询对应
post ID，不再在Wait阶段提交SQE。每个channel group拥有独立的64槽Payload arena（每槽8B，共512B），
Post Done Record按64B间隔复用该group的`recv_workspace`。
Host 调用方查询容量并提供 workspace，由 `SdmaWorkspaceManager` 初始化后传给 kernel。

### 测试流程

1. 每个 rank 在 HCCL shared memory 中准备发送数据（`PrepareSendBufferKernel`）
2. root rank 对每种传输规模分别执行 TGET 和 TGET_ASYNC
3. Host 侧计时测量带宽，device 侧通过 `SYS_CNT` 测量 cycle 数
4. 验证接收数据正确性

两条计时路径都只测量 Peer 对称内存 `sendShmem` 到 Root 对称内存 `recvShmem` 的传输。计时结束后仅将
接收结果复制一次到普通 GM 校验缓冲区，因此 Host 带宽和 Device cycle 均不包含 CopyOut。

### 规格

| 项目        | 值 |
| ----------- | ----- |
| 数据类型    | `float` |
| NPU 数量    | 2（点对点） |
| 传输规模    | 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 4 MB |
| 测量指标    | host 带宽（GB/s）、device 平均 cycle 数 |

## 实测性能（参考）

以下数据在 Ascend A2/A3 上测得（float 类型，2 卡点对点）。

| 传输大小 | TGET 带宽 (GB/s) | TGET_ASYNC 带宽 (GB/s) | TGET device 平均 cycles | TGET_ASYNC device 平均 cycles |
| -------- | ----------------: | ----------------------: | ----------------------: | ----------------------------: |
| 4 KB     | 0.20              | 0.18                    | 36.28                   | 136.51                        |
| 16 KB    | 0.77              | 0.64                    | 149.98                  | 161.40                        |
| 64 KB    | 1.50              | 2.44                    | 579.62                  | 277.25                        |
| 256 KB   | 2.40              | 7.90                    | 2409.72                 | 745.54                        |
| 1 MB     | 2.63              | 8.98                    | 9317.93                 | 2624.24                       |
| 4 MB     | 4.85              | 10.49                   | 37924.12                | 10273.68                      |

### 分析

- **TGET** 随传输规模增大，带宽逐步上升但在约 **4 GB/s** 处饱和——这是 UB 暂存路径的单核带宽上限。
- **TGET_ASYNC** 在大传输量（≥256 KB）时显著超越 TGET，4 MB 时 Host 观测带宽约为 **10.5 GB/s**。
- 在极小传输量（4 KB）下，TGET_ASYNC 由于 SDMA 启动开销反而略慢于 TGET。

### 带宽对比图

运行绘图脚本生成对比图：

```bash
python3 scripts/plot_bw_compare.py
```

## 构建与运行

### 前置条件

- CANN Toolkit >= 8.5.0（TGET 同步指令）；>= 9.0.0（TGET_ASYNC 异步指令）
- **MPICH**（推荐；Host 侧通过 `comm_mpi.h` 动态加载 `libmpi.so`，communicator handle 与 MPICH 一致）
- 2 张及以上 Ascend NPU

#### MPI 安装（推荐 MPICH）

```bash
# Ubuntu / Debian
sudo apt install mpich libmpich-dev

# 或用户目录安装（无 root）
# 见 tests/README_zh.md「从源码安装 MPICH」
export PATH=$HOME/mpich/bin:$PATH
export MPI_LIB_PATH=$HOME/mpich/lib/libmpi.so
```

`run.sh` 会自动搜索常见 MPICH 路径并设置 `MPI_LIB_PATH`。也可通过 `MPI_SEARCH_DIRS`（空格分隔的 `bin/` 目录列表）覆盖搜索路径。

> **说明**：本示例**不支持 OpenMPI**。`comm_mpi.h` 使用 MPICH 的 `MPI_COMM_WORLD` handle 编码；OpenMPI 的 communicator 表示不同，运行时 `MPI_Bcast` 等调用可能失败。`--allow-run-as-root` 为 OpenMPI 专用参数，MPICH 不支持。

### 步骤

1. 配置 Ascend CANN 环境：

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
# 或 source <workspace>/set_env_new.sh
```

2. 运行示例（默认 2 卡）。`run.sh` 会自动切换到本目录，可从仓库根或本目录执行：

```bash
# 方式 A：先进入示例目录
cd ${git_clone_path}/kernels/manual/a2a3/tget_bandwidth
bash run.sh -r npu -v a3

# 方式 B：在 pto-isa-main 根目录下
bash kernels/manual/a2a3/tget_bandwidth/run.sh -r npu -v a3
```

可通过 `-n` 参数指定 rank 数（默认为 2）：

```bash
bash run.sh -r npu -v a3 -n 2
```

`-v a3` 与 ST 测试脚本一致，内部映射为 CMake 的 `SOC_VERSION=Ascend910B1`（A2/A3 平台）。

成功时输出：

```text
================ TGET/TGET_ASYNC Bandwidth Sweep ================
peer_rank=1 dtype=float tile_elems=1024
[BW] instr=TGET bytes=4096 iters=1000 ...
[BW] instr=TGET_ASYNC bytes=4096 iters=1000 ...
...
test success
```

### 固定版SDMA Device Baseline

使用`device_baseline`模式可配置SQE大小、queue和多Post场景。每轮都会更新源数据、
将目的区域填为哨兵，并验证所有Post的结果。只需设置需要覆盖的参数：

```bash
export TGET_BENCH_MODE=device_baseline
export TGET_DEVICE_BASELINE_BYTES=131072
export TGET_DEVICE_BASELINE_BLOCK_DIVISOR=1
export TGET_DEVICE_BASELINE_QUEUE_NUM=1
export TGET_DEVICE_BASELINE_POST_COUNT=1
export TGET_DEVICE_BASELINE_OUTER_ITERS=20
export TGET_DEVICE_BASELINE_INNER_ITERS=300

bash run.sh -r npu -v a3 -n 2
```

`BLOCK_DIVISOR=1`表示`block_bytes=total_bytes`，即每次Post只生成一个data SQE。计时范围只包含对称内存
传输和Post完成过程，数据校验位于计时区间之外。

约束：

- `QUEUE_NUM <= 48`；
- 单group最多保留64个无需等待的Payload槽；
- 每次Post在单queue产生的数据和完成SQE总数不得超过该SQ深度；
- 默认只Wait最后一个Event，但仍验证所有Post的数据；设置`TGET_DEVICE_BASELINE_WAIT_EACH_EVENT=1`
  可逐个等待Event。

## 变更记录

| 日期       | 变更 |
| ---------- | ------ |
| 2026-07-16 | 固定post ID Event实现；增加queue、channel group和多Post测试 |
| 2026-06-01 | 文档与 `run.sh` 对齐 MPICH；移除 OpenMPI 专用 `mpirun` 参数 |
| 2026-04-02 | 从 ST 测试迁移为独立性能示例 |
