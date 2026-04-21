# Allgather 异步通信 Demo

使用 PTO 异步指令在多个 NPU 设备之间实现 allgather 集合通信操作。

- **A2/A3 构建**（默认 `SOC_VERSION=Ascend910B1`）：Demo 1–3，使用 SDMA 引擎（`TPUT_ASYNC` / `TGET_ASYNC`，通过 HCCL）
- **A5 构建**（`SOC_VERSION=Ascend950PR_9599`）：Demo 4–6，使用 URMA 引擎（`HCCP V2 Jetty RDMA`）。

两套引擎路径分别使用 `a2a3/common.hpp` 和 `a5/common.hpp`，ACL/运行时初始化方式不兼容，因此每次构建只编译和链接其中一套。

## 前置条件

- 已安装 CANN Toolkit（9.0.0 及以上版本），并通过 `set_env.sh` 设置 `ASCEND_HOME_PATH`
- 已安装 CANN Ops 包（9.0.0 及以上版本）
- 已安装 MPICH
- MPI rank 数量不超过可用 NPU 设备数（默认 `./run.sh` 为 8 ranks；`./run.sh 2 …` 为 2 ranks）。通常一 rank 对应一设备。

## 快速开始

```bash
source /path/to/set_env.sh
./run.sh                      # 8 ranks，默认 SoC Ascend910B1（A2/A3，Demo 1–3）
./run.sh 4                    # 4 ranks
./run.sh 2 Ascend950PR_9599   # 2 ranks，A5（Demo 4-6）
```

## 功能说明

每个 rank 贡献 256 个 `int32_t` 数据。allgather 操作完成后，每个 rank 都持有所有 rank 的完整数据。

### SDMA Demo（A2/A3 构建）

1. **TPUT_ASYNC Allgather（异步远程写，多核）**：以 `<<<nRanks, ...>>>` 启动，每个 AICORE 负责一个目标 rank 的通信，并行执行。其中 `block_idx == myRank` 的 AICORE 执行本地拷贝，其余 AICORE 通过 `pto::comm::TPUT_ASYNC` 将数据异步写入对应远端 rank。
2. **TGET_ASYNC Allgather（异步远程读，多核）**：以 `<<<nRanks, ...>>>` 启动，每个 AICORE 负责从一个源 rank 拉取数据，并行执行。其中 `block_idx == myRank` 的 AICORE 执行本地拷贝，其余 AICORE 通过 `pto::comm::TGET_ASYNC` 从对应远端 rank 异步读取数据。
3. **Ring TPUT_ASYNC Allgather（环形异步远程写）**：环形算法，N 个 rank 执行 N-1 轮。第 0 轮，每个 rank 将 `sendBuf` 本地拷贝到 `recvBuf[myRank]`，同时通过 `TPUT_ASYNC` 推送到下一个 rank。后续每轮，rank 将上一轮收到的数据块继续转发给下一个 rank。每轮通过独立的 kernel 启动，轮间由 Host 侧 barrier 同步。

### URMA Demo（A5 构建）

4. **URMA TPUT_ASYNC Allgather（多核）**：与 Demo 1 相同的多核算法，使用 `TPUT_ASYNC<DmaEngine::URMA>` + `UrmaPeerMrBaseAddr` 进行远端寻址。
5. **URMA TGET_ASYNC Allgather（多核）**：与 Demo 2 相同的多核算法，使用 `TGET_ASYNC<DmaEngine::URMA>`。
6. **URMA Ring TPUT_ASYNC Allgather**：与 Demo 3 相同的环形算法，使用 `TPUT_ASYNC<DmaEngine::URMA>`。执行 N-1 轮；2 卡时为 1 轮，验证基本 AllGather 正确性。recv→forward 路径在 N≥3 时自然被覆盖。

### 关键 PTO API

**Demos 1–3（SDMA / HCCL）**

- `pto::comm::AsyncSession`、`BuildAsyncSession`（与 `SdmaWorkspaceManager`、HCCL 上下文配合的 SDMA 重载）
- `pto::comm::TPUT_ASYNC`、`TGET_ASYNC`（默认 SDMA 引擎）
- `pto::comm::AsyncEvent`、`Wait`
- `SdmaWorkspaceManager`、`HcclRemotePtr`（Host）

**Demos 4–6（URMA）**

- `pto::comm::BuildAsyncSession<DmaEngine::URMA>`、`TPUT_ASYNC` / `TGET_ASYNC<DmaEngine::URMA>`
- `UrmaWorkspaceManager`、`UrmaPeerMrBaseAddr`（Host）

## 目录结构

```
allgather_async/
├── CMakeLists.txt                       # 构建配置（bisheng 编译器 + CCE）
├── csrc/
│   ├── kernel/
│   │   ├── allgather_kernel.cpp         # SDMA kernel 实现 + Host 启动函数（A2/A3）
│   │   ├── allgather_kernel.h           # SDMA Host 侧函数声明
│   │   ├── allgather_urma_kernel.cpp    # URMA kernel + Host launcher（A5）
│   │   └── allgather_urma_kernel.h      # URMA Host 侧函数声明
│   └── host/
│       └── main.cpp                     # 入口（MPI 初始化、运行 Demo、输出结果）
├── run.sh                               # 一键构建并运行
├── README.md                            # 英文说明
└── README_zh.md                         # 本文档
```

## 依赖安装

### 1. CANN Toolkit

CANN Toolkit 9.0.0 及以上版本，可通过以下两种方式获取：

- **方式一**：从[昇腾社区](https://www.hiascend.com/software/cann/community)下载
- **方式二**：直接下载尝鲜版安装包：[x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260305000326487/x86_64/Ascend-cann-toolkit_9.0.0_linux-x86_64.run) / [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260305000326487/aarch64/Ascend-cann-toolkit_9.0.0_linux-aarch64.run)

安装方式参考[快速安装 CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)。

安装完成后配置环境变量（默认安装路径）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

自定义安装路径：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 2. CANN Ops

CANN Ops 包（9.0.0 及以上版本），按硬件平台选择对应的 ops-legacy 包下载安装：

| 硬件平台 | x86_64 | aarch64 |
| --- | --- | --- |
| A2 | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-x86_64.run) | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-aarch64.run) |
| A3 | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-x86_64.run) | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-aarch64.run) |

安装方式与 Toolkit 相同，参考[快速安装 CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)。

### 3. MPICH

推荐版本 >= 3.2.1，通过源码编译安装：

```bash
# 以 3.2.1 版本为例
version='3.2.1'
wget https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz
tar -xzf mpich-${version}.tar.gz
cd mpich-${version}
./configure --prefix=/usr/local/mpich --disable-fortran
make && make install
```

设置环境变量：

```bash
export MPI_HOME=/usr/local/mpich
export PATH=${MPI_HOME}/bin:${PATH}
```

安装后确认 `mpirun` 可用：

```bash
mpirun --version
```

## 手动构建与运行

```bash
# A2/A3 构建（Demo 1-3）
mkdir -p build && cd build
cmake .. -DSOC_VERSION=Ascend910B1
make -j$(nproc)
cd ..
mpirun -n 8 ./build/bin/allgather_demo

# A5 构建（Demo 4-6）
rm -rf build
mkdir -p build && cd build
cmake .. -DSOC_VERSION=Ascend950PR_9599
make -j$(nproc)
cd ..
mpirun -n 2 ./build/bin/allgather_demo
```

`SOC_VERSION` 决定编译哪套 kernel：A2/A3 只编译 SDMA kernel，A5 只编译 URMA kernel。切换 SoC 目标时需要清除构建目录（`rm -rf build`）。

## 预期输出

### A5（2 ranks，URMA Demos 4–6）

```
========================================
 PTO Allgather Async Demo
 Ranks: 2
========================================

--- Demo 4: URMA Multi-core TPUT_ASYNC ---
[URMA_TPUT_MC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[URMA_TPUT_MC PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

--- Demo 5: URMA Multi-core TGET_ASYNC ---
[URMA_TGET_MC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[URMA_TGET_MC PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

--- Demo 6: URMA Ring TPUT_ASYNC ---
[URMA_RING_TPUT PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[URMA_RING_TPUT PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

========================================
 All demos PASSED
========================================
```

### A3（8 ranks，SDMA Demos 1–3）

```
========================================
 PTO Allgather Async Demo
 Ranks: 8
========================================

--- Demo 1: Multi-core TPUT_ASYNC ---
[TPUT_ASYNC_MC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...] slot[2]=[2000,2001,2002,...] ...
...

--- Demo 2: Multi-core TGET_ASYNC ---
[TGET_ASYNC_MC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...] slot[2]=[2000,2001,2002,...] ...
...

--- Demo 3: Ring TPUT_ASYNC ---
[RING_TPUT_ASYNC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...] slot[2]=[2000,2001,2002,...] ...
...

========================================
 All demos PASSED
========================================
```
