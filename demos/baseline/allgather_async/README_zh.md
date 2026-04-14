# Allgather 异步通信 Demo

本示例展示如何使用 PTO 的 `TPUT_ASYNC`（异步远程写）和 `TGET_ASYNC`（异步远程读）SDMA 指令在多个 NPU 设备之间实现多核 allgather 集合通信操作。

## 前置条件

- 已安装 CANN Toolkit（9.0.0 及以上版本），并通过 `set_env.sh` 设置 `ASCEND_HOME_PATH`
- 已安装 CANN Ops 包（9.0.0 及以上版本）
- 已安装 MPICH
- 机器上至少有 2 个 NPU 设备

## 快速开始

```bash
source /path/to/set_env.sh
./run.sh                      # 8 ranks，默认 SoC
./run.sh 4                    # 4 ranks
./run.sh 2 Ascend910_9599     # 2 ranks，A5 设备
```

## 功能说明

每个 rank 贡献 256 个 `int32_t` 数据。allgather 操作完成后，每个 rank 都持有所有 rank 的完整数据。

1. **TPUT_ASYNC Allgather（异步远程写，多核）**：以 `<<<nRanks, ...>>>` 启动，每个 AICORE 负责一个目标 rank 的通信，并行执行。其中 `block_idx == myRank` 的 AICORE 执行本地拷贝，其余 AICORE 通过 `pto::comm::TPUT_ASYNC` 将数据异步写入对应远端 rank。
2. **TGET_ASYNC Allgather（异步远程读，多核）**：以 `<<<nRanks, ...>>>` 启动，每个 AICORE 负责从一个源 rank 拉取数据，并行执行。其中 `block_idx == myRank` 的 AICORE 执行本地拷贝，其余 AICORE 通过 `pto::comm::TGET_ASYNC` 从对应远端 rank 异步读取数据。

### 关键 PTO API

- `pto::comm::AsyncSession` / `BuildAsyncSession`：SDMA 异步会话初始化
- `pto::comm::TPUT_ASYNC`：异步远程写（通过 SDMA 引擎）
- `pto::comm::TGET_ASYNC`：异步远程读（通过 SDMA 引擎）
- `pto::comm::AsyncEvent` / `Wait`：异步事件等待与同步
- `SdmaWorkspaceManager`：Host 侧 SDMA 工作空间管理
- `HcclRemotePtr`：将本地共享内存地址转换为远端可访问地址

## 目录结构

```
allgather_async/
├── CMakeLists.txt                       # 构建配置（bisheng 编译器 + CCE）
├── csrc/
│   ├── kernel/
│   │   ├── allgather_kernel.cpp         # AICORE kernel 实现 + Host 侧启动函数
│   │   └── allgather_kernel.h           # Host 侧函数声明
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


| 硬件平台 | x86_64                                                                                                                      | aarch64                                                                                                                      |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| A2   | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-x86_64.run) | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-aarch64.run) |
| A3   | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-x86_64.run)   | [下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-aarch64.run)   |


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
# 构建
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b1
make -j$(nproc)
cd ..

# 运行（N 个 MPI 进程，对应 N 个 NPU 设备）
mpirun -n 2 ./build/bin/allgather_demo
```

## 预期输出（2 ranks）

```
========================================
 PTO Allgather Async Demo
 Ranks: 2
========================================

--- Demo 1: Multi-core TPUT_ASYNC ---
[TPUT_ASYNC_MC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[TPUT_ASYNC_MC PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

--- Demo 2: Multi-core TGET_ASYNC ---
[TGET_ASYNC_MC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[TGET_ASYNC_MC PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

========================================
 All demos PASSED
========================================
```
