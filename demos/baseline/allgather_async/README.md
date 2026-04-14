# Allgather Async Demo

Demonstrates the allgather collective operation using PTO's `TPUT_ASYNC` (remote write) and `TGET_ASYNC` (remote read) SDMA-based async instructions across multiple NPU devices in multi-core mode.

## Prerequisites

- CANN Toolkit (version 9.0.0 or above) installed (`ASCEND_HOME_PATH` set via `set_env.sh`)
- CANN Ops package (version 9.0.0 or above) installed
- MPICH installed
- At least 2 NPU devices on the machine

## Quick Start

```bash
source /path/to/set_env.sh
./run.sh                      # 8 ranks, default SoC
./run.sh 4                    # 4 ranks
./run.sh 2 Ascend910_9599     # 2 ranks, A5 devices
```

## What It Does

Each rank contributes 256 `int32_t` values. After allgather, every rank holds all ranks' data.

1. **TPUT_ASYNC Allgather (multi-core)**: Launched with `<<<nRanks, ...>>>` — each AICORE handles one target rank's communication in parallel. The AICORE where `block_idx == myRank` performs a local copy; all others use `pto::comm::TPUT_ASYNC` to write data to the corresponding remote rank.
2. **TGET_ASYNC Allgather (multi-core)**: Launched with `<<<nRanks, ...>>>` — each AICORE pulls data from one source rank in parallel. The AICORE where `block_idx == myRank` performs a local copy; all others use `pto::comm::TGET_ASYNC` to read data from the corresponding remote rank.

## Project Structure

```
allgather_async/
  CMakeLists.txt                       -- Build configuration (bisheng + CCE)
  csrc/kernel/allgather_kernel.cpp     -- AICORE kernels + host-side launchers
  csrc/kernel/allgather_kernel.h       -- Host-side function declarations
  csrc/host/main.cpp                   -- Entry point (MPI init, run demos, report)
  run.sh                               -- One-click build and run
```

## Dependency Installation

### 1. CANN Toolkit

CANN Toolkit version 9.0.0 or above. Available via two methods:

- **Option 1**: Download from the [Ascend Community](https://www.hiascend.com/software/cann/community)
- **Option 2**: Direct download (preview build): [x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260305000326487/x86_64/Ascend-cann-toolkit_9.0.0_linux-x86_64.run) / [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260305000326487/aarch64/Ascend-cann-toolkit_9.0.0_linux-aarch64.run)

For installation instructions, refer to [Quick Install CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit).

After installation, set up the environment (default install path):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Custom install path:

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 2. CANN Ops

CANN Ops package (version 9.0.0 or above). Download the ops-legacy package for your hardware platform:

| Hardware | x86_64 | aarch64 |
| --- | --- | --- |
| A2 | [Download](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-x86_64.run) | [Download](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-aarch64.run) |
| A3 | [Download](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-x86_64.run) | [Download](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-aarch64.run) |

Installation follows the same procedure as the Toolkit. Refer to [Quick Install CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit).

### 3. MPICH

Recommended version >= 3.2.1. Build and install from source:

```bash
# Example with version 3.2.1
version='3.2.1'
wget https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz
tar -xzf mpich-${version}.tar.gz
cd mpich-${version}
./configure --prefix=/usr/local/mpich --disable-fortran
make && make install
```

Set environment variables:

```bash
export MPI_HOME=/usr/local/mpich
export PATH=${MPI_HOME}/bin:${PATH}
```

Verify that `mpirun` is available:

```bash
mpirun --version
```

## Manual Build

```bash
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b1
make -j$(nproc)
cd ..
mpirun -n 2 ./build/bin/allgather_demo
```
