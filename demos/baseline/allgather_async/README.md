# Allgather Async Demo

Demonstrates the allgather collective operation using PTO async instructions across multiple NPU devices.

- **A2/A3 build** (default `SOC_VERSION=Ascend910B1`): Demos 1–3 using SDMA engine (`TPUT_ASYNC` / `TGET_ASYNC` via HCCL)
- **A5 build** (`SOC_VERSION=Ascend950PR_9599`): Demos 4–6 using URMA engine (`HCCP V2 Jetty RDMA`).

The two engine paths use different host infrastructure (`a2a3/common.hpp` vs `a5/common.hpp`) with incompatible ACL/runtime initialization, so each build only compiles and links one set.

## Prerequisites

- CANN Toolkit (version 9.0.0 or above) installed (`ASCEND_HOME_PATH` set via `set_env.sh`)
- CANN Ops package (version 9.0.0 or above) installed
- MPICH installed
- Enough NPU devices for your MPI rank count (default `./run.sh` uses 8 ranks; `./run.sh 2 …` uses 2). Typically one rank maps to one device.

## Quick Start

```bash
source /path/to/set_env.sh
./run.sh                      # 8 ranks, default SoC Ascend910B1 (A2/A3, Demos 1–3)
./run.sh 4                    # 4 ranks
./run.sh 2 Ascend950PR_9599   # 2 ranks, A5 (Demos 4-6)
```

## What It Does

Each rank contributes 256 `int32_t` values. After allgather, every rank holds all ranks' data.

### SDMA Demos (A2/A3 build)

1. **TPUT_ASYNC Allgather (multi-core)**: Launched with `<<<nRanks, ...>>>` — each AICORE handles one target rank's communication in parallel. The AICORE where `block_idx == myRank` performs a local copy; all others use `pto::comm::TPUT_ASYNC` to write data to the corresponding remote rank.
2. **TGET_ASYNC Allgather (multi-core)**: Launched with `<<<nRanks, ...>>>` — each AICORE pulls data from one source rank in parallel. The AICORE where `block_idx == myRank` performs a local copy; all others use `pto::comm::TGET_ASYNC` to read data from the corresponding remote rank.
3. **Ring TPUT_ASYNC Allgather**: Ring algorithm with N-1 rounds for N ranks. In round 0, each rank copies its `sendBuf` locally and pushes it to the next rank via `TPUT_ASYNC`. In subsequent rounds, each rank forwards the chunk it received in the previous round to the next rank. Each round is a separate kernel launch with a host-side barrier in between.

### URMA Demos (A5 build)

4. **URMA TPUT_ASYNC Allgather (multi-core)**: Same algorithm as Demo 1, using `TPUT_ASYNC<DmaEngine::URMA>` with `UrmaPeerMrBaseAddr` for remote addressing.
5. **URMA TGET_ASYNC Allgather (multi-core)**: Same algorithm as Demo 2, using `TGET_ASYNC<DmaEngine::URMA>`.
6. **URMA Ring TPUT_ASYNC Allgather**: Same ring algorithm as Demo 3, using `TPUT_ASYNC<DmaEngine::URMA>`. Runs N-1 rounds; on 2 ranks this is a single round verifying basic AllGather correctness. The recv→forward path is naturally exercised when N≥3.

### Key PTO APIs

**Demos 1–3 (SDMA / HCCL)**

- `pto::comm::AsyncSession`, `BuildAsyncSession` (SDMA overload, used with `SdmaWorkspaceManager` and HCCL context)
- `pto::comm::TPUT_ASYNC`, `TGET_ASYNC` (default SDMA engine)
- `pto::comm::AsyncEvent`, `Wait`
- `SdmaWorkspaceManager`, `HcclRemotePtr` (host)

**Demos 4–6 (URMA)**

- `pto::comm::BuildAsyncSession<DmaEngine::URMA>`, `TPUT_ASYNC` / `TGET_ASYNC<DmaEngine::URMA>`
- `UrmaWorkspaceManager`, `UrmaPeerMrBaseAddr` (host)

## Project Structure

```
allgather_async/
├── CMakeLists.txt                       -- Build configuration (bisheng + CCE)
├── csrc/
│   ├── kernel/
│   │   ├── allgather_kernel.cpp         -- SDMA kernels + host launchers (A2/A3)
│   │   ├── allgather_kernel.h           -- SDMA host-side function declarations
│   │   ├── allgather_urma_kernel.cpp    -- URMA kernels + host launchers (A5)
│   │   └── allgather_urma_kernel.h      -- URMA host-side function declarations
│   └── host/
│       └── main.cpp                     -- Entry point (MPI init, run demos, report)
├── run.sh                               -- One-click build and run
├── README.md                            -- English documentation
└── README_zh.md                         -- Chinese documentation
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
# A2/A3 build (Demos 1-3)
mkdir -p build && cd build
cmake .. -DSOC_VERSION=Ascend910B1
make -j$(nproc)
cd ..
mpirun -n 8 ./build/bin/allgather_demo

# A5 build (Demos 4-6)
rm -rf build
mkdir -p build && cd build
cmake .. -DSOC_VERSION=Ascend950PR_9599
make -j$(nproc)
cd ..
mpirun -n 2 ./build/bin/allgather_demo
```

`SOC_VERSION` determines which kernel set is compiled: A2/A3 builds only the SDMA kernel; A5 builds only the URMA kernel. A clean rebuild (`rm -rf build`) is needed when switching between SoC targets.

## Expected Output

### A5 (2 ranks, URMA Demos 4–6)

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

### A3 (8 ranks, SDMA Demos 1–3)

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
