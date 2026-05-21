# tests/

PTO Tile Lib 的测试与示例，覆盖 CPU 仿真与 NPU（`sim` 和板上 `npu` 两种模式）。

## 测试入口

常见测试入口如下：

- CPU Simulator 全量运行：`python3 tests/run_cpu.py --clean --verbose`
- GEMM demo：`python3 tests/run_cpu.py --demo gemm --verbose`
- Flash Attention demo：`python3 tests/run_cpu.py --demo flash_attn --verbose`
- 单个 ST 用例：`python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]`
- 一键脚本：`./tests/run_st.sh`、`./tests/run_cpu_tests.sh`

## 目录结构

- `script/`：推荐的入口脚本
  - `run_st.py`：构建并运行 NPU ST
  - `build_st.py`：仅构建 NPU ST
  - `all_cpu_tests.py`：批量构建并运行 CPU ST 套件
  - `cpu_bfloat16.py`：CPU bfloat16 测试脚本
  - `README.md`：脚本使用说明
- `cpu/`：CPU 侧 ST 测试（gtest + CMake）
  - `st/`：CPU 计算 ST 工程与 testcase 数据生成脚本
  - `comm/st/`：CPU 通信 ST
- `npu/`：按 SoC 拆分的 NPU 侧 ST 测试
  - `a2a3/src/st/`：A2/A3 计算 ST
  - `a2a3/src/common/`：A2/A3 共享测试资源
  - `a2a3/comm/st/`：A2/A3 通信 ST
  - `a5/src/st/`：A5 计算 ST
  - `a5/src/common/`：A5 共享测试资源
  - `a5/comm/st/`：A5 通信 ST
  - `kirin9030/src/st/`：Kirin9030 计算 ST
  - `kirin9030/src/common/`：Kirin9030 共享测试资源
- `costmodel/`：代价模型测试
  - `st/`：代价模型 ST（算子代价测量）
  - `st_fit/`：代价模型拟合测试
- `common/`：共享测试资源
- `run_cpu.py`：CPU 仿真器全量运行入口
- `run_cpu_tests.sh`：CPU ST 一键脚本
- `run_st.sh`：NPU ST 一键脚本
- `run_comm_test.sh`：通信 ST 一键脚本（详见下方说明）
- `run_costmodel.py`：代价模型运行脚本
- `run_costmodel_tests.sh`：代价模型一键脚本
- `run_pipeline.sh`：流水线测试脚本
- `validate_op_coverage.py`：算子覆盖率验证脚本
- `validate_testcase_names.py`：用例命名规范验证脚本

## 通信测试（Comm ST）

通信测试验证多卡间的 PTO 通信原语（Put / Get / Broadcast / Gather / Scatter / Reduce / Notify / Wait / Test），基于 MPI + HCCL 实现。

### 前置依赖：MPI 安装

通信测试需要 MPI 环境（MPICH 或 OpenMPI 均可）。运行时需要两个组件：

1. **`mpirun`**：用于启动多进程
2. **`libmpi.so`**：运行时通过 `dlopen` 动态加载

#### 安装 MPICH（推荐）

```bash
# Ubuntu / Debian
sudo apt install mpich libmpich-dev

# CentOS / RHEL / EulerOS
sudo yum install mpich mpich-devel
# 安装后可能需要加载 module 或手动加入 PATH：
export PATH=/usr/lib64/mpich/bin:$PATH
```

#### 从源码安装 MPICH（无 root 权限时）

```bash
wget https://www.mpich.org/static/downloads/4.2.3/mpich-4.2.3.tar.gz
tar xzf mpich-4.2.3.tar.gz && cd mpich-4.2.3
./configure --prefix=$HOME/mpich --disable-fortran
make -j$(nproc) && make install
export MPI_HOME=$HOME/mpich
export PATH=$MPI_HOME/bin:$PATH
```

#### 环境变量

| 变量 | 说明 |
|------|------|
| `MPI_HOME` | MPI 安装根目录，脚本会自动搜索 `$MPI_HOME/bin/mpirun` |
| `MPI_LIB_PATH` | 直接指定 `libmpi.so` 路径（覆盖默认搜索） |

如果 `mpirun` 已在 `PATH` 中且 `libmpi.so` 在标准库路径下，则无需设置这些变量。

#### 验证安装

```bash
mpirun --version
mpirun -n 2 echo "MPI OK"
```

### 同步与异步指令测试

通信测试分为**同步指令**（如 `tput`、`tget`）和**异步指令**（如 `tput_async`、`tget_async`）两类：

| 类型 | 测试用例示例 | CANN 版本要求 |
|------|-------------|--------------|
| 同步指令 | `tput`、`tget`、`treduce`、`tbroadcast` 等 | CANN 8.x 及以上 |
| 异步指令 | `tput_async`、`tget_async` | **CANN 9.0 及以上** |

异步指令依赖 CANN 9.0 引入的 SDMA opapi 接口（如 `aclnnShmemSdmaStarsQuery`），在低版本 CANN 上会因符号缺失而运行失败。因此 `run_comm_test.sh` **默认不包含异步指令测试**，需通过 `-a` 参数显式启用。

### 快速开始

```bash
# 8 卡全量测试（默认 A2/A3，不含异步指令）
./run_comm_test.sh

# 包含异步指令测试（需 CANN 9.0+）
./run_comm_test.sh -a

# 仅跑异步 tput 用例
./run_comm_test.sh -t tput_async

# 指定 A5 SoC，2 卡
./run_comm_test.sh -v a5 -n 2

# 仅跑 tput 用例
./run_comm_test.sh -t tput

# 开启 debug 日志
./run_comm_test.sh -d -t tput
```

也可以通过 `run_st.py` 直接运行，脚本会自动按 rank 数分轮执行：

```bash
# 自动分轮运行 tput_async（2/4/8 rank）
python3 tests/script/run_st.py -r npu -v a3 -t comm/tput_async

# 限制最多 2 rank
python3 tests/script/run_st.py -r npu -v a3 -t comm/tput_async -n 2
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-n` | 可用 NPU 数量：2、4 或 8 | 8 |
| `-v` | SoC 版本：`a3`（Ascend910B）或 `a5`（Ascend950） | a3 |
| `-t` | 指定测试用例（可多次使用），如 `tput`、`treduce` | 全部 |
| `-a` | 包含异步指令测试（`*_async`），需 CANN 9.0+ | 关闭 |
| `-d` | 开启调试模式，打印详细初始化与同步日志 | 关闭 |

### 运行机制

脚本会根据 `-n` 指定的卡数，自动为每个测试用例分别以 2 / 4 / 8 rank 运行，通过 GTest Filter 确保每次只执行与当前 rank 数匹配的测试。例如 `-n 4` 时会先以 2 rank 跑默认用例，再以 4 rank 跑带 `4Ranks` 后缀的用例，跳过 8 rank 用例。

## 建议阅读顺序

- 入门指南（建议先 CPU，再 NPU）：[docs/getting-started.md](../docs/getting-started.md)
