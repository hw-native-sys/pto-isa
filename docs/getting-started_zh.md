<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# 快速开始

本指南涵盖在 **macOS / Linux / Windows** 系统上的先决条件与设置，并首先演示如何构建和运行 **CPU 模拟器**（推荐）。在 Ascend（NPU / 模拟器）上运行需要 Ascend CANN 环境，通常**仅限于 Linux**。

## 先决条件

### 必需项（CPU 模拟器）

- Git
- Python `>= 3.8`（推荐 3.10+）
- CMake `>= 3.16`
- 支持 C++20 的 C++ 编译器：
  - Linux: GCC 13+ 或 Clang 15+
  - macOS: Xcode/AppleClang（或 Homebrew LLVM）
  - Windows: Visual Studio 2022 Build Tools (MSVC)
- Python 包：`numpy`（CPU 测试数据生成器需要使用它）

`run_cpu.py` 可以自动安装 `numpy`（除非您传递 `--no-install` 参数）。

### 可选项（用于加速构建）

- Ninja (CMake 生成器)
- 有效的互联网连接（如果未在系统范围内安装 GoogleTest，CMake 可能需要获取它来编译 CPU ST 测试)

## 操作系统设置

### macOS

- 安装 Xcode 命令行工具：

  ```bash
  xcode-select --install
  ```

- 安装依赖项（推荐通过 Homebrew）：

  ```bash
  brew install cmake ninja python
  ```

如果不使用 Homebrew，请确保 `python3`、`cmake` 和现代的 `clang++` 在 `PATH` 环境变量中。

### Linux (Ubuntu 20.04)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-venv git
```
### Windows

安装以下软件：

- Git for Windows
- Python 3（并确保其在 `PATH` 中）
- CMake
- Visual Studio 2022 Build Tools（需包含“使用 C++ 的桌面开发”）

使用 `winget`（可选）：

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

安装完成后，打开 **Developer Command Prompt for VS 2022**（或确保 `cl.exe` 在 `PATH` 中）。

手动下载编译器（可选）：

可选方案：
- https://winlibs.com
- https://www.msys2.org

安装后，将 `path_to_compiler/bin` 目录添加到 `PATH` 环境变量中（确保在 PowerShell 中 `gcc -v` 可执行）。

## 获取代码

```bash
git clone <YOUR_REPO_URL>
cd pto-isa
```

## Python 环境

创建并激活虚拟环境：

- macOS / Linux：

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip
  python -m pip install numpy
  ```

- Windows (PowerShell)：

  ```powershell
  py -3 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install numpy
  ```

## 运行 CPU 模拟器

此命令将构建并运行 `tests/cpu/st` 目录下的 CPU ST 测试二进制文件，并执行所有测试用例：

```bash
python3 tests/run_cpu.py --clean --verbose
```

常用选项：

- 运行单个测试用例：

  ```bash
  python3 tests/run_cpu.py --testcase tadd
  ```

- 运行单个 gtest 用例：

  ```bash
  python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
  ```

- 构建并运行 GEMM 演示：

  ```bash
  python3 tests/run_cpu.py --demo gemm --verbose
  ```

- 构建并运行 Flash Attention 演示：

  ```bash
  python3 tests/run_cpu.py --demo flash_attn --verbose
  ```

- 可选参数：

  ```bash
  # 指定 cxx 编译器路径
  python3 tests/run_cpu.py --cxx=/path/to/compiler
  ```

  ```bash
  # 打印详细日志
  python3 tests/run_cpu.py --verbose
  ```

  ```bash
  # 删除构建目录并重新构建
  python3 tests/run_cpu.py --clean
  ```

  ```bash
  # 在 Windows 上，可能需要指定生成器和 cmake_prefix_path
  python3 tests/run_cpu.py --clean --generator "MinGW Makefiles" --cmake_prefix_path D:\gtest\
  ```

- 设置环境变量：
  ```bash
  export LD_LIBRARY_PATH=/path_to_compiler/lib64:$LD_LIBRARY_PATH
  ```

## 运行 NPU 测试

首先根据 ./getting-started.md#environment-variables 设置环境变量；

- 运行单个 ST 测试用例

  运行 ST 测试需要一个可用的 Ascend CANN 环境，并且通常仅限于 Linux。

  ```bash
  python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]
  ```

  注意：`a3` 后端覆盖 A2/A3 系列 (`include/pto/npu/a2a3`)。

  示例：

  ```bash
  python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
  python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
  ```

- 运行推荐的测试套件

  ```bash
  # 从项目根目录执行以下命令：
  chmod +x ./tests/run_st.sh
  ./tests/run_st.sh a5 npu simple
  ulimit -n 65536;./tests/run_st.sh a3 sim all # 如果要在模拟器上运行，请先使用 ulimit -n
  ```

- 运行完整的 ST 测试：

  ```bash
  chmod +x build.sh
  ./build.sh --run_all --a3 --sim
  ```
- 运行简化的 ST 测试：

  ```bash
  chmod +x build.sh
  ./build.sh --run_simple --a5 --npu
  ```
- 打包：

  ```bash
  chmod +x build.sh
  ./build.sh --pkg
  ```
- 设置环境

  注意：如果您尚未安装 toolkit，您需要先下载 toolkit 安装包。
  ```bash
  chmod +x ./scripts/install_pto.sh
  ./scripts/install_pto.sh <toolkit_install_path> [toolkit_package_path]
  ```

# 环境设置 (Ascend 910B/910C, Linux)

## 先决条件

在使用本项目前，请确保已安装以下基本依赖项以及 NPU 驱动/固件。

1.  **安装构建依赖**

    项目从源码构建需要以下依赖（请注意版本要求）：

    - Python >= 3.8.0
    - GCC >= 7.3.0
    - CMake >= 3.16.0
    - GoogleTest（仅在运行单元测试时需要；推荐版本：
      https://github.com/google/googletest/releases/tag/v1.14.0）

        下载
        https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz 后，
        通过以下方式安装：

        ```bash
        tar -xf googletest-1.14.0.tar.gz
        cd googletest-1.14.0
        mkdir temp && cd temp                # 在 googletest 源码目录下创建临时构建目录
        cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
        make
        make install                         # 以 root 用户安装
        # sudo make install                  # 以非 root 用户安装
        ```
      > **注意**
      > 
      > Python 需要下载 os、numpy、ctypes、struct、copy、math、enum、ml_dtypes、en_dtypes 等包。
      > 
      > 如果您已通过其他方式安装了 googletest，则需要相应地修改 CMakeLists.txt。例如，如果您在安装 googletest 时使用了 `cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"`，则需要在 tests/npu/[a2a3 | a5]/src/st/CMakeLists.txt 中添加 `add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)`。

2.  **安装驱动和固件（运行时依赖）**

    运行算子需要驱动和固件。如果仅需构建，可跳过此步骤。
    安装指南请参考：
    https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0001.html?Mode=VmIns&OS=Ubuntu&Software=cannToolKit。

## 安装软件包

本项目支持从源码构建。在构建前，请按以下步骤准备环境。

1.  **安装社区版 CANN toolkit**

    根据您的环境下载相应的 `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run` 安装包。https://www.hiascend.com/developer/download/community/result?module=cann.
   
    我们要求的 CANN 版本是 8.5.0 或更高。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - `${cann_version}`: CANN toolkit 版本。
    - `${arch}`: CPU 架构，例如 `aarch64` 或 `x86_64`。
    - `${install_path}`: 安装路径。
    - 如果省略 `--install-path`，将使用默认路径。如果以 root 用户安装，软件将放置在 `/usr/local/Ascend/cann` 下。如果以非 root 用户安装，将放置在 `$HOME/Ascend/cann` 下。

## 环境变量

- 默认路径（以 root 用户安装）

    ```bash
    source /usr/local/Ascend/cann/bin/setenv.bash
    ```

- 默认路径（以非 root 用户安装）
    ```bash
    source $HOME/Ascend/cann/bin/setenv.bash
    ```

- 自定义安装路径
    ```bash
    source ${install_path}/cann/bin/setenv.bash
    ```

## 源代码下载

通过以下命令下载源码：
```bash
# 克隆仓库（以 master 分支为例）
git clone https://github.com/PTO-ISA/pto-isa.git
```
