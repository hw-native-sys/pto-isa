<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# 快速开始

本指南帮助您设置和运行 PTO ISA 项目。它涵盖两种主要场景：

1. **CPU 模拟器**（推荐新手）- 跨平台支持 macOS、Linux 和 Windows
2. **NPU 环境**（高级）- Linux 上的 Ascend 910B/910C 与 CANN toolkit

请选择符合您需求的部分。大多数用户应该从 CPU 模拟器开始。

---

## 第一部分：CPU 模拟器（跨平台）

CPU 模拟器是最简单的入门方式。它可以在 macOS、Linux 和 Windows 上运行，无需专用硬件。

### 先决条件

**必需项：**
- Git
- Python `>= 3.8`（推荐 3.10+）
- CMake `>= 3.16`
- 支持 C++20 的 C++ 编译器：
  - Linux: GCC 13+ 或 Clang 15+（GCC >= 14 启用 bfloat16 支持）
  - macOS: Xcode/AppleClang（或 Homebrew LLVM）
  - Windows: Visual Studio 2022 Build Tools (MSVC)
- Python 包：`numpy`

`run_cpu.py` 可以自动安装 `numpy`（除非您传递 `--no-install` 参数）。

**可选项（用于加速构建）：**
- Ninja (CMake 生成器)
- 互联网连接（如果未在系统范围内安装 GoogleTest，CMake 可能需要获取它）

### 操作系统特定设置

#### macOS

安装 Xcode 命令行工具：

  ```bash
  xcode-select --install
  ```

安装依赖项（推荐通过 Homebrew）：

  ```bash
  brew install cmake ninja python
  ```

如果不使用 Homebrew，请确保 `python3`、`cmake` 和现代的 `clang++` 在 `PATH` 环境变量中。

#### Linux (Ubuntu 20.04+)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-venv git
```

#### Windows

安装以下软件：
- Git for Windows
- Python 3（并确保其在 `PATH` 中）
- CMake
- Visual Studio 2022 Build Tools（需包含"使用 C++ 的桌面开发"）

使用 `winget`（可选）：

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

安装完成后，打开 **Developer Command Prompt for VS 2022**（或确保 `cl.exe` 在 `PATH` 中）。

**替代方案：手动安装编译器**

如果您不想使用 Visual Studio：
- [WinLibs](https://winlibs.com)
- [MSYS2](https://www.msys2.org)

安装后，将 `path_to_compiler/bin` 目录添加到 `PATH` 环境变量中（在 PowerShell 中使用 `gcc -v` 验证）。

### 获取代码

```bash
git clone <YOUR_REPO_URL>
cd pto-isa
```

### Python 环境设置

创建并激活虚拟环境：

**macOS / Linux：**

  ```bash
  python3 -m venv .venv-mkdocs
  source .venv-mkdocs/bin/activate
  python -m pip install -U pip
  python -m pip install numpy
  ```

**Windows (PowerShell)：**

  ```powershell
  py -3 -m venv .venv-mkdocs
  .\.venv-mkdocs\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install numpy
  ```

### 运行 CPU 模拟器

构建并运行 `tests/cpu/st` 目录下的所有 CPU ST 测试二进制文件：

```bash
python3 tests/run_cpu.py --clean --verbose
```

**常用选项：**

运行单个测试用例：

  ```bash
  python3 tests/run_cpu.py --testcase tadd
  ```

运行特定的 gtest 用例：

  ```bash
  python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
  ```

构建并运行 GEMM 演示：

  ```bash
  python3 tests/run_cpu.py --demo gemm --verbose
  ```

构建并运行 Flash Attention 演示：

  ```bash
  python3 tests/run_cpu.py --demo flash_attn --verbose
  ```

**其他选项：**

指定编译器路径：

  ```bash
  python3 tests/run_cpu.py --cxx=/path/to/compiler
  ```

打印详细日志：

  ```bash
  python3 tests/run_cpu.py --verbose
  ```

删除构建目录并重新构建：

  ```bash
  python3 tests/run_cpu.py --clean
  ```

Windows 特定选项（如需要）：

  ```bash
  python3 tests/run_cpu.py --clean --generator "MinGW Makefiles" --cmake_prefix_path D:\gtest\
  ```

设置库路径（Linux）：

  ```bash
  export LD_LIBRARY_PATH=/path_to_compiler/lib64:$LD_LIBRARY_PATH
  ```

---

## 第二部分：NPU 环境（Ascend 910B/910C，仅限 Linux）

本部分适用于需要在 Ascend NPU 硬件或模拟器上运行的用户。它需要 Linux 环境和 Ascend CANN toolkit。

### 先决条件

**系统要求：**
- Linux（推荐 Ubuntu 20.04+）
- Python >= 3.8.0
- GCC >= 7.3.0
- CMake >= 3.16.0
- Ascend NPU 驱动和固件（用于硬件执行）
- CANN toolkit >= 8.5.0

> **注意：** 请确保bisheng -v显示的GCC版本与gcc -v显示的GCC版本兼容或一致。
>
> 如果不同，您可以在bisheng -v所示的gcc路径下，安装与gcc -v所显示版本一致的gcc，或将默认的gcc替换为与bisheng -v显示一致的版本。

**GoogleTest（单元测试所需）：**

下载 [GoogleTest 1.14.0](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) 并安装：

```bash
tar -xf googletest-1.14.0.tar.gz
cd googletest-1.14.0
mkdir temp && cd temp
cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
make
sudo make install
```

> **注意：** Python 需要以下包：os、numpy、ctypes、struct、copy、math、enum、ml_dtypes、en_dtypes 等。
>
> 如果您使用不同的标志安装了 GoogleTest（例如 `-D_GLIBCXX_USE_CXX11_ABI=0`），则必须在 `tests/npu/[a2a3|a5]/src/st/CMakeLists.txt` 中添加 `add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)` 进行相应更新。

### 安装选项

#### 选项 1：快速安装（推荐）

完整的安装指南（包括驱动、固件和 toolkit）：

<https://www.hiascend.com/cann/download>

此方法会自动处理所有依赖项。

#### 选项 2：手动安装

**步骤 1：安装驱动和固件**

在实际 NPU 硬件上运行时需要（如果仅构建或使用模拟器，可跳过）。

安装指南：[NPU 驱动和固件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0001.html?Mode=VmIns&OS=Ubuntu&Software=cannToolKit)

**步骤 2：安装 CANN Toolkit**

从 [CANN 下载页面](https://www.hiascend.com/developer/download/community/result?module=cann) 下载相应的 `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run` 安装包。

要求版本：CANN >= 8.5.0

```bash
# 使安装包可执行
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run

# 安装
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

参数说明：
- `${cann_version}`：CANN toolkit 版本
- `${arch}`：CPU 架构（`aarch64` 或 `x86_64`）
- `${install_path}`：安装路径（可选）

默认路径：
- Root 用户安装：`/usr/local/Ascend/cann`
- 非 root 用户安装：`$HOME/Ascend/cann`

### 环境变量

在运行 NPU 测试之前设置 CANN 环境：

**Root 用户安装（默认路径）：**

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
```

**非 root 用户安装（默认路径）：**

```bash
source $HOME/Ascend/cann/bin/setenv.bash
```

**自定义安装路径：**

```bash
source ${install_path}/cann/bin/setenv.bash
```

### 下载源代码

```bash
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa
```

### 运行 NPU 测试

**运行单个 ST 测试用例：**

  ```bash
  python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] [-a] -t [TEST_CASE] -g [GTEST_FILTER_CASE]
  ```

注意：`a3` 后端覆盖 A2/A3 系列（`include/pto/npu/a2a3`）；`-a`使能auto模式。

  示例：

  ```bash
  python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
  python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
  ```

**运行推荐的测试套件：**

  ```bash
# 从项目根目录执行
  chmod +x ./tests/run_st.sh
  ./tests/run_st.sh a5 npu simple

# 对于模拟器（首先增加文件描述符限制）
ulimit -n 65536
./tests/run_st.sh a3 sim all
  ```

**运行完整的 ST 测试：**

  ```bash
  chmod +x build.sh
  ./build.sh --run_all --a3 --sim
  ```

**运行简化的 ST 测试：**

  ```bash
  chmod +x build.sh
  ./build.sh --run_simple --a5 --npu
  ```

**打包：**

  ```bash
  chmod +x build.sh
  ./build.sh --pkg
  ```

**环境设置脚本：**

如果您尚未安装 toolkit，请先下载 toolkit 安装包，然后：

  ```bash
  chmod +x ./scripts/install_pto.sh
  ./scripts/install_pto.sh <toolkit_install_path> [toolkit_package_path]
  ```

---

## 下一步

- CPU 开发：探索 `tests/cpu/demos/` 下的演示
- NPU 开发：查看 `tests/npu/` 下的测试用例
- 查看 API 文档了解详细的指令使用方法
- 加入社区获取支持和讨论
