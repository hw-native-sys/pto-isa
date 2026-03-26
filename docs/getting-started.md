<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# Getting Started

This guide helps you set up and run the PTO ISA project. It covers two main scenarios:

1. **CPU Simulator** (Recommended for beginners) - Cross-platform support for macOS, Linux, and Windows
2. **NPU Environment** (Advanced) - Ascend 910B/910C on Linux with CANN toolkit

Choose the section that matches your needs. Most users should start with the CPU simulator.

---

## Part 1: CPU Simulator (Cross-Platform)

The CPU simulator is the easiest way to get started. It works on macOS, Linux, and Windows without requiring specialized hardware.

### Prerequisites

**Required:**
- Git
- Python `>= 3.8` (3.10+ recommended)
- CMake `>= 3.16`
- A C++ compiler with C++20 support:
  - Linux: GCC 13+ or Clang 15+ (bfloat16 support enabled for GCC >= 14)
  - macOS: Xcode/AppleClang (or Homebrew LLVM)
  - Windows: Visual Studio 2022 Build Tools (MSVC)
- Python package: `numpy`

`tests/run_cpu.py` can install `numpy` automatically (unless you pass `--no-install`).

**Optional (for faster builds):**
- Ninja (CMake generator)
- Internet connection (CMake may fetch GoogleTest if not installed system-wide)

### OS-Specific Setup

#### macOS

Install Xcode Command Line Tools:

  ```bash
  xcode-select --install
  ```

Install dependencies (recommended via Homebrew):

  ```bash
  brew install cmake ninja python
  ```

If you don't use Homebrew, ensure `python3`, `cmake`, and a modern `clang++` are on `PATH`.

#### Linux (Ubuntu 20.04+)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-venv git
```

#### Windows

Install the following:
- Git for Windows
- Python 3 (ensure it's on `PATH`)
- CMake
- Visual Studio 2022 Build Tools (Desktop development with C++)

Using `winget` (optional):

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

After installation, open a **Developer Command Prompt for VS 2022** (or ensure `cl.exe` is on `PATH`).

**Alternative: Manual compiler installation**

If you prefer not to use Visual Studio:
- [WinLibs](https://winlibs.com)
- [MSYS2](https://www.msys2.org)

After installation, add `path_to_compiler/bin` to `PATH` (verify with `gcc -v` in PowerShell).

### Get The Code

```bash
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa
```

### Python Environment Setup

Create and activate a virtual environment:

**macOS / Linux:**

  ```bash
  python3 -m venv .venv-mkdocs
  source .venv-mkdocs/bin/activate
  python -m pip install -U pip
  python -m pip install numpy
  ```

**Windows (PowerShell):**

  ```powershell
  py -3 -m venv .venv-mkdocs
  .\.venv-mkdocs\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install numpy
  ```

### Run CPU Simulator

Build and run all CPU ST test binaries under `tests/cpu/st`:

```bash
python3 tests/run_cpu.py --clean --verbose
```

**Common Options:**

Run a single testcase:

  ```bash
  python3 tests/run_cpu.py --testcase tadd
  ```

Run a specific gtest case:

  ```bash
  python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
  ```

Build & run the GEMM demo:

  ```bash
  python3 tests/run_cpu.py --demo gemm --verbose
  ```

Build & run the Flash Attention demo:

  ```bash
  python3 tests/run_cpu.py --demo flash_attn --verbose
  ```

**Additional Options:**

Specify compiler path:

  ```bash
  python3 tests/run_cpu.py --cxx=/path/to/compiler
  ```

Print detailed logs:

  ```bash
  python3 tests/run_cpu.py --verbose
  ```

Delete build directory and rebuild:

  ```bash
  python3 tests/run_cpu.py --clean
  ```

Windows-specific (if needed):

  ```bash
  python3 tests/run_cpu.py --clean --generator "MinGW Makefiles" --cmake_prefix_path D:\gtest\
  ```

Set library path (Linux):

  ```bash
  export LD_LIBRARY_PATH=/path_to_compiler/lib64:$LD_LIBRARY_PATH
  ```

---

## Part 2: NPU Environment (Ascend 910B/910C, Linux Only)

This section is for users who need to run on Ascend NPU hardware or simulator. It requires a Linux environment and the Ascend CANN toolkit.

### Prerequisites

**System Requirements:**
- Linux (Ubuntu 20.04+ recommended)
- Python >= 3.8.0
- GCC >= 7.3.0
- CMake >= 3.16.0

**GoogleTest (required for unit tests):**

Download [GoogleTest 1.14.0](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) and install:

```bash
tar -xf googletest-1.14.0.tar.gz
cd googletest-1.14.0
mkdir temp && cd temp
cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
make
sudo make install
```

> **Note:** Python requires packages: os, numpy, ctypes, struct, copy, math, enum, ml_dtypes, en_dtypes, etc.
>
> If you installed GoogleTest with different flags (e.g., `-D_GLIBCXX_USE_CXX11_ABI=0`), you must update `tests/npu/[a2a3|a5]/src/st/CMakeLists.txt` accordingly by adding `add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)`.

**CANN Toolkit Installation**

- Ascend NPU driver and firmware (for hardware execution)
- CANN toolkit >= 8.5.0

> **Important:** After installing the CANN toolkit, set up the environment variables by running:
>
> ```bash
> source /usr/local/Ascend/cann/bin/setenv.bash
> ```
>
> Then verify that the GCC version shown by bisheng is compatible with or matches the version shown by gcc:
>
> ```bash
> bisheng -v
> gcc -v
> ```
>
> If they differ, you can install a version of GCC that matches the version shown by gcc -v at the gcc path indicated by bisheng -v, or replace the default gcc with the version that matches what bisheng -v shows.

### Installation Options

#### Option 1: Quick Installation (Recommended)

For complete installation guidance including driver, firmware, and toolkit:

https://www.hiascend.com/cann/download

This method handles all dependencies automatically.

#### Option 2: Manual Installation

**Step 1: Install Driver and Firmware**

Required for running on actual NPU hardware (skip if only building or using simulator).

Installation guide: [NPU Driver and Firmware Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

**Step 2: Install CANN Toolkit**

Download the appropriate `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run` installer from [CANN Downloads](https://www.hiascend.com/developer/download/community/result?module=cann).

Required version: CANN >= 8.5.0

```bash
# Make installer executable
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run

# Install
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

Parameters:
- `${cann_version}`: CANN toolkit version
- `${arch}`: CPU architecture (`aarch64` or `x86_64`)
- `${install_path}`: Installation path (optional)

Default paths:
- Root installation: `/usr/local/Ascend/cann`
- Non-root installation: `$HOME/Ascend/cann`

#### Option 3: One-Step Environment Setup Script

If you want to install the CANN toolkit, PTO ISA package, and GoogleTest all at once, use the project's built-in `install_pto.sh` script. It automatically detects whether the CANN toolkit is already installed: if it is, the PTO ISA package is installed directly; if not, provide the CANN toolkit installer path via the second argument.

```bash
chmod +x ./scripts/install_pto.sh
./scripts/install_pto.sh <cann_toolkit_install_path> [cann_toolkit_package_path]
```

Parameters:
- `<cann_toolkit_install_path>`: The installation path of the CANN toolkit (used to detect whether it is already installed)
- `[cann_toolkit_package_path]`: Path to the CANN toolkit installer package (optional; only required if the CANN toolkit is not yet installed)

### Environment Variables

Set up the CANN environment before running NPU tests:

**Root installation (default path):**

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
```

**Non-root installation (default path):**

```bash
source $HOME/Ascend/cann/bin/setenv.bash
```

**Custom installation path:**

```bash
source ${install_path}/cann/bin/setenv.bash
```

### Download Source Code

```bash
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa
```

### Run NPU Tests

**Run a Single ST Test Case:**

  ```bash
  python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]
  ```

Note: The `a3` backend covers the A2/A3 family (`include/pto/npu/a2a3`).

Examples:

  ```bash
  python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
  python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
  ```

**Run Recommended Test Suites:**

  ```bash
# Execute from project root directory
  chmod +x ./tests/run_st.sh
  ./tests/run_st.sh a5 npu simple

# For simulator (increase file descriptor limit first)
ulimit -n 65536
./tests/run_st.sh a3 sim all
  ```

**Run Full ST Tests:**

  ```bash
  chmod +x build.sh
  ./build.sh --run_all --a3 --sim
  ```

**Run Simplified ST Tests:**

  ```bash
  chmod +x build.sh
  ./build.sh --run_simple --a5 --npu
  ```

**Packaging:**

  ```bash
  chmod +x build.sh
  ./build.sh --pkg
  ```

---

## Next Steps

- For CPU development: Explore the demos under `tests/cpu/demos/`
- For NPU development: Review the test cases under `tests/npu/`
- Check the API documentation for detailed instruction usage
- Join the community for support and discussions
