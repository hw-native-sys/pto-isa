# 编译流程详解

本文档详细介绍 PTO 算子的编译流程，帮助开发者理解从源代码到可执行文件的完整过程，掌握编译优化技巧。

## 目录

- [1. 编译流程概述](#1-%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B%E6%A6%82%E8%BF%B0)
- [2. 构建系统配置](#2-%E6%9E%84%E5%BB%BA%E7%B3%BB%E7%BB%9F%E9%85%8D%E7%BD%AE)
- [3. 编译步骤详解](#3-%E7%BC%96%E8%AF%91%E6%AD%A5%E9%AA%A4%E8%AF%A6%E8%A7%A3)
- [4. 编译选项说明](#4-%E7%BC%96%E8%AF%91%E9%80%89%E9%A1%B9%E8%AF%B4%E6%98%8E)
- [5. 交叉编译](#5-%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91)
- [6. 编译优化](#6-%E7%BC%96%E8%AF%91%E4%BC%98%E5%8C%96)
- [7. 常见问题排查](#7-%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E6%8E%92%E6%9F%A5)
- [8. 高级主题](#8-%E9%AB%98%E7%BA%A7%E4%B8%BB%E9%A2%98)

______________________________________________________________________

## 1. 编译流程概述

### 1.1 完整编译流程图

```text
┌─────────────────────────────────────────────────────────────┐
│                    PTO C++ 源码 (.cpp)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  预处理器 (Preprocessor)                                     │
│  - 宏展开 (#define)                                          │
│  - 头文件包含 (#include)                                     │
│  - 条件编译 (#ifdef)                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  C++ 编译器前端 (Frontend)                                   │
│  - 词法分析 (Lexer)                                          │
│  - 语法分析 (Parser)                                         │
│  - 语义分析 (Semantic Analysis)                              │
│  - 生成 AST (Abstract Syntax Tree)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PTO 内建函数展开                                            │
│  - TLOAD → 底层加载指令                                      │
│  - TSTORE → 底层存储指令                                     │
│  - TADD/TMUL → 底层计算指令                                  │
│  - 静态检查 (Tile 对齐、类型匹配)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  编译器中端 (Middle-end)                                     │
│  - 优化 Pass (内联、循环展开、常量折叠)                      │
│  - 生成中间表示 (IR)                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  编译器后端 (Backend)                                        │
│  - 指令选择                                                  │
│  - 寄存器分配                                                │
│  - 指令调度                                                  │
│  - 生成目标代码 (.o)                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  链接器 (Linker)                                             │
│  - 符号解析                                                  │
│  - 重定位                                                    │
│  - 生成可执行文件 / 共享库                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  可执行文件 / 共享库 (.so / .exe)                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 编译工具链

#### 必需工具

**CMake**：

- 版本要求：>= 3.16
- 用途：构建系统生成器
- 安装：

  ```bash
  # Ubuntu/Debian
  sudo apt install cmake

  # CentOS/RHEL
  sudo yum install cmake

  # macOS
  brew install cmake

  # Windows
  # 从 https://cmake.org/download/ 下载安装
  ```

**C++ 编译器**：

- 要求：支持 C++20 标准
- Linux 选项：
  - GCC >= 13.0
  - Clang >= 15.0
- Windows 选项：
  - MSVC 2022 (Visual Studio 17.0+)
  - MinGW-w64 (GCC 13+)
- 安装：

  ```bash
  # Ubuntu/Debian - GCC
  sudo apt install g++-13

  # Ubuntu/Debian - Clang
  sudo apt install clang-15

  # CentOS/RHEL
  sudo yum install gcc-toolset-13
  ```

**Python**：

- 版本要求：>= 3.8
- 用途：构建脚本、测试工具
- 安装：

  ```bash
  # Ubuntu/Debian
  sudo apt install python3 python3-pip

  # CentOS/RHEL
  sudo yum install python3 python3-pip
  ```

#### 可选工具

**Ninja**：

- 用途：加速构建（比 Make 快 2-3×）
- 安装：

  ```bash
  # Ubuntu/Debian
  sudo apt install ninja-build

  # CentOS/RHEL
  sudo yum install ninja-build

  # macOS
  brew install ninja
  ```

**ccache**：

- 用途：编译缓存（加速重复编译）
- 安装：

  ```bash
  # Ubuntu/Debian
  sudo apt install ccache

  # 配置
  export CC="ccache gcc"
  export CXX="ccache g++"
  ```

**clang-tidy**：

- 用途：静态代码分析
- 安装：

  ```bash
  sudo apt install clang-tidy
  ```

______________________________________________________________________

## 2. 构建系统配置

### 2.1 CMake 基础配置

**最小配置示例**：

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(MyPTOOperator VERSION 1.0.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 PTO 库
find_package(PTO REQUIRED)

# 添加可执行文件
add_executable(my_operator
  src/my_operator.cpp
)

# 链接 PTO 库
target_link_libraries(my_operator
  PRIVATE PTO::pto
)
```

**完整配置示例**：

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyPTOOperator VERSION 1.0.0 LANGUAGES CXX)

# ============ 编译选项 ============
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 导出编译命令（用于 IDE 和工具）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============ 构建类型 ============
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Debug 选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Release 选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# RelWithDebInfo 选项
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# ============ PTO 配置 ============
# 设置 PTO 后端
set(PTO_BACKEND "CPU" CACHE STRING "PTO backend: CPU or NPU")
set_property(CACHE PTO_BACKEND PROPERTY STRINGS CPU NPU)

# 设置 SOC 版本（NPU 后端）
if(PTO_BACKEND STREQUAL "NPU")
  set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")
  set_property(CACHE SOC_VERSION PROPERTY STRINGS
    Ascend910B1    # A2
    Ascend910B2    # A3
    Ascend910_9599 # A5
  )
endif()

# 查找 PTO 库
find_package(PTO REQUIRED)

# ============ 源文件 ============
file(GLOB_RECURSE SOURCES
  src/*.cpp
)

# ============ 可执行文件 ============
add_executable(my_operator ${SOURCES})

# 包含目录
target_include_directories(my_operator
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接库
target_link_libraries(my_operator
  PRIVATE
    PTO::pto
)

# 编译选项
target_compile_options(my_operator
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    $<$<CONFIG:Release>:-ffast-math>
)

# ============ 安装 ============
install(TARGETS my_operator
  RUNTIME DESTINATION bin
)

# ============ 测试 ============
enable_testing()
add_subdirectory(tests)
```

### 2.2 配置选项说明

**后端选择**：

```bash
# CPU 仿真构建（开发调试）
cmake -B build -DPTO_BACKEND=CPU

# NPU 构建（A2 芯片）
cmake -B build -DPTO_BACKEND=NPU -DSOC_VERSION=Ascend910B1

# NPU 构建（A3 芯片）
cmake -B build -DPTO_BACKEND=NPU -DSOC_VERSION=Ascend910B2

# NPU 构建（A5 芯片）
cmake -B build -DPTO_BACKEND=NPU -DSOC_VERSION=Ascend910_9599
```

**构建类型**：

```bash
# Debug 构建（无优化，包含调试符号）
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release 构建（完全优化，无调试符号）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo 构建（优化 + 调试符号）
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# MinSizeRel 构建（优化代码大小）
cmake -B build -DCMAKE_BUILD_TYPE=MinSizeRel
```

**编译器选择**：

```bash
# 使用 GCC
cmake -B build -DCMAKE_CXX_COMPILER=g++-13

# 使用 Clang
cmake -B build -DCMAKE_CXX_COMPILER=clang++-15

# 使用 ccache 加速
cmake -B build \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER=g++
```

**生成器选择**：

```bash
# 使用 Make（默认）
cmake -B build

# 使用 Ninja（推荐，更快）
cmake -B build -G Ninja

# 使用 Visual Studio（Windows）
cmake -B build -G "Visual Studio 17 2022"
```

### 2.3 构建命令

**标准构建流程**：

```bash
# 步骤1：配置
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 步骤2：编译
cmake --build build -j$(nproc)

# 步骤3：运行测试
ctest --test-dir build --output-on-failure

# 步骤4：安装
cmake --install build --prefix /path/to/install
```

**增量构建**：

```bash
# 只重新编译修改的文件
cmake --build build

# 强制重新编译所有文件
cmake --build build --clean-first
```

**并行构建**：

```bash
# 使用所有 CPU 核心
cmake --build build -j$(nproc)

# 使用指定数量的核心
cmake --build build -j8

# Ninja 自动并行
ninja -C build
```

**详细输出**：

```bash
# 显示编译命令
cmake --build build --verbose

# 或使用环境变量
VERBOSE=1 cmake --build build
```

______________________________________________________________________

## 3. 编译步骤详解

### 3.1 预处理阶段

**宏展开**：

```cpp
// 源码
#define TILE_SIZE 256
#define TILE_SHAPE 16, TILE_SIZE

using TileT = Tile<TileType::Vec, float, TILE_SHAPE>;

// 预处理后
using TileT = Tile<TileType::Vec, float, 16, 256>;
```

**头文件包含**：

```cpp
// 源码
#include <pto/pto-inst.hpp>

// 预处理后（展开为所有 PTO 头文件）
#include <pto/tile.hpp>
#include <pto/global_tensor.hpp>
#include <pto/intrinsics.hpp>
// ... 更多头文件
```

**条件编译**：

```cpp
// 源码
#ifdef PTO_BACKEND_CPU
  // CPU 仿真代码
  run_cpu_kernel();
#else
  // NPU 代码
  run_npu_kernel();
#endif

// 预处理后（CPU 后端）
run_cpu_kernel();

// 预处理后（NPU 后端）
run_npu_kernel();
```

**查看预处理结果**：

```bash
# GCC
g++ -E -P src/my_operator.cpp -o my_operator.i

# Clang
clang++ -E -P src/my_operator.cpp -o my_operator.i
```

### 3.2 编译阶段

**词法分析**：

```cpp
// 源码
TLOAD(tile, input);

// Token 流
IDENTIFIER(TLOAD)
LPAREN
IDENTIFIER(tile)
COMMA
IDENTIFIER(input)
RPAREN
SEMICOLON
```

**语法分析**：

```text
FunctionCall
├─ Function: TLOAD
└─ Arguments
    ├─ tile
    └─ input
```

**语义分析**：

```cpp
// 检查类型匹配
TLOAD(tile, input);
// tile: Tile<TileType::Vec, float, 16, 256>
// input: GlobalTensor<float>
// ✓ 类型兼容

// 检查对齐
static_assert(256 % 16 == 0, "Tile width must be aligned");
// ✓ 对齐检查通过
```

**PTO 内建函数展开**：

```cpp
// 源码
TLOAD(tile, input);

// 展开为底层指令
__builtin_pto_load(
  tile.data(),
  input.data(),
  tile.size(),
  tile.alignment()
);
```

**生成目标代码**：

```bash
# 编译为目标文件
g++ -std=c++20 -O3 -c src/my_operator.cpp -o build/my_operator.o

# 查看生成的汇编代码
g++ -std=c++20 -O3 -S src/my_operator.cpp -o build/my_operator.s
```

### 3.3 链接阶段

**符号解析**：

```text
my_operator.o:
  - 定义: main, my_kernel
  - 引用: TLOAD, TSTORE, TADD

libpto.a:
  - 定义: TLOAD, TSTORE, TADD, ...

链接器解析:
  my_operator.o::TLOAD → libpto.a::TLOAD ✓
  my_operator.o::TSTORE → libpto.a::TSTORE ✓
  my_operator.o::TADD → libpto.a::TADD ✓
```

**重定位**：

```text
my_operator.o 中的调用:
  call TLOAD  // 地址未知

链接后:
  call 0x12345678  // 解析为 libpto.a 中的实际地址
```

**生成可执行文件**：

```bash
# 链接
g++ build/my_operator.o \
    -L/path/to/pto/lib \
    -lpto \
    -o build/my_operator

# 查看依赖库
ldd build/my_operator
# 输出:
#   libpto.so => /path/to/pto/lib/libpto.so
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
```

______________________________________________________________________

## 4. 编译选项说明

### 4.1 优化级别

**-O0（无优化）**：

- 用途：调试
- 特点：
  - 编译最快
  - 代码与源码一一对应
  - 便于调试
- 性能：最慢

**-O1（基本优化）**：

- 用途：快速编译 + 基本优化
- 特点：
  - 编译较快
  - 基本优化（常量折叠、死代码消除）
- 性能：中等

**-O2（标准优化）**：

- 用途：生产环境（推荐）
- 特点：
  - 编译时间适中
  - 大部分优化（内联、循环优化）
  - 不影响调试
- 性能：快

**-O3（激进优化）**：

- 用途：性能关键代码
- 特点：
  - 编译最慢
  - 所有优化（向量化、循环展开）
  - 可能增加代码大小
- 性能：最快

**-Os（优化代码大小）**：

- 用途：嵌入式系统
- 特点：
  - 最小化代码大小
  - 牺牲部分性能
- 性能：中等

**-Ofast（超激进优化）**：

- 用途：不严格遵守标准的代码
- 特点：
  - 包含 -O3
  - 启用 -ffast-math（可能违反 IEEE 754）
- 性能：最快（但可能不正确）

**性能对比**：

```bash
# 测试不同优化级别
for opt in O0 O1 O2 O3 Ofast; do
  g++ -$opt src/my_operator.cpp -o build/my_operator_$opt
  time ./build/my_operator_$opt
done

# 典型结果：
# -O0: 1000 ms
# -O1: 500 ms
# -O2: 200 ms
# -O3: 150 ms
# -Ofast: 140 ms
```

### 4.2 架构特定选项

**-march=native**：

- 用途：针对当前 CPU 优化
- 特点：
  - 使用 CPU 特定指令（AVX2, AVX-512）
  - 性能提升 10-30%
  - 不可移植

**-march=x86-64**：

- 用途：通用 x86-64 代码
- 特点：
  - 兼容所有 x86-64 CPU
  - 不使用高级指令
  - 可移植

**示例**：

```bash
# 针对当前 CPU 优化
g++ -O3 -march=native src/my_operator.cpp

# 通用构建
g++ -O3 -march=x86-64 src/my_operator.cpp

# 针对特定 CPU
g++ -O3 -march=skylake src/my_operator.cpp
```

### 4.3 调试选项

**-g（包含调试符号）**：

```bash
# 基本调试信息
g++ -g src/my_operator.cpp

# 详细调试信息（包含宏定义）
g++ -g3 src/my_operator.cpp

# 使用 gdb 调试
gdb ./my_operator
```

**-fsanitize（运行时检查）**：

```bash
# 地址检查（检测内存错误）
g++ -fsanitize=address src/my_operator.cpp

# 未定义行为检查
g++ -fsanitize=undefined src/my_operator.cpp

# 线程检查
g++ -fsanitize=thread src/my_operator.cpp
```

### 4.4 警告选项

**推荐警告选项**：

```bash
g++ -Wall -Wextra -Wpedantic \
    -Werror \
    src/my_operator.cpp

# -Wall: 常见警告
# -Wextra: 额外警告
# -Wpedantic: 严格标准警告
# -Werror: 警告视为错误
```

______________________________________________________________________

## 5. 交叉编译

### 5.1 x86 → ARM 交叉编译

**安装交叉编译工具链**：

```bash
# Ubuntu/Debian
sudo apt install g++-aarch64-linux-gnu

# 验证
aarch64-linux-gnu-g++ --version
```

**CMake 配置**：

```cmake
# toolchain-aarch64.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

**构建**：

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64.cmake \
  -DPTO_BACKEND=NPU

cmake --build build
```

### 5.2 开发机 → NPU 交叉编译

**配置**：

```bash
# 设置 NPU 工具链路径
export NPU_TOOLCHAIN=/usr/local/Ascend/toolkit

# 配置 CMake
cmake -B build \
  -DPTO_BACKEND=NPU \
  -DSOC_VERSION=Ascend910B1 \
  -DCMAKE_TOOLCHAIN_FILE=${NPU_TOOLCHAIN}/cmake/toolchain.cmake

# 编译
cmake --build build
```

______________________________________________________________________

## 6. 编译优化

### 6.1 加速编译

**使用 Ninja**：

```bash
# 比 Make 快 2-3×
cmake -B build -G Ninja
ninja -C build
```

**使用 ccache**：

```bash
# 缓存编译结果
export CC="ccache gcc"
export CXX="ccache g++"

cmake -B build
cmake --build build

# 查看缓存统计
ccache -s
```

**并行编译**：

```bash
# 使用所有核心
cmake --build build -j$(nproc)

# 限制并行数（避免内存不足）
cmake --build build -j4
```

**预编译头文件**：

```cmake
# CMakeLists.txt
target_precompile_headers(my_operator
  PRIVATE
    <pto/pto-inst.hpp>
    <vector>
    <string>
)
```

### 6.2 减小二进制大小

**Strip 调试符号**：

```bash
# 编译时不包含调试符号
g++ -O3 -DNDEBUG src/my_operator.cpp

# 或编译后 strip
strip build/my_operator

# 大小对比：
# 带调试符号: 5.2 MB
# strip 后: 1.1 MB
```

**链接时优化（LTO）**：

```cmake
# CMakeLists.txt
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# 或手动指定
target_compile_options(my_operator PRIVATE -flto)
target_link_options(my_operator PRIVATE -flto)
```

______________________________________________________________________

## 7. 常见问题排查

### 7.1 编译错误

#### 问题1：找不到头文件

```text
error: pto/pto-inst.hpp: No such file or directory
```

**原因**：PTO 库路径未设置

**解决方案**：

```bash
# 方法1：设置环境变量
export PTO_LIB_PATH=/path/to/pto-isa

# 方法2：CMake 指定
cmake -B build -DPTO_ROOT=/path/to/pto-isa

# 方法3：手动指定包含路径
g++ -I/path/to/pto-isa/include src/my_operator.cpp
```

#### 问题2：静态断言失败

```text
static_assert failed: "Tile shape not aligned"
```

**原因**：Tile 尺寸不满足对齐要求

**解决方案**：

```cpp
// 错误：宽度 250 不是 16 的倍数
using TileT = Tile<TileType::Vec, float, 16, 250>;

// 正确：宽度 256 是 16 的倍数
using TileT = Tile<TileType::Vec, float, 16, 256>;
```

#### 问题3：链接错误

```text
undefined reference to `pto::TLOAD(...)`
```

**原因**：未链接 PTO 库

**解决方案**：

```cmake
# CMakeLists.txt
target_link_libraries(my_operator PRIVATE PTO::pto)

# 或手动链接
g++ build/my_operator.o -L/path/to/pto/lib -lpto -o build/my_operator
```

### 7.2 性能问题

#### 问题：Release 构建性能差

**诊断**：

```bash
# 检查优化级别
cmake --build build --verbose | grep "\-O"

# 应该看到 -O3 或 -O2
```

**解决方案**：

```cmake
# 显式设置优化选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# 或使用 LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
```

### 7.3 运行时错误

#### 问题：找不到共享库

```text
error while loading shared libraries: libpto.so: cannot open shared object file
```

**解决方案**：

```bash
# 方法1：设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/pto/lib:$LD_LIBRARY_PATH

# 方法2：添加到系统路径
sudo echo "/path/to/pto/lib" > /etc/ld.so.conf.d/pto.conf
sudo ldconfig

# 方法3：使用 RPATH
cmake -B build -DCMAKE_INSTALL_RPATH=/path/to/pto/lib
```

______________________________________________________________________

## 8. 高级主题

### 8.1 自定义编译 Pass

#### 示例：添加自定义优化

```cmake
# CMakeLists.txt
target_compile_options(my_operator
  PRIVATE
    -fplugin=/path/to/my_plugin.so
    -fplugin-arg-my_plugin-option=value
)
```

### 8.2 编译时间分析

**GCC 时间报告**：

```bash
g++ -ftime-report src/my_operator.cpp 2>&1 | grep "TOTAL"
```

**Clang 时间追踪**：

```bash
clang++ -ftime-trace src/my_operator.cpp
# 生成 my_operator.json
# 使用 chrome://tracing 查看
```

### 8.3 生成编译数据库

**用于 IDE 和工具**：

```bash
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 生成 build/compile_commands.json
# 用于 clangd, clang-tidy 等工具
```

______________________________________________________________________

## 参考资源

- [快速入门](../getting-started_zh.md)
- [算子调试指南](debug_zh.md)
- [性能优化指南](opt_zh.md)
- [CMake 官方文档](https://cmake.org/documentation/)
- [GCC 优化选项](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
