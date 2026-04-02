# Compilation Process

This document explains the PTO operator compilation process, helping developers understand the complete workflow from source code to executable files.

## Contents

- [1. Compilation Overview](#1-compilation-overview)
- [2. Build System Configuration](#2-build-system-configuration)
- [3. Compilation Steps](#3-compilation-steps)
- [4. Compilation Options](#4-compilation-options)
- [5. Cross Compilation](#5-cross-compilation)
- [6. Compilation Optimization](#6-compilation-optimization)
- [7. Troubleshooting](#7-troubleshooting)

______________________________________________________________________

## 1. Compilation Overview

### 1.1 Compilation Pipeline

```text
PTO C++ Source (.cpp)
    ↓
Preprocessor (macro expansion, #include, #ifdef)
    ↓
C++ Frontend (lexer, parser, semantic analysis, AST)
    ↓
PTO Intrinsic Expansion (TLOAD/TSTORE/TADD → low-level instructions)
    ↓
Middle-end (optimization passes, IR generation)
    ↓
Backend (instruction selection, register allocation, code generation)
    ↓
Linker (symbol resolution, relocation)
    ↓
Executable / Shared Library
```

### 1.2 Required Tools

**CMake** (>= 3.16):

```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake
```

**C++ Compiler** (C++20 support):

- GCC >= 13.0
- Clang >= 15.0
- MSVC 2022 (Windows)

**Python** (>= 3.8):

```bash
sudo apt install python3 python3-pip
```

### 1.3 Optional Tools

**Ninja** (faster builds):

```bash
sudo apt install ninja-build
```

**ccache** (compilation cache):

```bash
sudo apt install ccache
export CC="ccache gcc"
export CXX="ccache g++"
```

______________________________________________________________________

## 2. Build System Configuration

### 2.1 Minimal CMake Configuration

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyPTOOperator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PTO REQUIRED)

add_executable(my_operator src/my_operator.cpp)
target_link_libraries(my_operator PRIVATE PTO::pto)
```

### 2.2 Build Configuration

**Backend Selection**:

```bash
# CPU simulation
cmake -B build -DPTO_BACKEND=CPU

# NPU (A2/A3)
cmake -B build -DPTO_BACKEND=NPU -DSOC_VERSION=Ascend910B1

# NPU (A5)
cmake -B build -DPTO_BACKEND=NPU -DSOC_VERSION=Ascend910_9599
```

**Build Types**:

```bash
# Debug (no optimization, debug symbols)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release (full optimization)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo (optimization + debug symbols)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### 2.3 Build Commands

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Test
ctest --test-dir build --output-on-failure

# Install
cmake --install build --prefix /path/to/install
```

______________________________________________________________________

## 3. Compilation Steps

### 3.1 Preprocessing

**Macro Expansion**:

```cpp
// Source
#define TILE_SIZE 256
using TileT = Tile<TileType::Vec, float, 16, TILE_SIZE>;

// After preprocessing
using TileT = Tile<TileType::Vec, float, 16, 256>;
```

**View Preprocessed Output**:

```bash
g++ -E -P src/my_operator.cpp -o my_operator.i
```

### 3.2 Compilation

**PTO Intrinsic Expansion**:

```cpp
// Source
TLOAD(tile, input);

// Expanded to low-level instructions
__builtin_pto_load(tile.data(), input.data(), tile.size(), tile.alignment());
```

**Generate Object File**:

```bash
g++ -std=c++20 -O3 -c src/my_operator.cpp -o build/my_operator.o
```

### 3.3 Linking

**Symbol Resolution**:

```text
my_operator.o:
  - Defines: main, my_kernel
  - References: TLOAD, TSTORE, TADD

libpto.a:
  - Defines: TLOAD, TSTORE, TADD, ...

Linker resolves:
  my_operator.o::TLOAD → libpto.a::TLOAD ✓
```

**Generate Executable**:

```bash
g++ build/my_operator.o -L/path/to/pto/lib -lpto -o build/my_operator
```

______________________________________________________________________

## 4. Compilation Options

### 4.1 Optimization Levels

| Option   | Use Case                              | Performance |
| -------- | ------------------------------------- | ----------- |
| `-O0`    | Debugging                             | Slowest     |
| `-O1`    | Basic optimization                    | Medium      |
| `-O2`    | Production (recommended)              | Fast        |
| `-O3`    | Maximum optimization                  | Fastest     |
| `-Os`    | Size optimization                     | Medium      |
| `-Ofast` | Aggressive (may violate standards)    | Fastest     |

**Example**:

```bash
# Production build
g++ -O3 -march=native src/my_operator.cpp

# Debug build
g++ -O0 -g src/my_operator.cpp
```

### 4.2 Architecture-Specific Options

**-march=native**: Optimize for current CPU

```bash
g++ -O3 -march=native src/my_operator.cpp
```

**-march=x86-64**: Generic x86-64 code

```bash
g++ -O3 -march=x86-64 src/my_operator.cpp
```

### 4.3 Debug Options

**Debug Symbols**:

```bash
g++ -g src/my_operator.cpp
gdb ./my_operator
```

**Sanitizers**:

```bash
# Address sanitizer (memory errors)
g++ -fsanitize=address src/my_operator.cpp

# Undefined behavior sanitizer
g++ -fsanitize=undefined src/my_operator.cpp
```

### 4.4 Warning Options

```bash
g++ -Wall -Wextra -Wpedantic -Werror src/my_operator.cpp
```

______________________________________________________________________

## 5. Cross Compilation

### 5.1 x86 → ARM Cross Compilation

**Install Toolchain**:

```bash
sudo apt install g++-aarch64-linux-gnu
```

**CMake Toolchain File**:

```cmake
# toolchain-aarch64.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```

**Build**:

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64.cmake
cmake --build build
```

______________________________________________________________________

## 6. Compilation Optimization

### 6.1 Speed Up Compilation

**Use Ninja**:

```bash
cmake -B build -G Ninja
ninja -C build
```

**Use ccache**:

```bash
export CC="ccache gcc"
export CXX="ccache g++"
cmake -B build
cmake --build build
```

**Parallel Build**:

```bash
cmake --build build -j$(nproc)
```

**Precompiled Headers**:

```cmake
target_precompile_headers(my_operator PRIVATE <pto/pto-inst.hpp>)
```

### 6.2 Reduce Binary Size

**Strip Debug Symbols**:

```bash
strip build/my_operator
```

**Link-Time Optimization (LTO)**:

```cmake
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
```

______________________________________________________________________

## 7. Troubleshooting

### 7.1 Common Compilation Errors

#### Error: Header not found

```text
error: pto/pto-inst.hpp: No such file or directory
```

**Solution**:

```bash
export PTO_LIB_PATH=/path/to/pto-isa
cmake -B build -DPTO_ROOT=/path/to/pto-isa
```

#### Error: Static assertion failed

```text
static_assert failed: "Tile shape not aligned"
```

**Solution**:

```cpp
// Wrong: width 250 is not multiple of 16
using TileT = Tile<TileType::Vec, float, 16, 250>;

// Correct: width 256 is multiple of 16
using TileT = Tile<TileType::Vec, float, 16, 256>;
```

#### Error: Undefined reference

```text
undefined reference to `pto::TLOAD(...)`
```

**Solution**:

```cmake
target_link_libraries(my_operator PRIVATE PTO::pto)
```

### 7.2 Runtime Errors

#### Error: Shared library not found

```text
error while loading shared libraries: libpto.so
```

**Solution**:

```bash
export LD_LIBRARY_PATH=/path/to/pto/lib:$LD_LIBRARY_PATH
```

______________________________________________________________________

## References

- [Getting Started](../getting-started.md)
- [Debugging Guide](debug.md)
- [Performance Optimization](opt.md)
- [CMake Documentation](https://cmake.org/documentation/)
