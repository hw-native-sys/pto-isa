# AGENTS.md - PTO Tile Library

This file provides essential information for agentic coding agents working in this repository.

## Build / Lint / Test Commands

### Build Commands
```bash
# Build and run CPU simulator tests (recommended first step)
python3 tests/run_cpu.py --clean --verbose

# Build and run specific CPU demo
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose

# Build NPU tests (requires Ascend CANN environment)
python3 tests/script/build_st.py -r npu -v a3 -t all

# One-click build and run scripts
./build.sh --run_all --a3 --sim    # Full ST tests on simulator
./build.sh --run_simple --a5 --npu # Simplified ST tests on hardware
./build.sh --pkg                    # Build package
```

### Running Single Tests
```bash
# CPU simulator single test
python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.case_float_64x64_64x64'

# NPU single test (sim or npu)
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
python3 tests/script/run_st.py -r npu -v a3 -t tadd -g TADDTest.case_float_64x64_64x64

# Auto mode compilation
python3 tests/script/run_st.py -r sim -v a3 -a -t tadd -g TADDTest.case_float_64x64_64x64
```

### Lint / Format Commands
```bash
# Format C++ code (Google style, 120 char limit)
clang-format -i -style=file <file>

# Format Python code (Ruff)
ruff format <file>
ruff check <file>
```

## Code Style Guidelines

### C++ Code Style
- **Style**: Google style with customizations
- **Line length**: 120 characters
- **Indentation**: 4 spaces (no tabs)
- **Pointer alignment**: Right-aligned (`int* ptr`)
- **Braces**: Functions: opening brace on new line, other blocks: same line
- **Header guards**: `#ifndef <FILENAME>_H_` format

### File Headers
All source files must include the standard copyright header:
```cpp
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
```

### Naming Conventions
- **Classes/Structs**: `PascalCase` (e.g., `GlobalTensor`, `TileShape2D`)
- **Functions**: `PascalCase` for PTO instructions (e.g., `TADD`, `TMATMUL`), `camelCase` for helpers
- **Variables**: `camelCase` (e.g., `src0Tile`, `gmOffsetA`)
- **Constants/Enums**: `UPPER_SNAKE_CASE` (e.g., `BUFFER_NUM`, `PIPE_MTE1`)
- **Template parameters**: `PascalCase` (e.g., `LeftTile`, `RightTile`)
- **Macros**: `UPPER_SNAKE_CASE` (e.g., `PTO_STATIC_ASSERT`, `AICORE`)

### Import Organization
1. System C++ headers (`#include <cstdio>`)
2. Third-party headers (`#include <gtest/gtest.h>`)
3. PTO internal headers (`#include <pto/common/type.hpp>`)
4. Local headers

### PTO Instruction Patterns
```cpp
// Standard PTO instruction usage
#include <pto/pto-inst.hpp>
using namespace pto;

// Tile declaration
using TileData = Tile<TileType::Vec, T, kRows_, kCols_, BLayout::RowMajor, -1, -1>;
TileData srcTile(kRows_, kCols_);

// Global tensor declaration
using DynShape = Shape<1, 1, 1, kGRows_, kGCols_>;
using DynStride = Stride<1, 1, 1, kGCols_, 1>;
using GlobalData = GlobalTensor<T, DynShape, DynStride>;
GlobalData srcGlobal(src);

// PTO instruction pattern
TLOAD(srcTile, srcGlobal);
TADD(dstTile, src0Tile, src1Tile);
TSTORE(dstGlobal, dstTile);
```

### Template and Type Usage
- Use `constexpr` for compile-time constants
- Use `template <typename T, int kRows_, int kCols_>` for parameterized kernels
- Use `__gm__` attribute for global memory pointers
- Use `__out__` and `__in__` attributes for output/input parameters
- Use `AICORE` macro for AI Core functions (expands to `[aicore]` on NPU)
- Use `PTO_INST` for public PTO instruction declarations
- Use `PTO_INTERNAL` for internal implementations

### Assertions and Error Handling
```cpp
// Compile-time assertions
PTO_STATIC_ASSERT(condition);
PTO_STATIC_ASSERT(condition, "custom message");

// Runtime assertions (CPU simulator only)
PTO_CPU_ASSERT(condition);
PTO_CPU_ASSERT(condition, "custom message");

// Google Test assertions in test files
EXPECT_TRUE(condition);
ASSERT_EQ(expected, actual);
```

### Event Synchronization Pattern
```cpp
// Set flag
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

// Wait for flag
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

// Template-based flag helpers
template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetFlag(uint32_t id) {
    set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}
```

### Memory Layout and Buffers
- Use double buffering with `BUFFER_NUM = 2` constant
- Use `Tile<TileType::Vec, ...>` for vector operations
- Use `Tile<TileType::Cube, ...>` for cube operations
- Use `GlobalTensor<T, Shape, Stride>` for global memory access
- Buffer sizes typically use KiB units (e.g., `32 * 1024` for 32 KiB)

### Python Code Style
- **Formatter**: Ruff (configured in pyproject.toml)
- **Quotes**: Double quotes
- **Line length**: 120 characters
- **Indentation**: 4 spaces

### Test File Structure
```cpp
// Test kernel file: <testcase>_kernel.cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

using namespace pto;

template <typename T, int ...params>
AICORE void runTest(__gm__ T __out__ *out, __gm__ T __in__ *src) {
    // Kernel implementation
}

template <typename T, int ...params>
void LaunchTest(T *out, T * *src, void *stream) {
    runTest<T, ...params>(out, src);
}

// Explicit template instantiations
template void LaunchTest<float, ...params>(float *out, float * *src, void *stream);
```

### CMakeLists.txt Pattern
```cmake
# For test cases
pto_costmodel_sim_st(tadd)

# For kernel builds
pto_add_kernel(<target_name>)
```

## Key Directories
- `include/pto/`: Public API headers
- `include/pto/cpu/`: CPU simulator implementations
- `include/pto/npu/`: NPU implementations (a2a3, a5)
- `kernels/manual/`: Manual mode kernel implementations
- `tests/cpu/st/testcase/`: CPU simulator test cases
- `tests/npu/`: NPU test cases
- `tests/script/`: Test runner scripts
- `demos/`: Demo applications

## Important Notes
- Always test on CPU simulator before NPU hardware
- Use `--clean` flag with CPU tests for fresh builds
- NPU tests require `ASCEND_HOME_PATH` environment variable
- C++20 or later is required
- bfloat16 support requires GCC>=14 for CPU simulator
- PTO instructions are case-sensitive and use `T` prefix
