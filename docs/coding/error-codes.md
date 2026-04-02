# Error Codes Reference

This document lists common error codes, error messages, and solutions encountered in PTO development.

## Contents

- [1. Compilation Errors (E001-E099)](#1-compilation-errors-e001-e099)
- [2. Linking Errors (L001-L099)](#2-linking-errors-l001-l099)
- [3. Runtime Errors (R001-R099)](#3-runtime-errors-r001-r099)
- [4. Memory Errors (M001-M099)](#4-memory-errors-m001-m099)
- [5. Numerical Errors (N001-N099)](#5-numerical-errors-n001-n099)
- [6. Performance Issues (P001-P099)](#6-performance-issues-p001-p099)
- [7. Framework Integration Errors (F001-F099)](#7-framework-integration-errors-f001-f099)

______________________________________________________________________

## 1. Compilation Errors (E001-E099)

### E001: Header File Not Found

**Error Message**:

```text
error: pto/pto-inst.hpp: No such file or directory
```

**Cause**: PTO library path not set

**Solution**:

```bash
# Method 1: Set environment variable
export PTO_LIB_PATH=/path/to/pto-isa

# Method 2: CMake specify
cmake -B build -DPTO_ROOT=/path/to/pto-isa

# Method 3: Manual include path
g++ -I/path/to/pto-isa/include src/my_operator.cpp
```

### E002: Static Assertion Failed - Tile Alignment

**Error Message**:

```text
static_assert failed: "Tile shape not aligned"
static_assert failed: "Tile width must be multiple of 16"
```

**Cause**: Tile dimensions don't meet alignment requirements

**Solution**:

```cpp
// ❌ Wrong: width 250 is not multiple of 16
using TileT = Tile<TileType::Vec, float, 16, 250>;

// ✅ Correct: width 256 is multiple of 16
using TileT = Tile<TileType::Vec, float, 16, 256>;

// Alignment requirements:
// - Vec Tile: width % 16 == 0
// - Cube Tile: height % 16 == 0 && width % 16 == 0
// - Acc Tile: height % 16 == 0 && width % 16 == 0
```

### E003: Type Mismatch

**Error Message**:

```text
error: no matching function for call to 'TADD(Tile<float>&, Tile<half>&)'
```

**Cause**: Tile types are inconsistent

**Solution**:

```cpp
// ❌ Wrong: type mismatch
Tile<TileType::Vec, float, 16, 256> tile_a;
Tile<TileType::Vec, half, 16, 256> tile_b;
TADD(tile_a, tile_a, tile_b);  // Error!

// ✅ Correct: consistent types
Tile<TileType::Vec, float, 16, 256> tile_a, tile_b, tile_c;
TADD(tile_c, tile_a, tile_b);  // Correct

// Or use type conversion
TCAST(tile_b_float, tile_b);  // half → float
TADD(tile_c, tile_a, tile_b_float);
```

### E004: C++ Standard Not Supported

**Error Message**:

```text
error: 'concept' does not name a type
error: expected ';' before 'requires'
```

**Cause**: Compiler doesn't support C++20

**Solution**:

```bash
# Check compiler version
g++ --version  # Need >= 13.0
clang++ --version  # Need >= 15.0

# Explicitly specify C++20
g++ -std=c++20 src/my_operator.cpp

# CMake setting
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

______________________________________________________________________

## 2. Linking Errors (L001-L099)

### L001: Undefined Reference

**Error Message**:

```text
undefined reference to `pto::TLOAD(...)`
undefined reference to `pto::TSTORE(...)`
```

**Cause**: PTO library not linked

**Solution**:

```bash
# Manual linking
g++ build/my_operator.o -L/path/to/pto/lib -lpto -o build/my_operator

# CMake configuration
target_link_libraries(my_operator PRIVATE PTO::pto)
```

### L002: Shared Library Not Found

**Error Message**:

```text
error while loading shared libraries: libpto.so: cannot open shared object file
```

**Cause**: Runtime cannot find shared library

**Solution**:

```bash
# Method 1: Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/pto/lib:$LD_LIBRARY_PATH

# Method 2: Add to system path
sudo echo "/path/to/pto/lib" > /etc/ld.so.conf.d/pto.conf
sudo ldconfig

# Method 3: Use RPATH
cmake -B build -DCMAKE_INSTALL_RPATH=/path/to/pto/lib

# Verify
ldd ./my_operator
```

______________________________________________________________________

## 3. Runtime Errors (R001-R099)

### R001: Kernel Launch Failed

**Error Message**:

```text
PTO_ERROR: Failed to launch kernel
Error code: -1
```

**Cause**: Kernel parameters incorrect or insufficient resources

**Solution**:

```cpp
// Check block_num
int block_num = get_available_cores();  // Don't exceed available cores
EXEC_KERNEL_CMD(MyKernel, block_num, ...);

// Check parameter types
// ❌ Wrong: passed wrong pointer type
EXEC_KERNEL_CMD(MyKernel, 24, int_ptr, ...);  // Expected float*

// ✅ Correct
EXEC_KERNEL_CMD(MyKernel, 24, float_ptr, ...);
```

### R002: Assertion Failed

**Error Message**:

```text
PTO_ASSERT failed: condition 'size <= MAX_SIZE'
File: my_operator.cpp, Line: 42
```

**Cause**: Runtime condition check failed

**Solution**:

```cpp
// Add input validation
void my_kernel(..., uint32_t size) {
  // Check size limit
  if (size > MAX_SIZE) {
    printf("Error: size %u exceeds MAX_SIZE %u\n", size, MAX_SIZE);
    return;
  }

  // Continue execution
  // ...
}
```

### R003: Null Pointer Dereference

**Error Message**:

```text
Segmentation fault (core dumped)
```

**Cause**: Accessed null pointer or invalid memory

**Solution**:

```cpp
// Add null pointer checks
void my_kernel(__gm__ float* out, __gm__ const float* in) {
  if (out == nullptr || in == nullptr) {
    printf("Error: null pointer\n");
    return;
  }

  // Continue execution
  // ...
}

// Use AddressSanitizer for detection
g++ -fsanitize=address src/my_operator.cpp
```

______________________________________________________________________

## 4. Memory Errors (M001-M099)

### M001: L1 Memory Overflow

**Error Message**:

```text
PTO_ASSERT: L1 memory overflow
Required: 600 KB, Available: 512 KB
```

**Cause**: Tile memory usage exceeds L1 capacity

**Solution**:

```cpp
// Method 1: Reduce Tile size
// ❌ Wrong: 16 × 512 × 4 bytes = 32 KB, multiple Tiles exceed L1
using TileT = Tile<TileType::Vec, float, 16, 512>;

// ✅ Correct: Reduce to 256
using TileT = Tile<TileType::Vec, float, 16, 256>;

// Method 2: Use double buffering
Event e1, e2;
TileT tile_a, tile_b;

TLOAD(tile_a, input[0:size], e1);
for (int i = 1; i < N; i++) {
  TLOAD(tile_b, input[i*size:size], e2);
  WAIT(e1);
  COMPUTE(tile_a);
  WAIT(e2);
  COMPUTE(tile_b);
  swap(e1, e2);
  swap(tile_a, tile_b);
}
```

### M002: Memory Alignment Error

**Error Message**:

```text
PTO_ASSERT: Memory address not aligned
Address: 0x12345678, Required alignment: 64
```

**Cause**: Memory address doesn't meet alignment requirements

**Solution**:

```cpp
// Use aligned_alloc
void* ptr = aligned_alloc(64, size);

// Or use C++17 aligned_new
float* ptr = new(std::align_val_t{64}) float[size];

// Check alignment
assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
```

______________________________________________________________________

## 5. Numerical Errors (N001-N099)

### N001: Numerical Precision Error

**Error Message**:

```text
Numerical error: max_diff = 1e-2
Expected: 1.0, Got: 1.01
```

**Cause**: Floating-point precision issues or algorithm errors

**Solution**:

```cpp
// Method 1: Use higher precision
// ❌ half (FP16): precision ~1e-3
using TileT = Tile<TileType::Vec, half, 16, 256>;

// ✅ float (FP32): precision ~1e-7
using TileT = Tile<TileType::Vec, float, 16, 256>;

// Method 2: Adjust tolerance
const float TOLERANCE = 1e-5;  // Adjust based on data type
assert(abs(result - expected) < TOLERANCE);
```

### N002: NaN or Inf

**Error Message**:

```text
Numerical error: NaN detected
Numerical error: Inf detected
```

**Cause**: Division by zero, overflow, or invalid operations

**Solution**:

```cpp
// Add numerical checks
void check_numerical_stability(const Tile& tile) {
  for (int i = 0; i < tile.size(); i++) {
    float val = tile[i];
    if (std::isnan(val)) {
      printf("NaN detected at index %d\n", i);
    }
    if (std::isinf(val)) {
      printf("Inf detected at index %d\n", i);
    }
  }
}

// Avoid division by zero
TADDS(denominator, denominator, 1e-8f);  // Add small constant
TDIV(result, numerator, denominator);

// Use safe math functions
TCLIP(tile, tile, -1e10f, 1e10f);  // Limit range
```

______________________________________________________________________

## 6. Performance Issues (P001-P099)

### P001: Performance Below Expectations

**Symptoms**: Operator runtime far exceeds expectations

**Diagnosis**:

```bash
# Use msprof for analysis
msprof --output=./profiling_data \
       --application="./my_operator" \
       --ai-core=on

# View report
msprof --export=on --output=./profiling_data
```

**Common Causes and Solutions**:

1. **Memory Access Bottleneck**

```cpp
// ❌ Problem: Frequent GM access
for (int i = 0; i < N; i++) {
  TLOAD(tile, input[i]);
  COMPUTE(tile);
  TSTORE(output[i], tile);
}

// ✅ Optimization: Batch loading
const int BATCH = 8;
for (int i = 0; i < N; i += BATCH) {
  TLOAD(tiles[0:BATCH], input[i:BATCH]);
  for (int j = 0; j < BATCH; j++) {
    COMPUTE(tiles[j]);
  }
  TSTORE(output[i:BATCH], tiles[0:BATCH]);
}
```

1. **Low Pipeline Efficiency**

```cpp
// ❌ Problem: Serial execution
TLOAD(tile, input);
WAIT_LOAD();
COMPUTE(tile);
WAIT_COMPUTE();
TSTORE(output, tile);

// ✅ Optimization: Pipeline parallelism
Event load_event, compute_event;
TLOAD(tile_a, input[0], load_event);
for (int i = 1; i < N; i++) {
  TLOAD(tile_b, input[i], load_event);
  WAIT(load_event);
  COMPUTE(tile_a, compute_event);
  WAIT(compute_event);
  TSTORE(output[i-1], tile_a);
  swap(tile_a, tile_b);
}
```

______________________________________________________________________

## 7. Framework Integration Errors (F001-F099)

### F001: PyTorch Operator Registration Failed

**Error Message**:

```text
RuntimeError: No such operator npu::my_add
```

**Cause**: Operator not properly registered

**Solution**:

```cpp
// Ensure proper registration
TORCH_LIBRARY_FRAGMENT(npu, m) {
  m.def("my_add(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("my_add", TORCH_FN(my_add_impl));
}

// Python verification
import torch
print(torch.ops.npu.my_add)  # Should display operator info
```

### F002: Device Type Mismatch

**Error Message**:

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, npu:0 and cpu!
```

**Cause**: Input tensors on different devices

**Solution**:

```python
# Ensure all inputs on same device
x = x.npu()
y = y.npu()
z = torch.ops.npu.my_add(x, y)

# Or check in operator
at::Tensor my_add_impl(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.device() == y.device(),
              "Inputs must be on same device");
  // ...
}
```

______________________________________________________________________

## References

- [Debugging Guide](debug.md)
- [Performance Optimization](opt.md)
- [Compilation Process](compilation-process.md)
- [Framework Integration](framework-integration.md)
- [Memory Optimization](memory-optimization.md)
