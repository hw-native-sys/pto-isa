# 常见错误码说明

本文档详细列出 PTO 开发中常见的错误码、错误信息及其解决方案，帮助开发者快速定位和解决问题。

## 目录

- [1. 编译错误 (E001-E099)](#1-%E7%BC%96%E8%AF%91%E9%94%99%E8%AF%AF-e001-e099)
- [2. 链接错误 (L001-L099)](#2-%E9%93%BE%E6%8E%A5%E9%94%99%E8%AF%AF-l001-l099)
- [3. 运行时错误 (R001-R099)](#3-%E8%BF%90%E8%A1%8C%E6%97%B6%E9%94%99%E8%AF%AF-r001-r099)
- [4. 内存错误 (M001-M099)](#4-%E5%86%85%E5%AD%98%E9%94%99%E8%AF%AF-m001-m099)
- [5. 数值错误 (N001-N099)](#5-%E6%95%B0%E5%80%BC%E9%94%99%E8%AF%AF-n001-n099)
- [6. 性能问题 (P001-P099)](#6-%E6%80%A7%E8%83%BD%E9%97%AE%E9%A2%98-p001-p099)
- [7. 框架集成错误 (F001-F099)](#7-%E6%A1%86%E6%9E%B6%E9%9B%86%E6%88%90%E9%94%99%E8%AF%AF-f001-f099)

______________________________________________________________________

## 1. 编译错误 (E001-E099)

### E001: 头文件未找到

**错误信息**：

```text
error: pto/pto-inst.hpp: No such file or directory
```

**原因**：PTO 库路径未正确设置

**解决方案**：

```bash
# 方法1：设置环境变量
export PTO_LIB_PATH=/path/to/pto-isa

# 方法2：CMake 指定
cmake -B build -DPTO_ROOT=/path/to/pto-isa

# 方法3：手动指定包含路径
g++ -I/path/to/pto-isa/include src/my_operator.cpp
```

### E002: 静态断言失败 - Tile 对齐

**错误信息**：

```text
static_assert failed: "Tile shape not aligned"
static_assert failed: "Tile width must be multiple of 16"
```

**原因**：Tile 尺寸不满足对齐要求

**解决方案**：

```cpp
// ❌ 错误：宽度 250 不是 16 的倍数
using TileT = Tile<TileType::Vec, float, 16, 250>;

// ✅ 正确：宽度 256 是 16 的倍数
using TileT = Tile<TileType::Vec, float, 16, 256>;

// 对齐要求：
// - Vec Tile: width % 16 == 0
// - Cube Tile: height % 16 == 0 && width % 16 == 0
// - Acc Tile: height % 16 == 0 && width % 16 == 0
```

### E003: 类型不匹配

**错误信息**：

```text
error: no matching function for call to 'TADD(Tile<float>&, Tile<half>&)'
```

**原因**：Tile 类型不一致

**解决方案**：

```cpp
// ❌ 错误：类型不匹配
Tile<TileType::Vec, float, 16, 256> tile_a;
Tile<TileType::Vec, half, 16, 256> tile_b;
TADD(tile_a, tile_a, tile_b);  // 错误！

// ✅ 正确：类型一致
Tile<TileType::Vec, float, 16, 256> tile_a, tile_b, tile_c;
TADD(tile_c, tile_a, tile_b);  // 正确

// 或使用类型转换
TCAST(tile_b_float, tile_b);  // half → float
TADD(tile_c, tile_a, tile_b_float);
```

### E004: C++ 标准版本不支持

**错误信息**：

```text
error: 'concept' does not name a type
error: expected ';' before 'requires'
```

**原因**：编译器不支持 C++20

**解决方案**：

```bash
# 检查编译器版本
g++ --version  # 需要 >= 13.0
clang++ --version  # 需要 >= 15.0

# 显式指定 C++20
g++ -std=c++20 src/my_operator.cpp

# CMake 设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

### E005: 模板参数错误

**错误信息**：

```text
error: template argument 3 is invalid
```

**原因**：Tile 模板参数不正确

**解决方案**：

```cpp
// Tile 模板参数：
// Tile<TileType, DataType, Height, Width>

// ❌ 错误：参数顺序错误
Tile<float, TileType::Vec, 16, 256> tile;

// ✅ 正确
Tile<TileType::Vec, float, 16, 256> tile;

// ❌ 错误：缺少参数
Tile<TileType::Vec, float> tile;

// ✅ 正确：提供所有参数
Tile<TileType::Vec, float, 16, 256> tile;
```

### E006: 宏定义冲突

**错误信息**：

```text
error: 'TILE_SIZE' was not declared in this scope
warning: 'TILE_SIZE' macro redefined
```

**原因**：宏定义冲突或未定义

**解决方案**：

```cpp
// 使用 constexpr 代替宏
constexpr int TILE_SIZE = 256;

// 或使用命名空间避免冲突
namespace my_op {
  constexpr int TILE_SIZE = 256;
}

// 检查宏是否已定义
#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif
```

______________________________________________________________________

## 2. 链接错误 (L001-L099)

### L001: 未定义的引用

**错误信息**：

```text
undefined reference to `pto::TLOAD(...)`
undefined reference to `pto::TSTORE(...)`
```

**原因**：未链接 PTO 库

**解决方案**：

```bash
# 手动链接
g++ build/my_operator.o -L/path/to/pto/lib -lpto -o build/my_operator

# CMake 配置
target_link_libraries(my_operator PRIVATE PTO::pto)
```

### L002: 找不到共享库

**错误信息**：

```text
error while loading shared libraries: libpto.so: cannot open shared object file
```

**原因**：运行时找不到共享库

**解决方案**：

```bash
# 方法1：设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/pto/lib:$LD_LIBRARY_PATH

# 方法2：添加到系统路径
sudo echo "/path/to/pto/lib" > /etc/ld.so.conf.d/pto.conf
sudo ldconfig

# 方法3：使用 RPATH
cmake -B build -DCMAKE_INSTALL_RPATH=/path/to/pto/lib

# 验证
ldd ./my_operator
```

### L003: 符号版本不匹配

**错误信息**：

```text
version `GLIBCXX_3.4.30' not found
```

**原因**：编译器版本与运行时库版本不匹配

**解决方案**：

```bash
# 检查可用版本
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX

# 更新编译器
sudo apt install g++-13

# 或使用静态链接
g++ -static-libstdc++ src/my_operator.cpp
```

______________________________________________________________________

## 3. 运行时错误 (R001-R099)

### R001: Kernel 启动失败

**错误信息**：

```text
PTO_ERROR: Failed to launch kernel
Error code: -1
```

**原因**：Kernel 参数错误或资源不足

**解决方案**：

```cpp
// 检查 block_num
int block_num = get_available_cores();  // 不要超过可用核心数
EXEC_KERNEL_CMD(MyKernel, block_num, ...);

// 检查参数类型
// ❌ 错误：传递了错误的指针类型
EXEC_KERNEL_CMD(MyKernel, 24, int_ptr, ...);  // 期望 float*

// ✅ 正确
EXEC_KERNEL_CMD(MyKernel, 24, float_ptr, ...);
```

### R002: 断言失败

**错误信息**：

```text
PTO_ASSERT failed: condition 'size <= MAX_SIZE'
File: my_operator.cpp, Line: 42
```

**原因**：运行时条件检查失败

**解决方案**：

```cpp
// 添加输入验证
void my_kernel(..., uint32_t size) {
  // 检查大小限制
  if (size > MAX_SIZE) {
    printf("Error: size %u exceeds MAX_SIZE %u\n", size, MAX_SIZE);
    return;
  }

  // 继续执行
  // ...
}
```

### R003: 空指针解引用

**错误信息**：

```text
Segmentation fault (core dumped)
```

**原因**：访问了空指针或无效内存

**解决方案**：

```cpp
// 添加空指针检查
void my_kernel(__gm__ float* out, __gm__ const float* in) {
  if (out == nullptr || in == nullptr) {
    printf("Error: null pointer\n");
    return;
  }

  // 继续执行
  // ...
}

// 使用 AddressSanitizer 检测
g++ -fsanitize=address src/my_operator.cpp
```

### R004: 数组越界

**错误信息**：

```text
AddressSanitizer: heap-buffer-overflow
```

**原因**：访问了数组边界之外的内存

**解决方案**：

```cpp
// 添加边界检查
for (int i = start; i < end; i += TILE_SIZE) {
  int actual_size = min(TILE_SIZE, end - i);  // 防止越界

  TLOAD(tile, GlobalTensor(in + i, actual_size));
  // ...
}
```

______________________________________________________________________

## 4. 内存错误 (M001-M099)

### M001: L1 内存溢出

**错误信息**：

```text
PTO_ASSERT: L1 memory overflow
Required: 600 KB, Available: 512 KB
```

**原因**：Tile 占用的 L1 内存超过容量

**解决方案**：

```cpp
// 方法1：减小 Tile 尺寸
// ❌ 错误：16 × 512 × 4 bytes = 32 KB，多个 Tile 超出 L1
using TileT = Tile<TileType::Vec, float, 16, 512>;

// ✅ 正确：减小到 256
using TileT = Tile<TileType::Vec, float, 16, 256>;

// 方法2：使用双缓冲
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

### M002: GM 内存不足

**错误信息**：

```text
Failed to allocate GM memory: size = 4 GB
```

**原因**：全局内存不足

**解决方案**：

```cpp
// 分块处理
const int CHUNK_SIZE = 1024 * 1024;  // 1M 元素
for (int offset = 0; offset < total_size; offset += CHUNK_SIZE) {
  int chunk_size = min(CHUNK_SIZE, total_size - offset);
  process_chunk(input + offset, output + offset, chunk_size);
}
```

### M003: 内存泄漏

**错误信息**：

```text
Memory leak detected: 1024 KB not freed
```

**原因**：动态分配的内存未释放

**解决方案**：

```cpp
// 使用 RAII
class TileBuffer {
 public:
  TileBuffer(size_t size) {
    data_ = new float[size];
  }

  ~TileBuffer() {
    delete[] data_;
  }

 private:
  float* data_;
};

// 或使用智能指针
std::unique_ptr<float[]> buffer(new float[size]);
```

### M004: 内存对齐错误

**错误信息**：

```text
PTO_ASSERT: Memory address not aligned
Address: 0x12345678, Required alignment: 64
```

**原因**：内存地址不满足对齐要求

**解决方案**：

```cpp
// 使用 aligned_alloc
void* ptr = aligned_alloc(64, size);

// 或使用 C++17 aligned_new
float* ptr = new(std::align_val_t{64}) float[size];

// 检查对齐
assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
```

______________________________________________________________________

## 5. 数值错误 (N001-N099)

### N001: 数值精度误差

**错误信息**：

```text
Numerical error: max_diff = 1e-2
Expected: 1.0, Got: 1.01
```

**原因**：浮点精度问题或算法误差

**解决方案**：

```cpp
// 方法1：使用更高精度
// ❌ half (FP16): 精度 ~1e-3
using TileT = Tile<TileType::Vec, half, 16, 256>;

// ✅ float (FP32): 精度 ~1e-7
using TileT = Tile<TileType::Vec, float, 16, 256>;

// 方法2：调整容差
const float TOLERANCE = 1e-5;  // 根据数据类型调整
assert(abs(result - expected) < TOLERANCE);

// 方法3：使用 Kahan 求和（减少累积误差）
float sum = 0.0f, c = 0.0f;
for (int i = 0; i < n; i++) {
  float y = data[i] - c;
  float t = sum + y;
  c = (t - sum) - y;
  sum = t;
}
```

### N002: NaN 或 Inf

**错误信息**：

```text
Numerical error: NaN detected
Numerical error: Inf detected
```

**原因**：除零、溢出或无效操作

**解决方案**：

```cpp
// 添加数值检查
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

// 避免除零
TADDS(denominator, denominator, 1e-8f);  // 添加小常数
TDIV(result, numerator, denominator);

// 使用安全的数学函数
TCLIP(tile, tile, -1e10f, 1e10f);  // 限制范围
```

### N003: 数值溢出

**错误信息**：

```text
Numerical overflow: value exceeds float32 range
```

**原因**：计算结果超出数据类型范围

**解决方案**：

```cpp
// 使用数值稳定的算法
// ❌ 不稳定：直接计算 exp
TEXP(result, x);  // x 很大时溢出

// ✅ 稳定：减去最大值
TROWMAX(max_val, x);
TROWEXPANDSUB(shifted, x, max_val);
TEXP(result, shifted);  // 不会溢出
```

______________________________________________________________________

## 6. 性能问题 (P001-P099)

### P001: 性能低于预期

**症状**：算子运行时间远超预期

**诊断**：

```bash
# 使用 msprof 分析
msprof --output=./profiling_data \
       --application="./my_operator" \
       --ai-core=on

# 查看报告
msprof --export=on --output=./profiling_data
```

**常见原因和解决方案**：

1. **内存访问瓶颈**

```cpp
// ❌ 问题：频繁访问 GM
for (int i = 0; i < N; i++) {
  TLOAD(tile, input[i]);
  COMPUTE(tile);
  TSTORE(output[i], tile);
}

// ✅ 优化：批量加载
const int BATCH = 8;
for (int i = 0; i < N; i += BATCH) {
  TLOAD(tiles[0:BATCH], input[i:BATCH]);
  for (int j = 0; j < BATCH; j++) {
    COMPUTE(tiles[j]);
  }
  TSTORE(output[i:BATCH], tiles[0:BATCH]);
}
```

1. **流水线效率低**

```cpp
// ❌ 问题：串行执行
TLOAD(tile, input);
WAIT_LOAD();
COMPUTE(tile);
WAIT_COMPUTE();
TSTORE(output, tile);

// ✅ 优化：流水线并行
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

### P002: 核心利用率低

**症状**：msprof 显示核心利用率 < 50%

**原因**：负载不均衡或同步开销大

**解决方案**：

```cpp
// 动态负载均衡
int block_idx = get_block_idx();
int block_num = get_block_num();

// ❌ 静态划分：可能不均衡
int chunk_size = total_size / block_num;
int start = block_idx * chunk_size;
int end = (block_idx + 1) * chunk_size;

// ✅ 动态划分：更均衡
int chunk_size = (total_size + block_num - 1) / block_num;
int start = block_idx * chunk_size;
int end = min(start + chunk_size, total_size);
```

### P003: 缓存未命中率高

**症状**：L1 缓存命中率 < 80%

**原因**：数据访问模式不友好

**解决方案**：

```cpp
// 优化数据访问模式
// ❌ 列优先访问（缓存不友好）
for (int j = 0; j < cols; j++) {
  for (int i = 0; i < rows; i++) {
    process(data[i * cols + j]);
  }
}

// ✅ 行优先访问（缓存友好）
for (int i = 0; i < rows; i++) {
  for (int j = 0; j < cols; j++) {
    process(data[i * cols + j]);
  }
}
```

______________________________________________________________________

## 7. 框架集成错误 (F001-F099)

### F001: PyTorch 算子注册失败

**错误信息**：

```text
RuntimeError: No such operator npu::my_add
```

**原因**：算子未正确注册

**解决方案**：

```cpp
// 确保正确注册
TORCH_LIBRARY_FRAGMENT(npu, m) {
  m.def("my_add(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("my_add", TORCH_FN(my_add_impl));
}

// Python 验证
import torch
print(torch.ops.npu.my_add)  # 应该显示算子信息
```

### F002: 设备类型不匹配

**错误信息**：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, npu:0 and cpu!
```

**原因**：输入张量在不同设备上

**解决方案**：

```python
# 确保所有输入在同一设备
x = x.npu()
y = y.npu()
z = torch.ops.npu.my_add(x, y)

# 或在算子内部检查
at::Tensor my_add_impl(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.device() == y.device(),
              "Inputs must be on same device");
  // ...
}
```

### F003: 梯度计算错误

**错误信息**：

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**原因**：未正确实现反向传播

**解决方案**：

```cpp
// 注册 autograd
TORCH_LIBRARY_IMPL(npu, Autograd, m) {
  m.impl("my_add", TORCH_FN(my_add_autograd));
}

// 实现反向传播
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& y) {
    return my_add_impl(x, y);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto grad = grad_outputs[0];
    return {grad, grad};  // ∂z/∂x = 1, ∂z/∂y = 1
  }
};
```

______________________________________________________________________

## 参考资源

- [算子调试指南](debug_zh.md)
- [性能优化指南](opt_zh.md)
- [编译流程详解](compilation-process_zh.md)
- [框架集成指南](framework-integration_zh.md)
- [内存优化技巧](memory-optimization_zh.md)
