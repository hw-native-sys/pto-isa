# Fused Add-ReLU-Mul 自定义算子示例

本示例展示如何从零开始开发一个 PTO 自定义算子，实现算子融合优化。

## 算子功能

**Fused Add-ReLU-Mul**：将三个逐元素操作融合为一个 kernel

```
out = ReLU(x + bias) * scale
```

等价于：
```python
# 步骤1：Add
temp = x + bias

# 步骤2：ReLU
temp = max(0, temp)

# 步骤3：Mul
out = temp * scale
```

## 融合优势

| 指标 | 融合前（3个kernel） | 融合后（1个kernel） | 提升 |
|------|-------------------|-------------------|------|
| **GM 访问次数** | 6次（3读+3写） | 2次（1读+1写） | 3× |
| **Kernel 启动开销** | 3次 | 1次 | 3× |
| **数据局部性** | 差（中间结果写回GM） | 好（保持在L1/L0） | ✓ |
| **预期性能提升** | - | 2-3× | - |

## 目录结构

```
fused_add_relu_mul/
├── fused_add_relu_mul_kernel.cpp  # PTO Kernel 实现（3个版本）
├── main.cpp                       # Host 测试程序
├── CMakeLists.txt                 # 构建配置
├── run.sh                         # 构建和运行脚本
└── README_zh.md                   # 本文档
```

## Kernel 实现版本

本示例提供了三个版本的实现，展示不同的优化技术：

### 1. 基础版本（FusedAddReLUMulKernel）

**特点**：
- 简单直接的实现
- 适合学习和理解基本概念
- 性能基线

**核心代码**：
```cpp
// 加载数据
TLOAD(tile_x, GlobalTensor(x + i));

// 融合计算
TADDS(tile_result, tile_x, bias);    // Add
TRELU(tile_result, tile_result);     // ReLU
TMULS(tile_result, tile_result, scale); // Mul

// 存储结果
TSTORE(GlobalTensor(out + i), tile_result);
```

**Tile 配置**：
- 尺寸：16×256 = 4096 元素
- 内存：16 KB
- 适用平台：A2/A3/A5

### 2. 优化版本（FusedAddReLUMulOptimizedKernel）

**特点**：
- 使用双缓冲技术
- 重叠数据加载和计算
- 提高流水线效率

**优化策略**：
```cpp
// 双缓冲 Tile
TileT tile_x[2];
TileT tile_result[2];

// 预加载第一批数据
load_event[0] = TLOAD(tile_x[0], ...);

for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
    int curr = tile_idx % 2;
    int next = (tile_idx + 1) % 2;

    // 预加载下一批（异步）
    if (tile_idx + 1 < num_tiles) {
        load_event[next] = TLOAD(tile_x[next], ...);
    }

    // 等待当前数据
    WAIT(load_event[curr]);

    // 计算当前数据
    TADDS(tile_result[curr], tile_x[curr], bias);
    TRELU(tile_result[curr], tile_result[curr]);
    TMULS(tile_result[curr], tile_result[curr], scale);

    // 存储结果
    TSTORE(..., tile_result[curr]);
}
```

**性能提升**：相比基础版本 1.5-2×

### 3. 大 Tile 版本（FusedAddReLUMulLargeTileKernel）

**特点**：
- 使用更大的 Tile 尺寸
- 减少循环迭代次数
- 适用于 A5 平台（L1 容量更大）

**Tile 配置**：
- 尺寸：32×512 = 16384 元素
- 内存：64 KB
- 适用平台：A5（L1 ~1MB/核）

**优势**：
- 更少的循环开销
- 更好的数据复用
- 更高的内存带宽利用率

## 构建和运行

### 前置条件

1. **CANN 环境**：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. **设置 SOC 版本**：
   - A2/A3：`Ascend910B1` 或 `Ascend910B2`
   - A5：`Ascend910_9599`

### 快速开始

#### 方法1：使用脚本（推荐）

```bash
# CPU 仿真模式（开发调试）
./run.sh --sim --clean

# NPU 模式（性能测试）
./run.sh --npu --soc Ascend910B1

# Debug 模式
./run.sh --sim --debug
```

#### 方法2：手动构建

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置（CPU 仿真）
cmake .. -DRUN_MODE=sim -DSOC_VERSION=Ascend910B1

# 配置（NPU）
cmake .. -DRUN_MODE=npu -DSOC_VERSION=Ascend910B1

# 编译
make -j$(nproc)

# 运行
./fused_add_relu_mul
```

### 运行输出示例

```
==========================================
Fused Add-ReLU-Mul Custom Operator Test
==========================================

========== Functional Tests ==========

========== Testing Basic Kernel (Small) ==========
Parameters: length=1024, bias=1.00, scale=2.00
Max difference: 1.234e-06
Error count: 0 / 1024 (0.00%)
✓ Basic Kernel (Small) PASSED

========== Testing Basic Kernel (Medium) ==========
Parameters: length=1048576, bias=1.00, scale=2.00
Max difference: 2.345e-06
Error count: 0 / 1048576 (0.00%)
✓ Basic Kernel (Medium) PASSED

========== Testing Optimized Kernel (Double Buffer) ==========
Parameters: length=1048576, bias=1.00, scale=2.00
Max difference: 2.345e-06
Error count: 0 / 1048576 (0.00%)
✓ Optimized Kernel (Double Buffer) PASSED

========== Performance Benchmarks ==========

========== Benchmarking Basic Kernel ==========
Parameters: length=16777216, iterations=100
Average time: 0.8234 ms
Throughput: 155.23 GB/s

========== Benchmarking Optimized Kernel (Double Buffer) ==========
Parameters: length=16777216, iterations=100
Average time: 0.4567 ms
Throughput: 279.84 GB/s

========================================
✓ All tests PASSED
```

## 代码详解

### 1. 多核并行划分

```cpp
// 获取核心信息
int block_idx = get_block_idx();  // 当前核心ID（0-23）
int block_num = get_block_num();  // 总核心数（24）

// 计算数据范围
int elements_per_block = (totalLength + block_num - 1) / block_num;
int start = block_idx * elements_per_block;
int end = start + elements_per_block;
```

**说明**：
- 使用 SPMD 模式（Single Program, Multiple Data）
- 每个核心处理不同的数据块
- 自动负载均衡

### 2. Tile 配置

```cpp
// 定义 Tile 类型
constexpr int TILE_H = 16;
constexpr int TILE_W = 256;
using TileT = Tile<TileType::Vec, float, TILE_H, TILE_W>;
```

**参数说明**：
- `TileType::Vec`：向量 Tile（用于逐元素操作）
- `float`：数据类型
- `16×256`：Tile 形状（4096 元素 = 16 KB）

### 3. 数据加载

```cpp
TLOAD(tile_x, GlobalTensor(x + i));
```

**说明**：
- `TLOAD`：从全局内存（GM）加载数据到 Tile
- `GlobalTensor`：全局内存张量包装器
- 自动处理 GM → L1 → L0 的数据搬运

### 4. 融合计算

```cpp
TADDS(tile_result, tile_x, bias);        // Add: tile + scalar
TRELU(tile_result, tile_result);         // ReLU: max(0, tile)
TMULS(tile_result, tile_result, scale);  // Mul: tile * scalar
```

**指令说明**：
- `TADDS`：Tile 与标量加法
- `TRELU`：Tile 的 ReLU 激活
- `TMULS`：Tile 与标量乘法
- 所有操作在 L0/L1 中完成，无需访问 GM

### 5. 数据存储

```cpp
TSTORE(GlobalTensor(out + i), tile_result);
```

**说明**：
- `TSTORE`：将 Tile 数据写回全局内存
- 自动处理 L0 → L1 → GM 的数据搬运

## 性能优化技巧

### 1. 选择合适的 Tile 尺寸

**原则**：
- 平衡片上容量和数据复用
- A2/A3：单个 Tile 通常 2-32 KB
- A5：单个 Tile 可以更大（4-64 KB）

**示例**：
```cpp
// A2/A3 推荐
using TileT = Tile<TileType::Vec, float, 16, 256>;  // 16 KB

// A5 推荐
using TileT = Tile<TileType::Vec, float, 32, 512>;  // 64 KB
```

### 2. 使用双缓冲

**优势**：
- 重叠数据加载和计算
- 提高流水线效率
- 减少等待时间

**实现**：
```cpp
TileT tile[2];  // 双缓冲
Event event[2]; // 同步事件

// 预加载
event[0] = TLOAD(tile[0], ...);

for (int i = 0; i < N; i++) {
    int curr = i % 2;
    int next = (i + 1) % 2;

    // 加载下一批（异步）
    if (i + 1 < N) {
        event[next] = TLOAD(tile[next], ...);
    }

    // 等待当前批
    WAIT(event[curr]);

    // 计算当前批
    COMPUTE(tile[curr]);
}
```

### 3. 减少同步开销

**不好的做法**：
```cpp
for (int i = 0; i < N; i++) {
    TLOAD(tile, ...);
    TSYNC();  // 全局同步，开销大
    COMPUTE(tile);
}
```

**好的做法**：
```cpp
Event e;
for (int i = 0; i < N; i++) {
    e = TLOAD(tile, ...);
    COMPUTE(tile, e);  // 只等待 TLOAD 完成
}
```

### 4. 算子融合

**融合前**：
```cpp
// 3 个独立 kernel
kernel_add<<<...>>>(temp, x, bias);
kernel_relu<<<...>>>(temp2, temp);
kernel_mul<<<...>>>(out, temp2, scale);
```

**融合后**：
```cpp
// 1 个融合 kernel
fused_kernel<<<...>>>(out, x, bias, scale);
```

**收益**：
- 内存访问：6次 → 2次（3×）
- Kernel 启动：3次 → 1次（3×）
- 性能提升：2-3×

## 扩展练习

### 练习1：添加更多融合操作

尝试融合更多操作，例如：
```cpp
// Fused Add-ReLU-Mul-Sigmoid
out = Sigmoid(ReLU(x + bias) * scale)
```

**提示**：
```cpp
TADDS(tile, tile, bias);
TRELU(tile, tile);
TMULS(tile, tile, scale);
// TODO: 添加 Sigmoid
// Sigmoid(x) = 1 / (1 + exp(-x))
```

### 练习2：支持更多数据类型

添加 FP16 支持：
```cpp
template<typename T>
__global__ __aicore__ void FusedAddReLUMulKernel(
    __gm__ T* out,
    __gm__ const T* x,
    T bias,
    T scale,
    uint32_t totalLength)
{
    // 使用模板参数 T
    using TileT = Tile<TileType::Vec, T, 16, 256>;
    // ...
}
```

### 练习3：添加 Mask 支持

支持条件计算：
```cpp
// 只对 mask == 1 的元素进行计算
__global__ __aicore__ void FusedAddReLUMulMaskedKernel(
    __gm__ float* out,
    __gm__ const float* x,
    __gm__ const bool* mask,
    float bias,
    float scale,
    uint32_t totalLength)
{
    // TODO: 实现 mask 逻辑
}
```

### 练习4：集成到 PyTorch

参考 `demos/baseline/add` 示例，将此算子集成到 PyTorch：
```python
import torch
import torch_npu

# 调用自定义算子
out = torch.ops.npu.fused_add_relu_mul(x, bias, scale)
```

## 常见问题

### Q1: 编译错误 "Cannot find ASCEND_HOME_PATH"

**解决方案**：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Q2: 运行时错误 "Tile shape not aligned"

**原因**：Tile 尺寸不满足对齐要求

**解决方案**：
```cpp
// 确保宽度是 16 的倍数
using TileT = Tile<TileType::Vec, float, 16, 256>;  // ✓
using TileT = Tile<TileType::Vec, float, 16, 250>;  // ✗
```

### Q3: 性能不如预期

**检查清单**：
- [ ] 是否使用了双缓冲？
- [ ] Tile 尺寸是否合适？
- [ ] 是否有不必要的同步？
- [ ] 是否充分利用了多核？

**性能分析**：
```bash
# 使用 msprof 分析性能
msprof --application="./fused_add_relu_mul" --output=./profiling_data
```

## 参考资源

- [PTO 编程模型](../../../docs/coding/ProgrammingModel_zh.md)
- [算子融合技术](../../../docs/coding/operator-fusion_zh.md)
- [性能优化指南](../../../docs/coding/opt_zh.md)
- [流水线与并行执行](../../../docs/coding/pipeline-parallel_zh.md)
- [Add 算子示例](../../../demos/baseline/add/README_zh.md)

## 许可证

本项目基于 CANN Open Software License Agreement Version 2.0 进行许可。
