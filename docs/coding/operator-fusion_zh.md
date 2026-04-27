# 算子融合技术

本文档深入介绍 PTO 算子融合技术，帮助开发者通过融合多个算子减少内存访问，提升整体性能。

## 目录

- [1. 算子融合概述](#1-算子融合概述)
- [2. 融合模式分类](#2-融合模式分类)
- [3. 融合实现技术](#3-融合实现技术)
- [4. 融合收益分析](#4-融合收益分析)
- [5. 融合策略选择](#5-融合策略选择)
- [6. 实战案例](#6-实战案例)
- [7. 最佳实践](#7-最佳实践)

---

## 1. 算子融合概述

### 1.1 什么是算子融合

**定义**：将多个独立的算子合并为一个算子，在片上内存中完成所有计算，减少中间结果的 GM 存储和读取。

**核心思想**：
```
传统方式：
  Kernel1: GM → L1 → Compute → L1 → GM
  Kernel2: GM → L1 → Compute → L1 → GM
  Kernel3: GM → L1 → Compute → L1 → GM

融合方式：
  FusedKernel: GM → L1 → Compute1 → Compute2 → Compute3 → L1 → GM
```

### 1.2 融合的优势

#### 优势1：减少内存访问

**示例：Add + ReLU + Mul**
```cpp
// 融合前：3 个独立算子
y = Add(x, bias);      // Load x, Store y
z = ReLU(y);           // Load y, Store z
out = Mul(z, scale);   // Load z, Store out

// 内存访问统计：
// - Load: 3 次（x, y, z）
// - Store: 3 次（y, z, out）
// - 总计: 6 次 GM 访问

// 融合后：1 个融合算子
out = FusedAddReLUMul(x, bias, scale);

// 内存访问统计：
// - Load: 1 次（x）
// - Store: 1 次（out）
// - 总计: 2 次 GM 访问

// 内存访问减少：(6 - 2) / 6 = 67%
```

#### 优势2：减少内核启动开销

**内核启动开销**：
- 每次内核启动：~10-50 μs
- 3 个独立内核：30-150 μs
- 1 个融合内核：10-50 μs
- 节省：20-100 μs

#### 优势3：提高数据局部性

**缓存命中率提升**：
```
融合前：
  - 中间结果写回 GM
  - 可能被其他核心的数据驱逐出缓存
  - 下一个算子重新加载（缓存未命中）

融合后：
  - 中间结果保持在 L1
  - 无缓存驱逐
  - 100% 缓存命中
```

### 1.3 融合的挑战

**挑战1：片上内存限制**
```cpp
// 问题：融合后可能超出 L1 容量
// A2/A3: L1 ~512 KB/核
// A5: L1 ~1 MB/核

// 示例：3 个算子各需 200 KB
// 融合前：每个算子独立运行，200 KB < 512 KB ✓
// 融合后：需要 600 KB > 512 KB ✗
```

**挑战2：数据依赖复杂**
```cpp
// 问题：中间结果被多次使用
y = Add(x, bias);
z1 = ReLU(y);    // 使用 y
z2 = Sigmoid(y); // 再次使用 y
// 无法简单融合，需要保留 y
```

**挑战3：计算密集型算子**
```cpp
// 问题：计算时间远大于内存访问时间
// GEMM: 计算密集型
// - 计算时间：100 ms
// - 内存访问时间：10 ms
// - 融合收益：< 10%（不值得）
```

---

## 2. 融合模式分类

### 2.1 逐元素融合（Element-wise Fusion）

**特点**：
- 所有算子都是逐元素操作
- 无数据依赖
- 融合最简单，收益最大

**常见模式**：
```cpp
// 模式1：Add + ReLU
out = ReLU(Add(x, bias));

// 模式2：Add + ReLU + Mul
out = Mul(ReLU(Add(x, bias)), scale);

// 模式3：Add + BatchNorm + ReLU
out = ReLU(BatchNorm(Add(x, bias)));

// 模式4：Mul + Add + Sigmoid
out = Sigmoid(Add(Mul(x, scale), bias));
```

**实现示例**：
```cpp
__global__ __aicore__ void FusedAddReLUMul(
    __gm__ float* out,
    __gm__ const float* in,
    float bias,
    float scale,
    uint32_t length) {
  
  int block_idx = get_block_idx();
  int block_num = get_block_num();
  
  // 计算当前核心负责的数据范围
  int elements_per_block = (length + block_num - 1) / block_num;
  int start = block_idx * elements_per_block;
  int end = min(start + elements_per_block, length);
  
  using TileT = Tile<TileType::Vec, float, 16, 256>;
  
  for (int i = start; i < end; i += 16 * 256) {
    int size = min(16 * 256, end - i);
    
    TileT tile;
    
    // 加载数据
    TLOAD(tile, GlobalTensor(in + i));
    
    // 融合计算：Add + ReLU + Mul
    TADDS(tile, tile, bias);    // Add
    TRELU(tile, tile);          // ReLU
    TMULS(tile, tile, scale);   // Mul
    
    // 存储结果
    TSTORE(GlobalTensor(out + i), tile);
  }
}
```

**性能分析**：
```
数据量：1M 元素（4 MB）
平台：A3（24 核）

融合前：
- Add: 0.05 ms
- ReLU: 0.05 ms
- Mul: 0.05 ms
- 总计: 0.15 ms

融合后：
- FusedAddReLUMul: 0.05 ms

加速比：3×
```

### 2.2 归约融合（Reduction Fusion）

**特点**：
- 包含归约操作（sum, max, min）
- 需要保留归约结果
- 融合中等复杂度

**常见模式**：
```cpp
// 模式1：Softmax
// max → sub → exp → sum → div
out = exp(x - max(x)) / sum(exp(x - max(x)))

// 模式2：LayerNorm
// mean → sub → square → mean → sqrt → div
out = (x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps)

// 模式3：RMSNorm
// square → mean → sqrt → div
out = x / sqrt(mean(x^2) + eps)
```

**Softmax 融合实现**：
```cpp
__global__ __aicore__ void FusedSoftmax(
    __gm__ float* out,
    __gm__ const float* in,
    int rows,
    int cols) {
  
  int block_idx = get_block_idx();
  
  // 每个核心处理一行
  if (block_idx >= rows) return;
  
  using TileVec = Tile<TileType::Vec, float, 1, 256>;
  using TileScalar = Tile<TileType::Vec, float, 1, 1>;
  
  TileVec input, shifted, exp_vals, output;
  TileScalar max_val, sum_val;
  
  for (int col = 0; col < cols; col += 256) {
    int size = min(256, cols - col);
    
    // 加载输入
    TLOAD(input, in[block_idx * cols + col : size]);
    
    // 步骤1：计算最大值
    TROWMAX(max_val, input);
    
    // 步骤2：减去最大值（数值稳定）
    TROWEXPANDSUB(shifted, input, max_val);
    
    // 步骤3：计算指数
    TEXP(exp_vals, shifted);
    
    // 步骤4：计算和
    TROWSUM(sum_val, exp_vals);
    
    // 步骤5：归一化
    TROWEXPANDDIV(output, exp_vals, sum_val);
    
    // 存储结果
    TSTORE(out[block_idx * cols + col : size], output);
  }
}
```

**性能分析**：
```
数据量：1024 × 1024（4 MB）
平台：A3（24 核）

融合前（5 个独立算子）：
- RowMax: 0.08 ms
- Sub: 0.05 ms
- Exp: 0.12 ms
- RowSum: 0.08 ms
- Div: 0.05 ms
- 总计: 0.38 ms

融合后：
- FusedSoftmax: 0.15 ms

加速比：2.5×
```

### 2.3 矩阵融合（Matrix Fusion）

**特点**：
- 包含矩阵乘法
- 融合后处理（Bias, Activation）
- 高性能收益

**常见模式**：
```cpp
// 模式1：GEMM + Bias
out = MatMul(A, B) + bias

// 模式2：GEMM + Bias + ReLU
out = ReLU(MatMul(A, B) + bias)

// 模式3：GEMM + Bias + GELU
out = GELU(MatMul(A, B) + bias)

// 模式4：GEMM + Residual + LayerNorm
out = LayerNorm(MatMul(A, B) + residual)
```

**GEMM + Bias + ReLU 融合实现**：
```cpp
__global__ __aicore__ void FusedGEMMBiasReLU(
    __gm__ float* C,
    __gm__ const float* A,
    __gm__ const float* B,
    __gm__ const float* bias,
    int M, int K, int N) {
  
  int block_idx = get_block_idx();
  
  // 2D 划分
  int blocks_n = (N + TILE_N - 1) / TILE_N;
  int block_m = block_idx / blocks_n;
  int block_n = block_idx % blocks_n;
  
  int m_start = block_m * TILE_M;
  int n_start = block_n * TILE_N;
  
  if (m_start >= M || n_start >= N) return;
  
  using TileLeft = TileLeft<half, 128, 64>;
  using TileRight = TileRight<half, 64, 256>;
  using TileAcc = TileAcc<float, 128, 256>;
  using TileBias = Tile<TileType::Vec, float, 1, 256>;
  
  TileAcc acc;
  TFILL(acc, 0);
  
  // 矩阵乘法
  for (int k = 0; k < K; k += 64) {
    TileLeft tileA;
    TileRight tileB;
    
    TLOAD(tileA, A[m_start:128, k:64]);
    TLOAD(tileB, B[k:64, n_start:256]);
    TMATMUL_ACC(acc, tileA, tileB);
  }
  
  // 融合 Bias
  TileBias bias_tile;
  TLOAD(bias_tile, bias[n_start:256]);
  TROWEXPANDADD(acc, acc, bias_tile);
  
  // 融合 ReLU
  TRELU(acc, acc);
  
  // 存储结果
  TSTORE(C[m_start:128, n_start:256], acc);
}
```

**性能分析**：
```
矩阵大小：1024 × 1024 × 1024
平台：A3（24 核）

融合前：
- GEMM: 2.5 ms
- Bias: 0.05 ms
- ReLU: 0.05 ms
- 总计: 2.6 ms

融合后：
- FusedGEMMBiasReLU: 2.52 ms

加速比：1.03×（收益较小，但避免了额外的内核启动）
```

### 2.4 复杂融合（Complex Fusion）

**特点**：
- 多种操作类型混合
- 复杂的数据流
- 需要精心设计

**示例：Fused Multi-Head Attention**
```cpp
// QKV 投影 + Softmax + 输出投影
// Q = Linear(x)
// K = Linear(x)
// V = Linear(x)
// Attention = Softmax(Q @ K^T / sqrt(d))
// Out = Attention @ V
// Result = Linear(Out)
```

---

## 3. 融合实现技术

### 3.1 手动融合步骤

**步骤1：识别融合机会**
```python
# 分析计算图
def analyze_fusion_opportunities(graph):
    candidates = []
    
    for node in graph.nodes:
        # 查找连续的逐元素操作
        if is_elementwise(node):
            chain = find_elementwise_chain(node)
            if len(chain) >= 2:
                candidates.append(chain)
    
    return candidates
```

**步骤2：验证融合可行性**
```cpp
// 检查清单
bool can_fuse(Op op1, Op op2) {
  // 1. 检查数据依赖
  if (op2.input != op1.output) return false;
  
  // 2. 检查中间结果是否被其他算子使用
  if (op1.output.num_users > 1) return false;
  
  // 3. 检查片上内存容量
  size_t required_memory = op1.memory + op2.memory;
  if (required_memory > L1_CAPACITY) return false;
  
  // 4. 检查数据类型兼容性
  if (op1.output_type != op2.input_type) return false;
  
  return true;
}
```

**步骤3：实现融合 kernel**
```cpp
// 模板化融合 kernel
template<typename Op1, typename Op2, typename Op3>
__global__ __aicore__ void FusedKernel(
    __gm__ float* out,
    __gm__ const float* in,
    Op1 op1, Op2 op2, Op3 op3) {
  
  using TileT = Tile<TileType::Vec, float, 16, 256>;
  TileT tile;
  
  TLOAD(tile, in);
  
  // 依次执行融合的操作
  op1(tile, tile);
  op2(tile, tile);
  op3(tile, tile);
  
  TSTORE(out, tile);
}
```

**步骤4：性能验证**
```cpp
// 对比融合前后的性能
void benchmark_fusion() {
  // 融合前
  auto start = GetTime();
  kernel1<<<...>>>();
  kernel2<<<...>>>();
  kernel3<<<...>>>();
  auto time_unfused = GetTime() - start;
  
  // 融合后
  start = GetTime();
  fused_kernel<<<...>>>();
  auto time_fused = GetTime() - start;
  
  float speedup = time_unfused / time_fused;
  printf("Speedup: %.2fx\n", speedup);
}
```

### 3.2 编译器自动融合

**未来特性：PTO Tile Fusion**
```cpp
// 使用 pragma 指示编译器融合
#pragma pto_fusion_begin
{
  TADD(y, x, bias);
  TRELU(z, y);
  TMUL(out, z, scale);
}
#pragma pto_fusion_end

// 编译器自动：
// 1. 分析数据依赖
// 2. 检查融合可行性
// 3. 生成融合 kernel
// 4. 优化内存访问
```

---

## 4. 融合收益分析

### 4.1 理论收益计算

**公式**：
```
加速比 = T_unfused / T_fused

其中：
T_unfused = Σ(T_compute_i + T_memory_i + T_launch_i)
T_fused = T_compute_fused + T_memory_fused + T_launch_fused

通常：
T_compute_fused ≈ Σ T_compute_i（计算时间不变）
T_memory_fused << Σ T_memory_i（内存访问大幅减少）
T_launch_fused << Σ T_launch_i（启动开销减少）
```

**示例计算**：
```
Add + ReLU + Mul 融合：

融合前：
- Add: 0.01 ms (compute) + 0.04 ms (memory) + 0.02 ms (launch) = 0.07 ms
- ReLU: 0.01 ms + 0.04 ms + 0.02 ms = 0.07 ms
- Mul: 0.01 ms + 0.04 ms + 0.02 ms = 0.07 ms
- 总计: 0.21 ms

融合后：
- Compute: 0.03 ms（3 个操作）
- Memory: 0.04 ms（只加载和存储一次）
- Launch: 0.02 ms（只启动一次）
- 总计: 0.09 ms

加速比: 0.21 / 0.09 = 2.3×
```

### 4.2 实际收益测量

**测量方法**：
```cpp
// 使用 msprof 测量
void measure_fusion_benefit() {
  // 1. 测量融合前
  msprof_start();
  run_unfused_kernels();
  auto metrics_unfused = msprof_stop();

  // 2. 测量融合后
  msprof_start();
  run_fused_kernel();
  auto metrics_fused = msprof_stop();

  // 3. 分析收益
  printf("Memory access reduction: %.1f%%\n",
         100.0 * (1 - metrics_fused.memory_bytes /
                      metrics_unfused.memory_bytes));

  printf("Kernel launch reduction: %d → %d\n",
         metrics_unfused.num_kernels,
         metrics_fused.num_kernels);

  printf("Overall speedup: %.2fx\n",
         metrics_unfused.time / metrics_fused.time);
}
```

---

## 5. 融合策略选择

### 5.1 决策树

```
是否融合？
├─ 中间结果只使用一次？
│   ├─ 是 → 继续
│   └─ 否 → 不融合
├─ 融合后不超出 L1 容量？
│   ├─ 是 → 继续
│   └─ 否 → 不融合或部分融合
├─ 预期加速比 > 1.2×？
│   ├─ 是 → 融合
│   └─ 否 → 不融合
```

### 5.2 融合优先级

**高优先级（强烈推荐融合）**：
1. 多个逐元素操作
2. Softmax 类归约操作
3. BatchNorm + Activation
4. GEMM + Bias + Activation

**中优先级（视情况融合）**：
1. LayerNorm + 后续操作
2. Attention 内部操作
3. 卷积 + Bias + Activation

**低优先级（通常不融合）**：
1. 大型矩阵乘法（已经计算密集）
2. 中间结果被多次使用
3. 融合后超出片上内存

---

## 6. 实战案例

### 案例1：ResNet Block 融合

**原始实现**：
```python
# ResNet Block
def resnet_block(x, weight1, weight2, bias1, bias2):
    # Conv1 + BN1 + ReLU
    y = conv2d(x, weight1)
    y = batch_norm(y)
    y = relu(y)
    
    # Conv2 + BN2
    y = conv2d(y, weight2)
    y = batch_norm(y)
    
    # Residual + ReLU
    y = y + x
    y = relu(y)
    
    return y
```

**融合策略**：
```cpp
// 融合1：Conv + BN + ReLU
__global__ void FusedConvBNReLU(...) {
  // Conv
  TMATMUL(output, input, weight);
  
  // BN（融合）
  TROWEXPANDSUB(output, output, mean);
  TROWEXPANDDIV(output, output, std);
  TROWEXPANDMUL(output, output, gamma);
  TROWEXPANDADD(output, output, beta);
  
  // ReLU（融合）
  TRELU(output, output);
}

// 融合2：Add + ReLU
__global__ void FusedAddReLU(...) {
  TADD(output, y, residual);
  TRELU(output, output);
}
```

**性能提升**：
```
原始：8 个 kernel，1.2 ms
融合后：4 个 kernel，0.7 ms
加速比：1.7×
```

### 案例2：Transformer Layer 融合

**原始实现**：
```python
# Transformer Layer
def transformer_layer(x, Wq, Wk, Wv, Wo):
    # QKV 投影
    Q = linear(x, Wq)
    K = linear(x, Wk)
    V = linear(x, Wv)
    
    # Attention
    scores = matmul(Q, K.T) / sqrt(d)
    attn = softmax(scores)
    out = matmul(attn, V)
    
    # 输出投影
    out = linear(out, Wo)
    
    return out
```

**融合策略**：
```cpp
// 融合1：QKV 投影（3 个 GEMM 合并）
__global__ void FusedQKVProjection(...) {
  // 一次性计算 Q, K, V
  TMATMUL(Q, x, Wq);
  TMATMUL(K, x, Wk);
  TMATMUL(V, x, Wv);
}

// 融合2：Attention Score + Softmax
__global__ void FusedAttentionSoftmax(...) {
  // Score
  TMATMUL(scores, Q, K_T);
  
  // Scale（融合）
  TMULS(scores, scores, 1.0 / sqrt(d));
  
  // Softmax（融合）
  TROWMAX(max_val, scores);
  TROWEXPANDSUB(shifted, scores, max_val);
  TEXP(exp_vals, shifted);
  TROWSUM(sum_val, exp_vals);
  TROWEXPANDDIV(attn, exp_vals, sum_val);
}
```

**性能提升**：
```
原始：12 个 kernel，3.5 ms
融合后：6 个 kernel，2.1 ms
加速比：1.7×
```

---

## 7. 最佳实践

### 7.1 融合设计原则

✅ **DO**：
- 优先融合逐元素操作
- 融合 Softmax 等归约操作
- 在 GEMM 后融合 Bias 和 Activation
- 保持融合 kernel 简单易懂
- 测量实际性能收益

❌ **DON'T**：
- 不要融合中间结果被多次使用的算子
- 不要融合导致 L1 溢出的操作
- 不要过度融合（保持可维护性）
- 不要假设融合一定更快（需要测量）

### 7.2 融合检查清单

**融合前检查**：
- [ ] 中间结果只使用一次
- [ ] 融合后不超出 L1 容量
- [ ] 数据类型兼容
- [ ] 无复杂的控制流

**融合后验证**：
- [ ] 数值正确性验证
- [ ] 性能提升 > 20%
- [ ] 代码可维护性良好
- [ ] 建立性能回归测试

### 7.3 调试技巧

**验证正确性**：
```cpp
// 对比融合前后的输出
void verify_fusion() {
  // 运行融合前的版本
  run_unfused_kernels(input, output_ref);
  
  // 运行融合后的版本
  run_fused_kernel(input, output_test);
  
  // 对比结果
  float max_diff = 0;
  for (int i = 0; i < size; i++) {
    float diff = abs(output_ref[i] - output_test[i]);
    max_diff = max(max_diff, diff);
  }
  
  assert(max_diff < 1e-5);
  printf("Fusion verified: max_diff = %.2e\n", max_diff);
}
```

---

## 参考资源

- [性能优化指南](opt_zh.md)
- [内存优化技巧](memory-optimization_zh.md)
- [性能调优最佳实践](performance-best-practices_zh.md)
- [流水线与并行执行](pipeline-parallel_zh.md)
