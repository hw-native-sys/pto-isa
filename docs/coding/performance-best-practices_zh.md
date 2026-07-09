# 性能调优最佳实践

本文档总结 PTO 算子性能调优的常见方法。文中的数值示例仅用于帮助分析，不应视为固定的平台指标；实际可达性能取决于芯片代际、频率、存储层次、编译器行为、工作负载形态以及外部运行时环境。

---

## 1. 优化流程

### 1.1 标准优化流程

```
正确性验证 → 性能基线 → 瓶颈分析 → 针对性优化 → 验证 → 迭代
```

**详细步骤**：

#### 步骤 1：确保正确性
```bash
# CPU 仿真验证
python3 tests/run_cpu.py --testcase your_op --verbose

# NPU 验证
python3 tests/script/run_st.py -r npu -v a3 -t your_op
```

**检查点**：
- ✅ 数值误差 < 1e-5（fp32）或 < 1e-3（fp16）
- ✅ 所有测试用例通过
- ✅ 边界条件正确处理

#### 步骤 2：建立性能基线
```bash
# 使用 msprof 采集性能数据
msprof --application="your_app" --output=./baseline
```

**记录指标**：
- 总执行时间
- 各阶段时间占比（TLOAD/TMATMUL/TSTORE）
- 内存带宽利用率
- 计算单元利用率

#### 步骤 3：识别瓶颈

**分析 profiler 输出**：
```
TLOAD:    45%  ← 内存搬运
TEXTRACT: 10%  ← 布局转换
TMATMUL:  40%  ← 计算
TSTORE:    5%  ← 写回
```

**瓶颈类型**：
- **内存受限**：TLOAD/TSTORE 占比 > 60%
- **计算受限**：TMATMUL 占比 > 70%
- **转换受限**：TEXTRACT/TMOV 占比 > 20%

#### 步骤 4：针对性优化

根据瓶颈类型选择优化策略（见后续章节）。

#### 步骤 5：验证优化效果

**对比指标**：
- 性能提升百分比
- 各阶段时间变化
- 数值正确性保持

#### 步骤 6：迭代优化

重复步骤 3-5，直到达到性能目标或优化空间耗尽。

---

## 2. 性能分析方法

### 2.1 使用 msprof 工具

**基础用法**：
```bash
# 采集性能数据
msprof --application="./your_app" \
       --output=./profiling_data \
       --ai-core=on \
       --task-time=on

# 查看报告
msprof --export=on \
       --output=./profiling_data
```

**关键指标**：

| 指标 | 含义 | 目标值 |
|------|------|--------|
| **TMATMUL 占比** | Cube 单元利用率 | > 50% |
| **TLOAD 占比** | 内存搬运时间 | < 40% |
| **MTE 带宽** | 内存带宽利用率 | > 70% |
| **流水线气泡** | 空闲时间 | < 10% |

### 2.2 手动计时

在关键路径插入计时代码：

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// 关键代码段
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
}

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
printf("TLOAD time: %ld us\n", duration.count());
```

### 2.3 理论性能估算

理论峰值只适合做粗粒度上界分析。更可靠的做法是在相同测试条件下对比不同实现，并优先依据 profiler 数据判断瓶颈位置。

**建议的分析方式**：
```
理论吞吐上界 = 峰值算力 × 估计利用率
实际吞吐     = 实测工作负载 FLOPs / 实测执行时间
```

利用该对比判断内核更偏向算力受限还是带宽受限；除非数据来自目标平台的正式规格说明，否则不要把固定平台数值直接写成设计结论。

---

## 3. 常见性能问题

### 3.1 内存带宽受限

**症状**：
- TLOAD/TSTORE 占比 > 60%
- TMATMUL 占比 < 30%

**原因**：
- Tile 太小，数据复用不足
- 频繁的 GM ↔ L1 搬运
- 未使用流水线重叠

**解决方案**：

✅ **增大 Tile 尺寸**
```cpp
// 优化前：小 Tile
using TileT = Tile<TileType::Vec, float, 8, 64>;  // 2KB

// 优化后：大 Tile
using TileT = Tile<TileType::Vec, float, 16, 256>; // 16KB
```

✅ **提升数据复用**
```cpp
// GEMM：K 维度分块
for (int k = 0; k < K; k += TILE_K) {
  TLOAD(tileA, ...);  // 加载一次
  TLOAD(tileB, ...);  // 加载一次
  TMATMUL(acc, tileA, tileB);  // 复用多次
}
```

✅ **在适用时使用双缓冲或分阶段重叠**
```cpp
// 预加载
TLOAD(tile[0], ...);

for (int i = 0; i < N; i++) {
  int curr = i % 2;
  int next = (i + 1) % 2;

  // 处理当前 Tile
  process_tile(result[curr], tile[curr]);

  // 条件允许时并行加载下一个 Tile
  if (i + 1 < N) {
    TLOAD(tile[next], ...);
  }
}
```

### 3.2 计算单元利用率低

**症状**：
- TMATMUL 占比 < 40%
- 大量流水线气泡

**原因**：
- 数据搬运跟不上计算速度
- 同步过于频繁
- Tile 形状不匹配硬件

**解决方案**：

✅ **优化流水线重叠**
```cpp
// 使用事件而非全局同步
Event<Op::TLOAD, Op::TMATMUL> e;
e = TLOAD(tile, ...);
TMATMUL(acc, tile, ..., e);  // 只等待 TLOAD
```

✅ **调整 Tile 形状**
```cpp
// A2/A3 推荐：
// Left: 128×64, Right: 64×256, Acc: 128×256

// A5 推荐：
// Left: 256×128, Right: 128×512, Acc: 256×512
```

### 3.3 布局转换开销大

**症状**：
- TEXTRACT/TMOV 占比 > 20%
- TMATMUL 占比正常但总性能差

**原因**：
- 频繁的布局转换
- 输入输出布局不匹配

**解决方案**：

✅ **选择合适的输入布局**
```cpp
// 如果输入是 ND，直接使用 ND
using GT = GlobalTensor<float, Shape<...>, Stride<...>, Layout::ND>;

// 避免不必要的 NZ ↔ ND 转换
```

✅ **合并转换操作**
```cpp
// 不好：多次转换
TMOV(temp1, src);
TTRANS(temp2, temp1);
TMOV(dst, temp2);

// 好：一次转换
TTRANS(dst, src);  // 如果支持直接转置
```

### 3.4 核间负载不均衡

**症状**：
- 部分核心利用率高，部分低
- 总执行时间由最慢的核决定

**原因**：
- 数据划分不均匀
- 边界处理逻辑复杂

**解决方案**：

✅ **均匀划分数据**
```cpp
// 计算每个核的工作量
int total_work = M * N;
int num_cores = get_block_num();
int work_per_core = (total_work + num_cores - 1) / num_cores;

// 确保每个核的工作量相近
int block_idx = get_block_idx();
int work_start = block_idx * work_per_core;
int work_end = min(work_start + work_per_core, total_work);
```

✅ **简化边界处理**
```cpp
// 使用 padding 避免特殊处理
int padded_M = (M + TILE_M - 1) / TILE_M * TILE_M;
int padded_N = (N + TILE_N - 1) / TILE_N * TILE_N;
```

---

## 4. 优化技巧清单

### 4.1 Tiling 优化

✅ **选择合适的 Tile 大小**
- 平衡片上容量和数据复用
- A2/A3：单个 Tile 通常 2-32KB
- A5：单个 Tile 可以更大（4-64 KB）

✅ **多级 Tiling**
```cpp
// 全局 → 核级 → 块级
// M×K×N → singleCoreM×singleCoreK×singleCoreN → baseM×baseK×baseN
```

✅ **考虑硬件对齐要求**
- 行主序：Cols × sizeof(T) 对齐到 32字节
- 列主序：Rows × sizeof(T) 对齐到 32 字节
- NZ 布局：特殊的分形对齐要求

### 4.2 内存访问优化

✅ **连续访问**
```cpp
// 好：连续访问
for (int i = 0; i < M; i++) {
  TLOAD(tile, A[i, :]);  // 行连续
}

// 不好：跨步访问
for (int i = 0; i < M; i++) {
  TLOAD(tile, A[:, i]);  // 列访问，可能不连续
}
```

✅ **数据预取**
```cpp
// 提前加载下一批数据
TPREFETCH(next_data, ...);
```

✅ **减少 GM 访问次数**
```cpp
// 在 L1 中缓存频繁访问的数据
TLOAD(cached_tile, ...);  // 加载一次
for (int i = 0; i < N; i++) {
  TCOMPUTE(result, cached_tile, ...);  // 复用多次
}
```

### 4.3 计算优化

✅ **使用合适的数据类型**
```cpp
// fp16 计算更快，但精度较低
// fp32 精度高，但速度较慢
// 根据需求选择

// 混合精度：输入 fp16，累加 fp32
using TileLeft = TileLeft<half, 128, 64>;
using TileAcc = TileAcc<float, 128, 256>;
```

✅ **向量化操作**
```cpp
// 使用 Tile 操作而非标量循环
TADD(c, a, b);  // 并行处理所有元素

// 避免：
for (int i = 0; i < rows; i++) {
  for (int j = 0; j < cols; j++) {
    c[i][j] = a[i][j] + b[i][j];  // 串行
  }
}
```

✅ **算子融合**
```cpp
// 融合多个操作减少中间结果存储
// 例如：Softmax = exp(x - max) / sum(exp(x - max))
// 可以融合为一个 kernel
```

### 4.4 同步优化

✅ **使用细粒度事件**
```cpp
// 好：只等待必要的依赖
Event<Op::TLOAD, Op::TADD> e;
e = TLOAD(tile, ...);
TADD(result, tile, ..., e);

// 不好：全局同步
TLOAD(tile, ...);
TSYNC<Op::TLOAD>();  // 等待所有 TLOAD
TADD(result, tile, ...);
```

✅ **避免稳态循环中的 drain**
```cpp
// 不好：每次迭代都 drain
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
  TCOMPUTE(result, tile);
  TSYNC();  // 等待所有操作完成
}

// 好：只在循环外 drain
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
  TCOMPUTE(result, tile);
}
TSYNC();  // 只在最后同步一次
```

### 4.5 调试优化

✅ **保留正确性检查**
```cpp
#ifdef DEBUG
  // 验证中间结果
  float max_diff = CheckError(result, expected);
  assert(max_diff < 1e-5);
#endif
```

✅ **逐步优化**
- 每次只改一个优化点
- 优化后立即验证正确性和性能
- 记录每次优化的效果

✅ **性能回归测试**
```bash
# 建立性能基线
./benchmark --baseline > baseline.txt

# 优化后对比
./benchmark --compare baseline.txt
```

---

## 5. 平台特定优化

### 5.1 A2/A3 优化要点

**硬件特点**：
- 24 核
- L1 容量：~512KB/核
- Cube 峰值：~50 TFLOPS/核（fp16）

**推荐配置**：
```cpp
// GEMM Tile 大小
constexpr int baseM = 128;
constexpr int baseK = 64;
constexpr int baseN = 256;

// 分形大小
constexpr int fractalABSize = 512;  // A/B 操作数
constexpr int fractalCSize = 1024;  // 累加器
```

**优化重点**：
- 优先优化 K 维度的数据复用
- 使用双缓冲重叠 TLOAD 和 TMATMUL
- 注意 L1 容量限制

### 5.2 A5 优化要点

**硬件特点**：
- 更多核心
- 更大的 L1 容量：~1MB/核
- 更高的 Cube 峰值

**推荐配置**：
```cpp
// GEMM Tile 大小（可以更大）
constexpr int baseM = 256;
constexpr int baseK = 128;
constexpr int baseN = 512;
```

**优化重点**：
- 利用更大的 L1 容量增大 Tile
- 更激进的流水线优化
- 考虑使用 MXFP4/MXFP8 混合精度

### 5.3 CPU 仿真优化

**注意事项**：
- CPU 仿真主要用于验证正确性
- 性能特征与 NPU 不同
- 不要基于 CPU 性能做优化决策

**建议**：
```cpp
#ifdef __CPU_SIM
  // CPU 仿真：使用小 Tile 加快验证
  constexpr int TILE_SIZE = 16;
#else
  // NPU：使用大 Tile 优化性能
  constexpr int TILE_SIZE = 256;
#endif
```

---

## 6. 性能优化案例

### 6.1 GEMM 优化历程

**初始版本**：
- 性能：100TFLOPS
- TLOAD 占比：80%
- TMATMUL 占比：15%

**优化 1：增大 Tile 尺寸**
- 性能：180 TFLOPS（+80%）
- TLOAD 占比：65%
- TMATMUL 占比：30%

**优化 2：双缓冲**
- 性能：320 TFLOPS（+78%）
- TLOAD 占比：45%
- TMATMUL 占比：50%

**优化 3：K 维度分块优化**
- 性能：420 TFLOPS（+31%）
- TLOAD 占比：40%
- TMATMUL 占比：55%

**最终性能**：420 TFLOPS（初始版本的 4.2×）

详细分析：[GEMM 性能优化](../../kernels/manual/a2a3/gemm_performance/README_zh.md)

### 6.2 Flash Attention 优化

**关键优化点**：
- 动态 Tile 大小选择（128 vs 256）
- 多阶段流水线重叠
- 在线 softmax 算法

详细实现：[Flash Attention 优化](../../kernels/manual/common/flash_atten/README_zh.md)

---

## 7. 性能优化检查清单

### 开始优化前

- [ ] 正确性已验证（CPU + NPU）
- [ ] 建立了性能基线
- [ ] 采集了 profiler 数据
- [ ] 识别了性能瓶颈

### Tiling 优化

- [ ] Tile 大小合理（不超片上容量）
- [ ] 考虑了硬件对齐要求
- [ ] 数据复用充分

### 内存优化

- [ ] 使用了双缓冲或多缓冲
- [ ] 内存访问连续
- [ ] 减少了 GM 访问次数

### 计算优化

- [ ] 选择了合适的数据类型
- [ ] 使用了向量化操作
- [ ] 考虑了算子融合

### 并行优化

- [ ] 多核负载均衡
- [ ] 流水线充分重叠
- [ ] 同步开销最小化

### 验证

- [ ] 性能提升已量化
- [ ] 正确性保持
- [ ] 建立了性能回归测试

---

## 8. 常见误区

❌ **过早优化**
- 在验证正确性前就开始优化
- 没有 profiler 数据就盲目优化

❌ **过度优化**
- 为了 1% 的性能提升牺牲可读性
- 优化不是瓶颈的部分

❌ **忽略正确性**
- 优化后不验证数值结果
- 没有回归测试

❌ **平台特定优化**
- 针对单一平台过度优化
- 牺牲跨平台兼容性

---

## 参考资源

- [性能优化指南](opt_zh.md)
- [流水线与并行执行](pipeline-parallel_zh.md)
- [算子调试方法](debug_zh.md)
- [GEMM 优化案例](../../kernels/manual/a2a3/gemm_performance/README_zh.md)
- [Flash Attention 案例](../../kernels/manual/common/flash_atten/README_zh.md)





