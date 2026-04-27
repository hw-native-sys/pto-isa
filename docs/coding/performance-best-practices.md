# Performance Best Practices

This document summarizes performance tuning best practices for PTO operators, providing systematic optimization methods and experience.

## Contents

- [1. Optimization Workflow](#1-optimization-workflow)
- [2. Performance Analysis Methods](#2-performance-analysis-methods)
- [3. Common Performance Issues](#3-common-performance-issues)
- [4. Optimization Techniques Checklist](#4-optimization-techniques-checklist)
- [5. Platform-Specific Optimization](#5-platform-specific-optimization)

---

## 1. Optimization Workflow

### 1.1 Standard Optimization Process

```
Correctness Verification → Performance Baseline → Bottleneck Analysis →
Targeted Optimization → Verification → Iteration
```

**Detailed Steps**:

#### Step 1: Ensure Correctness
```bash
# CPU simulation verification
python3 tests/run_cpu.py --testcase your_op --verbose

# NPU verification
python3 tests/script/run_st.py -r npu -v a3 -t your_op
```

**Checkpoints**:
- ✅ Numerical error < 1e-5 (fp32) or < 1e-3 (fp16)
- ✅ All test cases pass
- ✅ Boundary conditions handled correctly

#### Step 2: Establish Performance Baseline
```bash
# Use msprof to collect performance data
msprof --application="your_app" --output=./baseline
```

**Record Metrics**:
- Total execution time
- Time proportion of each stage (TLOAD/TMATMUL/TSTORE)
- Memory bandwidth utilization
- Compute unit utilization

#### Step 3: Identify Bottlenecks

**Analyze profiler output**:
```
TLOAD:    45%  ← Memory transfer
TEXTRACT: 10%  ← Layout conversion
TMATMUL:  40%  ← Computation
TSTORE:    5%  ← Write back
```

**Bottleneck Types**:
- **Memory Bound**: TLOAD/TSTORE proportion > 60%
- **Compute Bound**: TMATMUL proportion > 70%
- **Conversion Bound**: TEXTRACT/TMOV proportion > 20%

#### Step 4: Targeted Optimization

Choose optimization strategy based on bottleneck type (see subsequent sections).

#### Step 5: Verify Optimization Effect

**Compare Metrics**:
- Performance improvement percentage
- Time changes in each stage
- Numerical correctness maintained

#### Step 6: Iterate Optimization

Repeat steps 3-5 until performance target is reached or optimization space is exhausted.

---

## 2. Performance Analysis Methods

### 2.1 Using msprof Tool

**Basic Usage**:
```bash
# Collect performance data
msprof --application="./your_app" \
       --output=./profiling_data \
       --ai-core=on \
       --task-time=on

# View report
msprof --export=on \
       --output=./profiling_data
```

**Key Metrics**:

| Metric | Meaning | Target Value |
|--------|---------|--------------|
| **TMATMUL Proportion** | Cube unit utilization | > 50% |
| **TLOAD Proportion** | Memory transfer time | < 40% |
| **MTE Bandwidth** | Memory bandwidth utilization | > 70% |
| **Pipeline Bubbles** | Idle time | < 10% |

### 2.2 Manual Timing

Insert timing code in critical paths:

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// Critical code section
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
}

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
printf("TLOAD time: %ld us\n", duration.count());
```

### 2.3 Theoretical Performance Calculation

**GEMM Theoretical Peak**:
```
Theoretical TFLOPS = Hardware Peak × Core Count × Utilization

Example A3 (24 cores):
- Hardware peak: ~50 TFLOPS/core (fp16)
- Theoretical peak: 50 × 24 = 1200 TFLOPS
- Achievable: ~70-80% = 840-960 TFLOPS
```

**Memory Bandwidth Theoretical Value**:
```
Theoretical Bandwidth = Hardware Bandwidth × Utilization

Example A3:
- Hardware bandwidth: ~900 GB/s
- Achievable: ~70-80% = 630-720 GB/s
```

---

## 3. Common Performance Issues

### 3.1 Memory Bandwidth Limited

**Symptoms**:
- TLOAD/TSTORE proportion > 60%
- TMATMUL proportion < 30%

**Causes**:
- Tile too small, insufficient data reuse
- Frequent GM ↔ L1 transfers
- Not using pipeline overlap

**Solutions**:

✅ **Increase Tile Size**
```cpp
// Before optimization: Small Tile
using TileT = Tile<TileType::Vec, float, 8, 64>;  // 2KB

// After optimization: Large Tile
using TileT = Tile<TileType::Vec, float, 16, 256>; // 16KB
```

✅ **Improve Data Reuse**
```cpp
// GEMM: K-dimension blocking
for (int k = 0; k < K; k += TILE_K) {
  TLOAD(tileA, ...);  // Load once
  TLOAD(tileB, ...);  // Load once
  TMATMUL(acc, tileA, tileB);  // Reuse multiple times
}
```

✅ **Use Double Buffering**
```cpp
// Preload
TLOAD(tile[0], ...);

for (int i = 0; i < N; i++) {
  int curr = i % 2;
  int next = (i + 1) % 2;

  // Compute current
  TCOMPUTE(result[curr], tile[curr]);

  // Load next simultaneously
  if (i + 1 < N) {
    TLOAD(tile[next], ...);
  }
}
```

### 3.2 Low Compute Unit Utilization

**Symptoms**:
- TMATMUL proportion < 40%
- Many pipeline bubbles

**Causes**:
- Data transfer can't keep up with computation speed
- Too frequent synchronization
- Tile shape doesn't match hardware

**Solutions**:

✅ **Optimize Pipeline Overlap**
```cpp
// Use events instead of global sync
Event<Op::TLOAD, Op::TMATMUL> e;
e = TLOAD(tile, ...);
TMATMUL(acc, tile, ..., e);  // Only wait for TLOAD
```

✅ **Adjust Tile Shape**
```cpp
// A2/A3 recommended:
// Left: 128×64, Right: 64×256, Acc: 128×256

// A5 recommended:
// Left: 256×128, Right: 128×512, Acc: 256×512
```

---

## 4. Optimization Techniques Checklist

### 4.1 Tiling Optimization

✅ **Choose Appropriate Tile Size**
- Balance on-chip capacity and data reuse
- A2/A3: Single Tile typically 2-32 KB
- A5: Single Tile can be larger (4-64 KB)

✅ **Multi-level Tiling**
```cpp
// Global → Core-level → Block-level
// M×K×N → singleCoreM×singleCoreK×singleCoreN → baseM×baseK×baseN
```

✅ **Consider Hardware Alignment Requirements**
- Row-major: Cols × sizeof(T) aligned to 32 bytes
- Column-major: Rows × sizeof(T) aligned to 32 bytes
- NZ layout: Special fractal alignment requirements

### 4.2 Memory Access Optimization

✅ **Contiguous Access**
```cpp
// Good: Contiguous access
for (int i = 0; i < M; i++) {
  TLOAD(tile, A[i, :]);  // Row contiguous
}

// Bad: Strided access
for (int i = 0; i < M; i++) {
  TLOAD(tile, A[:, i]);  // Column access, may not be contiguous
}
```

✅ **Data Prefetch**
```cpp
// Preload next batch of data
TPREFETCH(next_data, ...);
```

✅ **Reduce GM Access Count**
```cpp
// Cache frequently accessed data in L1
TLOAD(cached_tile, ...);  // Load once
for (int i = 0; i < N; i++) {
  TCOMPUTE(result, cached_tile, ...);  // Reuse multiple times
}
```

### 4.3 Computation Optimization

✅ **Use Appropriate Data Types**
```cpp
// fp16 computation faster but lower precision
// fp32 higher precision but slower
// Choose based on requirements

// Mixed precision: fp16 input, fp32 accumulation
using TileLeft = TileLeft<half, 128, 64>;
using TileAcc = TileAcc<float, 128, 256>;
```

✅ **Vectorized Operations**
```cpp
// Use Tile operations instead of scalar loops
TADD(c, a, b);  // Process all elements in parallel

// Avoid:
for (int i = 0; i < rows; i++) {
  for (int j = 0; j < cols; j++) {
    c[i][j] = a[i][j] + b[i][j];  // Serial
  }
}
```

✅ **Operator Fusion**
```cpp
// Fuse multiple operations to reduce intermediate result storage
// Example: Softmax = exp(x - max) / sum(exp(x - max))
// Can be fused into one kernel
```

### 4.4 Synchronization Optimization

✅ **Use Fine-grained Events**
```cpp
// Good: Only wait for necessary dependencies
Event<Op::TLOAD, Op::TADD> e;
e = TLOAD(tile, ...);
TADD(result, tile, ..., e);

// Bad: Global synchronization
TLOAD(tile, ...);
TSYNC<Op::TLOAD>();  // Wait for all TLOAD
TADD(result, tile, ...);
```

✅ **Avoid Drain in Steady-state Loops**
```cpp
// Bad: Drain every iteration
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
  TCOMPUTE(result, tile);
  TSYNC();  // Wait for all operations to complete
}

// Good: Only drain at loop end
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
  TCOMPUTE(result, tile);
}
TSYNC();  // Only sync once at the end
```

---

## 5. Platform-Specific Optimization

### 5.1 A2/A3 Optimization Points

**Hardware Characteristics**:
- 24 cores
- L1 capacity: ~512 KB/core
- Cube peak: ~50 TFLOPS/core (fp16)

**Recommended Configuration**:
```cpp
// GEMM Tile size
constexpr int baseM = 128;
constexpr int baseK = 64;
constexpr int baseN = 256;

// Fractal size
constexpr int fractalABSize = 512;  // A/B operands
constexpr int fractalCSize = 1024;  // Accumulator
```

**Optimization Focus**:
- Prioritize optimizing K-dimension data reuse
- Use double buffering to overlap TLOAD and TMATMUL
- Pay attention to L1 capacity limits

### 5.2 A5 Optimization Points

**Hardware Characteristics**:
- More cores
- Larger L1 capacity: ~1 MB/core
- Higher Cube peak

**Recommended Configuration**:
```cpp
// GEMM Tile size (can be larger)
constexpr int baseM = 256;
constexpr int baseK = 128;
constexpr int baseN = 512;
```

**Optimization Focus**:
- Utilize larger L1 capacity to increase Tile size
- More aggressive pipeline optimization
- Consider using MXFP4/MXFP8 mixed precision

---

## References

- [Performance Optimization Guide](opt.md)
- [Pipeline and Parallel Execution](pipeline-parallel.md)
- [Debugging Guide](debug.md)
- [GEMM Optimization Case](../../kernels/manual/a2a3/gemm_performance/README.md)
- [Flash Attention Case](../../kernels/manual/common/flash_atten/README.md)

