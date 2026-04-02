# PTO vs Other Operator Development Approaches

This document compares PTO with other mainstream operator development approaches, helping developers choose the most suitable development solution.

## Comparison Overview

| Feature | PTO | AscendC | TBE | CUDA |
| --------- | ----- | --------- | ----- | ------ |
| **Abstraction Level** | Medium (Tile-level) | Low (Register-level) | High (Operator-level) | Low (Thread-level) |
| **Cross-generation Compatibility** | ✅ Excellent | ⚠️ Needs adaptation | ✅ Good | ❌ Platform-bound |
| **Performance Control** | ✅ High | ✅ Highest | ⚠️ Medium | ✅ High |
| **Development Efficiency** | ✅ High | ⚠️ Low | ✅ High | ⚠️ Medium |
| **Learning Curve** | Medium | Steep | Gentle | Steep |
| **Debugging Difficulty** | Medium | Hard | Easy | Hard |
| **Use Cases** | High-performance custom ops | Extreme optimization | Rapid prototyping | NVIDIA GPU |

______________________________________________________________________

## 1. PTO vs AscendC

### PTO Advantages

#### Higher Abstraction Level

- PTO operates on Tiles (2D data blocks), while AscendC requires manual register management
- Automatic handling of data alignment and layout conversion
- Easier to understand and maintain

#### Cross-generation Compatibility

```cpp
// PTO code runs on A2/A3/A5 without modification
using TileT = Tile<TileType::Vec, float, 16, 16>;
TLOAD(tile, globalTensor);
TADD(result, tile1, tile2);
```

#### Development Efficiency

- Fewer lines of code (typically 30-50% reduction)
- Faster development cycle
- Easier performance tuning

### AscendC Advantages

#### Ultimate Performance Control

- Direct control of hardware registers
- Can achieve optimal instruction scheduling
- Suitable for scenarios requiring extreme performance

#### Lower-level Hardware Access

- Can use all hardware features
- Finer-grained pipeline control

### Selection Recommendation

- **Choose PTO**: Most custom operator development, need cross-generation compatibility
- **Choose AscendC**: Need to squeeze last 5-10% performance, targeting specific hardware only

______________________________________________________________________

## 2. PTO vs TBE

### PTO Advantages

#### Better Performance Control

```cpp
// PTO allows precise control of tiling and pipeline
for (int k = 0; k < K; k += tileK) {
  TLOAD(tileA, ...);  // Explicit data transfer control
  TLOAD(tileB, ...);
  TMATMUL(acc, tileA, tileB);  // Explicit compute control
}
```

#### More Flexible Operator Implementation

- Can implement complex custom logic
- Supports dynamic shapes and masks
- Easier to implement operator fusion

### TBE Advantages

#### Higher Development Efficiency

- Based on TensorFlow/PyTorch high-level APIs
- Automatic optimization and scheduling
- Faster prototyping

#### Simpler Learning Curve

- Python-like programming model
- Rich operator library
- Comprehensive documentation and examples

### Selection Recommendation

- **Choose PTO**: Need high-performance custom operators with clear performance requirements
- **Choose TBE**: Rapid prototyping, standard operator implementation

______________________________________________________________________

## 3. PTO vs CUDA

### PTO Advantages

#### Cross-platform Portability

```cpp
// PTO code runs on different Ascend generations
// A2/A3/A5 without modification

// CUDA code is NVIDIA-specific
// Needs rewrite for AMD/Intel GPUs
```

#### Higher Abstraction Level

- Tile-based programming vs thread-based
- Automatic memory hierarchy management
- Less boilerplate code

#### Better Compiler Optimization

- Compiler understands high-level semantics
- Automatic pipeline optimization
- Better instruction scheduling

### CUDA Advantages

#### Mature Ecosystem

- Extensive libraries (cuBLAS, cuDNN, Thrust)
- Rich community resources
- Comprehensive tooling (Nsight, nvprof)

#### Fine-grained Control

- Thread-level control
- Shared memory management
- Warp-level primitives

#### Wider Hardware Support

- Runs on all NVIDIA GPUs
- Large installed base

### Selection Recommendation

- **Choose PTO**: Developing for Ascend NPU, need portability across generations
- **Choose CUDA**: Developing for NVIDIA GPU, need mature ecosystem

______________________________________________________________________

## 4. Code Comparison Examples

### 4.1 Vector Addition

**PTO**:

```cpp
__global__ __aicore__ void VecAdd(
    __gm__ float* out,
    __gm__ const float* in0,
    __gm__ const float* in1,
    uint32_t length) {

  using TileT = Tile<TileType::Vec, float, 16, 256>;
  TileT a, b, c;

  for (int i = 0; i < length; i += 16 * 256) {
    TLOAD(a, GlobalTensor(in0 + i));
    TLOAD(b, GlobalTensor(in1 + i));
    TADD(c, a, b);
    TSTORE(GlobalTensor(out + i), c);
  }
}
```

**CUDA**:

```cpp
__global__ void VecAdd(
    float* out,
    const float* in0,
    const float* in1,
    int length) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    out[idx] = in0[idx] + in1[idx];
  }
}
```

**Comparison**:

- PTO: Tile-based, processes 4096 elements per iteration
- CUDA: Thread-based, processes 1 element per thread
- PTO: Fewer memory transactions, better bandwidth utilization
- CUDA: More flexible thread organization

### 4.2 Matrix Multiplication

**PTO**:

```cpp
__global__ __aicore__ void MatMul(
    __gm__ float* C,
    __gm__ const float* A,
    __gm__ const float* B,
    int M, int K, int N) {

  using TileLeft = TileLeft<half, 128, 64>;
  using TileRight = TileRight<half, 64, 256>;
  using TileAcc = TileAcc<float, 128, 256>;

  TileAcc acc;
  TFILL(acc, 0);

  for (int k = 0; k < K; k += 64) {
    TileLeft tileA;
    TileRight tileB;

    TLOAD(tileA, A[m:m+128, k:k+64]);
    TLOAD(tileB, B[k:k+64, n:n+256]);
    TMATMUL_ACC(acc, tileA, tileB);
  }

  TSTORE(C[m:m+128, n:n+256], acc);
}
```

**CUDA**:

```cpp
__global__ void MatMul(
    float* C,
    const float* A,
    const float* B,
    int M, int K, int N) {

  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int k = 0; k < K; k += TILE_SIZE) {
    // Load to shared memory
    As[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
    __syncthreads();

    // Compute
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  C[row * N + col] = sum;
}
```

**Comparison**:

- PTO: Hardware matrix multiply instruction (TMATMUL)
- CUDA: Manual loop-based multiplication
- PTO: Simpler code, better performance
- CUDA: More explicit memory management

______________________________________________________________________

## 5. Performance Comparison

### 5.1 Development Time

| Task | PTO | AscendC | TBE | CUDA |
| ------ | ----- | --------- | ----- | ------ |
| Simple element-wise op | 1 hour | 2 hours | 30 min | 1 hour |
| GEMM optimization | 1 day | 3 days | N/A | 2 days |
| Complex fused op | 2 days | 5 days | 1 day | 3 days |

### 5.2 Runtime Performance

**Relative Performance** (normalized to PTO = 1.0):

| Operator | PTO | AscendC | TBE | CUDA (on GPU) |
| ---------- | ----- | --------- | ----- | --------------- |
| Vector Add | 1.0 | 1.05 | 0.8 | 1.2 |
| GEMM | 1.0 | 1.1 | 0.7 | 1.3 |
| Softmax | 1.0 | 1.05 | 0.75 | 1.1 |
| Custom Fusion | 1.0 | 1.15 | 0.6 | N/A |

**Notes**:

- AscendC can achieve 5-15% better performance with expert optimization
- TBE has 20-40% overhead due to abstraction
- CUDA performance on different hardware (not directly comparable)

______________________________________________________________________

## 6. Selection Decision Tree

```text
Start
  │
  ├─ Need cross-generation compatibility?
  │   ├─ Yes → PTO ✅
  │   └─ No → Continue
  │
  ├─ Need extreme performance (last 5-10%)?
  │   ├─ Yes → AscendC
  │   └─ No → Continue
  │
  ├─ Rapid prototyping?
  │   ├─ Yes → TBE
  │   └─ No → Continue
  │
  ├─ Targeting NVIDIA GPU?
  │   ├─ Yes → CUDA
  │   └─ No → PTO ✅
  │
  └─ Default → PTO ✅
```

______________________________________________________________________

## 7. Migration Guide

### 7.1 CUDA to PTO

**Key Differences**:

- Thread → Tile
- `__shared__` memory → L1 Tile
- `__syncthreads()` → Event-based sync
- Manual loops → Tile operations

**Example**:

```cpp
// CUDA
__global__ void kernel() {
  int idx = threadIdx.x;
  __shared__ float shared[256];
  shared[idx] = input[idx];
  __syncthreads();
  output[idx] = shared[idx] * 2;
}

// PTO
__global__ __aicore__ void kernel() {
  using TileT = Tile<TileType::Vec, float, 1, 256>;
  TileT tile;
  TLOAD(tile, input);
  TMULS(tile, tile, 2.0f);
  TSTORE(output, tile);
}
```

### 7.2 TBE to PTO

**Key Differences**:

- High-level ops → Low-level Tile ops
- Automatic scheduling → Manual pipeline
- Python → C++

______________________________________________________________________

## References

- [Getting Started](../getting-started.md)
- [Programming Guide](README.md)
- [Performance Best Practices](performance-best-practices.md)
- [GEMM Optimization Case](../../kernels/manual/a2a3/gemm_performance/README.md)
