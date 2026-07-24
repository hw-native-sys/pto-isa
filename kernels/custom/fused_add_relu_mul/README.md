# Fused Add-ReLU-Mul Custom Operator Example

This example demonstrates how to develop a PTO custom operator from scratch, implementing operator fusion optimization.

## Operator Functionality

**Fused Add-ReLU-Mul**: Fuses three element-wise operations into a single kernel

```
out = ReLU(x + bias) * scale
```

Equivalent to:
```python
# Step 1: Add
temp = x + bias

# Step 2: ReLU
temp = max(0, temp)

# Step 3: Mul
out = temp * scale
```

## Fusion Benefits

| Metric | Before Fusion (3 kernels) | After Fusion (1 kernel) | Improvement |
|--------|---------------------------|-------------------------|-------------|
| **GM Access** | 6 times (3 read + 3 write) | 2 times (1 read + 1 write) | 3× |
| **Kernel Launch** | 3 times | 1 time | 3× |
| **Data Locality** | Poor (intermediate results to GM) | Good (stay in L1/L0) | ✓ |
| **Expected Speedup** | - | 2-3× | - |

## Directory Structure

```
fused_add_relu_mul/
├── fused_add_relu_mul_kernel.cpp  # PTO Kernel implementation (3 versions)
├── main.cpp                       # Host test program
├── CMakeLists.txt                 # Build configuration
├── run.sh                         # Build and run script
└── README.md                      # This document
```

## Kernel Implementation Versions

This example provides three implementation versions demonstrating different optimization techniques:

### 1. Basic Version (FusedAddReLUMulKernel)

**Features**:
- Simple and straightforward implementation
- Good for learning basic concepts
- Performance baseline

**Core Code**:
```cpp
// Load data
TLOAD(tile_x, GlobalTensor(x + i));

// Fused computation
TADDS(tile_result, tile_x, bias);    // Add
TRELU(tile_result, tile_result);     // ReLU
TMULS(tile_result, tile_result, scale); // Mul

// Store result
TSTORE(GlobalTensor(out + i), tile_result);
```

**Tile Configuration**:
- Size: 16×256 = 4096 elements
- Memory: 16 KB
- Platforms: A2/A3/A5

### 2. Optimized Version (FusedAddReLUMulOptimizedKernel)

**Features**:
- Uses double buffering technique
- Overlaps data loading and computation
- Improves pipeline efficiency

**Performance**: 1.5-2× faster than basic version

### 3. Large Tile Version (FusedAddReLUMulLargeTileKernel)

**Features**:
- Uses larger tile size
- Reduces loop iteration count
- Optimized for A5 platform (larger L1 capacity)

**Tile Configuration**:
- Size: 32×512 = 16384 elements
- Memory: 64 KB
- Platform: A5 (L1 ~1MB/core)

## Build and Run

### Prerequisites

1. **CANN Environment**:
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. **Set SOC Version**:
   - A2/A3: `Ascend910B1` or `Ascend910B2`
   - A5: `Ascend910_9599`

### Quick Start

#### Method 1: Using Script (Recommended)

```bash
# CPU simulation mode (development/debugging)
./run.sh --sim --clean

# NPU mode (performance testing)
./run.sh --npu --soc Ascend910B1

# Debug mode
./run.sh --sim --debug
```

#### Method 2: Manual Build

```bash
# Create build directory
mkdir -p build && cd build

# Configure (CPU simulation)
cmake .. -DRUN_MODE=sim -DSOC_VERSION=Ascend910B1

# Configure (NPU)
cmake .. -DRUN_MODE=npu -DSOC_VERSION=Ascend910B1

# Build
make -j$(nproc)

# Run
./fused_add_relu_mul
```

## Performance Optimization Tips

### 1. Choose Appropriate Tile Size

**Principles**:
- Balance on-chip capacity and data reuse
- A2/A3: Single tile typically 2-32 KB
- A5: Single tile can be larger (4-64 KB)

### 2. Use Double Buffering

**Benefits**:
- Overlap data loading and computation
- Improve pipeline efficiency
- Reduce wait time

### 3. Reduce Synchronization Overhead

**Good Practice**:
```cpp
Event e;
for (int i = 0; i < N; i++) {
    e = TLOAD(tile, ...);
    COMPUTE(tile, e);  // Only wait for TLOAD completion
}
```

### 4. Operator Fusion

**Benefits**:
- Memory access: 6 times → 2 times (3×)
- Kernel launch: 3 times → 1 time (3×)
- Performance: 2-3× speedup

## Extension Exercises

### Exercise 1: Add More Fused Operations

Try fusing more operations, for example:
```cpp
// Fused Add-ReLU-Mul-Sigmoid
out = Sigmoid(ReLU(x + bias) * scale)
```

### Exercise 2: Support More Data Types

Add FP16 support using templates.

### Exercise 3: Add Mask Support

Support conditional computation with masks.

### Exercise 4: Integrate with PyTorch

Refer to `demos/baseline/add` example to integrate this operator into PyTorch.

## References

- [PTO Programming Model](../../../docs/coding/ProgrammingModel_zh.md)
- [Operator Fusion](../../../docs/coding/operator-fusion_zh.md)
- [Performance Optimization](../../../docs/coding/opt_zh.md)
- [Pipeline and Parallel Execution](../../../docs/coding/pipeline-parallel_zh.md)
- [Add Operator Example](../../../demos/baseline/add/README_zh.md)

## License

This project is licensed under CANN Open Software License Agreement Version 2.0.
