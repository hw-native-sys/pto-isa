# High-Performance MXFP4 Operator Example for Unaligned Scenarios

## Overview

This sample implements high-performance MXFP4 matrix multiplication based on the PTO framework, which systematically integrates core optimization methods such as multi-core parallel partitioning, base-block selection, L1 cache optimization and double buffering. On the premise of ensuring computing accuracy, it maximizes the utilization of hardware computing power and storage bandwidth, and adapts to the non-aligned matrix multiplication requirements in high-performance computing scenarios.

## Supported AI Processors

- A5

## Directory Layout

```
kernels/manual/a5/matmul_mxfp8_performance/
├── scripts/
│   └── gen_data.py                      # Generate input and golden output
├── CMakeLists.txt                       # Build configuration
├── mxmatmul_performance_kernel.cpp      # Kernel implementation
├── main.cpp                             # Host-side entry point
└── run.sh                               # script
```

## Operator Description

### Function

The logic of MxMatmul implemented is as follows: first perform broadcast multiplication on the left/right quantization coefficient matrices with the corresponding input matrices, then execute matrix multiplication on the two groups of product results, and finally output the calculation results. Its mathematical expression is as follows:

$$
C = (scaleA ⊗ A) * (scaleB ⊗ B)
$$

where `⊗`  denotes broadcast multiplication and `*` enotes matrix multiplication. The input matrix formats are as follows:

- `A` is `m×k`
- `scaleA` is `m×scaleK`
- `B` is `k×n`
- `scaleB` is `scaleK×n`
- `C` is `m×n`

The default reference configuration in `main.cpp` is `m=2040, k=8192, n=8100` and `scaleK=k/32=256` (the k-dimension of the quantization coefficient matrix is 1/32 of the k-dimension of the data matrix).

### Specification

| Item         | Value |
| -----------  | ----- |
| OpType       | `MxMatmul` |
| Data Inputs  | `a`: `m×k`, `float8_e5m2`, `ND`; `b`: `n×k`, `float8_e5m2`, `DN` |
| Scale Inputs | `scaleA`: `m×scaleK`, `float8_e8m0`, `ND`; `scaleB`: `n×scaleK`, `float8_e8m0`, `DN` |
| Output       | `c`: `m×n`, `bfloat16`, `ND` |
| Kernel name  | `MxMatmulPerformance` |

## Optimization Notes

This example uses Ascend A5 platform as the performance validation platform.

- **Core Partitioning**：

  The core goal is to fully utilize multi-core parallel computing power and evenly split the overall computing task across different Cube cores.
  - In this example, `m=2040, k=8192, n=8100`; it is generally not recommended to partition the `k`，dimension within a single core, but instead partition the `m` and `n` dimensions.
  - The global task is partitioned across cores in a 4 × 8 manner, with a single core responsible for submatrices of dimensions `singleCoreM=512`、`singleCoreK=8192` and `singleCoreN=1024`, ensuring load balancing across all cores and maximizing parallelism.
- **Base Block Selection**：
  - choose base blocks that maximize compute-to-memory ratio. For FP16, a common choice is `[baseM, baseN, baseK] = [256, 256, 256]`, which achieves the highest computing-to-memory ratio for the basic block and is more conducive to maintaining 512-byte alignment for GM write-back.
- **L1 Caching**：
  - Batch caching strategy: move multiple base blocks from GM to L1 per transfer to improve bandwidth utilization. This example sets `stepKa=stepKb=2` to cache four `k` blocks at a time.
  - Independent caching: Scale and data are cached independently on L1, and the mxScalePara parameter is introduced to represent the cache ratio between the two.
- **Double Buffering**：
  - overlap DMA and compute by enabling double buffering in L1, L0A, L0B, L0ScaleA and L0ScaleB.

## Tiling Parameters

| 参数          | 值     |
| ------------- | ----- |
| `m`           | 2040  |
| `k`           | 8192  |
| `n`           | 8100  |
| `singleCoreM` | 512  |
| `singleCoreK` | 8192  |
| `singleCoreN` | 1024   |
| `baseM`       | 256   |
| `baseK`       | 256   |
| `baseN`       | 256   |
| `stepM`       | 1     |
| `stepKa`      | 2     |
| `stepKb`      | 2     |
| `stepN`       | 1     |
| `mxScalePara` | 4     |

## Measured Performance (Reference)

The following data were collected on Ascend A5, covering multiple sizes with m=k=n (fp8 input → fp16 output).

| Parameter | TMATMUL (Cube) Ratio | TEXTRACT Ratio | TLOAD Ratio | TSTORE Ratio | Execution time (ms) |
| --- | --- | --- | --- | --- | --- |
| `m=2048` `k=2048` `n=2048` | 44.7% | 46.6% | 22.1% | 25.6% | 0.0425 |
| `m=2048` `k=4096` `n=4096` | 77.4% | 76.7% | 38.5% | 7.7% | 0.1003 |
| `m=4096` `k=1024` `n=8192` | 64.9% | 58.4% | 29.3% | 25.7% | 0.1226 |
| `m=1024` `k=12288` `n=4096` | 84.9% | 87.4% | 43.4% | 2.8% | 0.1377 |
| `m=2048` `k=8192` `n=8192` | 90.7% | 88.1% | 45.8% | 4.6% | 0.3489 |
| `m=2040` `k=8192` `n=8100` | 83.0% | 83.0% | 42.1% | 12.3% | 0.3773 |

For the meaning of the parameters in the table and the performance optimization scheme, please refer to [gemm_performance Measured Performance](../../a2a3/gemm_performance/README.md#measured-performance-reference).

## Build and Run

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Generate input + golden output:

```bash
cd ${git_clone_path}/kernels/manual/a5/matmul_mxfp4_performance
python3 scripts/gen_data.py
```

3. Run the example:

```bash
bash run.sh -r npu -v Ascend950
```

If the run succeeds, the output prints:

```text
test success
```
