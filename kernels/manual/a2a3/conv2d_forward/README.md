# 2D Convolution Forward Operator Example

## Overview

This sample implements 2D convolution forward calculation based on the PTO framework, covering common optimization techniques (multi-core partitioning, base-block selection, L1 caching, and double buffering).

## Supported AI Processors

- A2/A3

## Directory Structure

```
kernels/manual/a2a3/conv2d_forward/
├── scripts/
│   └── gen_data.py                      # Generate input and golden output
├── CMakeLists.txt                       # Build configuration
├── conv2d_forward_kernel.cpp            # Kernel implementation
├── main.cpp                             # Host-side entry point
└── run.sh                               # Convenience script
```

## Operator Description

### Computational Function

This example implements 2D convolution forward calculation:

$$
Y = X * K
$$

where `*` denotes the convolution operation. The input matrix formats are as follows:

- `X` is of shape `[batch, cin, hin, win, c0]`
- `K` is in `Fractal_Z` format: `[cin * hk * wk, n/16, 16, c0]`
- `Y` is of shape `[batch, n/c0, hout, wout, c0]`
- `hout = (hin + padTop + padBottom - dilationH * (hk - 1) - 1) / strideH + 1`
- `wout = (win + padLeft + padRight - dilationW * (wk - 1) - 1) / strideW + 1`

The default reference configuration in `main.cpp` is `X=[4, 32, 16, 96, 16], K=[288, 384, 16, 16], Y=[4, 384, 16, 96, 16], strideH=strideW=1, dilationH=dilationW=1, padTop=padBottom=padLeft=padRight=1`.

### Specifications

| Item | Value |
| :--- | :--- |
| OpType | `Conv` |
| Input | `X`: `[batch,cin,hin,win,c0]`, `half`; `K`: `Fractal_Z`, `half`|
| Output | `Y`: `[batch,n/c0,hout,wout,c0]`, `half`|
| Kernel Name | `Conv` |

## Optimization Details

This example uses the 24-core A3 platform for performance verification.

-   **Multi-core Partitioning (core partitioning)**: Split the workload among Cube cores to maximize parallelism.It is generally not recommended to further split `k` within a single core. Instead, `m` and `n` are distributed across the 24 cores. This example uses a `4 × 6` grouping, corresponding to `singleCoreM=1536`, `singleCoreK=4608`, and `singleCoreN=1024`.
-   **Base block selection (base block selection)**: Choose a base block that offers a higher compute-to-memory-access ratio and better fits the on-chip capacity and alignment constraints. For FP16, a common choice is `[baseM, baseN, baseK] = [128, 256, 48]`. Compared to `[128, 128, 128]`, it provides higher arithmetic intensity and is easier to maintain 512-byte alignment for GM write-back.
-   **L1 caching (L1 caching)**: Load multiple base blocks from GM to L1 in one go to improve bandwidth utilization. In this example, `stepKa=stepKb=3`, meaning 3 `k` blocks are cached at a time.
-   **Double buffering (double buffering)**: Enable double buffering in L1/L0A/L0B to overlap DMA operations with computation as much as possible.

## Tiling Parameters

| Parameter | Value |
| :--- | :--- |
| `m` | 6144 |
| `k` | 4608 |
| `n` | 6144 |
| `batch` | 4 |
| `cin` | 32 |
| `hin` | 16 |
| `win` | 96 |
| `c0` | 16 |
| `hk` | 3 |
| `wk` | 3 |
| `strideH` | 1 |
| `strideW` | 1 |
| `dilationH` | 1 |
| `dilationW` | 1 |
| `padTop` | 1 |
| `padBottom` | 1 |
| `padLeft` | 1 |
| `padRight` | 1 |
| `singleCoreM` | 1536 |
| `singleCoreK` | 4608 |
| `singleCoreN` | 1024 |
| `baseM` | 128 |
| `baseK` | 48 |
| `baseN` | 256 |
| `stepM` | 1 |
| `stepKa` | 3 |
| `stepKb` | 3 |
| `stepN` | 1 |

## Measured Performance (Reference)

The following data was measured on Ascend A3 (24 cores), covering multiple input/output matrix sizes (fp16 input → fp32 output).

| Parameters | TMATMUL (Cube) % | TLOAD % | TEXTRACT % | TSTORE % | Execution Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `X=[4,8,8,48,16]`    `K=[72,96,16,16]`| 52.90% | 43.90% | 59.4% | 5.60% | 0.0322 |
| `X=[4,16,12,64,16]`  `K=[144,192,16,16]` | 86.1% | 75.1% | 57.3% | 4% | 0.1473 |
| `X=[4,24,12,96,16]`  `K=[216,288,16,16]` | 89.3% | 78.8% | 61.6% | 2.8% | 0.4709 |
| `X=[4,32,16,96,16]`  `K=[288,384,16,16]` | 90.8% | 80.4% | 61.1% | 2.1% | 1.0979 |
| `X=[4,40,15,128,16]` `K=[360,480,16,16]` | 91.3% | 80.7% | 62.4% | 1.7% | 2.1312 |


For the meaning of the parameters in the table and performance optimization schemes, please refer to [gemm_performance Measured Performance](../../a2a3/gemm_performance/README_zh.md#实测性能参考).

## Build and Run

1. Configure the Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Generate input + golden output:

```bash
cd ${git_clone_path}/kernels/manual/a2a3/conv2d_forward
python3 scripts/gen_data.py
```

3. Run the example:

```bash
bash run.sh -r npu -v Ascend910B1
```

If the run succeeds, the output prints:

```text
test success
```
