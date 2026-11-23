# High-Performance HiF4 (HiFloat4) GEMM Example for A6

## Overview

This sample implements a high-performance **HiF4 (HiFloat4)** matrix multiplication on the
Ascend A6 (dav-9201) platform, based on the PTO framework. It uses the HiF4 Cube matmul
pipeline:

```
GM(BF16) --TLOAD--> L1 --TEXTRACT--> L0A/L0B + L0AMX/L0BMX --TMATMUL_MX--> L0C(FP32) --TSTORE--> GM(BF16)
```

The kernel demonstrates multi-core partition, L1 double buffering for the data tiles and
L0A/L0B ping-pong for the `TEXTRACT`/`TMATMUL_MX` overlap.

## Supported AI Processors

- A6 (dav-9201)

## Directory Layout

```
kernels/manual/a6/matmul_hif4_performance/
├── scripts/
│   └── gen_data.py                     # Generate HiF4 input + golden output
├── CMakeLists.txt                      # Build configuration
├── hif4_matmul_performance_kernel.cpp  # Kernel implementation
├── main.cpp                            # Host-side entry point
└── run.sh                              # Build/run script
```

## Operator Description

### Function

`C = A * B`, where `A` and `B` are HiF4 (4-bit) quantized matrices accompanied by their
three-level scale metadata. The mathematical expression of the reconstruction is:

```
C = dequant(A, scaleA) * dequant(B, scaleB)
```

- `A` is `m x k` (HiF4 data, ND) + `m x k/64` (scale, `HIF4_A_ZZ`)
- `B` is `k x n` (HiF4 data, ND) + `k/64 x n` (scale, `HIF4_B_NN`)
- `C` is `m x n` (bfloat16)

The default configuration in `main.cpp` is `m=2048, k=2048, n=2048`.

### Specification

| Item         | Value |
| -----------  | ----- |
| OpType       | `Hif4Matmul` |
| Data Inputs  | `a`: `m x k`, `hifloat4x2_t`, ND; `b`: `k x n`, `hifloat4x2_t`, ND |
| Scale Inputs | `scaleA`: `m x k/64`, `uint8_t`, `HIF4_A_ZZ`; `scaleB`: `k/64 x n`, `uint8_t`, `HIF4_B_NN` |
| Output       | `c`: `m x n`, `bfloat16`, ND |
| Kernel name  | `Hif4MatmulPerformance` |

## Why k is smaller than the A5 MXFP4 example

The A5 `matmul_mxfp4_performance` example uses `k=8192` and K-slices the scale at
`TLOAD` time (its scale is plain 2D `MX_A_ND`/`MX_B_DN`). HiF4 scale is **pre-fractalized**
in GM as `[16,4]` cells (`HIF4_A_ZZ`/`HIF4_B_NN`), and the A6 `TLoad` consumes those cells
via contiguous `ZZ2ZZ`/`NN2NN` bursts — so a HiF4 scale panel cannot be K-sliced at TLOAD
time. This demo therefore loads the full-K scale once per base block and K-slices it at
`TEXTRACT` (L1 → L0AMX/L0BMX) instead. `k=2048` keeps the full-K scale tile small enough
for L1.

## Tiling Parameters

| Parameter     | Value |
| ------------- | ----- |
| `m`           | 2048  |
| `k`           | 2048  |
| `n`           | 2048  |
| `singleCoreM` | 512   |
| `singleCoreK` | 2048  |
| `singleCoreN` | 512   |
| `baseM`       | 256   |
| `baseK`       | 256   |
| `baseN`       | 256   |
| `stepKa`      | 2     |
| `stepKb`      | 2     |
| `blockDim`    | 16    |

## Build and Run

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Generate input + golden output:

```bash
cd ${git_clone_path}/kernels/manual/a6/matmul_hif4_performance
python3 scripts/gen_data.py
```

3. Build and run:

```bash
# NPU
./run.sh -r npu -v dav_9201

# Simulator (if available)
./run.sh -r sim -v dav_9201
```

The host verifies the device output against `output/golden.bin` with a 3% relative
tolerance (HiF4 quantization error).
