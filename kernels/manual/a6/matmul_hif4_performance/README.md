# A6 高性能 HiF4（HiFloat4）GEMM 示例

## 概述

本示例基于 PTO 框架，在 Ascend A6（dav-9201）平台上实现高性能 **HiF4（HiFloat4）** 矩阵乘。
使用 HiF4 Cube 矩阵乘流水线：

```
GM(BF16) --TLOAD--> L1 --TEXTRACT--> L0A/L0B + L0AMX/L0BMX --TMATMUL_MX--> L0C(FP32) --TSTORE--> GM(BF16)
```

kernel 演示了多核切分、数据 tile 的 L1 双缓冲以及 L0A/L0B 乒乓缓冲，以重叠
`TEXTRACT`/`TMATMUL_MX` 与访存。

## 支持的 AI 处理器

- A6（dav-9201）

## 目录结构

```
kernels/manual/a6/matmul_hif4_performance/
├── scripts/
│   └── gen_data.py                     # 生成 HiF4 输入与标杆输出
├── CMakeLists.txt                      # 构建配置
├── hif4_matmul_performance_kernel.cpp  # kernel 实现
├── main.cpp                            # 主机侧入口
└── run.sh                              # 构建/运行脚本
```

## 算子描述

### 功能

`C = A * B`，其中 `A` 与 `B` 为 HiF4（4-bit）量化矩阵，并携带三级缩放元数据。其重建表达式为：

```
C = dequant(A, scaleA) * dequant(B, scaleB)
```

- `A`：`m x k`（HiF4 数据，ND）+ `m x k/64`（缩放，`HIF4_A_ZZ`）
- `B`：`k x n`（HiF4 数据，ND）+ `k/64 x n`（缩放，`HIF4_B_NN`）
- `C`：`m x n`（bfloat16）

`main.cpp` 中的默认配置为 `m=2048, k=2048, n=2048`。

### 规格

| 项        | 值 |
| --------- | -- |
| OpType    | `Hif4Matmul` |
| 数据输入  | `a`：`m x k`，`hifloat4x2_t`，ND；`b`：`k x n`，`hifloat4x2_t`，ND |
| 缩放输入  | `scaleA`：`m x k/64`，`uint8_t`，`HIF4_A_ZZ`；`scaleB`：`k/64 x n`，`uint8_t`，`HIF4_B_NN` |
| 输出      | `c`：`m x n`，`bfloat16`，ND |
| Kernel 名 | `Hif4MatmulPerformance` |

## 为什么 k 比 A5 的 MXFP4 示例小

A5 的 `matmul_mxfp4_performance` 使用 `k=8192`，并在 `TLOAD` 阶段对缩放做 K 分片
（其缩放为普通二维 `MX_A_ND`/`MX_B_DN`）。HiF4 的缩放在 GM 中是**预先分块**的
`[16,4]` 单元（`HIF4_A_ZZ`/`HIF4_B_NN`），A6 的 `TLoad` 通过连续的 `ZZ2ZZ`/`NN2NN`
burst 读取这些单元 —— 因此 HiF4 缩放面板无法在 TLOAD 阶段做 K 分片。本示例改为在每个
base block 一次性加载 full-K 缩放，并在 `TEXTRACT`（L1 → L0AMX/L0BMX）阶段做 K 分片。
`k=2048` 使 full-K 缩放 tile 足够小，可放入 L1。

## 切分参数

| 参数         | 值    |
| ------------ | ----- |
| `m`          | 2048  |
| `k`          | 2048  |
| `n`          | 2048  |
| `singleCoreM`| 512   |
| `singleCoreK`| 2048  |
| `singleCoreN`| 512   |
| `baseM`      | 256   |
| `baseK`      | 256   |
| `baseN`      | 256   |
| `stepKa`     | 2     |
| `stepKb`     | 2     |
| `blockDim`   | 16    |

## 构建与运行

1. 配置 Ascend CANN 环境：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 生成输入与标杆输出：

```bash
cd ${git_clone_path}/kernels/manual/a6/matmul_hif4_performance
python3 scripts/gen_data.py
```

3. 构建并运行：

```bash
# NPU
./run.sh -r npu -v dav_9201

# 仿真器（如可用）
./run.sh -r sim -v dav_9201
```

主机侧以 3% 相对误差（HiF4 量化误差）比对设备输出与 `output/golden.bin`。
