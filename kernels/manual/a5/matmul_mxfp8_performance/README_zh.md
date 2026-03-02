# 高性能 MXFP8 算子示例

## 概览

本样例基于 PTO框架实现高性能MXFP8矩阵乘法，系统性整合多核并行切分、基准计算块（base-block）选型、L1 缓存优化及双缓冲等核心优化手段，在保证计算精度的前提下，最大化硬件算力与存储带宽利用率，适配高性能计算场景下的矩阵乘法需求。

## 支持的 AI 处理器

- A5

## 目录结构

```
kernels/manual/a5/matmul_mxfp8_performance/
├── scripts/
│   └── gen_data.py                      # 生成输入与 golden 输出
├── CMakeLists.txt                       # 构建配置
├── mxmatmul_performance_kernel.cpp      # Kernel 实现
├── main.cpp                             # Host 侧入口
└── run.sh                               # 便捷脚本
```

## 算子说明

### 计算功能

本示例实现的 MxMatmul 核心逻辑为：先将左 / 右量化系数矩阵与对应输入矩阵完成广播乘法，再对两组乘积结果执行矩阵乘法运算，最终输出计算结果。其数学表达式如下

$$
C = (scaleA ⊗ A) * (scaleB ⊗ B)
$$

其中`⊗` 表示广播乘法，`*` 表示矩阵乘法。输入矩阵格式如下

- `A` 为 `m×k`
- `scaleA` 为 `m×scaleK`
- `B` 为 `k×n`
- `scaleB` 为 `scaleK×n`
- `C` 为 `m×n`

`main.cpp` 中默认的参考配置为 `m=k=n=6144`， `scaleK=k/32=192`（量化系数矩阵的 k 维度为data矩阵 k 维度的 1/32）。

### 规格

| 项目        | 值 |
| ----------- | ----- |
| OpType          | `MxMatmul` |
| data输入         | `a`: `m×k`, `float8_e5m2_t`, `ND`; `b`: `n×k`, `float8_e5m2_t`, `ND` |
| scale输入        | `scaleA`: `m×scaleK`, `float8_e8m0_t`, `ND`; `scaleB`: `n×scaleK`, `float8_e8m0_t`, `ND` |
| 输出             | `c`: `m×n`, `bfloat16`, `ND` |
| Kernel 名称      | `MxMatmulPerformance` |

## 优化说明

本示例以 A5 平台为性能验证基准，针对矩阵乘法的算力与访存瓶颈，实施以下分层优化策略：

- **多核切分（core partitioning）**：
  
  核心目标是充分利用多核并行算力，将整体计算任务均衡拆分至不同Cube核上。
  - 本示例中 `m = n = k`，通常不建议在单核内再切 `k`，而是切分 `m` 和 `n` 。
  - 将全局任务按 4 × 8 分核，单核负责的子矩阵维度为 `singleCoreM=1536`、`singleCoreK=6144`、`singleCoreN=768`，确保各核负载均衡，最大化并行度。
- **Base block 选择（base block selection）**：
  - 选一个算力/访存比更高、且更贴合片上容量与对齐约束的 base block。对 FP8，常见选择 `[baseM, baseN, baseK] = [128, 256, 128]`，该基本块计算访存比最高，同时更容易保持 GM 写回的 512 字节对齐。
- **L1 缓存（L1 caching）**：
  - 批量缓存策略：一次从 GM 搬入多个 base block 到 L1，减少频繁的 GM-L1 数据搬运，提高带宽利用率。本示例 `stepKa=stepKb=4`，每次缓存 4 个 `k` block。
  - 独立缓存机制：L1 上scale和data独立缓存，引入mxScalePara参数表示两者缓存的比例关系。在复用已有tiling参数的基础上，保证scale读地址满足128B对齐，提高带宽利用率。
- **双缓冲（double buffering）**：
  - 在 L1/L0A/L0B/scaleA/scaleB 开启双缓冲，使 DMA 数据搬运与计算单元的运算过程最大化重叠，消除数据等待耗时，提升整体流水线执行效率。

## Tiling 参数

| 参数          | 值     |
| ------------- | ----- |
| `m`           | 6144  |
| `k`           | 6144  |
| `n`           | 6144  |
| `singleCoreM` | 1536  |
| `singleCoreK` | 6144  |
| `singleCoreN` | 768   |
| `baseM`       | 128   |
| `baseK`       | 128   |
| `baseN`       | 256   |
| `stepM`       | 1     |
| `stepKa`      | 4     |
| `stepKb`      | 4     |
| `stepN`       | 1     |
| `mxScalePara` | 8     |

## 实测性能（参考）

以下数据在 Ascend A5 上测得，覆盖多个 `m=k=n` 尺寸（fp8 输入 → fp16 输出）。

| 参数 | TMATMUL（Cube）占比 | TEXTRACT 占比 | TLOAD 占比 | TSTORE 占比 | 执行时间（ms） |
| --- | --- | --- | --- | --- | --- |
| `m=2048` `k=2048` `n=2048` | 58.2% | 35.1% | 84.6% | 7.3% | 0.0383 |
| `m=4096` `k=4096` `n=4096` | 87.7% | 54.2% | 96.3% | 5.3% | 0.1879 |
| `m=6144` `k=6144` `n=6144` | 91.6% | 57.3% | 98.2% | 3.5% | 0.6020 |
| `m=8192` `k=8192` `n=8192` | 90.1% | 56.2% | 99.6% | 2.6% | 1.4749 |

表中参数含义和性能优化方案请参考 [gemm_performance 文档](../../a2a3/gemm_performance/README_zh.md)。

## 构建与运行

1. 配置 Ascend CANN 环境：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 生成输入 + golden 输出：

```bash
cd ${git_clone_path}/kernels/manual/a5/matmul_mxfp8_performance
python3 scripts/gen_data.py
```

3. 运行示例：

```bash
bash run.sh -r npu -v Ascend910_9599
```

成功时输出：

```text
test success
```
