# 卷积2D正向算子示例

## 概览

本样例基于 PTO框架实现卷积2D正向计算，并覆盖常见优化手段（多核切分、base-block 选择、L1 缓存与双缓冲）。
## 支持的 AI 处理器

- A2/A3

## 目录结构

```
kernels/manual/a2a3/conv2d_forward/
├── scripts/
│   └── gen_data.py                      # 生成输入与 golden 输出
├── CMakeLists.txt                       # 构建配置
├── conv2d_forward_kernel.cpp            # Kernel 实现
├── main.cpp                             # Host 侧入口
└── run.sh                               # 便捷脚本
```

## 算子说明

### 计算功能

本示例实现卷积2D正向计算：

$$
Y = X * K
$$

其中`*` 表示卷积计算。输入矩阵格式如下

- `X` 为 `[batch,cin,hin,win,c0]`
- `K` 为 `Fractal_Z, 即:[cin*hk*wk,n/16,16,c0]`
- `Y` 为 `[batch,n/c0,hout,wout,c0]`
- `hout = (hin + padTop + padBottom - dilationH * (hk - 1) - 1) / strideH + 1`
- `wout = (win + padLeft + padRight - dilationW * (wk - 1) - 1) / strideW + 1`

`main.cpp` 中默认的参考配置为 `X=[4, 32, 16, 96, 16], K=[288, 384, 16, 16], Y=[4, 384, 16, 96, 16], strideH=strideW=1, dilationH=dilationW=1, padTop=padBottom=padLeft=padRight=1`。

### 规格

| 项目        | 值 |
| ----------- | ----- |
| OpType          | `Conv2dForward` |
| data输入         | `X`: `[batch,cin,hin,win,c0]`, `half`; `K`: `Fractal_Z`, `half`|
| 输出             | `Y`: `[batch,n/c0,hout,wout,c0]`, `half`|
| Kernel 名称      | `Conv2dForward` |

## 优化说明

本示例以 24 核的 A3 平台作为性能验证平台。

- **多核切分（core partitioning）**：在 Cube 核之间切分工作量，尽量把并行度吃满。通常不建议在单核内再切 `k`，而是把 `m` 与 `n` 分摊到 24 核。本示例使用 `4 × 6` 分组，对应 `singleCoreM=1536`、`singleCoreK=4608`、`singleCoreN=1024`。
- **Base block 选择（base block selection）**：选一个算力/访存比更高、且更贴合片上容量与对齐约束的 base block。对 FP16，常见选择 `[baseM, baseN, baseK] = [128, 256, 48]`；相比 `[128, 128, 128]` 算术强度更高，同时更容易保持 GM 写回的 512 字节对齐。
- **L1 缓存（L1 caching）**：一次从 GM 搬入多个 base block 到 L1，提高带宽利用率。本示例 `stepKa=stepKb=3`，每次缓存 3 个 `k` block。
- **双缓冲（double buffering）**：在 L1/L0A/L0B 开启双缓冲，让 DMA 与计算尽可能重叠。

## Tiling 参数

| 参数          | 值     |
| ------------- | ----- |
| `m`           | 6144  |
| `k`           | 4608  |
| `n`           | 6144  |
| `batch`       | 4     |
| `cin`         | 32    |
| `hin`         | 16    |
| `win`         | 96    |
| `c0`          | 16    |
| `hk`          | 3     |
| `wk`          | 3     |
| `strideH`     | 1     |
| `strideW`     | 1     |
| `dilationH`   | 1     |
| `dilationW`   | 1     |
| `padTop`      | 1     |
| `padBottom`   | 1     |
| `padLeft`     | 1     |
| `padRight`    | 1     |
| `singleCoreM` | 1536  |
| `singleCoreK` | 4608  |
| `singleCoreN` | 1024  |
| `baseM`       | 128   |
| `baseK`       | 48    |
| `baseN`       | 256   |
| `stepM`       | 1     |
| `stepKa`      | 3     |
| `stepKb`      | 3     |
| `stepN`       | 1     |

## 实测性能（参考）

以下数据在 Ascend A3（24 核）上测得，覆盖多个输入输出矩阵尺寸（fp16 输入 → fp16 输出）。

| 参数 | TMATMUL（Cube）占比 | TLOAD 占比 | TEXTRACT 占比 | TSTORE 占比 | 执行时间（ms） |
| --- | --- | --- | --- | --- | --- |
| `X=[4,8,8,48,16]`    `K=[72,96,16,16]`| 52.90% | 43.90% | 59.4% | 5.60% | 0.0322 |
| `X=[4,16,12,64,16]`  `K=[144,192,16,16]` | 86.1% | 75.1% | 57.3% | 4% | 0.1473 |
| `X=[4,24,12,96,16]`  `K=[216,288,16,16]` | 89.3% | 78.8% | 61.6% | 2.8% | 0.4709 |
| `X=[4,32,16,96,16]`  `K=[288,384,16,16]` | 90.8% | 80.4% | 61.1% | 2.1% | 1.0979 |
| `X=[4,40,15,128,16]` `K=[360,480,16,16]` | 91.3% | 80.7% | 62.4% | 1.7% | 2.1312 |


表中参数含义和性能优化方案请参考[gemm_performance实测性能](../gemm_performance/README.md#实测性能参考)。

## 构建与运行

1. 配置 Ascend CANN 环境：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 生成输入 + golden 输出：

```bash
cd ${git_clone_path}/kernels/manual/a2a3/conv2d_forward
python3 scripts/gen_data.py
```

3. 运行示例：

```bash
bash run.sh -r npu -v Ascend910B1
```

成功时输出：

```text
test success
```
