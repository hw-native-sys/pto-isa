# 高性能 GEMM 算子示例

## 概览

本示例演示如何使用 PTO 实现高性能 GEMM，并覆盖常见优化手段（多核切分、base-block 选择、L1 缓存与双缓冲）。

## 支持的 AI 处理器

- A2/A3

## 目录结构

```
kernels/manual/a2a3/gemm_performance/
├── scripts/
│   └── gen_data.py                  # 生成输入与 golden 输出
├── CMakeLists.txt                   # 构建配置
├── gemm_performance_kernel.cpp      # Kernel 实现
├── main.cpp                         # Host 侧入口
└── run.sh                           # 便捷脚本
```

## 算子说明

### 计算功能

本示例实现 GEMM：

$$
C = A \times B
$$

其中：

- `A` 为 `m×k`
- `B` 为 `k×n`
- `C` 为 `m×n`

`main.cpp` 中默认的参考配置为 `m=k=n=6144`。

### 规格

| 项目        | 值 |
| ----------- | ----- |
| OpType      | `GEMM` |
| 输入        | `a`: `m×k`, `float16`, `ND`; `b`: `n×k`, `float16`, `ND` |
| 输出        | `c`: `m×n`, `float`, `ND` |
| Kernel 名称 | `GEMMPerformance` |

## 优化说明

本示例以 24 核的 A3 平台作为性能验证平台。

- **多核切分（core partitioning）**：在 Cube 核之间切分工作量，尽量把并行度吃满。由于 `m`、`n`、`k` 相等，通常不建议在单核内再切 `k`，而是把 `m` 与 `n` 分摊到 24 核。本示例使用 `4 × 6` 分组，对应 `singleCoreM=1536`、`singleCoreK=6144`、`singleCoreN=1024`。
- **Base block 选择（base block selection）**：选一个算力/访存比更高、且更贴合片上容量与对齐约束的 base block。对 FP16，常见选择 `[baseM, baseN, baseK] = [128, 256, 64]`；相比 `[128, 128, 128]` 算术强度更高，同时更容易保持 GM 写回的 512 字节对齐。
- **L1 缓存（L1 caching）**：一次从 GM 搬入多个 base block 到 L1，提高带宽利用率。本示例 `stepKa=stepKb=4`，每次缓存 4 个 `k` block。
- **双缓冲（double buffering）**：在 L1/L0A/L0B 开启双缓冲，让 DMA 与计算尽可能重叠。

## Tiling 参数

| 参数          | 值     |
| ------------- | ----- |
| `m`           | 6144  |
| `k`           | 6144  |
| `n`           | 6144  |
| `singleCoreM` | 1536  |
| `singleCoreK` | 6144  |
| `singleCoreN` | 1024  |
| `baseM`       | 128   |
| `baseK`       | 64    |
| `baseN`       | 256   |
| `stepM`       | 1     |
| `stepKa`      | 4     |
| `stepKb`      | 4     |
| `stepN`       | 1     |

## 实测性能（参考）

以下数据在 Ascend A3（24 核）上测得，覆盖多个 `m=k=n` 尺寸（fp16 输入 → fp32 输出）。

| 参数 | TMATMUL（Cube）占比 | TEXTRACT 占比 | TLOAD 占比 | TSTORE 占比 | 执行时间（ms） |
| --- | --- | --- | --- | --- | --- |
| `m=1536` `k=1536` `n=1536` | 54.5% | 42.2% | 72.2% | 7.7% | 0.0388 |
| `m=3072` `k=3072` `n=3072` | 79.0% | 62.0% | 90.9% | 5.8% | 0.2067 |
| `m=6144` `k=6144` `n=6144` | 86.7% | 68.1% | 95.2% | 3.1% | 1.5060 |
| `m=7680` `k=7680` `n=7680` | 80.6% | 63.0% | 98.4% | 2.4% | 3.1680 |

### 这些数字意味着什么

这些指标主要用于回答一个问题：**端到端流水线到底被哪个引擎限制（瓶颈在哪）**。

- **规模效应**：执行时间随 `m=k=n` 超线性增长（符合 `O(n^3)` 计算量），吞吐通常会从小尺寸提升到中等尺寸后趋于平缓。
- **TMATMUL 利用率先升后降**：TMATMUL（Cube）占比从 54.5% → 86.7% 随规模增长而提升（更好的摊销与更稳定的流水），但在 `7680³` 降至 80.6%。这常表示在最大尺寸时，计算不再是唯一瓶颈。
- **大尺寸下 TLOAD 接近饱和**：TLOAD 占比在 `7680³` 达到 98.4%，说明 GM 供给路径接近极限并开始反过来限制计算（TMATMUL 占比下降）。
- **TSTORE 很小且继续下降**：GEMM 的输出写回在总耗时中占比很小，且规模越大越明显（一次写回对应大量 FMA）。
- **TEXTRACT 不可忽视**：42%→68% 表示 L1→L0 的 extract/layout 成本不低；优化这一阶段（并把它更好地与计算重叠）会直接影响整体性能。

一个实用的经验：当 **TLOAD 占比接近 ~100%** 时，往往意味着 **被内存供给限制**（即使 TMATMUL 看起来仍然“很忙”）。进一步加速通常来自于减少每 FLOP 的搬运字节数，以及提升阶段重叠。

## 性能优化指南（如何调这个 kernel）

本示例围绕一个标准 GEMM 流水线组织：

1. **TLOAD 阶段**：GM → L1（`TLOAD` 到 `aMatTile[]` / `bMatTile[]`）
2. **TEXTRACT 阶段**：L1 → L0A/L0B（`TEXTRACT` 到 `aTile[]` / `bTile[]`）
3. **TMATMUL 阶段**：L0A/L0B → L0C（`TMATMUL` / `TMATMUL_ACC` 写入 `cTile`）
4. **TSTORE 阶段**：L0C → GM（`TSTORE` 回写 `cTile`）

核心 kernel 位于 `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`，下文列出关键控制点与调优建议。

### 1) 优先做多核切分

关注 `InitGMOffsets(...)`：

- Kernel 将全局 `C[m,n]` 按 `blockDim` 切分为若干互不依赖的 tile。
- 对于近似方阵问题（`m≈n`），在 **`m` 和 `n` 两个维度同时切分** 往往比只切一个维度更均衡。

检查清单：

- 确保 `m % singleCoreM == 0` 且 `n % singleCoreN == 0`。
- 选择与 `blockDim` 匹配的二维网格分解（`m`-tiles × `n`-tiles），让每个核拿到连续的 `A` panel 与 `B` panel。

### 2) 选择能“干净”装进 L0A/L0B 的 base tile

关注 `InitBuffers(...)`：

- L0A 与 L0B 显式做了双缓冲，按 32 KiB 的 ping/pang（`0x0` 与 `0x0 + 32768`）划分。
- 因此有一个硬约束：每个 buffer 的 tile footprint 必须 ≤ 32 KiB。

对于 fp16 输入（2 bytes/elem）：

- L0A tile bytes ≈ `baseM * baseK * 2`
- L0B tile bytes ≈ `baseK * baseN * 2`

参考配置使用：

- `baseM=128, baseK=64` → `128*64*2 = 16 KiB`（空间充裕）
- `baseK=64, baseN=256` → `64*256*2 = 32 KiB`（刚好吃满预算）

指导原则：

- 尽量让 tile 尺寸 **充分利用** 32 KiB（尤其是 `B`），但不要超过。
- `baseK` 保持对齐到 Cube 更偏好的 K 粒度（通常与数据类型/布局有关，常见 32/64/128）。

### 3) 用 L1 的 “stepK” 缓存提升复用（避免溢出）

关注 `ProcessKIteration(...)` 与 `kModstepKa` 相关逻辑：

- `stepKa` / `stepKb` 控制一次 DMA 进入 L1 的 `K` 切片数。
- 本示例 `stepKa=stepKb=4`：一次 `TLOAD` 搬入 4 个 micro-panel，后续再 `TEXTRACT` 到 L0。

指导原则：

- 增大 `stepK` 可以减少 DMA 启动开销并提升 burst 效率，直到 **L1 容量限制或阶段重叠开始恶化**。
- 当 TLOAD 接近 100% 且 TMATMUL 下滑时，可以尝试：
  - 增大 `stepK`（提高复用），或
  - 增大计算强度（例如在 L0 允许的前提下增大 `baseN`/`baseM`），或
  - 改善重叠（见下一节）。

### 4) 保持流水线重叠（避免气泡）

双缓冲标志（`mte2DBFlag`, `mte1DBFlag`）以及事件流是本 kernel 的性能核心：

- **TLOAD** 在加载下一批 `aMatTile[]/bMatTile[]` 的同时，
- **TEXTRACT** 在提取下一批 `aTile[]/bTile[]` 的同时，
- **TMATMUL** 在计算当前批次的 `TMATMUL[_ACC]`。

当你观察到：

- **TLOAD 高而 TMATMUL 低** → Cube 可能“饿了”；要么重叠不足，要么 TLOAD 真正饱和。
- **TEXTRACT 高而 TMATMUL 低** → extract/layout 成为瓶颈；需要降低 `TEXTRACT` 成本或增大每次 extract 的计算量。

实用调优步骤：

- 确保“首轮 warmup”和“末轮 drain”不会把稳态循环串行化。本文件包含“补齐首末同步指令”的逻辑，重构时建议保留。
- 将计算与搬运按 buffer index（ping/pang）分离，只在真实依赖边界上 `wait_flag`。

### 5) 适配新形状时，先重新调 *core tile*

当 `m/k/n` 改变时，不要只改常量：

- 重新计算 `singleCoreM/singleCoreN`，让每个核承担相近的工作量。
- 重新检查 `mLoop`、`nLoop`、`kLoop`（`RunGemmE2E`），因为循环次数对重叠效率影响很大。

常见失效模式：

- `kLoop` 很大但 `stepK` 不足 → TLOAD 主导；
- `kLoop` 很小 → 固定开销主导。

### 6) 用占比指标决定“该优化什么”

从上面的测量可见：

- `7680³` 中 **TLOAD=98.4%** 且 TMATMUL 降到 **80.6%** → 更应该关注减少 GM 搬运（提升复用、更好的缓存 staging）与重叠，而不是单独微调 `TMATMUL`。
- 中等尺寸（`3072³`, `6144³`）同时表现出较高 TMATMUL 与 TLOAD → 流水线接近平衡；进一步提升需要更谨慎的端到端改动。

## 构建与运行

1. 配置 Ascend CANN 环境：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 生成输入 + golden 输出：

```bash
cd ${git_clone_path}/kernels/manual/a2a3/gemm_performance
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

## 变更记录

| 日期       | 变更 |
| ---------- | ------ |
| 2025-12-15 | 调整示例目录并添加本 README |
