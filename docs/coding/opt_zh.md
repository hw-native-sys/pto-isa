# PTO 性能优化指南

本文是面向 PTO kernel 的实用优化指南，重点关注**软件可见**的调优杠杆：

- Tiling 与跨 blocks/cores 的工作划分
- 数据搬运（GM ↔ 片上）与布局选择
- 重叠与同步（流水线、双缓冲）
- 指令选择与融合（Vector vs Cube 阶段）

需要端到端、示例驱动的深度讲解可参考（中文）：

- GEMM：[`kernels/manual/a2a3/gemm_performance/README_zh.md`](../../kernels/manual/a2a3/gemm_performance/README_zh.md)
- Flash Attention：[`kernels/manual/common/flash_atten/README_zh.md`](../../kernels/manual/common/flash_atten/README_zh.md)

## 1. 性能模型：按阶段思考

本仓库中多数高性能 kernel 可以抽象为一条阶段流水线：

1. **TLOAD**：全局内存（GM）→ 片上暂存（例如 Mat/Vec tiles）
2. **布局/暂存变换**：`TEXTRACT`、`TMOV`、`TTRANS`、`TRESHAPE`（取决于内核）
3. **计算**：
   - **Cube**：`TMATMUL`、`TMATMUL_ACC` 等
   - **Vector**：逐元素、归约、exp/log、compare/select 等
4. **TSTORE**：片上 → GM

优化目标通常相同：

- 最大化 “load / transform / compute / store” 的稳态重叠
- 降低每个有效 FLOP 需要搬运的字节数
- 避免流水线气泡（不必要的同步、差的 tiling、差的核划分）

当你有 profiling 的阶段占比时，可以把它当作“时间去哪了”的线索：

- **TLOAD 接近 100%**：流水线被喂不饱（feed-limited）；减少流量或提升复用/重叠。
- **Transform（`TEXTRACT`/`TMOV`）占主导**：减少每 FLOP 的布局工作，或提升每次 transform 对应的计算量以摊薄开销。
- **TMATMUL 很低而 TLOAD 很高**：Cube 在挨饿；重叠断了或内存带宽饱和。

## 2. 可重复的调优流程

1. **从正确性开始**
   - 先在 CPU 仿真验证：`python3 tests/run_cpu.py --verbose`
   - 在改调度前尽早加入数值检查（max diff / relative diff）。

2. **固定问题形态（shape set）**
   - 选择有代表性的 shape（包含“小”和“大”）。
   - 建议把结果记录在 kernel 目录 README 表格中，便于评审与回归。

3. **定位瓶颈阶段**
   - 使用 profiler 输出与阶段占比（若有）。
   - 若没有 profiler，可在 load/compute/store 等大阶段打点对比耗时。

4. **一次只改一个杠杆**
   - 只改 tiling、或只改核划分、或只改重叠策略（避免同时改多项）。
   - 对同一 shape set 重复验证。

5. **锁定稳定的稳态**
   - 确保 warm-up 与 drain（首尾迭代）不把主循环串行化。

## 3. 并行性：blocks、cores 与按 `block_idx` 的 tiling

PTO 遵循 SPMD 风格执行模型：所有核执行同一 kernel，`block_idx`（以及可选的 sub-block IDs）决定工作分配。

推荐阅读（中文）：

- 概览与示例：`docs/coding/tutorial_zh.md`
- 一个具体的 “tile-by-block-id” 例子：`docs/coding/tutorials/vec-add_zh.md`

建议：

- 当两个维度都很大时优先 **2D 划分**（例如 GEMM 按 `m` 与 `n` 同时切分）。
- 保持每个 block 的 GM 访问尽量 **连续** 与 **规则**（提升 burst 效率）。
- 选择每核工作量尽量均衡的划分（避免长尾 block）。

## 4. Tiling：选择能放下且能复用的尺寸

Tiling 是一阶调优旋钮：

- 决定片上占用（是否溢出/抖动/浪费 buffer）
- 决定复用（一次加载参与多少计算）
- 决定各阶段重叠潜力

检查清单：

- Tile 尺寸不超过片上限制（并满足 kernel 明确的 buffer 分区约束）。
- Tile 形状/布局与目标引擎匹配（Cube vs Vector）。
- 尽量提高算术强度：每搬运 1 字节做更多计算。

参考（中文）：

- Tile 定义与约束：`docs/coding/Tile_zh.md`
- GlobalTensor 视图与布局：`docs/coding/GlobalTensor_zh.md`

## 5. 数据搬运：减少流量并避免冗余变换

常见收益点：

- **复用**：每次 DMA 暂存更多数据并复用（例如 GEMM 的 stepK 缓存）。
- **更少的变换**：若能一开始就选对输入布局，尽量避免 `TTRANS`/`TRESHAPE`/额外 `TEXTRACT`。
- **简化输出**：写回 GM 友好的布局，并与下游消费模式匹配。

若 kernel 同时使用 Cube 与 Vector 阶段，尽量让中间数据保持在一个能减少阶段间转换成本的布局上。

## 6. 重叠与同步：让流水线保持满载

手工 kernel 常通过显式双缓冲与 event/flag 同步实现重叠：

- 当前 `TMATMUL` 运行时启动下一次 `TLOAD`
- 当前计算运行时做下一次 `TEXTRACT`
- 当前 `TSTORE` 运行时准备下一轮计算

经验法则：

- 只等待**真实依赖**（producer/consumer），避免在稳态循环中做全局性 “drain everything” 等待。
- 将流水线视为 **warm-up / steady state / drain**；优先把稳态调顺。

参考（中文）：

- 事件与同步模型：`docs/coding/Event_zh.md`

## 7. 示例驱动的深度指南

以下目录包含最完整的“如何调优”记录，并与真实代码绑定：

- GEMM（tiling、stepK 缓存、双缓冲）：
  - [`kernels/manual/a2a3/gemm_performance/README_zh.md`](../../kernels/manual/a2a3/gemm_performance/README_zh.md)
  - Kernel code：`kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`
- Flash Attention（分阶段 softmax、tiled QK/PV、逐阶段调优）：
  - [`kernels/manual/common/flash_atten/README_zh.md`](../../kernels/manual/common/flash_atten/README_zh.md)
  - Kernel code：`kernels/manual/common/flash_atten/fa_performance_kernel.cpp`
  - [`kernels/manual/common/flash_atten/README_zh.md`](../../kernels/manual/common/flash_atten/README_zh.md)
  - Kernel code：`kernels/manual/common/flash_atten/fa_performance_kernel.cpp`

## 8. 常见故障模式（以及处理方式）

- **某个 shape 很快，但其他 shape 很慢**
  - 针对不同 shape 类（小/中/大）分别重调核划分与 tile 尺寸。
  - 注意 “tile 太小”（开销主导）与 “tile 太大”（喂不饱/重叠断裂）。

- **TLOAD 占比高 + TMATMUL 占比低**
  - 提升复用（更大 tile 或更好的缓存），或改进重叠（双缓冲正确性）。
  - 减少冗余加载（例如避免在内层循环重复加载同一 panel）。

- **Transform 占主导（`TEXTRACT`/`TMOV`/layout）**
  - 提升每次 transform 对应的计算量（每次提取更多工作）。
  - 优先选择能减少 transform 次数的布局。

- **改流水线后正确性出问题**
  - 重新核对依赖边，确保每个 consumer 等待了正确的 producer event/flag。
  - 先用小 shape 验证；在优化前加入更强的正确性检查。
