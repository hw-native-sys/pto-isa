<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 手册与参考

本文档目录是 PTO ISA 的权威文档树。它将架构手册、指令集指南、家族契约和精确的指令参考分组整合在同一个位置。

## PTO ISA 中的文本汇编

本树是权威的 PTO ISA 手册。文本汇编拼写属于 PTO ISA 的语法层，而非第二份并行的架构手册。

- PTO ISA 定义了架构可见的语义、合法性、状态、排序、目标 profile 边界，以及 tile、vector、scalar、communication、系统调度操作的可见行为
- PTO-AS 是用于编写这些操作和操作数的汇编拼写。它是 PTO ISA 的表达方式的一部分，而非具有不同语义的分立 ISA

## 逐元素（Tile-Tile）
- [TADD](TADD_zh.md) - 两个 Tile 的逐元素加法。
- [TABS](TABS_zh.md) - Tile 的逐元素绝对值。
- [TAND](TAND_zh.md) - 两个 Tile 的逐元素按位与。
- [TOR](TOR_zh.md) - 两个 Tile 的逐元素按位或。
- [TSUB](TSUB_zh.md) - 两个 Tile 的逐元素减法。
- [TMUL](TMUL_zh.md) - 两个 Tile 的逐元素乘法。
- [TMIN](TMIN_zh.md) - 两个 Tile 的逐元素最小值。
- [TMAX](TMAX_zh.md) - 两个 Tile 的逐元素最大值。
- [TCMP](TCMP_zh.md) - 比较两个 Tile 并写入一个打包的谓词掩码。
- [TDIV](TDIV_zh.md) - 两个 Tile 的逐元素除法。
- [TSHL](TSHL_zh.md) - 两个 Tile 的逐元素左移。
- [TSHR](TSHR_zh.md) - 两个 Tile 的逐元素右移。
- [TXOR](TXOR_zh.md) - 两个 Tile 的逐元素按位异或。
- [TLOG](TLOG_zh.md) - Tile 的逐元素自然对数。
- [TRECIP](TRECIP_zh.md) - Tile 的逐元素倒数。
- [TPRELU](TPRELU_zh.md) - 带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。
- [TADDC](TADDC_zh.md) - 三元逐元素加法：`src0 + src1 + src2`。
- [TSUBC](TSUBC_zh.md) - 三元逐元素运算：`src0 - src1 + src2`。
- [TCVT](TCVT_zh.md) - 带指定舍入模式的逐元素类型转换。
- [TSEL](TSEL_zh.md) - 使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。
- [TRSQRT](TRSQRT_zh.md) - 逐元素倒数平方根。
- [TSQRT](TSQRT_zh.md) - 逐元素平方根。
- [TEXP](TEXP_zh.md) - 逐元素指数运算。
- [TNOT](TNOT_zh.md) - Tile 的逐元素按位取反。
- [TRELU](TRELU_zh.md) - Tile 的逐元素 ReLU。
- [TNEG](TNEG_zh.md) - Tile 的逐元素取负。
- [TREM](TREM_zh.md) - 两个 Tile 的逐元素余数，余数符号与除数相同。
- [TFMOD](TFMOD_zh.md) - 两个 Tile 的逐元素余数，余数符号与被除数相同。
- [TPOW](TPOW_zh.md) - 两个 Tile 的逐元素幂运算。

## Tile-标量 / Tile-立即数
- [TEXPANDS](TEXPANDS_zh.md) - 将标量广播到目标 Tile 中。
- [TCMPS](TCMPS_zh.md) - 将 Tile 与标量比较并写入逐元素比较结果。
- [TSELS](TSELS_zh.md) - 使用掩码 Tile 在源 Tile 和标量之间进行选择（源 Tile 逐元素选择）。
- [TMINS](TMINS_zh.md) - Tile 与标量的逐元素最小值。
- [TADDS](TADDS_zh.md) - Tile 与标量的逐元素加法。
- [TSUBS](TSUBS_zh.md) - 从 Tile 中逐元素减去一个标量。
- [TDIVS](TDIVS_zh.md) - 与标量的逐元素除法（Tile/标量 或 标量/Tile）。
- [TMULS](TMULS_zh.md) - Tile 与标量的逐元素乘法。
- [TFMODS](TFMODS_zh.md) - 与标量的逐元素余数：`fmod(src, scalar)`。
- [TREMS](TREMS_zh.md) - 与标量的逐元素余数：`remainder(src, scalar)`。
- [TMAXS](TMAXS_zh.md) - Tile 与标量的逐元素最大值：`max(src, scalar)`。
- [TANDS](TANDS_zh.md) - Tile 与标量的逐元素按位与。
- [TORS](TORS_zh.md) - Tile 与标量的逐元素按位或。
- [TSHLS](TSHLS_zh.md) - Tile 按标量逐元素左移。
- [TSHRS](TSHRS_zh.md) - Tile 按标量逐元素右移。
- [TXORS](TXORS_zh.md) - Tile 与标量的逐元素按位异或。
- [TLRELU](TLRELU_zh.md) - 带标量斜率的 Leaky ReLU。
- [TADDSC](TADDSC_zh.md) - 与标量和第二个 Tile 的融合逐元素加法：`src0 + scalar + src1`。
- [TSUBSC](TSUBSC_zh.md) - 融合逐元素运算：`src0 - scalar + src1`。
- [TPOWS](TPOWS_zh.md) - Tile 逐元素与标量幂运算。

## 模型层次

阅读顺序与手册章节地图一致：先编程模型与机器模型，再语法与状态，再内存，最后是操作码参考。

- [编程模型](programming-model/tiles-and-valid-regions_zh.md)
- [机器模型](machine-model/execution-agents_zh.md)
- [语法与操作数](syntax-and-operands/assembly-model_zh.md)
- [类型系统](state-and-types/type-system_zh.md)
- [位置意图与合法性](state-and-types/location-intent-and-legality_zh.md)
- [内存模型](memory-model/consistency-baseline_zh.md)

- [指令集总览](instruction-families/README_zh.md)
- [指令族](instruction-families/README_zh.md)
- [指令描述格式](reference/format-of-instruction-descriptions_zh.md)
- [Tile 指令集参考](tile/README_zh.md)
- [Vector 指令集参考](vector/README_zh.md)
- [标量与控制参考](scalar/README_zh.md)
- [通信指令集参考](comm/README_zh.md)
- [系统调度指令集参考](system/README_zh.md)
- [通用约定](conventions_zh.md)

## 支持性参考

- [参考注释](reference/README_zh.md)（术语表、诊断、可移植性、规范来源）

## 核间通信

- [TALLOC](TALLOC_zh.md) - 将 TPipe FIFO 槽位分配为一个 GlobalTensor 视图。
- [TPUSH](TPUSH_zh.md) - 将生产者 tile 推入 TPipe FIFO，用于 Cube-Vector 通信。
- [TPOP](TPOP_zh.md) - 从 TPipe FIFO 弹出消费者 tile/globalTensor，用于 Cube-Vector 通信。
- [TFREE](TFREE_zh.md) - 释放 TPipe 的 FIFO 空间；对于 TileData/GlobalTensor 的 TPOP 流程，该操作为空操作。

`tile/`、`vector/`、`scalar/`、`comm/` 和 `system/` 下的分组指令集树是权威的 PTO ISA 路径。

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
- `docs/isa/comm/`
- `docs/isa/system/ops/`
