<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 手册与参考

本文档目录是 PTO ISA 的权威文档树。它将架构手册、指令集指南、家族契约和精确的指令参考分组整合在同一个位置。

## PTO ISA 中的文本汇编

本树是权威的 PTO ISA 手册。文本汇编拼写属于 PTO ISA 的语法层，而非第二份并行的架构手册。

## 手动 / 资源绑定
- [TASSIGN](TASSIGN_zh.md) - 将 Tile 对象绑定到实现定义的片上地址（手动放置）。
- [SETFMATRIX](SETFMATRIX_zh.md) - 为类 IMG2COL 操作设置 FMATRIX 寄存器。
- [SET_IMG2COL_RPT](SET_IMG2COL_RPT_zh.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。
- [SET_IMG2COL_PADDING](SET_IMG2COL_PADDING_zh.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据。

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

## 复杂指令
- [TPRINT](TPRINT_zh.md) - 调试/打印 Tile 中的元素（实现定义）。
- [TMRGSORT](TMRGSORT_zh.md) - 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。
- [TSORT32](TSORT32_zh.md) - 对 `src` 的每个 32 元素块，与 `idx` 中对应的索引一起进行排序，并将排序后的值-索引对写入 `dst`。
- [TGATHER](TGATHER_zh.md) - 使用索引 Tile 或编译时掩码模式来收集/选择元素。
- [TCI](TCI_zh.md) - 生成连续整数序列到目标 Tile 中。
- [TTRI](TTRI_zh.md) - 生成三角（下/上）掩码 Tile。
- [TRANDOM](TRANDOM_zh.md) - 使用基于计数器的密码算法在目标 Tile 中生成随机数。
- [TPARTADD](TPARTADD_zh.md) - 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMUL](TPARTMUL_zh.md) - 部分逐元素乘法，对有效区域不一致的处理为实现定义。
- [TPARTMAX](TPARTMAX_zh.md) - 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMIN](TPARTMIN_zh.md) - 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTARGMAX](TPARTARGMAX_zh.md) - 部分逐元素最大值选择并返回对应索引（argmax），对不匹配的有效区域具有实现定义的处理方式。
- [TPARTARGMIN](TPARTARGMIN_zh.md) - 部分逐元素最小值选择并返回对应索引（argmin），对不匹配的有效区域具有实现定义的处理方式。
- [TGATHERB](TGATHERB_zh.md) - 使用字节偏移量收集元素。
- [TSCATTER](TSCATTER_zh.md) - 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。
- [TQUANT](TQUANT_zh.md) - 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。

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
