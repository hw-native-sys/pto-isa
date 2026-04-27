# PTO ISA 概述

本文档为根据 `docs/isa/manifest.yaml` 自动生成的 ISA 索引。

## 文档目录

| 领域 | 页面 | 描述 |
| --- | --- | --- |
| 概述 | [`docs/README_zh.md`](README_zh.md) | PTO ISA 指南入口与导航。 |
| 概述 | [`docs/PTOISA_zh.md`](PTOISA_zh.md) | 本页（概述 + 全量指令索引）。 |
| ISA 参考 | [`docs/isa/README_zh.md`](isa/scalar/ops/micro-instruction/README_zh.md) | 每条指令参考目录。 |
| ISA 参考 | [`docs/isa/conventions_zh.md`](isa/conventions_zh.md) | 通用符号、操作数、事件与修饰符。 |
| 语法与操作数 | [`docs/isa/syntax-and-operands/assembly-model_zh.md`](isa/syntax-and-operands/assembly-model_zh.md) | ISA 手册内的规范 PTO-AS 写法与操作数语法。 |
| 权威源 | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header_zh.md) | C++ intrinsic API（权威来源）。 |
| PTO auto 模式 | [`docs/auto_mode/README_zh.md`](README_zh.md) | PTO auto模式文档入口 |

## 指令索引（全部 PTO 指令）

| 分类 | 指令 | 描述 |
| --- | --- | --- |
| 同步 | [`TSYNC`](isa/tile/ops/sync-and-config/tsync_zh.md) | 同步 PTO 执行（等待事件或插入每操作流水线屏障）。 |
| 手动 / 资源绑定 | [`TASSIGN`](isa/tile/ops/sync-and-config/tassign_zh.md) | 将 Tile 对象绑定到实现定义的片上地址（手动放置）。 |
| 手动 / 资源绑定 | [`pto.setfmatrix`](isa/tile/ops/sync-and-config/setfmatrix.md) | 为类 IMG2COL 操作设置 FMATRIX 寄存器。 |
| 手动 / 资源绑定 | [`pto.set_img2col_rpt`](isa/tile/ops/sync-and-config/set-img2col-rpt.md) | 从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。 |
| 手动 / 资源绑定 | [`pto.set_img2col_padding`](isa/tile/ops/sync-and-config/set-img2col-padding.md) | 从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据。 |
| 逐元素（Tile-Tile） | [`TADD`](isa/tile/ops/elementwise-tile-tile/tadd_zh.md) | 两个 Tile 的逐元素加法。 |
| 逐元素（Tile-Tile） | [`TABS`](isa/tile/ops/elementwise-tile-tile/tabs_zh.md) | Tile 的逐元素绝对值。 |
| 逐元素（Tile-Tile） | [`TAND`](isa/tile/ops/elementwise-tile-tile/tand_zh.md) | 两个 Tile 的逐元素按位与。 |
| 逐元素（Tile-Tile） | [`TOR`](isa/tile/ops/elementwise-tile-tile/tor_zh.md) | 两个 Tile 的逐元素按位或。 |
| 逐元素（Tile-Tile） | [`TSUB`](isa/tile/ops/elementwise-tile-tile/tsub_zh.md) | 两个 Tile 的逐元素减法。 |
| 逐元素（Tile-Tile） | [`TMUL`](isa/tile/ops/elementwise-tile-tile/tmul_zh.md) | 两个 Tile 的逐元素乘法。 |
| 逐元素（Tile-Tile） | [`TMIN`](isa/tile/ops/elementwise-tile-tile/tmin_zh.md) | 两个 Tile 的逐元素最小值。 |
| 逐元素（Tile-Tile） | [`TMAX`](isa/tile/ops/elementwise-tile-tile/tmax_zh.md) | 两个 Tile 的逐元素最大值。 |
| 逐元素（Tile-Tile） | [`TCMP`](isa/tile/ops/elementwise-tile-tile/tcmp_zh.md) | 比较两个 Tile 并写入一个打包的谓词掩码。 |
| 逐元素（Tile-Tile） | [`TDIV`](isa/tile/ops/elementwise-tile-tile/tdiv_zh.md) | 两个 Tile 的逐元素除法。 |
| 逐元素（Tile-Tile） | [`TSHL`](isa/tile/ops/elementwise-tile-tile/tshl_zh.md) | 两个 Tile 的逐元素左移。 |
| 逐元素（Tile-Tile） | [`TSHR`](isa/tile/ops/elementwise-tile-tile/tshr_zh.md) | 两个 Tile 的逐元素右移。 |
| 逐元素（Tile-Tile） | [`TXOR`](isa/tile/ops/elementwise-tile-tile/txor_zh.md) | 两个 Tile 的逐元素按位异或。 |
| 逐元素（Tile-Tile） | [`TLOG`](isa/tile/ops/elementwise-tile-tile/tlog_zh.md) | Tile 的逐元素自然对数。 |
| 逐元素（Tile-Tile） | [`TRECIP`](isa/tile/ops/elementwise-tile-tile/trecip_zh.md) | Tile 的逐元素倒数。 |
| 逐元素（Tile-Tile） | [`TPRELU`](isa/tile/ops/elementwise-tile-tile/tprelu_zh.md) | 带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。 |
| 逐元素（Tile-Tile） | [`TADDC`](isa/tile/ops/elementwise-tile-tile/taddc_zh.md) | 三元逐元素加法：`src0 + src1 + src2`。 |
| 逐元素（Tile-Tile） | [`TSUBC`](isa/tile/ops/elementwise-tile-tile/tsubc_zh.md) | 三元逐元素运算：`src0 - src1 + src2`。 |
| 逐元素（Tile-Tile） | [`TCVT`](isa/tile/ops/elementwise-tile-tile/tcvt_zh.md) | 带指定舍入模式的逐元素类型转换。 |
| 逐元素（Tile-Tile） | [`TSEL`](isa/tile/ops/elementwise-tile-tile/tsel_zh.md) | 使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。 |
| 逐元素（Tile-Tile） | [`TRSQRT`](isa/tile/ops/elementwise-tile-tile/trsqrt_zh.md) | 逐元素倒数平方根。 |
| 逐元素（Tile-Tile） | [`TSQRT`](isa/tile/ops/elementwise-tile-tile/tsqrt_zh.md) | 逐元素平方根。 |
| 逐元素（Tile-Tile） | [`TEXP`](isa/tile/ops/elementwise-tile-tile/texp_zh.md) | 逐元素指数运算。 |
| 逐元素（Tile-Tile） | [`TNOT`](isa/tile/ops/elementwise-tile-tile/tnot_zh.md) | Tile 的逐元素按位取反。 |
| 逐元素（Tile-Tile） | [`TRELU`](isa/tile/ops/elementwise-tile-tile/trelu_zh.md) | Tile 的逐元素 ReLU。 |
| 逐元素（Tile-Tile） | [`TNEG`](isa/tile/ops/elementwise-tile-tile/tneg_zh.md) | Tile 的逐元素取负。 |
| 逐元素（Tile-Tile） | [`TREM`](isa/tile/ops/elementwise-tile-tile/trem_zh.md) | 两个 Tile 的逐元素余数，余数符号与除数相同。 |
| 逐元素（Tile-Tile） | [`TFMOD`](isa/tile/ops/elementwise-tile-tile/tfmod_zh.md) | 两个 Tile 的逐元素余数，余数符号与被除数相同。 |
| Tile-标量 / Tile-立即数 | [`TEXPANDS`](isa/tile/ops/tile-scalar-and-immediate/texpands_zh.md) | 将标量广播到目标 Tile 中。 |
| Tile-标量 / Tile-立即数 | [`TCMPS`](isa/tile/ops/elementwise-tile-tile/tcmps_zh.md) | 将 Tile 与标量比较并写入逐元素比较结果。 |
| Tile-标量 / Tile-立即数 | [`TSELS`](isa/tile/ops/elementwise-tile-tile/tsels_zh.md) | 使用掩码 Tile 在源 Tile 和标量之间进行选择（源 Tile 逐元素选择）。 |
| Tile-标量 / Tile-立即数 | [`TMINS`](isa/tile/ops/elementwise-tile-tile/tmins_zh.md) | Tile 与标量的逐元素最小值。 |
| Tile-标量 / Tile-立即数 | [`TADDS`](isa/tile/ops/elementwise-tile-tile/tadds_zh.md) | Tile 与标量的逐元素加法。 |
| Tile-标量 / Tile-立即数 | [`TSUBS`](isa/tile/ops/elementwise-tile-tile/tsubs_zh.md) | 从 Tile 中逐元素减去一个标量。 |
| Tile-标量 / Tile-立即数 | [`TDIVS`](isa/tile/ops/elementwise-tile-tile/tdivs_zh.md) | 与标量的逐元素除法（Tile/标量 或 标量/Tile）。 |
| Tile-标量 / Tile-立即数 | [`TMULS`](isa/tile/ops/elementwise-tile-tile/tmuls_zh.md) | Tile 与标量的逐元素乘法。 |
| Tile-标量 / Tile-立即数 | [`TFMODS`](isa/tile/ops/elementwise-tile-tile/tfmods_zh.md) | 与标量的逐元素余数：`fmod(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TREMS`](isa/tile/ops/elementwise-tile-tile/trems_zh.md) | 与标量的逐元素余数：`remainder(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TMAXS`](isa/tile/ops/elementwise-tile-tile/tmaxs_zh.md) | Tile 与标量的逐元素最大值：`max(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TANDS`](isa/tile/ops/elementwise-tile-tile/tands_zh.md) | Tile 与标量的逐元素按位与。 |
| Tile-标量 / Tile-立即数 | [`TORS`](isa/tile/ops/elementwise-tile-tile/tors_zh.md) | Tile 与标量的逐元素按位或。 |
| Tile-标量 / Tile-立即数 | [`TSHLS`](isa/tile/ops/tile-scalar-and-immediate/tshls_zh.md) | Tile 按标量逐元素左移。 |
| Tile-标量 / Tile-立即数 | [`TSHRS`](isa/tile/ops/tile-scalar-and-immediate/tshrs_zh.md) | Tile 按标量逐元素右移。 |
| Tile-标量 / Tile-立即数 | [`TXORS`](isa/tile/ops/elementwise-tile-tile/txors_zh.md) | Tile 与标量的逐元素按位异或。 |
| Tile-标量 / Tile-立即数 | [`TLRELU`](isa/tile/ops/tile-scalar-and-immediate/tlrelu_zh.md) | 带标量斜率的 Leaky ReLU。 |
| Tile-标量 / Tile-立即数 | [`TADDSC`](isa/tile/ops/tile-scalar-and-immediate/taddsc_zh.md) | 与标量和第二个 Tile 的融合逐元素加法：`src0 + scalar + src1`。 |
| Tile-标量 / Tile-立即数 | [`TSUBSC`](isa/tile/ops/tile-scalar-and-immediate/tsubsc_zh.md) | 融合逐元素运算：`src0 - scalar + src1`。 |
| 轴归约 / 扩展 | [`TROWSUM`](isa/tile/ops/reduce-and-expand/trowsum_zh.md) | 通过对列求和来归约每一行。 |
| 轴归约 / 扩展 | [`TROWPROD`](isa/tile/ops/reduce-and-expand/trowprod_zh.md) | 通过跨列乘积来归约每一行。 |
| 轴归约 / 扩展 | [`TCOLSUM`](isa/tile/ops/reduce-and-expand/tcolsum_zh.md) | 通过对行求和来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLPROD`](isa/tile/ops/reduce-and-expand/tcolprod_zh.md) | 通过跨行乘积来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLMAX`](isa/tile/ops/reduce-and-expand/tcolmax_zh.md) | 通过取行间最大值来归约每一列。 |
| 轴归约 / 扩展 | [`TROWMAX`](isa/tile/ops/reduce-and-expand/trowmax_zh.md) | 通过取列间最大值来归约每一行。 |
| 轴归约 / 扩展 | [`TROWMIN`](isa/tile/ops/reduce-and-expand/trowmin_zh.md) | 通过取列间最小值来归约每一行。 |
| 轴归约 / 扩展 | [`TROWARGMAX`](isa/tile/ops/reduce-and-expand/trowargmax_zh.md) | 获取每行最大值对应列索引。 |
| 轴归约 / 扩展 | [`TROWARGMIN`](isa/tile/ops/reduce-and-expand/trowargmin_zh.md) | 获取每行最小值对应列索引。 |
| 轴归约 / 扩展 | [`TCOLARGMAX`](isa/tile/ops/reduce-and-expand/tcolargmax_zh.md) | 获取每列最大值对应行索引。 |
| 轴归约 / 扩展 | [`TCOLARGMIN`](isa/tile/ops/reduce-and-expand/tcolargmin_zh.md) | 获取每列最小值对应行索引。 |
| 轴归约 / 扩展 | [`TROWEXPAND`](isa/tile/ops/reduce-and-expand/trowexpand_zh.md) | 将每个源行的第一个元素广播到目标行中。 |
| 轴归约 / 扩展 | [`TROWEXPANDDIV`](isa/tile/ops/reduce-and-expand/trowexpanddiv_zh.md) | 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDMUL`](isa/tile/ops/reduce-and-expand/trowexpandmul_zh.md) | 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDSUB`](isa/tile/ops/reduce-and-expand/trowexpandsub_zh.md) | 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDADD`](isa/tile/ops/reduce-and-expand/trowexpandadd_zh.md) | 行广播加法：加上一个每行标量向量。 |
| 轴归约 / 扩展 | [`TROWEXPANDMAX`](isa/tile/ops/reduce-and-expand/trowexpandmax_zh.md) | 行广播最大值：与每行标量向量取最大值。 |
| 轴归约 / 扩展 | [`TROWEXPANDMIN`](isa/tile/ops/reduce-and-expand/trowexpandmin_zh.md) | 行广播最小值：与每行标量向量取最小值。 |
| 轴归约 / 扩展 | [`TROWEXPANDEXPDIF`](isa/tile/ops/reduce-and-expand/trowexpandexpdif_zh.md) | 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。 |
| 轴归约 / 扩展 | [`TCOLMIN`](isa/tile/ops/reduce-and-expand/tcolmin_zh.md) | 通过取行间最小值来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLEXPAND`](isa/tile/ops/reduce-and-expand/tcolexpand_zh.md) | 将每个源列的第一个元素广播到目标列中。 |
| 轴归约 / 扩展 | [`TCOLEXPANDDIV`](isa/tile/ops/reduce-and-expand/tcolexpanddiv_zh.md) | 列广播除法：将每一列除以一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMUL`](isa/tile/ops/reduce-and-expand/tcolexpandmul_zh.md) | 列广播乘法：将每一列乘以一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDADD`](isa/tile/ops/reduce-and-expand/tcolexpandadd_zh.md) | 列广播加法：对每一列加上每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMAX`](isa/tile/ops/reduce-and-expand/tcolexpandmax_zh.md) | 列广播最大值：与每列标量向量取最大值。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMIN`](isa/tile/ops/reduce-and-expand/tcolexpandmin_zh.md) | 列广播最小值：与每列标量向量取最小值。 |
| 轴归约 / 扩展 | [`TCOLEXPANDSUB`](isa/tile/ops/reduce-and-expand/tcolexpandsub_zh.md) | 列广播减法：从每一列中减去一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDEXPDIF`](isa/tile/ops/reduce-and-expand/tcolexpandexpdif_zh.md) | 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。 |
| 内存（GM <-> Tile） | [`TLOAD`](isa/tile/ops/memory-and-data-movement/tload_zh.md) | 从 GlobalTensor (GM) 加载数据到 Tile。 |
| 内存（GM <-> Tile） | [`TPREFETCH`](isa/tile/ops/memory-and-data-movement/tprefetch_zh.md) | 将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。 |
| 内存（GM <-> Tile） | [`TSTORE`](isa/tile/ops/memory-and-data-movement/tstore_zh.md) | 将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。 |
| 内存（GM <-> Tile） | [`TSTORE_FP`](isa/tile/ops/memory-and-data-movement/tstore_zh.md) | 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 存储到全局内存。 |
| 内存（GM <-> Tile） | [`MGATHER`](isa/tile/ops/memory-and-data-movement/mgather_zh.md) | 使用逐元素索引从全局内存收集加载元素到 Tile 中。 |
| 内存（GM <-> Tile） | [`MSCATTER`](isa/tile/ops/memory-and-data-movement/mscatter_zh.md) | 使用逐元素索引将 Tile 中的元素散播存储到全局内存。 |
| 矩阵乘 | [`TGEMV_MX`](isa/tile/ops/matrix-and-matrix-vector/tgemv-mx_zh.md) | 带缩放 Tile 的 GEMV 变体，支持混合精度/量化矩阵向量计算。 |
| 矩阵乘 | [`TMATMUL_MX`](isa/tile/ops/matrix-and-matrix-vector/tmatmul-mx_zh.md) | 带额外缩放 Tile 的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。 |
| 矩阵乘 | [`TMATMUL`](isa/tile/ops/matrix-and-matrix-vector/tmatmul_zh.md) | 矩阵乘法 (GEMM)，生成累加器/输出 Tile。 |
| 矩阵乘 | [`TMATMUL_ACC`](isa/tile/ops/matrix-and-matrix-vector/tmatmul-acc_zh.md) | 带累加器输入的矩阵乘法（融合累加）。 |
| 矩阵乘 | [`TMATMUL_BIAS`](isa/tile/ops/matrix-and-matrix-vector/tmatmul-bias_zh.md) | 带偏置加法的矩阵乘法。 |
| 矩阵乘 | [`TGEMV`](isa/tile/ops/matrix-and-matrix-vector/tgemv_zh.md) | 通用矩阵-向量乘法，生成累加器/输出 Tile。 |
| 矩阵乘 | [`TGEMV_ACC`](isa/tile/ops/matrix-and-matrix-vector/tgemv-acc_zh.md) | 带显式累加器输入/输出 Tile 的 GEMV。 |
| 矩阵乘 | [`TGEMV_BIAS`](isa/tile/ops/matrix-and-matrix-vector/tgemv-bias_zh.md) | 带偏置加法的 GEMV。 |
| 数据搬运 / 布局 | [`TEXTRACT`](isa/tile/ops/layout-and-rearrangement/textract_zh.md) | 从源 Tile 中提取子 Tile。 |
| 数据搬运 / 布局 | [`TEXTRACT_FP`](isa/tile/ops/layout-and-rearrangement/textract_zh.md) | 带 fp/缩放 Tile 的提取（向量量化参数）。 |
| 数据搬运 / 布局 | [`TIMG2COL`](isa/tile/ops/layout-and-rearrangement/timg2col_zh.md) | 用于类卷积工作负载的图像到列变换。 |
| 数据搬运 / 布局 | [`TINSERT`](isa/tile/ops/layout-and-rearrangement/tinsert_zh.md) | 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。 |
| 数据搬运 / 布局 | [`TINSERT_FP`](isa/tile/ops/layout-and-rearrangement/tinsert_zh.md) | 带 fp/缩放 Tile 的插入（向量量化参数）。 |
| 数据搬运 / 布局 | [`TFILLPAD`](isa/tile/ops/layout-and-rearrangement/tfillpad_zh.md) | 复制 Tile 并在有效区域外使用编译时填充值进行填充。 |
| 数据搬运 / 布局 | [`TFILLPAD_INPLACE`](isa/tile/ops/layout-and-rearrangement/tfillpad-inplace_zh.md) | 原地填充/填充变体。 |
| 数据搬运 / 布局 | [`TFILLPAD_EXPAND`](isa/tile/ops/layout-and-rearrangement/tfillpad-expand_zh.md) | 填充/填充时允许目标大于源。 |
| 数据搬运 / 布局 | [`TMOV`](isa/tile/ops/layout-and-rearrangement/tmov_zh.md) | 在 Tile 之间移动/复制，可选应用实现定义的转换模式。 |
| 数据搬运 / 布局 | [`TMOV_FP`](isa/tile/ops/layout-and-rearrangement/tmov_zh.md) | 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 移动/转换到目标 Tile。 |
| 数据搬运 / 布局 | [`TRESHAPE`](isa/tile/ops/layout-and-rearrangement/treshape_zh.md) | 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。 |
| 数据搬运 / 布局 | [`TTRANS`](isa/tile/ops/layout-and-rearrangement/ttrans_zh.md) | 使用实现定义的临时 Tile 进行转置。 |
| 数据搬运 / 布局 | [`TSUBVIEW`](isa/tile/sync-and-config_zh.md) | 表达一个tile是另一个tile的subview |
| 数据搬运 / 布局 | [`TGET_SCALE_ADDR`](isa/tile/sync-and-config_zh.md) | 将输出tile的片上内存值绑定为扩展后的输入tile内存的值。 |
| 复杂指令 | [`TPRINT`](isa/tile/ops/irregular-and-complex/tprint_zh.md) | 调试/打印 Tile 中的元素（实现定义）。 |
| 复杂指令 | [`TMRGSORT`](isa/tile/ops/irregular-and-complex/tmrgsort_zh.md) | 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。 |
| 复杂指令 | [`TSORT32`](isa/tile/ops/irregular-and-complex/tsort32_zh.md) | 对 `src` 的每个 32 元素块连同对应的 `idx` 条目一起排序，并输出排序后的 value-index 对。 |
| 复杂指令 | [`TGATHER`](isa/tile/ops/irregular-and-complex/tgather_zh.md) | 使用索引 Tile 或编译时掩码模式来收集/选择元素。 |
| 复杂指令 | [`TCI`](isa/tile/ops/irregular-and-complex/tci_zh.md) | 生成连续整数序列到目标 Tile 中。 |
| 复杂指令 | [`TTRI`](isa/tile/ops/irregular-and-complex/ttri_zh.md) | 生成三角（下/上）掩码 Tile。 |
| 复杂指令 | [`TRANDOM`](isa/tile/irregular-and-complex_zh.md) | 使用基于计数器的密码算法在目标 Tile 中生成随机数。 |
| 复杂指令 | [`TPARTADD`](isa/tile/ops/irregular-and-complex/tpartadd_zh.md) | 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTMUL`](isa/tile/ops/irregular-and-complex/tpartmul_zh.md) | 部分逐元素乘法，对有效区域不一致的处理为实现定义。 |
| 复杂指令 | [`TPARTMAX`](isa/tile/ops/irregular-and-complex/tpartmax_zh.md) | 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTMIN`](isa/tile/ops/irregular-and-complex/tpartmin_zh.md) | 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TGATHERB`](isa/tile/ops/irregular-and-complex/tgatherb_zh.md) | 使用字节偏移量收集元素。 |
| 复杂指令 | [`TSCATTER`](isa/tile/ops/irregular-and-complex/tscatter_zh.md) | 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。 |
| 复杂指令 | [`TQUANT`](isa/tile/ops/irregular-and-complex/tquant_zh.md) | 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。 |
| 通信 | [`TPUT`](isa/comm/communication-runtime_zh.md) | 远程写：将本地数据传输到远端 NPU 内存（GM → UB → GM）。 |
| 通信 | [`TGET`](isa/comm/communication-runtime_zh.md) | 远程读：将远端 NPU 数据读取到本地内存（GM → UB → GM）。 |
| 通信 | [`TPUT_ASYNC`](isa/comm/communication-runtime_zh.md) | 异步远程写（本地 GM → DMA 引擎 → 远端 GM）。 |
| 通信 | [`TGET_ASYNC`](isa/comm/communication-runtime_zh.md) | 异步远程读（远端 GM → DMA 引擎 → 本地 GM）。 |
| 通信 | [`TNOTIFY`](isa/comm/communication-runtime_zh.md) | 向远端 NPU 发送标志通知。 |
| 通信 | [`TWAIT`](isa/comm/communication-runtime_zh.md) | 阻塞等待，直到信号满足比较条件。 |
| 通信 | [`TTEST`](isa/comm/communication-runtime_zh.md) | 非阻塞检测信号是否满足比较条件。 |
| 通信 | [`TGATHER`](isa/comm/communication-runtime_zh.md) | 从所有 rank 收集数据并沿 DIM_3 拼接。 |
| 通信 | [`TSCATTER`](isa/comm/communication-runtime_zh.md) | 将数据沿 DIM_3 拆分并分发到所有 rank。 |
| 通信 | [`TREDUCE`](isa/comm/communication-runtime_zh.md) | 从所有 rank 收集数据并逐元素归约到本地。 |
| 通信 | [`TBROADCAST`](isa/comm/communication-runtime_zh.md) | 将当前 NPU 的数据广播到所有 rank。 |
