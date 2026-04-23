# PTO ISA 概述

本文档为根据 `docs/isa/manifest.yaml` 自动生成的 ISA 索引。

## 文档目录

| 领域 | 页面 | 描述 |
| --- | --- | --- |
| 概述 | [`docs/README_zh.md`](README_zh.md) | PTO ISA 指南入口与导航。 |
| 概述 | [`docs/PTOISA_zh.md`](PTOISA_zh.md) | 本页（概述 + 全量指令索引）。 |
| ISA 参考 | [`docs/isa/README_zh.md`](isa/README_zh.md) | 每条指令参考目录。 |
| ISA 参考 | [`docs/isa/conventions_zh.md`](isa/conventions_zh.md) | 通用符号、操作数、事件与修饰符。 |
| 汇编 (PTO-AS) | [`docs/assembly/PTO-AS_zh.md`](assembly/PTO-AS_zh.md) | PTO-AS 语法参考。 |
| 权威源 | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header_zh.md) | C++ intrinsic API（权威来源）。 |
| PTO auto 模式 | [`docs/auto_mode/README_zh.md`](README_zh.md) | PTO auto模式文档入口 |

## 指令索引（全部 PTO 指令）

| 分类 | 指令 | 描述 |
| --- | --- | --- |
| 同步 | [`TSYNC`](isa/TSYNC_zh.md) | 同步 PTO 执行（等待事件或插入每操作流水线屏障）。 |
| 手动 / 资源绑定 | [`TASSIGN`](isa/TASSIGN_zh.md) | 将 Tile 对象绑定到实现定义的片上地址（手动放置）。 |
| 手动 / 资源绑定 | [`TSETFMATRIX`](isa/TSETFMATRIX_zh.md) | 为类 IMG2COL 操作设置 FMATRIX 寄存器。 |
| 手动 / 资源绑定 | [`TSET_IMG2COL_RPT`](isa/TSET_IMG2COL_RPT_zh.md) | 从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。 |
| 手动 / 资源绑定 | [`TSET_IMG2COL_PADDING`](isa/TSET_IMG2COL_PADDING_zh.md) | 从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据。 |
| 逐元素（Tile-Tile） | [`TADD`](isa/TADD_zh.md) | 两个 Tile 的逐元素加法。 |
| 逐元素（Tile-Tile） | [`TABS`](isa/TABS_zh.md) | Tile 的逐元素绝对值。 |
| 逐元素（Tile-Tile） | [`TAND`](isa/TAND_zh.md) | 两个 Tile 的逐元素按位与。 |
| 逐元素（Tile-Tile） | [`TOR`](isa/TOR_zh.md) | 两个 Tile 的逐元素按位或。 |
| 逐元素（Tile-Tile） | [`TSUB`](isa/TSUB_zh.md) | 两个 Tile 的逐元素减法。 |
| 逐元素（Tile-Tile） | [`TMUL`](isa/TMUL_zh.md) | 两个 Tile 的逐元素乘法。 |
| 逐元素（Tile-Tile） | [`TMIN`](isa/TMIN_zh.md) | 两个 Tile 的逐元素最小值。 |
| 逐元素（Tile-Tile） | [`TMAX`](isa/TMAX_zh.md) | 两个 Tile 的逐元素最大值。 |
| 逐元素（Tile-Tile） | [`TCMP`](isa/TCMP_zh.md) | 比较两个 Tile 并写入一个打包的谓词掩码。 |
| 逐元素（Tile-Tile） | [`TDIV`](isa/TDIV_zh.md) | 两个 Tile 的逐元素除法。 |
| 逐元素（Tile-Tile） | [`TSHL`](isa/TSHL_zh.md) | 两个 Tile 的逐元素左移。 |
| 逐元素（Tile-Tile） | [`TSHR`](isa/TSHR_zh.md) | 两个 Tile 的逐元素右移。 |
| 逐元素（Tile-Tile） | [`TXOR`](isa/TXOR_zh.md) | 两个 Tile 的逐元素按位异或。 |
| 逐元素（Tile-Tile） | [`TLOG`](isa/TLOG_zh.md) | Tile 的逐元素自然对数。 |
| 逐元素（Tile-Tile） | [`TRECIP`](isa/TRECIP_zh.md) | Tile 的逐元素倒数。 |
| 逐元素（Tile-Tile） | [`TPRELU`](isa/TPRELU_zh.md) | 带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。 |
| 逐元素（Tile-Tile） | [`TADDC`](isa/TADDC_zh.md) | 三元逐元素加法：`src0 + src1 + src2`。 |
| 逐元素（Tile-Tile） | [`TSUBC`](isa/TSUBC_zh.md) | 三元逐元素运算：`src0 - src1 + src2`。 |
| 逐元素（Tile-Tile） | [`TCVT`](isa/TCVT_zh.md) | 带指定舍入模式的逐元素类型转换。 |
| 逐元素（Tile-Tile） | [`TSEL`](isa/TSEL_zh.md) | 使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。 |
| 逐元素（Tile-Tile） | [`TRSQRT`](isa/TRSQRT_zh.md) | 逐元素倒数平方根。 |
| 逐元素（Tile-Tile） | [`TSQRT`](isa/TSQRT_zh.md) | 逐元素平方根。 |
| 逐元素（Tile-Tile） | [`TEXP`](isa/TEXP_zh.md) | 逐元素指数运算。 |
| 逐元素（Tile-Tile） | [`TPOW`](isa/TPOW_zh.md) | 逐元素幂运算。 |
| 逐元素（Tile-Tile） | [`TNOT`](isa/TNOT_zh.md) | Tile 的逐元素按位取反。 |
| 逐元素（Tile-Tile） | [`TRELU`](isa/TRELU_zh.md) | Tile 的逐元素 ReLU。 |
| 逐元素（Tile-Tile） | [`TNEG`](isa/TNEG_zh.md) | Tile 的逐元素取负。 |
| 逐元素（Tile-Tile） | [`TREM`](isa/TREM_zh.md) | 两个 Tile 的逐元素余数，余数符号与除数相同。 |
| 逐元素（Tile-Tile） | [`TFMOD`](isa/TFMOD_zh.md) | 两个 Tile 的逐元素余数，余数符号与被除数相同。 |
| Tile-标量 / Tile-立即数 | [`TEXPANDS`](isa/TEXPANDS_zh.md) | 将标量广播到目标 Tile 中。 |
| Tile-标量 / Tile-立即数 | [`TCMPS`](isa/TCMPS_zh.md) | 将 Tile 与标量比较并写入逐元素比较结果。 |
| Tile-标量 / Tile-立即数 | [`TSELS`](isa/TSELS_zh.md) | 使用掩码 Tile 在源 Tile 和标量之间进行选择（源 Tile 逐元素选择）。 |
| Tile-标量 / Tile-立即数 | [`TMINS`](isa/TMINS_zh.md) | Tile 与标量的逐元素最小值。 |
| Tile-标量 / Tile-立即数 | [`TADDS`](isa/TADDS_zh.md) | Tile 与标量的逐元素加法。 |
| Tile-标量 / Tile-立即数 | [`TSUBS`](isa/TSUBS_zh.md) | 从 Tile 中逐元素减去一个标量。 |
| Tile-标量 / Tile-立即数 | [`TDIVS`](isa/TDIVS_zh.md) | 与标量的逐元素除法（Tile/标量 或 标量/Tile）。 |
| Tile-标量 / Tile-立即数 | [`TMULS`](isa/TMULS_zh.md) | Tile 与标量的逐元素乘法。 |
| Tile-标量 / Tile-立即数 | [`TFMODS`](isa/TFMODS_zh.md) | 与标量的逐元素余数：`fmod(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TREMS`](isa/TREMS_zh.md) | 与标量的逐元素余数：`remainder(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TMAXS`](isa/TMAXS_zh.md) | Tile 与标量的逐元素最大值：`max(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TANDS`](isa/TANDS_zh.md) | Tile 与标量的逐元素按位与。 |
| Tile-标量 / Tile-立即数 | [`TORS`](isa/TORS_zh.md) | Tile 与标量的逐元素按位或。 |
| Tile-标量 / Tile-立即数 | [`TSHLS`](isa/TSHLS_zh.md) | Tile 按标量逐元素左移。 |
| Tile-标量 / Tile-立即数 | [`TSHRS`](isa/TSHRS_zh.md) | Tile 按标量逐元素右移。 |
| Tile-标量 / Tile-立即数 | [`TXORS`](isa/TXORS_zh.md) | Tile 与标量的逐元素按位异或。 |
| Tile-标量 / Tile-立即数 | [`TLRELU`](isa/TLRELU_zh.md) | 带标量斜率的 Leaky ReLU。 |
| Tile-标量 / Tile-立即数 | [`TPOWS`](isa/TPOWS_zh.md) | Tile 逐元素与标量幂运算。 |
| Tile-标量 / Tile-立即数 | [`TADDSC`](isa/TADDSC_zh.md) | 与标量和第二个 Tile 的融合逐元素加法：`src0 + scalar + src1`。 |
| Tile-标量 / Tile-立即数 | [`TSUBSC`](isa/TSUBSC_zh.md) | 融合逐元素运算：`src0 - scalar + src1`。 |
| 轴归约 / 扩展 | [`TROWSUM`](isa/TROWSUM_zh.md) | 通过对列求和来归约每一行。 |
| 轴归约 / 扩展 | [`TROWPROD`](isa/TROWPROD_zh.md) | 通过跨列乘积来归约每一行。 |
| 轴归约 / 扩展 | [`TCOLSUM`](isa/TCOLSUM_zh.md) | 通过对行求和来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLPROD`](isa/TCOLPROD_zh.md) | 通过跨行乘积来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLMAX`](isa/TCOLMAX_zh.md) | 通过取行间最大值来归约每一列。 |
| 轴归约 / 扩展 | [`TROWMAX`](isa/TROWMAX_zh.md) | 通过取列间最大值来归约每一行。 |
| 轴归约 / 扩展 | [`TROWMIN`](isa/TROWMIN_zh.md) | 通过取列间最小值来归约每一行。 |
| 轴归约 / 扩展 | [`TROWARGMAX`](isa/TROWARGMAX_zh.md) | 获取每行最大值对应列索引。 |
| 轴归约 / 扩展 | [`TROWARGMIN`](isa/TROWARGMIN_zh.md) | 获取每行最小值对应列索引。 |
| 轴归约 / 扩展 | [`TCOLARGMAX`](isa/TCOLARGMAX_zh.md) | 获取每列最大值对应行索引。 |
| 轴归约 / 扩展 | [`TCOLARGMIN`](isa/TCOLARGMIN_zh.md) | 获取每列最小值对应行索引。 |
| 轴归约 / 扩展 | [`TROWEXPAND`](isa/TROWEXPAND_zh.md) | 将每个源行的第一个元素广播到目标行中。 |
| 轴归约 / 扩展 | [`TROWEXPANDDIV`](isa/TROWEXPANDDIV_zh.md) | 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDMUL`](isa/TROWEXPANDMUL_zh.md) | 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDSUB`](isa/TROWEXPANDSUB_zh.md) | 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDADD`](isa/TROWEXPANDADD_zh.md) | 行广播加法：加上一个每行标量向量。 |
| 轴归约 / 扩展 | [`TROWEXPANDMAX`](isa/TROWEXPANDMAX_zh.md) | 行广播最大值：与每行标量向量取最大值。 |
| 轴归约 / 扩展 | [`TROWEXPANDMIN`](isa/TROWEXPANDMIN_zh.md) | 行广播最小值：与每行标量向量取最小值。 |
| 轴归约 / 扩展 | [`TROWEXPANDEXPDIF`](isa/TROWEXPANDEXPDIF_zh.md) | 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。 |
| 轴归约 / 扩展 | [`TCOLMIN`](isa/TCOLMIN_zh.md) | 通过取行间最小值来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLEXPAND`](isa/TCOLEXPAND_zh.md) | 将每个源列的第一个元素广播到目标列中。 |
| 轴归约 / 扩展 | [`TCOLEXPANDDIV`](isa/TCOLEXPANDDIV_zh.md) | 列广播除法：将每一列除以一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMUL`](isa/TCOLEXPANDMUL_zh.md) | 列广播乘法：将每一列乘以一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDADD`](isa/TCOLEXPANDADD_zh.md) | 列广播加法：对每一列加上每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMAX`](isa/TCOLEXPANDMAX_zh.md) | 列广播最大值：与每列标量向量取最大值。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMIN`](isa/TCOLEXPANDMIN_zh.md) | 列广播最小值：与每列标量向量取最小值。 |
| 轴归约 / 扩展 | [`TCOLEXPANDSUB`](isa/TCOLEXPANDSUB_zh.md) | 列广播减法：从每一列中减去一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDEXPDIF`](isa/TCOLEXPANDEXPDIF_zh.md) | 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。 |
| 内存（GM <-> Tile） | [`TLOAD`](isa/TLOAD_zh.md) | 从 GlobalTensor (GM) 加载数据到 Tile。 |
| 内存（GM <-> Tile） | [`TPREFETCH`](isa/TPREFETCH_zh.md) | 将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。 |
| 内存（GM <-> Tile） | [`TSTORE`](isa/TSTORE_zh.md) | 将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。 |
| 内存（GM <-> Tile） | [`TSTORE_FP`](isa/TSTORE_FP_zh.md) | 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 存储到全局内存。 |
| 内存（GM <-> Tile） | [`MGATHER`](isa/MGATHER_zh.md) | 使用逐元素索引从全局内存收集加载元素到 Tile 中。 |
| 内存（GM <-> Tile） | [`MSCATTER`](isa/MSCATTER_zh.md) | 使用逐元素索引将 Tile 中的元素散播存储到全局内存。 |
| 矩阵乘 | [`TGEMV_MX`](isa/TGEMV_MX_zh.md) | 带缩放 Tile 的 GEMV 变体，支持混合精度/量化矩阵向量计算。 |
| 矩阵乘 | [`TMATMUL_MX`](isa/TMATMUL_MX_zh.md) | 带额外缩放 Tile 的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。 |
| 矩阵乘 | [`TMATMUL`](isa/TMATMUL_zh.md) | 矩阵乘法 (GEMM)，生成累加器/输出 Tile。 |
| 矩阵乘 | [`TMATMUL_ACC`](isa/TMATMUL_ACC_zh.md) | 带累加器输入的矩阵乘法（融合累加）。 |
| 矩阵乘 | [`TMATMUL_BIAS`](isa/TMATMUL_BIAS_zh.md) | 带偏置加法的矩阵乘法。 |
| 矩阵乘 | [`TGEMV`](isa/TGEMV_zh.md) | 通用矩阵-向量乘法，生成累加器/输出 Tile。 |
| 矩阵乘 | [`TGEMV_ACC`](isa/TGEMV_ACC_zh.md) | 带显式累加器输入/输出 Tile 的 GEMV。 |
| 矩阵乘 | [`TGEMV_BIAS`](isa/TGEMV_BIAS_zh.md) | 带偏置加法的 GEMV。 |
| 数据搬运 / 布局 | [`TEXTRACT`](isa/TEXTRACT_zh.md) | 从源 Tile 中提取子 Tile。 |
| 数据搬运 / 布局 | [`TEXTRACT_FP`](isa/TEXTRACT_FP_zh.md) | 带 fp/缩放 Tile 的提取（向量量化参数）。 |
| 数据搬运 / 布局 | [`TIMG2COL`](isa/TIMG2COL_zh.md) | 用于类卷积工作负载的图像到列变换。 |
| 数据搬运 / 布局 | [`TINSERT`](isa/TINSERT_zh.md) | 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。 |
| 数据搬运 / 布局 | [`TINSERT_FP`](isa/TINSERT_FP_zh.md) | 带 fp/缩放 Tile 的插入（向量量化参数）。 |
| 数据搬运 / 布局 | [`TFILLPAD`](isa/TFILLPAD_zh.md) | 复制 Tile 并在有效区域外使用编译时填充值进行填充。 |
| 数据搬运 / 布局 | [`TFILLPAD_INPLACE`](isa/TFILLPAD_INPLACE_zh.md) | 原地填充/填充变体。 |
| 数据搬运 / 布局 | [`TFILLPAD_EXPAND`](isa/TFILLPAD_EXPAND_zh.md) | 填充/填充时允许目标大于源。 |
| 数据搬运 / 布局 | [`TMOV`](isa/TMOV_zh.md) | 在 Tile 之间移动/复制，可选应用实现定义的转换模式。 |
| 数据搬运 / 布局 | [`TMOV_FP`](isa/TMOV_FP_zh.md) | 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 移动/转换到目标 Tile。 |
| 数据搬运 / 布局 | [`TRESHAPE`](isa/TRESHAPE_zh.md) | 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。 |
| 数据搬运 / 布局 | [`TTRANS`](isa/TTRANS_zh.md) | 使用实现定义的临时 Tile 进行转置。 |
| 数据搬运 / 布局 | [`TSUBVIEW`](isa/TSUBVIEW_zh.md) | 表达一个tile是另一个tile的subview |
| 数据搬运 / 布局 | [`TGET_SCALE_ADDR`](isa/TGET_SCALE_ADDR_zh.md) | 将输出tile的片上内存值绑定为扩展后的输入tile内存的值。 |
| 复杂指令 | [`TPRINT`](isa/TPRINT_zh.md) | 调试/打印 Tile 中的元素（实现定义）。 |
| 复杂指令 | [`TMRGSORT`](isa/TMRGSORT_zh.md) | 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。 |
| 复杂指令 | [`TSORT32`](isa/TSORT32_zh.md) | 对 `src` 的每个 32 元素块连同对应的 `idx` 条目一起排序，并输出排序后的 value-index 对。 |
| 复杂指令 | [`TGATHER`](isa/TGATHER_zh.md) | 使用索引 Tile 或编译时掩码模式来收集/选择元素。 |
| 复杂指令 | [`TCI`](isa/TCI_zh.md) | 生成连续整数序列到目标 Tile 中。 |
| 复杂指令 | [`TTRI`](isa/TTRI_zh.md) | 生成三角（下/上）掩码 Tile。 |
| 复杂指令 | [`TRANDOM`](isa/TRANDOM_zh.md) | 使用基于计数器的密码算法在目标 Tile 中生成随机数。 |
| 复杂指令 | [`TPARTADD`](isa/TPARTADD_zh.md) | 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTMUL`](isa/TPARTMUL_zh.md) | 部分逐元素乘法，对有效区域不一致的处理为实现定义。 |
| 复杂指令 | [`TPARTMAX`](isa/TPARTMAX_zh.md) | 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTMIN`](isa/TPARTMIN_zh.md) | 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TGATHERB`](isa/TGATHERB_zh.md) | 使用字节偏移量收集元素。 |
| 复杂指令 | [`TSCATTER`](isa/TSCATTER_zh.md) | 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。 |
| 复杂指令 | [`TQUANT`](isa/TQUANT_zh.md) | 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。 |
| 通信 | [`TPUT`](isa/comm/TPUT_zh.md) | 远程写：将本地数据传输到远端 NPU 内存（GM → UB → GM）。 |
| 通信 | [`TGET`](isa/comm/TGET_zh.md) | 远程读：将远端 NPU 数据读取到本地内存（GM → UB → GM）。 |
| 通信 | [`TPUT_ASYNC`](isa/comm/TPUT_ASYNC_zh.md) | 异步远程写（本地 GM → DMA 引擎 → 远端 GM）。 |
| 通信 | [`TGET_ASYNC`](isa/comm/TGET_ASYNC_zh.md) | 异步远程读（远端 GM → DMA 引擎 → 本地 GM）。 |
| 通信 | [`TNOTIFY`](isa/comm/TNOTIFY_zh.md) | 向远端 NPU 发送标志通知。 |
| 通信 | [`TWAIT`](isa/comm/TWAIT_zh.md) | 阻塞等待，直到信号满足比较条件。 |
| 通信 | [`TTEST`](isa/comm/TTEST_zh.md) | 非阻塞检测信号是否满足比较条件。 |
| 通信 | [`TGATHER`](isa/comm/TGATHER_zh.md) | 从所有 rank 收集数据并沿 DIM_3 拼接。 |
| 通信 | [`TSCATTER`](isa/comm/TSCATTER_zh.md) | 将数据沿 DIM_3 拆分并分发到所有 rank。 |
| 通信 | [`TREDUCE`](isa/comm/TREDUCE_zh.md) | 从所有 rank 收集数据并逐元素归约到本地。 |
| 通信 | [`TBROADCAST`](isa/comm/TBROADCAST_zh.md) | 将当前 NPU 的数据广播到所有 rank。 |
