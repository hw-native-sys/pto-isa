<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 参考

本目录是 PTO Tile Lib ISA 的指令参考（每条指令一页）。

- 权威来源（C++ 内建函数）：`include/pto/common/pto_instr.hpp`
- [通用约定（操作数、事件、修饰符）](conventions_zh.md)

## 同步
- [TSYNC](TSYNC_zh.md) - 同步 PTO 执行（等待事件或插入每操作流水线屏障）。

## 手动 / 资源绑定
- [TASSIGN](TASSIGN_zh.md) - 将 Tile 对象绑定到实现定义的片上地址（手动放置）。
- [TSETFMATRIX](TSETFMATRIX_zh.md) - 为类 IMG2COL 操作设置 FMATRIX 寄存器。
- [TSET_IMG2COL_RPT](TSET_IMG2COL_RPT_zh.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。
- [TSET_IMG2COL_PADDING](TSET_IMG2COL_PADDING_zh.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据。

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

## 轴归约 / 扩展
- [TROWSUM](TROWSUM_zh.md) - 通过对列求和来归约每一行。
- [TROWPROD](TROWPROD_zh.md) - 通过跨列乘积来归约每一行。
- [TCOLSUM](TCOLSUM_zh.md) - 通过对行求和来归约每一列。
- [TCOLPROD](TCOLPROD_zh.md) - 通过跨行乘积来归约每一列。
- [TCOLMAX](TCOLMAX_zh.md) - 通过取行间最大值来归约每一列。
- [TROWMAX](TROWMAX_zh.md) - 通过取列间最大值来归约每一行。
- [TROWMIN](TROWMIN_zh.md) - 通过取列间最小值来归约每一行。
- [TROWARGMAX](TROWARGMAX_zh.md) - 获取每行最大值对应列索引。
- [TROWARGMIN](TROWARGMIN_zh.md) - 获取每行最小值对应列索引。
- [TROWEXPAND](TROWEXPAND_zh.md) - 将每个源行的第一个元素广播到目标行中。
- [TROWEXPANDDIV](TROWEXPANDDIV_zh.md) - 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。
- [TROWEXPANDMUL](TROWEXPANDMUL_zh.md) - 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。
- [TROWEXPANDSUB](TROWEXPANDSUB_zh.md) - 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。
- [TROWEXPANDADD](TROWEXPANDADD_zh.md) - 行广播加法：加上一个每行标量向量。
- [TROWEXPANDMAX](TROWEXPANDMAX_zh.md) - 行广播最大值：与每行标量向量取最大值。
- [TROWEXPANDMIN](TROWEXPANDMIN_zh.md) - 行广播最小值：与每行标量向量取最小值。
- [TROWEXPANDEXPDIF](TROWEXPANDEXPDIF_zh.md) - 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。
- [TCOLMIN](TCOLMIN_zh.md) - 通过取行间最小值来归约每一列。
- [TCOLEXPAND](TCOLEXPAND_zh.md) - 将每个源列的第一个元素广播到目标列中。
- [TCOLEXPANDDIV](TCOLEXPANDDIV_zh.md) - 列广播除法：将每一列除以一个每列标量向量。
- [TCOLEXPANDMUL](TCOLEXPANDMUL_zh.md) - 列广播乘法：将每一列乘以一个每列标量向量。
- [TCOLEXPANDADD](TCOLEXPANDADD_zh.md) - 列广播加法：对每一列加上每列标量向量。
- [TCOLEXPANDMAX](TCOLEXPANDMAX_zh.md) - 列广播最大值：与每列标量向量取最大值。
- [TCOLEXPANDMIN](TCOLEXPANDMIN_zh.md) - 列广播最小值：与每列标量向量取最小值。
- [TCOLEXPANDSUB](TCOLEXPANDSUB_zh.md) - 列广播减法：从每一列中减去一个每列标量向量。
- [TCOLEXPANDEXPDIF](TCOLEXPANDEXPDIF_zh.md) - 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。

## 内存（GM <-> Tile）
- [TLOAD](TLOAD_zh.md) - 从 GlobalTensor (GM) 加载数据到 Tile。
- [TPREFETCH](TPREFETCH_zh.md) - 将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。
- [TSTORE](TSTORE_zh.md) - 将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。
- [TSTORE_FP](TSTORE_FP_zh.md) - 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 存储到全局内存。
- [MGATHER](MGATHER_zh.md) - 使用逐元素索引从全局内存收集加载元素到 Tile 中。
- [MSCATTER](MSCATTER_zh.md) - 使用逐元素索引将 Tile 中的元素散播存储到全局内存。

## 矩阵乘
- [TGEMV_MX](TGEMV_MX_zh.md) - 带缩放 Tile 的 GEMV 变体，支持混合精度/量化矩阵向量计算。
- [TMATMUL_MX](TMATMUL_MX_zh.md) - 带额外缩放 Tile 的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。
- [TMATMUL](TMATMUL_zh.md) - 矩阵乘法 (GEMM)，生成累加器/输出 Tile。
- [TMATMUL_ACC](TMATMUL_ACC_zh.md) - 带累加器输入的矩阵乘法（融合累加）。
- [TMATMUL_BIAS](TMATMUL_BIAS_zh.md) - 带偏置加法的矩阵乘法。
- [TGEMV](TGEMV_zh.md) - 通用矩阵-向量乘法，生成累加器/输出 Tile。
- [TGEMV_ACC](TGEMV_ACC_zh.md) - 带显式累加器输入/输出 Tile 的 GEMV。
- [TGEMV_BIAS](TGEMV_BIAS_zh.md) - 带偏置加法的 GEMV。

## 数据搬运 / 布局
- [TEXTRACT](TEXTRACT_zh.md) - 从源 Tile 中提取子 Tile。
- [TEXTRACT_FP](TEXTRACT_FP_zh.md) - 带 fp/缩放 Tile 的提取（向量量化参数）。
- [TIMG2COL](TIMG2COL_zh.md) - 用于类卷积工作负载的图像到列变换。
- [TINSERT](TINSERT_zh.md) - 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。
- [TINSERT_FP](TINSERT_FP_zh.md) - 带 fp/缩放 Tile 的插入（向量量化参数）。
- [TFILLPAD](TFILLPAD_zh.md) - 复制 Tile 并在有效区域外使用编译时填充值进行填充。
- [TFILLPAD_INPLACE](TFILLPAD_INPLACE_zh.md) - 原地填充/填充变体。
- [TFILLPAD_EXPAND](TFILLPAD_EXPAND_zh.md) - 填充/填充时允许目标大于源。
- [TMOV](TMOV_zh.md) - 在 Tile 之间移动/复制，可选应用实现定义的转换模式。
- [TMOV_FP](TMOV_FP_zh.md) - 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 移动/转换到目标 Tile。
- [TRESHAPE](TRESHAPE_zh.md) - 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。
- [TTRANS](TTRANS_zh.md) - 使用实现定义的临时 Tile 进行转置。
- [TSUBVIEW](TSUBVIEW_zh.md) - 表达一个tile是另一个tile的subview。
- [TGET_SCALE_ADDR](TGET_SCALE_ADDR_zh.md) - 将输出tile的片上内存值绑定为扩展后的输入tile内存的值。

## 复杂指令
- [TPRINT](TPRINT_zh.md) - 调试/打印 Tile 中的元素（实现定义）。
- [TMRGSORT](TMRGSORT_zh.md) - 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。
- [TSORT32](TSORT32_zh.md) - 对 `src` 的每个 32 元素块，与 `idx` 中对应的索引一起进行排序，并将排序后的值-索引对写入 `dst`。
- [TGATHER](TGATHER_zh.md) - 使用索引 Tile 或编译时掩码模式来收集/选择元素。
- [TCI](TCI_zh.md) - 生成连续整数序列到目标 Tile 中。
- [TTRI](TTRI_zh.md) - 生成三角（下/上）掩码 Tile。
- [TPARTADD](TPARTADD_zh.md) - 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMUL](TPARTMUL_zh.md) - 部分逐元素乘法，对有效区域不一致的处理为实现定义。
- [TPARTMAX](TPARTMAX_zh.md) - 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMIN](TPARTMIN_zh.md) - 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。
- [TGATHERB](TGATHERB_zh.md) - 使用字节偏移量收集元素。
- [TSCATTER](TSCATTER_zh.md) - 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。
- [TQUANT](TQUANT_zh.md) - 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。

## 通信

完整的通信 ISA 指令参考（点对点、异步、同步原语及集合通信）见 [comm/README_zh.md](comm/README_zh.md)。
