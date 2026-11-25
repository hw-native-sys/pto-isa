<p align="center">
  <img src="../../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 参考

本目录是 PTO Tile Lib ISA 的指令参考（每条指令一页）。

- 权威来源：`include/pto/common/pto_instr.hpp`
- 通用约定（操作数、事件、修饰符）：`docs/zh/isa/conventions.md`

## 删除接口与迁移说明

当前 ISA 参考不再公开以下历史指令接口：

- `TADDC`
- `TAddDeqRelu`
- `TADDReluConv`
- `TADDSC`
- `TFUSEDMULADDRELU`
- `TGET_SCALE_ADDR`
- `TPairReduceSum`
- `TSUBC`
- `TSUBRELU`
- `TSUBRELUCONV`
- `TSUBSC`
- `TSUBVIEW`
- `TSYNC`

迁移建议：

- 生效版本：本次清理自 PTO ISA v9.1.0 起生效；这些历史 wrapper 的兼容窗口关闭，不再保留公开 wrapper。
- 将 `TSYNC(events...)` 替换为普通 event 顺序表达：把 event 对象传给消费端 intrinsic，或在确需显式等待时调用 `WaitAllEvents(events...)`。
- `TSUBVIEW` 不提供公开 ISA 替代接口。仓内实现可使用 `pto::detail::PtoSubTileView` 作为内部 helper；外部代码应通过受支持的 Tile 构造和公开数据搬运 API 表达数据视图。
- 将三元/标量融合算术形式替换为对应基础算术序列，例如 `TADD`、`TSUB`、`TADDS`、`TSUBS`、`TMUL`、`TMADD` 和 `TRELU`。
- 将历史融合乘加接口重命名：`TFUSEDMULADD` 改为 `TMADD`，`TMULADDDST` 改为 `TMULA`。
- 将融合 add/ReLU/convert 或 add/dequant/ReLU 形式拆分为显式算术、转换/反量化和 `TRELU` 步骤。
- 将 `TPairReduceSum` 替换为与目标 layout 匹配的现有行/列归约原语。
- 不再调用 `TGET_SCALE_ADDR`；AUTO 模式 MX 测试可在调用 MX matmul 原语前，在测试代码中根据数据 Tile 绑定 scale Tile 地址。

## 同步
- [SYNCALL](SYNCALL.md) - 跨核同步屏障（硬件 FFTS 或软件 GM 轮询）。

## 手动 / 资源绑定
- [TASSIGN](TASSIGN.md) - 将 Tile 对象绑定到实现定义的片上地址（手动放置）。
- [SETFMATRIX](SETFMATRIX.md) - 为类 IMG2COL 操作设置 FMATRIX 寄存器。
- [SET_IMG2COL_RPT](SET_IMG2COL_RPT.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。
- [SET_IMG2COL_PADDING](SET_IMG2COL_PADDING.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据。
- [SET_QUANT_SCALAR](SET_QUANT_SCALAR.md) - 设置标量量化参数，用于后续 TPUSH 操作。
- [SET_QUANT_VECTOR](SET_QUANT_VECTOR.md) - 从 Scaling Tile 设置向量量化参数，用于后续 TPUSH 操作。

## 逐元素（Tile-Tile）
- [TADD](TADD.md) - 两个 Tile 的逐元素加法。
- [TABS](TABS.md) - Tile 的逐元素绝对值。
- [TAND](TAND.md) - 两个 Tile 的逐元素按位与。
- [TOR](TOR.md) - 两个 Tile 的逐元素按位或。
- [TSUB](TSUB.md) - 两个 Tile 的逐元素减法。
- [TMUL](TMUL.md) - 两个 Tile 的逐元素乘法。
- [TMADD](TMADD.md) - 三元逐元素运算：`src0 * dst + src1`。
- [TMIN](TMIN.md) - 两个 Tile 的逐元素最小值。
- [TMAX](TMAX.md) - 两个 Tile 的逐元素最大值。
- [TCMP](TCMP.md) - 比较两个 Tile 并写入一个打包的谓词掩码。
- [TDIV](TDIV.md) - 两个 Tile 的逐元素除法。
- [TSHL](TSHL.md) - 两个 Tile 的逐元素左移。
- [TSHR](TSHR.md) - 两个 Tile 的逐元素右移。
- [TXOR](TXOR.md) - 两个 Tile 的逐元素按位异或。
- [TLOG](TLOG.md) - Tile 的逐元素自然对数。
- [TRECIP](TRECIP.md) - Tile 的逐元素倒数。
- [TPRELU](TPRELU.md) - 带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。
- [TCVT](TCVT.md) - 带指定舍入模式的逐元素类型转换。
- [TSEL](TSEL.md) - 使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。
- [TPARTSEL](TPARTSEL.md) - 允许不同数据 Tile 类型的掩码选择。
- [TRSQRT](TRSQRT.md) - 逐元素倒数平方根。
- [TSQRT](TSQRT.md) - 逐元素平方根。
- [TEXP](TEXP.md) - 逐元素指数运算。
- [TPOW](TPOW.md) - 逐元素幂运算。
- [TNOT](TNOT.md) - Tile 的逐元素按位取反。
- [TRELU](TRELU.md) - Tile 的逐元素 ReLU。
- [TNEG](TNEG.md) - Tile 的逐元素取负。
- [TREM](TREM.md) - 两个 Tile 的逐元素余数。除数非零时，非零余数的符号与除数相同。
- [TFMOD](TFMOD.md) - 两个 Tile 的逐元素余数，余数符号与被除数相同。
- [TMULA](TMULA.md) - 三元逐元素运算：`src0 * src1 + dst`。

## Tile-标量 / Tile-立即数
- [TEXPANDS](TEXPANDS.md) - 将标量广播到目标 Tile 中。
- [TCMPS](TCMPS.md) - 将 Tile 与标量比较并写入逐元素比较结果。
- [TSELS](TSELS.md) - 使用掩码 Tile 在源 Tile 和标量之间进行选择（源 Tile 逐元素选择）。
- [TMINS](TMINS.md) - Tile 与标量的逐元素最小值。
- [TADDS](TADDS.md) - Tile 与标量的逐元素加法。
- [TSUBS](TSUBS.md) - 从 Tile 中逐元素减去一个标量。
- [TAXPY](TAXPY.md) - 原位缩放累加（AXPY）：dst = scalar * src0 + dst。
- [TDIVS](TDIVS.md) - 与标量的逐元素除法（Tile/标量 或 标量/Tile）。
- [TMULS](TMULS.md) - Tile 与标量的逐元素乘法。
- [TFMODS](TFMODS.md) - 与标量的逐元素余数：`fmod(src, scalar)`。
- [TREMS](TREMS.md) - 与标量的逐元素余数：`remainder(src, scalar)`。
- [TMAXS](TMAXS.md) - Tile 与标量的逐元素最大值：`max(src, scalar)`。
- [TANDS](TANDS.md) - Tile 与标量的逐元素按位与。
- [TORS](TORS.md) - Tile 与标量的逐元素按位或。
- [TSHLS](TSHLS.md) - Tile 按标量逐元素左移。
- [TSHRS](TSHRS.md) - Tile 按标量逐元素右移。
- [TXORS](TXORS.md) - Tile 与标量的逐元素按位异或。
- [TLRELU](TLRELU.md) - 带标量斜率的 Leaky ReLU。
- [TPOWS](TPOWS.md) - Tile 逐元素与标量幂运算。

## 轴归约 / 扩展
- [TROWSUM](TROWSUM.md) - 通过对列求和来归约每一行。
- [TROWPROD](TROWPROD.md) - 通过跨列乘积来归约每一行。
- [TCOLSUM](TCOLSUM.md) - 通过对行求和来归约每一列。
- [TCOLPROD](TCOLPROD.md) - 通过跨行乘积来归约每一列。
- [TCOLMAX](TCOLMAX.md) - 通过取行间最大值来归约每一列。
- [TROWMAX](TROWMAX.md) - 通过取列间最大值来归约每一行。
- [TROWMIN](TROWMIN.md) - 通过取列间最小值来归约每一行。
- [TROWARGMAX](TROWARGMAX.md) - 获取每行最大值对应列索引。
- [TROWARGMIN](TROWARGMIN.md) - 获取每行最小值对应列索引。
- [TCOLARGMAX](TCOLARGMAX.md) - 获取每列最大值对应行索引/获取每列最大值对应行值和索引。
- [TCOLARGMIN](TCOLARGMIN.md) - 获取每列最小值对应行索引/获取每列最小值对应行值和索引。
- [TROWEXPAND](TROWEXPAND.md) - 将每个源行的第一个元素广播到目标行中。
- [TROWEXPANDDIV](TROWEXPANDDIV.md) - 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。
- [TROWEXPANDMUL](TROWEXPANDMUL.md) - 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。
- [TROWEXPANDSUB](TROWEXPANDSUB.md) - 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。
- [TROWEXPANDADD](TROWEXPANDADD.md) - 行广播加法：加上一个每行标量向量。
- [TROWEXPANDMAX](TROWEXPANDMAX.md) - 行广播最大值：与每行标量向量取最大值。
- [TROWEXPANDMIN](TROWEXPANDMIN.md) - 行广播最小值：与每行标量向量取最小值。
- [TROWEXPANDEXPDIF](TROWEXPANDEXPDIF.md) - 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。
- [TCOLMIN](TCOLMIN.md) - 通过取行间最小值来归约每一列。
- [TCOLEXPAND](TCOLEXPAND.md) - 将每个源列的第一个元素广播到目标列中。
- [TCOLEXPANDDIV](TCOLEXPANDDIV.md) - 列广播除法：将每一列除以一个每列标量向量。
- [TCOLEXPANDMUL](TCOLEXPANDMUL.md) - 列广播乘法：将每一列乘以一个每列标量向量。
- [TCOLEXPANDADD](TCOLEXPANDADD.md) - 列广播加法：对每一列加上每列标量向量。
- [TCOLEXPANDMAX](TCOLEXPANDMAX.md) - 列广播最大值：与每列标量向量取最大值。
- [TCOLEXPANDMIN](TCOLEXPANDMIN.md) - 列广播最小值：与每列标量向量取最小值。
- [TCOLEXPANDSUB](TCOLEXPANDSUB.md) - 列广播减法：从每一列中减去一个每列标量向量。
- [TCOLEXPANDEXPDIF](TCOLEXPANDEXPDIF.md) - 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。

## 内存（GM <-> Tile）
- [TLOAD](TLOAD.md) - 从 GlobalTensor (GM) 加载数据到 Tile。
- [TPREFETCH](TPREFETCH.md) - 将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。
- [TPREFETCH_ASYNC](TPREFETCH_ASYNC.md) - 通过 SDMA CMO 将 GlobalTensor 区域从 GM 异步预取到 L2 Cache。
- [TSTORE](TSTORE.md) - 将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。
- [TSTORE_FP](TSTORE_FP.md) - 历史 fp 量化存储形式的源码兼容 C++ 别名。
- [MGATHER](MGATHER.md) - 使用逐元素索引从全局内存收集加载元素到 Tile 中。
- [MSCATTER](MSCATTER.md) - 使用逐元素索引将 Tile 中的元素散播存储到全局内存。

## 矩阵乘
- [TGEMV_MX](TGEMV_MX.md) - 带缩放 Tile 的 GEMV 变体，支持混合精度/量化矩阵向量计算。
- [TMATMUL_MX](TMATMUL_MX.md) - 带额外缩放 Tile 的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。
- [TMATMUL_MX_HIF4](TMATMUL_MX_HIF4.md) - 用于混合精度 TMATMUL_MX 流程的 HiFloat4 Cube 矩阵乘变体。
- [TMATMUL](TMATMUL.md) - 矩阵乘法 (GEMM)，生成累加器/输出 Tile。
- [TMATMUL_ACC](TMATMUL_ACC.md) - 带累加器输入的矩阵乘法（融合累加）。
- [TMATMUL_BIAS](TMATMUL_BIAS.md) - 带偏置加法的矩阵乘法。
- [TGEMV](TGEMV.md) - 通用矩阵-向量乘法，生成累加器/输出 Tile。
- [TGEMV_ACC](TGEMV_ACC.md) - 带显式累加器输入/输出 Tile 的 GEMV。
- [TGEMV_BIAS](TGEMV_BIAS.md) - 带偏置加法的 GEMV。

## 数据搬运 / 布局
- [TEXTRACT](TEXTRACT.md) - 从源 Tile 中提取子 Tile。
- [TEXTRACT_FP](TEXTRACT_FP.md) - 历史 fp 量化提取形式的源码兼容 C++ 别名。
- [TIMG2COL](TIMG2COL.md) - 用于类卷积工作负载的图像到列变换。
- [TINSERT](TINSERT.md) - 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。
- [TINSERT_FP](TINSERT_FP.md) - 历史 fp 量化插入形式的源码兼容 C++ 别名。
- [TFILLPAD](TFILLPAD.md) - 复制 Tile 并在有效区域外使用编译时填充值进行填充。
- [TFILLPAD_INPLACE](TFILLPAD_INPLACE.md) - 原地填充/填充变体。
- [TFILLPAD_EXPAND](TFILLPAD_EXPAND.md) - 填充/填充时允许目标大于源。
- [TMOV](TMOV.md) - 在 Tile 之间移动/复制，可选应用实现定义的转换模式。
- [TMOV_FP](TMOV_FP.md) - 历史 fp 量化移动形式的源码兼容 C++ 别名。
- [TRESHAPE](TRESHAPE.md) - 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。
- [TTRANS](TTRANS.md) - 使用实现定义的临时 Tile 进行转置。
- [TCONCAT](TCONCAT.md) - 将两个 Tile 沿列维度水平拼接。
- [TINTERLEAVE](TINTERLEAVE.md) - 将两个源 Tile 交织为交替的偶/奇元素流，拆分为两个目标半部分。
- [TDEINTERLEAVE](TDEINTERLEAVE.md) - 将源 Tile 反交织为偶数位置和奇数位置的元素流（TINTERLEAVE 的逆操作）。

## 复杂指令
- [TPRINT](TPRINT.md) - 调试/打印 Tile 中的元素（实现定义）。
- [TMRGSORT](TMRGSORT.md) - 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。
- [TSORT32](TSORT32.md) - 对 `src` 的每个 32 元素块连同对应的 `idx` 条目一起排序，并输出排序后的 value-index 对。
- [TGATHER](TGATHER.md) - 使用索引 Tile 或编译时掩码模式来收集/选择元素。
- [TCI](TCI.md) - 生成连续整数序列到目标 Tile 中。
- [TTRI](TTRI.md) - 生成三角（下/上）掩码 Tile。
- [TRANDOM](TRANDOM.md) - 使用基于计数器的密码算法在目标 Tile 中生成随机数。
- [TPARTADD](TPARTADD.md) - 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMUL](TPARTMUL.md) - 部分逐元素乘法，对有效区域不一致的处理为实现定义。
- [TPARTMAX](TPARTMAX.md) - 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMIN](TPARTMIN.md) - 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTARGMAX](TPARTARGMAX.md) - 部分逐元素最大值选择并返回对应索引（argmax），对不匹配的有效区域具有实现定义的处理方式。
- [TPARTARGMIN](TPARTARGMIN.md) - 部分逐元素最小值选择并返回对应索引（argmin），对不匹配的有效区域具有实现定义的处理方式。
- [TGATHERB](TGATHERB.md) - 使用字节偏移量收集元素。
- [TSCATTER](TSCATTER.md) - 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。
- [TQUANT](TQUANT.md) - 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。
- [TQUANT_DN](TQUANT_DN.md) - Axis-0 分组量化与 DN 到 ZZ 指数布局转换。
- [TQUANT_HIF4](TQUANT_HIF4.md) - BF16 到 HiFloat4 的量化算法与 CCE 映射。
- [TDEQUANT](TDEQUANT.md) - 对量化 Tile 做仿射反量化（S8/S16 -> FP32）：dst = (src - offset) * scale。
- [THISTOGRAM](THISTOGRAM.md) - 对源元素的某个字节统计直方图（256 桶），可按高位字节级联过滤；基数排序的桶计数原语。

## 核间通信
- [TALLOC](TALLOC.md) - 将 TPipe FIFO 槽位分配为一个 GlobalTensor 视图。
- [TPUSH](TPUSH.md) - 将生产者 tile 推入 TPipe FIFO，用于 Cube-Vector 通信。
- [TPOP](TPOP.md) - 从 TPipe FIFO 弹出消费者 tile/globalTensor，用于 Cube-Vector 通信。
- [TFREE](TFREE.md) - 释放 TPipe 的 FIFO 空间；部分目标上对 TileData TPOP 流程为空操作。
