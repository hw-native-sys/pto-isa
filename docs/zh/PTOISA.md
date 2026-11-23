# PTO ISA 概述

本文档为根据 `docs/isa/manifest.yaml` 和通信 ISA 参考自动生成的 ISA 索引。

## 文档目录

| 领域 | 页面 | 描述 |
|---|---|---|
| 概述 | [`docs/zh/README.md`](README.md) | PTO ISA 指南入口与导航。 |
| 概述 | [`docs/zh/PTOISA.md`](PTOISA.md) | 本页（概述 + 全量指令索引）。 |
| ISA 参考 | [`docs/zh/isa/README.md`](isa/README.md) | 每条指令参考目录。 |
| ISA 参考 | [`docs/zh/isa/conventions.md`](isa/conventions.md) | 通用符号、操作数、事件与修饰符。 |
| 通信 ISA | [`docs/zh/isa/comm/README.md`](isa/comm/README.md) | 每条通信 ISA 指令参考。 |
| 权威源 | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header.md) | C++ intrinsic API（权威来源）。 |
| PTO auto 模式 | [`docs/zh/auto_mode/README.md`](README.md) | PTO auto模式文档入口 |

## 指令索引（全部 PTO 指令）

| 分类 | 指令 | 描述 |
|---|---|---|
| 同步 | [`SYNCALL`](isa/SYNCALL.md) | 跨核同步屏障（硬件 FFTS 或软件 GM 轮询）。 |
| 手动 / 资源绑定 | [`TASSIGN`](isa/TASSIGN.md) | 将 Tile 对象绑定到实现定义的片上地址（手动放置）。 |
| 手动 / 资源绑定 | [`SETFMATRIX`](isa/SETFMATRIX.md) | 为类 IMG2COL 操作设置 FMATRIX 寄存器。 |
| 手动 / 资源绑定 | [`SET_IMG2COL_RPT`](isa/SET_IMG2COL_RPT.md) | 从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。 |
| 手动 / 资源绑定 | [`SET_IMG2COL_PADDING`](isa/SET_IMG2COL_PADDING.md) | 从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据。 |
| 手动 / 资源绑定 | [`SET_QUANT_SCALAR`](isa/SET_QUANT_SCALAR.md) | 设置标量量化参数，用于后续 TPUSH 操作。 |
| 手动 / 资源绑定 | [`SET_QUANT_VECTOR`](isa/SET_QUANT_VECTOR.md) | 从 Scaling Tile 设置向量量化参数，用于后续 TPUSH 操作。 |
| 逐元素（Tile-Tile） | [`TADD`](isa/TADD.md) | 两个 Tile 的逐元素加法。 |
| 逐元素（Tile-Tile） | [`TABS`](isa/TABS.md) | Tile 的逐元素绝对值。 |
| 逐元素（Tile-Tile） | [`TAND`](isa/TAND.md) | 两个 Tile 的逐元素按位与。 |
| 逐元素（Tile-Tile） | [`TOR`](isa/TOR.md) | 两个 Tile 的逐元素按位或。 |
| 逐元素（Tile-Tile） | [`TSUB`](isa/TSUB.md) | 两个 Tile 的逐元素减法。 |
| 逐元素（Tile-Tile） | [`TMUL`](isa/TMUL.md) | 两个 Tile 的逐元素乘法。 |
| 逐元素（Tile-Tile） | [`TMADD`](isa/TMADD.md) | 三元逐元素运算：`src0 * dst + src1`。 |
| 逐元素（Tile-Tile） | [`TMIN`](isa/TMIN.md) | 两个 Tile 的逐元素最小值。 |
| 逐元素（Tile-Tile） | [`TMAX`](isa/TMAX.md) | 两个 Tile 的逐元素最大值。 |
| 逐元素（Tile-Tile） | [`TCMP`](isa/TCMP.md) | 比较两个 Tile 并写入一个打包的谓词掩码。 |
| 逐元素（Tile-Tile） | [`TDIV`](isa/TDIV.md) | 两个 Tile 的逐元素除法。 |
| 逐元素（Tile-Tile） | [`TSHL`](isa/TSHL.md) | 两个 Tile 的逐元素左移。 |
| 逐元素（Tile-Tile） | [`TSHR`](isa/TSHR.md) | 两个 Tile 的逐元素右移。 |
| 逐元素（Tile-Tile） | [`TXOR`](isa/TXOR.md) | 两个 Tile 的逐元素按位异或。 |
| 逐元素（Tile-Tile） | [`TLOG`](isa/TLOG.md) | Tile 的逐元素自然对数。 |
| 逐元素（Tile-Tile） | [`TRECIP`](isa/TRECIP.md) | Tile 的逐元素倒数。 |
| 逐元素（Tile-Tile） | [`TPRELU`](isa/TPRELU.md) | 带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。 |
| 逐元素（Tile-Tile） | [`TCVT`](isa/TCVT.md) | 带指定舍入模式的逐元素类型转换。 |
| 逐元素（Tile-Tile） | [`TSEL`](isa/TSEL.md) | 使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。 |
| 逐元素（Tile-Tile） | [`TPARTSEL`](isa/TPARTSEL.md) | 允许不同数据 Tile 类型的掩码选择。 |
| 逐元素（Tile-Tile） | [`TRSQRT`](isa/TRSQRT.md) | 逐元素倒数平方根。 |
| 逐元素（Tile-Tile） | [`TSQRT`](isa/TSQRT.md) | 逐元素平方根。 |
| 逐元素（Tile-Tile） | [`TEXP`](isa/TEXP.md) | 逐元素指数运算。 |
| 逐元素（Tile-Tile） | [`TPOW`](isa/TPOW.md) | 逐元素幂运算。 |
| 逐元素（Tile-Tile） | [`TNOT`](isa/TNOT.md) | Tile 的逐元素按位取反。 |
| 逐元素（Tile-Tile） | [`TRELU`](isa/TRELU.md) | Tile 的逐元素 ReLU。 |
| 逐元素（Tile-Tile） | [`TNEG`](isa/TNEG.md) | Tile 的逐元素取负。 |
| 逐元素（Tile-Tile） | [`TREM`](isa/TREM.md) | 两个 Tile 的逐元素余数。除数非零时，非零余数的符号与除数相同。 |
| 逐元素（Tile-Tile） | [`TFMOD`](isa/TFMOD.md) | 两个 Tile 的逐元素余数，余数符号与被除数相同。 |
| 逐元素（Tile-Tile） | [`TMULA`](isa/TMULA.md) | 三元逐元素运算：`src0 * src1 + dst`。 |
| Tile-标量 / Tile-立即数 | [`TEXPANDS`](isa/TEXPANDS.md) | 将标量广播到目标 Tile 中。 |
| Tile-标量 / Tile-立即数 | [`TCMPS`](isa/TCMPS.md) | 将 Tile 与标量比较并写入逐元素比较结果。 |
| Tile-标量 / Tile-立即数 | [`TSELS`](isa/TSELS.md) | 使用掩码 Tile 在源 Tile 和标量之间进行选择（源 Tile 逐元素选择）。 |
| Tile-标量 / Tile-立即数 | [`TMINS`](isa/TMINS.md) | Tile 与标量的逐元素最小值。 |
| Tile-标量 / Tile-立即数 | [`TADDS`](isa/TADDS.md) | Tile 与标量的逐元素加法。 |
| Tile-标量 / Tile-立即数 | [`TSUBS`](isa/TSUBS.md) | 从 Tile 中逐元素减去一个标量。 |
| Tile-标量 / Tile-立即数 | [`TAXPY`](isa/TAXPY.md) | 原位缩放累加（AXPY）：dst = scalar * src0 + dst。 |
| Tile-标量 / Tile-立即数 | [`TDIVS`](isa/TDIVS.md) | 与标量的逐元素除法（Tile/标量 或 标量/Tile）。 |
| Tile-标量 / Tile-立即数 | [`TMULS`](isa/TMULS.md) | Tile 与标量的逐元素乘法。 |
| Tile-标量 / Tile-立即数 | [`TFMODS`](isa/TFMODS.md) | 与标量的逐元素余数：`fmod(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TREMS`](isa/TREMS.md) | 与标量的逐元素余数：`remainder(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TMAXS`](isa/TMAXS.md) | Tile 与标量的逐元素最大值：`max(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | [`TANDS`](isa/TANDS.md) | Tile 与标量的逐元素按位与。 |
| Tile-标量 / Tile-立即数 | [`TORS`](isa/TORS.md) | Tile 与标量的逐元素按位或。 |
| Tile-标量 / Tile-立即数 | [`TSHLS`](isa/TSHLS.md) | Tile 按标量逐元素左移。 |
| Tile-标量 / Tile-立即数 | [`TSHRS`](isa/TSHRS.md) | Tile 按标量逐元素右移。 |
| Tile-标量 / Tile-立即数 | [`TXORS`](isa/TXORS.md) | Tile 与标量的逐元素按位异或。 |
| Tile-标量 / Tile-立即数 | [`TLRELU`](isa/TLRELU.md) | 带标量斜率的 Leaky ReLU。 |
| Tile-标量 / Tile-立即数 | [`TPOWS`](isa/TPOWS.md) | Tile 逐元素与标量幂运算。 |
| 轴归约 / 扩展 | [`TROWSUM`](isa/TROWSUM.md) | 通过对列求和来归约每一行。 |
| 轴归约 / 扩展 | [`TROWPROD`](isa/TROWPROD.md) | 通过跨列乘积来归约每一行。 |
| 轴归约 / 扩展 | [`TCOLSUM`](isa/TCOLSUM.md) | 通过对行求和来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLPROD`](isa/TCOLPROD.md) | 通过跨行乘积来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLMAX`](isa/TCOLMAX.md) | 通过取行间最大值来归约每一列。 |
| 轴归约 / 扩展 | [`TROWMAX`](isa/TROWMAX.md) | 通过取列间最大值来归约每一行。 |
| 轴归约 / 扩展 | [`TROWMIN`](isa/TROWMIN.md) | 通过取列间最小值来归约每一行。 |
| 轴归约 / 扩展 | [`TROWARGMAX`](isa/TROWARGMAX.md) | 获取每行最大值对应列索引。 |
| 轴归约 / 扩展 | [`TROWARGMIN`](isa/TROWARGMIN.md) | 获取每行最小值对应列索引。 |
| 轴归约 / 扩展 | [`TCOLARGMAX`](isa/TCOLARGMAX.md) | 获取每列最大值对应行索引/获取每列最大值对应行值和索引。 |
| 轴归约 / 扩展 | [`TCOLARGMIN`](isa/TCOLARGMIN.md) | 获取每列最小值对应行索引/获取每列最小值对应行值和索引。 |
| 轴归约 / 扩展 | [`TROWEXPAND`](isa/TROWEXPAND.md) | 将每个源行的第一个元素广播到目标行中。 |
| 轴归约 / 扩展 | [`TROWEXPANDDIV`](isa/TROWEXPANDDIV.md) | 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDMUL`](isa/TROWEXPANDMUL.md) | 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDSUB`](isa/TROWEXPANDSUB.md) | 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | [`TROWEXPANDADD`](isa/TROWEXPANDADD.md) | 行广播加法：加上一个每行标量向量。 |
| 轴归约 / 扩展 | [`TROWEXPANDMAX`](isa/TROWEXPANDMAX.md) | 行广播最大值：与每行标量向量取最大值。 |
| 轴归约 / 扩展 | [`TROWEXPANDMIN`](isa/TROWEXPANDMIN.md) | 行广播最小值：与每行标量向量取最小值。 |
| 轴归约 / 扩展 | [`TROWEXPANDEXPDIF`](isa/TROWEXPANDEXPDIF.md) | 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。 |
| 轴归约 / 扩展 | [`TCOLMIN`](isa/TCOLMIN.md) | 通过取行间最小值来归约每一列。 |
| 轴归约 / 扩展 | [`TCOLEXPAND`](isa/TCOLEXPAND.md) | 将每个源列的第一个元素广播到目标列中。 |
| 轴归约 / 扩展 | [`TCOLEXPANDDIV`](isa/TCOLEXPANDDIV.md) | 列广播除法：将每一列除以一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMUL`](isa/TCOLEXPANDMUL.md) | 列广播乘法：将每一列乘以一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDADD`](isa/TCOLEXPANDADD.md) | 列广播加法：对每一列加上每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMAX`](isa/TCOLEXPANDMAX.md) | 列广播最大值：与每列标量向量取最大值。 |
| 轴归约 / 扩展 | [`TCOLEXPANDMIN`](isa/TCOLEXPANDMIN.md) | 列广播最小值：与每列标量向量取最小值。 |
| 轴归约 / 扩展 | [`TCOLEXPANDSUB`](isa/TCOLEXPANDSUB.md) | 列广播减法：从每一列中减去一个每列标量向量。 |
| 轴归约 / 扩展 | [`TCOLEXPANDEXPDIF`](isa/TCOLEXPANDEXPDIF.md) | 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。 |
| 内存（GM <-> Tile） | [`TLOAD`](isa/TLOAD.md) | 从 GlobalTensor (GM) 加载数据到 Tile。 |
| 内存（GM <-> Tile） | [`TPREFETCH`](isa/TPREFETCH.md) | 将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。 |
| 内存（GM <-> Tile） | [`TPREFETCH_ASYNC`](isa/TPREFETCH_ASYNC.md) | 通过 SDMA CMO 将 GlobalTensor 区域从 GM 异步预取到 L2 Cache。 |
| 内存（GM <-> Tile） | [`TSTORE`](isa/TSTORE.md) | 将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。 |
| 内存（GM <-> Tile） | [`TSTORE_FP`](isa/TSTORE_FP.md) | 历史 fp 量化存储形式的源码兼容 C++ 别名。 |
| 内存（GM <-> Tile） | [`MGATHER`](isa/MGATHER.md) | 使用逐元素索引从全局内存收集加载元素到 Tile 中。 |
| 内存（GM <-> Tile） | [`MSCATTER`](isa/MSCATTER.md) | 使用逐元素索引将 Tile 中的元素散播存储到全局内存。 |
| 矩阵乘 | [`TGEMV_MX`](isa/TGEMV_MX.md) | 带缩放 Tile 的 GEMV 变体，支持混合精度/量化矩阵向量计算。 |
| 矩阵乘 | [`TMATMUL_MX`](isa/TMATMUL_MX.md) | 带额外缩放 Tile 的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。 |
| 矩阵乘 | [`TMATMUL_MX_HIF4`](isa/TMATMUL_MX_HIF4.md) | 用于混合精度 TMATMUL_MX 流程的 HiFloat4 Cube 矩阵乘变体。 |
| 矩阵乘 | [`TMATMUL`](isa/TMATMUL.md) | 矩阵乘法 (GEMM)，生成累加器/输出 Tile。 |
| 矩阵乘 | [`TMATMUL_ACC`](isa/TMATMUL_ACC.md) | 带累加器输入的矩阵乘法（融合累加）。 |
| 矩阵乘 | [`TMATMUL_BIAS`](isa/TMATMUL_BIAS.md) | 带偏置加法的矩阵乘法。 |
| 矩阵乘 | [`TGEMV`](isa/TGEMV.md) | 通用矩阵-向量乘法，生成累加器/输出 Tile。 |
| 矩阵乘 | [`TGEMV_ACC`](isa/TGEMV_ACC.md) | 带显式累加器输入/输出 Tile 的 GEMV。 |
| 矩阵乘 | [`TGEMV_BIAS`](isa/TGEMV_BIAS.md) | 带偏置加法的 GEMV。 |
| 数据搬运 / 布局 | [`TEXTRACT`](isa/TEXTRACT.md) | 从源 Tile 中提取子 Tile。 |
| 数据搬运 / 布局 | [`TEXTRACT_FP`](isa/TEXTRACT_FP.md) | 历史 fp 量化提取形式的源码兼容 C++ 别名。 |
| 数据搬运 / 布局 | [`TIMG2COL`](isa/TIMG2COL.md) | 用于类卷积工作负载的图像到列变换。 |
| 数据搬运 / 布局 | [`TINSERT`](isa/TINSERT.md) | 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。 |
| 数据搬运 / 布局 | [`TINSERT_FP`](isa/TINSERT_FP.md) | 历史 fp 量化插入形式的源码兼容 C++ 别名。 |
| 数据搬运 / 布局 | [`TFILLPAD`](isa/TFILLPAD.md) | 复制 Tile 并在有效区域外使用编译时填充值进行填充。 |
| 数据搬运 / 布局 | [`TFILLPAD_INPLACE`](isa/TFILLPAD_INPLACE.md) | 原地填充/填充变体。 |
| 数据搬运 / 布局 | [`TFILLPAD_EXPAND`](isa/TFILLPAD_EXPAND.md) | 填充/填充时允许目标大于源。 |
| 数据搬运 / 布局 | [`TMOV`](isa/TMOV.md) | 在 Tile 之间移动/复制，可选应用实现定义的转换模式。 |
| 数据搬运 / 布局 | [`TMOV_FP`](isa/TMOV_FP.md) | 历史 fp 量化移动形式的源码兼容 C++ 别名。 |
| 数据搬运 / 布局 | [`TRESHAPE`](isa/TRESHAPE.md) | 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。 |
| 数据搬运 / 布局 | [`TTRANS`](isa/TTRANS.md) | 使用实现定义的临时 Tile 进行转置。 |
| 数据搬运 / 布局 | [`TCONCAT`](isa/TCONCAT.md) | 将两个 Tile 沿列维度水平拼接。 |
| 数据搬运 / 布局 | [`TINTERLEAVE`](isa/TINTERLEAVE.md) | 将两个源 Tile 交织为交替的偶/奇元素流，拆分为两个目标半部分。 |
| 数据搬运 / 布局 | [`TDEINTERLEAVE`](isa/TDEINTERLEAVE.md) | 将源 Tile 反交织为偶数位置和奇数位置的元素流（TINTERLEAVE 的逆操作）。 |
| 复杂指令 | [`TPRINT`](isa/TPRINT.md) | 调试/打印 Tile 中的元素（实现定义）。 |
| 复杂指令 | [`TMRGSORT`](isa/TMRGSORT.md) | 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。 |
| 复杂指令 | [`TSORT32`](isa/TSORT32.md) | 对 `src` 的每个 32 元素块连同对应的 `idx` 条目一起排序，并输出排序后的 value-index 对。 |
| 复杂指令 | [`TGATHER`](isa/TGATHER.md) | 使用索引 Tile 或编译时掩码模式来收集/选择元素。 |
| 复杂指令 | [`TCI`](isa/TCI.md) | 生成连续整数序列到目标 Tile 中。 |
| 复杂指令 | [`TTRI`](isa/TTRI.md) | 生成三角（下/上）掩码 Tile。 |
| 复杂指令 | [`TRANDOM`](isa/TRANDOM.md) | 使用基于计数器的密码算法在目标 Tile 中生成随机数。 |
| 复杂指令 | [`TPARTADD`](isa/TPARTADD.md) | 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTMUL`](isa/TPARTMUL.md) | 部分逐元素乘法，对有效区域不一致的处理为实现定义。 |
| 复杂指令 | [`TPARTMAX`](isa/TPARTMAX.md) | 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTMIN`](isa/TPARTMIN.md) | 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTARGMAX`](isa/TPARTARGMAX.md) | 部分逐元素最大值选择并返回对应索引（argmax），对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TPARTARGMIN`](isa/TPARTARGMIN.md) | 部分逐元素最小值选择并返回对应索引（argmin），对不匹配的有效区域具有实现定义的处理方式。 |
| 复杂指令 | [`TGATHERB`](isa/TGATHERB.md) | 使用字节偏移量收集元素。 |
| 复杂指令 | [`TSCATTER`](isa/TSCATTER.md) | 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。 |
| 复杂指令 | [`TQUANT`](isa/TQUANT.md) | 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。 |
| 复杂指令 | [`TQUANT_DN`](isa/TQUANT_DN.md) | Axis-0 分组量化与 DN 到 ZZ 指数布局转换。 |
| 复杂指令 | [`TQUANT_HIF4`](isa/TQUANT_HIF4.md) | BF16 到 HiFloat4 的量化算法与 CCE 映射。 |
| 复杂指令 | [`TDEQUANT`](isa/TDEQUANT.md) | 对量化 Tile 做仿射反量化（S8/S16 -> FP32）：dst = (src - offset) * scale。 |
| 复杂指令 | [`THISTOGRAM`](isa/THISTOGRAM.md) | 对源元素的某个字节统计直方图（256 桶），可按高位字节级联过滤；基数排序的桶计数原语。 |
| 核间通信 | [`TALLOC`](isa/TALLOC.md) | 将 TPipe FIFO 槽位分配为一个 GlobalTensor 视图。 |
| 核间通信 | [`TPUSH`](isa/TPUSH.md) | 将生产者 tile 推入 TPipe FIFO，用于 Cube-Vector 通信。 |
| 核间通信 | [`TPOP`](isa/TPOP.md) | 从 TPipe FIFO 弹出消费者 tile/globalTensor，用于 Cube-Vector 通信。 |
| 核间通信 | [`TFREE`](isa/TFREE.md) | 释放 TPipe 的 FIFO 空间；部分目标上对 TileData TPOP 流程为空操作。 |
| 通信 | [`TPUT`](isa/comm/TPUT.md) | 远程写：将本地数据传输到远端 NPU 内存（GM → UB → GM）。 |
| 通信 | [`TGET`](isa/comm/TGET.md) | 远程读：将远端 NPU 数据读取到本地内存（GM → UB → GM）。 |
| 通信 | [`TPUT_ASYNC`](isa/comm/TPUT_ASYNC.md) | 异步远程写（本地 GM → DMA 引擎 → 远端 GM）。 |
| 通信 | [`TGET_ASYNC`](isa/comm/TGET_ASYNC.md) | 异步远程读（远端 GM → DMA 引擎 → 本地 GM）。 |
| 通信 | [`TNOTIFY`](isa/comm/TNOTIFY.md) | 向远端 NPU 发送标志通知。 |
| 通信 | [`TWAIT`](isa/comm/TWAIT.md) | 阻塞等待，直到信号满足比较条件。 |
| 通信 | [`TTEST`](isa/comm/TTEST.md) | 非阻塞检测信号是否满足比较条件。 |
| 通信 | [`TGATHER`](isa/comm/TGATHER.md) | 从所有 rank 收集数据并沿 DIM_3 拼接。 |
| 通信 | [`TSCATTER`](isa/comm/TSCATTER.md) | 将数据沿 DIM_3 拆分并分发到所有 rank。 |
| 通信 | [`TREDUCE`](isa/comm/TREDUCE.md) | 从所有 rank 收集数据并逐元素归约到本地。 |
| 通信 | [`TBROADCAST`](isa/comm/TBROADCAST.md) | 将当前 NPU 的数据广播到所有 rank。 |
