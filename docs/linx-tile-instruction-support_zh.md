# Linx 指令支持与受限实验接入

[English](linx-tile-instruction-support.md)

Linx 原生 NPU 后端当前定位为**受限实验接入**。接口及上游实现保留，已知不支持路径在编译时明确拒绝；CPU-SIM 提供相同调用接口的主机执行。这里的“原生实测”指本机 `gfrun` 功能模型执行，不是物理 NPU 验证。

## 按指令统计的验收口径

主口径为 **151 个普通指令调用名**；11 个 `_U` 和 64 个 `_ASS` 形式逐名另列。全部覆盖的活跃指令接口合计 226 个，不能称为 226 个独立硬件 opcode。4 个视图/构造 helper、32 个退休声明以及小写标量/Predicate 辅助接口不混入分母。

一个调用名只有在本矩阵中归属它的所有场景都通过时，才进入“通过指令数”。同名多个重载可显示“部分开放”；明确拒绝不是数值通过。覆盖指已构造并实际尝试这些场景，不表示穷尽 dtype、布局、shape、控制字、别名和所有重载。

| 统计组 | CPU：全部列出场景通过 | NPU 模型：全部列出场景通过 |
| --- | ---: | ---: |
| 普通指令调用名 | 151/151 | 109/151 |
| 显式维度 `_U` | 11/11 | 11/11 |
| Associated `_ASS` | 64/64 | 1/64 |
| 合计活跃指令接口名 | **226/226** | **121/226** |

上述结果来自 229 个场景：CPU 229 个通过；未加限制的导入实现与固定上游实现均为 122 个通过、107 个未通过。NPU 模型 122 个场景执行通过，107 个明确拒绝，没有其余未处理的失败。对应 105 个调用名全部或部分受限，不能把这 105 个计成执行通过。

## 固定环境与判定

| Component | Revision |
| --- | --- |
| Linx-TileOP-API (`linx`) | `74d7326c521aeb1b21c52602b67c55b42b3563a3` |
| SuperScalarModel / gfrun | `637bde0ec63955c4796828b297e16a057efad3cc` |
| LinxV5 Clang 15.0.4 | `b19e8c0e184cd7e3a1fa5ab57b368cbf978c3aa8` |
| PTO-spec reference | `42067d0f2890b8c013a868bd1247ebc6437b3179` |
| Host | aarch64 Linux / GCC 15.2.1 |

[测试清单](../tests/cpu/linx/instruction_catalog.json) 与两种公共入口的预处理定义逐项比较；遗漏或新增调用名会使检查失败。[场景构造](../tests/cpu/linx_instruction_cases.py) 为每个活跃调用名指定所属测试和 golden。

通过要求编译/链接成功、进程成功、guest 完成标志、无模型错误诊断、输出逐位符合独立 golden、16 个尾部 canary 完好。三路成功结果逐字节一致。模型 rc=0 或两条原生路径同样失败都不构成通过。

在固定依赖下，未加限制的原生实现有 107 个场景未通过，按实测阶段分为：9 个编译/汇编失败、26 个正常退出但数值/内存状态错误、11 个其他执行错误，以及 61 个 `_ASS` 场景的前置 TLOAD INIT 失败。后者没有执行到目标算术，不能据此断言目标指令自身算错。关联指令的端到端场景仍不能验收。11 个执行错误中还包括两个打印 ERROR 后 rc=0、输出仍错误的 reload；正常退出但输出错误的场景合计 28 个，其中 26 个没有错误诊断。

## 能力边界与明确拒绝

[限制定义](../include/pto/npu/linx/experimental.hpp) 和[机械接入脚本](../tests/script/sync_linx_headers.py) 在原 emitter 前加入诊断，保留原函数签名、校验及汇编文本。编译提示以 `PTO Linx experimental:` 开头，给出原因和文档位置。模板路径在实例化阶段拒绝，避免进入已知的编译器后端崩溃。少量具有非模板别名的辅助形式使用 error 属性，并在保留的原 asm 之前终止该路径；`-fsyntax-only` 对后者不足以验收，必须进行代码生成。

无绕过宏。诊断上游执行问题时使用独立的未加限制导入副本或固定上游源码，不在产品构建中关闭限制。CPU 生成器仅去除带边界标记的接入诊断，仍保留原生合法性检查。独立 admission 脚本使用 `NDEBUG`、`-O0/-O2` 和公共/内部头入口完成 428 次检查，要求明确诊断且无编译器内部崩溃；107 个受限场景全部满足。

| 受限路径 | 实测原因与边界 | 上游记录/证据 |
| --- | --- | --- |
| 带范围 `MGATHER_EXCH/MAX/MIN/ADD/INC/DEC/AND/OR/XOR`、`MSCATTER_MAX/MIN/ADD/INC/DEC/AND/OR/XOR/POPC` | 18 个原子场景得到错误输出，且模型可正常退出；range 编码与当前 schema 的 B.IOR 未用字段冲突。无范围形式继续开放。 | [TileOP #317](https://github.com/LinxISA/Linx-TileOP-API/issues/317)、[Model #1252](https://github.com/LinxISA/SuperScalarModel/issues/1252) |
| `TSTORE_ATOMIC_*` 八种模式 | 正常退出但执行覆盖写。例如 ADD 的首元素实际为 5，预期为 3+5=8；这不是可容忍的浮点误差。全部拒绝。 | [Model #1119](https://github.com/LinxISA/SuperScalarModel/issues/1119#issuecomment-6813968508) |
| TileArray/SubTileView 运算、通用 `range::Assemble` TLOAD、61 种 TEPL `_ASS` 形式 | `regions` 在 TADD INIT writer 被拒绝；`_ASS` 场景在前置 TLOAD INIT 被拒绝，目标算术尚未到达。限制的是该执行路径，不能认定 61 个算术操作各有独立缺陷。 | [Model #1253](https://github.com/LinxISA/SuperScalarModel/issues/1253)；前置 TLOAD INIT 的模型诊断 |
| `TGPR2T` | 模型拒绝属性、形状和 3+1 B.IOR schema；编译成功不代表执行成功。 | [TileOP #318](https://github.com/LinxISA/Linx-TileOP-API/issues/318) |
| Local `TIMG2COL` | 编译成功，空 shard 所在 PE 的 callee spill TSTORE 触发模型断言。 | [LLVM #164](https://github.com/LinxISA/llvm-project/issues/164) |
| `TMATMUL_MX_ACC` | basic ACC 的源绑定顺序与契约不符。独立 plain MX 和 BIAS 用例通过，继续开放。 | [TileOP #319](https://github.com/LinxISA/Linx-TileOP-API/issues/319) |
| Predicate `atomic_add`、`subview`、`concatenate` | mask/range 合同或 TMOV 整容量校验阻塞；组合后续步骤未到达，不推断其独立结果。 | [Model #1131](https://github.com/LinxISA/SuperScalarModel/issues/1131#issuecomment-6813972601)、[Model #1254](https://github.com/LinxISA/SuperScalarModel/issues/1254) |
| `linx_cvt`/packed 转换、BF16 scalar/packed 算术 | 标量转换触发 Clang legalization 崩溃；packed CVT/ADD 被汇编器拒绝。当前保守禁用该辅助族，不计入 Tile 指令分母。 | [LLVM #165](https://github.com/LinxISA/llvm-project/issues/165)、[LLVM #166](https://github.com/LinxISA/llvm-project/issues/166) |
| `TLOAD2_*`、`TLOAD4_*`、`TSTORE_PART` | 六种 load 的多 binder 编码汇编失败；`TSTORE.SPART` mnemonic 不受支持。 | 独立执行/汇编日志 |
| `TSTORE2_DN2DN` | 模型要求普通 Local TSTORE 只有一个源 Tile；双源形式触发断言。 | 独立用例 |
| `TMOV_L2S_INSERT/PUBLISH`、Shared ND/CUBE 转换 | Shared writer 范围或 TMOV 描述符被拒绝。复合流程首步失败时，后续操作状态为未到达；当前整体受限。 | 独立流程 |
| 普通已分配 Tile 的 `TLOAD_CUBE_ASS`/`TLOAD_ASS` reload | 模型打印 `ERROR: TLOAD with empty dstTile!` 后仍正常退出；128 个输出保留旧值而非新输入。专用 `LocalCubeAssembly` 的 `TLOAD_ASS` 继续开放。 | reload 对照 |

普通 `MGATHER_MASK/MSCATTER_MASK` 的 U8 字节掩码及带范围重载仍开放，不能与带范围原子操作混淆。RowMajor `TCMP/TCMPS` 的 packed-bit 结果不能当作 indexed U8 0/1 掩码。

开放受限路径需要先固定通过验证的 API/编译器/模型版本，复跑相同输入、独立 golden 和失败前无副作用检查，再移除对应诊断并更新中英文矩阵。上游 issue 关闭或单独编译成功不能替代执行验收。

## 数值与测试范围

- `TEXPDIF` 的同类型路径先按源/目标类型舍入减法，混合精度路径先扩展到 FP32；随后执行指数和输出舍入。每种 FP16/BF16 格式覆盖 128 个能区分中间舍入方式的输入。
- `TRSQRT` 的实验 CPU-SIM 契约明确跟随固定模型：先按操作数类型舍入 SQRT，再执行 DIV，包含 FP16/BF16。固定 PTO-spec 的高精度倒平方根参考与此不同，两者存在数值契约差异。每种低精度格式覆盖 128 个差异输入。
- `TPACK_ASS` 的 CPU 实现按当前 writer 的源 raw-word 数量计算宽度，读取范围限于当前 writer；独立 M32 两片段用例三路通过。
- FP32 向量、行列归约、比较/选择、索引和位操作使用非恒定位置数据；原子同时检查旧值与 GM 更新；Shared/slot 使用四 PE；MX ACC/BIAS 分别运行。TPREFETCH 只检查调用完成与后续数据，CPU 不模拟缓存。
- 所列测试不证明超越函数全输入一致、硬件资源/时序/cache、原子竞争的所有顺序或未定义数据行为。详细 CPU 约束见[后端说明](../include/pto/npu/linx/README_zh.md)。

## 逐指令验证矩阵

“NPU 模型”的部分开放表示至少一个场景通过、至少一个被拒绝。场景 ID 可直接传给 `--case`，具体 dtype、布局、输入、调用和 golden 均保存在测试源码及生成结果中。

### 普通调用名

| 指令 | CPU | NPU 模型 | 场景 ID |
| --- | --- | --- | --- |
| `GMOV` | 通过 | 通过 | `core_gmov` |
| `MGATHER` | 通过 | 通过 | `indexed_mask` |
| `MGATHER_ADD` | 通过 | 部分开放 | `gather_atomic_3`, `gather_atomic_3_unbounded` |
| `MGATHER_AND` | 通过 | 部分开放 | `gather_atomic_6`, `gather_atomic_6_unbounded` |
| `MGATHER_CAS` | 通过 | 通过 | `compare_exchange` |
| `MGATHER_DEC` | 通过 | 部分开放 | `gather_atomic_5`, `gather_atomic_5_unbounded` |
| `MGATHER_EXCH` | 通过 | 部分开放 | `gather_atomic_0`, `gather_atomic_0_unbounded` |
| `MGATHER_INC` | 通过 | 部分开放 | `gather_atomic_4`, `gather_atomic_4_unbounded` |
| `MGATHER_MASK` | 通过 | 通过 | `indexed_mask` |
| `MGATHER_MAX` | 通过 | 部分开放 | `gather_atomic_1`, `gather_atomic_1_unbounded` |
| `MGATHER_MIN` | 通过 | 部分开放 | `gather_atomic_2`, `gather_atomic_2_unbounded` |
| `MGATHER_OR` | 通过 | 部分开放 | `gather_atomic_7`, `gather_atomic_7_unbounded` |
| `MGATHER_XOR` | 通过 | 部分开放 | `gather_atomic_8`, `gather_atomic_8_unbounded` |
| `MSCATTER` | 通过 | 通过 | `MSCATTER` |
| `MSCATTER_ADD` | 通过 | 部分开放 | `scatter_atomic_2`, `scatter_atomic_2_unbounded` |
| `MSCATTER_AND` | 通过 | 部分开放 | `scatter_atomic_5`, `scatter_atomic_5_unbounded` |
| `MSCATTER_DEC` | 通过 | 部分开放 | `scatter_atomic_4`, `scatter_atomic_4_unbounded` |
| `MSCATTER_INC` | 通过 | 部分开放 | `scatter_atomic_3`, `scatter_atomic_3_unbounded` |
| `MSCATTER_MASK` | 通过 | 通过 | `indexed_mask` |
| `MSCATTER_MAX` | 通过 | 部分开放 | `scatter_atomic_0`, `scatter_atomic_0_unbounded` |
| `MSCATTER_MIN` | 通过 | 部分开放 | `scatter_atomic_1`, `scatter_atomic_1_unbounded` |
| `MSCATTER_OR` | 通过 | 部分开放 | `scatter_atomic_6`, `scatter_atomic_6_unbounded` |
| `MSCATTER_POPC` | 通过 | 部分开放 | `scatter_atomic_8`, `scatter_atomic_8_unbounded` |
| `MSCATTER_XOR` | 通过 | 部分开放 | `scatter_atomic_7`, `scatter_atomic_7_unbounded` |
| `TABS` | 通过 | 通过 | `TABS` |
| `TADD` | 通过 | 部分开放 | `TADD`, `regions` |
| `TADDS` | 通过 | 通过 | `TADDS`, `dynamic_shape` |
| `TAND` | 通过 | 通过 | `TAND` |
| `TANDS` | 通过 | 通过 | `select_bits` |
| `TCI` | 通过 | 通过 | `TCI` |
| `TCI_2D` | 通过 | 通过 | `TCI_2D` |
| `TCI_COL` | 通过 | 通过 | `TCI_COL` |
| `TCI_ROW` | 通过 | 通过 | `TCI_ROW` |
| `TCMP` | 通过 | 通过 | `compare_move` |
| `TCMPS` | 通过 | 通过 | `select_bits` |
| `TCOLARGMAX` | 通过 | 通过 | `TCOLARGMAX` |
| `TCOLARGMIN` | 通过 | 通过 | `TCOLARGMIN` |
| `TCOLEXPAND` | 通过 | 通过 | `broadcast_col_0` |
| `TCOLEXPANDADD` | 通过 | 通过 | `broadcast_col_1` |
| `TCOLEXPANDDIV` | 通过 | 通过 | `broadcast_col_4` |
| `TCOLEXPANDEXPDIF` | 通过 | 通过 | `broadcast_col_7` |
| `TCOLEXPANDMAX` | 通过 | 通过 | `broadcast_col_5` |
| `TCOLEXPANDMIN` | 通过 | 通过 | `broadcast_col_6` |
| `TCOLEXPANDMUL` | 通过 | 通过 | `broadcast_col_3` |
| `TCOLEXPANDSUB` | 通过 | 通过 | `broadcast_col_2` |
| `TCOLMAX` | 通过 | 通过 | `TCOLMAX` |
| `TCOLMIN` | 通过 | 通过 | `TCOLMIN` |
| `TCOLPROD` | 通过 | 通过 | `TCOLPROD` |
| `TCOLSUM` | 通过 | 通过 | `TCOLSUM` |
| `TCVT` | 通过 | 通过 | `fp16_convert` |
| `TCVT_T` | 通过 | 通过 | `TCVT_T` |
| `TDIV` | 通过 | 通过 | `TDIV` |
| `TDIVS` | 通过 | 通过 | `TDIVS` |
| `TEXP` | 通过 | 通过 | `TEXP` |
| `TEXPANDS` | 通过 | 通过 | `rearrange` |
| `TEXPDIF` | 通过 | 通过 | `TEXPDIF`, `TEXPDIF_fp16`, `TEXPDIF_bf16` |
| `TFMA` | 通过 | 通过 | `TFMA` |
| `TGATHER` | 通过 | 通过 | `tile_indexed` |
| `TGEMV` | 通过 | 通过 | `gemv` |
| `TGEMV_ACC` | 通过 | 通过 | `gemv` |
| `TGEMV_BIAS` | 通过 | 通过 | `gemv` |
| `TGEMV_MX` | 通过 | 通过 | `gemv` |
| `TGEMV_MX_ACC` | 通过 | 通过 | `gemv` |
| `TGEMV_MX_BIAS` | 通过 | 通过 | `gemv` |
| `TGPR2T` | 通过 | 拒绝 | `gpr_bits` |
| `TIMG2COL` | 通过 | 拒绝 | `image_nhwc` |
| `TIMG2COL_SPART` | 通过 | 通过 | `image_nchw` |
| `TIMG2COL_SPART_SLOT` | 通过 | 通过 | `matrix_slots` |
| `TLOAD` | 通过 | 通过 | `shared_load`, `shared_single_producer`, `weights`, `dynamic_shape` |
| `TLOAD2_DN2ZN` | 通过 | 拒绝 | `TLOAD2_DN2ZN` |
| `TLOAD2_ND2NZ` | 通过 | 拒绝 | `TLOAD2_ND2NZ` |
| `TLOAD2_ND2ZN` | 通过 | 拒绝 | `TLOAD2_ND2ZN` |
| `TLOAD4_DN2ZN` | 通过 | 拒绝 | `TLOAD4_DN2ZN` |
| `TLOAD4_ND2NZ` | 通过 | 拒绝 | `TLOAD4_ND2NZ` |
| `TLOAD4_ND2ZN` | 通过 | 拒绝 | `TLOAD4_ND2ZN` |
| `TLOAD_CUBE` | 通过 | 通过 | `cube_transport` |
| `TLOAD_SLOT` | 通过 | 通过 | `matrix_slots` |
| `TLOG` | 通过 | 通过 | `TLOG` |
| `TMATMUL` | 通过 | 通过 | `matrix_shared`, `matrix_aux` |
| `TMATMUL_ACC` | 通过 | 通过 | `matrix_acc_scale` |
| `TMATMUL_BIAS` | 通过 | 通过 | `TMATMUL_BIAS` |
| `TMATMUL_MX` | 通过 | 通过 | `matrix_hif4`, `TMATMUL_MX_isolated` |
| `TMATMUL_MX_ACC` | 通过 | 拒绝 | `TMATMUL_MX_ACC_isolated` |
| `TMATMUL_MX_BIAS` | 通过 | 通过 | `TMATMUL_MX_BIAS_isolated` |
| `TMATMUL_SLOT` | 通过 | 通过 | `matrix_slots` |
| `TMAX` | 通过 | 通过 | `TMAX` |
| `TMAXS` | 通过 | 通过 | `TMAXS` |
| `TMIN` | 通过 | 通过 | `TMIN` |
| `TMINS` | 通过 | 通过 | `TMINS` |
| `TMOV` | 通过 | 通过 | `compare_move` |
| `TMOV_CUBE_TO_SHARED_ND` | 通过 | 拒绝 | `shared_nd` |
| `TMOV_L2S_INSERT` | 通过 | 拒绝 | `local_publish` |
| `TMOV_L2S_PUBLISH` | 通过 | 拒绝 | `local_publish` |
| `TMOV_S2L_BROADCAST` | 通过 | 通过 | `shared_load` |
| `TMOV_S2L_EXTRACT` | 通过 | 通过 | `image_nchw` |
| `TMOV_SHARED_ND_TO_CUBE` | 通过 | 拒绝 | `shared_nd` |
| `TMUL` | 通过 | 通过 | `TMUL` |
| `TMULS` | 通过 | 通过 | `TMULS` |
| `TNEG` | 通过 | 通过 | `TNEG` |
| `TNOT` | 通过 | 通过 | `select_bits` |
| `TOR` | 通过 | 通过 | `TOR` |
| `TORS` | 通过 | 通过 | `select_bits` |
| `TPACK` | 通过 | 通过 | `rearrange` |
| `TPERMUTE` | 通过 | 通过 | `rearrange` |
| `TPREFETCH` | 通过 | 通过 | `compare_move` |
| `TRECIP` | 通过 | 通过 | `TRECIP` |
| `TRELU` | 通过 | 通过 | `TRELU` |
| `TREM` | 通过 | 通过 | `TREM` |
| `TREMS` | 通过 | 通过 | `TREMS` |
| `TROWARGMAX` | 通过 | 通过 | `TROWARGMAX` |
| `TROWARGMIN` | 通过 | 通过 | `TROWARGMIN` |
| `TROWEXPAND` | 通过 | 通过 | `broadcast_row_0` |
| `TROWEXPANDADD` | 通过 | 通过 | `broadcast_row_1` |
| `TROWEXPANDDIV` | 通过 | 通过 | `broadcast_row_4` |
| `TROWEXPANDEXPDIF` | 通过 | 通过 | `broadcast_row_7` |
| `TROWEXPANDMAX` | 通过 | 通过 | `broadcast_row_5` |
| `TROWEXPANDMIN` | 通过 | 通过 | `broadcast_row_6` |
| `TROWEXPANDMUL` | 通过 | 通过 | `broadcast_row_3` |
| `TROWEXPANDSUB` | 通过 | 通过 | `broadcast_row_2` |
| `TROWMAX` | 通过 | 通过 | `TROWMAX` |
| `TROWMIN` | 通过 | 通过 | `TROWMIN` |
| `TROWPROD` | 通过 | 通过 | `TROWPROD` |
| `TROWSUM` | 通过 | 通过 | `TROWSUM` |
| `TRSQRT` | 通过 | 通过 | `TRSQRT`, `TRSQRT_fp16`, `TRSQRT_bf16` |
| `TSCATTER` | 通过 | 通过 | `tile_indexed` |
| `TSEL` | 通过 | 通过 | `compare_move` |
| `TSELS` | 通过 | 通过 | `select_bits` |
| `TSHL` | 通过 | 通过 | `TSHL` |
| `TSHLS` | 通过 | 通过 | `select_bits` |
| `TSHR` | 通过 | 通过 | `TSHR` |
| `TSHRS` | 通过 | 通过 | `select_bits` |
| `TSHUF` | 通过 | 通过 | `rearrange` |
| `TSQRT` | 通过 | 通过 | `TSQRT` |
| `TSTORE` | 通过 | 通过 | `dynamic_shape` |
| `TSTORE2_DN2DN` | 通过 | 拒绝 | `TSTORE2_DN2DN` |
| `TSTORE_ATOMIC_ADD` | 通过 | 拒绝 | `store_atomic_2` |
| `TSTORE_ATOMIC_AND` | 通过 | 拒绝 | `store_atomic_5` |
| `TSTORE_ATOMIC_DEC` | 通过 | 拒绝 | `store_atomic_4` |
| `TSTORE_ATOMIC_INC` | 通过 | 拒绝 | `store_atomic_3` |
| `TSTORE_ATOMIC_MAX` | 通过 | 拒绝 | `store_atomic_0` |
| `TSTORE_ATOMIC_MIN` | 通过 | 拒绝 | `store_atomic_1` |
| `TSTORE_ATOMIC_OR` | 通过 | 拒绝 | `store_atomic_6` |
| `TSTORE_ATOMIC_XOR` | 通过 | 拒绝 | `store_atomic_7` |
| `TSTORE_CUBE` | 通过 | 通过 | `cube_transport` |
| `TSTORE_PART` | 通过 | 拒绝 | `TSTORE_PART` |
| `TSUB` | 通过 | 通过 | `TSUB` |
| `TSUBS` | 通过 | 通过 | `TSUBS` |
| `TTRI` | 通过 | 通过 | `TTRI` |
| `TUNPACK` | 通过 | 通过 | `rearrange` |
| `TXOR` | 通过 | 通过 | `TXOR` |
| `TXORS` | 通过 | 通过 | `select_bits` |

### 显式维度形式

| 指令 | CPU | NPU 模型 | 场景 ID |
| --- | --- | --- | --- |
| `TADD_U` | 通过 | 通过 | `TADD_U` |
| `TAND_U` | 通过 | 通过 | `TAND_U` |
| `TDIV_U` | 通过 | 通过 | `TDIV_U` |
| `TMAX_U` | 通过 | 通过 | `TMAX_U` |
| `TMIN_U` | 通过 | 通过 | `TMIN_U` |
| `TMUL_U` | 通过 | 通过 | `TMUL_U` |
| `TOR_U` | 通过 | 通过 | `TOR_U` |
| `TSHL_U` | 通过 | 通过 | `TSHL_U` |
| `TSHR_U` | 通过 | 通过 | `TSHR_U` |
| `TSUB_U` | 通过 | 通过 | `TSUB_U` |
| `TXOR_U` | 通过 | 通过 | `TXOR_U` |

### Associated 形式

| 指令 | CPU | NPU 模型 | 场景 ID |
| --- | --- | --- | --- |
| `TABS_ASS` | 通过 | 拒绝 | `TABS_ASS` |
| `TADDS_ASS` | 通过 | 拒绝 | `TADDS_ASS` |
| `TADD_ASS` | 通过 | 拒绝 | `TADD_ASS` |
| `TANDS_ASS` | 通过 | 拒绝 | `TANDS_ASS` |
| `TAND_ASS` | 通过 | 拒绝 | `TAND_ASS` |
| `TCMPS_ASS` | 通过 | 拒绝 | `TCMPS_ASS` |
| `TCMP_ASS` | 通过 | 拒绝 | `TCMP_ASS` |
| `TCOLEXPANDADD_ASS` | 通过 | 拒绝 | `TCOLEXPANDADD_ASS` |
| `TCOLEXPANDDIV_ASS` | 通过 | 拒绝 | `TCOLEXPANDDIV_ASS` |
| `TCOLEXPANDEXPDIF_ASS` | 通过 | 拒绝 | `TCOLEXPANDEXPDIF_ASS` |
| `TCOLEXPANDMAX_ASS` | 通过 | 拒绝 | `TCOLEXPANDMAX_ASS` |
| `TCOLEXPANDMIN_ASS` | 通过 | 拒绝 | `TCOLEXPANDMIN_ASS` |
| `TCOLEXPANDMUL_ASS` | 通过 | 拒绝 | `TCOLEXPANDMUL_ASS` |
| `TCOLEXPANDSUB_ASS` | 通过 | 拒绝 | `TCOLEXPANDSUB_ASS` |
| `TCOLEXPAND_ASS` | 通过 | 拒绝 | `TCOLEXPAND_ASS` |
| `TCOLMAX_ASS` | 通过 | 拒绝 | `TCOLMAX_ASS` |
| `TCOLMIN_ASS` | 通过 | 拒绝 | `TCOLMIN_ASS` |
| `TCOLPROD_ASS` | 通过 | 拒绝 | `TCOLPROD_ASS` |
| `TCOLSUM_ASS` | 通过 | 拒绝 | `TCOLSUM_ASS` |
| `TCVT_ASS` | 通过 | 拒绝 | `TCVT_ASS` |
| `TDIVS_ASS` | 通过 | 拒绝 | `TDIVS_ASS` |
| `TDIV_ASS` | 通过 | 拒绝 | `TDIV_ASS` |
| `TEXP_ASS` | 通过 | 拒绝 | `TEXP_ASS` |
| `TFMA_ASS` | 通过 | 拒绝 | `TFMA_ASS` |
| `TLOAD_ASS` | 通过 | 部分开放 | `local_assembly`, `TLOAD_ASS_reload` |
| `TLOAD_CUBE_ASS` | 通过 | 拒绝 | `TLOAD_CUBE_ASS` |
| `TLOG_ASS` | 通过 | 拒绝 | `TLOG_ASS` |
| `TMAXS_ASS` | 通过 | 拒绝 | `TMAXS_ASS` |
| `TMAX_ASS` | 通过 | 拒绝 | `TMAX_ASS` |
| `TMINS_ASS` | 通过 | 拒绝 | `TMINS_ASS` |
| `TMIN_ASS` | 通过 | 拒绝 | `TMIN_ASS` |
| `TMULS_ASS` | 通过 | 拒绝 | `TMULS_ASS` |
| `TMUL_ASS` | 通过 | 拒绝 | `TMUL_ASS` |
| `TNEG_ASS` | 通过 | 拒绝 | `TNEG_ASS` |
| `TNOT_ASS` | 通过 | 拒绝 | `TNOT_ASS` |
| `TORS_ASS` | 通过 | 拒绝 | `TORS_ASS` |
| `TOR_ASS` | 通过 | 拒绝 | `TOR_ASS` |
| `TPACK_ASS` | 通过 | 通过 | `TPACK_ASS` |
| `TRECIP_ASS` | 通过 | 拒绝 | `TRECIP_ASS` |
| `TRELU_ASS` | 通过 | 拒绝 | `TRELU_ASS` |
| `TREMS_ASS` | 通过 | 拒绝 | `TREMS_ASS` |
| `TREM_ASS` | 通过 | 拒绝 | `TREM_ASS` |
| `TROWEXPANDADD_ASS` | 通过 | 拒绝 | `TROWEXPANDADD_ASS` |
| `TROWEXPANDDIV_ASS` | 通过 | 拒绝 | `TROWEXPANDDIV_ASS` |
| `TROWEXPANDEXPDIF_ASS` | 通过 | 拒绝 | `TROWEXPANDEXPDIF_ASS` |
| `TROWEXPANDMAX_ASS` | 通过 | 拒绝 | `TROWEXPANDMAX_ASS` |
| `TROWEXPANDMIN_ASS` | 通过 | 拒绝 | `TROWEXPANDMIN_ASS` |
| `TROWEXPANDMUL_ASS` | 通过 | 拒绝 | `TROWEXPANDMUL_ASS` |
| `TROWEXPANDSUB_ASS` | 通过 | 拒绝 | `TROWEXPANDSUB_ASS` |
| `TROWEXPAND_ASS` | 通过 | 拒绝 | `TROWEXPAND_ASS` |
| `TROWMAX_ASS` | 通过 | 拒绝 | `TROWMAX_ASS` |
| `TROWMIN_ASS` | 通过 | 拒绝 | `TROWMIN_ASS` |
| `TROWPROD_ASS` | 通过 | 拒绝 | `TROWPROD_ASS` |
| `TROWSUM_ASS` | 通过 | 拒绝 | `TROWSUM_ASS` |
| `TRSQRT_ASS` | 通过 | 拒绝 | `TRSQRT_ASS` |
| `TSHLS_ASS` | 通过 | 拒绝 | `TSHLS_ASS` |
| `TSHL_ASS` | 通过 | 拒绝 | `TSHL_ASS` |
| `TSHRS_ASS` | 通过 | 拒绝 | `TSHRS_ASS` |
| `TSHR_ASS` | 通过 | 拒绝 | `TSHR_ASS` |
| `TSQRT_ASS` | 通过 | 拒绝 | `TSQRT_ASS` |
| `TSUBS_ASS` | 通过 | 拒绝 | `TSUBS_ASS` |
| `TSUB_ASS` | 通过 | 拒绝 | `TSUB_ASS` |
| `TXORS_ASS` | 通过 | 拒绝 | `TXORS_ASS` |
| `TXOR_ASS` | 通过 | 拒绝 | `TXOR_ASS` |

## 非指令项

`TPARTVIEW/TASSEMBLY` 在 `regions` 中执行，CPU 通过、原生受限。`TPARTELEMENT/TREDUCEPREFIXVIEW` 尚无数值验收结果；它们属于视图 helper，不计入 226 的指令覆盖声明。小写辅助接口另有 `scalar_rounding`、`packed_bf16`、`predicate_atomic`、`predicate_regions` 场景；其通过/拒绝不能换算成 Tile opcode 数量。

退休声明仅作接口清单和拒绝约束保留，不计为可运行指令：

`ACCCVT`, `ACCCVT_RMAX_SCALE_NZ2DN`, `ACCSCALE_NZ2DN`, `ACCSCALE_T`, `TADDC`, `TADDSC`, `TAXPY`, `TCONCAT`, `TCONCAT_ASS`, `TDEQUANT`, `TEXTRACT`, `TFILLPAD`, `TFMOD`, `TFMODS`, `TGATHERB`, `THISTOGRAM`, `TINSERT`, `TLRELU`, `TMRGSORT`, `TPARTADD`, `TPARTMAX`, `TPARTMIN`, `TPARTMUL`, `TPRELU`, `TQUANT`, `TRANDOM`, `TSORT`, `TSORT32`, `TSUBC`, `TSUBSC`, `TTRANS`, `TTRANS_ASS`

## 复跑

```bash
# Run the instruction suite through CPU-SIM and the integrated NPU backend.
python3 tests/cpu/test_linx_models.py --compiler /path/to/g++-15 \
  --linx-compiler /path/to/linx-clang++ \
  --model-root /path/to/SuperScalarModel --runtime-dir /path/to/target-libs \
  --upstream /path/to/Linx-TileOP-API --output-dir /tmp/linx-instructions \
  --routes cpu device --jobs 8 --timeout 30
# --case TPACK_ASS (repeatable) selects an independent profile.
# --routes upstream runs the pinned upstream implementation for comparison.
# Admission requires matching profiles from an unguarded diagnostic import.
# --device-include /path/to/unguarded-import/include selects that import.
python3 tests/cpu/test_linx_admission.py \
  --baseline /tmp/unguarded-instructions/result.json \
  --restricted /tmp/linx-instructions/result.json \
  --output-dir /tmp/linx-admission --jobs 8
python3 tests/cpu/test_linx_native.py --compiler /path/to/g++-15 \
  --linx-compiler /path/to/linx-clang++ --upstream /path/to/Linx-TileOP-API \
  --survey --sanitize --release --output-dir /tmp/linx-regression
python3 tests/script/sync_linx_headers.py --upstream /path/to/Linx-TileOP-API
```

完整数值审计以非零退出码报告原生受限/失败项；admission 脚本单独报告“所有开放场景通过、已知失败全部明确拒绝”的检查结果。两者不可混作同一个通过率。测试结果目录保存 ELF、原始 dump、trace 和完整命令，供复核执行结果。
