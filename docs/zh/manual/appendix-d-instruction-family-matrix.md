# 附录 D. 指令族矩阵

## D.1 范围

本附录由 `docs/isa/manifest.yaml` 自动生成，用于给出 PTO 虚拟指令族的源同步矩阵。

## D.2 覆盖统计

| 分类 | 指令数量 |
|---|---:|
| 同步 | 1 |
| 手动 / 资源绑定 | 6 |
| 逐元素（Tile-Tile） | 29 |
| Tile-标量 / Tile-立即数 | 19 |
| 轴归约 / 扩展 | 28 |
| 内存（GM <-> Tile） | 7 |
| 矩阵乘 | 9 |
| 数据搬运 / 布局 | 15 |
| 复杂指令 | 20 |
| Cross-core Communication | 4 |
| 总计 | 138 |

## D.3 头文件同步状态

- 头文件清单来源：`include/pto/common/pto_instr.hpp`（134 个唯一指令 API）
- Manifest 清单来源：`docs/isa/manifest.yaml`（138 条目）
- 头文件有但 manifest 缺失：无
- manifest 有但头文件缺失：TMATMUL_MX_HIF4, TPREFETCH_ASYNC, TQUANT_DN, TQUANT_HIF4

## D.4 指令族矩阵

| 分类 | 指令 | 图示模板 | 操作数契约 | 语义页面 |
|---|---|---|---|---|
| 同步 | `SYNCALL` | `sync` | `participants` | `docs/zh/isa/SYNCALL.md` |
| 手动 / 资源绑定 | `TASSIGN` | `config` | `config, state` | `docs/zh/isa/TASSIGN.md` |
| 手动 / 资源绑定 | `SETFMATRIX` | `config` | `config, state` | `docs/zh/isa/SETFMATRIX.md` |
| 手动 / 资源绑定 | `SET_IMG2COL_RPT` | `config` | `config, state` | `docs/zh/isa/SET_IMG2COL_RPT.md` |
| 手动 / 资源绑定 | `SET_IMG2COL_PADDING` | `config` | `config, state` | `docs/zh/isa/SET_IMG2COL_PADDING.md` |
| 手动 / 资源绑定 | `SET_QUANT_SCALAR` | `config` | `scale` | `docs/zh/isa/SET_QUANT_SCALAR.md` |
| 手动 / 资源绑定 | `SET_QUANT_VECTOR` | `config` | `fpTile` | `docs/zh/isa/SET_QUANT_VECTOR.md` |
| 逐元素（Tile-Tile） | `TADD` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TADD.md` |
| 逐元素（Tile-Tile） | `TABS` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TABS.md` |
| 逐元素（Tile-Tile） | `TAND` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TAND.md` |
| 逐元素（Tile-Tile） | `TOR` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TOR.md` |
| 逐元素（Tile-Tile） | `TSUB` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TSUB.md` |
| 逐元素（Tile-Tile） | `TMUL` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TMUL.md` |
| 逐元素（Tile-Tile） | `TMADD` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TMADD.md` |
| 逐元素（Tile-Tile） | `TMIN` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TMIN.md` |
| 逐元素（Tile-Tile） | `TMAX` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TMAX.md` |
| 逐元素（Tile-Tile） | `TCMP` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TCMP.md` |
| 逐元素（Tile-Tile） | `TDIV` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TDIV.md` |
| 逐元素（Tile-Tile） | `TSHL` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TSHL.md` |
| 逐元素（Tile-Tile） | `TSHR` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TSHR.md` |
| 逐元素（Tile-Tile） | `TXOR` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TXOR.md` |
| 逐元素（Tile-Tile） | `TLOG` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TLOG.md` |
| 逐元素（Tile-Tile） | `TRECIP` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TRECIP.md` |
| 逐元素（Tile-Tile） | `TPRELU` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TPRELU.md` |
| 逐元素（Tile-Tile） | `TCVT` | `elementwise` | `dst, src, tmp (overload)` | `docs/zh/isa/TCVT.md` |
| 逐元素（Tile-Tile） | `TSEL` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TSEL.md` |
| 逐元素（Tile-Tile） | `TRSQRT` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TRSQRT.md` |
| 逐元素（Tile-Tile） | `TSQRT` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TSQRT.md` |
| 逐元素（Tile-Tile） | `TEXP` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TEXP.md` |
| 逐元素（Tile-Tile） | `TPOW` | `elementwise` | `dst, base, exp, tmp` | `docs/zh/isa/TPOW.md` |
| 逐元素（Tile-Tile） | `TNOT` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TNOT.md` |
| 逐元素（Tile-Tile） | `TRELU` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TRELU.md` |
| 逐元素（Tile-Tile） | `TNEG` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TNEG.md` |
| 逐元素（Tile-Tile） | `TREM` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TREM.md` |
| 逐元素（Tile-Tile） | `TFMOD` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TFMOD.md` |
| 逐元素（Tile-Tile） | `TMULA` | `elementwise` | `dst, src0, src1` | `docs/zh/isa/TMULA.md` |
| Tile-标量 / Tile-立即数 | `TEXPANDS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TEXPANDS.md` |
| Tile-标量 / Tile-立即数 | `TCMPS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TCMPS.md` |
| Tile-标量 / Tile-立即数 | `TSELS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TSELS.md` |
| Tile-标量 / Tile-立即数 | `TMINS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TMINS.md` |
| Tile-标量 / Tile-立即数 | `TADDS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TADDS.md` |
| Tile-标量 / Tile-立即数 | `TSUBS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TSUBS.md` |
| Tile-标量 / Tile-立即数 | `TAXPY` | `scalar` | `dst, src0, scalar` | `docs/zh/isa/TAXPY.md` |
| Tile-标量 / Tile-立即数 | `TDIVS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TDIVS.md` |
| Tile-标量 / Tile-立即数 | `TMULS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TMULS.md` |
| Tile-标量 / Tile-立即数 | `TFMODS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TFMODS.md` |
| Tile-标量 / Tile-立即数 | `TREMS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TREMS.md` |
| Tile-标量 / Tile-立即数 | `TMAXS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TMAXS.md` |
| Tile-标量 / Tile-立即数 | `TANDS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TANDS.md` |
| Tile-标量 / Tile-立即数 | `TORS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TORS.md` |
| Tile-标量 / Tile-立即数 | `TSHLS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TSHLS.md` |
| Tile-标量 / Tile-立即数 | `TSHRS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TSHRS.md` |
| Tile-标量 / Tile-立即数 | `TXORS` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TXORS.md` |
| Tile-标量 / Tile-立即数 | `TLRELU` | `scalar` | `dst, src, scalar` | `docs/zh/isa/TLRELU.md` |
| Tile-标量 / Tile-立即数 | `TPOWS` | `tile_scalar` | `dst, base, exp, tmp` | `docs/zh/isa/TPOWS.md` |
| 轴归约 / 扩展 | `TROWSUM` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWSUM.md` |
| 轴归约 / 扩展 | `TROWPROD` | `reduce` | `dst, src, tmp` | `docs/zh/isa/TROWPROD.md` |
| 轴归约 / 扩展 | `TCOLSUM` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLSUM.md` |
| 轴归约 / 扩展 | `TCOLPROD` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLPROD.md` |
| 轴归约 / 扩展 | `TCOLMAX` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLMAX.md` |
| 轴归约 / 扩展 | `TROWMAX` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWMAX.md` |
| 轴归约 / 扩展 | `TROWMIN` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWMIN.md` |
| 轴归约 / 扩展 | `TROWARGMAX` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWARGMAX.md` |
| 轴归约 / 扩展 | `TROWARGMIN` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWARGMIN.md` |
| 轴归约 / 扩展 | `TCOLARGMAX` | `reduce` | `dst, src, tmp` | `docs/zh/isa/TCOLARGMAX.md` |
| 轴归约 / 扩展 | `TCOLARGMIN` | `reduce` | `dst, src, tmp` | `docs/zh/isa/TCOLARGMIN.md` |
| 轴归约 / 扩展 | `TROWEXPAND` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPAND.md` |
| 轴归约 / 扩展 | `TROWEXPANDDIV` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPANDDIV.md` |
| 轴归约 / 扩展 | `TROWEXPANDMUL` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPANDMUL.md` |
| 轴归约 / 扩展 | `TROWEXPANDSUB` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPANDSUB.md` |
| 轴归约 / 扩展 | `TROWEXPANDADD` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPANDADD.md` |
| 轴归约 / 扩展 | `TROWEXPANDMAX` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPANDMAX.md` |
| 轴归约 / 扩展 | `TROWEXPANDMIN` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPANDMIN.md` |
| 轴归约 / 扩展 | `TROWEXPANDEXPDIF` | `reduce_expand` | `dst, src` | `docs/zh/isa/TROWEXPANDEXPDIF.md` |
| 轴归约 / 扩展 | `TCOLMIN` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLMIN.md` |
| 轴归约 / 扩展 | `TCOLEXPAND` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPAND.md` |
| 轴归约 / 扩展 | `TCOLEXPANDDIV` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPANDDIV.md` |
| 轴归约 / 扩展 | `TCOLEXPANDMUL` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPANDMUL.md` |
| 轴归约 / 扩展 | `TCOLEXPANDADD` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPANDADD.md` |
| 轴归约 / 扩展 | `TCOLEXPANDMAX` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPANDMAX.md` |
| 轴归约 / 扩展 | `TCOLEXPANDMIN` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPANDMIN.md` |
| 轴归约 / 扩展 | `TCOLEXPANDSUB` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPANDSUB.md` |
| 轴归约 / 扩展 | `TCOLEXPANDEXPDIF` | `reduce_expand` | `dst, src` | `docs/zh/isa/TCOLEXPANDEXPDIF.md` |
| 内存（GM <-> Tile） | `TLOAD` | `memory` | `tile, global` | `docs/zh/isa/TLOAD.md` |
| 内存（GM <-> Tile） | `TPREFETCH` | `memory` | `tile, global` | `docs/zh/isa/TPREFETCH.md` |
| 内存（GM <-> Tile） | `TPREFETCH_ASYNC` | `memory` | `global, context` | `docs/zh/isa/TPREFETCH_ASYNC.md` |
| 内存（GM <-> Tile） | `TSTORE` | `memory` | `tile, global` | `docs/zh/isa/TSTORE.md` |
| 内存（GM <-> Tile） | `TSTORE_FP` | `memory` | `tile, fp, global` | `docs/zh/isa/TSTORE_FP.md` |
| 内存（GM <-> Tile） | `MGATHER` | `memory` | `tile, global` | `docs/zh/isa/MGATHER.md` |
| 内存（GM <-> Tile） | `MSCATTER` | `memory` | `tile, global` | `docs/zh/isa/MSCATTER.md` |
| 矩阵乘 | `TGEMV_MX` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TGEMV_MX.md` |
| 矩阵乘 | `TMATMUL_MX` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TMATMUL_MX.md` |
| 矩阵乘 | `TMATMUL_MX_HIF4` | `matmul` | `dst, left, right, scale` | `docs/zh/isa/TMATMUL_MX_HIF4.md` |
| 矩阵乘 | `TMATMUL` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TMATMUL.md` |
| 矩阵乘 | `TMATMUL_ACC` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TMATMUL_ACC.md` |
| 矩阵乘 | `TMATMUL_BIAS` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TMATMUL_BIAS.md` |
| 矩阵乘 | `TGEMV` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TGEMV.md` |
| 矩阵乘 | `TGEMV_ACC` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TGEMV_ACC.md` |
| 矩阵乘 | `TGEMV_BIAS` | `matmul` | `dst, lhs, rhs` | `docs/zh/isa/TGEMV_BIAS.md` |
| 数据搬运 / 布局 | `TEXTRACT` | `reshape_move` | `dst, src` | `docs/zh/isa/TEXTRACT.md` |
| 数据搬运 / 布局 | `TEXTRACT_FP` | `reshape_move` | `dst, src, fp` | `docs/zh/isa/TEXTRACT_FP.md` |
| 数据搬运 / 布局 | `TIMG2COL` | `reshape_move` | `dst, src` | `docs/zh/isa/TIMG2COL.md` |
| 数据搬运 / 布局 | `TINSERT` | `reshape_move` | `dst, src` | `docs/zh/isa/TINSERT.md` |
| 数据搬运 / 布局 | `TINSERT_FP` | `reshape_move` | `dst, src, fp` | `docs/zh/isa/TINSERT_FP.md` |
| 数据搬运 / 布局 | `TFILLPAD` | `reshape_move` | `dst, src` | `docs/zh/isa/TFILLPAD.md` |
| 数据搬运 / 布局 | `TFILLPAD_INPLACE` | `reshape_move` | `dst, src` | `docs/zh/isa/TFILLPAD_INPLACE.md` |
| 数据搬运 / 布局 | `TFILLPAD_EXPAND` | `reshape_move` | `dst, src` | `docs/zh/isa/TFILLPAD_EXPAND.md` |
| 数据搬运 / 布局 | `TMOV` | `reshape_move` | `dst, src` | `docs/zh/isa/TMOV.md` |
| 数据搬运 / 布局 | `TMOV_FP` | `reshape_move` | `dst, src, fp` | `docs/zh/isa/TMOV_FP.md` |
| 数据搬运 / 布局 | `TRESHAPE` | `reshape_move` | `dst, src` | `docs/zh/isa/TRESHAPE.md` |
| 数据搬运 / 布局 | `TTRANS` | `reshape_move` | `dst, src` | `docs/zh/isa/TTRANS.md` |
| 数据搬运 / 布局 | `TCONCAT` | `layout` | `dst, src0, src1` | `docs/zh/isa/TCONCAT.md` |
| 数据搬运 / 布局 | `TINTERLEAVE` | `layout` | `dst1, dst0, src1, src0` | `docs/zh/isa/TINTERLEAVE.md` |
| 数据搬运 / 布局 | `TDEINTERLEAVE` | `layout` | `dst1, dst0, src1, src0` | `docs/zh/isa/TDEINTERLEAVE.md` |
| 复杂指令 | `TPRINT` | `complex` | `dst, src0, src1` | `docs/zh/isa/TPRINT.md` |
| 复杂指令 | `TMRGSORT` | `complex` | `dst, src0, src1` | `docs/zh/isa/TMRGSORT.md` |
| 复杂指令 | `TSORT32` | `complex` | `dst, src0, src1` | `docs/zh/isa/TSORT32.md` |
| 复杂指令 | `TGATHER` | `complex` | `dst, src0, src1` | `docs/zh/isa/TGATHER.md` |
| 复杂指令 | `TCI` | `complex` | `dst, start, tmp (overload)` | `docs/zh/isa/TCI.md` |
| 复杂指令 | `TTRI` | `complex` | `dst, src0, src1` | `docs/zh/isa/TTRI.md` |
| 复杂指令 | `TRANDOM` | `complex` | `dst, key, counter` | `docs/zh/isa/TRANDOM.md` |
| 复杂指令 | `TPARTADD` | `complex` | `dst, src0, src1` | `docs/zh/isa/TPARTADD.md` |
| 复杂指令 | `TPARTMUL` | `complex` | `dst, src0, src1` | `docs/zh/isa/TPARTMUL.md` |
| 复杂指令 | `TPARTMAX` | `complex` | `dst, src0, src1` | `docs/zh/isa/TPARTMAX.md` |
| 复杂指令 | `TPARTMIN` | `complex` | `dst, src0, src1` | `docs/zh/isa/TPARTMIN.md` |
| 复杂指令 | `TPARTARGMAX` | `complex` | `dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx` | `docs/zh/isa/TPARTARGMAX.md` |
| 复杂指令 | `TPARTARGMIN` | `complex` | `dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx` | `docs/zh/isa/TPARTARGMIN.md` |
| 复杂指令 | `TGATHERB` | `complex` | `dst, src0, src1` | `docs/zh/isa/TGATHERB.md` |
| 复杂指令 | `TSCATTER` | `complex` | `dst, src0, src1` | `docs/zh/isa/TSCATTER.md` |
| 复杂指令 | `TQUANT` | `complex` | `dst, src0, src1` | `docs/zh/isa/TQUANT.md` |
| 复杂指令 | `TQUANT_DN` | `complex` | `dst, src, scale` | `docs/zh/isa/TQUANT_DN.md` |
| 复杂指令 | `TQUANT_HIF4` | `complex` | `dst, src, scale` | `docs/zh/isa/TQUANT_HIF4.md` |
| 复杂指令 | `TDEQUANT` | `elementwise` | `dst, src, scale, offset` | `docs/zh/isa/TDEQUANT.md` |
| 复杂指令 | `THISTOGRAM` | `complex` | `dst, src, idx` | `docs/zh/isa/THISTOGRAM.md` |
| Cross-core Communication | `TALLOC` | `comm` | `pipe, gmTensor` | `docs/zh/isa/TALLOC.md` |
| Cross-core Communication | `TPUSH` | `comm` | `pipe, tile` | `docs/zh/isa/TPUSH.md` |
| Cross-core Communication | `TPOP` | `comm` | `pipe, tile` | `docs/zh/isa/TPOP.md` |
| Cross-core Communication | `TFREE` | `comm` | `pipe` | `docs/zh/isa/TFREE.md` |

## D.5 说明

- 逐条指令语义仍以 `docs/zh/isa/*.md` 为准。
- 本附录用于分类与覆盖追踪，不替代逐条指令的规范化语义描述。
