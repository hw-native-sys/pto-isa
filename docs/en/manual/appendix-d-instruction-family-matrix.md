# Appendix D. Instruction Family Matrix

## D.1 Scope

This appendix is generated from `docs/isa/manifest.yaml` and provides a source-synchronized matrix of PTO virtual instruction families.

## D.2 Coverage summary

| Category | Instruction Count |
|---|---:|
| Synchronization | 1 |
| Manual / Resource Binding | 6 |
| Elementwise (Tile-Tile) | 29 |
| Tile-Scalar / Tile-Immediate | 19 |
| Axis Reduce / Expand | 28 |
| Memory (GM <-> Tile) | 7 |
| Matrix Multiply | 9 |
| Data Movement / Layout | 15 |
| Complex | 20 |
| Cross-core Communication | 4 |
| Total | 138 |

## D.3 Header synchronization status

- Header inventory source: `include/pto/common/pto_instr.hpp` (134 unique instruction APIs)
- Manifest inventory source: `docs/isa/manifest.yaml` (138 entries)
- Missing in manifest: none
- Present in manifest but missing in header: TMATMUL_MX_HIF4, TPREFETCH_ASYNC, TQUANT_DN, TQUANT_HIF4

## D.4 Family matrix

| Category | Instruction | Diagram Template | Operand Contract | Semantic Page |
|---|---|---|---|---|
| Synchronization | `SYNCALL` | `sync` | `participants` | `docs/en/isa/SYNCALL.md` |
| Manual / Resource Binding | `TASSIGN` | `config` | `config, state` | `docs/en/isa/TASSIGN.md` |
| Manual / Resource Binding | `SETFMATRIX` | `config` | `config, state` | `docs/en/isa/SETFMATRIX.md` |
| Manual / Resource Binding | `SET_IMG2COL_RPT` | `config` | `config, state` | `docs/en/isa/SET_IMG2COL_RPT.md` |
| Manual / Resource Binding | `SET_IMG2COL_PADDING` | `config` | `config, state` | `docs/en/isa/SET_IMG2COL_PADDING.md` |
| Manual / Resource Binding | `SET_QUANT_SCALAR` | `config` | `scale` | `docs/en/isa/SET_QUANT_SCALAR.md` |
| Manual / Resource Binding | `SET_QUANT_VECTOR` | `config` | `fpTile` | `docs/en/isa/SET_QUANT_VECTOR.md` |
| Elementwise (Tile-Tile) | `TADD` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TADD.md` |
| Elementwise (Tile-Tile) | `TABS` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TABS.md` |
| Elementwise (Tile-Tile) | `TAND` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TAND.md` |
| Elementwise (Tile-Tile) | `TOR` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TOR.md` |
| Elementwise (Tile-Tile) | `TSUB` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TSUB.md` |
| Elementwise (Tile-Tile) | `TMUL` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TMUL.md` |
| Elementwise (Tile-Tile) | `TMADD` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TMADD.md` |
| Elementwise (Tile-Tile) | `TMIN` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TMIN.md` |
| Elementwise (Tile-Tile) | `TMAX` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TMAX.md` |
| Elementwise (Tile-Tile) | `TCMP` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TCMP.md` |
| Elementwise (Tile-Tile) | `TDIV` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TDIV.md` |
| Elementwise (Tile-Tile) | `TSHL` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TSHL.md` |
| Elementwise (Tile-Tile) | `TSHR` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TSHR.md` |
| Elementwise (Tile-Tile) | `TXOR` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TXOR.md` |
| Elementwise (Tile-Tile) | `TLOG` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TLOG.md` |
| Elementwise (Tile-Tile) | `TRECIP` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TRECIP.md` |
| Elementwise (Tile-Tile) | `TPRELU` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TPRELU.md` |
| Elementwise (Tile-Tile) | `TCVT` | `elementwise` | `dst, src, tmp (overload)` | `docs/en/isa/TCVT.md` |
| Elementwise (Tile-Tile) | `TSEL` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TSEL.md` |
| Elementwise (Tile-Tile) | `TRSQRT` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TRSQRT.md` |
| Elementwise (Tile-Tile) | `TSQRT` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TSQRT.md` |
| Elementwise (Tile-Tile) | `TEXP` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TEXP.md` |
| Elementwise (Tile-Tile) | `TPOW` | `elementwise` | `dst, base, exp, tmp` | `docs/en/isa/TPOW.md` |
| Elementwise (Tile-Tile) | `TNOT` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TNOT.md` |
| Elementwise (Tile-Tile) | `TRELU` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TRELU.md` |
| Elementwise (Tile-Tile) | `TNEG` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TNEG.md` |
| Elementwise (Tile-Tile) | `TREM` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TREM.md` |
| Elementwise (Tile-Tile) | `TFMOD` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TFMOD.md` |
| Elementwise (Tile-Tile) | `TMULA` | `elementwise` | `dst, src0, src1` | `docs/en/isa/TMULA.md` |
| Tile-Scalar / Tile-Immediate | `TEXPANDS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TEXPANDS.md` |
| Tile-Scalar / Tile-Immediate | `TCMPS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TCMPS.md` |
| Tile-Scalar / Tile-Immediate | `TSELS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TSELS.md` |
| Tile-Scalar / Tile-Immediate | `TMINS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TMINS.md` |
| Tile-Scalar / Tile-Immediate | `TADDS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TADDS.md` |
| Tile-Scalar / Tile-Immediate | `TSUBS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TSUBS.md` |
| Tile-Scalar / Tile-Immediate | `TAXPY` | `scalar` | `dst, src0, scalar` | `docs/en/isa/TAXPY.md` |
| Tile-Scalar / Tile-Immediate | `TDIVS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TDIVS.md` |
| Tile-Scalar / Tile-Immediate | `TMULS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TMULS.md` |
| Tile-Scalar / Tile-Immediate | `TFMODS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TFMODS.md` |
| Tile-Scalar / Tile-Immediate | `TREMS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TREMS.md` |
| Tile-Scalar / Tile-Immediate | `TMAXS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TMAXS.md` |
| Tile-Scalar / Tile-Immediate | `TANDS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TANDS.md` |
| Tile-Scalar / Tile-Immediate | `TORS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TORS.md` |
| Tile-Scalar / Tile-Immediate | `TSHLS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TSHLS.md` |
| Tile-Scalar / Tile-Immediate | `TSHRS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TSHRS.md` |
| Tile-Scalar / Tile-Immediate | `TXORS` | `scalar` | `dst, src, scalar` | `docs/en/isa/TXORS.md` |
| Tile-Scalar / Tile-Immediate | `TLRELU` | `scalar` | `dst, src, scalar` | `docs/en/isa/TLRELU.md` |
| Tile-Scalar / Tile-Immediate | `TPOWS` | `tile_scalar` | `dst, base, exp, tmp` | `docs/en/isa/TPOWS.md` |
| Axis Reduce / Expand | `TROWSUM` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWSUM.md` |
| Axis Reduce / Expand | `TROWPROD` | `reduce` | `dst, src, tmp` | `docs/en/isa/TROWPROD.md` |
| Axis Reduce / Expand | `TCOLSUM` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLSUM.md` |
| Axis Reduce / Expand | `TCOLPROD` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLPROD.md` |
| Axis Reduce / Expand | `TCOLMAX` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLMAX.md` |
| Axis Reduce / Expand | `TROWMAX` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWMAX.md` |
| Axis Reduce / Expand | `TROWMIN` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWMIN.md` |
| Axis Reduce / Expand | `TROWARGMAX` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWARGMAX.md` |
| Axis Reduce / Expand | `TROWARGMIN` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWARGMIN.md` |
| Axis Reduce / Expand | `TCOLARGMAX` | `reduce` | `dst, src, tmp` | `docs/en/isa/TCOLARGMAX.md` |
| Axis Reduce / Expand | `TCOLARGMIN` | `reduce` | `dst, src, tmp` | `docs/en/isa/TCOLARGMIN.md` |
| Axis Reduce / Expand | `TROWEXPAND` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPAND.md` |
| Axis Reduce / Expand | `TROWEXPANDDIV` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPANDDIV.md` |
| Axis Reduce / Expand | `TROWEXPANDMUL` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPANDMUL.md` |
| Axis Reduce / Expand | `TROWEXPANDSUB` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPANDSUB.md` |
| Axis Reduce / Expand | `TROWEXPANDADD` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPANDADD.md` |
| Axis Reduce / Expand | `TROWEXPANDMAX` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPANDMAX.md` |
| Axis Reduce / Expand | `TROWEXPANDMIN` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPANDMIN.md` |
| Axis Reduce / Expand | `TROWEXPANDEXPDIF` | `reduce_expand` | `dst, src` | `docs/en/isa/TROWEXPANDEXPDIF.md` |
| Axis Reduce / Expand | `TCOLMIN` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLMIN.md` |
| Axis Reduce / Expand | `TCOLEXPAND` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPAND.md` |
| Axis Reduce / Expand | `TCOLEXPANDDIV` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPANDDIV.md` |
| Axis Reduce / Expand | `TCOLEXPANDMUL` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPANDMUL.md` |
| Axis Reduce / Expand | `TCOLEXPANDADD` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPANDADD.md` |
| Axis Reduce / Expand | `TCOLEXPANDMAX` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPANDMAX.md` |
| Axis Reduce / Expand | `TCOLEXPANDMIN` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPANDMIN.md` |
| Axis Reduce / Expand | `TCOLEXPANDSUB` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPANDSUB.md` |
| Axis Reduce / Expand | `TCOLEXPANDEXPDIF` | `reduce_expand` | `dst, src` | `docs/en/isa/TCOLEXPANDEXPDIF.md` |
| Memory (GM <-> Tile) | `TLOAD` | `memory` | `tile, global` | `docs/en/isa/TLOAD.md` |
| Memory (GM <-> Tile) | `TPREFETCH` | `memory` | `tile, global` | `docs/en/isa/TPREFETCH.md` |
| Memory (GM <-> Tile) | `TPREFETCH_ASYNC` | `memory` | `global, context` | `docs/en/isa/TPREFETCH_ASYNC.md` |
| Memory (GM <-> Tile) | `TSTORE` | `memory` | `tile, global` | `docs/en/isa/TSTORE.md` |
| Memory (GM <-> Tile) | `TSTORE_FP` | `memory` | `tile, fp, global` | `docs/en/isa/TSTORE_FP.md` |
| Memory (GM <-> Tile) | `MGATHER` | `memory` | `tile, global` | `docs/en/isa/MGATHER.md` |
| Memory (GM <-> Tile) | `MSCATTER` | `memory` | `tile, global` | `docs/en/isa/MSCATTER.md` |
| Matrix Multiply | `TGEMV_MX` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TGEMV_MX.md` |
| Matrix Multiply | `TMATMUL_MX` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TMATMUL_MX.md` |
| Matrix Multiply | `TMATMUL_MX_HIF4` | `matmul` | `dst, left, right, scale` | `docs/en/isa/TMATMUL_MX_HIF4.md` |
| Matrix Multiply | `TMATMUL` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TMATMUL.md` |
| Matrix Multiply | `TMATMUL_ACC` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TMATMUL_ACC.md` |
| Matrix Multiply | `TMATMUL_BIAS` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TMATMUL_BIAS.md` |
| Matrix Multiply | `TGEMV` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TGEMV.md` |
| Matrix Multiply | `TGEMV_ACC` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TGEMV_ACC.md` |
| Matrix Multiply | `TGEMV_BIAS` | `matmul` | `dst, lhs, rhs` | `docs/en/isa/TGEMV_BIAS.md` |
| Data Movement / Layout | `TEXTRACT` | `reshape_move` | `dst, src` | `docs/en/isa/TEXTRACT.md` |
| Data Movement / Layout | `TEXTRACT_FP` | `reshape_move` | `dst, src, fp` | `docs/en/isa/TEXTRACT_FP.md` |
| Data Movement / Layout | `TIMG2COL` | `reshape_move` | `dst, src` | `docs/en/isa/TIMG2COL.md` |
| Data Movement / Layout | `TINSERT` | `reshape_move` | `dst, src` | `docs/en/isa/TINSERT.md` |
| Data Movement / Layout | `TINSERT_FP` | `reshape_move` | `dst, src, fp` | `docs/en/isa/TINSERT_FP.md` |
| Data Movement / Layout | `TFILLPAD` | `reshape_move` | `dst, src` | `docs/en/isa/TFILLPAD.md` |
| Data Movement / Layout | `TFILLPAD_INPLACE` | `reshape_move` | `dst, src` | `docs/en/isa/TFILLPAD_INPLACE.md` |
| Data Movement / Layout | `TFILLPAD_EXPAND` | `reshape_move` | `dst, src` | `docs/en/isa/TFILLPAD_EXPAND.md` |
| Data Movement / Layout | `TMOV` | `reshape_move` | `dst, src` | `docs/en/isa/TMOV.md` |
| Data Movement / Layout | `TMOV_FP` | `reshape_move` | `dst, src, fp` | `docs/en/isa/TMOV_FP.md` |
| Data Movement / Layout | `TRESHAPE` | `reshape_move` | `dst, src` | `docs/en/isa/TRESHAPE.md` |
| Data Movement / Layout | `TTRANS` | `reshape_move` | `dst, src` | `docs/en/isa/TTRANS.md` |
| Data Movement / Layout | `TCONCAT` | `layout` | `dst, src0, src1` | `docs/en/isa/TCONCAT.md` |
| Data Movement / Layout | `TINTERLEAVE` | `layout` | `dst1, dst0, src1, src0` | `docs/en/isa/TINTERLEAVE.md` |
| Data Movement / Layout | `TDEINTERLEAVE` | `layout` | `dst1, dst0, src1, src0` | `docs/en/isa/TDEINTERLEAVE.md` |
| Complex | `TPRINT` | `complex` | `dst, src0, src1` | `docs/en/isa/TPRINT.md` |
| Complex | `TMRGSORT` | `complex` | `dst, src0, src1` | `docs/en/isa/TMRGSORT.md` |
| Complex | `TSORT32` | `complex` | `dst, src0, src1` | `docs/en/isa/TSORT32.md` |
| Complex | `TGATHER` | `complex` | `dst, src0, src1` | `docs/en/isa/TGATHER.md` |
| Complex | `TCI` | `complex` | `dst, start, tmp (overload)` | `docs/en/isa/TCI.md` |
| Complex | `TTRI` | `complex` | `dst, src0, src1` | `docs/en/isa/TTRI.md` |
| Complex | `TRANDOM` | `complex` | `dst, key, counter` | `docs/en/isa/TRANDOM.md` |
| Complex | `TPARTADD` | `complex` | `dst, src0, src1` | `docs/en/isa/TPARTADD.md` |
| Complex | `TPARTMUL` | `complex` | `dst, src0, src1` | `docs/en/isa/TPARTMUL.md` |
| Complex | `TPARTMAX` | `complex` | `dst, src0, src1` | `docs/en/isa/TPARTMAX.md` |
| Complex | `TPARTMIN` | `complex` | `dst, src0, src1` | `docs/en/isa/TPARTMIN.md` |
| Complex | `TPARTARGMAX` | `complex` | `dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx` | `docs/en/isa/TPARTARGMAX.md` |
| Complex | `TPARTARGMIN` | `complex` | `dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx` | `docs/en/isa/TPARTARGMIN.md` |
| Complex | `TGATHERB` | `complex` | `dst, src0, src1` | `docs/en/isa/TGATHERB.md` |
| Complex | `TSCATTER` | `complex` | `dst, src0, src1` | `docs/en/isa/TSCATTER.md` |
| Complex | `TQUANT` | `complex` | `dst, src0, src1` | `docs/en/isa/TQUANT.md` |
| Complex | `TQUANT_DN` | `complex` | `dst, src, scale` | `docs/en/isa/TQUANT_DN.md` |
| Complex | `TQUANT_HIF4` | `complex` | `dst, src, scale` | `docs/en/isa/TQUANT_HIF4.md` |
| Complex | `TDEQUANT` | `elementwise` | `dst, src, scale, offset` | `docs/en/isa/TDEQUANT.md` |
| Complex | `THISTOGRAM` | `complex` | `dst, src, idx` | `docs/en/isa/THISTOGRAM.md` |
| Cross-core Communication | `TALLOC` | `comm` | `pipe, gmTensor` | `docs/en/isa/TALLOC.md` |
| Cross-core Communication | `TPUSH` | `comm` | `pipe, tile` | `docs/en/isa/TPUSH.md` |
| Cross-core Communication | `TPOP` | `comm` | `pipe, tile` | `docs/en/isa/TPOP.md` |
| Cross-core Communication | `TFREE` | `comm` | `pipe` | `docs/en/isa/TFREE.md` |

## D.5 Notes

- Per-instruction semantics remain canonical in `docs/en/isa/*.md`.
- This appendix is a taxonomy and coverage matrix, not a replacement for per-op normative semantics.
