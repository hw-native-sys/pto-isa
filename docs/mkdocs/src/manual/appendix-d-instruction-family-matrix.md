# Appendix D. Instruction Family Matrix

## D.1 Scope

This appendix is generated from `docs/isa/manifest.yaml` and provides a source-synchronized matrix of PTO virtual instruction families.

## D.2 Coverage summary

| Category | Instruction Count |
|---|---:|
| Synchronization | 1 |
| Manual / Resource Binding | 4 |
| Elementwise (Tile-Tile) | 28 |
| Tile-Scalar / Tile-Immediate | 19 |
| Axis Reduce / Expand | 24 |
| Memory (GM <-> Tile) | 6 |
| Matrix Multiply | 8 |
| Data Movement / Layout | 12 |
| Complex | 13 |
| Communication | 11 |
| Total | 126 |

## D.3 Header synchronization status

- Header inventory source: `include/pto/common/pto_instr.hpp` (115 unique instruction APIs)
- Manifest inventory source: `docs/isa/manifest.yaml` (115 entries)
- Missing in manifest: none
- Present in manifest but missing in header: none

## D.4 Family matrix

| Category | Instruction | Diagram Template | Operand Contract | Semantic Page |
|---|---|---|---|---|
| Synchronization | [TSYNC](/docs/isa/TSYNC.md) | `sync` | `producer, consumer` | `docs/isa/TSYNC.md` |
| Manual / Resource Binding | [TASSIGN](/docs/isa/TASSIGN.md) | `config` | `config, state` | `docs/isa/TASSIGN.md` |
| Manual / Resource Binding | [TSETFMATRIX](/docs/isa/TSETFMATRIX.md) | `config` | `config, state` | `docs/isa/TSETFMATRIX.md` |
| Elementwise (Tile-Tile) | [TADD](/docs/isa/TADD.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TADD.md` |
| Elementwise (Tile-Tile) | [TABS](/docs/isa/TABS.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TABS.md` |
| Elementwise (Tile-Tile) | [TAND](/docs/isa/TAND.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TAND.md` |
| Elementwise (Tile-Tile) | [TOR](/docs/isa/TOR.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TOR.md` |
| Elementwise (Tile-Tile) | [TSUB](/docs/isa/TSUB.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSUB.md` |
| Elementwise (Tile-Tile) | [TMUL](/docs/isa/TMUL.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMUL.md` |
| Elementwise (Tile-Tile) | [TMIN](/docs/isa/TMIN.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMIN.md` |
| Elementwise (Tile-Tile) | [TMAX](/docs/isa/TMAX.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMAX.md` |
| Elementwise (Tile-Tile) | [TCMP](/docs/isa/TCMP.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TCMP.md` |
| Elementwise (Tile-Tile) | [TDIV](/docs/isa/TDIV.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TDIV.md` |
| Elementwise (Tile-Tile) | [TSHL](/docs/isa/TSHL.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSHL.md` |
| Elementwise (Tile-Tile) | [TSHR](/docs/isa/TSHR.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSHR.md` |
| Elementwise (Tile-Tile) | [TXOR](/docs/isa/TXOR.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TXOR.md` |
| Elementwise (Tile-Tile) | [TLOG](/docs/isa/TLOG.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TLOG.md` |
| Elementwise (Tile-Tile) | [TRECIP](/docs/isa/TRECIP.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRECIP.md` |
| Elementwise (Tile-Tile) | [TPRELU](/docs/isa/TPRELU.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TPRELU.md` |
| Elementwise (Tile-Tile) | [TADDC](/docs/isa/TADDC.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TADDC.md` |
| Elementwise (Tile-Tile) | [TSUBC](/docs/isa/TSUBC.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSUBC.md` |
| Elementwise (Tile-Tile) | [TCVT](/docs/isa/TCVT.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TCVT.md` |
| Elementwise (Tile-Tile) | [TSEL](/docs/isa/TSEL.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSEL.md` |
| Elementwise (Tile-Tile) | [TRSQRT](/docs/isa/TRSQRT.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRSQRT.md` |
| Elementwise (Tile-Tile) | [TSQRT](/docs/isa/TSQRT.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSQRT.md` |
| Elementwise (Tile-Tile) | [TEXP](/docs/isa/TEXP.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TEXP.md` |
| Elementwise (Tile-Tile) | [TNOT](/docs/isa/TNOT.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TNOT.md` |
| Elementwise (Tile-Tile) | [TRELU](/docs/isa/TRELU.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRELU.md` |
| Elementwise (Tile-Tile) | [TNEG](/docs/isa/TNEG.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TNEG.md` |
| Elementwise (Tile-Tile) | [TREM](/docs/isa/TREM.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TREM.md` |
| Elementwise (Tile-Tile) | [TFMOD](/docs/isa/TFMOD.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TFMOD.md` |
| Tile-Scalar / Tile-Immediate | [TEXPANDS](/docs/isa/TEXPANDS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TEXPANDS.md` |
| Tile-Scalar / Tile-Immediate | [TCMPS](/docs/isa/TCMPS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TCMPS.md` |
| Tile-Scalar / Tile-Immediate | [TSELS](/docs/isa/TSELS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSELS.md` |
| Tile-Scalar / Tile-Immediate | [TMINS](/docs/isa/TMINS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMINS.md` |
| Tile-Scalar / Tile-Immediate | [TADDS](/docs/isa/TADDS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TADDS.md` |
| Tile-Scalar / Tile-Immediate | [TSUBS](/docs/isa/TSUBS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSUBS.md` |
| Tile-Scalar / Tile-Immediate | [TDIVS](/docs/isa/TDIVS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TDIVS.md` |
| Tile-Scalar / Tile-Immediate | [TMULS](/docs/isa/TMULS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMULS.md` |
| Tile-Scalar / Tile-Immediate | [TFMODS](/docs/isa/TFMODS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TFMODS.md` |
| Tile-Scalar / Tile-Immediate | [TREMS](/docs/isa/TREMS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TREMS.md` |
| Tile-Scalar / Tile-Immediate | [TMAXS](/docs/isa/TMAXS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMAXS.md` |
| Tile-Scalar / Tile-Immediate | [TANDS](/docs/isa/TANDS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TANDS.md` |
| Tile-Scalar / Tile-Immediate | [TORS](/docs/isa/TORS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TORS.md` |
| Tile-Scalar / Tile-Immediate | [TSHLS](/docs/isa/TSHLS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSHLS.md` |
| Tile-Scalar / Tile-Immediate | [TSHRS](/docs/isa/TSHRS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSHRS.md` |
| Tile-Scalar / Tile-Immediate | [TXORS](/docs/isa/TXORS.md) | `scalar` | `dst, src, scalar` | `docs/isa/TXORS.md` |
| Tile-Scalar / Tile-Immediate | [TLRELU](/docs/isa/TLRELU.md) | `scalar` | `dst, src, scalar` | `docs/isa/TLRELU.md` |
| Tile-Scalar / Tile-Immediate | [TADDSC](/docs/isa/TADDSC.md) | `scalar` | `dst, src, scalar` | `docs/isa/TADDSC.md` |
| Tile-Scalar / Tile-Immediate | [TSUBSC](/docs/isa/TSUBSC.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSUBSC.md` |
| Axis Reduce / Expand | [TROWSUM](/docs/isa/TROWSUM.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWSUM.md` |
| Axis Reduce / Expand | [TROWPROD](/docs/isa/TROWPROD.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWPROD.md` |
| Axis Reduce / Expand | [TCOLSUM](/docs/isa/TCOLSUM.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLSUM.md` |
| Axis Reduce / Expand | [TCOLPROD](/docs/isa/TCOLPROD.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLPROD.md` |
| Axis Reduce / Expand | [TCOLMAX](/docs/isa/TCOLMAX.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLMAX.md` |
| Axis Reduce / Expand | [TROWMAX](/docs/isa/TROWMAX.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWMAX.md` |
| Axis Reduce / Expand | [TROWMIN](/docs/isa/TROWMIN.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWMIN.md` |
| Axis Reduce / Expand | [TROWEXPAND](/docs/isa/TROWEXPAND.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPAND.md` |
| Axis Reduce / Expand | [TROWEXPANDDIV](/docs/isa/TROWEXPANDDIV.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDDIV.md` |
| Axis Reduce / Expand | [TROWEXPANDMUL](/docs/isa/TROWEXPANDMUL.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMUL.md` |
| Axis Reduce / Expand | [TROWEXPANDSUB](/docs/isa/TROWEXPANDSUB.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDSUB.md` |
| Axis Reduce / Expand | [TROWEXPANDADD](/docs/isa/TROWEXPANDADD.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDADD.md` |
| Axis Reduce / Expand | [TROWEXPANDMAX](/docs/isa/TROWEXPANDMAX.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMAX.md` |
| Axis Reduce / Expand | [TROWEXPANDMIN](/docs/isa/TROWEXPANDMIN.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMIN.md` |
| Axis Reduce / Expand | [TROWEXPANDEXPDIF](/docs/isa/TROWEXPANDEXPDIF.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDEXPDIF.md` |
| Axis Reduce / Expand | [TCOLMIN](/docs/isa/TCOLMIN.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLMIN.md` |
| Axis Reduce / Expand | [TCOLEXPAND](/docs/isa/TCOLEXPAND.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPAND.md` |
| Axis Reduce / Expand | [TCOLEXPANDDIV](/docs/isa/TCOLEXPANDDIV.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDDIV.md` |
| Axis Reduce / Expand | [TCOLEXPANDMUL](/docs/isa/TCOLEXPANDMUL.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMUL.md` |
| Axis Reduce / Expand | [TCOLEXPANDADD](/docs/isa/TCOLEXPANDADD.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDADD.md` |
| Axis Reduce / Expand | [TCOLEXPANDMAX](/docs/isa/TCOLEXPANDMAX.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMAX.md` |
| Axis Reduce / Expand | [TCOLEXPANDMIN](/docs/isa/TCOLEXPANDMIN.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMIN.md` |
| Axis Reduce / Expand | [TCOLEXPANDSUB](/docs/isa/TCOLEXPANDSUB.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDSUB.md` |
| Axis Reduce / Expand | [TCOLEXPANDEXPDIF](/docs/isa/TCOLEXPANDEXPDIF.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDEXPDIF.md` |
| Memory (GM <-> Tile) | [TLOAD](/docs/isa/TLOAD.md) | `memory` | `tile, global` | `docs/isa/TLOAD.md` |
| Memory (GM <-> Tile) | [TPREFETCH](/docs/isa/TPREFETCH.md) | `memory` | `tile, global` | `docs/isa/TPREFETCH.md` |
| Memory (GM <-> Tile) | [TSTORE](/docs/isa/TSTORE.md) | `memory` | `tile, global` | `docs/isa/TSTORE.md` |
| Memory (GM <-> Tile) | [TSTORE_FP](/docs/isa/TSTORE_FP.md) | `memory` | `tile, global` | `docs/isa/TSTORE_FP.md` |
| Memory (GM <-> Tile) | [MGATHER](/docs/isa/MGATHER.md) | `memory` | `tile, global` | `docs/isa/MGATHER.md` |
| Memory (GM <-> Tile) | [MSCATTER](/docs/isa/MSCATTER.md) | `memory` | `tile, global` | `docs/isa/MSCATTER.md` |
| Matrix Multiply | [TGEMV_MX](/docs/isa/TGEMV_MX.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_MX.md` |
| Matrix Multiply | [TMATMUL_MX](/docs/isa/TMATMUL_MX.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_MX.md` |
| Matrix Multiply | [TMATMUL](/docs/isa/TMATMUL.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL.md` |
| Matrix Multiply | [TMATMUL_ACC](/docs/isa/TMATMUL_ACC.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_ACC.md` |
| Matrix Multiply | [TMATMUL_BIAS](/docs/isa/TMATMUL_BIAS.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_BIAS.md` |
| Matrix Multiply | [TGEMV](/docs/isa/TGEMV.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV.md` |
| Matrix Multiply | [TGEMV_ACC](/docs/isa/TGEMV_ACC.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_ACC.md` |
| Matrix Multiply | [TGEMV_BIAS](/docs/isa/TGEMV_BIAS.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_BIAS.md` |
| Data Movement / Layout | [TEXTRACT](/docs/isa/TEXTRACT.md) | `reshape_move` | `dst, src` | `docs/isa/TEXTRACT.md` |
| Data Movement / Layout | [TEXTRACT_FP](/docs/isa/TEXTRACT_FP.md) | `reshape_move` | `dst, src` | `docs/isa/TEXTRACT_FP.md` |
| Data Movement / Layout | [TIMG2COL](/docs/isa/TIMG2COL.md) | `reshape_move` | `dst, src` | `docs/isa/TIMG2COL.md` |
| Data Movement / Layout | [TINSERT](/docs/isa/TINSERT.md) | `reshape_move` | `dst, src` | `docs/isa/TINSERT.md` |
| Data Movement / Layout | [TINSERT_FP](/docs/isa/TINSERT_FP.md) | `reshape_move` | `dst, src` | `docs/isa/TINSERT_FP.md` |
| Data Movement / Layout | [TFILLPAD](/docs/isa/TFILLPAD.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD.md` |
| Data Movement / Layout | [TFILLPAD_INPLACE](/docs/isa/TFILLPAD_INPLACE.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD_INPLACE.md` |
| Data Movement / Layout | [TFILLPAD_EXPAND](/docs/isa/TFILLPAD_EXPAND.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD_EXPAND.md` |
| Data Movement / Layout | [TMOV](/docs/isa/TMOV.md) | `reshape_move` | `dst, src` | `docs/isa/TMOV.md` |
| Data Movement / Layout | [TMOV_FP](/docs/isa/TMOV_FP.md) | `reshape_move` | `dst, src` | `docs/isa/TMOV_FP.md` |
| Data Movement / Layout | [TRESHAPE](/docs/isa/TRESHAPE.md) | `reshape_move` | `dst, src` | `docs/isa/TRESHAPE.md` |
| Data Movement / Layout | [TTRANS](/docs/isa/TTRANS.md) | `reshape_move` | `dst, src` | `docs/isa/TTRANS.md` |
| Complex | [TPRINT](/docs/isa/TPRINT.md) | `complex` | `dst, src0, src1` | `docs/isa/TPRINT.md` |
| Complex | [TMRGSORT](/docs/isa/TMRGSORT.md) | `complex` | `dst, src0, src1` | `docs/isa/TMRGSORT.md` |
| Complex | [TSORT32](/docs/isa/TSORT32.md) | `complex` | `dst, src0, src1` | `docs/isa/TSORT32.md` |
| Complex | [TGATHER](/docs/isa/TGATHER.md) | `complex` | `dst, src0, src1` | `docs/isa/TGATHER.md` |
| Complex | [TCI](/docs/isa/TCI.md) | `complex` | `dst, src0, src1` | `docs/isa/TCI.md` |
| Complex | [TTRI](/docs/isa/TTRI.md) | `complex` | `dst, src0, src1` | `docs/isa/TTRI.md` |
| Complex | [TPARTADD](/docs/isa/TPARTADD.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTADD.md` |
| Complex | [TPARTMUL](/docs/isa/TPARTMUL.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMUL.md` |
| Complex | [TPARTMAX](/docs/isa/TPARTMAX.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMAX.md` |
| Complex | [TPARTMIN](/docs/isa/TPARTMIN.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMIN.md` |
| Complex | [TGATHERB](/docs/isa/TGATHERB.md) | `complex` | `dst, src0, src1` | `docs/isa/TGATHERB.md` |
| Complex | [TSCATTER](/docs/isa/TSCATTER.md) | `complex` | `dst, src0, src1` | `docs/isa/TSCATTER.md` |
| Complex | [TQUANT](/docs/isa/TQUANT.md) | `complex` | `dst, src0, src1` | `docs/isa/TQUANT.md` |
| Communication | [TPUT](/docs/isa/comm/TPUT.md) | `comm` | `dst, src, staging` | `docs/isa/comm/TPUT.md` |
| Communication | [TGET](/docs/isa/comm/TGET.md) | `comm` | `dst, src, staging` | `docs/isa/comm/TGET.md` |
| Communication | [TPUT_ASYNC](/docs/isa/comm/TPUT_ASYNC.md) | `comm` | `dst, src, session` | `docs/isa/comm/TPUT_ASYNC.md` |
| Communication | [TGET_ASYNC](/docs/isa/comm/TGET_ASYNC.md) | `comm` | `dst, src, session` | `docs/isa/comm/TGET_ASYNC.md` |
| Communication | [TNOTIFY](/docs/isa/comm/TNOTIFY.md) | `comm` | `signal, value, op` | `docs/isa/comm/TNOTIFY.md` |
| Communication | [TWAIT](/docs/isa/comm/TWAIT.md) | `comm` | `signal, value, cmp` | `docs/isa/comm/TWAIT.md` |
| Communication | [TTEST](/docs/isa/comm/TTEST.md) | `comm` | `signal, value, cmp` | `docs/isa/comm/TTEST.md` |
| Communication | [TGATHER](/docs/isa/comm/TGATHER.md) | `comm` | `group, dst, staging` | `docs/isa/comm/TGATHER.md` |
| Communication | [TSCATTER](/docs/isa/comm/TSCATTER.md) | `comm` | `group, src, staging` | `docs/isa/comm/TSCATTER.md` |
| Communication | [TREDUCE](/docs/isa/comm/TREDUCE.md) | `comm` | `group, dst, acc, recv` | `docs/isa/comm/TREDUCE.md` |
| Communication | [TBROADCAST](/docs/isa/comm/TBROADCAST.md) | `comm` | `group, src, staging` | `docs/isa/comm/TBROADCAST.md` |

## D.5 Notes

- Per-instruction semantics remain canonical in `docs/isa/*.md`.
- This appendix is a taxonomy and coverage matrix, not a replacement for per-op normative semantics.
