# PTO Virtual Instruction Set

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T05:36:15.526Z pushedAt=2026-08-29T09:05:18.489Z -->

- [Header Files and Library Files](menu/header_and_library.md)

  - [Header Files and Library Files](PTO-ISA-Header-and-Library-Description.md)

- [Element-Wise Binary Operations](menu/arithmetic.md)

  - [Arithmetic Operations](menu/arithmetic.md)

    - [TADD](isa/TADD.md)

    - [TSUB](isa/TSUB.md)

    - [TMUL](isa/TMUL.md)

    - [TMAX](isa/TMAX.md)

    - [TMIN](isa/TMIN.md)

  - [Logical Operations](menu/binary_logic.md)

    - [TAND](isa/TAND.md)

    - [TOR](isa/TOR.md)

    - [TXOR](isa/TXOR.md)

    - [TSHL](isa/TSHL.md)

    - [TSHR](isa/TSHR.md)

    - [TCMP](isa/TCMP.md)

    - [TSEL](isa/TSEL.md)

- [Element-Wise Unary Operations](menu/unary.md)

  - [TABS](isa/TABS.md)

  - [TNOT](isa/TNOT.md)

  - [TNEG](isa/TNEG.md)

  - [TRELU](isa/TRELU.md)

- [Element-Wise Arithmetic and Transcendental Functions](menu/transcendental.md)

  - [TDIV](isa/TDIV.md)

  - [TREM](isa/TREM.md)

  - [TSQRT](isa/TSQRT.md)

  - [TLOG](isa/TLOG.md)

  - [TRECIP](isa/TRECIP.md)

  - [TEXP](isa/TEXP.md)

  - [TRSQRT](isa/TRSQRT.md)

- [Element-Wise Scalar Operations](menu/scalar_arithmetic.md)

  - [Arithmetic Operations](menu/scalar_arithmetic.md)

    - [TADDS](isa/TADDS.md)

    - [TAXPY](isa/TAXPY.md)

    - [TSUBS](isa/TSUBS.md)

    - [TMULS](isa/TMULS.md)

    - [TDIVS](isa/TDIVS.md)

    - [TMINS](isa/TMINS.md)

    - [TMAXS](isa/TMAXS.md)

    - [TREMS](isa/TREMS.md)

  - [Logical Operations](menu/scalar_logic.md)

    - [TANDS](isa/TANDS.md)

    - [TORS](isa/TORS.md)

    - [TXORS](isa/TXORS.md)

    - [TCMPS](isa/TCMPS.md)

    - [TSELS](isa/TSELS.md)

    - [TSHLS](isa/TSHLS.md)

    - [TSHRS](isa/TSHRS.md)

- [Reduction Operations](menu/reduce_to_col.md)

  - [Reduction to a Column](menu/reduce_to_col.md)

    - [TROWSUM](isa/TROWSUM.md)

    - [TROWPROD](isa/TROWPROD.md)

    - [TROWMAX](isa/TROWMAX.md)

    - [TROWMIN](isa/TROWMIN.md)

    - [TROWARGMAX](isa/TROWARGMAX.md)

    - [TROWARGMIN](isa/TROWARGMIN.md)

  - [Reduction to a Row](menu/reduce_to_row.md)

    - [TCOLSUM](isa/TCOLSUM.md)

    - [TCOLPROD](isa/TCOLPROD.md)

    - [TCOLMAX](isa/TCOLMAX.md)

    - [TCOLMIN](isa/TCOLMIN.md)

    - [TCOLARGMAX](isa/TCOLARGMAX.md)

    - [TCOLARGMIN](isa/TCOLARGMIN.md)

- [Broadcast Operations](menu/broadcast_row.md)

  - [Row-Wise Broadcast](menu/broadcast_row.md)

    - [TROWEXPAND](isa/TROWEXPAND.md)

    - [TROWEXPANDADD](isa/TROWEXPANDADD.md)

    - [TROWEXPANDSUB](isa/TROWEXPANDSUB.md)

    - [TROWEXPANDMUL](isa/TROWEXPANDMUL.md)

    - [TROWEXPANDDIV](isa/TROWEXPANDDIV.md)

    - [TROWEXPANDMAX](isa/TROWEXPANDMAX.md)

    - [TROWEXPANDMIN](isa/TROWEXPANDMIN.md)

    - [TROWEXPANDEXPDIF](isa/TROWEXPANDEXPDIF.md)

  - [Column-Wise Broadcast](menu/broadcast_col.md)

    - [TCOLEXPAND](isa/TCOLEXPAND.md)

    - [TCOLEXPANDADD](isa/TCOLEXPANDADD.md)

    - [TCOLEXPANDSUB](isa/TCOLEXPANDSUB.md)

    - [TCOLEXPANDMUL](isa/TCOLEXPANDMUL.md)

    - [TCOLEXPANDDIV](isa/TCOLEXPANDDIV.md)

    - [TCOLEXPANDMAX](isa/TCOLEXPANDMAX.md)

    - [TCOLEXPANDMIN](isa/TCOLEXPANDMIN.md)

    - [TCOLEXPANDEXPDIF](isa/TCOLEXPANDEXPDIF.md)

- [Matrix Operations](menu/matmul_mat.md)

  - [Matrix-Matrix Multiplication](menu/matmul_mat.md)

    - [TMATMUL](isa/TMATMUL.md)

    - [TMATMUL_BIAS](isa/TMATMUL_BIAS.md)

    - [TMATMUL_ACC](isa/TMATMUL_ACC.md)

    - [TMATMUL_MX](isa/TMATMUL_MX.md)

  - [Matrix-Vector Multiplication](menu/matmul_vec.md)

    - [TGEMV](isa/TGEMV.md)

    - [TGEMV_BIAS](isa/TGEMV_BIAS.md)

    - [TGEMV_ACC](isa/TGEMV_ACC.md)

    - [TGEMV_MX](isa/TGEMV_MX.md)

- [Data Movement and Memory Access](menu/regular_access.md)

  - [Regular Memory Access](menu/regular_access.md)

    - [TLOAD](isa/TLOAD.md)

    - [TSTORE](isa/TSTORE.md)

    - [TPREFETCH](isa/TPREFETCH.md)

  - [Irregular Memory Access](menu/irregular_access.md)

    - [MGATHER](isa/MGATHER.md)

    - [MSCATTER](isa/MSCATTER.md)

- [Complex Transformation Computation](menu/init.md)

  - [Initialization](menu/init.md)

    - [TEXPANDS](isa/TEXPANDS.md)

    - [TCI](isa/TCI.md)

    - [TTRI](isa/TTRI.md)

    - [TRANDOM](isa/TRANDOM.md)

    - [TFILLPAD](isa/TFILLPAD.md)

  - [Data Type Conversion](menu/cast.md)

    - [TCVT](isa/TCVT.md)

    - [TQUANT](isa/TQUANT.md)

    - [TDEQUANT](isa/TDEQUANT.md)

  - [Layout Transformation](menu/layout.md)

    - [TEXTRACT](isa/TEXTRACT.md)

    - [TINSERT](isa/TINSERT.md)

    - [TGATHER](isa/TGATHER.md)

    - [TSCATTER](isa/TSCATTER.md)

    - [TCONCAT](isa/TCONCAT.md)

    - [TTRANS](isa/TTRANS.md)

    - [TIMG2COL](isa/TIMG2COL.md)

    - [TMOV](isa/TMOV.md)

    - [TGATHERB](isa/TGATHERB.md)

    - [TDEINTERLEAVE](isa/TDEINTERLEAVE.md)

    - [TINTERLEAVE](isa/TINTERLEAVE.md)

    - [TRESHAPE](isa/TRESHAPE.md)

  - [Sorting](menu/sort.md)

    - [TSORT32](isa/TSORT32.md)

    - [TMRGSORT](isa/TMRGSORT.md)

    - [THISTOGRAM](isa/THISTOGRAM.md)

  - [Union Computation](menu/union.md)

    - [TPARTADD](isa/TPARTADD.md)

    - [TPARTMUL](isa/TPARTMUL.md)

    - [TPARTMAX](isa/TPARTMAX.md)

    - [TPARTMIN](isa/TPARTMIN.md)

    - [TPARTARGMAX](isa/TPARTARGMAX.md)

    - [TPARTARGMIN](isa/TPARTARGMIN.md)

- [System and Control](menu/debug.md)

  - [Debugging](menu/debug.md)

    - [TASSIGN](isa/TASSIGN.md)

    - [TPRINT](isa/TPRINT.md)

  - [CV Communication](menu/cv_comm.md)

    - [TPUSH](isa/TPUSH.md)

    - [TPOP](isa/TPOP.md)

    - [TALLOC](isa/TALLOC.md)

    - [TFREE](isa/TFREE.md)

  - [Synchronization](menu/sync.md)

    - [SYNCALL](isa/SYNCALL.md)

- [Communication](menu/sync_comm.md)

  - [Synchronous Communication](menu/sync_comm.md)

    - [TPUT](isa/comm/TPUT.md)

    - [TGET](isa/comm/TGET.md)

    - [TBROADCAST](isa/comm/TBROADCAST.md)

    - [TREDUCE](isa/comm/TREDUCE.md)

    - [TNOTIFY](isa/comm/TNOTIFY.md)

    - [TWAIT](isa/comm/TWAIT.md)

    - [TTEST](isa/comm/TTEST.md)

  - [Asynchronous Communication](menu/async_comm.md)

    - [TPUT_ASYNC](isa/comm/TPUT_ASYNC.md)

    - [TPUT_ASYNC_NOTIFY](isa/comm/TPUT_ASYNC_NOTIFY.md)

    - [TGET_ASYNC](isa/comm/TGET_ASYNC.md)
