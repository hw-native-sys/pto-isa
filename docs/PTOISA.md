# PTO ISA Overview

This page is the source-synchronized ISA index generated from `docs/isa/manifest.yaml`.

## Docs Contents

| Area | Page | Description |
|---|---|---|
| Overview | [`docs/README.md`](README.md) | PTO ISA guide entry point and navigation. |
| Overview | [`docs/PTOISA.md`](PTOISA.md) | This page (overview + full instruction index). |
| ISA reference | [`docs/isa/README.md`](isa/README.md) | Per-instruction reference directory index. |
| ISA reference | [`docs/isa/conventions.md`](isa/conventions.md) | Shared notation, operands, events, and modifiers. |
| Assembly (PTO-AS) | [`docs/assembly/PTO-AS.md`](assembly/PTO-AS.md) | PTO-AS syntax reference. |
| Source of truth | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header.md) | C++ intrinsic API (authoritative). |
| PTO Auto Mode | [`docs/auto_mode/README.md`](README.md) | PTO auto mode guide entry point. |

## Instruction Index (All PTO Instructions)

| Category | Instruction | Description |
|---|---|---|
| Synchronization | [`TSYNC`](isa/TSYNC.md) | Synchronize PTO execution (wait on events or insert a per-op pipeline barrier). |
| Manual / Resource Binding | [`TASSIGN`](isa/TASSIGN.md) | Bind a Tile object to an implementation-defined on-chip address (manual placement). |
| Manual / Resource Binding | [`SETFMATRIX`](isa/SETFMATRIX.md) | Set FMATRIX register(s) for IMG2COL-like ops. |
| Manual / Resource Binding | [`SET_IMG2COL_RPT`](isa/SET_IMG2COL_RPT.md) | Set IMG2COL repeat metadata from an IMG2COL configuration tile. |
| Manual / Resource Binding | [`SET_IMG2COL_PADDING`](isa/SET_IMG2COL_PADDING.md) | Set IMG2COL padding metadata from an IMG2COL configuration tile. |
| Elementwise (Tile-Tile) | [`TADD`](isa/TADD.md) | Elementwise add of two tiles. |
| Elementwise (Tile-Tile) | [`TABS`](isa/TABS.md) | Elementwise absolute value of a tile. |
| Elementwise (Tile-Tile) | [`TAND`](isa/TAND.md) | Elementwise bitwise AND of two tiles. |
| Elementwise (Tile-Tile) | [`TOR`](isa/TOR.md) | Elementwise bitwise OR of two tiles. |
| Elementwise (Tile-Tile) | [`TSUB`](isa/TSUB.md) | Elementwise subtract of two tiles. |
| Elementwise (Tile-Tile) | [`TMUL`](isa/TMUL.md) | Elementwise multiply of two tiles. |
| Elementwise (Tile-Tile) | [`TMIN`](isa/TMIN.md) | Elementwise minimum of two tiles. |
| Elementwise (Tile-Tile) | [`TMAX`](isa/TMAX.md) | Elementwise maximum of two tiles. |
| Elementwise (Tile-Tile) | [`TCMP`](isa/TCMP.md) | Compare two tiles and write a packed predicate mask. |
| Elementwise (Tile-Tile) | [`TDIV`](isa/TDIV.md) | Elementwise division of two tiles. |
| Elementwise (Tile-Tile) | [`TSHL`](isa/TSHL.md) | Elementwise shift-left of two tiles. |
| Elementwise (Tile-Tile) | [`TSHR`](isa/TSHR.md) | Elementwise shift-right of two tiles. |
| Elementwise (Tile-Tile) | [`TXOR`](isa/TXOR.md) | Elementwise bitwise XOR of two tiles. |
| Elementwise (Tile-Tile) | [`TLOG`](isa/TLOG.md) | Elementwise natural logarithm of a tile. |
| Elementwise (Tile-Tile) | [`TRECIP`](isa/TRECIP.md) | Elementwise reciprocal of a tile. |
| Elementwise (Tile-Tile) | [`TPRELU`](isa/TPRELU.md) | Elementwise PReLU (parametric ReLU) with a per-element slope tile. |
| Elementwise (Tile-Tile) | [`TADDC`](isa/TADDC.md) | Elementwise ternary add: `src0 + src1 + src2`. |
| Elementwise (Tile-Tile) | [`TSUBC`](isa/TSUBC.md) | Elementwise ternary op: `src0 - src1 + src2`. |
| Elementwise (Tile-Tile) | [`TCVT`](isa/TCVT.md) | Elementwise type conversion with a specified rounding mode. |
| Elementwise (Tile-Tile) | [`TSEL`](isa/TSEL.md) | Select between two tiles using a mask tile (per-element selection). |
| Elementwise (Tile-Tile) | [`TRSQRT`](isa/TRSQRT.md) | Elementwise reciprocal square root. |
| Elementwise (Tile-Tile) | [`TSQRT`](isa/TSQRT.md) | Elementwise square root. |
| Elementwise (Tile-Tile) | [`TEXP`](isa/TEXP.md) | Elementwise exponential. |
| Elementwise (Tile-Tile) | [`TNOT`](isa/TNOT.md) | Elementwise bitwise NOT of a tile. |
| Elementwise (Tile-Tile) | [`TRELU`](isa/TRELU.md) | Elementwise ReLU of a tile. |
| Elementwise (Tile-Tile) | [`TNEG`](isa/TNEG.md) | Elementwise negation of a tile. |
| Elementwise (Tile-Tile) | [`TREM`](isa/TREM.md) | Elementwise remainder of two tiles. |
| Elementwise (Tile-Tile) | [`TFMOD`](isa/TFMOD.md) | Elementwise fmod of two tiles. |
| Elementwise (Tile-Tile) | [`TPOW`](isa/TPOW.md) | Elementwise power of two tiles. |
| Tile-Scalar / Tile-Immediate | [`TEXPANDS`](isa/TEXPANDS.md) | Broadcast a scalar into a destination tile. |
| Tile-Scalar / Tile-Immediate | [`TCMPS`](isa/TCMPS.md) | Compare a tile against a scalar and write per-element comparison results. |
| Tile-Scalar / Tile-Immediate | [`TSELS`](isa/TSELS.md) | Select between source tile and scalar using a mask tile (per-element selection for source tile). |
| Tile-Scalar / Tile-Immediate | [`TMINS`](isa/TMINS.md) | Elementwise minimum of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TADDS`](isa/TADDS.md) | Elementwise add a scalar to a tile. |
| Tile-Scalar / Tile-Immediate | [`TSUBS`](isa/TSUBS.md) | Elementwise subtract a scalar from a tile. |
| Tile-Scalar / Tile-Immediate | [`TDIVS`](isa/TDIVS.md) | Elementwise division with a scalar (tile/scalar or scalar/tile). |
| Tile-Scalar / Tile-Immediate | [`TMULS`](isa/TMULS.md) | Elementwise multiply a tile by a scalar. |
| Tile-Scalar / Tile-Immediate | [`TFMODS`](isa/TFMODS.md) | Elementwise remainder with a scalar: `fmod(src, scalar)`. |
| Tile-Scalar / Tile-Immediate | [`TREMS`](isa/TREMS.md) | Elementwise remainder with a scalar: `remainder(src, scalar)`. |
| Tile-Scalar / Tile-Immediate | [`TMAXS`](isa/TMAXS.md) | Elementwise max of a tile and a scalar: `max(src, scalar)`. |
| Tile-Scalar / Tile-Immediate | [`TANDS`](isa/TANDS.md) | Elementwise bitwise AND of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TORS`](isa/TORS.md) | Elementwise bitwise OR of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TSHLS`](isa/TSHLS.md) | Elementwise shift-left a tile by a scalar. |
| Tile-Scalar / Tile-Immediate | [`TSHRS`](isa/TSHRS.md) | Elementwise shift-right a tile by a scalar. |
| Tile-Scalar / Tile-Immediate | [`TXORS`](isa/TXORS.md) | Elementwise bitwise XOR of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TLRELU`](isa/TLRELU.md) | Leaky ReLU with a scalar slope. |
| Tile-Scalar / Tile-Immediate | [`TADDSC`](isa/TADDSC.md) | Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`. |
| Tile-Scalar / Tile-Immediate | [`TSUBSC`](isa/TSUBSC.md) | Elementwise fused op: `src0 - scalar + src1`. |
| Tile-Scalar / Tile-Immediate | [`TPOWS`](isa/TPOWS.md) | Elementwise power of a tile by a scalar. |
| Axis Reduce / Expand | [`TROWSUM`](isa/TROWSUM.md) | Reduce each row by summing across columns. |
| Axis Reduce / Expand | [`TROWPROD`](isa/TROWPROD.md) | Reduce each row by multiplying across columns. |
| Axis Reduce / Expand | [`TCOLSUM`](isa/TCOLSUM.md) | Reduce each column by summing across rows. |
| Axis Reduce / Expand | [`TCOLPROD`](isa/TCOLPROD.md) | Reduce each column by multiplying across rows. |
| Axis Reduce / Expand | [`TCOLMAX`](isa/TCOLMAX.md) | Reduce each column by taking the maximum across rows. |
| Axis Reduce / Expand | [`TROWMAX`](isa/TROWMAX.md) | Reduce each row by taking the maximum across columns. |
| Axis Reduce / Expand | [`TROWMIN`](isa/TROWMIN.md) | Reduce each row by taking the minimum across columns. |
| Axis Reduce / Expand | [`TROWARGMAX`](isa/TROWARGMAX.md) | Get the column index of the maximum element for each row. |
| Axis Reduce / Expand | [`TROWARGMIN`](isa/TROWARGMIN.md) | Get the column index of the minimum element for each row. |
| Axis Reduce / Expand | [`TCOLARGMAX`](isa/TCOLARGMAX.md) | Get the row index of the maximum element for each column. |
| Axis Reduce / Expand | [`TCOLARGMIN`](isa/TCOLARGMIN.md) | Get the row index of the minimum element for each column. |
| Axis Reduce / Expand | [`TROWEXPAND`](isa/TROWEXPAND.md) | Broadcast the first element of each source row across the destination row. |
| Axis Reduce / Expand | [`TROWEXPANDDIV`](isa/TROWEXPANDDIV.md) | Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`. |
| Axis Reduce / Expand | [`TROWEXPANDMUL`](isa/TROWEXPANDMUL.md) | Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`. |
| Axis Reduce / Expand | [`TROWEXPANDSUB`](isa/TROWEXPANDSUB.md) | Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`. |
| Axis Reduce / Expand | [`TROWEXPANDADD`](isa/TROWEXPANDADD.md) | Row-wise broadcast add: add a per-row scalar vector. |
| Axis Reduce / Expand | [`TROWEXPANDMAX`](isa/TROWEXPANDMAX.md) | Row-wise broadcast max with a per-row scalar vector. |
| Axis Reduce / Expand | [`TROWEXPANDMIN`](isa/TROWEXPANDMIN.md) | Row-wise broadcast min with a per-row scalar vector. |
| Axis Reduce / Expand | [`TROWEXPANDEXPDIF`](isa/TROWEXPANDEXPDIF.md) | Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars. |
| Axis Reduce / Expand | [`TCOLMIN`](isa/TCOLMIN.md) | Reduce each column by taking the minimum across rows. |
| Axis Reduce / Expand | [`TCOLEXPAND`](isa/TCOLEXPAND.md) | Broadcast the first element of each source column across the destination column. |
| Axis Reduce / Expand | [`TCOLEXPANDDIV`](isa/TCOLEXPANDDIV.md) | Column-wise broadcast divide: divide each column by a per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDMUL`](isa/TCOLEXPANDMUL.md) | Column-wise broadcast multiply: multiply each column by a per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDADD`](isa/TCOLEXPANDADD.md) | Column-wise broadcast add with per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDMAX`](isa/TCOLEXPANDMAX.md) | Column-wise broadcast max with per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDMIN`](isa/TCOLEXPANDMIN.md) | Column-wise broadcast min with per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDSUB`](isa/TCOLEXPANDSUB.md) | Column-wise broadcast subtract: subtract a per-column scalar vector from each column. |
| Axis Reduce / Expand | [`TCOLEXPANDEXPDIF`](isa/TCOLEXPANDEXPDIF.md) | Column-wise exp-diff: compute exp(src0 - src1) with per-column scalars. |
| Memory (GM <-> Tile) | [`TLOAD`](isa/TLOAD.md) | Load data from a GlobalTensor (GM) into a Tile. |
| Memory (GM <-> Tile) | [`TPREFETCH`](isa/TPREFETCH.md) | Prefetch data from global memory into a tile-local cache/buffer (hint). |
| Memory (GM <-> Tile) | [`TSTORE`](isa/TSTORE.md) | Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters. |
| Memory (GM <-> Tile) | [`TSTORE_FP`](isa/TSTORE_FP.md) | Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters. |
| Memory (GM <-> Tile) | [`MGATHER`](isa/MGATHER.md) | Gather-load elements from global memory into a tile using per-element indices. |
| Memory (GM <-> Tile) | [`MSCATTER`](isa/MSCATTER.md) | Scatter-store elements from a tile into global memory using per-element indices. |
| Matrix Multiply | [`TGEMV_MX`](isa/TGEMV_MX.md) | GEMV with additional scaling tiles for mixed-precision / quantized matrix-vector compute. |
| Matrix Multiply | [`TMATMUL_MX`](isa/TMATMUL_MX.md) | Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets. |
| Matrix Multiply | [`TMATMUL`](isa/TMATMUL.md) | Matrix multiply (GEMM) producing an accumulator/output tile. |
| Matrix Multiply | [`TMATMUL_ACC`](isa/TMATMUL_ACC.md) | Matrix multiply with accumulator input (fused accumulate). |
| Matrix Multiply | [`TMATMUL_BIAS`](isa/TMATMUL_BIAS.md) | Matrix multiply with bias add. |
| Matrix Multiply | [`TGEMV`](isa/TGEMV.md) | General Matrix-Vector multiplication producing an accumulator/output tile. |
| Matrix Multiply | [`TGEMV_ACC`](isa/TGEMV_ACC.md) | GEMV with explicit accumulator input/output tiles. |
| Matrix Multiply | [`TGEMV_BIAS`](isa/TGEMV_BIAS.md) | GEMV with bias add. |
| Data Movement / Layout | [`TEXTRACT`](isa/TEXTRACT.md) | Extract a sub-tile from a source tile. |
| Data Movement / Layout | [`TEXTRACT_FP`](isa/TEXTRACT_FP.md) | Extract with fp/scaling tile (vector-quantization parameters). |
| Data Movement / Layout | [`TIMG2COL`](isa/TIMG2COL.md) | Image-to-column transform for convolution-like workloads. |
| Data Movement / Layout | [`TINSERT`](isa/TINSERT.md) | Insert a sub-tile into a destination tile at an (indexRow, indexCol) offset. |
| Data Movement / Layout | [`TINSERT_FP`](isa/TINSERT_FP.md) | Insert with fp/scaling tile (vector-quantization parameters). |
| Data Movement / Layout | [`TFILLPAD`](isa/TFILLPAD.md) | Copy+pad a tile outside the valid region with a compile-time pad value. |
| Data Movement / Layout | [`TFILLPAD_INPLACE`](isa/TFILLPAD_INPLACE.md) | In-place fill/pad variant. |
| Data Movement / Layout | [`TFILLPAD_EXPAND`](isa/TFILLPAD_EXPAND.md) | Fill/pad while allowing dst to be larger than src. |
| Data Movement / Layout | [`TMOV`](isa/TMOV.md) | Move/copy between tiles, optionally applying implementation-defined conversion modes. |
| Data Movement / Layout | [`TMOV_FP`](isa/TMOV_FP.md) | Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters. |
| Data Movement / Layout | [`TRESHAPE`](isa/TRESHAPE.md) | Reinterpret a tile as another tile type/shape while preserving the underlying bytes. |
| Data Movement / Layout | [`TTRANS`](isa/TTRANS.md) | Transpose with an implementation-defined temporary tile. |
| Data Movement / Layout | [`TSUBVIEW`](isa/TSUBVIEW.md) | Reinterpret a tile as a subtile of another tile. |
| Data Movement / Layout | [`TGET_SCALE_ADDR`](isa/TGET_SCALE_ADDR.md) | Bind the on-chip address of output tile to a scaled factor of that of input tile. |
| Complex | [`TPRINT`](isa/TPRINT.md) | Debug/print elements from a tile (implementation-defined). |
| Complex | [`TMRGSORT`](isa/TMRGSORT.md) | Merge sort for multiple sorted lists (implementation-defined element format and layout). |
| Complex | [`TSORT32`](isa/TSORT32.md) | Sort 32-element blocks of `src` with accompanying `idx` entries and output sorted value-index pairs. |
| Complex | [`TGATHER`](isa/TGATHER.md) | Gather/select elements using either an index tile or a compile-time mask pattern. |
| Complex | [`TCI`](isa/TCI.md) | Generate a contiguous integer sequence into a destination tile. |
| Complex | [`TTRI`](isa/TTRI.md) | Generate a triangular (lower/upper) mask tile. |
| Complex | [`TRANDOM`](isa/TRANDOM.md) | Generates random numbers in the destination tile using a counter-based cipher algorithm. |
| Complex | [`TPARTADD`](isa/TPARTADD.md) | Partial elementwise add with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMUL`](isa/TPARTMUL.md) | Partial elementwise multiply with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMAX`](isa/TPARTMAX.md) | Partial elementwise max with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMIN`](isa/TPARTMIN.md) | Partial elementwise min with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTARGMAX`](isa/TPARTARGMAX.md) | Partial elementwise max selection returning corresponding index (argmax), with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTARGMIN`](isa/TPARTARGMIN.md) | Partial elementwise min selection returning corresponding index (argmin), with implementation-defined handling of mismatched valid regions. |
| Complex | [`TGATHERB`](isa/TGATHERB.md) | Gather elements using byte offsets. |
| Complex | [`TSCATTER`](isa/TSCATTER.md) | Scatter rows of a source tile into a destination tile using per-element row indices. |
| Complex | [`TQUANT`](isa/TQUANT.md) | Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs. |
| Communication | [`TPUT`](isa/comm/TPUT.md) | Remote write: transfer local data to remote NPU memory (GM → UB → GM). |
| Communication | [`TGET`](isa/comm/TGET.md) | Remote read: read remote NPU data to local memory (GM → UB → GM). |
| Communication | [`TPUT_ASYNC`](isa/comm/TPUT_ASYNC.md) | Asynchronous remote write (local GM → DMA engine → remote GM). |
| Communication | [`TGET_ASYNC`](isa/comm/TGET_ASYNC.md) | Asynchronous remote read (remote GM → DMA engine → local GM). |
| Communication | [`TNOTIFY`](isa/comm/TNOTIFY.md) | Send flag notification to remote NPU. |
| Communication | [`TWAIT`](isa/comm/TWAIT.md) | Blocking wait until signal(s) meet comparison condition. |
| Communication | [`TTEST`](isa/comm/TTEST.md) | Non-blocking test if signal(s) meet comparison condition. |
| Communication | [`TGATHER`](isa/comm/TGATHER.md) | Gather data from all ranks and concatenate along DIM_3. |
| Communication | [`TSCATTER`](isa/comm/TSCATTER.md) | Scatter data to all ranks by splitting along DIM_3. |
| Communication | [`TREDUCE`](isa/comm/TREDUCE.md) | Gather and reduce data from all ranks element-wise to local. |
| Communication | [`TBROADCAST`](isa/comm/TBROADCAST.md) | Broadcast data from current NPU to all ranks. |
