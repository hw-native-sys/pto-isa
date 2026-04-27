# PTO ISA Overview

This page is the source-synchronized ISA index generated from `docs/isa/manifest.yaml`.

## Docs Contents

| Area | Page | Description |
|---|---|---|
| Overview | [`docs/README.md`](README.md) | PTO ISA guide entry point and navigation. |
| Overview | [`docs/PTOISA.md`](PTOISA.md) | This page (overview + full instruction index). |
| ISA reference | [`docs/isa/README.md`](isa/scalar/ops/micro-instruction/README.md) | Per-instruction reference directory index. |
| ISA reference | [`docs/isa/conventions.md`](isa/conventions.md) | Shared notation, operands, events, and modifiers. |
| Syntax and operands | [`docs/isa/syntax-and-operands/assembly-model.md`](isa/syntax-and-operands/assembly-model.md) | Canonical PTO-AS spelling and operand syntax inside the ISA manual. |
| Source of truth | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header.md) | C++ intrinsic API (authoritative). |
| PTO Auto Mode | [`docs/auto_mode/README.md`](README.md) | PTO auto mode guide entry point. |

## Instruction Index (All PTO Instructions)

| Category | Instruction | Description |
|---|---|---|
| Synchronization | [`TSYNC`](isa/tile/ops/sync-and-config/tsync.md) | Synchronize PTO execution (wait on events or insert a per-op pipeline barrier). |
| Manual / Resource Binding | [`TASSIGN`](isa/tile/ops/sync-and-config/tassign.md) | Bind a Tile object to an implementation-defined on-chip address (manual placement). |
| Manual / Resource Binding | [`pto.setfmatrix`](isa/tile/ops/sync-and-config/setfmatrix.md) | Set FMATRIX register(s) for IMG2COL-like ops. |
| Manual / Resource Binding | [`pto.set_img2col_rpt`](isa/tile/ops/sync-and-config/set-img2col-rpt.md) | Set IMG2COL repeat metadata from an IMG2COL configuration tile. |
| Manual / Resource Binding | [`pto.set_img2col_padding`](isa/tile/ops/sync-and-config/set-img2col-padding.md) | Set IMG2COL padding metadata from an IMG2COL configuration tile. |
| Elementwise (Tile-Tile) | [`TADD`](isa/tile/ops/elementwise-tile-tile/tadd.md) | Elementwise add of two tiles. |
| Elementwise (Tile-Tile) | [`TABS`](isa/tile/ops/elementwise-tile-tile/tabs.md) | Elementwise absolute value of a tile. |
| Elementwise (Tile-Tile) | [`TAND`](isa/tile/ops/elementwise-tile-tile/tand.md) | Elementwise bitwise AND of two tiles. |
| Elementwise (Tile-Tile) | [`TOR`](isa/tile/ops/elementwise-tile-tile/tor.md) | Elementwise bitwise OR of two tiles. |
| Elementwise (Tile-Tile) | [`TSUB`](isa/tile/ops/elementwise-tile-tile/tsub.md) | Elementwise subtract of two tiles. |
| Elementwise (Tile-Tile) | [`TMUL`](isa/tile/ops/elementwise-tile-tile/tmul.md) | Elementwise multiply of two tiles. |
| Elementwise (Tile-Tile) | [`TMIN`](isa/tile/ops/elementwise-tile-tile/tmin.md) | Elementwise minimum of two tiles. |
| Elementwise (Tile-Tile) | [`TMAX`](isa/tile/ops/elementwise-tile-tile/tmax.md) | Elementwise maximum of two tiles. |
| Elementwise (Tile-Tile) | [`TCMP`](isa/tile/ops/elementwise-tile-tile/tcmp.md) | Compare two tiles and write a packed predicate mask. |
| Elementwise (Tile-Tile) | [`TDIV`](isa/tile/ops/elementwise-tile-tile/tdiv.md) | Elementwise division of two tiles. |
| Elementwise (Tile-Tile) | [`TSHL`](isa/tile/ops/elementwise-tile-tile/tshl.md) | Elementwise shift-left of two tiles. |
| Elementwise (Tile-Tile) | [`TSHR`](isa/tile/ops/elementwise-tile-tile/tshr.md) | Elementwise shift-right of two tiles. |
| Elementwise (Tile-Tile) | [`TXOR`](isa/tile/ops/elementwise-tile-tile/txor.md) | Elementwise bitwise XOR of two tiles. |
| Elementwise (Tile-Tile) | [`TLOG`](isa/tile/ops/elementwise-tile-tile/tlog.md) | Elementwise natural logarithm of a tile. |
| Elementwise (Tile-Tile) | [`TRECIP`](isa/tile/ops/elementwise-tile-tile/trecip.md) | Elementwise reciprocal of a tile. |
| Elementwise (Tile-Tile) | [`TPRELU`](isa/tile/ops/elementwise-tile-tile/tprelu.md) | Elementwise PReLU (parametric ReLU) with a per-element slope tile. |
| Elementwise (Tile-Tile) | [`TADDC`](isa/tile/ops/elementwise-tile-tile/taddc.md) | Elementwise ternary add: `src0 + src1 + src2`. |
| Elementwise (Tile-Tile) | [`TSUBC`](isa/tile/ops/elementwise-tile-tile/tsubc.md) | Elementwise ternary op: `src0 - src1 + src2`. |
| Elementwise (Tile-Tile) | [`TCVT`](isa/tile/ops/elementwise-tile-tile/tcvt.md) | Elementwise type conversion with a specified rounding mode. |
| Elementwise (Tile-Tile) | [`TSEL`](isa/tile/ops/elementwise-tile-tile/tsel.md) | Select between two tiles using a mask tile (per-element selection). |
| Elementwise (Tile-Tile) | [`TRSQRT`](isa/tile/ops/elementwise-tile-tile/trsqrt.md) | Elementwise reciprocal square root. |
| Elementwise (Tile-Tile) | [`TSQRT`](isa/tile/ops/elementwise-tile-tile/tsqrt.md) | Elementwise square root. |
| Elementwise (Tile-Tile) | [`TEXP`](isa/tile/ops/elementwise-tile-tile/texp.md) | Elementwise exponential. |
| Elementwise (Tile-Tile) | [`TNOT`](isa/tile/ops/elementwise-tile-tile/tnot.md) | Elementwise bitwise NOT of a tile. |
| Elementwise (Tile-Tile) | [`TRELU`](isa/tile/ops/elementwise-tile-tile/trelu.md) | Elementwise ReLU of a tile. |
| Elementwise (Tile-Tile) | [`TNEG`](isa/tile/ops/elementwise-tile-tile/tneg.md) | Elementwise negation of a tile. |
| Elementwise (Tile-Tile) | [`TREM`](isa/tile/ops/elementwise-tile-tile/trem.md) | Elementwise remainder of two tiles. |
| Elementwise (Tile-Tile) | [`TFMOD`](isa/tile/ops/elementwise-tile-tile/tfmod.md) | Elementwise fmod of two tiles. |
| Tile-Scalar / Tile-Immediate | [`TEXPANDS`](isa/tile/ops/tile-scalar-and-immediate/texpands.md) | Broadcast a scalar into a destination tile. |
| Tile-Scalar / Tile-Immediate | [`TCMPS`](isa/tile/ops/tile-scalar-and-immediate/tcmps.md) | Compare a tile against a scalar and write per-element comparison results. |
| Tile-Scalar / Tile-Immediate | [`TSELS`](isa/tile/ops/tile-scalar-and-immediate/tsels.md) | Select between source tile and scalar using a mask tile (per-element selection for source tile). |
| Tile-Scalar / Tile-Immediate | [`TMINS`](isa/tile/ops/tile-scalar-and-immediate/tmins.md) | Elementwise minimum of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TADDS`](isa/tile/ops/tile-scalar-and-immediate/tadds.md) | Elementwise add a scalar to a tile. |
| Tile-Scalar / Tile-Immediate | [`TSUBS`](isa/tile/ops/tile-scalar-and-immediate/tsubs.md) | Elementwise subtract a scalar from a tile. |
| Tile-Scalar / Tile-Immediate | [`TDIVS`](isa/tile/ops/tile-scalar-and-immediate/tdivs.md) | Elementwise division with a scalar (tile/scalar or scalar/tile). |
| Tile-Scalar / Tile-Immediate | [`TMULS`](isa/tile/ops/tile-scalar-and-immediate/tmuls.md) | Elementwise multiply a tile by a scalar. |
| Tile-Scalar / Tile-Immediate | [`TFMODS`](isa/tile/ops/tile-scalar-and-immediate/tfmods.md) | Elementwise remainder with a scalar: `fmod(src, scalar)`. |
| Tile-Scalar / Tile-Immediate | [`TREMS`](isa/tile/ops/tile-scalar-and-immediate/trems.md) | Elementwise remainder with a scalar: `remainder(src, scalar)`. |
| Tile-Scalar / Tile-Immediate | [`TMAXS`](isa/tile/ops/tile-scalar-and-immediate/tmaxs.md) | Elementwise max of a tile and a scalar: `max(src, scalar)`. |
| Tile-Scalar / Tile-Immediate | [`TANDS`](isa/tile/ops/tile-scalar-and-immediate/tands.md) | Elementwise bitwise AND of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TORS`](isa/tile/ops/tile-scalar-and-immediate/tors.md) | Elementwise bitwise OR of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TSHLS`](isa/tile/ops/tile-scalar-and-immediate/tshls.md) | Elementwise shift-left a tile by a scalar. |
| Tile-Scalar / Tile-Immediate | [`TSHRS`](isa/tile/ops/tile-scalar-and-immediate/tshrs.md) | Elementwise shift-right a tile by a scalar. |
| Tile-Scalar / Tile-Immediate | [`TXORS`](isa/tile/ops/tile-scalar-and-immediate/txors.md) | Elementwise bitwise XOR of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TLRELU`](isa/tile/ops/tile-scalar-and-immediate/tlrelu.md) | Leaky ReLU with a scalar slope. |
| Tile-Scalar / Tile-Immediate | [`TADDSC`](isa/tile/ops/tile-scalar-and-immediate/taddsc.md) | Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`. |
| Tile-Scalar / Tile-Immediate | [`TSUBSC`](isa/tile/ops/tile-scalar-and-immediate/tsubsc.md) | Elementwise fused op: `src0 - scalar + src1`. |
| Axis Reduce / Expand | [`TROWSUM`](isa/tile/ops/reduce-and-expand/trowsum.md) | Reduce each row by summing across columns. |
| Axis Reduce / Expand | [`TROWPROD`](isa/tile/ops/reduce-and-expand/trowprod.md) | Reduce each row by multiplying across columns. |
| Axis Reduce / Expand | [`TCOLSUM`](isa/tile/ops/reduce-and-expand/tcolsum.md) | Reduce each column by summing across rows. |
| Axis Reduce / Expand | [`TCOLPROD`](isa/tile/ops/reduce-and-expand/tcolprod.md) | Reduce each column by multiplying across rows. |
| Axis Reduce / Expand | [`TCOLMAX`](isa/tile/ops/reduce-and-expand/tcolmax.md) | Reduce each column by taking the maximum across rows. |
| Axis Reduce / Expand | [`TROWMAX`](isa/tile/ops/reduce-and-expand/trowmax.md) | Reduce each row by taking the maximum across columns. |
| Axis Reduce / Expand | [`TROWMIN`](isa/tile/ops/reduce-and-expand/trowmin.md) | Reduce each row by taking the minimum across columns. |
| Axis Reduce / Expand | [`TROWARGMAX`](isa/tile/ops/reduce-and-expand/trowargmax.md) | Get the column index of the maximum element for each row. |
| Axis Reduce / Expand | [`TROWARGMIN`](isa/tile/ops/reduce-and-expand/trowargmin.md) | Get the column index of the minimum element for each row. |
| Axis Reduce / Expand | [`TCOLARGMAX`](isa/tile/ops/reduce-and-expand/tcolargmax.md) | Get the row index of the maximum element for each column. |
| Axis Reduce / Expand | [`TCOLARGMIN`](isa/tile/ops/reduce-and-expand/tcolargmin.md) | Get the row index of the minimum element for each column. |
| Axis Reduce / Expand | [`TROWEXPAND`](isa/tile/ops/reduce-and-expand/trowexpand.md) | Broadcast the first element of each source row across the destination row. |
| Axis Reduce / Expand | [`TROWEXPANDDIV`](isa/tile/ops/reduce-and-expand/trowexpanddiv.md) | Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`. |
| Axis Reduce / Expand | [`TROWEXPANDMUL`](isa/tile/ops/reduce-and-expand/trowexpandmul.md) | Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`. |
| Axis Reduce / Expand | [`TROWEXPANDSUB`](isa/tile/ops/reduce-and-expand/trowexpandsub.md) | Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`. |
| Axis Reduce / Expand | [`TROWEXPANDADD`](isa/tile/ops/reduce-and-expand/trowexpandadd.md) | Row-wise broadcast add: add a per-row scalar vector. |
| Axis Reduce / Expand | [`TROWEXPANDMAX`](isa/tile/ops/reduce-and-expand/trowexpandmax.md) | Row-wise broadcast max with a per-row scalar vector. |
| Axis Reduce / Expand | [`TROWEXPANDMIN`](isa/tile/ops/reduce-and-expand/trowexpandmin.md) | Row-wise broadcast min with a per-row scalar vector. |
| Axis Reduce / Expand | [`TROWEXPANDEXPDIF`](isa/tile/ops/reduce-and-expand/trowexpandexpdif.md) | Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars. |
| Axis Reduce / Expand | [`TCOLMIN`](isa/tile/ops/reduce-and-expand/tcolmin.md) | Reduce each column by taking the minimum across rows. |
| Axis Reduce / Expand | [`TCOLEXPAND`](isa/tile/ops/reduce-and-expand/tcolexpand.md) | Broadcast the first element of each source column across the destination column. |
| Axis Reduce / Expand | [`TCOLEXPANDDIV`](isa/tile/ops/reduce-and-expand/tcolexpanddiv.md) | Column-wise broadcast divide: divide each column by a per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDMUL`](isa/tile/ops/reduce-and-expand/tcolexpandmul.md) | Column-wise broadcast multiply: multiply each column by a per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDADD`](isa/tile/ops/reduce-and-expand/tcolexpandadd.md) | Column-wise broadcast add with per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDMAX`](isa/tile/ops/reduce-and-expand/tcolexpandmax.md) | Column-wise broadcast max with per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDMIN`](isa/tile/ops/reduce-and-expand/tcolexpandmin.md) | Column-wise broadcast min with per-column scalar vector. |
| Axis Reduce / Expand | [`TCOLEXPANDSUB`](isa/tile/ops/reduce-and-expand/tcolexpandsub.md) | Column-wise broadcast subtract: subtract a per-column scalar vector from each column. |
| Axis Reduce / Expand | [`TCOLEXPANDEXPDIF`](isa/tile/ops/reduce-and-expand/tcolexpandexpdif.md) | Column-wise exp-diff: compute exp(src0 - src1) with per-column scalars. |
| Memory (GM <-> Tile) | [`TLOAD`](isa/tile/ops/memory-and-data-movement/tload.md) | Load data from a GlobalTensor (GM) into a Tile. |
| Memory (GM <-> Tile) | [`TPREFETCH`](isa/tile/ops/memory-and-data-movement/tprefetch.md) | Prefetch data from global memory into a tile-local cache/buffer (hint). |
| Memory (GM <-> Tile) | [`TSTORE`](isa/tile/ops/memory-and-data-movement/tstore.md) | Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters. |
| Memory (GM <-> Tile) | [`TSTORE_FP`](isa/tile/ops/memory-and-data-movement/tstore.md) | Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters. |
| Memory (GM <-> Tile) | [`MGATHER`](isa/tile/ops/memory-and-data-movement/mgather.md) | Gather-load elements from global memory into a tile using per-element indices. |
| Memory (GM <-> Tile) | [`MSCATTER`](isa/tile/ops/memory-and-data-movement/mscatter.md) | Scatter-store elements from a tile into global memory using per-element indices. |
| Matrix Multiply | [`TGEMV_MX`](isa/tile/ops/matrix-and-matrix-vector/tgemv-mx.md) | GEMV with additional scaling tiles for mixed-precision / quantized matrix-vector compute. |
| Matrix Multiply | [`TMATMUL_MX`](isa/tile/ops/matrix-and-matrix-vector/tmatmul-mx.md) | Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets. |
| Matrix Multiply | [`TMATMUL`](isa/tile/ops/matrix-and-matrix-vector/tmatmul.md) | Matrix multiply (GEMM) producing an accumulator/output tile. |
| Matrix Multiply | [`TMATMUL_ACC`](isa/tile/ops/matrix-and-matrix-vector/tmatmul-acc.md) | Matrix multiply with accumulator input (fused accumulate). |
| Matrix Multiply | [`TMATMUL_BIAS`](isa/tile/ops/matrix-and-matrix-vector/tmatmul-bias.md) | Matrix multiply with bias add. |
| Matrix Multiply | [`TGEMV`](isa/tile/ops/matrix-and-matrix-vector/tgemv.md) | General Matrix-Vector multiplication producing an accumulator/output tile. |
| Matrix Multiply | [`TGEMV_ACC`](isa/tile/ops/matrix-and-matrix-vector/tgemv-acc.md) | GEMV with explicit accumulator input/output tiles. |
| Matrix Multiply | [`TGEMV_BIAS`](isa/tile/ops/matrix-and-matrix-vector/tgemv-bias.md) | GEMV with bias add. |
| Data Movement / Layout | [`TEXTRACT`](isa/tile/ops/layout-and-rearrangement/textract.md) | Extract a sub-tile from a source tile. |
| Data Movement / Layout | [`TEXTRACT_FP`](isa/tile/ops/layout-and-rearrangement/textract.md) | Extract with fp/scaling tile (vector-quantization parameters). |
| Data Movement / Layout | [`TIMG2COL`](isa/tile/ops/layout-and-rearrangement/timg2col.md) | Image-to-column transform for convolution-like workloads. |
| Data Movement / Layout | [`TINSERT`](isa/tile/ops/layout-and-rearrangement/tinsert.md) | Insert a sub-tile into a destination tile at an (indexRow, indexCol) offset. |
| Data Movement / Layout | [`TINSERT_FP`](isa/tile/ops/layout-and-rearrangement/tinsert.md) | Insert with fp/scaling tile (vector-quantization parameters). |
| Data Movement / Layout | [`TFILLPAD`](isa/tile/ops/layout-and-rearrangement/tfillpad.md) | Copy+pad a tile outside the valid region with a compile-time pad value. |
| Data Movement / Layout | [`TFILLPAD_INPLACE`](isa/tile/ops/layout-and-rearrangement/tfillpad-inplace.md) | In-place fill/pad variant. |
| Data Movement / Layout | [`TFILLPAD_EXPAND`](isa/tile/ops/layout-and-rearrangement/tfillpad-expand.md) | Fill/pad while allowing dst to be larger than src. |
| Data Movement / Layout | [`TMOV`](isa/tile/ops/layout-and-rearrangement/tmov.md) | Move/copy between tiles, optionally applying implementation-defined conversion modes. |
| Data Movement / Layout | [`TMOV_FP`](isa/tile/ops/layout-and-rearrangement/tmov.md) | Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters. |
| Data Movement / Layout | [`TRESHAPE`](isa/tile/ops/layout-and-rearrangement/treshape.md) | Reinterpret a tile as another tile type/shape while preserving the underlying bytes. |
| Data Movement / Layout | [`TTRANS`](isa/tile/ops/layout-and-rearrangement/ttrans.md) | Transpose with an implementation-defined temporary tile. |
| Data Movement / Layout | [`TSUBVIEW`](isa/tile/ops/sync-and-config/subview.md) | Reinterpret a tile as a subtile of another tile. |
| Data Movement / Layout | [`TGET_SCALE_ADDR`](isa/tile/ops/sync-and-config/get-scale-addr.md) | Bind the on-chip address of output tile to a scaled factor of that of input tile. |
| Complex | [`TPRINT`](isa/tile/ops/irregular-and-complex/tprint.md) | Debug/print elements from a tile (implementation-defined). |
| Complex | [`TMRGSORT`](isa/tile/ops/irregular-and-complex/tmrgsort.md) | Merge sort for multiple sorted lists (implementation-defined element format and layout). |
| Complex | [`TSORT32`](isa/tile/ops/irregular-and-complex/tsort32.md) | Sort 32-element blocks of `src` with accompanying `idx` entries and output sorted value-index pairs. |
| Complex | [`TGATHER`](isa/tile/ops/irregular-and-complex/tgather.md) | Gather/select elements using either an index tile or a compile-time mask pattern. |
| Complex | [`TCI`](isa/tile/ops/irregular-and-complex/tci.md) | Generate a contiguous integer sequence into a destination tile. |
| Complex | [`TTRI`](isa/tile/ops/irregular-and-complex/ttri.md) | Generate a triangular (lower/upper) mask tile. |
| Complex | [`TRANDOM`](isa/tile/ops/irregular-and-complex/trandom.md) | Generates random numbers in the destination tile using a counter-based cipher algorithm. |
| Complex | [`TPARTADD`](isa/tile/ops/irregular-and-complex/tpartadd.md) | Partial elementwise add with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMUL`](isa/tile/ops/irregular-and-complex/tpartmul.md) | Partial elementwise multiply with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMAX`](isa/tile/ops/irregular-and-complex/tpartmax.md) | Partial elementwise max with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMIN`](isa/tile/ops/irregular-and-complex/tpartmin.md) | Partial elementwise min with implementation-defined handling of mismatched valid regions. |
| Complex | [`TGATHERB`](isa/tile/ops/irregular-and-complex/tgatherb.md) | Gather elements using byte offsets. |
| Complex | [`TSCATTER`](isa/tile/ops/irregular-and-complex/tscatter.md) | Scatter rows of a source tile into a destination tile using per-element row indices. |
| Complex | [`TQUANT`](isa/tile/ops/irregular-and-complex/tquant.md) | Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs. |
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
