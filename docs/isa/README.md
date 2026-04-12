<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA Reference

This directory contains the per-instruction reference for the PTO Tile Lib ISA.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- [Common conventions (operands, events, modifiers)](conventions.md)

## Synchronization

- [TSYNC](TSYNC.md) - Synchronize PTO execution (wait on events or insert a per-op pipeline barrier).

## Manual / Resource Binding

- [TASSIGN](TASSIGN.md) - Bind a Tile object to an implementation-defined on-chip address (manual placement).
- [TSETFMATRIX](TSETFMATRIX.md) - Set FMATRIX register(s) for IMG2COL-like ops.
- [TSET_IMG2COL_RPT](TSET_IMG2COL_RPT.md) - Set IMG2COL repeat metadata from an IMG2COL configuration tile.
- [TSET_IMG2COL_PADDING](TSET_IMG2COL_PADDING.md) - Set IMG2COL padding metadata from an IMG2COL configuration tile.

## Elementwise (Tile-Tile)

- [TADD](TADD.md) - Elementwise add of two tiles.
- [TABS](TABS.md) - Elementwise absolute value of a tile.
- [TAND](TAND.md) - Elementwise bitwise AND of two tiles.
- [TOR](TOR.md) - Elementwise bitwise OR of two tiles.
- [TSUB](TSUB.md) - Elementwise subtract of two tiles.
- [TMUL](TMUL.md) - Elementwise multiply of two tiles.
- [TMIN](TMIN.md) - Elementwise minimum of two tiles.
- [TMAX](TMAX.md) - Elementwise maximum of two tiles.
- [TCMP](TCMP.md) - Compare two tiles and write a packed predicate mask.
- [TDIV](TDIV.md) - Elementwise division of two tiles.
- [TSHL](TSHL.md) - Elementwise shift-left of two tiles.
- [TSHR](TSHR.md) - Elementwise shift-right of two tiles.
- [TXOR](TXOR.md) - Elementwise bitwise XOR of two tiles.
- [TLOG](TLOG.md) - Elementwise natural logarithm of a tile.
- [TRECIP](TRECIP.md) - Elementwise reciprocal of a tile.
- [TPRELU](TPRELU.md) - Elementwise PReLU (parametric ReLU) with a per-element slope tile.
- [TADDC](TADDC.md) - Elementwise ternary add: `src0 + src1 + src2`.
- [TSUBC](TSUBC.md) - Elementwise ternary op: `src0 - src1 + src2`.
- [TCVT](TCVT.md) - Elementwise type conversion with a specified rounding mode.
- [TSEL](TSEL.md) - Select between two tiles using a mask tile (per-element selection).
- [TRSQRT](TRSQRT.md) - Elementwise reciprocal square root.
- [TSQRT](TSQRT.md) - Elementwise square root.
- [TEXP](TEXP.md) - Elementwise exponential.
- [TNOT](TNOT.md) - Elementwise bitwise NOT of a tile.
- [TRELU](TRELU.md) - Elementwise ReLU of a tile.
- [TNEG](TNEG.md) - Elementwise negation of a tile.
- [TREM](TREM.md) - Elementwise remainder of two tiles.
- [TFMOD](TFMOD.md) - Elementwise fmod of two tiles.

## Tile-Scalar / Tile-Immediate

- [TEXPANDS](TEXPANDS.md) - Broadcast a scalar into a destination tile.
- [TCMPS](TCMPS.md) - Compare a tile against a scalar and write per-element comparison results.
- [TSELS](TSELS.md) - Select between source tile and scalar using a mask tile (per-element selection for source tile).
- [TMINS](TMINS.md) - Elementwise minimum of a tile and a scalar.
- [TADDS](TADDS.md) - Elementwise add a scalar to a tile.
- [TSUBS](TSUBS.md) - Elementwise subtract a scalar from a tile.
- [TDIVS](TDIVS.md) - Elementwise division with a scalar (tile/scalar or scalar/tile).
- [TMULS](TMULS.md) - Elementwise multiply a tile by a scalar.
- [TFMODS](TFMODS.md) - Elementwise remainder with a scalar: `fmod(src, scalar)`.
- [TREMS](TREMS.md) - Elementwise remainder with a scalar: `remainder(src, scalar)`.
- [TMAXS](TMAXS.md) - Elementwise max of a tile and a scalar: `max(src, scalar)`.
- [TANDS](TANDS.md) - Elementwise bitwise AND of a tile and a scalar.
- [TORS](TORS.md) - Elementwise bitwise OR of a tile and a scalar.
- [TSHLS](TSHLS.md) - Elementwise shift-left a tile by a scalar.
- [TSHRS](TSHRS.md) - Elementwise shift-right a tile by a scalar.
- [TXORS](TXORS.md) - Elementwise bitwise XOR of a tile and a scalar.
- [TLRELU](TLRELU.md) - Leaky ReLU with a scalar slope.
- [TADDSC](TADDSC.md) - Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`.
- [TSUBSC](TSUBSC.md) - Elementwise fused op: `src0 - scalar + src1`.

## Axis Reduce / Expand

- [TROWSUM](TROWSUM.md) - Reduce each row by summing across columns.
- [TROWPROD](TROWPROD.md) - Reduce each row by multiplying across columns.
- [TCOLSUM](TCOLSUM.md) - Reduce each column by summing across rows.
- [TCOLPROD](TCOLPROD.md) - Reduce each column by multiplying across rows.
- [TCOLMAX](TCOLMAX.md) - Reduce each column by taking the maximum across rows.
- [TROWMAX](TROWMAX.md) - Reduce each row by taking the maximum across columns.
- [TROWMIN](TROWMIN.md) - Reduce each row by taking the minimum across columns.
- [TROWARGMAX](TROWARGMAX.md) - Get the column index of the maximum element for each row.
- [TROWARGMIN](TROWARGMIN.md) - Get the column index of the minimum element for each row.
- [TCOLARGMAX](TCOLARGMAX.md) - Get the row index of the maximum element for each column.
- [TCOLARGMIN](TCOLARGMIN.md) - Get the row index of the minimum element for each column.
- [TROWEXPAND](TROWEXPAND.md) - Broadcast the first element of each source row across the destination row.
- [TROWEXPANDDIV](TROWEXPANDDIV.md) - Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`.
- [TROWEXPANDMUL](TROWEXPANDMUL.md) - Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`.
- [TROWEXPANDSUB](TROWEXPANDSUB.md) - Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`.
- [TROWEXPANDADD](TROWEXPANDADD.md) - Row-wise broadcast add: add a per-row scalar vector.
- [TROWEXPANDMAX](TROWEXPANDMAX.md) - Row-wise broadcast max with a per-row scalar vector.
- [TROWEXPANDMIN](TROWEXPANDMIN.md) - Row-wise broadcast min with a per-row scalar vector.
- [TROWEXPANDEXPDIF](TROWEXPANDEXPDIF.md) - Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars.
- [TCOLMIN](TCOLMIN.md) - Reduce each column by taking the minimum across rows.
- [TCOLEXPAND](TCOLEXPAND.md) - Broadcast the first element of each source column across the destination column.
- [TCOLEXPANDDIV](TCOLEXPANDDIV.md) - Column-wise broadcast divide: divide each column by a per-column scalar vector.
- [TCOLEXPANDMUL](TCOLEXPANDMUL.md) - Column-wise broadcast multiply: multiply each column by a per-column scalar vector.
- [TCOLEXPANDADD](TCOLEXPANDADD.md) - Column-wise broadcast add with per-column scalar vector.
- [TCOLEXPANDMAX](TCOLEXPANDMAX.md) - Column-wise broadcast max with per-column scalar vector.
- [TCOLEXPANDMIN](TCOLEXPANDMIN.md) - Column-wise broadcast min with per-column scalar vector.
- [TCOLEXPANDSUB](TCOLEXPANDSUB.md) - Column-wise broadcast subtract: subtract a per-column scalar vector from each column.
- [TCOLEXPANDEXPDIF](TCOLEXPANDEXPDIF.md) - Column-wise exp-diff: compute exp(src0 - src1) with per-column scalars.

## Memory (GM <-> Tile)

- [TLOAD](TLOAD.md) - Load data from a GlobalTensor (GM) into a Tile.
- [TPREFETCH](TPREFETCH.md) - Prefetch data from global memory into a tile-local cache/buffer (hint).
- [TSTORE](TSTORE.md) - Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters.
- [TSTORE_FP](TSTORE_FP.md) - Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters.
- [MGATHER](MGATHER.md) - Gather-load elements from global memory into a tile using per-element indices.
- [MSCATTER](MSCATTER.md) - Scatter-store elements from a tile into global memory using per-element indices.

## Matrix Multiply

- [TGEMV_MX](TGEMV_MX.md) - GEMV with additional scaling tiles for mixed-precision / quantized matrix-vector compute.
- [TMATMUL_MX](TMATMUL_MX.md) - Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets.
- [TMATMUL](TMATMUL.md) - Matrix multiply (GEMM) producing an accumulator/output tile.
- [TMATMUL_ACC](TMATMUL_ACC.md) - Matrix multiply with accumulator input (fused accumulate).
- [TMATMUL_BIAS](TMATMUL_BIAS.md) - Matrix multiply with bias add.
- [TGEMV](TGEMV.md) - General Matrix-Vector multiplication producing an accumulator/output tile.
- [TGEMV_ACC](TGEMV_ACC.md) - GEMV with explicit accumulator input/output tiles.
- [TGEMV_BIAS](TGEMV_BIAS.md) - GEMV with bias add.

## Data Movement / Layout

- [TEXTRACT](TEXTRACT.md) - Extract a sub-tile from a source tile.
- [TEXTRACT_FP](TEXTRACT_FP.md) - Extract with fp/scaling tile (vector-quantization parameters).
- [TIMG2COL](TIMG2COL.md) - Image-to-column transform for convolution-like workloads.
- [TINSERT](TINSERT.md) - Insert a sub-tile into a destination tile at an (indexRow, indexCol) offset.
- [TINSERT_FP](TINSERT_FP.md) - Insert with fp/scaling tile (vector-quantization parameters).
- [TFILLPAD](TFILLPAD.md) - Copy+pad a tile outside the valid region with a compile-time pad value.
- [TFILLPAD_INPLACE](TFILLPAD_INPLACE.md) - In-place fill/pad variant.
- [TFILLPAD_EXPAND](TFILLPAD_EXPAND.md) - Fill/pad while allowing dst to be larger than src.
- [TMOV](TMOV.md) - Move/copy between tiles, optionally applying implementation-defined conversion modes.
- [TMOV_FP](TMOV_FP.md) - Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters.
- [TRESHAPE](TRESHAPE.md) - Reinterpret a tile as another tile type/shape while preserving the underlying bytes.
- [TTRANS](TTRANS.md) - Transpose with an implementation-defined temporary tile.
- [TSUBVIEW](TSUBVIEW.md) - Reinterpret a tile as a subtile of another tile.
- [TGET_SCALE_ADDR](TGET_SCALE_ADDR.md) - Bind the on-chip address of output tile to a scaled factor of that of input tile.

## Complex

- [TPRINT](TPRINT.md) - Debug/print elements from a tile (implementation-defined).
- [TMRGSORT](TMRGSORT.md) - Merge sort for multiple sorted lists (implementation-defined element format and layout).
- [TSORT32](TSORT32.md) - Sort each 32-element block of `src` together with the corresponding indices from `idx`, and write the sorted value-index pairs into `dst`.
- [TGATHER](TGATHER.md) - Gather/select elements using either an index tile or a compile-time mask pattern.
- [TCI](TCI.md) - Generate a contiguous integer sequence into a destination tile.
- [TTRI](TTRI.md) - Generate a triangular (lower/upper) mask tile.
- [TRANDOM](TRANDOM.md) - Generates random numbers in the destination tile using a counter-based cipher algorithm.
- [TPARTADD](TPARTADD.md) - Partial elementwise add with implementation-defined handling of mismatched valid regions.
- [TPARTMUL](TPARTMUL.md) - Partial elementwise multiply with implementation-defined handling of mismatched valid regions.
- [TPARTMAX](TPARTMAX.md) - Partial elementwise max with implementation-defined handling of mismatched valid regions.
- [TPARTMIN](TPARTMIN.md) - Partial elementwise min with implementation-defined handling of mismatched valid regions.
- [TGATHERB](TGATHERB.md) - Gather elements using byte offsets.
- [TSCATTER](TSCATTER.md) - Scatter rows of a source tile into a destination tile using per-element row indices.
- [TQUANT](TQUANT.md) - Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs.

## Communication

See [comm/README.md](comm/README.md) for the full per-instruction communication ISA reference (point-to-point, async, synchronization, and collective operations).
