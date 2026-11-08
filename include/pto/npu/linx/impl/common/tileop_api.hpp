#ifndef TILEOP_API_HPP
#define TILEOP_API_HPP

#include "pto/npu/linx/impl/common/tileop_api_impl.hpp"

// The Linx backend exports TCI and its CUBE 2D variants directly from
// jcore/template_asm.hpp.  Keep the portable forwarding wrappers out of that
// configuration to avoid duplicate definitions.
#ifndef __linx
template <is_tile_data_v tile_shape, typename T, int descending = 0>
void TCI(tile_shape &dst, T s) {
  TCI_Impl<tile_shape, T, descending>(dst, s);
}

// Two-dimensional CUBE sequence generation.  The step values are part of the
// type so that the hardware form can be validated before emitting a bundle.
// The original TCI entry point above intentionally remains RowMajor-only.
template <is_tile_data_v tile_shape, typename T, int row_step, int col_step>
void TCI_2D(tile_shape &dst, T s) {
  TCI_2D_Impl<tile_shape, T, row_step, col_step>(dst, s);
}

template <is_tile_data_v tile_shape, typename T>
void TCI_ROW(tile_shape &dst, T s) {
  TCI_2D<tile_shape, T, 1, 0>(dst, s);
}

template <is_tile_data_v tile_shape, typename T>
void TCI_COL(tile_shape &dst, T s) {
  TCI_2D<tile_shape, T, 0, 1>(dst, s);
}

template <is_tile_data_v D, is_tile_data_v A, is_tile_data_v B>
void TEXPDIF(D &dst, A &src0, B &src1) {
  validate_texpdif_operands<D, A, B>();
  TEXPDIF_Impl(dst, src0, src1);
}
#ifdef __cpu_sim__
template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TLOAD(tile_shape &dst, gm_shape &src) {
  TLOAD_Impl(dst, src);
}
template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
void TSTORE(gm_shape &dst, tile_shape &src) {
  TSTORE_Impl(dst, src);
}
#endif

// TGATHER / TSCATTER: defined as one-layer inline-asm in
// jcore/template_asm.hpp (no __vec__ kernel), so no wrapper here. The
// jcore/cpu_sim *_Impl definitions remain available for callers that
// prefer the kernel-launch form.
#endif
#endif
