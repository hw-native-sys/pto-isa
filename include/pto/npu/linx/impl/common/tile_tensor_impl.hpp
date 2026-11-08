#ifndef TILE_TENSOR_IMPL
#define TILE_TENSOR_IMPL

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include "pto/npu/linx/impl/common/tileop_api.hpp"

using namespace pto;

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_,
          const CompactMode Compact_, const bool ElementProfile_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_, Compact_, ElementProfile_>::
  Tile(std::enable_if_t<(RowMask > 0) && (ColMask > 0), DType> s) {
  TEXPANDS(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_,
          const CompactMode Compact_, const bool ElementProfile_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_, Compact_, ElementProfile_>::
  Tile(typename Tile::DType s,
       std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidRow,
       std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidCol) {
  RowMaskInternal = ValidRow;
  ColMaskInternal = ValidCol;
  TEXPANDS(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_,
          const CompactMode Compact_, const bool ElementProfile_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_, Compact_, ElementProfile_>::
  Tile(std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidRow,
       std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> ValidCol) {
  RowMaskInternal = ValidRow;
  ColMaskInternal = ValidCol;
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_,
          const CompactMode Compact_, const bool ElementProfile_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_, Compact_, ElementProfile_>::
  Tile(typename Tile::DType s,
       std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> ValidRow) {
  RowMaskInternal = ValidRow;
  TEXPANDS(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_,
          const CompactMode Compact_, const bool ElementProfile_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_, Compact_, ElementProfile_>::
  Tile(std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> ValidRow) {
  RowMaskInternal = ValidRow;
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_,
          const CompactMode Compact_, const bool ElementProfile_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_, Compact_, ElementProfile_>::
  Tile(DType s,
       std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> ValidCol) {
  ColMaskInternal = ValidCol;
  TEXPANDS(*this, s);
}

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout B_Fractal_, const int RowValid_, const int ColValid_,
          const SLayout S_Fractal_, const int S_FractalSize_, const PadValue PadVal_,
          const CompactMode Compact_, const bool ElementProfile_>
template <int RowMask, int ColMask>
inline Tile<Loc_, Element_, Rows_, Cols_, B_Fractal_, RowValid_, ColValid_,
            S_Fractal_, S_FractalSize_, PadVal_, Compact_, ElementProfile_>::
  Tile(std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> ValidCol) {
  ColMaskInternal = ValidCol;
}

#endif
