/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TSUBVIEW_A2A3_HPP
#define TSUBVIEW_A2A3_HPP
#include <pto/common/type.hpp>

#ifdef __PTO_AUTO__
// only needed for auto mode
template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TSubView(typename TileDataDst::TileDType __out__ dst,
                                  typename TileDataSrc::TileDType __in__ src, uint16_t subTileValidRow,
                                  uint16_t subTileValidCol, uint16_t srcTileRowStride, uint16_t srcTileColStride,
                                  uint16_t rowIdx, uint16_t colIdx)
{
    return;
}
#endif

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSUBVIEW_IMPL(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx)
{
#ifndef __PTO_AUTO__
    // implement for manual mode (e.g. using TASSIGN_IMPL(dst, src.data() + ...);)
#else
    // we simply call a dummy tile function for auto mode, and the compiler
    // will generate the implementation during memory allocation
    static_assert(TileDataDst::Loc == TileDataSrc::Loc,
                  "The destination and source tiles must have the same TileType!");
    static_assert(TileDataDst::Rows == TileDataSrc::Rows,
                  "The destination and source tiles must have the same number of rows!");
    static_assert(TileDataDst::Cols == TileDataSrc::Cols,
                  "The destination and source tiles must have the same number of columns!");
    static_assert(std::is_same<typename TileDataDst::DType, typename TileDataSrc::DType>::value,
                  "The destination and source tiles must have the same scalar "
                  "element type!");
    static_assert(TileDataDst::BFractal == TileDataSrc::BFractal,
                  "The destination and source tiles must have the same BFractal");
    PTO_ASSERT(src.GetValidRow() % dst.GetValidRow() == 0,
               "The source tile's validRow must be a multiple of the destination "
               "tile's validRow!");
    PTO_ASSERT(src.GetValidCol() % dst.GetValidCol() == 0,
               "The source tile's validCol must be a multiple of the destination "
               "tile's validCol!");

    TSubView<TileDataDst, TileDataSrc>(dst.data(), src.data(), (uint16_t)dst.GetValidRow(), (uint16_t)dst.GetValidCol(),
                                       (uint16_t)TileDataSrc::RowStride, (uint16_t)TileDataSrc::ColStride, rowIdx,
                                       colIdx);
#endif
}

#endif