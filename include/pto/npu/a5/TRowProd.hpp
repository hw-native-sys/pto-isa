/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TROWPROD_HPP
#define TROWPROD_HPP

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/common/constants.hpp"
#include <math.h>
#include <type_traits>

namespace pto {
template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRowProdCheck(uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow)
{
    using T = typename TileDataIn::DType;
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>,
                  "TRowProd only supports 'half' or 'float' data types. "
                  "Fix: Define TileDataIn with DType = half or float.");
    static_assert(std::is_same_v<T, typename TileDataOut::DType>,
                  "Input and output tile data types must match. "
                  "Fix: Ensure TileDataOut uses the same DType as TileDataIn.");
    static_assert(TileDataOut::Loc == pto::TileType::Vec && TileDataIn::Loc == pto::TileType::Vec,
                  "TRowProd only works on vector tiles (TileType::Vec). "
                  "Fix: Instantiate TileDataIn and TileDataOut with Loc_ = TileType::Vec.");
    static_assert(TileDataIn::isRowMajor && !TileDataIn::isBoxedLayout,
                  "TRowProd input tile must use standard ND layout (row-major, non-fractal). "
                  "Fix: Define TileDataIn with BFractal_ = BLayout::RowMajor and SFractal_ "
                  "= SLayout::NoneBox, e.g.,\n"
                  "     Tile<TileType::Vec, T, ROWS, COLS, BLayout::RowMajor, ..., "
                  "SLayout::NoneBox>");
    static_assert((!TileDataOut::isBoxedLayout &&
                   (TileDataOut::isRowMajor || (!TileDataOut::isRowMajor && TileDataOut::Cols == 1))),
                  "TRowProd output tile layout must be either:\n"
                  "  (a) ND layout: BLayout::RowMajor + SLayout::NoneBox, OR\n"
                  "  (b) DN layout with exactly one column: BLayout::ColMajor + "
                  "SLayout::NoneBox + Cols=1.\n"
                  "Fix: Choose one of the following for TileDataOut:\n"
                  "     - Tile<..., ROWS, COLS, BLayout::RowMajor, ValidRows, 1>   // ND\n"
                  "     - Tile<..., ROWS, 1, BLayout::ColMajor, ValidRows, 1>  // DN with Cols=1");
    // runtime checks
    PTO_ASSERT(srcValidRows != 0 && srcValidCols != 0,
               "TRowProd input source valid rows or columns is zero — TRowProd requires at "
               "least one element per row. "
               "Fix: Ensure srcValidRows > 0 and srcValidCols > 0.");
    PTO_ASSERT(srcValidRows == dstValidRow,
               "TRowProd input and output valid row counts must be equal in TRowProd "
               "(row count is preserved). "
               "Fix: Pass dstValidRow = srcValidRows.");
}

template <typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL OP_NAME(TROWPROD)
    OP_TYPE(reduce) void TRowProd(typename TileDataOut::TileDType __out__ dstData,
                                  typename TileDataIn::TileDType __in__ srcData, unsigned rows, unsigned cols,
                                  unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataIn::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);

    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        RegTensor<T> oneReg;
        RegTensor<T> intlvReg1;
        RegTensor<T> intlvReg2;
        vbr(oneReg, 1);

        constexpr auto oneElemMask =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_ONEPT>())>();
        constexpr uint16_t nLoop = (sizeof(T) == 2) ? TROW_PROD_LOOP_B16 : TROW_PROD_LOOP_B32;
        MaskReg pReg;
        uint32_t sreg = cols;
        uint32_t tmp = elementsPerRepeat;
        MaskReg reducePReg = CreatePredicate<T>(tmp);

        for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
            sreg = cols;
            vbr(dstReg, 1);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                pReg = CreatePredicate<T>(sreg);
                vlds(srcReg, src, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
                vmul(dstReg, dstReg, srcReg, pReg, MODE_MERGING);
            }

            for (uint16_t k = 0; k < nLoop; k++) {
                vintlv(intlvReg1, intlvReg2, dstReg, oneReg);
                vmul(dstReg, intlvReg1, intlvReg2, reducePReg, MODE_ZEROING);
            }
            // reducePReg is neglected when oneElemMask mode set
            vsts(dstReg, dst, i * TileDataOut::RowStride, oneElemMask, reducePReg);
        }
    } // end VF
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWPROD_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();
    TRowProdCheck<TileDataOut, TileDataIn>(rows, cols, dst.GetValidRow());
    TRowProd<TileDataOut, TileDataIn>(dst.data(), src.data(), rows, cols);
}
} // namespace pto

#endif