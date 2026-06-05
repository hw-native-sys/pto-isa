/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMS_HPP
#define TREMS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/TBinSOp.hpp>
#include <pto/npu/kirin9030/TRem.hpp>

namespace pto {

template <typename T>
struct RemSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T> &dst, RegTensor<T> &src0, T scalar, MaskReg &preg)
    {
        RegTensor<T> src1;
        vdup(src1, scalar, preg, MODE_ZEROING);
        RemOp<T>::BinInstr(dst, src0, src1, preg);
    }
};

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TREMS)
    OP_TYPE(element_wise) void TRemS(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                     typename SrcTile::DType scalar, unsigned kValidRows, unsigned kValidCols,
                                     VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned blockSizeElem = CCE_VL / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = DstTile::RowStride;
    constexpr unsigned srcRowStride = SrcTile::RowStride;

    BinaryInstr<RemSOp<T>, DstTile, SrcTile, T, elementsPerRepeat, blockSizeElem, dstRowStride, srcRowStride>(
        dstPtr, srcPtr, scalar, kValidRows, kValidCols, version);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TRemSCheck(unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol)
{
    using T = typename DstTile::DType;
    static_assert(std::is_same_v<T, typename SrcTile::DType>, "The data type must be same of src and dst");
    static_assert((sizeof(T) == 2) || (sizeof(T) == 4), "TREMS: Invalid data type");
    static_assert((DstTile::Loc == TileType::Vec) && (SrcTile::Loc == TileType::Vec),
                  "TileType of dst and src tiles must be TileType::Vec.");
    static_assert((DstTile::ValidCol <= DstTile::Cols) && (DstTile::ValidRow <= DstTile::Rows) &&
                      (SrcTile::ValidCol <= SrcTile::Cols) && (SrcTile::ValidRow <= SrcTile::Rows),
                  "Number of valid columns and rows must not be greater than number of tile columns and rows.");
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename DstTile, typename SrcTile, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(DstTile &dst, SrcTile &src, typename SrcTile::DType scalar, TileDataTmp &tmp)
{
    using T = typename DstTile::DType;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    PTO_ASSERT((src.GetValidCol() == validCol) && (src.GetValidRow() == validRow),
               "Number of validColumns and validRows of src and dst must be the same.");

    TRemSCheck<DstTile, SrcTile>(src.GetValidRow(), src.GetValidCol(), validRow, validCol);
    TRemS<DstTile, SrcTile>(dst.data(), src.data(), scalar, validRow, validCol);
}
} // namespace pto
#endif
