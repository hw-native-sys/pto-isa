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

namespace pto {

template <typename T>
struct RemSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T>& dst, RegTensor<T>& src0, T scalar, MaskReg& preg)
    {
        RegTensor<T> src1;
        vdup(src1, scalar, preg, MODE_ZEROING);
        if constexpr (std::is_same_v<T, float>) {
            vdiv(dst, src0, src1, preg, MODE_ZEROING);
            vtrc(dst, dst, ROUND_F, preg);
            vmuls(dst, dst, scalar, preg, MODE_ZEROING);
            vsub(dst, src0, dst, preg, MODE_ZEROING);
        } else if constexpr (std::is_same_v<T, half>) {
            RegTensor<float> src0_even, src1_even, even, src0_odd, src1_odd, odd;
            RegTensor<T> dst_even, dst_odd;
            vcvt(src0_even, src0, preg, PART_EVEN);
            vcvt(src1_even, src1, preg, PART_EVEN);
            vcvt(src0_odd, src0, preg, PART_ODD);
            vcvt(src1_odd, src1, preg, PART_ODD);

            vdiv(even, src0_even, src1_even, preg, MODE_ZEROING);
            vdiv(odd, src0_odd, src1_odd, preg, MODE_ZEROING);

            vtrc(even, even, ROUND_F, preg);
            vtrc(odd, odd, ROUND_F, preg);

            vmuls(even, even, (float)scalar, preg, MODE_ZEROING);
            vmuls(odd, odd, (float)scalar, preg, MODE_ZEROING);

            vsub(even, src0_even, even, preg, MODE_ZEROING);
            vsub(odd, src0_odd, odd, preg, MODE_ZEROING);

            vcvt(dst_even, even, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
            vcvt(dst_odd, odd, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(dst, dst_even, dst_odd, preg);
        } else {
            // using cce intrinsic implement
        }
    }
};

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TREMS) OP_TYPE(element_wise) void TRemS(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, typename SrcTile::DType scalar,
    unsigned kValidRows, unsigned kValidCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);

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
    static_assert(
        (DstTile::Loc == TileType::Vec) && (SrcTile::Loc == TileType::Vec),
        "TileType of dst and src tiles must be TileType::Vec.");
    static_assert(
        (DstTile::ValidCol <= DstTile::Cols) && (DstTile::ValidRow <= DstTile::Rows) &&
            (SrcTile::ValidCol <= SrcTile::Cols) && (SrcTile::ValidRow <= SrcTile::Rows),
        "Number of valid columns and rows must not be greater than number of tile columns and rows.");
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename DstTile, typename SrcTile, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(DstTile& dst, SrcTile& src, typename SrcTile::DType scalar, TileDataTmp& tmp)
{
    using T = typename DstTile::DType;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    PTO_ASSERT(
        (src.GetValidCol() == validCol) && (src.GetValidRow() == validRow),
        "Number of validColumns and validRows of src and dst must be the same.");

    TRemSCheck<DstTile, SrcTile>(src.GetValidRow(), src.GetValidCol(), validRow, validCol);
    TRemS<DstTile, SrcTile>(dst.data(), src.data(), scalar, validRow, validCol);
}
} // namespace pto
#endif
