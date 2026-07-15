/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSUBS_HPP
#define TSUBS_HPP

#include "pto/npu/a5/TBinSOp.hpp"

namespace pto {

template <typename T>
struct SubSOp {
    static constexpr bool isDynFunc = true;
    RegTensor<T> src1Reg;

    PTO_INTERNAL SubSOp(T scalar) { vbr(src1Reg, scalar); };
    PTO_INTERNAL void BinSInstr(RegTensor<T>& dstReg, RegTensor<T>& src0Reg, T scalar, MaskReg& pReg)
    {
        vsub(dstReg, src0Reg, src1Reg, pReg, MODE_ZEROING);
    }
};

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TSubS(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src0, T src1, unsigned kValidRows,
    unsigned kValidCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    constexpr unsigned repeatElem = CCE_VL / sizeof(T);
    constexpr unsigned blockElem = BLOCK_BYTE_SIZE / sizeof(T);
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    BinaryInstr<SubSOp<T>, DstTile, SrcTile, T, repeatElem, blockElem, DstTile::RowStride, SrcTile::RowStride>(
        dstPtr, src0Ptr, src1, kValidRows, kValidCols, version);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TSUBS_IMPL(DstTile& dst, SrcTile& src0, typename SrcTile::DType src1)
{
    using T = typename DstTile::DType;
    static_assert(
        std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t> ||
            std::is_same_v<T, uint32_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t> ||
            std::is_same_v<T, half> || std::is_same_v<T, float32_t>,
        "TSUBS: Invalid data type");
    static_assert(
        (DstTile::Loc == TileType::Vec) && (SrcTile::Loc == TileType::Vec),
        "TileType of dst and src tiles must be TileType::Vec.");
    static_assert(
        (DstTile::ValidCol <= DstTile::Cols) && (DstTile::ValidRow <= DstTile::Rows) &&
            (SrcTile::ValidCol <= SrcTile::Cols) && (SrcTile::ValidRow <= SrcTile::Rows),
        "Number of valid columns and rows must not be greater than number of tile columns and rows.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRow, "Number of rows of src and dst must be the same.");
    PTO_ASSERT(src0.GetValidCol() == validCol, "Number of columns of src and dst must be the same.");

    TSubS<T, DstTile, SrcTile>(dst.data(), src0.data(), src1, validRow, validCol);
}
} // namespace pto
#endif
