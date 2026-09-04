/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/common/debug.h>
#include "pto/npu/kirinDev0000/custom/DivIntSoft.hpp"
#include "pto/common/arch/register/trem_common.hpp"

namespace pto {

template <typename T>
struct RemOp : RemOpBase<T> {
    PTO_INTERNAL static void RemInt(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& preg)
    {
        MaskReg diffSignMask;
        RegTensor<T> invalidRes, q, prod;
        MaskReg dstNegMask, src1NegMask, zeroMask, curMask;
        curMask = preg;
        DivSoftIntImpl(q, src0, src1, preg);
        vmul(prod, q, src1, preg, MODE_ZEROING);
        vsub(dst, src0, prod, preg, MODE_ZEROING);

        // x/0 = -1
        vcmps_eq(zeroMask, src1, 0, preg);
        vdup(invalidRes, (T)-1, zeroMask, MODE_ZEROING);
        vsel(dst, invalidRes, dst, zeroMask);
        pxor(curMask, zeroMask, preg, preg);

        // dst[x] == 0 do not participate in subsequent processing.
        vcmps_eq(zeroMask, dst, 0, preg);
        pxor(curMask, zeroMask, curMask, curMask);

        // multiplication may cause overflow to become negative
        vcmps_lt(dstNegMask, dst, 0, curMask);
        vcmps_lt(src1NegMask, src1, 0, curMask);
        pxor(diffSignMask, dstNegMask, src1NegMask, curMask); // same is 0, diff is 1
        vadd(dst, dst, src1, diffSignMask, MODE_MERGING);     // add src1 if diff sign
    }

    PTO_INTERNAL static void BinInstr(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& preg)
    {
        if constexpr (std::is_same_v<T, float>) {
            RemOpBase<T>::RemFloat(dst, src0, src1, preg);
        } else if constexpr (std::is_same_v<T, half>) {
            RemOpBase<T>::RemHalf(dst, src0, src1, preg);
        } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
            RemInt(dst, src0, src1, preg);
        }
    }
};

template <typename DstTile, typename Src0Tile, typename Src1Tile>
__tf__ PTO_INTERNAL OP_NAME(TREM) OP_TYPE(element_wise) void TRem(
    typename DstTile::TileDType __out__ dst, typename Src0Tile::TileDType __in__ src0,
    typename Src1Tile::TileDType __in__ src1, unsigned validRows, unsigned validCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);

    constexpr unsigned blockSizeElem = CCE_VL / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    // Note: tmp parameter is not used in a5 implementation (no sign correction needed)
    BinaryInstr<RemOp<T>, DstTile, Src0Tile, Src1Tile, elementsPerRepeat, blockSizeElem>(
        dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
}

template <typename DstTile, typename Src0Tile, typename Src1Tile>
PTO_INTERNAL void TRemCheck(const DstTile& dst, const Src0Tile& src0, const Src1Tile& src1)
{
    using T = typename DstTile::DType;
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, uint16_t> ||
            std::is_same_v<T, int16_t>,
        "Fix: TREM has invalid data type.");
    static_assert(
        !std::is_same_v<T, int32_t> && !std::is_same_v<T, uint32_t>,
        "Fix: TREM does not support int32/uint32 on kirinDev0000: hardware lacks integer vdiv.");
    static_assert(
        DstTile::isRowMajor && Src0Tile::isRowMajor && Src1Tile::isRowMajor,
        "Fix: TREM only support row major layout.");
    static_assert(
        std::is_same_v<T, typename Src0Tile::DType> && std::is_same_v<T, typename Src1Tile::DType>,
        "Fix: TREM input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
}

template <
    auto PrecisionType = RemAlgorithm::DEFAULT, typename DstTile, typename Src0Tile, typename Src1Tile,
    typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(DstTile& dst, Src0Tile& src0, Src1Tile& src1, TileDataTmp& tmp)
{
    using T = typename DstTile::DType;
    TRemCheck<DstTile, Src0Tile, Src1Tile>(dst, src0, src1);

    TRem<DstTile, Src0Tile, Src1Tile>(dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
