/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <pto/npu/a5/TBinOp.hpp>

namespace pto {

template <typename T>
struct RemOp {
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    static const U inf = sizeof(T) == sizeof(int32_t) ? 0x7F800000 : 0x7C00;
    static const U abs = sizeof(T) == sizeof(int32_t) ? 0x7FFFFFFF : 0x7FFF;
    static const U nan = sizeof(T) == sizeof(int32_t) ? 0x7FC00000 : 0x7E00;
    PTO_INTERNAL static void RemFloat(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& preg)
    {
        MaskReg infMask, diffSignMask;
        RegTensor<T> diffSign, src0Abs, nanReg, absReg;
        vdiv(dst, src0, src1, preg, MODE_ZEROING);
        vtrc(dst, dst, ROUND_F, preg);
        vmul(dst, dst, src1, preg, MODE_ZEROING);
        vsub(dst, src0, dst, preg, MODE_ZEROING);

        vmul(diffSign, src1, dst, preg, MODE_ZEROING);
        vcmps_lt(diffSignMask, diffSign, 0.0f, preg);
        vadd(diffSign, dst, src1, diffSignMask, MODE_MERGING);

        vdup((RegTensor<U>&)absReg, abs, preg, MODE_ZEROING);
        vand((RegTensor<U>&)src0Abs, (RegTensor<U>&)src0, (RegTensor<U>&)absReg, preg);
        vcmps_eq(infMask, (RegTensor<U>&)src0Abs, inf, preg);
        vdup((RegTensor<U>&)nanReg, nan, infMask, MODE_ZEROING);
        vsel(dst, nanReg, dst, infMask);
    }

    PTO_INTERNAL static void RemHalf(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& preg)
    {
        MaskReg infMask, diffSignMask;
        RegTensor<float> even0, even1, evenQuotient, odd0, odd1, oddQuotient;
        RegTensor<T> evenDst, oddDst, diffSign, src0Abs, nanReg, absReg;
        vcvt(even0, src0, preg, PART_EVEN);
        vcvt(even1, src1, preg, PART_EVEN);
        vcvt(odd0, src0, preg, PART_ODD);
        vcvt(odd1, src1, preg, PART_ODD);

        vdiv(evenQuotient, even0, even1, preg, MODE_ZEROING);
        vdiv(oddQuotient, odd0, odd1, preg, MODE_ZEROING);

        vtrc(evenQuotient, evenQuotient, ROUND_F, preg);
        vtrc(oddQuotient, oddQuotient, ROUND_F, preg);

        vmul(evenQuotient, evenQuotient, even1, preg, MODE_ZEROING);
        vmul(oddQuotient, oddQuotient, odd1, preg, MODE_ZEROING);

        vsub(evenQuotient, even0, evenQuotient, preg, MODE_ZEROING);
        vsub(oddQuotient, odd0, oddQuotient, preg, MODE_ZEROING);

        vcvt(evenDst, evenQuotient, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
        vcvt(oddDst, oddQuotient, preg, ROUND_Z, RS_ENABLE, PART_ODD);

        vor(dst, evenDst, oddDst, preg);

        vmul(diffSign, src1, dst, preg, MODE_ZEROING);
        vcmps_lt(diffSignMask, diffSign, 0.0f, preg);
        vadd(dst, dst, src1, diffSignMask, MODE_MERGING);

        vdup((RegTensor<U>&)absReg, abs, preg, MODE_ZEROING);
        vand((RegTensor<U>&)src0Abs, (RegTensor<U>&)src0, (RegTensor<U>&)absReg, preg);
        vcmps_eq(infMask, (RegTensor<U>&)src0Abs, inf, preg);
        vdup((RegTensor<U>&)nanReg, nan, infMask, MODE_ZEROING);
        vsel(dst, nanReg, dst, infMask);
    }

    PTO_INTERNAL static void RemInt(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& preg)
    {
        MaskReg diffSignMask;
        RegTensor<T> invalidRes;
        MaskReg dstNegMask, src1NegMask, zeroMask, curMask;
        curMask = preg;
        vdiv(dst, src0, src1, preg, MODE_ZEROING);
        vmul(dst, dst, src1, preg, MODE_ZEROING);
        vsub(dst, src0, dst, preg, MODE_ZEROING);

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
            MaskReg sign_diff_mask;
            RegTensor<T> reg_tmp;
            vdiv(dstReg, reg_src0, reg_src1, preg, MODE_ZEROING);
            vtrc(dstReg, dstReg, ROUND_F, preg);
            vmul(dstReg, dstReg, reg_src1, preg, MODE_ZEROING);
            vsub(dstReg, reg_src0, dstReg, preg, MODE_ZEROING);

            vmul(reg_tmp, reg_src1, dstReg, preg, MODE_ZEROING);
            vcmps_lt(sign_diff_mask, reg_tmp, 0.0f, preg);
            vadd(reg_tmp, dstReg, reg_src1, sign_diff_mask, MODE_ZEROING);
            vsel(dstReg, reg_tmp, dstReg, sign_diff_mask);
        } else if constexpr (std::is_same_v<T, half>) {
            RegTensor<float> reg_even0, reg_even1, reg_even2, reg_odd0, reg_odd1, reg_odd2;
            RegTensor<T> reg_dst_even, reg_dst_odd, reg_tmp;
            MaskReg sign_diff_mask;
            vcvt(reg_even0, reg_src0, preg, PART_EVEN);
            vcvt(reg_even1, reg_src1, preg, PART_EVEN);
            vcvt(reg_odd0, reg_src0, preg, PART_ODD);
            vcvt(reg_odd1, reg_src1, preg, PART_ODD);

            vdiv(reg_even2, reg_even0, reg_even1, preg, MODE_ZEROING);
            vdiv(reg_odd2, reg_odd0, reg_odd1, preg, MODE_ZEROING);

            vtrc(reg_even2, reg_even2, ROUND_F, preg);
            vtrc(reg_odd2, reg_odd2, ROUND_F, preg);

            vmul(reg_even2, reg_even2, reg_even1, preg, MODE_ZEROING);
            vmul(reg_odd2, reg_odd2, reg_odd1, preg, MODE_ZEROING);

            vsub(reg_even2, reg_even0, reg_even2, preg, MODE_ZEROING);
            vsub(reg_odd2, reg_odd0, reg_odd2, preg, MODE_ZEROING);

            vcvt(reg_dst_even, reg_even2, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
            vcvt(reg_dst_odd, reg_odd2, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(dstReg, reg_dst_even, reg_dst_odd, preg);

            vmul(reg_tmp, reg_src1, dstReg, preg, MODE_ZEROING);
            vcmps_lt(sign_diff_mask, reg_tmp, 0.0f, preg);
            vadd(reg_tmp, dstReg, reg_src1, sign_diff_mask, MODE_ZEROING);
            vsel(dstReg, reg_tmp, dstReg, sign_diff_mask);
        } else {
            // using cce intrinsic implement
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
            std::is_same_v<T, int16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>,
        "Fix: TREM has invalid data type.");
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
