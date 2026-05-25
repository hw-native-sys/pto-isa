/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTIALBINOPS_HPP
#define TPARTIALBINOPS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

template <typename T>
struct Padding {
    using Type = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t,
                                    std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint8_t>>;

    PTO_INTERNAL static constexpr Type GetPaddingMin()
    {
        if constexpr (std::is_same_v<T, float>) {
            return (Type)0xff800000UL;
        } else if constexpr (std::is_same_v<T, half>) {
            return (Type)0xfc00UL;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return (Type)0xff80UL;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return (Type)0x80000000UL;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return (Type)0x8000UL;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return (Type)0x80UL;
        } else {
            return (Type)0;
        }
    }

    PTO_INTERNAL static constexpr Type GetPaddingMax()
    {
        if constexpr (std::is_same_v<T, float>) {
            return (Type)0x7f800000UL;
        } else if constexpr (std::is_same_v<T, half>) {
            return (Type)0x7c00UL;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return (Type)0x7f80UL;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return (Type)0x7fffffffUL;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return (Type)0x7fffUL;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return (Type)0x7fUL;
        } else {
            return (Type)(~(Type)0);
        }
    }

    static constexpr Type Null = (Type)0;
    static constexpr Type Zero = (Type)0;
    static constexpr Type Min = GetPaddingMin();
    static constexpr Type Max = GetPaddingMax();
};

template <typename Op, typename DstTileData, typename Src0TileData, typename Src1TileData, unsigned elementsPerRepeat,
          unsigned src0Stride, unsigned src1Stride, unsigned dstStride>
__tf__ PTO_INTERNAL void TCopyPadOp(typename DstTileData::TileDType __out__ dst,
                                    typename Src0TileData::TileDType __in__ src0,
                                    typename Src1TileData::TileDType __in__ src1, uint32_t Src0validRow,
                                    uint32_t Src0validCol, uint32_t Src1validRow, uint32_t Src1validCol,
                                    uint32_t DstvalidRow, uint32_t DstvalidCol)
{
    using T = typename DstTileData::DType;
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __VEC_SCOPE__
    {
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        MaskReg dstMask, src0Mask, src1Mask;
        RegTensor<T> src0Reg;
        RegTensor<T> src1Reg;
        RegTensor<T> dstReg;
        uint16_t repeatTimes = CeilDivision(DstvalidCol, elementsPerRepeat);
        for (uint16_t i = 0; i < (uint16_t)DstvalidRow; ++i) {
            uint32_t dstSReg = DstvalidCol;
            uint32_t src0SReg = i < Src0validRow ? Src0validCol : 0;
            uint32_t src1SReg = i < Src1validRow ? Src1validCol : 0;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                dstMask = CreatePredicate<T>(dstSReg);
                src0Mask = CreatePredicate<T>(src0SReg);
                src1Mask = CreatePredicate<T>(src1SReg);
                vdup((RegTensor<typename Padding<T>::Type> &)dstReg, Op::PadVal, dstMask, MODE_ZEROING);
                vlds(src0Reg, src0Ptr + i * src0Stride, j * elementsPerRepeat, NORM);
                vlds(src1Reg, src1Ptr + i * src1Stride, j * elementsPerRepeat, NORM);
                Op::BinInstr(dstReg, dstReg, src0Reg, src0Mask);
                Op::BinInstr(dstReg, dstReg, src1Reg, src1Mask);
                vsts(dstReg, dstPtr + i * dstStride, j * elementsPerRepeat, distValue, dstMask);
            }
        }
    }
}

template <typename Op, typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void TPartMasterImpl(DstTileData &dst, Src0TileData &src0, Src1TileData &src1, VFImplKind version)
{
    using T = typename DstTileData::DType;
    using S0 = typename Src0TileData::DType;
    using S1 = typename Src1TileData::DType;

    static_assert(std::is_same_v<T, S0> && std::is_same_v<T, S1>,
                  "Fix: TPARTMAX/MIN Input and output types should match");

    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint16_t> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>,
                  "Fix: TPARTMAX/MIN Invalid data type.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    constexpr unsigned DstRowStride = DstTileData::RowStride;
    constexpr unsigned Src0RowStride = Src0TileData::RowStride;
    constexpr unsigned Src1RowStride = Src1TileData::RowStride;

    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();

    // dst has to be larger than or equal to both sources
    bool condDstgeSrc = (src1ValidRow <= dstValidRow && src1ValidCol <= dstValidCol) &&
                        (src0ValidRow <= dstValidRow && src0ValidCol <= dstValidCol);

    if (condDstgeSrc) { // src0 <= dst && src1 <= dst
        TCopyPadOp<Op, DstTileData, Src0TileData, Src1TileData, elementsPerRepeat, Src0RowStride, Src1RowStride,
                   DstRowStride>(dst.data(), src0.data(), src1.data(), src0ValidRow, src0ValidCol, src1ValidRow,
                                 src1ValidCol, dstValidRow, dstValidCol);
    } // other conditions not supported
}

} // namespace pto

#endif
