/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLREDUCEIDX_HPP
#define TCOLREDUCEIDX_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {
template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TColReduceIdxCheck(unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow,
                                     unsigned dstValidCol)
{
    static_assert((sizeof(typename TileDataIn::DType) == 1) || (sizeof(typename TileDataIn::DType) == 2) ||
                      (sizeof(typename TileDataIn::DType) == 4),
                  "Fix: TCOLREDUCEIDX data type must be b8/b16/b32");
    static_assert(TileDataIn::Loc == pto::TileType::Vec, "Fix: TCOLREDUCEIDX Src TileType must be Vec Tile!");
    static_assert(TileDataOut::Loc == pto::TileType::Vec, "Fix: TCOLREDUCEIDX Dst TileType must be Vec Tile!");
    static_assert(TileDataIn::SFractal == SLayout::NoneBox, "Fix: TCOLREDUCEIDX only support Nd or Dn fractal Tile");
    static_assert(TileDataOut::isRowMajor && TileDataOut::SFractal == SLayout::NoneBox,
                  "Fix: TCOLREDUCEIDX only support Nd fractal Tile");
    static_assert(
        std::is_same_v<typename TileDataOut::DType, uint32_t> || std::is_same_v<typename TileDataOut::DType, int32_t>,
        "Fix: TCOLREDUCEIDX output data type must be s32 or u32.");
    PTO_ASSERT(srcValidRow != 0 && srcValidCol != 0,
               "Fix: TCOLREDUCEIDX input shape is invalid, validCol or validRow is 0.");
    PTO_ASSERT(dstValidRow != 1, "Fix: TCOLREDUCEIDX output validRow must be 1");
    PTO_ASSERT(srcValidCol != dstValidCol,
               "Fix: TCOLREDUCEIDX input validCol must be consistent with the output validCol");
}
template <typename TileDataOut, typename TileDataIn, bool IsArgMax>
__tf__ PTO_INTERNAL void TColReduceIdx8(typename TileDataOut::TileDType __out__ dst,
                                        typename TileDataIn::TileDType __in__ src, unsigned srcValidRow,
                                        unsigned srcValidCol)
{
    using TOUT = typename TileDataOut::DType;
    using TIN = typename TileDataIn::DType;
    using T = std::conditional_t<std::is_same_v<TIN, int8_t>, vector_s16,
                                 std::conditional_t<std::is_same_v<TIN, uint8_t>, vector_u16, void>>;

    constexpr unsigned srcRowStride = TileDataIn::Cols;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TIN);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __ubuf__ TOUT *dstPtr = (__ubuf__ TOUT *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);
    __VEC_SCOPE__
    {
        vector_s16 vregIndexOldEven;
        vector_s16 vregIndexOldOdd;
        vector_s16 vregIndexOutput0;
        vector_s16 vregIndexOutput1;
        vector_s16 vregIndexNewEven;
        vector_s16 vregIndexNewOdd;
        RegTensor<TOUT> outputIndexEven;
        RegTensor<TOUT> outputIndexOdd;
        RegTensor<TOUT> outputIndex0;
        RegTensor<TOUT> outputIndex1;
        RegTensor<TIN> vregOld;
        RegTensor<TIN> vregNew;
        T vregOldEven;
        T vregOldOdd;
        T vregNewEven;
        T vregNewOdd;
        MaskReg preg = pset_b8(PAT_ALL);
        MaskReg selectEven;
        MaskReg selectOdd;
        MaskReg preg0;
        MaskReg preg1;
        MaskReg preg2;
        MaskReg preg3;
        uint32_t sreg = srcValidCol;

        for (uint16_t j = 0; j < repeatTimes; j++) {
            preg0 = plt_b32(sreg, POST_UPDATE);
            preg1 = plt_b32(sreg, POST_UPDATE);
            preg2 = plt_b32(sreg, POST_UPDATE);
            preg3 = plt_b32(sreg, POST_UPDATE);
            vdup(vregIndexOldEven, 0, preg, MODE_ZEROING);
            vdup(vregIndexOldOdd, 0, preg, MODE_ZEROING);
            vdup(vregIndexNewEven, 0, preg, MODE_ZEROING);
            vdup(vregIndexNewOdd, 0, preg, MODE_ZEROING);
            vlds(vregOld, srcPtr, j * elementsPerRepeat, NORM);
            vcvt(vregOldEven, vregOld, preg, PART_EVEN);
            vcvt(vregOldOdd, vregOld, preg, PART_ODD);

            for (uint16_t i = 1; i < (uint16_t)srcValidRow; i++) {
                vadds(vregIndexNewEven, vregIndexNewEven, 1, preg, MODE_ZEROING);
                vadds(vregIndexNewOdd, vregIndexNewOdd, 1, preg, MODE_ZEROING);
                vlds(vregNew, srcPtr, i * srcRowStride + j * elementsPerRepeat, NORM);
                vcvt(vregNewEven, vregNew, preg, PART_EVEN);
                vcvt(vregNewOdd, vregNew, preg, PART_ODD);
                if constexpr (IsArgMax) {
                    vcmp_gt(selectEven, vregNewEven, vregOldEven, preg);
                    vcmp_gt(selectOdd, vregNewOdd, vregOldOdd, preg);
                    vsel(vregIndexOldEven, vregIndexNewEven, vregIndexOldEven, selectEven);
                    vsel(vregIndexOldOdd, vregIndexNewOdd, vregIndexOldOdd, selectOdd);
                    vmax(vregOldEven, vregOldEven, vregNewEven, preg, MODE_ZEROING);
                    vmax(vregOldOdd, vregOldOdd, vregNewOdd, preg, MODE_ZEROING);
                } else {
                    vcmp_lt(selectEven, vregNewEven, vregOldEven, preg);
                    vcmp_lt(selectOdd, vregNewOdd, vregOldOdd, preg);
                    vsel(vregIndexOldEven, vregIndexNewEven, vregIndexOldEven, selectEven);
                    vsel(vregIndexOldOdd, vregIndexNewOdd, vregIndexOldOdd, selectOdd);
                    vmin(vregOldEven, vregOldEven, vregNewEven, preg, MODE_ZEROING);
                    vmin(vregOldOdd, vregOldOdd, vregNewOdd, preg, MODE_ZEROING);
                }
            }
            vintlv(vregIndexOutput0, vregIndexOutput1, vregIndexOldEven, vregIndexOldOdd);
            vcvt(outputIndexEven, vregIndexOutput0, preg, PART_EVEN);
            vcvt(outputIndexOdd, vregIndexOutput0, preg, PART_ODD);
            vintlv(outputIndex0, outputIndex1, outputIndexEven, outputIndexOdd);
            vsts(outputIndex0, dst, j * elementsPerRepeat, NORM_B32, preg0);
            vsts(outputIndex1, dst, j * elementsPerRepeat + ELE_CNT_B32, NORM_B32, preg1);

            vcvt(outputIndexEven, vregIndexOutput1, preg, PART_EVEN);
            vcvt(outputIndexOdd, vregIndexOutput1, preg, PART_ODD);
            vintlv(outputIndex0, outputIndex1, outputIndexEven, outputIndexOdd);
            vsts(outputIndex0, dst, j * elementsPerRepeat + 2 * ELE_CNT_B32, NORM_B32, preg2);
            vsts(outputIndex1, dst, j * elementsPerRepeat + 3 * ELE_CNT_B32, NORM_B32, preg3);
        }
    }
}

template <typename TileDataOut, typename TileDataIn, bool IsArgMax>
__tf__ PTO_INTERNAL void TColReduceIdx16(typename TileDataOut::TileDType __out__ dst,
                                         typename TileDataIn::TileDType __in__ src, unsigned srcValidRow,
                                         unsigned srcValidCol)
{
    using TIN = typename TileDataIn::DType;
    using TOUT = typename TileDataOut::DType;
    constexpr unsigned srcRowStride = TileDataIn::Cols;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TIN);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    __ubuf__ TOUT *dstPtr = (__ubuf__ TOUT *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

    __VEC_SCOPE__
    {
        vector_s16 vregIndexOld;
        vector_s16 vregIndexNew;
        RegTensor<TOUT> outputIndexEven;
        RegTensor<TOUT> outputIndexOdd;
        RegTensor<TOUT> outputIndex0;
        RegTensor<TOUT> outputIndex1;
        MaskReg preg = pset_b8(PAT_ALL);
        MaskReg preg0;
        MaskReg preg1;
        MaskReg select;
        RegTensor<TIN> vregOld;
        RegTensor<TIN> vregNew;
        uint32_t sreg = srcValidCol;

        for (uint16_t j = 0; j < repeatTimes; j++) {
            preg0 = plt_b32(sreg, POST_UPDATE);
            preg1 = plt_b32(sreg, POST_UPDATE);
            vdup(vregIndexOld, 0, preg, MODE_ZEROING);
            vdup(vregIndexNew, 0, preg, MODE_ZEROING);
            vlds(vregOld, srcPtr, j * elementsPerRepeat, NORM);
            for (uint16_t i = 1; i < (uint16_t)srcValidRow; i++) {
                vadds(vregIndexNew, vregIndexNew, 1, preg, MODE_ZEROING);
                vlds(vregNew, srcPtr, i * srcRowStride + j * elementsPerRepeat, NORM);
                if constexpr (IsArgMax) {
                    vcmp_gt(select, vregNew, vregOld, preg);
                    vsel(vregIndexOld, vregIndexNew, vregIndexOld, select);
                    vmax(vregOld, vregOld, vregNew, preg, MODE_ZEROING);
                } else {
                    vcmp_lt(select, vregNew, vregOld, preg);
                    vsel(vregIndexOld, vregIndexNew, vregIndexOld, select);
                    vmin(vregOld, vregOld, vregNew, preg, MODE_ZEROING);
                }
            }
            vcvt(outputIndexEven, vregIndexOld, preg, PART_EVEN);
            vcvt(outputIndexOdd, vregIndexOld, preg, PART_ODD);
            vintlv(outputIndex0, outputIndex1, outputIndexEven, outputIndexOdd);
            vsts(outputIndex0, dst, j * elementsPerRepeat, NORM_B32, preg0);
            vsts(outputIndex1, dst, j * elementsPerRepeat + ELE_CNT_B32, NORM_B32, preg1);
        }
    }
}

template <typename TileDataOut, typename TileDataIn, bool IsArgMax>
__tf__ PTO_INTERNAL void TColReduceIdx32(typename TileDataOut::TileDType __out__ dst,
                                         typename TileDataIn::TileDType __in__ src, unsigned srcValidRow,
                                         unsigned srcValidCol)
{
    using TIN = typename TileDataIn::DType;
    using TOUT = typename TileDataOut::DType;

    constexpr unsigned srcRowStride = TileDataIn::Cols;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TIN);
    __ubuf__ TOUT *dstPtr = (__ubuf__ TOUT *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<TOUT> vregIndexOld;
        RegTensor<TOUT> vregIndexNew;
        RegTensor<TIN> vregOld;
        RegTensor<TIN> vregNew;
        MaskReg pg;
        MaskReg select;
        uint32_t sreg = srcValidCol;

        for (uint16_t j = 0; j < repeatTimes; j++) {
            pg = plt_b32(sreg, POST_UPDATE);
            vdup(vregIndexOld, 0, pg, MODE_ZEROING);
            vdup(vregIndexNew, 0, pg, MODE_ZEROING);
            vlds(vregOld, srcPtr, j * elementsPerRepeat, NORM);
            for (uint16_t i = 1; i < (uint16_t)srcValidRow; i++) {
                vadds(vregIndexNew, vregIndexNew, (uint32_t)1, pg, MODE_ZEROING);
                vlds(vregNew, srcPtr, i * srcRowStride + j * elementsPerRepeat, NORM);
                if constexpr (IsArgMax) {
                    vcmp_gt(select, vregNew, vregOld, pg);
                    vsel(vregIndexOld, vregIndexNew, vregIndexOld, select);
                    vmax(vregOld, vregOld, vregNew, pg, MODE_ZEROING);
                } else {
                    vcmp_lt(select, vregNew, vregOld, pg);
                    vsel(vregIndexOld, vregIndexNew, vregIndexOld, select);
                    vmin(vregOld, vregOld, vregNew, pg, MODE_ZEROING);
                }
            }
            vsts(vregIndexOld, dstPtr, j * elementsPerRepeat, NORM_B32, pg);
        }
    }
}
template <typename TileDataOut, typename TileDataIn, bool IsArgMax>
PTO_INTERNAL void TCOLARG_DISPATCH(TileDataOut &dst, TileDataIn &src)
{
    unsigned srcValidRow = src.GetValidRow();
    unsigned srcValidCol = src.GetValidCol();
    TColReduceIdxCheck<TileDataOut, TileDataIn>(srcValidRow, srcValidCol, dst.GetValidRow(), dst.GetValidCol());

    if constexpr (sizeof(typename TileDataIn::DType) == 1) {
        TColReduceIdx8<TileDataOut, TileDataIn, IsArgMax>(dst.data(), src.data(), srcValidRow, srcValidCol);
    } else if (sizeof(typename TileDataIn::DType) == 2) {
        TColReduceIdx16<TileDataOut, TileDataIn, IsArgMax>(dst.data(), src.data(), srcValidRow, srcValidCol);
    } else if (sizeof(typename TileDataIn::DType) == 4) {
        TColReduceIdx32<TileDataOut, TileDataIn, IsArgMax>(dst.data(), src.data(), srcValidRow, srcValidCol);
    }
}
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TCOLARG_DISPATCH<TileDataOut, TileDataIn, false>(dst, src); // Min
}
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TCOLARG_DISPATCH<TileDataOut, TileDataIn, true>(dst, src); // Max
}
} // namespace pto
#endif