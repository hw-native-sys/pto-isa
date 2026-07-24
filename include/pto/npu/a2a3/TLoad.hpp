/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP
#define TLOAD_HPP

namespace pto {

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckNzFormat(
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int validRow, int validCol);

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2ub(
    __ubuf__ typename TileData::DType* dst, typename GlobalData::DType* src, uint16_t nBurst, uint32_t lenBurst,
    uint32_t gmGap, uint32_t ubGap, uint32_t ubPad)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad * 2, gmGap, ubGap);
    }
}

#include <pto/common/utils.hpp>

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2L1(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, uint16_t nBurst, uint16_t lenBurst,
    uint16_t gmGap, uint16_t l1Gap)
{
    pto_copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, gmGap, l1Gap);
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNDC1HWC0(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcN, int srcD, int srcC1,
    int srcH, int srcW, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int dstN, int dstD,
    int dstC1, int dstH, int dstW)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    constexpr uint32_t maxSupportBurst = 4095;
    uint32_t gmGap = ((gStride2 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride3 == dstW * c0ElemCount || dstH == 1) && gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst &&
        dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint16_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t srcAddr1 = i * gStride0;
            int64_t dstAddr1 = i * dstD * dstH * dstW * dstC1 * c0ElemCount;
            for (uint32_t j = 0; j < dstD; j++) {
                srcAddrP = srcAddr + srcAddr1 + j * gStride1;
                dstAddrP = dstAddr + dstAddr1 + j * dstH * dstW * dstC1 * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, 0);
            }
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint16_t srcGap = ((gStride3 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t srcAddr1 = i * gStride0;
            int64_t dstAddr1 = i * dstD * dstH * dstW * dstC1 * c0ElemCount;
            for (uint32_t j = 0; j < dstD; j++) {
                int64_t srcAddr2 = j * gStride1;
                int64_t dstAddr2 = j * dstH * dstW * dstC1 * c0ElemCount;
                for (uint32_t k = 0; k < dstC1; k++) {
                    srcAddrP = srcAddr + srcAddr1 + srcAddr2 + k * gStride2;
                    dstAddrP = dstAddr + dstAddr1 + dstAddr2 + k * dstH * dstW * c0ElemCount;
                    TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, l1Gap);
                }
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nd2nd(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(
        gShape4 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
        "The 5th dim of ND shape must be 32 bytes aligned!");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(
        validRow == gShape0 * gShape1 * gShape2 * gShape3,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape3;
    uint16_t lenBurst = (validCol * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride3 - gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Cols - validCol) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * gStride0;
        int64_t dstAddr0 = i * dstStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t dstAddr1 = j * dstStride1;
            int64_t srcAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Dn2dn(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(
        gShape3 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
        "The 4th dim of DN shape must be 32 bytes aligned!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape2 * gShape4,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape4;
    uint16_t lenBurst = (validRow * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride4 - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Rows - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    typename GlobalData::DType* srcAddrP = srcAddr;

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * gStride1;
            int64_t dstAddr1 = j * dstStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nz2nz(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    CheckNzFormat<TileData, GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, validRow, validCol);
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow;
    uint32_t gmGap = ((gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t l1Gap = TileData::Rows - validRow;
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    int64_t tileStride = TileData::Rows * gShape1 * gShape4;

    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + i * gStride0;
        dstAddrP = dstAddr + i * tileStride;
        TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoad5HD(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcN, int srcC1, int srcH,
    int srcW, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int dstN, int dstC1, int dstH,
    int dstW)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    constexpr uint32_t maxSupportBurst = 4095;
    uint32_t gmGap = ((gStride1 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride2 == dstW * c0ElemCount || dstH == 1) && gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst &&
        dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint16_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * dstH * dstW * dstC1 * c0ElemCount;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, 0);
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint16_t srcGap = ((gStride2 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t dstAddr1 = i * dstH * dstW * dstC1 * c0ElemCount;
            int64_t srcAddr1 = i * gStride0;
            for (uint32_t j = 0; j < dstC1; j++) {
                srcAddrP = srcAddr + srcAddr1 + j * gStride1;
                dstAddrP = dstAddr + dstAddr1 + j * dstH * dstW * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadFractalZ(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcShape0, int srcShape1,
    int srcShape2, int srcShape3, int srcShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;

    if constexpr (TileData::totalDimCount == 4) {
        static_assert(
            TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
            "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        static_assert(
            GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
            "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

        uint16_t nBurst = dstShape0;
        uint16_t lenBurst = dstShape1 * dstShape2;
        uint16_t gmGap =
            ((gStride1 - srcShape2 * srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);

    } else {
        PTO_ASSERT(
            srcShape1 == dstShape1 && srcShape2 == dstShape2,
            "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
        PTO_ASSERT(dstShape3 <= UINT16_MAX, "Fix: max support dstN is UINT16_MAX!");

        uint16_t lenBurst = dstShape3;
        uint16_t gmGap = ((gStride2 - srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        constexpr uint32_t maxSupportBurst = 4095;

        if (dstShape0 * dstShape1 * dstShape2 <= maxSupportBurst) {
            uint16_t nBurst = dstShape0 * dstShape1 * dstShape2;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
        } else {
            uint16_t nBurst = dstShape1 * dstShape2;
            for (uint32_t i = 0; i < dstShape0; i++) {
                srcAddrP = srcAddr + i * gStride0;
                dstAddrP = dstAddr + i * dstShape1 * dstShape2 * dstShape3 * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckConvTileData(TileData& dst, GlobalData& src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z_3D && TileData::layout == pto::Layout::FRACTAL_Z_3D) ||
        (GlobalData::layout == pto::Layout::NDC1HWC0 && TileData::layout == pto::Layout::NDC1HWC0);
    static_assert(
        isSameLayout == true, "Fix: Src Dst layout must be NC1HWC0 or FRACTAL_Z or FRACTAL_Z_3D or NDC1HWC0!");
}

#include "pto/common/arch/memory/tload_common.hpp"

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData& dst, GlobalData& src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL<TileData, GlobalData>(dst, src);
    }
}

} // namespace pto
#endif // TLOAD_HPP
