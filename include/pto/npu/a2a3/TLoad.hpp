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
PTO_INTERNAL void TLoadInstrGm2ub(__ubuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                  uint16_t nBurst, uint32_t lenBurst, uint32_t gmGap, uint32_t ubGap, uint32_t ubPad)
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

#include "pto/common/arch/memory/tload_common.hpp"

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNDC1HWC0(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                       int srcN, int srcD, int srcC1, int srcH, int srcW, int gStride0, int gStride1,
                                       int gStride2, int gStride3, int gStride4, int dstN, int dstD, int dstC1,
                                       int dstH, int dstW)
{
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    constexpr uint32_t maxSupportBurst = 4095;
    // gmGap unit is 32B
    uint32_t gmGap = ((gStride2 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride3 == dstW * c0ElemCount || dstH == 1) && // process for W direction all load or H=1
        gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst && dstH * dstW <= UINT16_MAX) {
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
PTO_INTERNAL void CheckConvTileData(TileData &dst, GlobalData &src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z_3D && TileData::layout == pto::Layout::FRACTAL_Z_3D) ||
        (GlobalData::layout == pto::Layout::NDC1HWC0 && TileData::layout == pto::Layout::NDC1HWC0);
    static_assert(isSameLayout == true,
                  "Fix: Src Dst layout must be NC1HWC0 or FRACTAL_Z or FRACTAL_Z_3D or NDC1HWC0!");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(TileData &dst, GlobalData &src)
{
    CheckConvTileData<TileData, GlobalData>(dst, src);
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) { // layout is NC1HWC0, dst dim4 is c0Size
        TLoad5HD<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                       src.GetShape(3), src.GetStride(0), src.GetStride(1), src.GetStride(2),
                                       src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1),
                                       dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z ||
                         GlobalData::layout == pto::Layout::FRACTAL_Z_3D) {
        // [C1HW,N/16,16,C0] or [C1DHW,N/16,16,C0], dst dim4 is c0Size
        TLoadFractalZ<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                            src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1),
                                            src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::NDC1HWC0) { // NDC1HWC0, globaltensor is NDC1HW
        TLoadNDC1HWC0<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                            src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1),
                                            src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3), dst.GetShape(4));
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}
} // namespace pto
#endif // TLOAD_HPP
