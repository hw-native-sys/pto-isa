/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
    using T = std::conditional_t<sizeof(typename TileDataDst::DType) == 1, int8_t,
                                 std::conditional_t<sizeof(typename TileDataDst::DType) == 2, int16_t, int32_t>>;
    if constexpr (sizeof(typename TileDataDst::DType) == 8) {
        ubPad = ubPad * 2;
    }

    copy_gm_to_ubuf_align(reinterpret_cast<__ubuf__ T *>(dst), reinterpret_cast<__gm__ T *>(src), 0, nBurst, lenBurst,
                          0, ubPad, gmGap, ubGap);
}

#include "pto/common/arch/memory/tload_common.hpp"

template <typename TileDataDst, typename GlobalDataSrc>
PTO_INTERNAL void CheckConvTileData(TileDataDst &dst, GlobalDataSrc &src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float!");
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

template <typename TileDataDst, typename GlobalDataSrc>
PTO_INTERNAL void TLOAD_IMPL(TileDataDst &dst, GlobalDataSrc &src)
{
    if constexpr (is_conv_tile_v<TileDataDst>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL<TileDataDst, GlobalDataSrc>(dst, src);
    }
}

} // namespace pto
#endif // TLOAD_HPP
