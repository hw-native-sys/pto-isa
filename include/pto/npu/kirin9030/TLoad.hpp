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
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "pto/common/arch/register/tload_common.hpp"

namespace pto {
struct Kirin9030LoadOp : LoadOpBase {
    using LoadOpBase::TLoadCubeInstr;
    template <Layout Layout = Layout::ND, typename T>
    PTO_INTERNAL static void TLoadCubeInstr(
        __cbuf__ T* dst, __gm__ T* src, uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue,
        uint64_t loop4SrcStride)
    {
        if constexpr (Layout == Layout::ND) {
            pto_copy_gm_to_cbuf_multi_nd2nz(dst, src, 0 /*sid*/, loop1SrcStride, 0, nValue, dValue, loop4SrcStride);
        } else {
            static_assert(sizeof(T) == 0, "Fix: TLoad does not support DN2NZ.");
        }
    }
};

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData& dst, GlobalData& src)
{
    StaticCheck<TileData, GlobalData>();
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        TLoad<Kirin9030LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Mat) {
        static_assert(!IsScale<TileData, GlobalData>(), "Fix: TLOAD not supported Mx cube.");
        TLoadCubeCheck<TileData, GlobalData>();
        TLoadCube<Kirin9030LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckConvTileData(TileData& dst, GlobalData& src)
{
    static_assert(
        caps::IsInt8<typename TileData::DType>() || caps::IsInt16<typename TileData::DType>() ||
            caps::IsInt32<typename TileData::DType>() || caps::IsFP16<typename TileData::DType>() ||
            caps::IsFP32<typename TileData::DType>(),
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z);
    static_assert(isSameLayout == true, "Fix: Src Dst layout must be NC1HWC0 or FRACTAL_Z!");
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(DstTile& dst, SrcGlobal& src)
{
    CheckConvTileData<DstTile, SrcGlobal>(dst, src);
    if constexpr (SrcGlobal::layout == pto::Layout::NC1HWC0) { // layout is [N,C1,H,W,C0]
        TLoad5HD<Kirin9030LoadOp, DstTile, SrcGlobal>(
            dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
            src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (SrcGlobal::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (DstTile::totalDimCount == 4) { // layout is [C1HW,N/16,16,C0]
            TLoadFractalZ<Kirin9030LoadOp, DstTile, SrcGlobal>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
                src.GetStride(4), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
        } else { // layout is [C1,H,W,N,C0]
            TLoad5HD<Kirin9030LoadOp, DstTile, SrcGlobal>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4),
                dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
        }
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLOAD_IMPL(DstTile& dst, SrcGlobal& src)
{
    if constexpr (is_conv_tile_v<DstTile>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}
} // namespace pto
#endif // TLOAD_HPP
