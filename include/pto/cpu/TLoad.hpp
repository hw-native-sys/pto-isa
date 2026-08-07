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

#include <unistd.h>
#include <cassert>
#include "pto/cpu/parallel.hpp"
#include "nz_utils.hpp"

namespace pto {
template <typename TileData>
AICORE constexpr typename TileData::DType getPadValue()
{
    switch (TileData::PadVal) {
        case PadValue::Null:
        case PadValue::Zero: {
            if constexpr (std::is_same_v<typename TileData::DType, int4b_t>) {
                return int4b_t(0);
            } else {
                return typename TileData::DType(0);
            }
        }
        case PadValue::Min:
            if constexpr (std::numeric_limits<typename TileData::DType>::has_infinity) {
                return -std::numeric_limits<typename TileData::DType>::infinity();
            } else {
                return std::numeric_limits<typename TileData::DType>::min();
            }
        case PadValue::Max:
            if constexpr (std::numeric_limits<typename TileData::DType>::has_infinity) {
                return std::numeric_limits<typename TileData::DType>::infinity();
            } else {
                return std::numeric_limits<typename TileData::DType>::max();
            }
    }
    if constexpr (std::is_same_v<typename TileData::DType, int4b_t>) {
        return int4b_t(0);
    } else {
        return 0;
    }
}

template <typename TileData, typename GlobalData>
PTO_INLINE void CheckTileData(TileData& dst, GlobalData& src)
{
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Source dtype must be same with dst dtype");
    static_assert(
        GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN ||
            GlobalData::layout == pto::Layout::NZ,
        "Only ND, DN and NZ GLobal Tensors are currently supported");

    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        assert(
            dst.GetValidRow() == src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) &&
            dst.GetValidCol() == src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                                     src.GetShape(GlobalTensorDim::DIM_4));
    } else {
        assert(
            (src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                     src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) ==
                 dst.GetValidRow() &&
             src.GetShape(GlobalTensorDim::DIM_4) == dst.GetValidCol() && TileData::isRowMajor) ||
            (src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                     src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_4) ==
                 dst.GetValidCol() &&
             src.GetShape(GlobalTensorDim::DIM_3) == dst.GetValidRow() && !TileData::isRowMajor));
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
        (GlobalData::layout == pto::Layout::NDC1HWC0 && TileData::layout == pto::Layout::NDC1HWC0);
    static_assert(
        isSameLayout == true, "Fix: Src and Dst layout must be the same in case of NC1HWC0, NDC1HWC0 or FRACTAL_Z!");

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (TileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
            static_assert(
                TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
                "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
            static_assert(
                GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
                "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        } else { //  [C1,H,W,N,C0]
            assert(
                src.GetShape(GlobalTensorDim::DIM_1) == dst.GetShape(GlobalTensorDim::DIM_1) &&
                src.GetShape(GlobalTensorDim::DIM_2) == dst.GetShape(GlobalTensorDim::DIM_2) &&
                "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
            assert(dst.GetShape(GlobalTensorDim::DIM_3) <= UINT16_MAX && "Fix: max support dstN is UINT16_MAX!");
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData& dst, GlobalData& src)
{
    CheckTileData<TileData, GlobalData>(dst, src);

    const size_t validRow = dst.GetValidRow();
    const size_t validCol = dst.GetValidCol();

    // Filling padding
    std::fill(dst.data(), dst.data() + TileData::GetSizeInUnits(), getPadValue<TileData>());

    const std::vector<int64_t> shapes = {
        src.GetShape(GlobalTensorDim::DIM_0), src.GetShape(GlobalTensorDim::DIM_1),
        src.GetShape(GlobalTensorDim::DIM_2), src.GetShape(GlobalTensorDim::DIM_3),
        src.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> strides = {
        src.GetStride(GlobalTensorDim::DIM_0), src.GetStride(GlobalTensorDim::DIM_1),
        src.GetStride(GlobalTensorDim::DIM_2), src.GetStride(GlobalTensorDim::DIM_3),
        src.GetStride(GlobalTensorDim::DIM_4)};

    for (size_t row = 0; row < validRow; ++row) {
        for (size_t col = 0; col < validCol; ++col) {
            const size_t dstOffset = MapTileIndicesToGlobalOffset<GlobalData>(row, col, shapes, strides);
            dst.SetElement(row, col, GetProperDataPart(src.data(), dstOffset));
        }
    }
}

template <typename ConTile, typename GlobalData>
__tf__ PTO_INLINE void TLOAD_CONVTILE_IMPL(ConTile& dst, GlobalData& src)
{
    CheckConvTileData<ConTile, GlobalData>(dst, src);

    using T = typename ConTile::DType;
    const size_t validRow = CalculateValidRowFromTile(dst);
    const size_t validCol = CalculateValidColFromTile(dst);

    const std::vector<int64_t> tile_shapes = {
        dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4)};

    const std::vector<int64_t> shapes = {
        src.GetShape(GlobalTensorDim::DIM_0), src.GetShape(GlobalTensorDim::DIM_1),
        src.GetShape(GlobalTensorDim::DIM_2), src.GetShape(GlobalTensorDim::DIM_3),
        src.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> strides = {
        src.GetStride(GlobalTensorDim::DIM_0), src.GetStride(GlobalTensorDim::DIM_1),
        src.GetStride(GlobalTensorDim::DIM_2), src.GetStride(GlobalTensorDim::DIM_3),
        src.GetStride(GlobalTensorDim::DIM_4)};

    for (size_t row = 0; row < validRow; ++row) {
        for (size_t col = 0; col < validCol; ++col) {
            const size_t srcOffset = MapTileIndicesToGlobalOffset<GlobalData>(row, col, shapes, strides);
            dst.data()[GetConvTileElementOffset<ConTile>(row, col, tile_shapes)] = src.data()[srcOffset];
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData& dst, GlobalData& src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}

} // namespace pto
#endif
