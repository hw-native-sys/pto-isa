/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP

#include <pto/common/constants.hpp>
#include <cassert>
#include "pto/cpu/parallel.hpp"
#include "common.hpp"
#include "nz_utils.hpp"

namespace pto {

template <typename GlobalData, typename TileData>
PTO_INLINE void CheckTileDataStore(GlobalData& dst, TileData& src)
{
    constexpr size_t C0 = C0_SIZE_BYTE / sizeof(typename GlobalData::DType);
    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        assert(
            src.GetValidRow() == dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) &&
            src.GetValidCol() == dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                                     dst.GetShape(GlobalTensorDim::DIM_4));

    } else {
        assert(
            dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) *
                dst.GetShape(GlobalTensorDim::DIM_4) * C0 >=
            src.GetValidRow() * src.GetValidCol());
    }
}

template <
    typename GlobalData, typename TileData, QuantMode_t quantMode, bool applyRelu,
    AtomicType atomicType = AtomicType::AtomicNone>
__tf__ PTO_INLINE void TStore(GlobalData& dst, TileData& src, const std::vector<uint64_t>& scalars)
{
    using DT = typename GlobalData::DType;
    using ST = typename TileData::DType;

    CheckTileDataStore(dst, src);

    const size_t validRow = src.GetValidRow();
    const size_t validCol = src.GetValidCol();

    const std::vector<int64_t> shapes = {
        dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> strides = {
        dst.GetStride(GlobalTensorDim::DIM_0), dst.GetStride(GlobalTensorDim::DIM_1),
        dst.GetStride(GlobalTensorDim::DIM_2), dst.GetStride(GlobalTensorDim::DIM_3),
        dst.GetStride(GlobalTensorDim::DIM_4)};

    uint64_t scalar = 0;
    for (size_t row = 0; row < validRow; ++row) {
        for (size_t col = 0; col < validCol; ++col) {
            if constexpr (quantMode != QuantMode_t::NoQuant) {
                scalar = scalars[TileData::isRowMajor ? col : row];
            }
            ST val = src.GetElement(row, col);
            DT dstVal = ConvertStoreValue<DT, ST, quantMode, applyRelu>(val, scalar);
            const size_t dstOffset = MapTileIndicesToGlobalOffset<GlobalData>(row, col, shapes, strides);
            if constexpr (atomicType == AtomicType::AtomicAdd) {
                dst.AddToElement(dstOffset, dstVal);
            } else {
                dst.SetElement(dstOffset, dstVal);
            }
        }
    }
}

template <typename GlobalData, typename ConTile, AtomicType atomicType = AtomicType::AtomicNone>
__tf__ PTO_INLINE void TStoreConv(GlobalData& dst, ConTile& src)
{
    using T = typename ConTile::DType;
    CheckConvTileData<ConTile, GlobalData>(src, dst);

    const size_t validRow = CalculateValidRowFromTile(src);
    const size_t validCol = CalculateValidColFromTile(src);

    const std::vector<int64_t> tile_shapes = {
        src.GetShape(GlobalTensorDim::DIM_0), src.GetShape(GlobalTensorDim::DIM_1),
        src.GetShape(GlobalTensorDim::DIM_2), src.GetShape(GlobalTensorDim::DIM_3),
        src.GetShape(GlobalTensorDim::DIM_4)};

    const std::vector<int64_t> shapes = {
        dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> strides = {
        dst.GetStride(GlobalTensorDim::DIM_0), dst.GetStride(GlobalTensorDim::DIM_1),
        dst.GetStride(GlobalTensorDim::DIM_2), dst.GetStride(GlobalTensorDim::DIM_3),
        dst.GetStride(GlobalTensorDim::DIM_4)};

    uint64_t scalar = 0;
    for (size_t row = 0; row < validRow; ++row) {
        for (size_t col = 0; col < validCol; ++col) {
            T val = src.data()[GetConvTileElementOffset<ConTile>(row, col, tile_shapes)];
            const size_t dstOffset = MapTileIndicesToGlobalOffset<GlobalData>(row, col, shapes, strides);
            if constexpr (atomicType == AtomicType::AtomicAdd) {
                dst.AddToElement(dstOffset, val);
            } else {
                dst.SetElement(dstOffset, val);
            }
        }
    }
}

template <
    typename TileData, typename GlobalData, QuantMode_t quantMode, bool applyRelu,
    AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, const std::vector<uint64_t>& scalars = {})
{
    static_assert(
        GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN ||
            GlobalData::layout == pto::Layout::NZ || GlobalData::layout == pto::Layout::NDC1HWC0 ||
            GlobalData::layout == pto::Layout::NC1HWC0,
        "Only ND, DN, NZ, NC1HWC0 and NDC1HWC0 GLobal Tensors are currently supported");
    if constexpr (is_conv_tile_v<TileData>) {
        TStoreConv<GlobalData, TileData, atomicType>(dst, src);
    } else {
        TStore<GlobalData, TileData, quantMode, applyRelu, atomicType>(dst, src, scalars);
    }
}

template <typename TileData, typename GlobalData, AtomicType atomicType, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    (void)Phase;
    TSTORE_IMPL<TileData, GlobalData, QuantMode_t::NoQuant, false, atomicType>(dst, src);
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    (void)Phase;
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    TSTORE_IMPL<TileData, GlobalData, QuantMode_t::NoQuant, useRelu, atomicType>(dst, src);
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src, uint64_t preQuantScalar)
{
    (void)Phase;
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    size_t vector_size = 0;
    if constexpr (TileData::isRowMajor) {
        vector_size = src.GetValidCol();
    } else {
        vector_size = src.GetValidRow();
    }
    std::vector<uint64_t> scalars(vector_size, preQuantScalar);
    TSTORE_IMPL<TileData, GlobalData, quantPre, useRelu, atomicType>(dst, src, scalars);
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src, FpTileData& fp)
{
    (void)Phase;
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;

    std::vector<uint64_t> scalars(fp.GetValidCol(), 0);
    for (size_t i = 0; i < fp.GetValidCol(); i++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIdx];
    }
    TSTORE_IMPL<TileData, GlobalData, quantPre, useRelu, atomicType>(dst, src, scalars);
}
} // namespace pto
#endif
