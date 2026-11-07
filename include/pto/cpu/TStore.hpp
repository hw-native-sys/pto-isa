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
    constexpr size_t C0 = GetC0ElemCount<typename GlobalData::DType, TileData>();
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
                scalar = scalars[col];
            }
            ST val = src.GetElement(row, col);
            DT dstVal = ConvertStoreValue<DT, ST, quantMode, applyRelu>(val, scalar);
            const size_t dstOffset = MapTransferIndicesToGlobalOffset<GlobalData, TileData>(row, col, shapes, strides);
            if constexpr (atomicType == AtomicType::AtomicAdd) {
                dst.AddToElement(dstOffset, dstVal);
            } else {
                dst.SetElement(dstOffset, dstVal);
            }
        }
    }
}

// ConvTile TSTORE writes only the source valid box (each dimension starts at the logical origin);
// GM outside it is left untouched. The tile and the GlobalTensor are packed with their own declared
// shapes, mirroring the TLOAD valid-box implementation.
template <typename GlobalData, typename ConTile, AtomicType atomicType>
PTO_INLINE void TStoreConvTileElement(
    GlobalData& dst, ConTile& src, int64_t srcRow, int64_t srcCol, int64_t dstRow, int64_t dstCol,
    const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& dstShapes,
    const std::vector<int64_t>& dstStrides)
{
    using T = typename ConTile::DType;
    const T val = src.GetElement(GetConvTileElementOffset<ConTile>(srcRow, srcCol, tile_shapes));
    const size_t dstOffset = MapTileIndicesToGlobalOffset<GlobalData>(dstRow, dstCol, dstShapes, dstStrides);
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        dst.AddToElement(dstOffset, val);
    } else {
        dst.SetElement(dstOffset, val);
    }
}

template <typename GlobalData, typename ConTile, AtomicType atomicType>
PTO_INLINE void TStoreConvTileNC1HWC0(
    GlobalData& dst, ConTile& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& dstShapes,
    const std::vector<int64_t>& dstStrides)
{
    using T = typename ConTile::DType;
    constexpr int64_t c0 = static_cast<int64_t>(GetC0ElemCount<T>());

    const int64_t tH = src.GetShape(GlobalTensorDim::DIM_2);
    const int64_t tW = src.GetShape(GlobalTensorDim::DIM_3);
    const int64_t gH = dst.GetShape(GlobalTensorDim::DIM_2);
    const int64_t gW = dst.GetShape(GlobalTensorDim::DIM_3);
    const int64_t gC0 = dst.GetShape(GlobalTensorDim::DIM_4);
    const int64_t vN = src.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t vC1 = src.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t vH = src.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t vW = src.GetValidShape(GlobalTensorDim::DIM_3);
    int64_t vC0 = c0;
    if constexpr (ConTile::totalDimCount >= 5) {
        vC0 = src.GetValidShape(GlobalTensorDim::DIM_4);
    }
    assert(vC0 <= c0 && "ConvTile valid C0 must not exceed the 32-byte C0 block!");
    assert(gC0 == c0 && "NC1HWC0 GlobalTensor C0 must equal the 32-byte C0 block!");

    for (int64_t n = 0; n < vN; ++n) {
        for (int64_t c1 = 0; c1 < vC1; ++c1) {
            for (int64_t h = 0; h < vH; ++h) {
                for (int64_t w = 0; w < vW; ++w) {
                    const int64_t tileRow = (n * tH + h) * tW + w;
                    const int64_t dstRow = (n * gH + h) * gW + w;
                    for (int64_t c = 0; c < vC0; ++c) {
                        TStoreConvTileElement<GlobalData, ConTile, atomicType>(
                            dst, src, tileRow, c1 * c0 + c, dstRow, c1 * gC0 + c, tile_shapes, dstShapes, dstStrides);
                    }
                }
            }
        }
    }
}

template <typename GlobalData, typename ConTile, AtomicType atomicType>
PTO_INLINE void TStoreConvTileNDC1HWC0(
    GlobalData& dst, ConTile& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& dstShapes,
    const std::vector<int64_t>& dstStrides)
{
    using T = typename ConTile::DType;
    constexpr int64_t c0 = static_cast<int64_t>(GetC0ElemCount<T>());

    const int64_t tC1 = src.GetShape(GlobalTensorDim::DIM_2);
    const int64_t tH = src.GetShape(GlobalTensorDim::DIM_3);
    const int64_t tW = src.GetShape(GlobalTensorDim::DIM_4);
    const int64_t gC1 = dst.GetShape(GlobalTensorDim::DIM_2);
    const int64_t gH = dst.GetShape(GlobalTensorDim::DIM_3);
    const int64_t gW = dst.GetShape(GlobalTensorDim::DIM_4);
    const int64_t vN = src.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t vD = src.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t vC1 = src.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t vH = src.GetValidShape(GlobalTensorDim::DIM_3);
    const int64_t vW = src.GetValidShape(GlobalTensorDim::DIM_4);
    int64_t vC0 = c0;
    if constexpr (ConTile::totalDimCount >= 6) {
        vC0 = src.GetValidShape(5);
    }
    assert(vC0 <= c0 && "ConvTile valid C0 must not exceed the 32-byte C0 block!");

    for (int64_t n = 0; n < vN; ++n) {
        for (int64_t d = 0; d < vD; ++d) {
            for (int64_t c1 = 0; c1 < vC1; ++c1) {
                for (int64_t h = 0; h < vH; ++h) {
                    for (int64_t w = 0; w < vW; ++w) {
                        const int64_t tileRow = (n * tH + h) * tW + w;
                        const int64_t dstRow = (n * gH + h) * gW + w;
                        const int64_t tileColBase = (d * tC1 + c1) * c0;
                        const int64_t dstColBase = (d * gC1 + c1) * c0;
                        for (int64_t c = 0; c < vC0; ++c) {
                            TStoreConvTileElement<GlobalData, ConTile, atomicType>(
                                dst, src, tileRow, tileColBase + c, dstRow, dstColBase + c, tile_shapes, dstShapes,
                                dstStrides);
                        }
                    }
                }
            }
        }
    }
}

template <typename GlobalData, typename ConTile, AtomicType atomicType>
PTO_INLINE void TStoreConvTileFractalZ4D(
    GlobalData& dst, ConTile& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& dstShapes,
    const std::vector<int64_t>& dstStrides)
{
    const int64_t d1 = src.GetShape(GlobalTensorDim::DIM_1);
    const int64_t d3 = src.GetShape(GlobalTensorDim::DIM_3);
    const int64_t v0 = src.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t v1 = src.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t v2 = src.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t v3 = src.GetValidShape(GlobalTensorDim::DIM_3);

    for (int64_t i0 = 0; i0 < v0; ++i0) {
        for (int64_t i1 = 0; i1 < v1; ++i1) {
            const int64_t rowBase = (i0 * d1 + i1) * d3;
            for (int64_t i2 = 0; i2 < v2; ++i2) {
                for (int64_t i3 = 0; i3 < v3; ++i3) {
                    TStoreConvTileElement<GlobalData, ConTile, atomicType>(
                        dst, src, rowBase + i3, i2, rowBase + i3, i2, tile_shapes, dstShapes, dstStrides);
                }
            }
        }
    }
}

template <typename GlobalData, typename ConTile, AtomicType atomicType>
PTO_INLINE void TStoreConvTileFractalZ5D(
    GlobalData& dst, ConTile& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& dstShapes,
    const std::vector<int64_t>& dstStrides)
{
    const int64_t tH = src.GetShape(GlobalTensorDim::DIM_1);
    const int64_t tW = src.GetShape(GlobalTensorDim::DIM_2);
    const int64_t tC0 = src.GetShape(GlobalTensorDim::DIM_4);
    const int64_t gH = dst.GetShape(GlobalTensorDim::DIM_1);
    const int64_t gW = dst.GetShape(GlobalTensorDim::DIM_2);
    const int64_t gC0 = dst.GetShape(GlobalTensorDim::DIM_4);
    const int64_t vC1 = src.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t vH = src.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t vW = src.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t vN = src.GetValidShape(GlobalTensorDim::DIM_3);
    const int64_t vC0 = src.GetValidShape(GlobalTensorDim::DIM_4);

    for (int64_t c1 = 0; c1 < vC1; ++c1) {
        for (int64_t h = 0; h < vH; ++h) {
            for (int64_t w = 0; w < vW; ++w) {
                const int64_t tileRowBase = ((c1 * tH + h) * tW + w) * tC0;
                const int64_t dstRowBase = ((c1 * gH + h) * gW + w) * gC0;
                for (int64_t n = 0; n < vN; ++n) {
                    for (int64_t c = 0; c < vC0; ++c) {
                        TStoreConvTileElement<GlobalData, ConTile, atomicType>(
                            dst, src, tileRowBase + c, n, dstRowBase + c, n, tile_shapes, dstShapes, dstStrides);
                    }
                }
            }
        }
    }
}

template <typename GlobalData, typename ConTile, AtomicType atomicType = AtomicType::AtomicNone>
__tf__ PTO_INLINE void TStoreConv(GlobalData& dst, ConTile& src)
{
    CheckConvTileData<ConTile, GlobalData>(src, dst);
    CheckConvTileValidShape(src);

    const std::vector<int64_t> tile_shapes = {
        src.GetShape(GlobalTensorDim::DIM_0), src.GetShape(GlobalTensorDim::DIM_1),
        src.GetShape(GlobalTensorDim::DIM_2), src.GetShape(GlobalTensorDim::DIM_3),
        src.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> dstShapes = {
        dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> dstStrides = {
        dst.GetStride(GlobalTensorDim::DIM_0), dst.GetStride(GlobalTensorDim::DIM_1),
        dst.GetStride(GlobalTensorDim::DIM_2), dst.GetStride(GlobalTensorDim::DIM_3),
        dst.GetStride(GlobalTensorDim::DIM_4)};

    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        TStoreConvTileNC1HWC0<GlobalData, ConTile, atomicType>(dst, src, tile_shapes, dstShapes, dstStrides);
    } else if constexpr (GlobalData::layout == pto::Layout::NDC1HWC0) {
        TStoreConvTileNDC1HWC0<GlobalData, ConTile, atomicType>(dst, src, tile_shapes, dstShapes, dstStrides);
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (ConTile::totalDimCount == 4) {
            TStoreConvTileFractalZ4D<GlobalData, ConTile, atomicType>(dst, src, tile_shapes, dstShapes, dstStrides);
        } else {
            TStoreConvTileFractalZ5D<GlobalData, ConTile, atomicType>(dst, src, tile_shapes, dstShapes, dstStrides);
        }
    }
}

template <
    typename TileData, typename GlobalData, QuantMode_t quantMode, bool applyRelu,
    AtomicType atomicType = AtomicType::AtomicNone, TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, const std::vector<uint64_t>& scalars = {})
{
    if constexpr (is_conv_tile_v<TileData>) {
        // ConvTile stores only write the valid box, so every layout TStoreConv expands is accepted here.
        static_assert(
            GlobalData::layout == pto::Layout::NC1HWC0 || GlobalData::layout == pto::Layout::NDC1HWC0 ||
                GlobalData::layout == pto::Layout::FRACTAL_Z,
            "ConvTile TSTORE only supports NC1HWC0, NDC1HWC0 and FRACTAL_Z Global Tensors");
        TStoreConv<GlobalData, TileData, atomicType>(dst, src);
    } else {
        static_assert(
            GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN ||
                GlobalData::layout == pto::Layout::NZ || GlobalData::layout == pto::Layout::NDC1HWC0 ||
                GlobalData::layout == pto::Layout::NC1HWC0,
            "Only ND, DN, NZ, NC1HWC0 and NDC1HWC0 GLobal Tensors are currently supported");
        TStore<GlobalData, TileData, quantMode, applyRelu, atomicType>(dst, src, scalars);
    }
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType, STPhase Phase = STPhase::Unspecified,
    TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    (void)Phase;
    TSTORE_IMPL<TileData, GlobalData, QuantMode_t::NoQuant, false, atomicType>(dst, src);
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified, TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    (void)Phase;
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    TSTORE_IMPL<TileData, GlobalData, QuantMode_t::NoQuant, useRelu, atomicType>(dst, src);
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified, TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src, uint64_t preQuantScalar)
{
    (void)Phase;
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    std::vector<uint64_t> scalars(src.GetValidCol(), preQuantScalar);
    TSTORE_IMPL<TileData, GlobalData, quantPre, useRelu, atomicType>(dst, src, scalars);
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified, TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
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
