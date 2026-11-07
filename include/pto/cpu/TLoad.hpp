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

#include <pto/cpu/target_config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unistd.h>

#include <pto/common/constants.hpp>
#include <pto/cpu/NPUMemoryModel.hpp>
#include "pto/cpu/parallel.hpp"

namespace pto {
PTO_CPU_TARGET_NAMESPACE_BEGIN
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
            GlobalData::layout == pto::Layout::NZ || GlobalData::layout == pto::Layout::MX_A_ND ||
            GlobalData::layout == pto::Layout::MX_A_DN || GlobalData::layout == pto::Layout::MX_A_ZZ ||
            GlobalData::layout == pto::Layout::MX_B_ND || GlobalData::layout == pto::Layout::MX_B_DN ||
            GlobalData::layout == pto::Layout::MX_B_NN,
        "Only ND, DN, NZ and MX_* GLobal Tensors are currently supported");

    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        if constexpr (
            TileData::Loc == TileType::Vec && !TileData::isRowMajor && TileData::SFractal == SLayout::RowMajor) {
            // A5 NZ-to-UB DMA takes its row count from the tile and column blocks from GM.
            if (GetCpuSimArch() == NPUArch::A5) {
                [[maybe_unused]] constexpr size_t C0_ELEMENTS =
                    C0_SIZE_BYTE / sizeof(typename TileData::DType) * (IsTwinType<typename TileData::DType>() ? 2 : 1);
                assert(
                    GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == C0_ELEMENTS &&
                    "A5 NZ-to-UB requires a static 16-row, 32-byte fractal");
                return;
            }
        }
        assert(
            dst.GetValidRow() == src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) &&
            dst.GetValidCol() == src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                                     src.GetShape(GlobalTensorDim::DIM_4));
    } else if constexpr (GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN) {
        [[maybe_unused]] const bool rowsMerged =
            src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                    src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) ==
                dst.GetValidRow() &&
            src.GetShape(GlobalTensorDim::DIM_4) == dst.GetValidCol();
        [[maybe_unused]] const bool colsMerged =
            src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                    src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_4) ==
                dst.GetValidCol() &&
            src.GetShape(GlobalTensorDim::DIM_3) == dst.GetValidRow();

        // An ND source always merges DIM_0..DIM_3 into rows, including ND2NZ into a Mat tile that is
        // not row major: there DIM_2 is the nd-matrix count, so the tile has DIM_2 * DIM_3 rows.
        [[maybe_unused]] const bool ndIntoNzTile = GlobalData::layout == pto::Layout::ND && rowsMerged;
        [[maybe_unused]] bool partialMultiNd = false;
        constexpr bool ndToNz =
            GlobalData::layout == Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ;
        constexpr bool dnToZn =
            GlobalData::layout == Layout::DN && TileData::isRowMajor && TileData::SFractal == SLayout::ColMajor;
        constexpr bool validDnToZnDescriptor = GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 &&
                                               TileData::SFractalSize == 512 && TileData::Cols <= 65535 &&
                                               !IsTwinType<typename TileData::DType>();
        if constexpr (
            (ndToNz || (dnToZn && validDnToZnDescriptor)) && GlobalData::staticShape[2] != 1 &&
            TileData::Loc == TileType::Mat && sizeof(typename TileData::DType) <= 4) {
            const int64_t matrixRows = src.GetShape(dnToZn ? GlobalTensorDim::DIM_4 : GlobalTensorDim::DIM_3);
            const int64_t matrixCols = src.GetShape(dnToZn ? GlobalTensorDim::DIM_3 : GlobalTensorDim::DIM_4);
            const int validRows = dnToZn ? dst.GetValidCol() : dst.GetValidRow();
            const int validCols = dnToZn ? dst.GetValidRow() : dst.GetValidCol();
            partialMultiNd = GetCpuSimArch() == NPUArch::A5 &&
                             (!dnToZn || src.GetStride(GlobalTensorDim::DIM_3) == 1) &&
                             src.GetShape(GlobalTensorDim::DIM_0) == 1 && src.GetShape(GlobalTensorDim::DIM_1) == 1 &&
                             src.GetShape(GlobalTensorDim::DIM_2) > 0 &&
                             src.GetShape(GlobalTensorDim::DIM_2) <= 65535 && matrixRows > 0 && matrixRows <= 16384 &&
                             validRows > 0 && validRows <= (dnToZn ? TileData::Cols : TileData::Rows) &&
                             validRows <= src.GetShape(GlobalTensorDim::DIM_2) * matrixRows && validCols > 0 &&
                             validCols <= (dnToZn ? TileData::Rows : TileData::Cols) && validCols <= matrixCols;
        }
        if constexpr (dnToZn && GlobalData::staticShape[2] != 1 && TileData::Loc == TileType::Mat) {
            // The row-merged fallback does not describe a DN source loaded into a ZN tile.
            assert(partialMultiNd);
            return;
        }
        assert(
            (rowsMerged && TileData::isRowMajor) || (colsMerged && !TileData::isRowMajor) || ndIntoNzTile ||
            partialMultiNd);
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
            std::is_same_v<typename TileData::DType, float> || IsTwinType<typename TileData::DType>(),
        "Fix: Data type must be "
        "int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/"
        "float4_e1m2x2_t/float4_e2m1x2_t/half/bfloat16_t/float!");
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

template <typename TileData, typename T>
PTO_INTERNAL void FillTLoadRowTail(TileData& dst, size_t validRow, size_t validCol, size_t padCols, T padVal)
{
    if (validRow != 0 && padCols != 0) {
        const size_t lastElement = GetTileElementOffset<TileData>(validRow - 1, validCol + padCols - 1);
        const size_t spanBytes =
            (lastElement / TileData::kPackedElementsPerByte + 1) * sizeof(typename TileData::DType);
        NPUMemoryModel::Instance().CheckAccess(dst.data(), 0, spanBytes);
    }
    for (size_t row = 0; row < validRow; ++row) {
        for (size_t col = validCol; col < validCol + padCols; ++col) {
            dst.SetElement(row, col, padVal);
        }
    }
}

template <typename TileData, typename T>
PTO_INTERNAL void FillTLoadColTail(TileData& dst, size_t validRow, size_t validCol, size_t padRows, T padVal)
{
    if (validCol != 0 && padRows != 0) {
        const size_t lastElement = GetTileElementOffset<TileData>(validRow + padRows - 1, validCol - 1);
        const size_t spanBytes =
            (lastElement / TileData::kPackedElementsPerByte + 1) * sizeof(typename TileData::DType);
        NPUMemoryModel::Instance().CheckAccess(dst.data(), 0, spanBytes);
    }
    for (size_t col = 0; col < validCol; ++col) {
        for (size_t row = validRow; row < validRow + padRows; ++row) {
            dst.SetElement(row, col, padVal);
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void FillTLoadPadding(TileData& dst, size_t validRow, size_t validCol)
{
    constexpr size_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr auto tileLayout = GetTileLayoutCustom<TileData>();

    if constexpr (TileData::Loc == TileType::Vec) {
        // A2/A3 and A5 GM-to-UB DMA pad only the sub-32-byte tail. Full-block gaps are skipped.
        if constexpr (TileData::PadVal == PadValue::Null || GlobalData::layout == Layout::NZ) {
            return;
        } else if constexpr (TileData::isRowMajor) {
            const size_t padCols = (TileData::Cols - validCol) % blockSizeElem;
            FillTLoadRowTail(dst, validRow, validCol, padCols, getPadValue<TileData>());
        } else {
            const size_t padRows = (TileData::Rows - validRow) % blockSizeElem;
            FillTLoadColTail(dst, validRow, validCol, padRows, getPadValue<TileData>());
        }
        return;
    }

    if constexpr (TileData::Loc == TileType::Mat) {
        // ND/DN-to-fractal DMA always zero-fills the final partial C0 block.
        constexpr bool convertsToNz = tileLayout == TileLayoutCustom::NZ &&
                                      (GlobalData::layout == Layout::ND || GlobalData::layout == Layout::DN);
        constexpr bool convertsToZn = tileLayout == TileLayoutCustom::ZN && GlobalData::layout == Layout::DN;
        if constexpr (convertsToNz) {
            const size_t padCols = (TileData::Cols - validCol) % blockSizeElem;
            FillTLoadRowTail(dst, validRow, validCol, padCols, typename TileData::DType(0));
            return;
        } else if constexpr (convertsToZn) {
            const size_t padRows = (TileData::Rows - validRow) % blockSizeElem;
            FillTLoadColTail(dst, validRow, validCol, padRows, typename TileData::DType(0));
            return;
        }

        // A2/A3 uses ND-to-NZ DMA for the single-row/single-column Mat special case.
        if (GetCpuSimArch() == NPUArch::A2A3) {
            if constexpr (
                tileLayout == TileLayoutCustom::ND && GlobalData::layout == Layout::ND && TileData::Rows == 1) {
                const size_t padCols = (TileData::Cols - validCol) % blockSizeElem;
                FillTLoadRowTail(dst, validRow, validCol, padCols, typename TileData::DType(0));
            } else if constexpr (
                tileLayout == TileLayoutCustom::DN && GlobalData::layout == Layout::DN && TileData::Cols == 1) {
                const size_t padRows = (TileData::Rows - validRow) % blockSizeElem;
                FillTLoadColTail(dst, validRow, validCol, padRows, typename TileData::DType(0));
            }
            return;
        }

        // A5 aligned GM-to-L1 DMA pads only the final partial block for same-layout ND/DN loads.
        if constexpr (tileLayout == TileLayoutCustom::ND && GlobalData::layout == Layout::ND) {
            const size_t padCols = (TileData::Cols - validCol) % blockSizeElem;
            FillTLoadRowTail(dst, validRow, validCol, padCols, getPadValue<TileData>());
        } else if constexpr (tileLayout == TileLayoutCustom::DN && GlobalData::layout == Layout::DN) {
            const size_t padRows = (TileData::Rows - validRow) % blockSizeElem;
            FillTLoadColTail(dst, validRow, validCol, padRows, getPadValue<TileData>());
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void loadA5VecNz(TileData& dst, GlobalData& src)
{
    using T = typename TileData::DType;
    constexpr size_t ELEMENTS_PER_STORAGE_UNIT = IsTwinType<T>() ? 2 : 1;
    const size_t groupCount = src.GetShape(GlobalTensorDim::DIM_0);
    const size_t blockCount = src.GetShape(GlobalTensorDim::DIM_1);
    const size_t srcGroupStrideBytes = src.GetStride(GlobalTensorDim::DIM_0) * sizeof(T) / ELEMENTS_PER_STORAGE_UNIT;
    const size_t srcBlockStrideBytes = src.GetStride(GlobalTensorDim::DIM_1) * sizeof(T) / ELEMENTS_PER_STORAGE_UNIT;
    const size_t dstGroupStrideBytes =
        blockCount * TileData::Rows * src.GetShape(GlobalTensorDim::DIM_4) * sizeof(T) / ELEMENTS_PER_STORAGE_UNIT;
    constexpr size_t DST_BLOCK_STRIDE_BYTES = TileData::Rows * C0_SIZE_BYTE;
    const size_t burstBytes = dst.GetValidRow() * C0_SIZE_BYTE;
    auto* dstBytes = reinterpret_cast<uint8_t*>(dst.data());
    const auto* srcBytes = reinterpret_cast<const uint8_t*>(src.data());

    // A5 TLoadVecNZ2NZ copies every GM column block, including columns beyond validCol.
    for (size_t group = 0; group < groupCount; ++group) {
        for (size_t block = 0; block < blockCount; ++block) {
            const size_t dstOffsetBytes = group * dstGroupStrideBytes + block * DST_BLOCK_STRIDE_BYTES;
            NPUMemoryModel::Instance().CheckAccess(dstBytes, dstOffsetBytes, burstBytes);
            std::memcpy(
                dstBytes + dstOffsetBytes, srcBytes + group * srcGroupStrideBytes + block * srcBlockStrideBytes,
                burstBytes);
        }
    }
}

template <TLoadL2Hint l2Control = TLoadL2Hint::NormalFirstVictim, typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData& dst, GlobalData& src)
{
    CheckTileData<TileData, GlobalData>(dst, src);

    const size_t validRow = dst.GetValidRow();
    const size_t validCol = dst.GetValidCol();

    if constexpr (
        GlobalData::layout == Layout::NZ && TileData::Loc == TileType::Vec && !TileData::isRowMajor &&
        TileData::SFractal == SLayout::RowMajor) {
        if (GetCpuSimArch() == NPUArch::A5) {
            loadA5VecNz(dst, src);
            return;
        }
    }

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
            const size_t dstOffset = MapTileIndicesToGlobalOffset<GlobalData, TileData>(row, col, shapes, strides);
            dst.SetElement(row, col, src.GetElement(dstOffset));
        }
    }

    FillTLoadPadding<TileData, GlobalData>(dst, validRow, validCol);
}

// ConvTile TLOAD copies only the destination valid box. The box is always anchored at the logical origin,
// and everything outside it is left untouched. Declared Shape still provides every stride/pitch: (row, col)
// is packed with the declared dims so GetConvTileElementOffset/MapTileIndicesToGlobalOffset keep their
// existing meaning, while the loop bounds come from GetValidShape.
template <typename ConTile>
PTO_INLINE void CheckConvTileValidShape(ConTile& dst)
{
    for (int i = 0; i < ConTile::totalDimCount; ++i) {
        assert(dst.GetShape(i) >= dst.GetValidShape(i) && "ConvTile valid extent must not exceed the declared extent!");
    }
}

// The destination tile and the source GlobalTensor are packed with their own declared shapes: a cropped
// GlobalTensor declaration changes the source packing without touching the tile placement.
template <typename ConTile, typename GlobalData>
PTO_INLINE void TLoadConvTileElement(
    ConTile& dst, GlobalData& src, int64_t dstRow, int64_t dstCol, int64_t srcRow, int64_t srcCol,
    const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& srcShapes,
    const std::vector<int64_t>& srcStrides)
{
    const size_t srcOffset = MapTileIndicesToGlobalOffset<GlobalData>(srcRow, srcCol, srcShapes, srcStrides);
    dst.SetElement(GetConvTileElementOffset<ConTile>(dstRow, dstCol, tile_shapes), src.GetElement(srcOffset));
}

template <typename ConTile, typename GlobalData>
PTO_INLINE void TLoadConvTileNC1HWC0(
    ConTile& dst, GlobalData& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& srcShapes,
    const std::vector<int64_t>& srcStrides)
{
    using T = typename ConTile::DType;
    constexpr int64_t c0 = static_cast<int64_t>(GetC0ElemCount<T>());

    const int64_t dH = dst.GetShape(GlobalTensorDim::DIM_2);
    const int64_t dW = dst.GetShape(GlobalTensorDim::DIM_3);
    const int64_t sH = src.GetShape(GlobalTensorDim::DIM_2);
    const int64_t sW = src.GetShape(GlobalTensorDim::DIM_3);
    const int64_t sC0 = src.GetShape(GlobalTensorDim::DIM_4);
    const int64_t vN = dst.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t vC1 = dst.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t vH = dst.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t vW = dst.GetValidShape(GlobalTensorDim::DIM_3);
    int64_t vC0 = c0;
    if constexpr (ConTile::totalDimCount >= 5) {
        vC0 = dst.GetValidShape(GlobalTensorDim::DIM_4);
    }
    assert(vC0 <= c0 && "ConvTile valid C0 must not exceed the 32-byte C0 block!");
    assert(sC0 == c0 && "NC1HWC0 GlobalTensor C0 must equal the 32-byte C0 block!");

    for (int64_t n = 0; n < vN; ++n) {
        for (int64_t c1 = 0; c1 < vC1; ++c1) {
            for (int64_t h = 0; h < vH; ++h) {
                for (int64_t w = 0; w < vW; ++w) {
                    const int64_t dstRow = (n * dH + h) * dW + w;
                    const int64_t srcRow = (n * sH + h) * sW + w;
                    for (int64_t c = 0; c < vC0; ++c) {
                        TLoadConvTileElement<ConTile, GlobalData>(
                            dst, src, dstRow, c1 * c0 + c, srcRow, c1 * sC0 + c, tile_shapes, srcShapes, srcStrides);
                    }
                }
            }
        }
    }
}

template <typename ConTile, typename GlobalData>
PTO_INLINE void TLoadConvTileNDC1HWC0(
    ConTile& dst, GlobalData& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& srcShapes,
    const std::vector<int64_t>& srcStrides)
{
    using T = typename ConTile::DType;
    constexpr int64_t c0 = static_cast<int64_t>(GetC0ElemCount<T>());

    const int64_t dC1 = dst.GetShape(GlobalTensorDim::DIM_2);
    const int64_t dH = dst.GetShape(GlobalTensorDim::DIM_3);
    const int64_t dW = dst.GetShape(GlobalTensorDim::DIM_4);
    const int64_t sC1 = src.GetShape(GlobalTensorDim::DIM_2);
    const int64_t sH = src.GetShape(GlobalTensorDim::DIM_3);
    const int64_t sW = src.GetShape(GlobalTensorDim::DIM_4);
    const int64_t vN = dst.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t vD = dst.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t vC1 = dst.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t vH = dst.GetValidShape(GlobalTensorDim::DIM_3);
    const int64_t vW = dst.GetValidShape(GlobalTensorDim::DIM_4);
    int64_t vC0 = c0;
    if constexpr (ConTile::totalDimCount >= 6) {
        vC0 = dst.GetValidShape(5);
    }
    assert(vC0 <= c0 && "ConvTile valid C0 must not exceed the 32-byte C0 block!");

    for (int64_t n = 0; n < vN; ++n) {
        for (int64_t d = 0; d < vD; ++d) {
            for (int64_t c1 = 0; c1 < vC1; ++c1) {
                for (int64_t h = 0; h < vH; ++h) {
                    for (int64_t w = 0; w < vW; ++w) {
                        const int64_t dstRow = (n * dH + h) * dW + w;
                        const int64_t srcRow = (n * sH + h) * sW + w;
                        const int64_t dstColBase = (d * dC1 + c1) * c0;
                        const int64_t srcColBase = (d * sC1 + c1) * c0;
                        for (int64_t c = 0; c < vC0; ++c) {
                            TLoadConvTileElement<ConTile, GlobalData>(
                                dst, src, dstRow, dstColBase + c, srcRow, srcColBase + c, tile_shapes, srcShapes,
                                srcStrides);
                        }
                    }
                }
            }
        }
    }
}

template <typename ConTile, typename GlobalData>
PTO_INLINE void TLoadConvTileFractalZ4D(
    ConTile& dst, GlobalData& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& srcShapes,
    const std::vector<int64_t>& srcStrides)
{
    const int64_t d1 = dst.GetShape(GlobalTensorDim::DIM_1);
    const int64_t d3 = dst.GetShape(GlobalTensorDim::DIM_3);
    const int64_t v0 = dst.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t v1 = dst.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t v2 = dst.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t v3 = dst.GetValidShape(GlobalTensorDim::DIM_3);

    for (int64_t i0 = 0; i0 < v0; ++i0) {
        for (int64_t i1 = 0; i1 < v1; ++i1) {
            const int64_t rowBase = (i0 * d1 + i1) * d3;
            for (int64_t i2 = 0; i2 < v2; ++i2) {
                for (int64_t i3 = 0; i3 < v3; ++i3) {
                    // The FRACTAL_Z (row, col) packing is shared by the tile and the two supported source
                    // views ([C1,H,W,N,C0] with N1 == 1, or the flat [1,C1HW,N1,16,C0] view).
                    TLoadConvTileElement<ConTile, GlobalData>(
                        dst, src, rowBase + i3, i2, rowBase + i3, i2, tile_shapes, srcShapes, srcStrides);
                }
            }
        }
    }
}

template <typename ConTile, typename GlobalData>
PTO_INLINE void TLoadConvTileFractalZ5D(
    ConTile& dst, GlobalData& src, const std::vector<int64_t>& tile_shapes, const std::vector<int64_t>& srcShapes,
    const std::vector<int64_t>& srcStrides)
{
    const int64_t dH = dst.GetShape(GlobalTensorDim::DIM_1);
    const int64_t dW = dst.GetShape(GlobalTensorDim::DIM_2);
    const int64_t dC0 = dst.GetShape(GlobalTensorDim::DIM_4);
    const int64_t sH = src.GetShape(GlobalTensorDim::DIM_1);
    const int64_t sW = src.GetShape(GlobalTensorDim::DIM_2);
    const int64_t sC0 = src.GetShape(GlobalTensorDim::DIM_4);
    const int64_t vC1 = dst.GetValidShape(GlobalTensorDim::DIM_0);
    const int64_t vH = dst.GetValidShape(GlobalTensorDim::DIM_1);
    const int64_t vW = dst.GetValidShape(GlobalTensorDim::DIM_2);
    const int64_t vN = dst.GetValidShape(GlobalTensorDim::DIM_3);
    const int64_t vC0 = dst.GetValidShape(GlobalTensorDim::DIM_4);

    for (int64_t c1 = 0; c1 < vC1; ++c1) {
        for (int64_t h = 0; h < vH; ++h) {
            for (int64_t w = 0; w < vW; ++w) {
                const int64_t dstRowBase = ((c1 * dH + h) * dW + w) * dC0;
                const int64_t srcRowBase = ((c1 * sH + h) * sW + w) * sC0;
                for (int64_t n = 0; n < vN; ++n) {
                    for (int64_t c = 0; c < vC0; ++c) {
                        TLoadConvTileElement<ConTile, GlobalData>(
                            dst, src, dstRowBase + c, n, srcRowBase + c, n, tile_shapes, srcShapes, srcStrides);
                    }
                }
            }
        }
    }
}

template <TLoadL2Hint l2Control = TLoadL2Hint::NormalFirstVictim, typename ConTile, typename GlobalData>
__tf__ PTO_INLINE void TLOAD_CONVTILE_IMPL(ConTile& dst, GlobalData& src)
{
    CheckConvTileData<ConTile, GlobalData>(dst, src);
    CheckConvTileValidShape(dst);

    const std::vector<int64_t> tile_shapes = {
        dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> srcShapes = {
        src.GetShape(GlobalTensorDim::DIM_0), src.GetShape(GlobalTensorDim::DIM_1),
        src.GetShape(GlobalTensorDim::DIM_2), src.GetShape(GlobalTensorDim::DIM_3),
        src.GetShape(GlobalTensorDim::DIM_4)};
    const std::vector<int64_t> srcStrides = {
        src.GetStride(GlobalTensorDim::DIM_0), src.GetStride(GlobalTensorDim::DIM_1),
        src.GetStride(GlobalTensorDim::DIM_2), src.GetStride(GlobalTensorDim::DIM_3),
        src.GetStride(GlobalTensorDim::DIM_4)};

    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        TLoadConvTileNC1HWC0<ConTile, GlobalData>(dst, src, tile_shapes, srcShapes, srcStrides);
    } else if constexpr (GlobalData::layout == pto::Layout::NDC1HWC0) {
        TLoadConvTileNDC1HWC0<ConTile, GlobalData>(dst, src, tile_shapes, srcShapes, srcStrides);
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (ConTile::totalDimCount == 4) {
            TLoadConvTileFractalZ4D<ConTile, GlobalData>(dst, src, tile_shapes, srcShapes, srcStrides);
        } else {
            TLoadConvTileFractalZ5D<ConTile, GlobalData>(dst, src, tile_shapes, srcShapes, srcStrides);
        }
    }
}

template <TLoadL2Hint l2Control = TLoadL2Hint::NormalFirstVictim, typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData& dst, GlobalData& src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL<l2Control>(dst, src);
    } else {
        TLOAD_TILE_IMPL<l2Control>(dst, src);
    }
}

PTO_CPU_TARGET_NAMESPACE_END
} // namespace pto
#endif
