/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_TCI_HPP
#define PTO_LINX_VENDOR_TCI_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape, int desc>
void TCI_RowMajor_Imp(typename tile_shape::TileDType dst, const typename tile_shape::DType s)
{
    for (size_t i = 0; i < tile_shape::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
            size_t idx = i * tile_shape::RowStride + j;
            using UnsignedDType = typename std::make_unsigned<typename tile_shape::DType>::type;
            const UnsignedDType start = static_cast<UnsignedDType>(s);
            const UnsignedDType offset = static_cast<UnsignedDType>(idx);
            const UnsignedDType value = desc ? start - offset : start + offset;
            dst[idx] = static_cast<typename tile_shape::DType>(value);
        }
}
template <typename tile_shape, int desc>
void TCI_ColMajor_Imp(typename tile_shape::TileDType dst, const typename tile_shape::DType s)
{
    for (size_t i = 0; i < tile_shape::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            size_t idx = i * tile_shape::ColStride + j;
            if constexpr (desc) {
                dst[idx] = s - static_cast<typename tile_shape::DType>(idx);
            } else {
                dst[idx] = s + static_cast<typename tile_shape::DType>(idx);
            }
        }
}

template <is_tile_data_v tile_shape, typename T, int descending>
void TCI_Impl(tile_shape& dst, T s)
{
    static constexpr size_t row = tile_shape::ValidRow;
    static constexpr size_t col = tile_shape::ValidCol;

    static_assert(std::is_same<typename tile_shape::DType, T>::value, "Dst and scalar must be same data type!");
    static_assert((descending == 0) || (descending == 1), "descending must be 0 or 1!");
    static_assert(row != DYNAMIC && col != DYNAMIC, "TODO: Support tile dynamic shape!");
    static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
    static_assert(tile_shape::isRowMajor, "TCI requires RowMajor layout");
    static_assert(row == 1, "TCI requires ValidRow == 1");
    static_assert(col > 0 && tile_shape::Cols >= col, "TCI requires 0 < ValidCol <= Cols");
    static_assert(!tile_shape::isBoxedLayout, "TCI not support Boxed Layout!");
    if constexpr (
        std::is_same<typename tile_shape::DType, int32_t>::value ||
        std::is_same<typename tile_shape::DType, uint32_t>::value ||
        std::is_same<typename tile_shape::DType, int16_t>::value ||
        std::is_same<typename tile_shape::DType, uint16_t>::value) {
        TCI_RowMajor_Imp<tile_shape, descending>(dst.data(), s);
    } else {
        static_assert(
            std::is_same<typename tile_shape::DType, int32_t>::value ||
                std::is_same<typename tile_shape::DType, uint32_t>::value ||
                std::is_same<typename tile_shape::DType, int16_t>::value ||
                std::is_same<typename tile_shape::DType, uint16_t>::value,
            "Data type not supported");
    }
}

template <is_tile_data_v tile_shape, typename T, int row_step, int col_step>
void TCI_2D_Impl(tile_shape& dst, T s)
{
    static_assert(
        std::is_same<typename tile_shape::DType, T>::value, "TCI_2D destination and start must have the same type");
    static_assert(
        row_step >= -1 && row_step <= 1 && col_step >= -1 && col_step <= 1, "TCI_2D steps must be -1, 0, or 1");
    static_assert(
        tile_shape::IsCubeLayout &&
            (tile_shape::BFractal == BLayout::CubeM16 || tile_shape::BFractal == BLayout::CubeM32),
        "TCI_2D requires a CUBE_M16 or CUBE_M32 tile");
    static_assert(
        tile_shape::ValidRow > 0 && tile_shape::ValidCol > 0 && tile_shape::ValidRow <= tile_shape::Rows &&
            tile_shape::ValidCol <= tile_shape::Cols,
        "TCI_2D requires positive valid dimensions within the tile");
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value || std::is_same<T, uint32_t>::value ||
            std::is_same<T, uint16_t>::value,
        "TCI_2D supports only S32, S16, U32, and U16");
    for (size_t r = 0; r < tile_shape::ValidRow; ++r) {
        for (size_t c = 0; c < tile_shape::ValidCol; ++c) {
            using U = typename std::make_unsigned<T>::type;
            const int64_t offset = static_cast<int64_t>(r) * row_step + static_cast<int64_t>(c) * col_step;
            const U value = static_cast<U>(s) + static_cast<U>(offset);
            dst.data()[tile_shape::CubeStorageIndex(static_cast<int>(r), static_cast<int>(c))] = static_cast<T>(value);
        }
    }
}

#endif
