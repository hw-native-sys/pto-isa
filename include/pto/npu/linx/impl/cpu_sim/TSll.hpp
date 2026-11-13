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

#ifndef PTO_LINX_VENDOR_TSLL_HPP
#define PTO_LINX_VENDOR_TSLL_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

// Host reference: uniform left shift of the whole tile by a scalar amount.
template <typename tile_shape>
void TSll_RowMajor_Imp(typename tile_shape::TileDType dst, const typename tile_shape::TileDType src, unsigned shamt)
{
    for (size_t i = 0; i < tile_shape::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
            size_t idx = i * tile_shape::RowStride + j;
            dst[idx] = src[idx] << shamt;
        }
}

template <typename tile_shape>
void TSll_ColMajor_Imp(typename tile_shape::TileDType dst, const typename tile_shape::TileDType src, unsigned shamt)
{
    for (size_t i = 0; i < tile_shape::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            size_t idx = i * tile_shape::ColStride + j;
            dst[idx] = src[idx] << shamt;
        }
}

template <is_tile_data_v tile_shape>
void TSLL_Impl(tile_shape& dst, tile_shape& src, unsigned shamt)
{
    static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
    static_assert(!tile_shape::isBoxedLayout, "TSLL not support Boxed Layout!");

    if constexpr (
        std::is_same<typename tile_shape::DType, int64_t>::value ||
        std::is_same<typename tile_shape::DType, int32_t>::value ||
        std::is_same<typename tile_shape::DType, int16_t>::value ||
        std::is_same<typename tile_shape::DType, int8_t>::value ||
        std::is_same<typename tile_shape::DType, unsigned long>::value ||
        std::is_same<typename tile_shape::DType, unsigned int>::value ||
        std::is_same<typename tile_shape::DType, unsigned short>::value ||
        std::is_same<typename tile_shape::DType, unsigned char>::value) {
        if constexpr (tile_shape::isRowMajor) {
            TSll_RowMajor_Imp<tile_shape>(dst.data(), src.data(), shamt);
        } else {
            TSll_ColMajor_Imp<tile_shape>(dst.data(), src.data(), shamt);
        }
    } else {
        static_assert(
            std::is_same<typename tile_shape::DType, int64_t>::value ||
                std::is_same<typename tile_shape::DType, int32_t>::value ||
                std::is_same<typename tile_shape::DType, int16_t>::value ||
                std::is_same<typename tile_shape::DType, int8_t>::value ||
                std::is_same<typename tile_shape::DType, unsigned long>::value ||
                std::is_same<typename tile_shape::DType, unsigned int>::value ||
                std::is_same<typename tile_shape::DType, unsigned short>::value ||
                std::is_same<typename tile_shape::DType, unsigned char>::value,
            "Only int data type are supported");
    }
}

#endif
