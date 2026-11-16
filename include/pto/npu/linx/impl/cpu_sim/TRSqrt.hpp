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

#ifndef PTO_LINX_VENDOR_TRSQRT_HPP
#define PTO_LINX_VENDOR_TRSQRT_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TRSqrt_RowMajor_Imp(typename tile_shape::TileDType dst, const typename tile_shape::TileDType src)
{
    for (uint16_t i = 0; i < tile_shape::ValidRow; ++i)
        for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
            typename tile_shape::DType temp = std::sqrt(src[i * tile_shape::RowStride + j]);
            dst[i * tile_shape::RowStride + j] = static_cast<typename tile_shape::DType>(1.0) / temp;
        }
}

template <is_tile_data_v tile_shape>
void TRSQRT_Impl(tile_shape& dst, tile_shape& src)
{
    static_assert(
        tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC, "TODO: Support tile dynamic shape!");
    if constexpr (tile_shape::isRowMajor) {
        TRSqrt_RowMajor_Imp<tile_shape>(dst.data(), src.data());
    } else {
        static_assert(tile_shape::isRowMajor, "Storage layout type not supported");
    }
}

#endif
