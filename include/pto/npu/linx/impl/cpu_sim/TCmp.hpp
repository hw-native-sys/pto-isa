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

#ifndef PTO_LINX_VENDOR_TCMP_HPP
#define PTO_LINX_VENDOR_TCMP_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <assert.h>
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in, CmpMode mode>
void TCmp_Vec_RowMajor(
    typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src0,
    const typename tile_shape_in::TileDType src1)
{
    for (size_t i = 0; i < tile_shape_in::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape_in::ValidCol; ++j) {
            size_t idx = i * tile_shape_in::RowStride + j;
            if constexpr (mode == CmpMode::EQ) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] == src1[idx]);
            } else if constexpr (mode == CmpMode::NE) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] != src1[idx]);
            } else if constexpr (mode == CmpMode::GT) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] > src1[idx]);
            } else if constexpr (mode == CmpMode::LT) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] < src1[idx]);
            } else if constexpr (mode == CmpMode::GE) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] >= src1[idx]);
            } else if constexpr (mode == CmpMode::LE) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] <= src1[idx]);
            } else {
                static_assert(
                    mode == CmpMode::EQ || mode == CmpMode::NE || mode == CmpMode::LT || mode == CmpMode::GT ||
                        mode == CmpMode::LE || mode == CmpMode::GE,
                    "TCMP mode must be one of the six ISA comparison modes");
            }
        }
}

template <typename tile_shape_out, typename tile_shape_in, CmpMode mode>
void TCmp_Vec_ColMajor(
    typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src0,
    const typename tile_shape_in::TileDType src1)
{
    for (size_t i = 0; i < tile_shape_in::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape_in::ValidRow; ++j) {
            size_t idx = i * tile_shape_in::ColStride + j;
            if constexpr (mode == CmpMode::EQ) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] == src1[idx]);
            } else if constexpr (mode == CmpMode::NE) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] != src1[idx]);
            } else if constexpr (mode == CmpMode::GT) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] > src1[idx]);
            } else if constexpr (mode == CmpMode::LT) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] < src1[idx]);
            } else if constexpr (mode == CmpMode::GE) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] >= src1[idx]);
            } else if constexpr (mode == CmpMode::LE) {
                dst[idx] = static_cast<typename tile_shape_out::DType>(src0[idx] <= src1[idx]);
            } else {
                static_assert(
                    mode == CmpMode::EQ || mode == CmpMode::NE || mode == CmpMode::LT || mode == CmpMode::GT ||
                        mode == CmpMode::LE || mode == CmpMode::GE,
                    "TCMP mode must be one of the six ISA comparison modes");
            }
        }
}

template <typename tile_shape_out, typename tile_shape_in>
void TCMP_Impl(tile_shape_out& dst, tile_shape_in& src0, tile_shape_in& src1, CmpMode cmpMode)
{
    static_assert(
        tile_shape_in::Rows == tile_shape_out::Rows && tile_shape_in::Cols == tile_shape_out::Cols,
        "Error! Input shape != Output shape");
    static_assert(
        tile_shape_in::InnerRows == tile_shape_out::InnerRows && tile_shape_in::InnerCols == tile_shape_out::InnerCols,
        "Error! Inner shape is not equal!");
    static_assert(
        tile_shape_out::Loc == Location::Vec && tile_shape_in::Loc == Location::Vec,
        "Only VEC tile type are supported");
    static_assert(
        tile_shape_out::isBoxedLayout == false && tile_shape_in::isBoxedLayout == false,
        "TCMP not support Boxed Layout!");

    if constexpr (
        std::is_same<typename tile_shape_in::DType, int64_t>::value ||
        std::is_same<typename tile_shape_in::DType, int32_t>::value ||
        std::is_same<typename tile_shape_in::DType, float>::value ||
        std::is_same<typename tile_shape_in::DType, __half>::value) {
        if constexpr (tile_shape_in::isRowMajor) {
            switch (cmpMode) {
                case CmpMode::EQ:
                    if (std::is_same<typename tile_shape_in::DType, int32_t>::value) {
                        TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::EQ>(
                            dst.data(), src0.data(), src1.data());
                    } else {
                        assert(false && "Only Int32_t type supports the EQ mode!");
                    }
                    break;
                case CmpMode::NE:
                    TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::NE>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::GT:
                    TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::GT>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::LT:
                    TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::LT>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::GE:
                    TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::GE>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::LE:
                    TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::LE>(dst.data(), src0.data(), src1.data());
                    break;
                default:
                    assert(false && "Unsupported CmpMode value");
                    break;
            }
        } else {
            switch (cmpMode) {
                case CmpMode::EQ:
                    if (std::is_same<typename tile_shape_in::DType, int32_t>::value) {
                        TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::EQ>(
                            dst.data(), src0.data(), src1.data());
                    } else {
                        assert(false && "Only Int32_t type supports the EQ mode!");
                    }
                    break;
                case CmpMode::NE:
                    TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::NE>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::GT:
                    TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::GT>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::LT:
                    TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::LT>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::GE:
                    TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::GE>(dst.data(), src0.data(), src1.data());
                    break;
                case CmpMode::LE:
                    TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::LE>(dst.data(), src0.data(), src1.data());
                    break;
                default:
                    assert(false && "Unsupported CmpMode value");
                    break;
            }
        }
    } else {
        static_assert(
            std::is_same<typename tile_shape_in::DType, int64_t>::value ||
                std::is_same<typename tile_shape_in::DType, int32_t>::value ||
                std::is_same<typename tile_shape_in::DType, float>::value ||
                std::is_same<typename tile_shape_in::DType, __half>::value,
            "Dtype not support!");
    }
}
#endif
