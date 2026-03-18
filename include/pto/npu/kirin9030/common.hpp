/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef COMMON_HPP
#define COMMON_HPP

#include "datatype.hpp"
#include <pto/common/type.hpp>

namespace pto {

template <typename T>
PTO_INTERNAL uint32_t GetByteSize(const uint32_t value)
{
    return sizeof(T) * value;
}

template <typename T, int U, int... Args>
AICORE constexpr bool SupportBytes()
{
    if constexpr (sizeof...(Args) > 0) {
        return sizeof(T) == U || SupportBytes<T, Args...>();
    }
    return sizeof(T) == U;
}

using MaskReg = vector_bool;
using UnalignReg = vector_align;
using AddrReg = vector_address;

template <typename T>
PTO_INTERNAL MaskReg CreatePredicate(uint32_t &scalar)
{
    MaskReg reg;
    if constexpr (sizeof(T) == 1) {
        reg = plt_b8(scalar, POST_UPDATE);
    } else if constexpr (sizeof(T) == 2) {
        reg = plt_b16(scalar, POST_UPDATE);
    } else if constexpr (sizeof(T) == 4) {
        reg = plt_b32(scalar, POST_UPDATE);
    }
    return reg;
}

template <typename T>
struct RegTensor {
    using RegType = typename TypeGet<T>::T;
    RegType reg;

    PTO_INTERNAL RegTensor(){};
    PTO_INTERNAL operator RegType &()
    {
        return reg;
    }
};

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetCastPreQuantMode()
{
    return QuantMode_t::NoQuant;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetScalarPreQuantMode()
{
    if constexpr (std::is_same_v<SrcType, int32_t>) {
        if constexpr (std::is_same_v<DstType, half>) {
            return QuantMode_t::DEQF16;
        } else if constexpr (std::is_same_v<DstType, int16_t>) {
            return QuantMode_t::DEQS16;
        } else if constexpr (std::is_same_v<DstType, int8_t> || std::is_same_v<DstType, uint8_t>) {
            return QuantMode_t::REQ8;
        }
    } else if constexpr (std::is_same_v<SrcType, half>) {
        if constexpr (std::is_same_v<DstType, int16_t>) {
            return QuantMode_t::QF162S16_PRE;
        } else if constexpr (std::is_same_v<DstType, int8_t> || std::is_same_v<DstType, uint8_t>) {
            return QuantMode_t::QF162B8_PRE;
        }
    }
    return QuantMode_t::NoQuant;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetVectorPreQuantMode()
{
    if constexpr (std::is_same_v<SrcType, int32_t>) {
        if constexpr (std::is_same_v<DstType, half>) {
            return QuantMode_t::VDEQF16;
        } else if constexpr (std::is_same_v<DstType, int16_t>) {
            return QuantMode_t::VDEQS16;
        } else if constexpr (std::is_same_v<DstType, int8_t> || std::is_same_v<DstType, uint8_t>) {
            return QuantMode_t::VREQ8;
        }
    } else if constexpr (std::is_same_v<SrcType, half>) {
        if constexpr (std::is_same_v<DstType, int16_t>) {
            return QuantMode_t::VQF162S16_PRE;
        } else if constexpr (std::is_same_v<DstType, int8_t> || std::is_same_v<DstType, uint8_t>) {
            return QuantMode_t::VQF162B8_PRE;
        }
    }

    return QuantMode_t::NoQuant;
}

template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType, bool isQuant = false>
PTO_INTERNAL void CheckTMovAccValid()
{
    static_assert((SrcTileData::Loc == TileType::Acc), "Source TileType only support Acc.");
    static_assert((!SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor),
                  "Src fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    static_assert(std::is_same_v<SrcType, half> || std::is_same_v<SrcType, int32_t>,
                  "Src data type only support half or int32_t.");
    if constexpr (isQuant) {
        if constexpr (std::is_same_v<SrcType, half>) {
            static_assert(std::is_same_v<DstType, half> || std::is_same_v<DstType, int8_t> ||
                              std::is_same_v<DstType, uint8_t> || std::is_same_v<DstType, int16_t>,
                          "The output data type must be int8/uint8/half/int16 when input is data type half.");
        } else if constexpr (std::is_same_v<SrcType, int32_t>) {
            static_assert(std::is_same_v<DstType, half> || std::is_same_v<DstType, int8_t> ||
                              std::is_same_v<DstType, uint8_t> || std::is_same_v<DstType, int16_t>,
                          "The output data type must be int8/uint8/half/int16/int32 when input is data type int32.");
        }
    } else {
        static_assert(std::is_same_v<DstType, SrcType>,
                      "The input data type must be consistent with the output data type when preQuantScalar is not "
                      "configured");
        static_assert(std::is_same_v<DstType, half> || std::is_same_v<DstType, int32_t>,
                      "The data type must be half or int32 when preQuantScalar is not configured");
    }
    static_assert((DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                      (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                      (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Only support nz2nz, nz2nd or nz2dn.");
}
} // namespace pto

#endif