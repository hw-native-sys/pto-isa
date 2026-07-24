/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPRINT_CPU_HPP
#define TPRINT_CPU_HPP

#include <iomanip>
#include <iostream>
#include <type_traits>

#include "pto/cpu/tile_offsets.hpp"
#include <pto/common/pto_tile.hpp>

namespace pto {

template <PrintFormat Format>
struct PrintFormatTraits;

template <>
struct PrintFormatTraits<PrintFormat::Width8_Precision4> {
    static constexpr int Width = 8;
    static constexpr int Precision = 4;
};

template <>
struct PrintFormatTraits<PrintFormat::Width8_Precision2> {
    static constexpr int Width = 8;
    static constexpr int Precision = 2;
};

template <>
struct PrintFormatTraits<PrintFormat::Width10_Precision6> {
    static constexpr int Width = 10;
    static constexpr int Precision = 6;
};

template <PrintFormat Format, typename T>
PTO_INTERNAL void PrintValue(const T& value)
{
    constexpr int width = PrintFormatTraits<Format>::Width;
    if constexpr (std::is_floating_point_v<T>) {
        constexpr int precision = PrintFormatTraits<Format>::Precision;
        std::cout << std::fixed << std::setw(width) << std::setprecision(precision) << value;
    } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        std::cout << std::setw(width) << static_cast<int>(value);
    } else if constexpr (std::is_integral_v<T>) {
        std::cout << std::setw(width) << value;
    } else {
        std::cout << value;
    }
}

template <PrintFormat Format = PrintFormat::Width8_Precision4, typename T>
PTO_INTERNAL void TPRINT_IMPL(T& src)
{
    using DType = typename T::DType;
    std::cout << "TPRINT " << src.GetValidRow() << "x" << src.GetValidCol() << '\n';
    for (unsigned r = 0; r < src.GetValidRow(); ++r) {
        for (unsigned c = 0; c < src.GetValidCol(); ++c) {
            if (c != 0) {
                std::cout << ' ';
            }
            using DataType = typename T::DType;
            PrintValue<Format, DataType>(src.data()[GetTileElementOffset<T>(r, c)]);
        }
        std::cout << '\n';
    }
}

template <PrintFormat Format = PrintFormat::Width8_Precision4, typename TileData, typename GlobalData>
PTO_INTERNAL void TPRINT_IMPL(TileData& src, GlobalData& tmp)
{
    TPRINT_IMPL<Format>(src);
}

} // namespace pto

#endif
