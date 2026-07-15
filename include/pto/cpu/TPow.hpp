/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPOW_HPP
#define TPOW_HPP

#include <cmath>
#include <type_traits>
#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename T>
inline T integer_pow(T base, int64_t exponent)
{
    static_assert(std::is_integral_v<T>, "integer_pow requires an integral type.");
    if (exponent == 0 || base == 1)
        return 1;
    if (exponent < 0) {
        if (base == -1) {
            return exponent % 2 == 0 ? 1 : -1;
        }
        return 0;
    }
    T result = 1;
    T base_val = base;
    uint64_t exp = static_cast<uint64_t>(exponent);
    while (exp > 0) {
        if (exp & 1) {
            result *= base_val;
        }
        base_val *= base_val;
        exp >>= 1;
    }
    return result;
}

template <typename T>
inline T compute_pow(T base_val, T exp_val)
{
    if constexpr (std::is_integral_v<T>) {
        int64_t exponent = static_cast<int64_t>(exp_val);
        return integer_pow(base_val, exponent);
    } else {
        double result = std::pow(static_cast<double>(base_val), static_cast<double>(exp_val));
        if constexpr (std::is_same_v<T, half>) {
            return static_cast<T>(static_cast<float>(result));
        } else {
            return static_cast<T>(result);
        }
    }
}

template <typename tile_shape>
PTO_INTERNAL void TPow_Impl(
    typename tile_shape::TileDType dst, typename tile_shape::TileDType base, typename tile_shape::TileDType exp,
    unsigned validRow, unsigned validCol)
{
    using T = typename tile_shape::DType;
    if constexpr (tile_shape::SFractal == SLayout::NoneBox) {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                const std::size_t rowBase = r * tile_shape::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = rowBase + c;
                    dst[idx] = compute_pow<T>(base[idx], exp[idx]);
                }
            });

        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                const std::size_t colBase = c * tile_shape::Rows;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < validRow; ++r) {
                    const std::size_t idx = colBase + r;
                    dst[idx] = compute_pow<T>(base[idx], exp[idx]);
                }
            });
        }

    } else {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    dst[idx] = compute_pow<T>(base[idx], exp[idx]);
                }
            });

        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                for (std::size_t r = 0; r < validRow; ++r) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    dst[idx] = compute_pow<T>(base[idx], exp[idx]);
                }
            });
        }
    }
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile>
PTO_INTERNAL void TPOW_IMPL(DstTile& dst, BaseTile& base, ExpTile& exp, TmpTile& tmp)
{
    (void)tmp;
    (void)algo;
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    TPow_Impl<DstTile>(dst.data(), base.data(), exp.data(), row, col);
}
} // namespace pto
#endif
