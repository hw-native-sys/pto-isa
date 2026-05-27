/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TTRANS_HPP
#define TTRANS_HPP

#include <array>

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include <type_traits>
#include <cstdint>

namespace pto {

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNCHW2NC1HWC0_Impl(DstTileData &dst, SrcTileData &src, int64_t G, int64_t N, int64_t C, int64_t H,
                                      int64_t W)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto *src_ptr = reinterpret_cast<const SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    constexpr int64_t C0 = 32 / sizeof(SrcDType);
    int64_t C1 = (C + C0 - 1) / C0;
    size_t Size = G * N * C1 * H * W * C0;

    std::fill(dst.data(), dst.data() + Size, 0);

    const size_t HW = H * W;
    const size_t CHW = C * HW;
    const size_t C1HW_C0 = C1 * HW * C0;

    for (int64_t g = 0; g < G; ++g) {
        size_t gOffset_src = g * (N * CHW);
        size_t gOffset_dst = g * (N * C1HW_C0);

        for (int64_t n = 0; n < N; ++n) {
            size_t nOffset_src = n * CHW;
            size_t nOffset_dst = n * C1HW_C0;

            for (int64_t c = 0; c < C; ++c) {
                size_t r = c / C0;
                size_t cl = c % C0;

                size_t cOffset_src = c * HW;
                size_t cOffset_dst = (r * HW * C0) + cl;

                for (int64_t h = 0; h < H; ++h) {
                    size_t hOffset_src = h * W;
                    size_t hOffset_dst = h * (W * C0);

                    size_t base_src = gOffset_src + nOffset_src + cOffset_src + hOffset_src;
                    size_t base_dst = gOffset_dst + nOffset_dst + cOffset_dst + hOffset_dst;

                    for (int64_t w = 0; w < W; ++w) {
                        dst_ptr[base_dst + w * C0] = src_ptr[base_src + w];
                    }
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NCHW2NC1HWC0(DstTileData &dst, SrcTileData &src)
{
    int64_t G = 1; // 4D layout has 1 implicit group
    int64_t N = src.GetShape(0);
    int64_t C = src.GetShape(1);
    int64_t H = src.GetShape(2);
    int64_t W = src.GetShape(3);

    TTRANS_GNCHW2NC1HWC0_Impl(dst, src, G, N, C, H, W);
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNCHW2NC1HWC0(DstTileData &dst, SrcTileData &src)
{
    int64_t G = src.GetShape(0);
    int64_t N = src.GetShape(1);
    int64_t C = src.GetShape(2);
    int64_t H = src.GetShape(3);
    int64_t W = src.GetShape(4);

    TTRANS_GNCHW2NC1HWC0_Impl(dst, src, G, N, C, H, W);
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNC1HWC02C1HWN1N0C0_Impl(DstTileData &dst, SrcTileData &src, int64_t G, int64_t N, int64_t C1,
                                            int64_t H, int64_t W)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto *src_ptr = reinterpret_cast<const SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    int64_t N1 = dst.GetShape(1);
    int64_t N0 = dst.GetShape(2);
    int64_t C0 = dst.GetShape(3);

    if (N0 <= 0) {
        throw std::invalid_argument("N0 must be greater than 0!");
    }

    size_t Size = dst.GetShape(0) * N1 * N0 * C0;
    std::fill(dst.data(), dst.data() + Size, 0);

    const size_t HW = H * W;
    const size_t C1HW = C1 * HW;
    const size_t C0_N0 = C0 * N0;
    const size_t C0_N0_N1 = C0_N0 * N1;

    for (int64_t g = 0; g < G; ++g) {
        // Source offset for group g
        const size_t g_src_offset = g * N * C1HW * C0;

        for (int64_t n = 0; n < N; ++n) {
            const size_t n1 = n / N0;
            const size_t n0 = n % N0;

            const size_t n_src_offset = g_src_offset + (n * C1HW * C0);
            const size_t dst_n_base = (n0 * C0) + (n1 * C0_N0);

            for (int64_t c1 = 0; c1 < C1; ++c1) {
                const size_t c1_src_offset = n_src_offset + (c1 * HW * C0);

                // Destination block base for (g, c1)
                const size_t g_c1_dst_base = (g * C1HW + c1 * HW) * C0_N0_N1;

                for (int64_t h = 0; h < H; ++h) {
                    const size_t h_src_offset = c1_src_offset + (h * W * C0);
                    const size_t h_dst_offset = h * W * C0_N0_N1;

                    for (int64_t w = 0; w < W; ++w) {
                        size_t src_base = h_src_offset + (w * C0);
                        size_t dst_base = dst_n_base + g_c1_dst_base + h_dst_offset + (w * C0_N0_N1);

                        // Continuous block copy
                        for (size_t c0 = 0; c0 < C0; ++c0) {
                            dst_ptr[dst_base + c0] = src_ptr[src_base + c0];
                        }
                    }
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NC1HWC02C1HWN1N0C0(DstTileData &dst, SrcTileData &src)
{
    int64_t G = 1; // 4D layout implies 1 implicit group
    int64_t N = src.GetShape(0);
    int64_t C1 = src.GetShape(1);
    int64_t H = src.GetShape(2);
    int64_t W = src.GetShape(3);

    TTRANS_GNC1HWC02C1HWN1N0C0_Impl(dst, src, G, N, C1, H, W);
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNC1HWC02C1HWN1N0C0(DstTileData &dst, SrcTileData &src)
{
    int64_t G = src.GetShape(0);
    int64_t N = src.GetShape(1);
    int64_t C1 = src.GetShape(2);
    int64_t H = src.GetShape(3);
    int64_t W = src.GetShape(4);

    TTRANS_GNC1HWC02C1HWN1N0C0_Impl(dst, src, G, N, C1, H, W);
}

template <typename DstTileData, typename SrcTileData>
void TTrans_Impl(typename DstTileData::TileDType dst, typename SrcTileData::TileDType src, unsigned validRow,
                 unsigned validCol)
{
    using SrcDType = typename SrcTileData::DType;
    std::array<SrcDType, SrcTileData::Rows * SrcTileData::Cols> srcSnapshot;

    for (size_t r = 0; r < validRow; r++) {
        for (size_t c = 0; c < validCol; c++) {
            size_t srcTileIdx = GetTileElementOffset<SrcTileData>(r, c);
            srcSnapshot[r * validCol + c] = src[srcTileIdx];
        }
    }

    for (size_t c = 0; c < validCol; c++) {
        for (size_t r = 0; r < validRow; r++) {
            size_t dstTileIdx = GetTileElementOffset<DstTileData>(c, r);
            dst[dstTileIdx] = srcSnapshot[r * validCol + c];
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData>
PTO_INTERNAL void TTRANS_IMPL(DstTileData &dst, SrcTileData &src, TmpTileData &tmp)
{
    // Validate matching element widths at compilation
    static_assert(sizeof(typename SrcTileData::DType) == sizeof(typename DstTileData::DType),
                  "Data type sizes between source and destination tiles must match.");

    if constexpr (is_conv_tile_v<SrcTileData> && is_conv_tile_v<DstTileData>) {
        constexpr Layout src_layout = SrcTileData::layout;
        constexpr Layout dst_layout = DstTileData::layout;

        if constexpr (src_layout == Layout::NCHW && dst_layout == Layout::NC1HWC0) {
            TTRANS_NCHW2NC1HWC0(dst, src);
        } else if constexpr (src_layout == Layout::NC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
            TTRANS_NC1HWC02C1HWN1N0C0(dst, src);
        } else if constexpr (src_layout == Layout::GNCHW && dst_layout == Layout::GNC1HWC0) {
            TTRANS_GNCHW2NC1HWC0(dst, src);
        } else if constexpr (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
            TTRANS_GNC1HWC02C1HWN1N0C0(dst, src);
        }
    } else if constexpr (is_tile_data_v<SrcTileData> && is_tile_data_v<DstTileData>) {
        static_assert(SrcTileData::ValidRow == DstTileData::ValidCol && SrcTileData::ValidCol == DstTileData::ValidRow,
                      "Hardware matrix tiles transpose dimension sizes must mirror match.");
        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        TTrans_Impl<DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol);
    }
}

} // namespace pto

#endif // TTRANS_HPP
