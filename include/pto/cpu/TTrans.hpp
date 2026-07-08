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

template <typename T>
struct is_one_of_mx_types : std::disjunction<std::is_same<T, float4_e2m1x2_t>, std::is_same<T, float4_e1m2x2_t>,
                                             std::is_same<T, float8_e8m0_t>, std::is_same<T, float8_e4m3_t>,
                                             std::is_same<T, float8_e5m2_t> > {};

template <typename T>
inline constexpr bool is_one_of_mx_types_v = is_one_of_mx_types<T>::value;

template <typename T>
inline constexpr int GetTypeSize()
{
    if constexpr (is_one_of_mx_types_v<T>)
        return 1;
    else
        return static_cast<int>(sizeof(T));
}

template <typename DstTileData, typename SrcTileData>
inline void CheckValidConvShape(DstTileData &dst, SrcTileData &src)
{
    using T = typename DstTileData::DType;
    constexpr int64_t C0 = 32 / GetTypeSize<T>() * (isTwinType<T>() ? 2 : 1);

    constexpr Layout src_layout = SrcTileData::layout;
    constexpr Layout dst_layout = DstTileData::layout;
    constexpr int DIM_0 = pto::GlobalTensorDim::DIM_0;
    constexpr int DIM_1 = pto::GlobalTensorDim::DIM_1;
    constexpr int DIM_2 = pto::GlobalTensorDim::DIM_2;
    constexpr int DIM_3 = pto::GlobalTensorDim::DIM_3;
    constexpr int DIM_4 = pto::GlobalTensorDim::DIM_4;
    constexpr int DIM_5 = 5;

    if constexpr (src_layout == Layout::NCHW && dst_layout == Layout::NC1HWC0) {
        // NCHW (N, C, H, W) -> NC1HWC0 (N, C1, H, W, C0)
        // C1 = ceil(C / C0)
        assert(dst.GetShape(DIM_0) == src.GetShape(DIM_0) &&                 // N
               dst.GetShape(DIM_1) == (src.GetShape(DIM_1) + C0 - 1) / C0 && // C1
               dst.GetShape(DIM_2) == src.GetShape(DIM_2) &&                 // H
               dst.GetShape(DIM_3) == src.GetShape(DIM_3) &&                 // W
               dst.GetShape(DIM_4) == C0 &&                                  // C0
               "Shape mismatch: NCHW to NC1HWC0");
    } else if constexpr (src_layout == Layout::NC1HWC0 && dst_layout == Layout::NCHW) {
        // NCHW (N, C, H, W) -> NC1HWC0 (N, C1, H, W, C0)
        // C1 = ceil(C / C0)
        assert(dst.GetShape(DIM_0) == src.GetShape(DIM_0) &&      // N
               dst.GetShape(DIM_1) == src.GetShape(DIM_1) * C0 && // C1
               dst.GetShape(DIM_2) == src.GetShape(DIM_2) &&      // H
               dst.GetShape(DIM_3) == src.GetShape(DIM_3) &&      // W
               src.GetShape(DIM_4) == C0 && "Shape mismatch: NC1HWC0 to NCHW");
    } else if constexpr (src_layout == Layout::NC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
        // NC1HWC0 (N, C1, H, W, C0) -> C1HWN1N0C0 (C1, H, W, N1, N0, C0)
        // N = N1 * N0
        assert(dst.GetShape(DIM_0) == src.GetShape(DIM_1) * src.GetShape(DIM_2) * src.GetShape(DIM_3) && // C1*H*W
               dst.GetShape(DIM_1) * dst.GetShape(DIM_2) >= src.GetShape(DIM_0) &&                       // N1*N0 = N
               dst.GetShape(DIM_3) == src.GetShape(DIM_4) &&                                             // C0
               "Shape mismatch: NC1HWC0 to FRACTAL_Z");
    } else if constexpr (src_layout == Layout::GNCHW && dst_layout == Layout::GNC1HWC0) {
        // GNCHW (G, N, C, H, W) -> GNC1HWC0 (G, N, C1, H, W, C0)
        assert(dst.GetShape(DIM_0) == src.GetShape(DIM_0) &&                      // G
               dst.GetShape(DIM_1) == src.GetShape(DIM_1) &&                      // N
               dst.GetShape(DIM_3) == src.GetShape(DIM_3) &&                      // H
               dst.GetShape(DIM_4) == src.GetShape(DIM_4) &&                      // W
               dst.GetShape(DIM_2) == (src.GetShape(DIM_2) + C0 - 1) / C0 &&      // C1
               dst.GetShape(DIM_5) == C0 && "Shape mismatch: GNCHW to GNC1HWC0"); // C0
    } else if constexpr (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
        // GNC1HWC0 (G, N, C1, H, W, C0) -> C1HWGN1N0C0 (C1, H, W, G, N1, N0, C0)
        // Note: Assuming Dst shape maps dimensions G*C1*H*W as outer dimension
        assert(dst.GetShape(DIM_0) == src.GetShape(DIM_0) * src.GetShape(DIM_2) * src.GetShape(DIM_3) *
                                          src.GetShape(DIM_4) && // C1, H, W, G
               dst.GetShape(DIM_1) == (src.GetShape(DIM_1) + dst.GetShape(DIM_2) - 1) / dst.GetShape(DIM_2) && // N1*N0
               dst.GetShape(DIM_3) == src.GetShape(DIM_5) &&                                                   // C0
               "Shape mismatch: GNC1HWC0 to FRACTAL_Z");
    } else if constexpr (src_layout == Layout::NCDHW && dst_layout == Layout::FRACTAL_Z_3D) {
        // NCDHW (N, C, D, H, W) -> FRACTAL_Z_3D (D, C1, H, W, N1, N0, C0)
        // Note: D, C1, H, W are merged into DIM 0 for dst.
        size_t dstC1 = dst.GetShape(DIM_0) / (src.GetShape(DIM_2) * src.GetShape(DIM_3) * src.GetShape(DIM_4));
        assert(dstC1 == (src.GetShape(DIM_1) + C0 - 1) / C0 &&
               dst.GetShape(DIM_1) == (src.GetShape(DIM_0) + dst.GetShape(DIM_2) - 1) / dst.GetShape(DIM_2) && // N1*N0
               dst.GetShape(DIM_3) == C0 &&                                                                    // C0
               "Shape mismatch: NCDHW to FRACTAL_Z_3D");
    }
}

template <typename DstTileData, typename SrcTileData, bool reverse = false>
inline void TTRANS_GNCHW2NC1HWC0_Impl(DstTileData &dst, SrcTileData &src, int64_t G, int64_t N, int64_t C, int64_t H,
                                      int64_t W)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    auto *src_ptr = reinterpret_cast<SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    constexpr int64_t C0 = (32 / GetTypeSize<SrcDType>()) * (isTwinType<DstDType>() ? 2 : 1);
    int64_t C1 = (C + C0 - 1) / C0;
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
                        if constexpr (reverse) {
                            src_ptr[base_src + w] = dst_ptr[base_dst + w * C0];
                        } else {
                            dst_ptr[base_dst + w * C0] = src_ptr[base_src + w];
                        }
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
inline void TTRANS_NC1HWC02NCHW(DstTileData &dst, SrcTileData &src)
{
    int64_t G = 1; // 4D layout has 1 implicit group
    int64_t N = dst.GetShape(0);
    int64_t C = dst.GetShape(1);
    int64_t H = dst.GetShape(2);
    int64_t W = dst.GetShape(3);

    TTRANS_GNCHW2NC1HWC0_Impl<SrcTileData, DstTileData, true>(src, dst, G, N, C, H, W);
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

    auto *src_ptr = reinterpret_cast<SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    int64_t N1 = dst.GetShape(1);
    int64_t N0 = dst.GetShape(2);
    int64_t C0 = dst.GetShape(3);

    assert(N0 > 0 && "N0 must be greater than 0!");

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
inline void TTRANS_NCDHW2DC1HWN1N0C0(DstTileData &dst, SrcTileData &src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto *src_ptr = reinterpret_cast<const SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    // Shape extraction
    const int64_t N = src.GetShape(0);
    const int64_t C = src.GetShape(1);
    const int64_t D = src.GetShape(2);
    const int64_t H = src.GetShape(3);
    const int64_t W = src.GetShape(4);

    const int64_t C0 = dst.GetShape(3); // Assuming index 3 is C0
    const int64_t N0 = dst.GetShape(2);
    const int64_t N1 = dst.GetShape(1);
    const int64_t C1 = (C + C0 - 1) / C0;

    const size_t HW = H * W;
    const size_t DHW = D * HW;
    const size_t CDHW = C * DHW;
    const size_t C0_N0 = C0 * N0;
    const size_t C0_N0_N1 = C0_N0 * N1;
    const size_t C1HW = C1 * H * W;

    for (int64_t n = 0; n < N; ++n) {
        const size_t n1 = n / N0;
        const size_t n0 = n % N0;
        const size_t dst_base_n = n1 * C0_N0 + n0 * C0;
        const size_t src_base_n = n * CDHW;
        for (int64_t c = 0; c < C; ++c) {
            const size_t c1 = c / C0;
            const size_t c0 = c % C0;
            const size_t src_base_c = src_base_n + c * DHW;
            const size_t dst_base_c1 = c1 * HW;
            for (int64_t d = 0; d < D; ++d) {
                const size_t src_base_d = src_base_c + d * HW;
                const size_t dst_base_d = d * C1HW + dst_base_c1;
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        const size_t base_w = h * W + w;
                        // Offset in NCDHW source
                        size_t src_offset = src_base_d + base_w;

                        // Base for fractal block
                        size_t dst_base = (dst_base_d + base_w) * C0_N0_N1 + dst_base_n + c0;

                        dst_ptr[dst_base] = src_ptr[src_offset];
                    }
                }
            }
        }
    }
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

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TTRANS_CONV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckValidConvShape<DstTileData, SrcTileData>(dst, src);
    constexpr Layout src_layout = SrcTileData::layout;
    constexpr Layout dst_layout = DstTileData::layout;

    if constexpr (src_layout == Layout::NCHW && dst_layout == Layout::NC1HWC0) {
        TTRANS_NCHW2NC1HWC0(dst, src);
    } else if constexpr (src_layout == Layout::NC1HWC0 && dst_layout == Layout::NCHW) {
        TTRANS_NC1HWC02NCHW(dst, src);
    } else if constexpr (src_layout == Layout::NC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
        TTRANS_NC1HWC02C1HWN1N0C0(dst, src);
    } else if constexpr (src_layout == Layout::GNCHW && dst_layout == Layout::GNC1HWC0) {
        TTRANS_GNCHW2NC1HWC0(dst, src);
    } else if constexpr (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
        TTRANS_GNC1HWC02C1HWN1N0C0(dst, src);
    } else if constexpr (src_layout == Layout::NCDHW && dst_layout == Layout::FRACTAL_Z_3D) {
        TTRANS_NCDHW2DC1HWN1N0C0(dst, src);
    }
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData>
PTO_INTERNAL void TTRANS_IMPL(DstTileData &dst, SrcTileData &src, TmpTileData &tmp)
{
    // Validate matching element widths at compilation
    static_assert(sizeof(typename SrcTileData::DType) == sizeof(typename DstTileData::DType),
                  "Data type sizes between source and destination tiles must match.");

    if constexpr (is_conv_tile_v<SrcTileData> && is_conv_tile_v<DstTileData>) {
        TTRANS_CONV_IMPL(dst, src);
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
