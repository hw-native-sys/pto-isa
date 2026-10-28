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
struct is_one_of_mx_types
    : std::disjunction<
          std::is_same<T, float4_e2m1x2_t>, std::is_same<T, float4_e1m2x2_t>, std::is_same<T, float8_e8m0_t>,
          std::is_same<T, float8_e4m3_t>, std::is_same<T, float8_e5m2_t> > {};

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
inline void CheckValidConvShape(DstTileData& dst, SrcTileData& src)
{
    using T = typename DstTileData::DType;
    constexpr int64_t C0 = 32 / GetTypeSize<T>() * (IsTwinType<T>() ? 2 : 1);
    constexpr int64_t N0 = 16; // 分形布局的 N0 是格式常量

    constexpr Layout src_layout = SrcTileData::layout;
    constexpr Layout dst_layout = DstTileData::layout;
    constexpr int DIM_0 = pto::GlobalTensorDim::DIM_0;
    constexpr int DIM_1 = pto::GlobalTensorDim::DIM_1;
    constexpr int DIM_2 = pto::GlobalTensorDim::DIM_2;
    constexpr int DIM_3 = pto::GlobalTensorDim::DIM_3;
    constexpr int DIM_4 = pto::GlobalTensorDim::DIM_4;
    constexpr int DIM_5 = 5;

    // raw 只负责"装得下"：每个 tile 的 raw 各维都必须覆盖其 valid 各维
    for (int i = 0; i < DstTileData::totalDimCount; ++i) {
        assert(dst.GetShape(i) >= dst.GetValidShape(i) && "dst raw shape must cover its valid shape");
    }
    for (int i = 0; i < SrcTileData::totalDimCount; ++i) {
        assert(src.GetShape(i) >= src.GetValidShape(i) && "src raw shape must cover its valid shape");
    }

    // 格式关系只作用于 valid：raw 与格式转换无关，C0 / N0 是格式常量
    if constexpr (src_layout == Layout::NCHW && dst_layout == Layout::NC1HWC0) {
        assert(
            dst.GetValidShape(DIM_4) == C0 &&                                       // C0 is a format constant
            dst.GetValidShape(DIM_0) == src.GetValidShape(DIM_0) &&                 // N
            dst.GetValidShape(DIM_1) == (src.GetValidShape(DIM_1) + C0 - 1) / C0 && // C1v = ceil(Cv / C0)
            dst.GetValidShape(DIM_2) == src.GetValidShape(DIM_2) &&                 // H
            dst.GetValidShape(DIM_3) == src.GetValidShape(DIM_3) &&                 // W
            "Valid shape mismatch: NCHW to NC1HWC0");
    } else if constexpr (src_layout == Layout::NC1HWC0 && dst_layout == Layout::NCHW) {
        assert(
            src.GetValidShape(DIM_4) == C0 &&                            // C0 is a format constant
            dst.GetValidShape(DIM_0) == src.GetValidShape(DIM_0) &&      // N
            dst.GetValidShape(DIM_1) == src.GetValidShape(DIM_1) * C0 && // Cv = C1v * C0
            dst.GetValidShape(DIM_2) == src.GetValidShape(DIM_2) &&      // H
            dst.GetValidShape(DIM_3) == src.GetValidShape(DIM_3) &&      // W
            "Valid shape mismatch: NC1HWC0 to NCHW");
    } else if constexpr (src_layout == Layout::NC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
        assert(
            src.GetValidShape(DIM_4) == C0 && dst.GetValidShape(DIM_3) == C0 && // C0 is a format constant
            dst.GetValidShape(DIM_2) == N0 &&                                   // N0 is a format constant
            dst.GetValidShape(DIM_0) ==
                src.GetValidShape(DIM_1) * src.GetValidShape(DIM_2) * src.GetValidShape(DIM_3) && // C1v*Hv*Wv
            dst.GetValidShape(DIM_1) == (src.GetValidShape(DIM_0) + N0 - 1) / N0 &&               // N1v
            "Valid shape mismatch: NC1HWC0 to FRACTAL_Z");
    } else if constexpr (src_layout == Layout::GNCHW && dst_layout == Layout::GNC1HWC0) {
        assert(
            dst.GetValidShape(DIM_5) == C0 &&                                       // C0 is a format constant
            dst.GetValidShape(DIM_0) == src.GetValidShape(DIM_0) &&                 // G
            dst.GetValidShape(DIM_1) == src.GetValidShape(DIM_1) &&                 // N
            dst.GetValidShape(DIM_2) == (src.GetValidShape(DIM_2) + C0 - 1) / C0 && // C1v = ceil(Cv / C0)
            dst.GetValidShape(DIM_3) == src.GetValidShape(DIM_3) &&                 // H
            dst.GetValidShape(DIM_4) == src.GetValidShape(DIM_4) &&                 // W
            "Valid shape mismatch: GNCHW to GNC1HWC0");
    } else if constexpr (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::GNCHW) {
        assert(
            src.GetValidShape(DIM_5) == C0 &&                            // C0 is a format constant
            dst.GetValidShape(DIM_0) == src.GetValidShape(DIM_0) &&      // G
            dst.GetValidShape(DIM_1) == src.GetValidShape(DIM_1) &&      // N
            dst.GetValidShape(DIM_2) == src.GetValidShape(DIM_2) * C0 && // C = C1 * C0
            dst.GetValidShape(DIM_3) == src.GetValidShape(DIM_3) &&      // H
            dst.GetValidShape(DIM_4) == src.GetValidShape(DIM_4) &&      // W
            "Valid shape mismatch: GNC1HWC0 to GNCHW");
    } else if constexpr (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
        assert(
            src.GetValidShape(DIM_5) == C0 && dst.GetValidShape(DIM_3) == C0 && // C0 is a format constant
            dst.GetValidShape(DIM_2) == N0 &&                                   // N0 is a format constant
            dst.GetValidShape(DIM_0) == src.GetValidShape(DIM_0) * src.GetValidShape(DIM_2) * src.GetValidShape(DIM_3) *
                                            src.GetValidShape(DIM_4) &&             // Gv*C1v*Hv*Wv
            dst.GetValidShape(DIM_1) == (src.GetValidShape(DIM_1) + N0 - 1) / N0 && // N1v
            "Valid shape mismatch: GNC1HWC0 to FRACTAL_Z");
    } else if constexpr (src_layout == Layout::NCDHW && dst_layout == Layout::FRACTAL_Z_3D) {
        const int64_t vC0 = dst.GetValidShape(DIM_3);
        const int64_t vC1 = (src.GetValidShape(DIM_1) + vC0 - 1) / vC0;
        assert(
            vC0 == C0 && dst.GetValidShape(DIM_2) == N0 && // C0 / N0 是格式常量
            dst.GetValidShape(DIM_0) ==
                src.GetValidShape(DIM_2) * vC1 * src.GetValidShape(DIM_3) * src.GetValidShape(DIM_4) && // Dv*C1v*Hv*Wv
            dst.GetValidShape(DIM_1) == (src.GetValidShape(DIM_0) + N0 - 1) / N0 &&                     // N1v
            "Valid shape mismatch: NCDHW to FRACTAL_Z_3D");
    } else if constexpr (
        src_layout == Layout::NCHW && (dst_layout == Layout::FRACTAL_Z || dst_layout == Layout::FRACTAL_Z_3D)) {
        const int64_t vC0 = dst.GetValidShape(DIM_3);
        const int64_t vC1 = (src.GetValidShape(DIM_1) + vC0 - 1) / vC0;
        assert(
            vC0 == C0 && dst.GetValidShape(DIM_2) == N0 && // C0 / N0 是格式常量
            dst.GetValidShape(DIM_0) == vC1 * src.GetValidShape(DIM_2) * src.GetValidShape(DIM_3) && // C1v*Hv*Wv
            dst.GetValidShape(DIM_1) == (src.GetValidShape(DIM_0) + N0 - 1) / N0 &&                  // N1v
            "Valid shape mismatch: NCHW to FRACTAL_Z(_3D)");
    }
}

// NC1HWC0 (N, C1, H, W, C0) -> NCHW (N, C, H, W)
// raw 管地址（每维 stride），valid 管处理域：只读 src valid 盒、只写 dst valid 盒。
template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NC1HWC02NCHW(DstTileData& dst, SrcTileData& src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;
    auto* srcPtr = reinterpret_cast<SrcDType*>(src.data());
    auto* dstPtr = reinterpret_cast<DstDType*>(dst.data());

    // raw -> strides
    const int64_t srcWStride = src.GetShape(4);
    const int64_t srcHStride = src.GetShape(3) * srcWStride;
    const int64_t srcC1Stride = src.GetShape(2) * srcHStride;
    const int64_t srcNStride = src.GetShape(1) * srcC1Stride;
    const int64_t dstHStride = dst.GetShape(3);
    const int64_t dstCStride = dst.GetShape(2) * dstHStride;
    const int64_t dstNStride = dst.GetShape(1) * dstCStride;

    // valid -> 处理域；块宽取 src valid 的 C0（= 格式常量）
    const int64_t vN = dst.GetValidShape(0);
    const int64_t vC = dst.GetValidShape(1);
    const int64_t vH = dst.GetValidShape(2);
    const int64_t vW = dst.GetValidShape(3);
    const int64_t vC0 = src.GetValidShape(4);

    for (int64_t n = 0; n < vN; ++n) {
        for (int64_t c = 0; c < vC; ++c) {
            const int64_t c1 = c / vC0;
            const int64_t c0 = c % vC0;
            for (int64_t h = 0; h < vH; ++h) {
                for (int64_t w = 0; w < vW; ++w) {
                    SetProperDataPart(
                        dstPtr, n * dstNStride + c * dstCStride + h * dstHStride + w,
                        GetProperDataPart(
                            srcPtr, n * srcNStride + c1 * srcC1Stride + h * srcHStride + w * srcWStride + c0));
                }
            }
        }
    }
}

// GNCHW (G, N, C, H, W) -> GNC1HWC0 (G, N, C1, H, W, C0)
// dst valid 盒内没有源对应的位置（末 C1 块的 C0 尾巴）补 0；盒外不写。
template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNCHW2NC1HWC0(DstTileData& dst, SrcTileData& src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;
    auto* srcPtr = reinterpret_cast<SrcDType*>(src.data());
    auto* dstPtr = reinterpret_cast<DstDType*>(dst.data());

    // raw -> strides
    const int64_t srcHStride = src.GetShape(4);
    const int64_t srcCStride = src.GetShape(3) * srcHStride;
    const int64_t srcNStride = src.GetShape(2) * srcCStride;
    const int64_t srcGStride = src.GetShape(1) * srcNStride;
    const int64_t dstWStride = dst.GetShape(5); // GNC1HWC0 是 6 维：W 的 stride 是 C0
    const int64_t dstHStride = dst.GetShape(4) * dstWStride;
    const int64_t dstC1Stride = dst.GetShape(3) * dstHStride;
    const int64_t dstNStride = dst.GetShape(2) * dstC1Stride;
    const int64_t dstGStride = dst.GetShape(1) * dstNStride;

    // valid -> 处理域
    const int64_t vG = src.GetValidShape(0);
    const int64_t vN = src.GetValidShape(1);
    const int64_t vC = src.GetValidShape(2);
    const int64_t vH = src.GetValidShape(3);
    const int64_t vW = src.GetValidShape(4);
    const int64_t dG = dst.GetValidShape(0);
    const int64_t dN = dst.GetValidShape(1);
    const int64_t dC1 = dst.GetValidShape(2);
    const int64_t dH = dst.GetValidShape(3);
    const int64_t dW = dst.GetValidShape(4);
    const int64_t dC0 = dst.GetValidShape(5);

    for (int64_t g = 0; g < dG; ++g) {
        for (int64_t n = 0; n < dN; ++n) {
            for (int64_t c1 = 0; c1 < dC1; ++c1) {
                for (int64_t h = 0; h < dH; ++h) {
                    for (int64_t w = 0; w < dW; ++w) {
                        const int64_t baseDst =
                            g * dstGStride + n * dstNStride + c1 * dstC1Stride + h * dstHStride + w * dstWStride;
                        const int64_t baseSrc = g * srcGStride + n * srcNStride + h * srcHStride + w;
                        for (int64_t c0 = 0; c0 < dC0; ++c0) {
                            const int64_t c = c1 * dC0 + c0;
                            DstDType value = static_cast<DstDType>(0);
                            if (g < vG && n < vN && c < vC && h < vH && w < vW) {
                                value = GetProperDataPart(srcPtr, baseSrc + c * srcCStride);
                            }
                            SetProperDataPart(dstPtr, baseDst + c0, value);
                        }
                    }
                }
            }
        }
    }
}

// GNC1HWC0 (G, N, C1, H, W, C0) -> GNCHW (G, N, C1*C0, H, W)
// raw 管地址（每维 stride），valid 管处理域：只读 src valid 盒、只写 dst valid 盒。
template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNC1HWC02GNCHW(DstTileData& dst, SrcTileData& src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;
    auto* srcPtr = reinterpret_cast<SrcDType*>(src.data());
    auto* dstPtr = reinterpret_cast<DstDType*>(dst.data());
    constexpr int64_t C0 = 32 / GetTypeSize<DstDType>();

    // raw -> strides（src 最内层是 C0，dst 最内层是 W）
    const int64_t srcWStride = src.GetShape(5);
    const int64_t srcHStride = src.GetShape(4) * srcWStride;
    const int64_t srcC1Stride = src.GetShape(3) * srcHStride;
    const int64_t srcNStride = src.GetShape(2) * srcC1Stride;
    const int64_t srcGStride = src.GetShape(1) * srcNStride;
    const int64_t dstWStride = 1;
    const int64_t dstHStride = dst.GetShape(4) * dstWStride;
    const int64_t dstCStride = dst.GetShape(3) * dstHStride;
    const int64_t dstNStride = dst.GetShape(2) * dstCStride;
    const int64_t dstGStride = dst.GetShape(1) * dstNStride;

    // valid -> 处理域
    const int64_t vG = src.GetValidShape(0);
    const int64_t vN = src.GetValidShape(1);
    const int64_t vC1 = src.GetValidShape(2);
    const int64_t vH = src.GetValidShape(3);
    const int64_t vW = src.GetValidShape(4);
    const int64_t vC0 = src.GetValidShape(5);
    const int64_t dG = dst.GetValidShape(0);
    const int64_t dN = dst.GetValidShape(1);
    const int64_t dC = dst.GetValidShape(2);
    const int64_t dH = dst.GetValidShape(3);
    const int64_t dW = dst.GetValidShape(4);

    for (int64_t g = 0; g < dG; ++g) {
        for (int64_t n = 0; n < dN; ++n) {
            for (int64_t c = 0; c < dC; ++c) {
                const int64_t c1 = c / C0;
                const int64_t c0 = c % C0;
                for (int64_t h = 0; h < dH; ++h) {
                    for (int64_t w = 0; w < dW; ++w) {
                        DstDType value = static_cast<DstDType>(0);
                        if (g < vG && n < vN && c1 < vC1 && h < vH && w < vW && c0 < vC0) {
                            const int64_t srcOffset = g * srcGStride + n * srcNStride + c1 * srcC1Stride +
                                                      h * srcHStride + w * srcWStride + c0;
                            value = GetProperDataPart(srcPtr, srcOffset);
                        }
                        const int64_t dstOffset =
                            g * dstGStride + n * dstNStride + c * dstCStride + h * dstHStride + w * dstWStride;
                        SetProperDataPart(dstPtr, dstOffset, value);
                    }
                }
            }
        }
    }
}

// NCHW -> NC1HWC0.
// The declared (raw) shapes only describe memory: every dimension contributes its stride and its capacity,
// and takes no part in the format conversion. The processing domain comes from the valid shapes of both sides:
//   - the source valid box (Nv, Cv, Hv, Wv) is what gets read (always addressed with the *raw* strides);
//   - the destination valid box (Nv, C1v, Hv, Wv, C0) is what gets written, with C1v = ceil(Cv / C0) and
//     C0 = 32 / sizeof(T) being properties of the format;
//   - positions inside the destination valid box that have no source counterpart (the C0 tail of the last
//     C1 block) are zero-filled; everything outside the destination valid box is left untouched.
// For a tile that does not opt into a valid shape, GetValidShape() equals GetShape() and the loop below
// degenerates to the plain full-shape conversion.
template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NCHW2NC1HWC0(DstTileData& dst, SrcTileData& src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    auto* srcPtr = reinterpret_cast<SrcDType*>(src.data());
    auto* dstPtr = reinterpret_cast<DstDType*>(dst.data());

    // Raw (declared) extents: addressing only.
    const int64_t srcC = src.GetShape(1);
    const int64_t srcH = src.GetShape(2);
    const int64_t srcW = src.GetShape(3);
    const int64_t dstC1 = dst.GetShape(1);
    const int64_t dstH = dst.GetShape(2);
    const int64_t dstW = dst.GetShape(3);
    const int64_t dstC0 = dst.GetShape(4);

    const int64_t srcCStride = srcH * srcW;
    const int64_t srcHStride = srcW;
    const int64_t srcNStride = srcC * srcCStride;
    const int64_t dstWStride = dstC0;
    const int64_t dstHStride = dstW * dstC0;
    const int64_t dstC1Stride = dstH * dstHStride;
    const int64_t dstNStride = dstC1 * dstC1Stride;

    // Valid extents: the processing domain.
    const int64_t vC = src.GetValidShape(1);
    const int64_t dN = dst.GetValidShape(0);
    const int64_t dC1 = dst.GetValidShape(1);
    const int64_t dH = dst.GetValidShape(2);
    const int64_t dW = dst.GetValidShape(3);
    const int64_t dC0 = dst.GetValidShape(4);

    for (int64_t n = 0; n < dN; ++n) {
        for (int64_t c1 = 0; c1 < dC1; ++c1) {
            for (int64_t h = 0; h < dH; ++h) {
                for (int64_t w = 0; w < dW; ++w) {
                    const int64_t baseSrc = n * srcNStride + h * srcHStride + w;
                    const int64_t baseDst = n * dstNStride + c1 * dstC1Stride + h * dstHStride + w * dstWStride;
                    for (int64_t c0 = 0; c0 < dC0; ++c0) {
                        const int64_t c = c1 * dC0 + c0;
                        // channels beyond the source valid range are format padding inside the valid box
                        DstDType value = static_cast<DstDType>(0);
                        if (c < vC) {
                            value = GetProperDataPart(srcPtr, baseSrc + c * srcCStride);
                        }
                        SetProperDataPart(dstPtr, baseDst + c0, value);
                    }
                }
            }
        }
    }
}

// NC1HWC0 -> FRACTAL_Z
// GNC1HWC0 -> FRACTAL_Z
// 扁平维按 valid 的 (c1, h, w) 分解，因此写入范围是 dim0 上的连续前缀 [0, C1v*Hv*Wv)；
// N 方向写到 hull = N1v*N0，其中 [Nv, N1v*N0) 没有源对应，补 0。hasGroup 为 false 时 source 为
// 4D 的 NC1HWC0（隐式 G = 1）。
template <typename DstTileData, typename SrcTileData, bool hasGroup>
inline void TTRANS_GNC1HWC02C1HWN1N0C0_Impl(DstTileData& dst, SrcTileData& src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;
    auto* srcPtr = reinterpret_cast<SrcDType*>(src.data());
    auto* dstPtr = reinterpret_cast<DstDType*>(dst.data());

    constexpr int kOff = hasGroup ? 1 : 0; // G 为可选维，其余维依次后移

    // raw -> strides
    const int64_t srcWStride = src.GetShape(kOff + 4);
    const int64_t srcHStride = src.GetShape(kOff + 3) * srcWStride;
    const int64_t srcC1Stride = src.GetShape(kOff + 2) * srcHStride;
    const int64_t srcNStride = src.GetShape(kOff + 1) * srcC1Stride;
    const int64_t srcGStride = hasGroup ? src.GetShape(1) * srcNStride : 0; // G 的 stride = 其后所有维之积
    const int64_t dstN0Stride = dst.GetShape(3);
    const int64_t dstN1Stride = dst.GetShape(2) * dstN0Stride;
    const int64_t dstD0Stride = dst.GetShape(1) * dstN1Stride;

    // valid -> 处理域
    const int64_t vG = hasGroup ? src.GetValidShape(0) : 1;
    const int64_t vN = src.GetValidShape(kOff + 0);
    const int64_t vC1 = src.GetValidShape(kOff + 1);
    const int64_t vH = src.GetValidShape(kOff + 2);
    const int64_t vW = src.GetValidShape(kOff + 3);
    const int64_t d0 = dst.GetValidShape(0);
    const int64_t dN1 = dst.GetValidShape(1);
    const int64_t dN0 = dst.GetValidShape(2);
    const int64_t dC0 = dst.GetValidShape(3);

    const int64_t hw = vH * vW;
    const int64_t perGroup = vC1 * hw;

    for (int64_t n = 0; n < dN1 * dN0; ++n) {
        const int64_t n1 = n / dN0;
        const int64_t n0 = n % dN0;
        for (int64_t flat = 0; flat < d0; ++flat) {
            const int64_t g = flat / perGroup;
            const int64_t rem = flat % perGroup;
            const int64_t c1 = rem / hw;
            const int64_t hwIdx = rem % hw;
            const int64_t h = hwIdx / vW;
            const int64_t w = hwIdx % vW;
            const int64_t baseSrc =
                g * srcGStride + n * srcNStride + c1 * srcC1Stride + h * srcHStride + w * srcWStride;
            const int64_t baseDst = flat * dstD0Stride + n1 * dstN1Stride + n0 * dstN0Stride;
            for (int64_t c0 = 0; c0 < dC0; ++c0) {
                // N 方向超出 src valid 的部分是格式补零
                DstDType value = static_cast<DstDType>(0);
                if (n < vN) {
                    value = GetProperDataPart(srcPtr, baseSrc + c0);
                }
                SetProperDataPart(dstPtr, baseDst + c0, value);
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NC1HWC02C1HWN1N0C0(DstTileData& dst, SrcTileData& src)
{
    TTRANS_GNC1HWC02C1HWN1N0C0_Impl<DstTileData, SrcTileData, false>(dst, src);
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNC1HWC02C1HWN1N0C0(DstTileData& dst, SrcTileData& src)
{
    TTRANS_GNC1HWC02C1HWN1N0C0_Impl<DstTileData, SrcTileData, true>(dst, src);
}

// NCHW (N, C, H, W) -> FRACTAL_Z (C1*H*W, N1, N0, C0)
// 4D NCHW 源产出 FRACTAL_Z_3D 等价于隐式 D = 1，布局相同，共用本实现。
// 扁平维按 valid 的 (c1, h, w) 分解；N / C 方向超出 src valid 的部分（N0 尾巴、C0 尾巴）补 0。
template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NCHW2C1HWN1N0C0(DstTileData& dst, SrcTileData& src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;
    auto* srcPtr = reinterpret_cast<SrcDType*>(src.data());
    auto* dstPtr = reinterpret_cast<DstDType*>(dst.data());

    // raw -> strides
    const int64_t srcHStride = src.GetShape(3);
    const int64_t srcCStride = src.GetShape(2) * srcHStride;
    const int64_t srcNStride = src.GetShape(1) * srcCStride;
    const int64_t dstN0Stride = dst.GetShape(3);
    const int64_t dstN1Stride = dst.GetShape(2) * dstN0Stride;
    const int64_t dstD0Stride = dst.GetShape(1) * dstN1Stride;

    // valid -> 处理域
    const int64_t vN = src.GetValidShape(0);
    const int64_t vC = src.GetValidShape(1);
    const int64_t vH = src.GetValidShape(2);
    const int64_t vW = src.GetValidShape(3);
    const int64_t d0 = dst.GetValidShape(0);
    const int64_t dN1 = dst.GetValidShape(1);
    const int64_t dN0 = dst.GetValidShape(2);
    const int64_t dC0 = dst.GetValidShape(3);

    const int64_t hw = vH * vW;

    for (int64_t n = 0; n < dN1 * dN0; ++n) {
        const int64_t n1 = n / dN0;
        const int64_t n0 = n % dN0;
        for (int64_t flat = 0; flat < d0; ++flat) {
            const int64_t c1 = flat / hw;
            const int64_t hwIdx = flat % hw;
            const int64_t h = hwIdx / vW;
            const int64_t w = hwIdx % vW;
            const int64_t baseSrc = n * srcNStride + h * srcHStride + w;
            const int64_t baseDst = flat * dstD0Stride + n1 * dstN1Stride + n0 * dstN0Stride;
            for (int64_t c0 = 0; c0 < dC0; ++c0) {
                const int64_t c = c1 * dC0 + c0;
                DstDType value = static_cast<DstDType>(0);
                if (n < vN && c < vC) {
                    value = GetProperDataPart(srcPtr, baseSrc + c * srcCStride);
                }
                SetProperDataPart(dstPtr, baseDst + c0, value);
            }
        }
    }
}

// NCDHW (N, C, D, H, W) -> FRACTAL_Z_3D (D*C1*H*W, N1, N0, C0)
// 扁平维按 valid 的 (d, c1, h, w) 分解（D 在最外）；N / C 方向超出 src valid 的部分补 0。
template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NCDHW2DC1HWN1N0C0(DstTileData& dst, SrcTileData& src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;
    auto* srcPtr = reinterpret_cast<SrcDType*>(src.data());
    auto* dstPtr = reinterpret_cast<DstDType*>(dst.data());

    // raw -> strides
    const int64_t srcHStride = src.GetShape(4);
    const int64_t srcDStride = src.GetShape(3) * srcHStride;
    const int64_t srcCStride = src.GetShape(2) * srcDStride;
    const int64_t srcNStride = src.GetShape(1) * srcCStride;
    const int64_t dstN0Stride = dst.GetShape(3);
    const int64_t dstN1Stride = dst.GetShape(2) * dstN0Stride;
    const int64_t dstD0Stride = dst.GetShape(1) * dstN1Stride;

    // valid -> 处理域
    const int64_t vN = src.GetValidShape(0);
    const int64_t vC = src.GetValidShape(1);
    const int64_t vD = src.GetValidShape(2);
    const int64_t vH = src.GetValidShape(3);
    const int64_t vW = src.GetValidShape(4);
    const int64_t d0 = dst.GetValidShape(0);
    const int64_t dN1 = dst.GetValidShape(1);
    const int64_t dN0 = dst.GetValidShape(2);
    const int64_t dC0 = dst.GetValidShape(3);

    const int64_t hw = vH * vW;
    const int64_t c1v = (vC + dC0 - 1) / dC0;
    const int64_t perD = c1v * hw;

    for (int64_t n = 0; n < dN1 * dN0; ++n) {
        const int64_t n1 = n / dN0;
        const int64_t n0 = n % dN0;
        for (int64_t flat = 0; flat < d0; ++flat) {
            const int64_t d = flat / perD;
            const int64_t remD = flat % perD;
            const int64_t c1 = remD / hw;
            const int64_t hwIdx = remD % hw;
            const int64_t h = hwIdx / vW;
            const int64_t w = hwIdx % vW;
            const int64_t baseSrc = n * srcNStride + d * srcDStride + h * srcHStride + w;
            const int64_t baseDst = flat * dstD0Stride + n1 * dstN1Stride + n0 * dstN0Stride;
            for (int64_t c0 = 0; c0 < dC0; ++c0) {
                const int64_t c = c1 * dC0 + c0;
                DstDType value = static_cast<DstDType>(0);
                if (n < vN && c < vC) {
                    value = GetProperDataPart(srcPtr, baseSrc + c * srcCStride);
                }
                SetProperDataPart(dstPtr, baseDst + c0, value);
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
void TTrans_Impl(DstTileData& dst, SrcTileData& src)
{
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    for (size_t c = 0; c < validCol; c++) {
        for (size_t r = 0; r < validRow; r++) {
            dst.SetElement(c, r, src.GetElement(r, c));
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TTRANS_CONV_IMPL(DstTileData& dst, SrcTileData& src)
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
    } else if constexpr (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::GNCHW) {
        TTRANS_GNC1HWC02GNCHW(dst, src);
    } else if constexpr (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
        TTRANS_GNC1HWC02C1HWN1N0C0(dst, src);
    } else if constexpr (src_layout == Layout::NCDHW && dst_layout == Layout::FRACTAL_Z_3D) {
        TTRANS_NCDHW2DC1HWN1N0C0(dst, src);
    } else if constexpr (
        src_layout == Layout::NCHW && (dst_layout == Layout::FRACTAL_Z || dst_layout == Layout::FRACTAL_Z_3D)) {
        TTRANS_NCHW2C1HWN1N0C0(dst, src);
    }
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData>
PTO_INTERNAL void TTRANS_IMPL(DstTileData& dst, SrcTileData& src, TmpTileData& tmp)
{
    // Validate matching element widths at compilation
    static_assert(
        sizeof(typename SrcTileData::DType) == sizeof(typename DstTileData::DType),
        "Data type sizes between source and destination tiles must match.");

    if constexpr (is_conv_tile_v<SrcTileData> && is_conv_tile_v<DstTileData>) {
        TTRANS_CONV_IMPL(dst, src);
    } else if constexpr (is_tile_data_v<SrcTileData> && is_tile_data_v<DstTileData>) {
        static_assert(
            SrcTileData::ValidRow == DstTileData::ValidCol && SrcTileData::ValidCol == DstTileData::ValidRow,
            "Hardware matrix tiles transpose dimension sizes must mirror match.");
        TTrans_Impl<DstTileData, SrcTileData>(dst, src);
    }
}

} // namespace pto

#endif // TTRANS_HPP
