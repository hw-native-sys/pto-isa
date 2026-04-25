/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP
#define TLOAD_HPP

#include <unistd.h>
#include <cassert>
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename TileData>
AICORE constexpr typename TileData::DType getPadValue()
{
    switch (TileData::PadVal) {
        case PadValue::Null:
        case PadValue::Zero:
            return typename TileData::DType(0);
        case PadValue::Min:
            if constexpr (std::numeric_limits<typename TileData::DType>::has_infinity) {
                return -std::numeric_limits<typename TileData::DType>::infinity();
            } else {
                return std::numeric_limits<typename TileData::DType>::min();
            }
        case PadValue::Max:
            if constexpr (std::numeric_limits<typename TileData::DType>::has_infinity) {
                return std::numeric_limits<typename TileData::DType>::infinity();
            } else {
                return std::numeric_limits<typename TileData::DType>::max();
            }
    }
    return 0;
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INLINE void LoadPlainMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                                       int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol,
                                       size_t idx3)
{
    size_t offsetDstBase, shape, shapeBase, strideBase, stride;
    if constexpr (TileData::isRowMajor) {
        offsetDstBase = idx3 * gShape3 * TileData::Cols;
        shape = static_cast<std::size_t>(gShape4);
        stride = static_cast<std::size_t>(gStride4);
        shapeBase = static_cast<std::size_t>(gShape3);
        strideBase = static_cast<std::size_t>(gStride3);
    } else {
        offsetDstBase = idx3 * gShape4 * TileData::Rows;
        shape = static_cast<std::size_t>(gShape3);
        stride = static_cast<std::size_t>(gStride3);
        shapeBase = static_cast<std::size_t>(gShape4);
        strideBase = static_cast<std::size_t>(gStride4);
    }
    cpu::parallel_for_1d(0, shapeBase, static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t r) {
        const std::size_t dstBase = offsetDstBase + r * (TileData::isRowMajor ? TileData::Cols : TileData::Rows);
        const std::size_t srcBase = r * strideBase;
        PTO_CPU_VECTORIZE_LOOP
        for (std::size_t c = 0; c < shape; c++) {
            dst[dstBase + c] = getProperDataPart(src, srcBase + c * stride);
        }
    });
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INLINE void LoadPlainDnClassic(typename GlobalData::DType __out__ *dst,
                                          typename TileData::TileDType __in__ src, int gShape0, int gShape1,
                                          int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                          int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    int64_t dstStride1 = gShape2;
    int64_t dstStride0 = gShape1 * dstStride1;

    for (uint32_t i = 0; i < static_cast<uint32_t>(gShape0); i++) {
        int64_t dstAddr0 = static_cast<int64_t>(i) * dstStride0;
        int64_t srcAddr0 = static_cast<int64_t>(i) * gStride0;
        for (uint32_t j = 0; j < static_cast<uint32_t>(gShape1); j++) {
            int64_t dstAddr1 = static_cast<int64_t>(j) * dstStride1;
            int64_t srcAddr1 = static_cast<int64_t>(j) * gStride1;
            for (uint32_t k = 0; k < static_cast<uint32_t>(gShape2); k++) {
                size_t offsetSrcBase = srcAddr0 + srcAddr1 + static_cast<int64_t>(k) * gStride2;
                LoadPlainMatrix<GlobalData, TileData>(dst, src + offsetSrcBase, gShape3, gShape4, gStride3, gStride4,
                                                      validRow, validCol, dstAddr0 + dstAddr1 + k);
            }
        }
    }
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INLINE void LoadPlainDnFlattenRows(typename GlobalData::DType __out__ *dst,
                                              typename TileData::TileDType __in__ src, int gShape0, int gShape1,
                                              int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                              int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    (void)validRow;
    (void)validCol;
    for (uint32_t i = 0; i < static_cast<uint32_t>(gShape0); i++) {
        const std::size_t srcAddr0 = static_cast<std::size_t>(i) * static_cast<std::size_t>(gStride0);
        const std::size_t dstAddr0 =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(gShape1) * gShape2 * gShape3;
        for (uint32_t j = 0; j < static_cast<uint32_t>(gShape1); j++) {
            const std::size_t srcAddr1 = static_cast<std::size_t>(j) * static_cast<std::size_t>(gStride1);
            const std::size_t dstAddr1 = static_cast<std::size_t>(j) * static_cast<std::size_t>(gShape2) * gShape3;
            for (uint32_t k = 0; k < static_cast<uint32_t>(gShape2); k++) {
                const std::size_t srcAddr2 = static_cast<std::size_t>(k) * static_cast<std::size_t>(gStride2);
                const std::size_t dstRowBase = dstAddr0 + dstAddr1 + static_cast<std::size_t>(k) * gShape3;
                for (std::size_t c = 0; c < static_cast<std::size_t>(gShape4); c++) {
                    const std::size_t dstBase = c * static_cast<std::size_t>(TileData::Rows) + dstRowBase;
                    const std::size_t srcBase = srcAddr0 + srcAddr1 + srcAddr2 + c * static_cast<std::size_t>(gStride4);
                    PTO_CPU_VECTORIZE_LOOP
                    for (std::size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                        dst[dstBase + r] = src[srcBase + r * static_cast<std::size_t>(gStride3)];
                    }
                }
            }
        }
    }
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INLINE void LoadPlain(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                                 int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                 int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    if constexpr (!TileData::isRowMajor) {
        const int64_t flattenedRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * static_cast<int64_t>(gShape3);
        if (flattenedRows == validRow && gShape4 == validCol) {
            LoadPlainDnFlattenRows<GlobalData, TileData>(dst, src, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                         gStride0, gStride1, gStride2, gStride3, gStride4, validRow,
                                                         validCol);
            return;
        }
        LoadPlainDnClassic<GlobalData, TileData>(dst, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                 gStride1, gStride2, gStride3, gStride4, validRow, validCol);
        return;
    }

    int64_t dstStride1 = gShape2;
    int64_t dstStride0 = gShape1 * dstStride1;

    for (uint32_t i = 0; i < static_cast<uint32_t>(gShape0); i++) {
        int64_t dstAddr0 = static_cast<int64_t>(i) * dstStride0;
        int64_t srcAddr0 = static_cast<int64_t>(i) * gStride0;
        for (uint32_t j = 0; j < static_cast<uint32_t>(gShape1); j++) {
            int64_t dstAddr1 = static_cast<int64_t>(j) * dstStride1;
            int64_t srcAddr1 = static_cast<int64_t>(j) * gStride1;
            for (uint32_t k = 0; k < static_cast<uint32_t>(gShape2); k++) {
                size_t offsetSrcBase = srcAddr0 + srcAddr1 + static_cast<int64_t>(k) * gStride2;
                LoadPlainMatrix<GlobalData, TileData>(dst, src + offsetSrcBase, gShape3, gShape4, gStride3, gStride4,
                                                      validRow, validCol, dstAddr0 + dstAddr1 + k);
            }
        }
    }
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INLINE void LoadSubfractalMatrix(typename GlobalData::DType __out__ *dst,
                                            typename TileData::TileDType __in__ src, int gShape3, int gShape4,
                                            int gStride3, int gStride4, int validRow, int validCol)
{
    cpu::parallel_for_1d(
        0, static_cast<std::size_t>(gShape4), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t c) {
            size_t subTileC = c / TileData::InnerCols;
            size_t innerC = c % TileData::InnerCols;
            for (size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                size_t subTileR = r / TileData::InnerRows;
                size_t innerR = r % TileData::InnerRows;
                size_t tile_idx;
                tile_idx = GetTileElementOffsetSubfractals<TileData>(subTileR, innerR, subTileC, innerC);

                size_t gd_idx = r * static_cast<std::size_t>(gStride3) + c * static_cast<std::size_t>(gStride4);
                dst[tile_idx] = getProperDataPart(src, gd_idx);
            }
        });
}

template <typename TileData, typename GlobalData>
__tf__ AICORE void TLoad(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src, int gShape0,
                         int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
                         int gStride3, int gStride4, int validRow, int validCol)
{
    const bool isFlattenRowsColMajor =
        !TileData::isRowMajor &&
        static_cast<int64_t>(gShape0) * gShape1 * gShape2 * static_cast<int64_t>(gShape3) == validRow &&
        gShape4 == validCol;
    assert((static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3 == validRow && gShape4 == validCol &&
            TileData::isRowMajor) ||
           (static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape4 == validCol && gShape3 == validRow &&
            !TileData::isRowMajor) ||
           isFlattenRowsColMajor);

    // Filling padding
    std::fill(dst, dst + (TileData::Cols * TileData::Rows), getPadValue<TileData>());

    // Filling data
    if (TileData::SFractal == SLayout::NoneBox) {
        LoadPlain<GlobalData, TileData>(dst, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1,
                                        gStride2, gStride3, gStride4, validRow, validCol);
    } else {
        assert(gShape0 == 1 && gShape1 == 1 && gShape2 == 1 && "ND,DN -> Nz,Zn convertion does support only 2D GMs");
        LoadSubfractalMatrix<GlobalData, TileData>(dst, src, gShape3, gShape4, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData &dst, GlobalData &src)
{
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Source dtype must be same with dst dtype");
    static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN,
                  "Only ND and DN GLobal Tensors are currently supported");
    TLoad<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
                                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2L1(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t gmGap, uint16_t l1Gap)
{
    // Use reinterpret_cast for pointer reinterpretation
    uint8_t *dP = reinterpret_cast<uint8_t *>(dst);
    const uint8_t *sP = reinterpret_cast<const uint8_t *>(src);

    const uint32_t blockSize = C0_SIZE_BYTE;

    uint16_t srcStride = (static_cast<size_t>(lenBurst) + gmGap) * blockSize / sizeof(typename TileData::DType);
    uint16_t dstStride = (static_cast<size_t>(lenBurst) + l1Gap) * blockSize / sizeof(typename TileData::DType);
    uint8_t elemNum = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    for (uint16_t i = 0; i < nBurst; i++) {
        for (size_t j = 0; j < lenBurst * elemNum; j++) {
            dst[dstStride * i + j] = getProperDataPart(src, srcStride * i + j);
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoad5HD(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                  int srcN, int srcC1, int srcH, int srcW, int gStride0, int gStride1, int gStride2,
                                  int gStride3, int gStride4, int dstN, int dstC1, int dstH, int dstW)
{
    __cbuf__ typename TileData::DType *dstAddr = dst;
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    constexpr uint32_t maxSupportBurst = 4095;
    // gmGap unit is 32B
    uint32_t gmGap = ((gStride1 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride2 == dstW * c0ElemCount || dstH == 1) && // process for W direction all load or H=1
        gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst && dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint16_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * dstH * dstW * dstC1 * c0ElemCount;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, 0);
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint16_t srcGap = ((gStride2 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t dstAddr1 = i * dstH * dstW * dstC1 * c0ElemCount;
            int64_t srcAddr1 = i * gStride0;
            for (uint32_t j = 0; j < dstC1; j++) {
                srcAddrP = srcAddr + srcAddr1 + j * gStride1;
                dstAddrP = dstAddr + dstAddr1 + j * dstH * dstW * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadFractalZ(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                       int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
                                       int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
                                       int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    __cbuf__ typename TileData::DType *dstAddr = dst;
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;

    if constexpr (TileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
                      "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
                      "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

        uint16_t nBurst = dstShape0;
        uint16_t lenBurst = dstShape1 * dstShape2;
        uint16_t gmGap =
            ((gStride1 - srcShape2 * srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);

    } else { //  [C1,H,W,N,C0]
        PTO_ASSERT(srcShape1 == dstShape1 && srcShape2 == dstShape2,
                   "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
        PTO_ASSERT(dstShape3 <= UINT16_MAX, "Fix: max support dstN is UINT16_MAX!");

        uint16_t lenBurst = dstShape3;
        uint16_t gmGap = ((gStride2 - srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        constexpr uint32_t maxSupportBurst = 4095;

        if (dstShape0 * dstShape1 * dstShape2 <= maxSupportBurst) { // if burst <= 4095, only load once
            uint16_t nBurst = dstShape0 * dstShape1 * dstShape2;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
        } else {
            uint16_t nBurst = dstShape1 * dstShape2;
            for (uint32_t i = 0; i < dstShape0; i++) {
                srcAddrP = srcAddr + i * gStride0;
                dstAddrP = dstAddr + i * dstShape1 * dstShape2 * dstShape3 * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckConvTileData(TileData &dst, GlobalData &src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z);
    static_assert(isSameLayout == true, "Fix: Src Dst layout must be NC1HWC0 or FRACTAL_Z!");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(TileData &dst, GlobalData &src)
{
    CheckConvTileData<TileData, GlobalData>(dst, src);
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) { // layout is NC1HWC0, dst dim4 is c0Size
        TLoad5HD<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                       src.GetShape(3), src.GetStride(0), src.GetStride(1), src.GetStride(2),
                                       src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1),
                                       dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) { // C1HWNC0, dst dim4 is c0Size
        TLoadFractalZ<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                            src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1),
                                            src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}

} // namespace pto
#endif
