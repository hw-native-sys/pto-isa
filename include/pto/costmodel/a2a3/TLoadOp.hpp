/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOADOP_HPP
#define TLOADOP_HPP

namespace pto {

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2ub(std::vector<CostModelStats> &stats, uint16_t nBurst, uint32_t lenBurst,
                                  uint32_t gmGap, uint32_t ubGap, uint32_t ubPad)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b8", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad);
        stats.emplace_back(costModelStats);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b16", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad);
        stats.emplace_back(costModelStats);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b32", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad);
        stats.emplace_back(costModelStats);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b32", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad * 2);
        stats.emplace_back(costModelStats);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2L1(std::vector<CostModelStats> &stats, uint16_t nBurst, uint16_t lenBurst,
                                  uint16_t gmGap, uint16_t l1Gap)
{
    CostModelStats costModelStats("copy_gm_to_cbuf", nBurst, lenBurst, gmGap, l1Gap);
    costModelStats.setPadMode(0);
    stats.emplace_back(costModelStats);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNd2nd(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    static_assert(TileData::Rows < 4096, "Fix: TLOAD Rows>=4095 not supported in A2/A3");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for A2/A3");
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint16_t nBurst = gShape3;
    uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
    uint64_t gmGapValue = (gStride3 - gShape4) * sizeof(typename TileData::DType);
    uint32_t gmGap = (uint32_t)gmGapValue;
    uint32_t ubGapElement = (TileData::Cols - validCol);
    uint32_t ubGap = (ubGapElement * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t ubPad = 0;
    if constexpr (TileData::PadVal != PadValue::Null) {
        ubPad = ubGapElement % blockSizeElem;
    }

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2ub<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, ubGap, ubPad);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubDn2dn(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for A2/A3");
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint16_t nBurst = gShape4;
    uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
    uint64_t gmGapValue = (gStride4 - gShape3) * sizeof(typename TileData::DType);
    uint32_t gmGap = (uint32_t)gmGapValue;
    uint32_t ubGapEle = (TileData::Rows - gShape3);
    uint32_t ubGap = (ubGapEle * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t ubPadTmp = 0;
    if constexpr (TileData::PadVal != PadValue::Null) {
        ubPadTmp = ubGapEle % blockSizeElem;
    }

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2ub<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, ubGap, ubPadTmp);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNz2nz(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
    uint32_t ubGap = TileData::Rows - validRow;
    for (uint32_t i = 0; i < gShape0; i++) {
        TLoadInstrGm2ub<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, ubGap, 0);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ub(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2, int gShape3,
                             int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
                             int validRow, int validCol)
{
    if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        TLoadGm2ubNd2nd<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1,
                                              gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        TLoadGm2ubDn2dn<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1,
                                              gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2ubNz2nz<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1,
                                              gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nd2nd(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(gShape4 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
               "The 5th dim of ND shape must be 32 bytes aligned!");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape3;
    uint16_t lenBurst = (validCol * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride3 - gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Cols - validCol) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Dn2dn(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(gShape3 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
               "The 4th dim of DN shape must be 32 bytes aligned!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape4;
    uint16_t lenBurst = (validRow * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride4 - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Rows - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nz2nz(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow;
    uint32_t gmGap = ((gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t l1Gap = TileData::Rows - validRow;

    for (uint32_t i = 0; i < gShape0; i++) {
        TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, l1Gap);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoad5HD(std::vector<CostModelStats> &stats, int srcN, int srcC1, int srcH, int srcW, int gStride0,
                           int gStride1, int gStride2, int gStride3, int gStride4, int dstN, int dstC1, int dstH,
                           int dstW)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t maxSupportBurst = 4095;
    // gmGap unit is 32B
    uint32_t gmGap = ((gStride1 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride2 == dstW * c0ElemCount || dstH == 1) && // process for W direction all load or H=1
        gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst && dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint16_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, srcGap, 0);
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint16_t srcGap = ((gStride2 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            for (uint32_t j = 0; j < dstC1; j++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, srcGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadFractalZ(std::vector<CostModelStats> &stats, int srcShape0, int srcShape1, int srcShape2,
                                int srcShape3, int srcShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                int gStride4, int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    if constexpr (TileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
                      "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
                      "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

        uint16_t nBurst = dstShape0;
        uint16_t lenBurst = dstShape1 * dstShape2;
        uint16_t gmGap =
            ((gStride1 - srcShape2 * srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, 0);

    } else { //  [C1,H,W,N,C0]
        PTO_ASSERT(srcShape1 == dstShape1 && srcShape2 == dstShape2,
                   "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
        PTO_ASSERT(dstShape3 <= UINT16_MAX, "Fix: max support dstN is UINT16_MAX!");

        uint16_t lenBurst = dstShape3;
        uint16_t gmGap = ((gStride2 - srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        constexpr uint32_t maxSupportBurst = 4095;

        if (dstShape0 * dstShape1 * dstShape2 <= maxSupportBurst) { // if burst <= 4095, only load once
            uint16_t nBurst = dstShape0 * dstShape1 * dstShape2;
            TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, 0);
        } else {
            uint16_t nBurst = dstShape1 * dstShape2;
            for (uint32_t i = 0; i < dstShape0; i++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, 0);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(std::vector<CostModelStats> &stats, TileData &dst, GlobalData &src)
{
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) { // layout is NC1HWC0, dst dim4 is c0Size
        TLoad5HD<TileData, GlobalData>(stats, src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                                       src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
                                       src.GetStride(4), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2),
                                       dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) { // C1HWNC0, dst dim4 is c0Size
        TLoadFractalZ<TileData, GlobalData>(stats, src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                                            src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2),
                                            src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1),
                                            dst.GetShape(2), dst.GetShape(3));
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL std::vector<CostModelStats> runTLoadOp(TileData &dst, GlobalData &src)
{
    std::vector<CostModelStats> stats;
    TLOAD_CONVTILE_IMPL(stats, dst, src);
    return stats;
}
} // namespace pto
#endif