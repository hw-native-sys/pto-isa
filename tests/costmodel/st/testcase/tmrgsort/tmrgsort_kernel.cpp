/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace std;
using namespace pto;

// TMRGSORT cycle formulas:
//   Multi-src (2/3/4 sources): vmrgsort4(1) = startup(14) + 1*2 = 16
//   Single-src: R = kTCols_ / (blockLen * 4), vmrgsort4(R) = 14 + R*2
//   TopK: last TMRGSORT on dstTile is always multi-src → 16

// ─── Multi-src helpers ────────────────────────────────────────────────────────

template <typename DstTileData, typename TmpTileData, typename TileData, bool EXHAUSTED>
PTO_INTERNAL void Sort2Lists(DstTileData &dstTile, TileData &src0Tile, TileData &src1Tile, TmpTileData &tmpTile)
{
    MrgSortExecutedNumList executedNumList;
    TMRGSORT<DstTileData, TmpTileData, TileData, TileData, EXHAUSTED>(dstTile, executedNumList, tmpTile, src0Tile,
                                                                      src1Tile);
}

template <typename DstTileData, typename TmpTileData, typename TileData, bool EXHAUSTED>
PTO_INTERNAL void Sort3Lists(DstTileData &dstTile, TileData &src0Tile, TileData &src1Tile, TileData &src2Tile,
                             TmpTileData &tmpTile)
{
    MrgSortExecutedNumList executedNumList;
    TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, EXHAUSTED>(dstTile, executedNumList, tmpTile,
                                                                                src0Tile, src1Tile, src2Tile);
}

template <typename DstTileData, typename TmpTileData, typename TileData, bool EXHAUSTED>
PTO_INTERNAL void Sort4Lists(DstTileData &dstTile, TileData &src0Tile, TileData &src1Tile, TileData &src2Tile,
                             TileData &src3Tile, TmpTileData &tmpTile)
{
    MrgSortExecutedNumList executedNumList;
    TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, EXHAUSTED>(
        dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
}

// ─── Multi-src kernel ─────────────────────────────────────────────────────────

template <typename T, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3, int TOPK, int LISTNUM,
          bool EXHAUSTED, float profiling, float accuracy>
AICORE void RunTMrgsort()
{
    using TileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, 1, TOPK, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, kTCols_ * LISTNUM, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(1, kTCols_);
    TileData src1Tile(1, kTCols_src1);
    DstTileData dstTile(1, TOPK);
    TmpTileData tmpTile(1, kTCols_ * LISTNUM);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x0 + kTCols_ * sizeof(T));
    TASSIGN(dstTile, 0x0 + (kTCols_ + kTCols_src1) * sizeof(T));
    TASSIGN(tmpTile, 0x0 + (kTCols_ + kTCols_src1 + TOPK) * sizeof(T));

    if constexpr (LISTNUM == 4) {
        TileData src2Tile(1, kTCols_src2);
        TileData src3Tile(1, kTCols_src3);
        TASSIGN(src2Tile, 0x0 + (kTCols_ + kTCols_src1 + kTCols_src2) * sizeof(T));
        TASSIGN(src3Tile, 0x0 + (kTCols_ + kTCols_src1 + kTCols_src2 + kTCols_src3) * sizeof(T));
        Sort4Lists<DstTileData, TmpTileData, TileData, EXHAUSTED>(dstTile, src0Tile, src1Tile, src2Tile, src3Tile,
                                                                  tmpTile);
    } else if constexpr (LISTNUM == 3) {
        TileData src2Tile(1, kTCols_src2);
        TASSIGN(src2Tile, 0x0 + (kTCols_ + kTCols_src1 + kTCols_src2) * sizeof(T));
        Sort3Lists<DstTileData, TmpTileData, TileData, EXHAUSTED>(dstTile, src0Tile, src1Tile, src2Tile, tmpTile);
    } else {
        Sort2Lists<DstTileData, TmpTileData, TileData, EXHAUSTED>(dstTile, src0Tile, src1Tile, tmpTile);
    }

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

// ─── Single-src kernel ────────────────────────────────────────────────────────

template <typename T, int kTCols_, uint32_t blockLen, float profiling, float accuracy>
AICORE void RunTMrgsortSingle()
{
    using TileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(1, kTCols_);
    TileData dstTile(1, kTCols_);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + kTCols_ * sizeof(T));

    TMRGSORT<TileData, TileData>(dstTile, srcTile, blockLen);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

// ─── TopK kernel ──────────────────────────────────────────────────────────────

template <typename T, int kTCols_, int topk, float profiling, float accuracy>
AICORE void RunTMrgsortTopk()
{
    using TileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, 1, topk, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(1, kTCols_);
    DstTileData dstTile(1, topk);
    TmpTileData tmpTile(1, kTCols_);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + kTCols_ * sizeof(T));
    TASSIGN(tmpTile, 0x0 + kTCols_ * sizeof(T) + topk * sizeof(T));

    // Each iteration: merge sorted blocks into tmpTile
    uint32_t blockLen = 64;
    for (; blockLen * 4 <= static_cast<uint32_t>(kTCols_); blockLen *= 4) {
        TMRGSORT<TmpTileData, TileData>(tmpTile, srcTile, blockLen);
    }

    if (blockLen < static_cast<uint32_t>(kTCols_)) {
        // Tail: final 2-src merge writes to dstTile
        using Src0TileData = Tile<TileType::Vec, T, 1, topk, BLayout::RowMajor, -1, -1>;
        using Src1TileData = Tile<TileType::Vec, T, 1, topk, BLayout::RowMajor, -1, -1>;
        Src0TileData src0Tile(1, topk);
        Src1TileData src1Tile(1, topk);
        TASSIGN(src0Tile, 0x0 + (kTCols_ * 2 + topk) * sizeof(T));
        TASSIGN(src1Tile, 0x0 + (kTCols_ * 2 + topk * 2) * sizeof(T));
        TmpTileData tmp1Tile(1, kTCols_);
        TASSIGN(tmp1Tile, 0x0 + (kTCols_ * 3 + topk * 2) * sizeof(T));
        MrgSortExecutedNumList executedNumList;
        TMRGSORT<DstTileData, TmpTileData, Src0TileData, Src1TileData, 0>(dstTile, executedNumList, tmp1Tile, src0Tile,
                                                                          src1Tile);
    } else {
        // blockLen == kTCols_: data fully sorted in tmpTile, propagate cycle
        dstTile.SetCycle(tmpTile.GetCycle());
    }

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

// ─── Launch functions ─────────────────────────────────────────────────────────

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM, bool EXHAUSTED, float profiling, float accuracy>
void LanchTMrgsortMulti(void *stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        RunTMrgsort<half, kTCols_ * TYPE_COEF, kTCols_src1 * TYPE_COEF, kTCols_src2 * TYPE_COEF,
                    kTCols_src3 * TYPE_COEF, TOPK * TYPE_COEF, LISTNUM, EXHAUSTED, profiling, accuracy>();
    } else {
        RunTMrgsort<T, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM, EXHAUSTED, profiling, accuracy>();
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen, float profiling,
          float accuracy>
void LanchTMrgsortSingle(void *stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        RunTMrgsortSingle<half, kTCols_ * TYPE_COEF, blockLen * TYPE_COEF, profiling, accuracy>();
    } else {
        RunTMrgsortSingle<T, kTCols_, blockLen, profiling, accuracy>();
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk, float profiling, float accuracy>
void LanchTMrgsortTopK(void *stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        RunTMrgsortTopk<half, kTCols_ * TYPE_COEF, topk * TYPE_COEF, profiling, accuracy>();
    } else {
        RunTMrgsortTopk<T, kTCols_, topk, profiling, accuracy>();
    }
}

// ─── Template instantiations ──────────────────────────────────────────────────

// multi case: vmrgsort4(1) = costmodel=20
template void LanchTMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 128, 512, 4, false, 20.0f, 1.0f>(void *stream);
template void LanchTMrgsortMulti<uint16_t, 1, 128, 1, 128, 128, 128, 128, 512, 4, false, 20.0f, 1.0f>(void *stream);
// multi exhausted case
template void LanchTMrgsortMulti<float, 1, 64, 1, 64, 64, 0, 0, 128, 2, true, 20.0f, 1.0f>(void *stream);
template void LanchTMrgsortMulti<uint16_t, 1, 256, 1, 256, 256, 256, 0, 768, 3, true, 20.0f, 1.0f>(void *stream);
// single case: costmodel output
// case_single1: float,    kTCols=256, blockLen=64:  costmodel=20
template void LanchTMrgsortSingle<float, 1, 256, 1, 256, 64, 20.0f, 1.0f>(void *stream);
// case_single3: float,    kTCols=512, blockLen=64:  costmodel=26
template void LanchTMrgsortSingle<float, 1, 512, 1, 512, 64, 26.0f, 1.0f>(void *stream);
// case_single5: uint16_t, kTCols=256, blockLen=64:  costmodel=20
template void LanchTMrgsortSingle<uint16_t, 1, 256, 1, 256, 64, 20.0f, 1.0f>(void *stream);
// case_single7: uint16_t, kTCols=512, blockLen=64:  costmodel=26
template void LanchTMrgsortSingle<uint16_t, 1, 512, 1, 512, 64, 26.0f, 1.0f>(void *stream);
// case_single8: uint16_t, kTCols=1024, blockLen=256: costmodel=20
template void LanchTMrgsortSingle<uint16_t, 1, 1024, 1, 1024, 256, 20.0f, 1.0f>(void *stream);
// topk case: final TMRGSORT on dstTile costmodel=20
template void LanchTMrgsortTopK<float, 1, 2048, 1, 2048, 2048, 20.0f, 1.0f>(void *stream);
template void LanchTMrgsortTopK<uint16_t, 1, 2048, 1, 2048, 2048, 20.0f, 1.0f>(void *stream);
