/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;

template <
    typename GlobalData, typename DstTileData, typename TmpTileData, typename TileData, typename T, bool EXHAUSTED>
PTO_INTERNAL void Sort2Lists(
    DstTileData& dstTile, GlobalData& src0Global, GlobalData& src1Global, TileData& src0Tile, TileData& src1Tile,
    TmpTileData& tmpTile)
{
    MrgSortExecutedNumList executedNumList;
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMRGSORT<DstTileData, TmpTileData, TileData, TileData, EXHAUSTED>(
        dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
}

template <
    typename GlobalData, typename DstTileData, typename TmpTileData, typename TileData, typename T, bool EXHAUSTED>
PTO_INTERNAL void Sort3Lists(
    DstTileData& dstTile, GlobalData& src0Global, GlobalData& src1Global, GlobalData& src2Global, TileData& src0Tile,
    TileData& src1Tile, TileData& src2Tile, TmpTileData& tmpTile)
{
    MrgSortExecutedNumList executedNumList;
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(src2Tile, src2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, EXHAUSTED>(
        dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
}

template <
    typename GlobalData, typename DstTileData, typename TmpTileData, typename TileData, typename T, bool EXHAUSTED>
PTO_INTERNAL void Sort4Lists(
    DstTileData& dstTile, GlobalData& src0Global, GlobalData& src1Global, GlobalData& src2Global,
    GlobalData& src3Global, TileData& src0Tile, TileData& src1Tile, TileData& src2Tile, TileData& src3Tile,
    TmpTileData& tmpTile)
{
    MrgSortExecutedNumList executedNumList;
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(src2Tile, src2Global);
    TLOAD(src3Tile, src3Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, EXHAUSTED>(
        dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
}

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3,
    int TOPK, int LISTNUM, bool EXHAUSTED>
__global__ AICORE void RunTMrgsort(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2, __gm__ T* src3)
{
    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, kGRows_, kGCols_>, pto::Stride<1, 1, 1, kGCols_, 1>>;
    using TileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_ * LISTNUM>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_ * LISTNUM, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using TmpGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, 1, TOPK, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, kTCols_ * LISTNUM, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(1, kTCols_);
    TileData src1Tile(1, kTCols_src1);
    TileData src2Tile(1, kTCols_src2);
    TileData src3Tile(1, kTCols_src3);
    DstTileData dstTile(1, TOPK);
    TmpTileData tmpTile(1, kTCols_ * LISTNUM);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x0 + kTCols_ * sizeof(T));
    TASSIGN(src2Tile, 0x0 + (kTCols_ + kTCols_src1) * sizeof(T));
    TASSIGN(src3Tile, 0x0 + (kTCols_ + kTCols_src1 + kTCols_src2) * sizeof(T));
    TASSIGN(dstTile, 0x0 + (kTCols_ + kTCols_src1 + kTCols_src2 + kTCols_src3) * sizeof(T));
    TASSIGN(tmpTile, 0x0 + (kTCols_ + kTCols_src1 + kTCols_src2 + kTCols_src3 + TOPK) * sizeof(T));

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData src2Global(src2);
    GlobalData src3Global(src3);
    DstGlobalData dstGlobal(out);

    if constexpr (LISTNUM == 4) {
        Sort4Lists<GlobalData, DstTileData, TmpTileData, TileData, T, EXHAUSTED>(
            dstTile, src0Global, src1Global, src2Global, src3Global, src0Tile, src1Tile, src2Tile, src3Tile, tmpTile);
    } else if constexpr (LISTNUM == 3) {
        Sort3Lists<GlobalData, DstTileData, TmpTileData, TileData, T, EXHAUSTED>(
            dstTile, src0Global, src1Global, src2Global, src0Tile, src1Tile, src2Tile, tmpTile);
    } else if constexpr (LISTNUM == 2) {
        Sort2Lists<GlobalData, DstTileData, TmpTileData, TileData, T, EXHAUSTED>(
            dstTile, src0Global, src1Global, src0Tile, src1Tile, tmpTile);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
__global__ AICORE void RunTMrgsortSingle(__gm__ T* out, __gm__ T* src0)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(1, kTCols_);
    DstTileData dstTile(1, kTCols_);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0 + kTCols_ * sizeof(T));

    int offset = 0;
    GlobalData src0Global(src0 + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMRGSORT<DstTileData, TileData>(dstTile, src0Tile, blockLen);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    out = dstGlobal.data();
}

template <int kTCols_>
PTO_INTERNAL int32_t FillMrgArray(int32_t* mrgArray, int blockLen)
{
    int32_t arrayCount = 0;
    int32_t tmpInner = kTCols_;
    for (int32_t i = blockLen; i >= 64; i /= 4) {
        int32_t count;
        for (count = 0; count < tmpInner / i; count++) {
            mrgArray[arrayCount++] = i;
        }
        tmpInner -= count * i;
    }
    return arrayCount;
}

template <typename T>
void DataCopy(void* dst, void* src, unsigned len)
{
    for (unsigned i = 0; i < len / sizeof(T); i++) {
        (static_cast<T*>(dst))[i] = (static_cast<T*>(src))[i];
    }
}

template <
    typename GlobalData, typename DstGlobalData, typename DstTileData, typename TmpTileData, typename T, int kTCols_,
    int topk>
PTO_INTERNAL void SortTailBlock(
    DstGlobalData& dstGlobal, DstTileData& dstTile, __gm__ T* src, __ubuf__ T* srcAddr, int blockLen)
{
    TmpTileData tmp1Tile(1, kTCols_);
    TASSIGN(tmp1Tile, 0x0 + (kTCols_ * 2 + topk) * sizeof(T));

    int32_t mrgArray[15] = {0};
    int32_t arrayCount = FillMrgArray<kTCols_>(mrgArray, blockLen);
    uint16_t mrgSortedLen = 0;
    GlobalData srcGlobal(src);
    MrgSortExecutedNumList executedNumList;
    for (int32_t i = 0; i < arrayCount - 1; ++i) {
        using Src0TileData = Tile<TileType::Vec, T, 1, topk, BLayout::RowMajor, -1, -1>;
        using Src1TileData = Tile<TileType::Vec, T, 1, topk, BLayout::RowMajor, -1, -1>;
        mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
        uint64_t tmpMrgSortedLen = mrgSortedLen;
        uint64_t tmpMrgArray = mrgArray[i + 1];
        if (tmpMrgSortedLen > topk) {
            tmpMrgSortedLen = topk;
        }
        if (tmpMrgArray > topk) {
            tmpMrgArray = topk;
        }
        Src0TileData src0Tile(1, tmpMrgSortedLen);
        Src1TileData src1Tile(1, tmpMrgArray);
        TASSIGN(src0Tile, 0x0 + (kTCols_ * 3 + topk) * sizeof(T));
        TASSIGN(src1Tile, 0x0 + (kTCols_ * 3 + topk + tmpMrgSortedLen) * sizeof(T));
        DataCopy<T>(src0Tile.data(), (__ubuf__ void*)srcAddr, (tmpMrgSortedLen * sizeof(T)));
        pipe_barrier(PIPE_V);
        DataCopy<T>(src1Tile.data(), (__ubuf__ void*)(srcAddr + mrgSortedLen), (tmpMrgArray * sizeof(T)));
        pipe_barrier(PIPE_V);
        TMRGSORT<DstTileData, TmpTileData, Src0TileData, Src1TileData, 0>(
            dstTile, executedNumList, tmp1Tile, src0Tile, src1Tile);
        pipe_barrier(PIPE_V);
        DataCopy<T>((__ubuf__ void*)srcAddr, dstTile.data(), (topk * sizeof(T)));
        pipe_barrier(PIPE_V);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
__global__ AICORE void RunTMrgsortTopk(__gm__ T* out, __gm__ T* src)
{
    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, kGRows_, kGCols_>, pto::Stride<1, 1, 1, kGCols_, 1>>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstGlobalData = GlobalTensor<T, Shape<1, 1, 1, kGRows_, kGCols_>, pto::Stride<1, 1, 1, topk, 1>>;
    using TmpGlobalData = GlobalTensor<T, Shape<1, 1, 1, kGRows_, kGCols_>, pto::Stride<1, 1, 1, kGCols_, 1>>;
    using DstTileData = Tile<TileType::Vec, T, kTRows_, topk, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(1, kTCols_);
    DstTileData dstTile(1, topk);
    TmpTileData tmpTile(1, kTCols_);
    uint32_t dstAddr = kTCols_ * sizeof(T);
    uint32_t tmpAddr = dstAddr + topk * sizeof(T);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + dstAddr);
    TASSIGN(tmpTile, 0x0 + tmpAddr);

    GlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    uint32_t blockLen = 64;

    // 每4个合并，计算整块
    TLOAD(srcTile, srcGlobal);
    for (; blockLen * 4 <= kTCols_; blockLen *= 4) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TMRGSORT<TmpTileData, TileData>(tmpTile, srcTile, blockLen);
        pipe_barrier(PIPE_V);
        uint16_t cols = kTCols_ / (blockLen * 4) * (blockLen * 4);
        DataCopy<T>(srcTile.data(), tmpTile.data(), (cols * sizeof(T)));
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    // 合并尾块
    if (blockLen < kTCols_) {
        SortTailBlock<GlobalData, DstGlobalData, DstTileData, TmpTileData, T, kTCols_, topk>(
            dstGlobal, dstTile, src, srcTile.data(), blockLen);
    } else {
        DataCopy<T>(dstTile.data(), tmpTile.data(), (topk * sizeof(T)));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, dstTile);
    }
}

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3,
    int TOPK, int LISTNUM, bool EXHAUSTED>
void LanchTMrgsortMulti(float* out, float* src0, float* src1, float* src2, float* src3, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        RunTMrgsort<
            half, kGRows_, kGCols_ * TYPE_COEF, kTRows_, kTCols_ * TYPE_COEF, kTCols_src1 * TYPE_COEF,
            kTCols_src2 * TYPE_COEF, kTCols_src3 * TYPE_COEF, TOPK * TYPE_COEF, LISTNUM, EXHAUSTED>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<half*>(src3));
    } else {
        RunTMrgsort<
            T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM, EXHAUSTED>(
            out, src0, src1, src2, src3);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
void LanchTMrgsortSingle(float* out, float* src, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        RunTMrgsortSingle<half, kGRows_, kGCols_ * TYPE_COEF, kTRows_, kTCols_ * TYPE_COEF, blockLen * TYPE_COEF>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src));
    } else {
        RunTMrgsortSingle<T, kGRows_, kGCols_, kTRows_, kTCols_, blockLen>(out, src);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
void LanchTMrgsortTopK(float* out, float* src, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        RunTMrgsortTopk<half, kGRows_, kGCols_ * TYPE_COEF, kTRows_, kTCols_ * TYPE_COEF, topk * TYPE_COEF>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src));
    } else {
        RunTMrgsortTopk<T, kGRows_, kGCols_, kTRows_, kTCols_, topk>(out, src);
    }
}

// multi case
template void LanchTMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 128, 512, 4, false>(
    float* out, float* src0, float* src1, float* src2, float* src3, void* stream);
template void LanchTMrgsortMulti<uint16_t, 1, 128, 1, 128, 128, 128, 128, 512, 4, false>(
    float* out, float* src0, float* src1, float* src2, float* src3, void* stream);
template void LanchTMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 64, 448, 4, false>(
    float* out, float* src0, float* src1, float* src2, float* src3, void* stream);
template void LanchTMrgsortMulti<float, 1, 128, 1, 128, 128, 64, 0, 128, 3, false>(
    float* out, float* src0, float* src1, float* src2, float* src3, void* stream);
// multi exhausted case
template void LanchTMrgsortMulti<float, 1, 64, 1, 64, 64, 0, 0, 128, 2, true>(
    float* out, float* src0, float* src1, float* src2, float* src3, void* stream);
template void LanchTMrgsortMulti<uint16_t, 1, 256, 1, 256, 256, 256, 0, 768, 3, true>(
    float* out, float* src0, float* src1, float* src2, float* src3, void* stream);
// single case
template void LanchTMrgsortSingle<float, 1, 256, 1, 256, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<float, 1, 512, 1, 512, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<uint16_t, 1, 256, 1, 256, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<uint16_t, 1, 512, 1, 512, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<uint16_t, 1, 1024, 1, 1024, 256>(float* out, float* src, void* stream);

// topk case
template void LanchTMrgsortTopK<float, 1, 2048, 1, 2048, 2048>(float* out, float* src, void* stream);
template void LanchTMrgsortTopK<uint16_t, 1, 2048, 1, 2048, 2048>(float* out, float* src, void* stream);