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
#include <pto/common/constants.hpp>
#include <pto/npu/kirinX90/TInsert.hpp>

using namespace pto;

#if defined(__DAV_CUBE__)
AICORE inline void ReadbackCbufToUbuf(
    __ubuf__ void* dstUb, __cbuf__ void* srcCbuf, uint16_t burstNum, uint16_t burstLen, uint16_t srcGap, uint8_t syncId,
    uint8_t eventIdNum)
{
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUb, srcCbuf, 0, burstNum, burstLen, srcGap, 0);
    copy_cbuf_to_ubuf(dstUb, srcCbuf, 1, burstNum, burstLen, srcGap, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
}
#endif

#if defined(__DAV_VEC__)
template <typename GlobalData, typename VecTile>
AICORE inline void WaitAndStore(GlobalData& dstGlobal, VecTile& dstTile, uint8_t syncId)
{
    TSTORE(dstGlobal, dstTile);
}
#endif

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZ(__gm__ T* out, __gm__ T* src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile(Rows, Cols);
    TmpVecTile tmpTile;
    DstVecTile dstTile(Rows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t alignedRow = ((Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    constexpr uint16_t srcGap = 0;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf(
        (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, srcGap, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZPlusOne(__gm__ T* out, __gm__ T* src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<
        TileType::Vec, T, Rows + 1, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile(Rows, Cols);
    TmpVecTile tmpTile;
    DstVecTile dstTile(Rows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t alignedRow = ((Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), Rows, Cols, Rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZSplit(__gm__ T* out, __gm__ T* src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<
        TileType::Vec, T, Rows + 1, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile(Rows, Cols);
    TmpVecTile tmpTile;
    DstVecTile dstTile(Rows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t alignedRow = ((Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), Rows, Cols, Rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT<Mode>(matTile, tmpTile);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZ<T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZPlusOneKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZPlusOne<T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZSplitKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZSplit<Mode, T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, uint32_t ValidRow, uint32_t TileRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZLargeTile(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using SrcNZShapeDim5 = pto::Shape<1, Cols / c0Size, TileRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcNZStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * TileRows, TileRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcNZGlobalData = GlobalTensor<T, SrcNZShapeDim5, SrcNZStridDim5, Layout::NZ>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRow, Cols, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile;
    DstVecTile dstTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(matTile, 0x0);

    SrcNZGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    {
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(1), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < dstRepeats; ++i) {
                vsts(vreg, dstUbAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)dstUbAddr, 0, burstNum, burstLen, 0, 0);

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID2);
#endif

#if defined(__DAV_CUBE__)
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID2);
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
    set_flag(PIPE_MTE1, PIPE_V, EVENT_ID3);
#endif

#if defined(__DAV_VEC__)
    wait_flag(PIPE_MTE1, PIPE_V, EVENT_ID3);
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t ValidRow, uint32_t TileRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZLargeTileKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZLargeTile<T, ValidRow, TileRows, DstRows, Cols, IdxRow>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertNZ(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZKernel<float, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZPlusOneKernel<float, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZPlusOneKernel<float, 32, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZPlusOneKernel<int32_t, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertNZSplitKernel<TInsertMode::SPLIT2, float, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZSplitKernel<TInsertMode::SPLIT4, float, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertNZKernel<float, 64, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTInsertNZLargeTileKernel<float, 16, 32, 32, 32, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTInsertNZLargeTileKernel<float, 16, 32, 32, 32, 16><<<1, nullptr, stream>>>(out, src);
    }
}

template <
    typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
    uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVec(__gm__ T* out, __gm__ T* srcIn, __gm__ T* dstIn)
{
    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using DstStride = pto::Stride<DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using DstVec = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor>;

    SrcVec srcTile;
    DstVec dstTile;

    TASSIGN(srcTile, 0x0);
    constexpr uint32_t srcSize = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstTile, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstIn);
    DstGlobal outGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    TLOAD(dstTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TINSERT(dstTile, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
#endif
}

template <int32_t testKey>
void launchTInsertNDVec(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVec<float, 8, 8, 16, 16, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstIn));
    } else if constexpr (testKey == 2) {
        RunTInsertNDVec<float, 8, 8, 16, 16, 4, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstIn));
    } else if constexpr (testKey == 3) {
        RunTInsertNDVec<half, 16, 16, 32, 32, 8, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    } else if constexpr (testKey == 4) {
        RunTInsertNDVec<int8_t, 32, 32, 64, 64, 0, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstIn));
    } else if constexpr (testKey == 5) {
        RunTInsertNDVec<half, 16, 16, 32, 48, 4, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    } else if constexpr (testKey == 6) {
        RunTInsertNDVec<float, 8, 8, 16, 24, 3, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstIn));
    } else if constexpr (testKey == 7) {
        RunTInsertNDVec<float, 8, 8, 16, 24, 0, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstIn));
    } else if constexpr (testKey == 8) {
        RunTInsertNDVec<half, 8, 16, 16, 48, 2, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    } else if constexpr (testKey == 9) {
        RunTInsertNDVec<int8_t, 32, 32, 64, 64, 0, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstIn));
    } else if constexpr (testKey == 10) {
        RunTInsertNDVec<half, 4, 128, 8, 144, 0, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    } else if constexpr (testKey == 11) {
        RunTInsertNDVec<half, 4, 144, 8, 160, 0, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    }
}

template <
    typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t SrcValidCols, uint32_t DstRows, uint32_t DstCols,
    uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVecValid(__gm__ T* out, __gm__ T* srcIn, __gm__ T* dstIn)
{
    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using DstStride = pto::Stride<DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcLoadVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using SrcInsertVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor, SrcRows, SrcValidCols>;
    using DstVec = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor>;

    SrcLoadVec srcLoad;
    SrcInsertVec srcInsert;
    DstVec dstTile;

    TASSIGN(srcLoad, 0x0);
    TASSIGN(srcInsert, 0x0);
    constexpr uint32_t srcSize = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstTile, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstIn);
    DstGlobal outGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcLoad, srcGlobal);
    TLOAD(dstTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TINSERT(dstTile, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
#endif
}

template <int32_t testKey>
void launchTInsertNDVecValidShape(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVecValid<float, 4, 8, 5, 16, 16, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstIn));
    } else if constexpr (testKey == 2) {
        RunTInsertNDVecValid<half, 8, 16, 10, 16, 32, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    } else if constexpr (testKey == 3) {
        RunTInsertNDVecValid<int8_t, 16, 32, 20, 32, 64, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstIn));
    } else if constexpr (testKey == 4) {
        RunTInsertNDVecValid<float, 4, 8, 5, 16, 16, 2, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstIn));
    } else if constexpr (testKey == 5) {
        RunTInsertNDVecValid<half, 8, 16, 10, 16, 32, 4, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    } else if constexpr (testKey == 6) {
        RunTInsertNDVecValid<int8_t, 16, 32, 20, 32, 64, 8, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstIn));
    }
}

template <typename T, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVecScalar(__gm__ T* out, __gm__ T* srcIn, __gm__ T* dstIn)
{
    constexpr uint32_t MinAlignedCols = 32 / sizeof(T);
    using SrcShape = pto::Shape<1, 1, 1, 1, MinAlignedCols>;
    using SrcStride = pto::Stride<MinAlignedCols, MinAlignedCols, MinAlignedCols, MinAlignedCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using DstStride = pto::Stride<DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcLoadVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor>;
    using SrcInsertVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor, 1, 1>;
    using DstVec = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor>;

    SrcLoadVec srcLoad;
    SrcInsertVec srcInsert;
    DstVec dstTile;

    TASSIGN(srcLoad, 0x0);
    TASSIGN(srcInsert, 0x0);
    constexpr uint32_t srcSize = 1 * MinAlignedCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstTile, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstIn);
    DstGlobal outGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcLoad, srcGlobal);
    TLOAD(dstTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TINSERT(dstTile, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
#endif
}

template <int32_t testKey>
void launchTInsertNDVecScalar(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVecScalar<float, 16, 16, 5, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstIn));
    } else if constexpr (testKey == 2) {
        RunTInsertNDVecScalar<half, 32, 32, 10, 15><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstIn));
    } else if constexpr (testKey == 3) {
        RunTInsertNDVecScalar<int8_t, 64, 64, 20, 30><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstIn));
    }
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZUnaligned(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t AlignedRow = ((SrcRows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;

    using SrcShapeDim5 = pto::Shape<1, 1, 1, SrcRows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<
        TileType::Vec, T, AlignedRow + 1, Cols, BLayout::ColMajor, SrcRows, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using ZeroVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::RowMajor, -1, -1>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile(SrcRows, Cols);
    TmpVecTile tmpTile;
    DstVecTile dstTile(DstRows, Cols);
    ZeroVecTile zeroTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(zeroTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();
    __ubuf__ T* tmpAddr = tmpTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    {
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t tmpElements = (AlignedRow + 1) * Cols;
        constexpr uint16_t tmpRepeats =
            static_cast<uint16_t>((tmpElements + elementsPerRepeat - 1) / elementsPerRepeat);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(1), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < tmpRepeats; ++i) {
                vsts(vreg, tmpAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
            for (uint16_t i = 0; i < dstRepeats; ++i) {
                vsts(vreg, dstUbAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)dstUbAddr, 0, burstNum, burstLen, 0, 0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), SrcRows, Cols, SrcRows);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t SrcRows1, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow2>
AICORE void runTInsertNZTwoInsert(__gm__ T* out, __gm__ T* src1, __gm__ T* src2)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t AlignedRow1 = ((SrcRows1 + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t AlignedRow2 = ((SrcRows2 + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t MaxAlignedRow = (AlignedRow1 > AlignedRow2) ? AlignedRow1 : AlignedRow2;

    using Src1ShapeDim5 = pto::Shape<1, 1, 1, SrcRows1, Cols>;
    using Src1StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using Src1GlobalData = GlobalTensor<T, Src1ShapeDim5, Src1StridDim5>;

    using Src2ShapeDim5 = pto::Shape<1, 1, 1, SrcRows2, Cols>;
    using Src2StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using Src2GlobalData = GlobalTensor<T, Src2ShapeDim5, Src2StridDim5>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using Src1VecTile = Tile<TileType::Vec, T, SrcRows1, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile1 = Tile<
        TileType::Vec, T, AlignedRow1 + 1, Cols, BLayout::ColMajor, SrcRows1, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using Src2VecTile = Tile<TileType::Vec, T, SrcRows2, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile2 = Tile<
        TileType::Vec, T, AlignedRow2 + 1, Cols, BLayout::ColMajor, SrcRows2, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    Src1VecTile src1Tile(SrcRows1, Cols);
    Src2VecTile src2Tile(SrcRows2, Cols);
    TmpVecTile1 tmpTile1;
    TmpVecTile2 tmpTile2;
    DstVecTile dstTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    TASSIGN(src1Tile, 0x0);
    TASSIGN(src2Tile, 0x4000);
    TASSIGN(tmpTile1, 0x10000);
    TASSIGN(tmpTile2, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    Src1GlobalData src1Global(src1);
    Src2GlobalData src2Global(src2);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();
    __ubuf__ T* tmpAddr = tmpTile1.data();

#if defined(__DAV_VEC__)
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    {
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t tmpElements = (MaxAlignedRow + 1) * Cols;
        constexpr uint16_t tmpRepeats =
            static_cast<uint16_t>((tmpElements + elementsPerRepeat - 1) / elementsPerRepeat);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(1), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < tmpRepeats; ++i) {
                vsts(vreg, tmpAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
            for (uint16_t i = 0; i < dstRepeats; ++i) {
                vsts(vreg, dstUbAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)dstUbAddr, 0, burstNum, burstLen, 0, 0);

    pto::TMovToVecNd2Nz<T, TmpVecTile1, Src1VecTile>(tmpTile1.data(), src1Tile.data(), SrcRows1, Cols, SrcRows1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile1, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    TLOAD(src2Tile, src2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    {
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t tmpElements2 = (AlignedRow2 + 1) * Cols;
        constexpr uint16_t tmpRepeats2 =
            static_cast<uint16_t>((tmpElements2 + elementsPerRepeat - 1) / elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(1), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < tmpRepeats2; ++i) {
                vsts(vreg, tmpAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }
    pto::TMovToVecNd2Nz<T, TmpVecTile2, Src2VecTile>(tmpTile2.data(), src2Tile.data(), SrcRows2, Cols, SrcRows2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile2, static_cast<uint16_t>(IdxRow2), static_cast<uint16_t>(0));

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZUnalignedKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZUnaligned<T, SrcRows, DstRows, Cols, IdxRow>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, uint32_t SrcRows1, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow2>
__global__ AICORE void launchTInsertNZTwoInsertKernel(
    __gm__ uint64_t* out, __gm__ uint64_t* src1, __gm__ uint64_t* src2)
{
    runTInsertNZTwoInsert<T, SrcRows1, SrcRows2, DstRows, Cols, IdxRow2>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src1), reinterpret_cast<__gm__ T*>(src2));
}

template <int32_t testKey>
void launchTInsertNZUnaligned(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZUnalignedKernel<float, 15, 16, 32, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZUnalignedKernel<float, 10, 32, 32, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZUnalignedKernel<float, 10, 32, 32, 4><<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTInsertNZTwoInsert(uint64_t* out, uint64_t* src1, uint64_t* src2, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZTwoInsertKernel<float, 15, 10, 32, 32, 15><<<1, nullptr, stream>>>(out, src1, src2);
    } else if constexpr (testKey == 2) {
        launchTInsertNZTwoInsertKernel<float, 8, 8, 16, 256, 8><<<1, nullptr, stream>>>(out, src1, src2);
    }
}

template <typename T, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZOverwrite(__gm__ T* out, __gm__ T* src1, __gm__ T* src2)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t AlignedRow2 = ((SrcRows2 + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;

    using Src1ShapeDim5 = pto::Shape<1, 1, 1, DstRows, Cols>;
    using Src1StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using Src1GlobalData = GlobalTensor<T, Src1ShapeDim5, Src1StridDim5>;

    using Src2ShapeDim5 = pto::Shape<1, 1, 1, SrcRows2, Cols>;
    using Src2StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using Src2GlobalData = GlobalTensor<T, Src2ShapeDim5, Src2StridDim5>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using Src1VecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile1 = Tile<
        TileType::Vec, T, DstRows + 1, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;

    using Src2VecTile = Tile<TileType::Vec, T, SrcRows2, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile2 = Tile<
        TileType::Vec, T, AlignedRow2 + 1, Cols, BLayout::ColMajor, SrcRows2, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;

    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    Src1VecTile src1Tile(DstRows, Cols);
    Src2VecTile src2Tile(SrcRows2, Cols);
    TmpVecTile1 tmpTile1;
    TmpVecTile2 tmpTile2;
    DstVecTile dstTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    TASSIGN(src1Tile, 0x0);
    TASSIGN(src2Tile, 0x4000);
    TASSIGN(tmpTile1, 0x10000);
    TASSIGN(tmpTile2, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    Src1GlobalData src1Global(src1);
    Src2GlobalData src2Global(src2);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();
    __ubuf__ T* tmpAddr = tmpTile1.data();

#if defined(__DAV_VEC__)
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto::TMovToVecNd2Nz<T, TmpVecTile1, Src1VecTile>(tmpTile1.data(), src1Tile.data(), DstRows, Cols, DstRows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile1, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    TLOAD(src2Tile, src2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    {
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t tmpElements = (AlignedRow2 + 1) * Cols;
        constexpr uint16_t tmpRepeats =
            static_cast<uint16_t>((tmpElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(0), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < tmpRepeats; ++i) {
                vsts(vreg, tmpAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }
    pto::TMovToVecNd2Nz<T, TmpVecTile2, Src2VecTile>(tmpTile2.data(), src2Tile.data(), SrcRows2, Cols, SrcRows2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile2, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZOverwriteKernel(
    __gm__ uint64_t* out, __gm__ uint64_t* src1, __gm__ uint64_t* src2)
{
    runTInsertNZOverwrite<T, SrcRows2, DstRows, Cols, IdxRow>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src1), reinterpret_cast<__gm__ T*>(src2));
}

template <int32_t testKey>
void launchTInsertNZOverwrite(uint64_t* out, uint64_t* src1, uint64_t* src2, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZOverwriteKernel<float, 10, 32, 32, 4><<<1, nullptr, stream>>>(out, src1, src2);
    }
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZVecToVec(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using SrcShapeDim5 = pto::Shape<1, 1, 1, SrcRows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcNDTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::RowMajor, -1, -1>;
    using SrcNZTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::ColMajor, SrcRows, Cols, SLayout::RowMajor>;
    using DstNZTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcNDTile srcNDTile(SrcRows, Cols);
    SrcNZTile srcNZTile;
    DstNZTile dstNZTile(DstRows, Cols);

    TASSIGN(srcNDTile, 0x0);
    TASSIGN(srcNZTile, 0x10000);
    TASSIGN(dstNZTile, 0x20000);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcNDTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    {
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __ubuf__ T* dstAddr = dstNZTile.data();
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(1), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < dstRepeats; ++i) {
                vsts(vreg, dstAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }

    TMOV(srcNZTile, srcNDTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(dstNZTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstNZTile);
#endif
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZPlusOneVecToVec(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t AlignedSrcRow = ((SrcRows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;

    using SrcShapeDim5 = pto::Shape<1, 1, 1, SrcRows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcNDTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::RowMajor, -1, -1>;
    using SrcNZTile = Tile<
        TileType::Vec, T, AlignedSrcRow + 1, Cols, BLayout::ColMajor, SrcRows, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using DstNZTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcNDTile srcNDTile(SrcRows, Cols);
    SrcNZTile srcNZTile;
    DstNZTile dstNZTile(DstRows, Cols);

    TASSIGN(srcNDTile, 0x0);
    TASSIGN(srcNZTile, 0x10000);
    TASSIGN(dstNZTile, 0x20000);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcNDTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    {
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __ubuf__ T* dstAddr = dstNZTile.data();
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(1), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < dstRepeats; ++i) {
                vsts(vreg, dstAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }

    pto::TMovToVecNd2Nz<T, SrcNZTile, SrcNDTile>(srcNZTile.data(), srcNDTile.data(), SrcRows, Cols, SrcRows);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(dstNZTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstNZTile);
#endif
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZVecToVecKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZVecToVec<T, SrcRows, DstRows, Cols, IdxRow>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZPlusOneVecToVecKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZPlusOneVecToVec<T, SrcRows, DstRows, Cols, IdxRow>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertNZVecToVec(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZVecToVecKernel<float, 16, 16, 32, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZPlusOneVecToVecKernel<float, 16, 16, 32, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZPlusOneVecToVecKernel<float, 16, 32, 32, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZVecToVecKernel<half, 16, 16, 32, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertNZPlusOneVecToVecKernel<half, 16, 16, 32, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZVecToVecKernel<int8_t, 16, 16, 64, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertNZPlusOneVecToVecKernel<int8_t, 16, 16, 64, 0><<<1, nullptr, stream>>>(out, src);
    }
}

template <TInsertMode Mode, typename T, uint32_t ValidRow, uint32_t DstRows, uint32_t Cols>
AICORE void runTInsertNZSplitCustom(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using SrcShapeDim5 = pto::Shape<1, 1, 1, ValidRow, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, ValidRow, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<
        TileType::Vec, T, DstRows + 1, Cols, BLayout::ColMajor, ValidRow, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile(ValidRow, Cols);
    TmpVecTile tmpTile;
    DstVecTile dstTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    constexpr uint32_t srcSize = ValidRow * Cols * sizeof(T);
    constexpr uint32_t tmpOffset = ((srcSize + 0xFF) / 0x100) * 0x100;

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, tmpOffset);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    constexpr uint32_t dstUbOffset = ((tmpOffset + (DstRows + 1) * Cols * sizeof(T) + 0xFF) / 0x100) * 0x100;

    TASSIGN(dstTile, dstUbOffset);

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    // Start TLOAD (MTE2) to overlap with V-pipe zero-fill
    TLOAD(srcTile, srcGlobal);

    // Fill tmpTile UB with non-zero constant so rows beyond ValidRow carry a known value
    // after NZ conversion. TINSERT writes ALL alignedRows from tmpTile to L1.
    {
        constexpr uint32_t tmpTileBytes = burstNum * (DstRows + 1) * c0Size * sizeof(T);
        constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
        constexpr uint32_t tmpElements = tmpTileBytes / sizeof(T);
        constexpr uint16_t zeroRepeats =
            static_cast<uint16_t>((tmpElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __ubuf__ T* tmpUbAddr = tmpTile.data();
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(1), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < zeroRepeats; ++i) {
                vsts(vreg, tmpUbAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }

    // Wait for TLOAD to complete
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Barrier: ensure both TLOAD (MTE2) and zero-fill (V pipe vsts) are fully committed
    pipe_barrier(PIPE_ALL);

    // Convert ND source to NZ format in tmpTile (writes only ValidRow rows, rest stays zero)
    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), ValidRow, Cols, ValidRow);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT<Mode>(matTile, tmpTile);

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <TInsertMode Mode, typename T, uint32_t ValidRow, uint32_t DstRows, uint32_t Cols>
__global__ AICORE void launchTInsertNZSplitCustomKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZSplitCustom<Mode, T, ValidRow, DstRows, Cols>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertNZSplitCustom(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZSplitCustomKernel<TInsertMode::SPLIT2, float, 8, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZSplitCustomKernel<TInsertMode::SPLIT4, float, 8, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZSplitCustomKernel<TInsertMode::SPLIT2, float, 128, 128, 128><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZSplitCustomKernel<TInsertMode::SPLIT4, float, 128, 128, 128><<<1, nullptr, stream>>>(out, src);
    }
}

template <TInsertMode Mode, typename T, uint32_t ValidRows, uint32_t DstRows, uint32_t Cols>
AICORE void runTInsertNZTwoInputSplit(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;
    constexpr uint32_t AlignedRow = ((ValidRows + nzRow - 1) / nzRow) * nzRow;
    constexpr uint32_t c0Dim = c0Size;
    constexpr uint32_t byteCols = Cols;

    using SrcVecTile = Tile<
        TileType::Vec, T, AlignedRow + 1, Cols, BLayout::ColMajor, ValidRows, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Dim, DstRows / nzRow, nzRow, c0Dim>;
    using OutStridDim5 = pto::Stride<Cols / c0Dim * c0Dim * DstRows, DstRows * c0Dim, nzRow * c0Dim, c0Dim, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    SrcVecTile srcTile;
    DstVecTile dstTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(matTile, 0x0);

    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = byteCols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    constexpr uint32_t nz1TotalBytes = burstNum * (AlignedRow + 1) * c0Size * sizeof(T);
    constexpr uint16_t nz1BurstLen = static_cast<uint16_t>(nz1TotalBytes / BLOCK_BYTE_SIZE);
    constexpr uint32_t zeroElements = DstRows * byteCols;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* ubAddr = srcTile.data();
    __gm__ T* nz1GmAddr = src + zeroElements;

#if defined(__DAV_VEC__)
    // Load zero_region from GM to UB, then copy to L1 to initialize L1 with zeros
    copy_gm_to_ubuf((__ubuf__ void*)ubAddr, (__gm__ void*)src, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)ubAddr, 0, burstNum, burstLen, 0, 0);

    // Barrier: ensure L1 zero-fill (MTE3) finishes before MTE2 writes to same UB
    pipe_barrier(PIPE_ALL);

    // Load NZ data from GM to UB (MTE2) — overwrites zeros in UB
    copy_gm_to_ubuf((__ubuf__ void*)ubAddr, (__gm__ void*)nz1GmAddr, 0, 1, nz1BurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT<Mode>(matTile, srcTile);

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)ubAddr, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <TInsertMode Mode, typename T, uint32_t ValidRows, uint32_t DstRows, uint32_t Cols>
__global__ AICORE void launchTInsertNZTwoInputSplitKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZTwoInputSplit<Mode, T, ValidRows, DstRows, Cols>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertNZTwoInput(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZTwoInputSplitKernel<TInsertMode::SPLIT2, half, 8, 16, 128><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZTwoInputSplitKernel<TInsertMode::SPLIT2, float, 8, 16, 128><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZTwoInputSplitKernel<TInsertMode::SPLIT2, int8_t, 8, 16, 128><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZTwoInputSplitKernel<TInsertMode::SPLIT2, half, 129, 256, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertNZTwoInputSplitKernel<TInsertMode::SPLIT2, float, 129, 256, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZTwoInputSplitKernel<TInsertMode::SPLIT2, int8_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template <
    typename T, uint32_t TileRows, uint32_t Cols, uint32_t ValidRows1, uint32_t IndexRow1, uint32_t ValidRows2,
    uint32_t IndexRow2, uint32_t DstRows>
AICORE void runTInsertNZDoubleInput(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;
    constexpr uint32_t c0Dim = c0Size;
    constexpr uint32_t byteCols = Cols;
    constexpr uint32_t nz1Size = byteCols * TileRows * sizeof(T);
    constexpr uint32_t ubSrc2Offset = ((nz1Size + 511) / 512) * 512;

    using SrcVecTile1 = Tile<
        TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRows1, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>;
    using SrcVecTile2 = Tile<
        TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRows2, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Dim, DstRows / nzRow, nzRow, c0Dim>;
    using OutStridDim5 = pto::Stride<Cols / c0Dim * c0Dim * DstRows, DstRows * c0Dim, nzRow * c0Dim, c0Dim, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    SrcVecTile1 src1Tile;
    SrcVecTile2 src2Tile;
    DstVecTile dstTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    TASSIGN(src1Tile, 0x0);
    TASSIGN(src2Tile, ubSrc2Offset);
    TASSIGN(dstTile, 0x0);
    TASSIGN(matTile, 0x0);

    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = byteCols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    constexpr uint32_t nz1TotalBytes = burstNum * TileRows * c0Size * sizeof(T);
    constexpr uint16_t nz1BurstLen = static_cast<uint16_t>(nz1TotalBytes / BLOCK_BYTE_SIZE);
    constexpr uint32_t zeroElements = DstRows * byteCols;
    constexpr uint32_t nz1Elements = nz1TotalBytes / sizeof(T);

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* ubAddr1 = src1Tile.data();
    __ubuf__ T* ubAddr2 = src2Tile.data();
    __gm__ T* nz1Addr1 = src + zeroElements;
    __gm__ T* nz1Addr2 = nz1Addr1 + nz1Elements;

#if defined(__DAV_VEC__)
    // Load zero_region from GM to UB, then copy to L1 to initialize L1 with zeros
    copy_gm_to_ubuf((__ubuf__ void*)ubAddr1, (__gm__ void*)src, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)ubAddr1, 0, burstNum, burstLen, 0, 0);

    // Barrier: ensure L1 zero-fill (MTE3) finishes before MTE2 writes to same UB
    pipe_barrier(PIPE_ALL);

    // Load NZ data for both tiles from GM to UB (MTE2) — overwrites zeros in UB
    copy_gm_to_ubuf((__ubuf__ void*)ubAddr1, (__gm__ void*)nz1Addr1, 0, 1, nz1BurstLen, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void*)ubAddr2, (__gm__ void*)nz1Addr2, 0, 1, nz1BurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT(matTile, src1Tile, static_cast<uint16_t>(IndexRow1), static_cast<uint16_t>(0));
    TINSERT(matTile, src2Tile, static_cast<uint16_t>(IndexRow2), static_cast<uint16_t>(0));

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void*)ubAddr1, (__cbuf__ void*)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <
    typename T, uint32_t TileRows, uint32_t Cols, uint32_t ValidRows1, uint32_t IndexRow1, uint32_t ValidRows2,
    uint32_t IndexRow2, uint32_t DstRows>
__global__ AICORE void launchTInsertNZDoubleInputKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZDoubleInput<T, TileRows, Cols, ValidRows1, IndexRow1, ValidRows2, IndexRow2, DstRows>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertNZDoubleInput(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZDoubleInputKernel<half, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZDoubleInputKernel<float, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZDoubleInputKernel<int8_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZDoubleInputKernel<half, 129, 256, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertNZDoubleInputKernel<float, 129, 128, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZDoubleInputKernel<int8_t, 129, 256, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    }
}

template <
    typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t ValidRow, uint32_t ValidCol, uint32_t DstRows,
    uint32_t DstCols, uint32_t IdxRow, uint32_t IdxCol>
AICORE void runTInsertCompactNullTLoad(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using InitGmShape = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using InitGmStride =
        pto::Stride<DstCols / c0Size * DstRows * c0Size, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using InitGlobal = GlobalTensor<T, InitGmShape, InitGmStride, Layout::NZ>;

    using SrcGmShape = pto::Shape<1, SrcCols / c0Size, SrcRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcGmStride =
        pto::Stride<SrcCols / c0Size * SrcRows * c0Size, SrcRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobal = GlobalTensor<T, SrcGmShape, SrcGmStride, Layout::NZ>;

    using DstGmShape = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using DstGmStride =
        pto::Stride<DstCols / c0Size * DstRows * c0Size, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using DstGlobal = GlobalTensor<T, DstGmShape, DstGmStride, Layout::NZ>;

    using NzSrcTile = Tile<
        TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, ValidRow, ValidCol, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::Null>;
    using DstUbTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, DstCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    NzSrcTile nzTile;
    DstUbTile dstTile;
    MatTile matTile(IdxRow + ValidRow, IdxCol + ValidCol);

    TASSIGN(nzTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    InitGlobal initGlobal(src);
    SrcGlobal srcGlobal(src + DstRows * DstCols);
    DstGlobal dstGlobal(out);

    constexpr uint32_t initBurstNum = DstCols / c0Size;
    constexpr uint16_t initBurstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();
    __ubuf__ T* nzUbAddr = nzTile.data();

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

#if defined(__DAV_VEC__)
    copy_gm_to_ubuf((__ubuf__ void*)dstUbAddr, (__gm__ void*)initGlobal.data(), 0, initBurstNum, initBurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)dstUbAddr, 0, initBurstNum, initBurstLen, 0, 0);

    pipe_barrier(PIPE_ALL);

    TLOAD(nzTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    (void)nzUbAddr;
    TINSERT(matTile, nzTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf(
        (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, initBurstNum, initBurstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <
    typename T, uint32_t NzRows, uint32_t NzCols, uint32_t ValidRow, uint32_t ValidCol, uint32_t DstRows,
    uint32_t DstCols, uint32_t IdxRow, uint32_t IdxCol, CompactMode CMode>
AICORE void runTInsertCompactTMov(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using InitGmShape = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using InitGmStride =
        pto::Stride<DstCols / c0Size * DstRows * c0Size, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using InitGlobal = GlobalTensor<T, InitGmShape, InitGmStride, Layout::NZ>;

    using SrcNdGmShape = pto::Shape<1, 1, 1, ValidRow, ValidCol>;
    using SrcNdGmStride = pto::Stride<1, 1, 1, ValidCol, 1>;
    using SrcNdGlobal = GlobalTensor<T, SrcNdGmShape, SrcNdGmStride>;

    using DstGmShape = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using DstGmStride =
        pto::Stride<DstCols / c0Size * DstRows * c0Size, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using DstGlobal = GlobalTensor<T, DstGmShape, DstGmStride, Layout::NZ>;

    using NdSrcTile = Tile<TileType::Vec, T, ValidRow, ValidCol, BLayout::RowMajor, -1, -1>;
    using NzSrcTile = Tile<
        TileType::Vec, T, NzRows, NzCols, BLayout::ColMajor, ValidRow, ValidCol, SLayout::RowMajor, 512, PadValue::Null,
        CMode>;
    using DstUbTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, DstCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    NdSrcTile ndTile(ValidRow, ValidCol);
    NzSrcTile nzTile;
    DstUbTile dstTile;
    MatTile matTile(IdxRow + ValidRow, IdxCol + ValidCol);

    TASSIGN(ndTile, 0x0);
    TASSIGN(nzTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    InitGlobal initGlobal(src);
    SrcNdGlobal srcGlobal(src + DstRows * DstCols);
    DstGlobal dstGlobal(out);

    constexpr uint32_t initBurstNum = DstCols / c0Size;
    constexpr uint16_t initBurstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

#if defined(__DAV_VEC__)
    copy_gm_to_ubuf((__ubuf__ void*)dstUbAddr, (__gm__ void*)initGlobal.data(), 0, initBurstNum, initBurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void*)matAddr, (__ubuf__ void*)dstUbAddr, 0, initBurstNum, initBurstLen, 0, 0);

    pipe_barrier(PIPE_ALL);

    TLOAD(ndTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(nzTile, ndTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

    TINSERT(matTile, nzTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf(
        (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, initBurstNum, initBurstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <
    typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t ValidRow, uint32_t ValidCol, uint32_t DstRows,
    uint32_t DstCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void launchTInsertCompactNullTLoadKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertCompactNullTLoad<T, SrcRows, SrcCols, ValidRow, ValidCol, DstRows, DstCols, IdxRow, IdxCol>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <
    typename T, uint32_t NzRows, uint32_t NzCols, uint32_t ValidRow, uint32_t ValidCol, uint32_t DstRows,
    uint32_t DstCols, uint32_t IdxRow, uint32_t IdxCol, CompactMode CMode>
__global__ AICORE void launchTInsertCompactTMovKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertCompactTMov<T, NzRows, NzCols, ValidRow, ValidCol, DstRows, DstCols, IdxRow, IdxCol, CMode>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertCompactNullTLoad(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertCompactNullTLoadKernel<half, 128, 64, 64, 32, 128, 128, 0, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertCompactNullTLoadKernel<half, 128, 64, 64, 32, 128, 128, 0, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertCompactNullTLoadKernel<float, 64, 32, 32, 16, 64, 64, 0, 8><<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTInsertCompactNormalTMov(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertCompactTMovKernel<half, 128, 64, 64, 32, 128, 128, 0, 0, CompactMode::Normal>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertCompactTMovKernel<half, 128, 64, 64, 32, 128, 128, 0, 32, CompactMode::Normal>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertCompactTMovKernel<float, 128, 32, 32, 16, 128, 64, 0, 8, CompactMode::Normal>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTInsertCompactRowPlusOneTMov(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertCompactTMovKernel<half, 65, 64, 64, 32, 128, 128, 0, 0, CompactMode::RowPlusOne>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertCompactTMovKernel<half, 65, 64, 64, 32, 128, 128, 0, 32, CompactMode::RowPlusOne>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertCompactTMovKernel<float, 65, 32, 32, 16, 128, 64, 0, 8, CompactMode::RowPlusOne>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTInsertCompactNullTLoad<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNullTLoad<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNullTLoad<3>(uint64_t* out, uint64_t* src, void* stream);

template void launchTInsertCompactNormalTMov<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNormalTMov<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNormalTMov<3>(uint64_t* out, uint64_t* src, void* stream);

template void launchTInsertCompactRowPlusOneTMov<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactRowPlusOneTMov<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactRowPlusOneTMov<3>(uint64_t* out, uint64_t* src, void* stream);

template void launchTInsertNZDoubleInput<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<5>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<6>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<5>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<6>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<5>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<6>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<7>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<8>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZ<9>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZVecToVec<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZVecToVec<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZVecToVec<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZVecToVec<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZVecToVec<5>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZVecToVec<6>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZVecToVec<7>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZOverwrite<1>(uint64_t* out, uint64_t* src1, uint64_t* src2, void* stream);
template void launchTInsertNZTwoInsert<1>(uint64_t* out, uint64_t* src1, uint64_t* src2, void* stream);
template void launchTInsertNZTwoInsert<2>(uint64_t* out, uint64_t* src1, uint64_t* src2, void* stream);
template void launchTInsertNZUnaligned<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZUnaligned<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZUnaligned<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNDVecScalar<1>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecScalar<2>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecScalar<3>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecValidShape<1>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecValidShape<2>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecValidShape<3>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecValidShape<4>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecValidShape<5>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVecValidShape<6>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<1>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<2>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<3>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<4>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<5>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<6>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<7>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<8>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<9>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<10>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
template void launchTInsertNDVec<11>(uint8_t* out, uint8_t* srcIn, uint8_t* dstIn, void* stream);
