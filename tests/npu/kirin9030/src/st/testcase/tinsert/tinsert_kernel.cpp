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

using namespace pto;

template <
    typename TileMatAData, typename TileMatBData, typename LeftTile, typename RightTile, typename AccTile,
    typename DstMatTile, typename GlobalDataSrc0, typename GlobalDataSrc1>
AICORE inline void LoadMatmulInsert(
    TileMatAData& aMatTile, TileMatBData& bMatTile, LeftTile& aTile, RightTile& bTile, AccTile& cTile,
    DstMatTile& dstMatTile, GlobalDataSrc0& src0Global, GlobalDataSrc1& src1Global)
{
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TINSERT(dstMatTile, cTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
}

template <typename AType, typename CType, int M, int K, int N>
__global__ AICORE void RunTInsertAcc2Mat(__gm__ CType* out, __gm__ AType* src0, __gm__ AType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<AType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, AType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<(M * K + K * N) * sizeof(AType)>(dstMatTile);

    using LeftTile = TileLeft<AType, M, K, M, K>;
    using RightTile = TileRight<AType, K, N, K, N>;
    using AccTile = TileAcc<CType, M, N, M, N>;
    AccTile cTile;
    LeftTile aTile;
    RightTile bTile;
    TASSIGN<0x0>(aTile);
    TASSIGN<0x0>(bTile);
    TASSIGN<0x0>(cTile);

    using DstVecTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(CType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGCols * sGRows, kGRows * sGCols * sGRows, sGCols * sGRows, sGCols, 1>;
    using NZOutputGlobalData = GlobalTensor<CType, ShapeDim5, StridDim5, Layout::NZ>;
    NZOutputGlobalData dstGlobal(out);

    LoadMatmulInsert(aMatTile, bMatTile, aTile, bTile, cTile, dstMatTile, src0Global, src1Global);

    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(CType));
    constexpr uint16_t burstLen = M * c0Size * sizeof(CType) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ CType* dstUbAddr = dstVecTile.data();
    __cbuf__ CType* srcMatAddr = dstMatTile.data();
    copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <int32_t testKey>
void launchTInsertAcc2Mat(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (testKey == 1) {
        RunTInsertAcc2Mat<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 2) {
        RunTInsertAcc2Mat<half, half, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    }
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZ(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    SrcVecTile srcTile;
    DstVecTile tmpTile;
    DstVecTile dstTile;
    MatTile matTile;

    TASSIGN<0x0>(srcTile);
    TASSIGN<Rows * Cols * sizeof(T)>(tmpTile);
    TASSIGN<2 * Rows * Cols * sizeof(T)>(dstTile);
    TASSIGN<0x0>(matTile);

    constexpr uint32_t alignedRow = ((Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    constexpr uint16_t srcGap = 0;
    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT(matTile, tmpTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, 0, burstNum, burstLen, srcGap, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZPlusOne(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    using TmpVecTile = Tile<
        TileType::Vec, T, Rows + 1, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    SrcVecTile srcTile;
    TmpVecTile tmpTile;
    DstVecTile dstTile;
    MatTile matTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<Rows * Cols * sizeof(T)>(dstTile);
    TASSIGN<2 * Rows * Cols * sizeof(T)>(tmpTile);
    TASSIGN<0x0>(matTile);

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();
    constexpr uint32_t alignedRow = ((Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZSplit(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    using TmpVecTile = Tile<
        TileType::Vec, T, Rows + 1, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    SrcVecTile srcTile;
    DstVecTile dstTile;
    TmpVecTile tmpTile;
    MatTile matTile;

    TASSIGN<0x0>(srcTile);
    TASSIGN<Rows * Cols * sizeof(T)>(dstTile);
    TASSIGN<2 * Rows * Cols * sizeof(T)>(tmpTile);
    TASSIGN<0x0>(matTile);

    __cbuf__ T* matAddr = matTile.data();
    __ubuf__ T* dstUbAddr = dstTile.data();
    constexpr uint32_t alignedRow = ((Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT<Mode>(matTile, tmpTile);
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
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

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZSplitKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZSplit<Mode, T, Rows, Cols>(reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <typename T, uint32_t ValidRow, uint32_t TileRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZLargeTile(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using SrcNZShapeDim5 = pto::Shape<1, Cols / c0Size, TileRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcNZStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * TileRows, TileRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcNZGlobalData = GlobalTensor<T, SrcNZShapeDim5, SrcNZStridDim5, Layout::NZ>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    SrcNZGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcVecTile = Tile<TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRow, Cols, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;

    SrcVecTile srcTile;
    DstVecTile dstTile;
    MatTile matTile;

    TASSIGN<0x0>(srcTile);
    TASSIGN<TileRows * Cols * sizeof(T)>(dstTile);
    TASSIGN<0x0>(matTile);

    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* dstUbAddr = dstTile.data();
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    TEXPANDS(dstTile, 1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf(matAddr, dstUbAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TINSERT(matTile, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUbAddr, matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
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
        launchTInsertNZSplitKernel<pto::TInsertMode::SPLIT2, float, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZSplitKernel<pto::TInsertMode::SPLIT4, float, 32, 32><<<1, nullptr, stream>>>(out, src);
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
    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using DstStride = pto::Stride<DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;
    SrcGlobal srcGlobal(srcIn);
    DstGlobal outGlobal(out);
    DstGlobal dstInitGlobal(dstIn);

    using SrcVec = Tile<TileType::Vec, T, SrcRows, SrcCols>;
    using DstVec = Tile<TileType::Vec, T, DstRows, DstCols>;
    SrcVec srcTile;
    DstVec dstTile;

    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcVec::Numel * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);
    TLOAD(dstTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TINSERT(dstTile, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, dstTile);
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
    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using DstStride = pto::Stride<DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;
    SrcGlobal srcGlobal(srcIn);
    DstGlobal outGlobal(out);
    DstGlobal dstInitGlobal(dstIn);

    using SrcLoadVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using SrcInsertVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor, SrcRows, SrcValidCols>;
    using DstVec = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor>;

    SrcLoadVec srcLoad;
    SrcInsertVec srcInsert;
    DstVec dstTile;

    TASSIGN<0x0>(srcLoad);
    TASSIGN<0x0>(srcInsert);
    constexpr uint32_t srcSize = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN<dstAssignAddr>(dstTile);

    TLOAD(srcLoad, srcGlobal);
    TLOAD(dstTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TINSERT(dstTile, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, dstTile);
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
    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using SrcStride = pto::Stride<MinAlignedCols, MinAlignedCols, MinAlignedCols, MinAlignedCols, 1>;
    using DstStride = pto::Stride<DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;
    SrcGlobal srcGlobal(srcIn);
    DstGlobal outGlobal(out);
    DstGlobal dstInitGlobal(dstIn);

    using SrcLoadVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor>;
    using SrcInsertVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor, 1, 1>;
    using DstVec = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor>;
    SrcLoadVec srcLoad;
    SrcInsertVec srcInsert;
    DstVec dstTile;

    TASSIGN<0x0>(srcLoad);
    TASSIGN<0x0>(srcInsert);
    constexpr uint32_t srcSize = 1 * MinAlignedCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN<dstAssignAddr>(dstTile);

    TLOAD(srcLoad, srcGlobal);
    TLOAD(dstTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TINSERT(dstTile, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, dstTile);
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
    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcVecTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::RowMajor, SrcRows, Cols>;
    using TmpVecTile = Tile<
        TileType::Vec, T, AlignedRow + 1, Cols, BLayout::ColMajor, SrcRows, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    using ZeroVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::RowMajor, DstRows, Cols>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;

    SrcVecTile srcTile;
    TmpVecTile tmpTile;
    DstVecTile dstTile;
    ZeroVecTile zeroTile;
    MatTile matTile;
    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* dstUbAddr = dstTile.data();
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcRows * Cols * sizeof(T)>(tmpTile);
    TASSIGN<(SrcRows + AlignedRow + 1) * Cols * sizeof(T)>(dstTile);
    TASSIGN<(SrcRows + AlignedRow + 1) * Cols * sizeof(T)>(zeroTile);
    TASSIGN<0x0>(matTile);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    using TmpTileAlias = Tile<TileType::Vec, T, AlignedRow + 1, Cols>;
    using DstTileAlias = Tile<TileType::Vec, T, DstRows, Cols>;
    TEXPANDS((TmpTileAlias&)tmpTile, 1);
    TEXPANDS((DstTileAlias&)dstTile, 1);
    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    copy_ubuf_to_cbuf(matAddr, dstUbAddr, 0, burstNum, burstLen, 0, 0);
    pipe_barrier(PIPE_MTE3);
    TINSERT(matTile, tmpTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUbAddr, matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, uint32_t SrcRows1, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow2>
AICORE void runTInsertNZTwoInsert(__gm__ T* out, __gm__ T* src1, __gm__ T* src2)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t AlignedRow1 = ((SrcRows1 + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t AlignedRow2 = ((SrcRows2 + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t MaxAlignedRow = (AlignedRow1 > AlignedRow2) ? AlignedRow1 : AlignedRow2;

    using Src1ShapeDim5 = pto::Shape<1, 1, 1, SrcRows1, Cols>;
    using Src2ShapeDim5 = pto::Shape<1, 1, 1, SrcRows2, Cols>;
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using Src1StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using Src2StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using Src1GlobalData = GlobalTensor<T, Src1ShapeDim5, Src1StridDim5>;
    using Src2GlobalData = GlobalTensor<T, Src2ShapeDim5, Src2StridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    Src1GlobalData src1Global(src1);
    Src2GlobalData src2Global(src2);
    OutGlobalData dstGlobal(out);

    using Src1VecTile = Tile<TileType::Vec, T, SrcRows1, Cols, BLayout::RowMajor, SrcRows1, Cols>;
    using Src2VecTile = Tile<TileType::Vec, T, SrcRows2, Cols, BLayout::RowMajor, SrcRows2, Cols>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    using TmpVecTile1 = Tile<
        TileType::Vec, T, AlignedRow1 + 1, Cols, BLayout::ColMajor, SrcRows1, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using TmpVecTile2 = Tile<
        TileType::Vec, T, AlignedRow2 + 1, Cols, BLayout::ColMajor, SrcRows2, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;

    Src1VecTile src1Tile;
    Src2VecTile src2Tile;
    TmpVecTile1 tmpTile1;
    TmpVecTile2 tmpTile2;
    DstVecTile dstTile;
    MatTile matTile;

    TASSIGN<0x0>(src1Tile);
    TASSIGN<SrcRows1 * Cols * sizeof(T)>(src2Tile);
    TASSIGN<(SrcRows1 + SrcRows2) * Cols * sizeof(T)>(tmpTile1);
    TASSIGN<(SrcRows1 + SrcRows2) * Cols * sizeof(T)>(tmpTile2);
    TASSIGN<(SrcRows1 + SrcRows2 + MaxAlignedRow + 1) * Cols * sizeof(T)>(dstTile);
    TASSIGN<0x0>(matTile);

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* dstUbAddr = dstTile.data();
    __ubuf__ void* tmpAddr = tmpTile1.data();

    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    using TmpTileAlias = Tile<TileType::Vec, T, MaxAlignedRow + 1, Cols>;
    using DstTileAlias = Tile<TileType::Vec, T, DstRows, Cols>;
    TEXPANDS((TmpTileAlias&)tmpTile1, 1);
    TEXPANDS((DstTileAlias&)dstTile, 1);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf(matAddr, dstUbAddr, 0, burstNum, burstLen, 0, 0);

    TMOV(tmpTile1, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile1, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    TLOAD(src2Tile, src2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    using TmpTileAlias2 = Tile<TileType::Vec, T, AlignedRow2 + 1, Cols>;
    TEXPANDS((TmpTileAlias2&)tmpTile1, 1);

    TMOV(tmpTile2, src2Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile2, static_cast<uint16_t>(IdxRow2), static_cast<uint16_t>(0));

    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUbAddr, matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
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
    constexpr uint32_t MaxAlignedRow = (DstRows > AlignedRow2) ? DstRows : AlignedRow2;

    using Src1ShapeDim5 = pto::Shape<1, 1, 1, DstRows, Cols>;
    using Src2ShapeDim5 = pto::Shape<1, 1, 1, SrcRows2, Cols>;
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using Src1StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using Src2StridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using Src1GlobalData = GlobalTensor<T, Src1ShapeDim5, Src1StridDim5>;
    using Src2GlobalData = GlobalTensor<T, Src2ShapeDim5, Src2StridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    Src1GlobalData src1Global(src1);
    Src2GlobalData src2Global(src2);
    OutGlobalData dstGlobal(out);

    using Src1VecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::RowMajor>;
    using Src2VecTile = Tile<TileType::Vec, T, SrcRows2, Cols, BLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    using TmpVecTile1 = Tile<
        TileType::Vec, T, DstRows + 1, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using TmpVecTile2 = Tile<
        TileType::Vec, T, AlignedRow2 + 1, Cols, BLayout::ColMajor, SrcRows2, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;

    Src1VecTile src1Tile;
    Src2VecTile src2Tile;
    TmpVecTile1 tmpTile1;
    TmpVecTile2 tmpTile2;
    DstVecTile dstTile;
    MatTile matTile;

    TASSIGN<0x0>(src1Tile);
    TASSIGN<DstRows * Cols * sizeof(T)>(src2Tile);
    TASSIGN<(DstRows + SrcRows2) * Cols * sizeof(T)>(tmpTile1);
    TASSIGN<(DstRows + SrcRows2) * Cols * sizeof(T)>(tmpTile2);
    TASSIGN<(DstRows + SrcRows2 + MaxAlignedRow + 1) * Cols * sizeof(T)>(dstTile);
    TASSIGN<0x0>(matTile);

    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(tmpTile1, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile1, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    TLOAD(src2Tile, src2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    using TmpTileAlias = Tile<TileType::Vec, T, AlignedRow2 + 1, Cols>;
    TEXPANDS((TmpTileAlias&)tmpTile1, 1);

    TMOV(tmpTile2, src2Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile2, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));

    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* dstUbAddr = dstTile.data();
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    copy_cbuf_to_ubuf(dstUbAddr, matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
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
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcNDTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::RowMajor>;
    using SrcNZTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::ColMajor, SrcRows, Cols, SLayout::RowMajor>;
    using DstNZTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;

    SrcNDTile srcNDTile;
    SrcNZTile srcNZTile;
    DstNZTile dstNZTile;

    TASSIGN<0x0>(srcNDTile);
    TASSIGN<sizeof(T) * SrcRows * Cols>(srcNZTile);
    TASSIGN<2 * sizeof(T) * SrcRows * Cols>(dstNZTile);

    TLOAD(srcNDTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    using DstNZTileAlias = Tile<TileType::Vec, T, 1, DstRows * Cols>;
    TEXPANDS((DstNZTileAlias&)dstNZTile, 1);

    TMOV(srcNZTile, srcNDTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT(dstNZTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstNZTile);
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZPlusOneVecToVec(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t AlignedSrcRow = ((SrcRows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;

    using SrcShapeDim5 = pto::Shape<1, 1, 1, SrcRows, Cols>;
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcNDTile = Tile<TileType::Vec, T, SrcRows, Cols, BLayout::RowMajor>;
    using SrcNZTile = Tile<
        TileType::Vec, T, AlignedSrcRow + 1, Cols, BLayout::ColMajor, SrcRows, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using DstNZTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;

    SrcNDTile srcNDTile;
    SrcNZTile srcNZTile;
    DstNZTile dstNZTile;

    TASSIGN<0x0>(srcNDTile);
    TASSIGN<sizeof(T) * SrcRows * Cols>(srcNZTile);
    TASSIGN<sizeof(T) * (SrcRows + AlignedSrcRow + 1) * Cols>(dstNZTile);

    TLOAD(srcNDTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    using DstNZTileAlias = Tile<TileType::Vec, T, 1, DstRows * Cols>;
    TEXPANDS((DstNZTileAlias&)dstNZTile, 1);

    TMOV(srcNZTile, srcNDTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT(dstNZTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstNZTile);
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

template <pto::TInsertMode Mode, typename T, uint32_t ValidRow, uint32_t DstRows, uint32_t Cols>
AICORE void runTInsertNZSplitCustom(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using SrcShapeDim5 = pto::Shape<1, 1, 1, ValidRow, Cols>;
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    using SrcVecTile = Tile<TileType::Vec, T, ValidRow, Cols, BLayout::RowMajor, ValidRow, Cols>;
    using TmpVecTile = Tile<
        TileType::Vec, T, DstRows + 1, Cols, BLayout::ColMajor, ValidRow, Cols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    SrcVecTile srcTile;
    TmpVecTile tmpTile;
    DstVecTile dstTile;
    MatTile matTile;

    constexpr uint32_t srcSize = ValidRow * Cols * sizeof(T);
    constexpr uint32_t tmpOffset = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN<0x0>(srcTile);
    TASSIGN<tmpOffset>(tmpTile);
    TASSIGN<0x0>(matTile);

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    constexpr uint32_t dstUbOffset = ((tmpOffset + (DstRows + 1) * Cols * sizeof(T) + 0xFF) / 0x100) * 0x100;
    TASSIGN<dstUbOffset>(dstTile);

    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* dstUbAddr = dstTile.data();

    // Start TLOAD (MTE2) to overlap with V-pipe zero-fill
    TLOAD(srcTile, srcGlobal);

    // Wait for TLOAD to complete
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    using TmpVecTileAlias = Tile<TileType::Vec, T, 1, (DstRows + 1) * Cols>;
    TEXPANDS((TmpVecTileAlias&)tmpTile, 1);
    // Convert ND source to NZ format in tmpTile (writes only ValidRow rows, rest stays zero)
    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT<Mode>(matTile, tmpTile);
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUbAddr, matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <pto::TInsertMode Mode, typename T, uint32_t ValidRow, uint32_t DstRows, uint32_t Cols>
__global__ AICORE void launchTInsertNZSplitCustomKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZSplitCustom<Mode, T, ValidRow, DstRows, Cols>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertNZSplitCustom(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT2, float, 8, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT4, float, 8, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT2, float, 128, 128, 64>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT4, float, 128, 128, 64>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template <pto::TInsertMode Mode, typename T, uint32_t ValidRows, uint32_t DstRows, uint32_t Cols>
AICORE void runTInsertNZTwoInputSplit(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;
    constexpr uint32_t AlignedRow = ((ValidRows + nzRow - 1) / nzRow) * nzRow;

    using SrcVecTile = Tile<
        TileType::Vec, T, AlignedRow + 1, Cols, BLayout::ColMajor, ValidRows, Cols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / nzRow, nzRow, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, nzRow * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    SrcVecTile srcTile;
    DstVecTile dstTile(DstRows, Cols);
    MatTile matTile(DstRows, Cols);

    TASSIGN<0x0>(srcTile);
    TASSIGN<0x0>(dstTile);
    TASSIGN<0x0>(matTile);

    OutGlobalData dstGlobal(out);

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    constexpr uint32_t nz1TotalBytes = burstNum * (AlignedRow + 1) * c0Size * sizeof(T);
    constexpr uint16_t nz1BurstLen = static_cast<uint16_t>(nz1TotalBytes / BLOCK_BYTE_SIZE);
    constexpr uint32_t zeroElements = DstRows * Cols;

    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* ubAddr = srcTile.data();
    __gm__ void* nz1GmAddr = src + zeroElements;

    // Load zero_region from GM to UB, then copy to L1 to initialize L1 with zeros
    copy_gm_to_ubuf(ubAddr, src, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf(matAddr, ubAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // Load NZ data from GM to UB (MTE2) — overwrites zeros in UB
    copy_gm_to_ubuf(ubAddr, nz1GmAddr, 0, 1, nz1BurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TINSERT<Mode>(matTile, srcTile);
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(ubAddr, matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <pto::TInsertMode Mode, typename T, uint32_t ValidRows, uint32_t DstRows, uint32_t Cols>
__global__ AICORE void launchTInsertNZTwoInputSplitKernel(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTInsertNZTwoInputSplit<Mode, T, ValidRows, DstRows, Cols>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTInsertNZTwoInput(uint64_t* out, uint64_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, half, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, int8_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, half, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 10) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float, 129, 256, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 11) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, int8_t, 129, 256, 256>
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
    constexpr uint32_t nz1Size = Cols * TileRows * sizeof(T);
    constexpr uint32_t ubSrc2Offset = ((nz1Size + 511) / 512) * 512;

    using SrcVecTile1 = Tile<TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRows1, Cols, SLayout::RowMajor>;
    using SrcVecTile2 = Tile<TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRows2, Cols, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor>;

    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, DstRows / nzRow, nzRow, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * DstRows, DstRows * c0Size, nzRow * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;
    OutGlobalData dstGlobal(out);

    SrcVecTile1 src1Tile;
    SrcVecTile2 src2Tile;
    DstVecTile dstTile;
    MatTile matTile;

    TASSIGN<0x0>(src1Tile);
    TASSIGN<ubSrc2Offset>(src2Tile);
    TASSIGN<0x0>(dstTile);
    TASSIGN<0x0>(matTile);

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    constexpr uint32_t nz1TotalBytes = burstNum * TileRows * c0Size * sizeof(T);
    constexpr uint16_t nz1BurstLen = static_cast<uint16_t>(nz1TotalBytes / BLOCK_BYTE_SIZE);
    constexpr uint32_t zeroElements = DstRows * Cols;
    constexpr uint32_t nz1Elements = nz1TotalBytes / sizeof(T);

    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* ubAddr1 = src1Tile.data();
    __ubuf__ void* ubAddr2 = src2Tile.data();
    __gm__ void* nz1Addr1 = src + zeroElements;
    __gm__ void* nz1Addr2 = src + zeroElements + nz1Elements;

    // Load zero_region from GM to UB, then copy to L1 to initialize L1 with zeros
    copy_gm_to_ubuf(ubAddr1, src, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf(matAddr, ubAddr1, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // Load NZ data for both tiles from GM to UB (MTE2) — overwrites zeros in UB
    copy_gm_to_ubuf(ubAddr1, nz1Addr1, 0, 1, nz1BurstLen, 0, 0);
    copy_gm_to_ubuf(ubAddr2, nz1Addr2, 0, 1, nz1BurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TINSERT(matTile, src1Tile, static_cast<uint16_t>(IndexRow1), static_cast<uint16_t>(0));
    TINSERT(matTile, src2Tile, static_cast<uint16_t>(IndexRow2), static_cast<uint16_t>(0));
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(ubAddr1, matAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
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
    if constexpr (testKey == 7) {
        launchTInsertNZDoubleInputKernel<half, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTInsertNZDoubleInputKernel<float, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 10) {
        launchTInsertNZDoubleInputKernel<int8_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 13) {
        launchTInsertNZDoubleInputKernel<half, 129, 128, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 15) {
        launchTInsertNZDoubleInputKernel<float, 129, 64, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 16) {
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
    InitGlobal initGlobal(src);
    SrcGlobal srcGlobal(src + DstRows * DstCols);
    DstGlobal dstGlobal(out);

    using NzSrcTile = Tile<
        TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, ValidRow, ValidCol, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::Null>;
    using DstUbTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, DstCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    NzSrcTile nzTile;
    DstUbTile dstTile;
    MatTile matTile(IdxRow + ValidRow, IdxCol + ValidCol);

    TASSIGN<0x0>(nzTile);
    TASSIGN<sizeof(T) * SrcRows * SrcCols>(dstTile);
    TASSIGN<0x0>(matTile);

    constexpr uint32_t initBurstNum = DstCols / c0Size;
    constexpr uint16_t initBurstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* dstUbAddr = dstTile.data();

    TLOAD(nzTile, srcGlobal);
    copy_gm_to_ubuf(dstUbAddr, (__gm__ void*)src, 0, initBurstNum, initBurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf(matAddr, dstUbAddr, 0, initBurstNum, initBurstLen, 0, 0);

    pipe_barrier(PIPE_MTE3);

    TINSERT(matTile, nzTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUbAddr, matAddr, 0, initBurstNum, initBurstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <
    typename T, uint32_t NzRows, uint32_t NzCols, uint32_t ValidRow, uint32_t ValidCol, uint32_t DstRows,
    uint32_t DstCols, uint32_t IdxRow, uint32_t IdxCol, CompactMode CMode>
AICORE void runTInsertCompactTMov(__gm__ T* out, __gm__ T* src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using InitGmShape = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcNdGmShape = pto::Shape<1, 1, 1, ValidRow, ValidCol>;
    using DstGmShape = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using InitGmStride =
        pto::Stride<DstCols / c0Size * DstRows * c0Size, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcNdGmStride = pto::Stride<1, 1, 1, ValidCol, 1>;
    using DstGmStride =
        pto::Stride<DstCols / c0Size * DstRows * c0Size, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using InitGlobal = GlobalTensor<T, InitGmShape, InitGmStride, Layout::NZ>;
    using SrcNdGlobal = GlobalTensor<T, SrcNdGmShape, SrcNdGmStride>;
    using DstGlobal = GlobalTensor<T, DstGmShape, DstGmStride, Layout::NZ>;
    InitGlobal initGlobal(src);
    SrcNdGlobal srcGlobal(src + DstRows * DstCols);
    DstGlobal dstGlobal(out);

    using NdSrcTile = Tile<TileType::Vec, T, ValidRow, ValidCol, BLayout::RowMajor, ValidRow, ValidCol>;
    using NzSrcTile = Tile<
        TileType::Vec, T, NzRows, NzCols, BLayout::ColMajor, ValidRow, ValidCol, SLayout::RowMajor, 512, PadValue::Null,
        CMode>;
    using DstUbTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, DstRows, DstCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    NdSrcTile ndTile;
    NzSrcTile nzTile;
    DstUbTile dstTile;
    MatTile matTile(IdxRow + ValidRow, IdxCol + ValidCol);

    TASSIGN<0x0>(ndTile);
    TASSIGN<sizeof(T) * ValidRow * ValidCol>(nzTile);
    TASSIGN<sizeof(T) * (ValidRow * ValidCol + NzRows * NzCols)>(dstTile);
    TASSIGN<0x0>(matTile);

    constexpr uint32_t initBurstNum = DstCols / c0Size;
    constexpr uint16_t initBurstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ void* matAddr = matTile.data();
    __ubuf__ void* dstUbAddr = dstTile.data();

    TLOAD(ndTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMOV(nzTile, ndTile);

    copy_gm_to_ubuf(dstUbAddr, (__gm__ void*)src, 0, initBurstNum, initBurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf(matAddr, dstUbAddr, 0, initBurstNum, initBurstLen, 0, 0);

    pipe_barrier(PIPE_MTE3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TINSERT(matTile, nzTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUbAddr, matAddr, 0, initBurstNum, initBurstLen, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
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
    } else if constexpr (testKey == 4) {
        launchTInsertCompactNullTLoadKernel<half, 128, 128, 80, 48, 128, 128, 0, 16><<<1, nullptr, stream>>>(out, src);
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
    } else if constexpr (testKey == 4) {
        launchTInsertCompactTMovKernel<half, 128, 128, 64, 48, 128, 128, 0, 16, CompactMode::Normal>
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
    } else if constexpr (testKey == 4) {
        launchTInsertCompactTMovKernel<half, 65, 128, 64, 48, 128, 128, 0, 16, CompactMode::RowPlusOne>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTInsertCompactNullTLoad<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNullTLoad<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNullTLoad<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNullTLoad<4>(uint64_t* out, uint64_t* src, void* stream);

template void launchTInsertCompactNormalTMov<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNormalTMov<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNormalTMov<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactNormalTMov<4>(uint64_t* out, uint64_t* src, void* stream);

template void launchTInsertCompactRowPlusOneTMov<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactRowPlusOneTMov<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactRowPlusOneTMov<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertCompactRowPlusOneTMov<4>(uint64_t* out, uint64_t* src, void* stream);

template void launchTInsertNZDoubleInput<7>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<9>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<10>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<13>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<15>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZDoubleInput<16>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<8>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<10>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZTwoInput<11>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertNZSplitCustom<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTInsertAcc2Mat<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
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
