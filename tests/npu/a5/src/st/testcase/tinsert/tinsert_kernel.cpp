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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <pto/npu/a5/TInsert.hpp>

using namespace pto;

template <int M, int N, typename OutType>
struct NZOutputFormat {
    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(OutType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGCols * sGRows, kGRows * sGCols * sGRows, sGCols * sGRows, sGCols, 1>;
    using GlobalData = GlobalTensor<OutType, ShapeDim5, StridDim5, Layout::NZ>;
};

template <typename TileMatAData, typename TileMatBData, typename LeftTile, typename RightTile, typename AccTile,
          typename DstMatTile, typename GlobalDataSrc0, typename GlobalDataSrc1>
AICORE inline void LoadMatmulInsert(TileMatAData &aMatTile, TileMatBData &bMatTile, LeftTile &aTile, RightTile &bTile,
                                    AccTile &cTile, DstMatTile &dstMatTile, GlobalDataSrc0 &src0Global,
                                    GlobalDataSrc1 &src1Global)
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

AICORE inline void ReadbackCbufToUbuf(__ubuf__ void *dstUb, __cbuf__ void *srcCbuf, uint16_t burstNum,
                                      uint16_t burstLen, uint16_t srcGap, uint8_t syncId, uint8_t eventIdNum)
{
    wait_intra_block(PIPE_MTE1, syncId);
    wait_intra_block(PIPE_MTE1, syncId + eventIdNum);
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(dstUb, srcCbuf, 0, burstNum, burstLen, srcGap, 0);
    copy_cbuf_to_ubuf(dstUb, srcCbuf, 1, burstNum, burstLen, srcGap, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncId);
    set_intra_block(PIPE_MTE1, syncId + eventIdNum);
}

template <typename GlobalData, typename VecTile>
AICORE inline void WaitAndStore(GlobalData &dstGlobal, VecTile &dstTile, uint8_t syncId)
{
    wait_intra_block(PIPE_MTE3, syncId);
    TSTORE(dstGlobal, dstTile);
}

template <typename AType, typename BType, int M, int K, int N>
__global__ AICORE void RunTInsertAcc2Mat(__gm__ float *out, __gm__ AType *src0, __gm__ BType *src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<BType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    using LeftTile = TileLeft<AType, M, K, M, K>;
    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;
    AccTile cTile;
    LeftTile aTile;
    RightTile bTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using DstMatTile = Tile<TileType::Mat, float, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using DstVecTile = Tile<TileType::Vec, float, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstMatTile dstMatTile;
    DstVecTile dstVecTile;
    TASSIGN(dstMatTile, 0x0);
    TASSIGN(dstVecTile, 0x0);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(float));
    constexpr uint16_t burstLen = M * c0Size * sizeof(float) / 32;
    constexpr uint16_t burstNum = N / c0Size;

    using OutFmt = NZOutputFormat<M, N, float>;
    typename OutFmt::GlobalData dstGlobal(out);

    uint8_t syncId = 0;

#if defined(__DAV_CUBE__)
    LoadMatmulInsert(aMatTile, bMatTile, aTile, bTile, cTile, dstMatTile, src0Global, src1Global);

    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

    __ubuf__ float *dstUbAddr = dstVecTile.data();
    __cbuf__ float *srcMatAddr = dstMatTile.data();
    copy_cbuf_to_ubuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)srcMatAddr, 0, burstNum, burstLen, 0, 0);
    copy_cbuf_to_ubuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)srcMatAddr, 1, burstNum, burstLen, 0, 0);

    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);

    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();
    if (idx == 0) {
        TSTORE(dstGlobal, dstVecTile);
    }
#endif
}

template <int32_t testKey>
void launchTInsertAcc2Mat(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertAcc2Mat<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (testKey == 2) {
        RunTInsertAcc2Mat<half, half, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    }
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZ(__gm__ T *out, __gm__ T *src)
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, srcGap, syncId,
                       eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZPlusOne(__gm__ T *out, __gm__ T *src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<TileType::Vec, T, Rows + 1, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::RowPlusOne>;
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), Rows, Cols, Rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZSplit(__gm__ T *out, __gm__ T *src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<TileType::Vec, T, Rows + 1, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::RowPlusOne>;
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), Rows, Cols, Rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT<Mode>(matTile, tmpTile);
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZ<T, Rows, Cols>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZPlusOneKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZPlusOne<T, Rows, Cols>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZSplitKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZSplit<Mode, T, Rows, Cols>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <typename T, uint32_t ValidRow, uint32_t TileRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZLargeTile(__gm__ T *out, __gm__ T *src)
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    {
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(0), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < dstRepeats; ++i) {
                vsts(vreg, dstUbAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void *)matAddr, (__ubuf__ void *)dstUbAddr, 0, burstNum, burstLen, 0, 0);

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t ValidRow, uint32_t TileRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZLargeTileKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZLargeTile<T, ValidRow, TileRows, DstRows, Cols, IdxRow>(reinterpret_cast<__gm__ T *>(out),
                                                                        reinterpret_cast<__gm__ T *>(src));
}

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZHif8(__gm__ T *out, __gm__ T *src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using NZShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using NZStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalData = GlobalTensor<T, NZShapeDim5, NZStridDim5, Layout::NZ>;
    using OutGlobalData = GlobalTensor<T, NZShapeDim5, NZStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile;
    DstVecTile dstTile(Rows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (Rows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, srcTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZSplitHif8(__gm__ T *out, __gm__ T *src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using NZShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using NZStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalData = GlobalTensor<T, NZShapeDim5, NZStridDim5, Layout::NZ>;
    using OutGlobalData = GlobalTensor<T, NZShapeDim5, NZStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile;
    DstVecTile dstTile(Rows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (Rows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT<Mode>(matTile, srcTile);
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZHif8Kernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZHif8<T, Rows, Cols>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZSplitHif8Kernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZSplitHif8<Mode, T, Rows, Cols>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertNZ(uint64_t *out, uint64_t *src, void *stream)
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
    } else if constexpr (testKey == 10) {
        launchTInsertNZHif8Kernel<hifloat8_t, 16, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 11) {
        launchTInsertNZSplitHif8Kernel<pto::TInsertMode::SPLIT2, hifloat8_t, 16, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 12) {
        launchTInsertNZSplitHif8Kernel<pto::TInsertMode::SPLIT4, hifloat8_t, 16, 128><<<1, nullptr, stream>>>(out, src);
    }
}

template <uint32_t Rows, uint32_t Cols>
AICORE void runTInsertND(__gm__ int8_t *out, __gm__ int8_t *src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<int8_t, SrcShapeDim5, SrcStridDim5>;
    using OutGlobalData = GlobalTensor<int8_t, SrcShapeDim5, SrcStridDim5>;

    using SrcTileData = Tile<TileType::Vec, int8_t, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, int8_t, Rows, Cols, BLayout::RowMajor, -1, -1>;

    using InsertSrcTile =
        Tile<TileType::Vec, float8_e8m0_t, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::RowMajor, 32>;
    using InsertDstTile =
        Tile<TileType::Mat, float8_e8m0_t, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::RowMajor, 32>;

    SrcTileData srcTile(Rows, Cols);
    DstTileData dstTile(Rows, Cols);
    InsertSrcTile insertSrc;
    InsertDstTile insertDst;

    TASSIGN(srcTile, 0x0);
    TASSIGN(insertSrc, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(insertDst, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;
    uint16_t blockLen = Rows * Cols * sizeof(int8_t) / BLOCK_BYTE_SIZE;
    __cbuf__ float8_e8m0_t *srcMatAddr = insertDst.data();
    __ubuf__ int8_t *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT(insertDst, insertSrc, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNDKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertND<Rows, Cols>(reinterpret_cast<__gm__ int8_t *>(out), reinterpret_cast<__gm__ int8_t *>(src));
}

template <int32_t testKey>
void launchTInsertND(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNDKernel<64, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNDKernel<128, 64><<<1, nullptr, stream>>>(out, src);
    }
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
          uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVec(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstIn)
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
void launchTInsertNDVec(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVec<float, 8, 8, 16, 16, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstIn));
    } else if constexpr (testKey == 2) {
        RunTInsertNDVec<float, 8, 8, 16, 16, 4, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstIn));
    } else if constexpr (testKey == 3) {
        RunTInsertNDVec<half, 16, 16, 32, 32, 8, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    } else if constexpr (testKey == 4) {
        RunTInsertNDVec<int8_t, 32, 32, 64, 64, 0, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstIn));
    } else if constexpr (testKey == 5) {
        RunTInsertNDVec<half, 16, 16, 32, 48, 4, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    } else if constexpr (testKey == 6) {
        RunTInsertNDVec<float, 8, 8, 16, 24, 3, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstIn));
    } else if constexpr (testKey == 7) {
        RunTInsertNDVec<float, 8, 8, 16, 24, 0, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstIn));
    } else if constexpr (testKey == 8) {
        RunTInsertNDVec<half, 8, 16, 16, 48, 2, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    } else if constexpr (testKey == 9) {
        RunTInsertNDVec<int8_t, 32, 32, 64, 64, 0, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstIn));
    } else if constexpr (testKey == 10) {
        RunTInsertNDVec<half, 4, 128, 8, 144, 0, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    } else if constexpr (testKey == 11) {
        RunTInsertNDVec<half, 4, 144, 8, 160, 0, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    }
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t SrcValidCols, uint32_t DstRows, uint32_t DstCols,
          uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVecValid(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstIn)
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
void launchTInsertNDVecValidShape(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVecValid<float, 4, 8, 5, 16, 16, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstIn));
    } else if constexpr (testKey == 2) {
        RunTInsertNDVecValid<half, 8, 16, 10, 16, 32, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    } else if constexpr (testKey == 3) {
        RunTInsertNDVecValid<int8_t, 16, 32, 20, 32, 64, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstIn));
    } else if constexpr (testKey == 4) {
        RunTInsertNDVecValid<float, 4, 8, 5, 16, 16, 2, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstIn));
    } else if constexpr (testKey == 5) {
        RunTInsertNDVecValid<half, 8, 16, 10, 16, 32, 4, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    } else if constexpr (testKey == 6) {
        RunTInsertNDVecValid<int8_t, 16, 32, 20, 32, 64, 8, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstIn));
    }
}

template <typename T, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVecScalar(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstIn)
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
void launchTInsertNDVecScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVecScalar<float, 16, 16, 5, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstIn));
    } else if constexpr (testKey == 2) {
        RunTInsertNDVecScalar<half, 32, 32, 10, 15><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstIn));
    } else if constexpr (testKey == 3) {
        RunTInsertNDVecScalar<int8_t, 64, 64, 20, 30><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstIn));
    }
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZUnaligned(__gm__ T *out, __gm__ T *src)
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
    using TmpVecTile = Tile<TileType::Vec, T, AlignedRow + 1, Cols, BLayout::ColMajor, SrcRows, Cols, SLayout::RowMajor,
                            512, PadValue::Null, CompactMode::RowPlusOne>;
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();
    __ubuf__ T *tmpAddr = tmpTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    {
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
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
            vdup(vreg, static_cast<T>(0), preg, MODE_ZEROING);
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
    copy_ubuf_to_cbuf((__cbuf__ void *)matAddr, (__ubuf__ void *)dstUbAddr, 0, burstNum, burstLen, 0, 0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), SrcRows, Cols, SrcRows);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t SrcRows1, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow2>
AICORE void runTInsertNZTwoInsert(__gm__ T *out, __gm__ T *src1, __gm__ T *src2)
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
    using TmpVecTile1 = Tile<TileType::Vec, T, AlignedRow1 + 1, Cols, BLayout::ColMajor, SrcRows1, Cols,
                             SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
    using Src2VecTile = Tile<TileType::Vec, T, SrcRows2, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile2 = Tile<TileType::Vec, T, AlignedRow2 + 1, Cols, BLayout::ColMajor, SrcRows2, Cols,
                             SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();
    __ubuf__ T *tmpAddr = tmpTile1.data();

#if defined(__DAV_VEC__)
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    {
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
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
            vdup(vreg, static_cast<T>(0), preg, MODE_ZEROING);
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
    copy_ubuf_to_cbuf((__cbuf__ void *)matAddr, (__ubuf__ void *)dstUbAddr, 0, burstNum, burstLen, 0, 0);

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
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr uint32_t tmpElements2 = (AlignedRow2 + 1) * Cols;
        constexpr uint16_t tmpRepeats2 =
            static_cast<uint16_t>((tmpElements2 + elementsPerRepeat - 1) / elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(0), preg, MODE_ZEROING);
            for (uint16_t i = 0; i < tmpRepeats2; ++i) {
                vsts(vreg, tmpAddr, static_cast<uint32_t>(i) * elementsPerRepeat, NORM_B32, preg);
            }
        }
    }
    pto::TMovToVecNd2Nz<T, TmpVecTile2, Src2VecTile>(tmpTile2.data(), src2Tile.data(), SrcRows2, Cols, SrcRows2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT(matTile, tmpTile2, static_cast<uint16_t>(IdxRow2), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZUnalignedKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZUnaligned<T, SrcRows, DstRows, Cols, IdxRow>(reinterpret_cast<__gm__ T *>(out),
                                                             reinterpret_cast<__gm__ T *>(src));
}

template <typename T, uint32_t SrcRows1, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow2>
__global__ AICORE void launchTInsertNZTwoInsertKernel(__gm__ uint64_t *out, __gm__ uint64_t *src1,
                                                      __gm__ uint64_t *src2)
{
    runTInsertNZTwoInsert<T, SrcRows1, SrcRows2, DstRows, Cols, IdxRow2>(
        reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src1), reinterpret_cast<__gm__ T *>(src2));
}

template <int32_t testKey>
void launchTInsertNZUnaligned(uint64_t *out, uint64_t *src, void *stream)
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
void launchTInsertNZTwoInsert(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZTwoInsertKernel<float, 15, 10, 32, 32, 15><<<1, nullptr, stream>>>(out, src1, src2);
    } else if constexpr (testKey == 2) {
        launchTInsertNZTwoInsertKernel<float, 8, 8, 16, 256, 8><<<1, nullptr, stream>>>(out, src1, src2);
    }
}

template <typename T, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZOverwrite(__gm__ T *out, __gm__ T *src1, __gm__ T *src2)
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
    using TmpVecTile1 = Tile<TileType::Vec, T, DstRows + 1, Cols, BLayout::ColMajor, DstRows, Cols, SLayout::RowMajor,
                             512, PadValue::Null, CompactMode::RowPlusOne>;

    using Src2VecTile = Tile<TileType::Vec, T, SrcRows2, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile2 = Tile<TileType::Vec, T, AlignedRow2 + 1, Cols, BLayout::ColMajor, SrcRows2, Cols,
                             SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;

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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();
    __ubuf__ T *tmpAddr = tmpTile1.data();

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
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
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
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t SrcRows2, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZOverwriteKernel(__gm__ uint64_t *out, __gm__ uint64_t *src1,
                                                      __gm__ uint64_t *src2)
{
    runTInsertNZOverwrite<T, SrcRows2, DstRows, Cols, IdxRow>(
        reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src1), reinterpret_cast<__gm__ T *>(src2));
}

template <int32_t testKey>
void launchTInsertNZOverwrite(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZOverwriteKernel<float, 10, 32, 32, 4><<<1, nullptr, stream>>>(out, src1, src2);
    }
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
AICORE void runTInsertNZVecToVec(__gm__ T *out, __gm__ T *src)
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
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __ubuf__ T *dstAddr = dstNZTile.data();
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(0), preg, MODE_ZEROING);
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
AICORE void runTInsertNZPlusOneVecToVec(__gm__ T *out, __gm__ T *src)
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
    using SrcNZTile = Tile<TileType::Vec, T, AlignedSrcRow + 1, Cols, BLayout::ColMajor, SrcRows, Cols,
                           SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
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
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr uint32_t dstElements = DstRows * Cols;
        constexpr uint16_t dstRepeats =
            static_cast<uint16_t>((dstElements + elementsPerRepeat - 1) / elementsPerRepeat);
        __ubuf__ T *dstAddr = dstNZTile.data();
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t predCount = elementsPerRepeat;
            MaskReg preg = CreatePredicate<T>(predCount);
            vdup(vreg, static_cast<T>(0), preg, MODE_ZEROING);
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
__global__ AICORE void launchTInsertNZVecToVecKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZVecToVec<T, SrcRows, DstRows, Cols, IdxRow>(reinterpret_cast<__gm__ T *>(out),
                                                            reinterpret_cast<__gm__ T *>(src));
}

template <typename T, uint32_t SrcRows, uint32_t DstRows, uint32_t Cols, uint32_t IdxRow>
__global__ AICORE void launchTInsertNZPlusOneVecToVecKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZPlusOneVecToVec<T, SrcRows, DstRows, Cols, IdxRow>(reinterpret_cast<__gm__ T *>(out),
                                                                   reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertNZVecToVec(uint64_t *out, uint64_t *src, void *stream)
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
AICORE void runTInsertNZSplitCustom(__gm__ T *out, __gm__ T *src)
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
    using TmpVecTile = Tile<TileType::Vec, T, DstRows + 1, Cols, BLayout::ColMajor, ValidRow, Cols, SLayout::RowMajor,
                            512, PadValue::Null, CompactMode::RowPlusOne>;
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
    TASSIGN(dstTile, 0x0);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), ValidRow, Cols, ValidRow);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT<Mode>(matTile, tmpTile);
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <pto::TInsertMode Mode, typename T, uint32_t ValidRow, uint32_t DstRows, uint32_t Cols>
__global__ AICORE void launchTInsertNZSplitCustomKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZSplitCustom<Mode, T, ValidRow, DstRows, Cols>(reinterpret_cast<__gm__ T *>(out),
                                                              reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertNZSplitCustom(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT2, float, 8, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT4, float, 8, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT2, float, 128, 128, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZSplitCustomKernel<pto::TInsertMode::SPLIT4, float, 128, 128, 128>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template <pto::TInsertMode Mode, typename T, uint32_t ValidRows, uint32_t DstRows, uint32_t Cols>
AICORE void runTInsertNZTwoInputSplit(__gm__ T *out, __gm__ T *src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;
    constexpr uint32_t AlignedRow = ((ValidRows + nzRow - 1) / nzRow) * nzRow;
    constexpr bool isFp4 = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    constexpr uint32_t fp4Factor = isFp4 ? 2 : 1;
    constexpr uint32_t c0Dim = c0Size * fp4Factor;  // c0 in element units (64 for fp4, c0Size for others)
    constexpr uint32_t byteCols = Cols / fp4Factor; // byte-level columns

    using SrcVecTile = Tile<TileType::Vec, T, AlignedRow + 1, Cols, BLayout::ColMajor, ValidRows, Cols,
                            SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *ubAddr = srcTile.data();
    __gm__ T *nz1GmAddr = src + zeroElements;

#if defined(__DAV_VEC__)
    copy_gm_to_ubuf((__ubuf__ void *)ubAddr, (__gm__ void *)src, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void *)matAddr, (__ubuf__ void *)ubAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    copy_gm_to_ubuf((__ubuf__ void *)ubAddr, (__gm__ void *)nz1GmAddr, 0, 1, nz1BurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT<Mode>(matTile, srcTile);
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)ubAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <pto::TInsertMode Mode, typename T, uint32_t ValidRows, uint32_t DstRows, uint32_t Cols>
__global__ AICORE void launchTInsertNZTwoInputSplitKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZTwoInputSplit<Mode, T, ValidRows, DstRows, Cols>(reinterpret_cast<__gm__ T *>(out),
                                                                 reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertNZTwoInputGroup1(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, half, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, bfloat16_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, int8_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float8_e5m2_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float8_e4m3_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, hifloat8_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, half, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, bfloat16_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 10) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTInsertNZTwoInputGroup2(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 11) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, int8_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 12) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float8_e5m2_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 13) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float8_e4m3_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 14) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, hifloat8_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 15) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float4_e2m1x2_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 16) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float4_e1m2x2_t, 8, 16, 128>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 17) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float4_e2m1x2_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 18) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float4_e1m2x2_t, 129, 256, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 19) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float4_e2m1x2_t, 8, 16, 192>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 20) {
        launchTInsertNZTwoInputSplitKernel<pto::TInsertMode::SPLIT2, float4_e1m2x2_t, 8, 16, 192>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template <int32_t testKey>
void launchTInsertNZTwoInput(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey <= 10) {
        launchTInsertNZTwoInputGroup1<testKey>(out, src, stream);
    } else {
        launchTInsertNZTwoInputGroup2<testKey>(out, src, stream);
    }
}

template <typename T, uint32_t TileRows, uint32_t Cols, uint32_t ValidRows1, uint32_t IndexRow1, uint32_t ValidRows2,
          uint32_t IndexRow2, uint32_t DstRows>
AICORE void runTInsertNZDoubleInput(__gm__ T *out, __gm__ T *src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;
    constexpr bool isFp4 = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    constexpr uint32_t fp4Factor = isFp4 ? 2 : 1;
    constexpr uint32_t c0Dim = c0Size * fp4Factor;
    constexpr uint32_t byteCols = Cols / fp4Factor;
    constexpr uint32_t nz1Size = byteCols * TileRows * sizeof(T);
    constexpr uint32_t ubSrc2Offset = ((nz1Size + 511) / 512) * 512;

    using SrcVecTile1 = Tile<TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRows1, Cols, SLayout::RowMajor,
                             512, PadValue::Null, CompactMode::RowPlusOne>;
    using SrcVecTile2 = Tile<TileType::Vec, T, TileRows, Cols, BLayout::ColMajor, ValidRows2, Cols, SLayout::RowMajor,
                             512, PadValue::Null, CompactMode::RowPlusOne>;
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

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *ubAddr1 = src1Tile.data();
    __ubuf__ T *ubAddr2 = src2Tile.data();
    __gm__ T *nz1Addr1 = src + zeroElements;
    __gm__ T *nz1Addr2 = nz1Addr1 + nz1Elements;

#if defined(__DAV_VEC__)
    copy_gm_to_ubuf((__ubuf__ void *)ubAddr1, (__gm__ void *)src, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void *)matAddr, (__ubuf__ void *)ubAddr1, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    copy_gm_to_ubuf((__ubuf__ void *)ubAddr1, (__gm__ void *)nz1Addr1, 0, 1, nz1BurstLen, 0, 0);
    copy_gm_to_ubuf((__ubuf__ void *)ubAddr2, (__gm__ void *)nz1Addr2, 0, 1, nz1BurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT(matTile, src1Tile, static_cast<uint16_t>(IndexRow1), static_cast<uint16_t>(0));
    TINSERT(matTile, src2Tile, static_cast<uint16_t>(IndexRow2), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)ubAddr1, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t TileRows, uint32_t Cols, uint32_t ValidRows1, uint32_t IndexRow1, uint32_t ValidRows2,
          uint32_t IndexRow2, uint32_t DstRows>
__global__ AICORE void launchTInsertNZDoubleInputKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZDoubleInput<T, TileRows, Cols, ValidRows1, IndexRow1, ValidRows2, IndexRow2, DstRows>(
        reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertNZDoubleInput(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZDoubleInputKernel<float4_e2m1x2_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZDoubleInputKernel<float4_e1m2x2_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZDoubleInputKernel<hifloat8_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZDoubleInputKernel<float4_e2m1x2_t, 129, 256, 1, 128, 128, 0, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertNZDoubleInputKernel<float4_e1m2x2_t, 129, 256, 1, 128, 128, 0, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZDoubleInputKernel<hifloat8_t, 129, 256, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertNZDoubleInputKernel<half, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTInsertNZDoubleInputKernel<bfloat16_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTInsertNZDoubleInputKernel<float, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 10) {
        launchTInsertNZDoubleInputKernel<int8_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 11) {
        launchTInsertNZDoubleInputKernel<float8_e5m2_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 12) {
        launchTInsertNZDoubleInputKernel<float8_e4m3_t, 17, 128, 4, 0, 4, 4, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 13) {
        launchTInsertNZDoubleInputKernel<half, 129, 256, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 14) {
        launchTInsertNZDoubleInputKernel<bfloat16_t, 129, 256, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 15) {
        launchTInsertNZDoubleInputKernel<float, 129, 128, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 16) {
        launchTInsertNZDoubleInputKernel<int8_t, 129, 256, 1, 128, 128, 0, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 17) {
        launchTInsertNZDoubleInputKernel<float8_e5m2_t, 129, 256, 1, 128, 128, 0, 256>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 18) {
        launchTInsertNZDoubleInputKernel<float8_e4m3_t, 129, 256, 1, 128, 128, 0, 256>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template <typename T, uint32_t DstBurstNum, uint16_t DstBurstLen, uint16_t SrcBurstLen, typename MatTile,
          typename SrcVecTile, uint32_t IndexRow, uint32_t IndexCol>
AICORE void LoadAndInsertFp4(__cbuf__ T *matAddr, __ubuf__ T *dstUbAddr, __ubuf__ T *srcUbAddr, __gm__ T *src,
                             __gm__ T *srcGmAddr, MatTile &matTile, SrcVecTile &srcTile, uint8_t &syncId,
                             uint8_t eventIdNum)
{
#if defined(__DAV_VEC__)
    copy_gm_to_ubuf((__ubuf__ void *)dstUbAddr, (__gm__ void *)src, 0, DstBurstNum, DstBurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void *)matAddr, (__ubuf__ void *)dstUbAddr, 0, DstBurstNum, DstBurstLen, 0, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    copy_gm_to_ubuf((__ubuf__ void *)srcUbAddr, (__gm__ void *)srcGmAddr, 0, 1, SrcBurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TINSERT(matTile, srcTile, static_cast<uint16_t>(IndexRow), static_cast<uint16_t>(IndexCol));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, DstBurstNum, DstBurstLen, 0, syncId,
                       eventIdNum);
#endif
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t ValidRows, uint32_t IndexRow, uint32_t IndexCol,
          uint32_t DstRows, uint32_t DstCols>
AICORE void runTInsertNZFp4Offset(__gm__ T *out, __gm__ T *src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;
    constexpr bool isFp4 = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    constexpr uint32_t fp4Factor = isFp4 ? 2 : 1;
    constexpr uint32_t c0Dim = c0Size * fp4Factor;
    constexpr uint32_t dstByteCols = DstCols / fp4Factor;
    constexpr uint32_t srcByteCols = SrcCols / fp4Factor;

    using SrcVecTile =
        Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, ValidRows, SrcCols, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTileT = Tile<TileType::Mat, T, DstRows, DstCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    using OutShapeDim5 = pto::Shape<1, DstCols / c0Dim, DstRows / nzRow, nzRow, c0Dim>;
    using OutStridDim5 = pto::Stride<DstCols / c0Dim * c0Dim * DstRows, DstRows * c0Dim, nzRow * c0Dim, c0Dim, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    constexpr uint32_t dstUbBytes = DstRows * dstByteCols;
    constexpr uint32_t srcUbOffset = ((dstUbBytes + 511) / 512) * 512;
    constexpr uint32_t dstBurstNum = dstByteCols / c0Size;
    constexpr uint16_t dstBurstLen = (DstRows * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    constexpr uint16_t srcBurstLen = static_cast<uint16_t>(srcByteCols * SrcRows * sizeof(T) / BLOCK_BYTE_SIZE);
    constexpr uint32_t zeroElements = DstRows * dstByteCols;

    SrcVecTile srcTile;
    DstVecTile dstTile(DstRows, DstCols);
    MatTileT matTile(DstRows, DstCols);
    TASSIGN(srcTile, srcUbOffset);
    TASSIGN(dstTile, 0x0);
    TASSIGN(matTile, 0x0);

    OutGlobalData dstGlobal(out);
    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    LoadAndInsertFp4<T, dstBurstNum, dstBurstLen, srcBurstLen, MatTileT, SrcVecTile, IndexRow, IndexCol>(
        matTile.data(), dstTile.data(), srcTile.data(), src, src + zeroElements, matTile, srcTile, syncId, eventIdNum);

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t ValidRows, uint32_t IndexRow, uint32_t IndexCol,
          uint32_t DstRows, uint32_t DstCols>
__global__ AICORE void launchTInsertNZFp4OffsetKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZFp4Offset<T, SrcRows, SrcCols, ValidRows, IndexRow, IndexCol, DstRows, DstCols>(
        reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertNZFp4Offset(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 1) {
        // fp4e2m1: insert 16×64 at col=64 into 16×256 dst
        launchTInsertNZFp4OffsetKernel<float4_e2m1x2_t, 16, 64, 16, 0, 64, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        // fp4e1m2: insert 16×64 at col=64 into 16×256 dst
        launchTInsertNZFp4OffsetKernel<float4_e1m2x2_t, 16, 64, 16, 0, 64, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        // fp4e2m1: insert 16×64 at (row=4, col=128) into 16×256 dst
        launchTInsertNZFp4OffsetKernel<float4_e2m1x2_t, 16, 64, 8, 4, 128, 16, 256><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        // fp4e1m2: insert 16×64 at (row=4, col=128) into 16×256 dst
        launchTInsertNZFp4OffsetKernel<float4_e1m2x2_t, 16, 64, 8, 4, 128, 16, 256><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTInsertNZFp4Offset<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZFp4Offset<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZFp4Offset<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZFp4Offset<4>(uint64_t *out, uint64_t *src, void *stream);

template void launchTInsertNZDoubleInput<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<4>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<5>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<6>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<7>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<8>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<9>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<10>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<11>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<12>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<13>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<14>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<15>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<16>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<17>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZDoubleInput<18>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<4>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<5>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<6>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<7>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<8>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<9>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<10>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<11>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<12>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<13>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<14>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<15>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<16>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<17>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<18>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<19>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZTwoInput<20>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZSplitCustom<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZSplitCustom<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZSplitCustom<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZSplitCustom<4>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertAcc2Mat<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTInsertAcc2Mat<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTInsertNZ<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<4>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<5>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<6>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<7>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<8>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<9>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<10>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<11>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<12>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZVecToVec<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZVecToVec<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZVecToVec<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZVecToVec<4>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZVecToVec<5>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZVecToVec<6>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZVecToVec<7>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZOverwrite<1>(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream);
template void launchTInsertNZTwoInsert<1>(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream);
template void launchTInsertNZTwoInsert<2>(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream);
template void launchTInsertNZUnaligned<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZUnaligned<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZUnaligned<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNDVecScalar<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecScalar<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecScalar<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<4>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<5>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<6>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<4>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<5>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<6>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<7>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<8>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<9>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<10>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVec<11>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertND<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertND<2>(uint64_t *out, uint64_t *src, void *stream);