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

// NZ output format type helper
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

// Helper: TLOAD + TMOV + TMATMUL + TINSERT on CUBE side
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

// Helper: Read back L1/cbuf to UB and cross-core sync
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

// Helper: VEC wait and store
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

template void launchTInsertAcc2Mat<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTInsertAcc2Mat<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

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

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTInsertNZWithMode(__gm__ T *out, __gm__ T *src)
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

    constexpr uint16_t srcGap = 0;

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
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, srcGap, syncId,
                       eventIdNum);
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

template <pto::TInsertMode Mode, typename T, uint32_t Rows, uint32_t Cols>
__global__ AICORE void launchTInsertNZWithModeKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertNZWithMode<Mode, T, Rows, Cols>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertNZ(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 1) {
        launchTInsertNZKernel<float, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertNZWithModeKernel<pto::TInsertMode::NZ_PLUS_1, float, 16, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertNZKernel<float, 32, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertNZWithModeKernel<pto::TInsertMode::NZ_PLUS_1, int32_t, 32, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertNZWithModeKernel<pto::TInsertMode::SPLIT2_NZ_PLUS_1, float, 32, 32>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertNZWithModeKernel<pto::TInsertMode::SPLIT4_NZ_PLUS_1, float, 32, 32>
            <<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertNZKernel<float, 64, 64><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTInsertNZ<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<4>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<5>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<6>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZ<7>(uint64_t *out, uint64_t *src, void *stream);

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

template void launchTInsertND<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertND<2>(uint64_t *out, uint64_t *src, void *stream);

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

template void launchTInsertNDVecValidShape<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<4>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<5>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecValidShape<6>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

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

template void launchTInsertNDVecScalar<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecScalar<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);
template void launchTInsertNDVecScalar<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

// UB→L1 NZ unaligned test: SrcRows < 16 and/or non-zero offset
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
    // Load source ND data
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Zero-fill tmpTile and dstTile buffers
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

    // Zero L1 via copy_ubuf_to_cbuf
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_cbuf((__cbuf__ void *)matAddr, (__ubuf__ void *)dstUbAddr, 0, burstNum, burstLen, 0, 0);

    pto::TMovToVecNd2Nz<T, TmpVecTile, SrcVecTile>(tmpTile.data(), srcTile.data(), SrcRows, Cols, SrcRows);

    // Wait for both L1 zero-fill and NZ conversion, then insert
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT<pto::TInsertMode::NZ_PLUS_1>(matTile, tmpTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(0));
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    ReadbackCbufToUbuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, burstNum, burstLen, 0, syncId, eventIdNum);
#endif

#if defined(__DAV_VEC__)
    WaitAndStore(dstGlobal, dstTile, syncId);
#endif
}

// Two-insert unaligned scenario: insert src1 at (0,0) then src2 at (IdxRow,0)
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
    // Load src1 ND data
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Zero-fill tmpTile and dstTile, then zero L1
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

    // Convert src1 ND→NZ+1 and insert at (0,0)
    pto::TMovToVecNd2Nz<T, TmpVecTile1, Src1VecTile>(tmpTile1.data(), src1Tile.data(), SrcRows1, Cols, SrcRows1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT<pto::TInsertMode::NZ_PLUS_1>(matTile, tmpTile1, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    // Load src2 ND data
    TLOAD(src2Tile, src2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    // Zero-fill tmpTile again, convert src2, insert at (IdxRow2,0)
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

    TINSERT<pto::TInsertMode::NZ_PLUS_1>(matTile, tmpTile2, static_cast<uint16_t>(IdxRow2), static_cast<uint16_t>(0));
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
        // case_nz_8: 15 rows into 16-row dest (unaligned rows < 16)
        launchTInsertNZUnalignedKernel<float, 15, 16, 32, 0><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        // case_nz_9: 10 rows into 32-row dest at offset (16,0) (aligned boundary)
        launchTInsertNZUnalignedKernel<float, 10, 32, 32, 16><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTInsertNZUnaligned<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTInsertNZUnaligned<2>(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZTwoInsert(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream)
{
    if constexpr (testKey == 1) {
        // case_nz_10: 15+10 rows into 32-row dest at (0,0) + (15,0)
        launchTInsertNZTwoInsertKernel<float, 15, 10, 32, 32, 15><<<1, nullptr, stream>>>(out, src1, src2);
    }
}

template void launchTInsertNZTwoInsert<1>(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream);
