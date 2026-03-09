/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace pto;

template <typename T, int row, int col, typename DstTileData, typename SrcTileData>
AICORE inline void TMOVMat2Vec(DstTileData &dst, SrcTileData &src)
{
    __ubuf__ typename DstTileData::DType *dstAddr = dst.data();
    __cbuf__ typename SrcTileData::DType *srcAddr = src.data();
    __ubuf__ typename DstTileData::DType *dstTileAddr = dstAddr;
    __cbuf__ typename SrcTileData::DType *srcTileAddr = srcAddr;

    uint16_t nBurst = 1;
    uint16_t lenBurst = row * col * sizeof(T) / 32;

    copy_cbuf_to_ubuf(dstTileAddr, srcTileAddr, 0, nBurst, lenBurst, 0, 0);
}

template <typename T, uint32_t Rows, uint32_t Cols, uint32_t ExtraRows, uint32_t ValidRows = Rows,
          uint32_t ValidCols = Cols, uint32_t IndexRows = 0, uint32_t IndexCols = 0>
AICORE void runTmovUb2l1(__gm__ T *out, __gm__ T *src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, ValidCols / c0Size, ValidRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * ValidRows, ValidRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, ExtraRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using DstTileData = Tile<TileType::Vec, T, ValidRows, ValidCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTileData = Tile<TileType::Mat, T, ValidRows, ValidCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcTileData srcTile(Rows, Cols);
    TmpTileData tmpTile(Rows, Cols);
    DstTileData dstTile(ValidRows, ValidCols);
    MatTileData matTile(ValidRows, ValidCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal); // gm->ub
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMOV(tmpTile, srcTile); // ub2Ub
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if constexpr (IndexRows != 0 || IndexCols != 0) {
        TEXTRACT(matTile, tmpTile, IndexRows, IndexCols);
    } else {
        TMOV(matTile, tmpTile); // ub2l1
    }

    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    TMOVMat2Vec<T, ValidRows, ValidCols>(dstTile, matTile);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile); // UB -> GM : AIV
}

template <typename T, uint32_t Rows, uint32_t Cols, uint32_t ExtraRows, uint32_t ValidRows = Rows,
          uint32_t ValidCols = Cols, uint32_t IndexRows = 0, uint32_t IndexCols = 0>
__global__ AICORE void launchTmovUb2l1(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTmovUb2l1<T, Rows, Cols, ExtraRows, ValidRows, ValidCols, IndexRows, IndexCols>(
        reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTmovUb2l1(uint64_t *out, uint64_t *src, void *stream)
{
    cout << "launchTmovUb2l1 start!" << endl;
    if constexpr (testKey == 1) {
        launchTmovUb2l1<half, 16, 32, 17><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTmovUb2l1<half, 64, 256, 65><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTmovUb2l1<int32_t, 48, 72, 49><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTmovUb2l1<int32_t, 96, 8, 97><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTmovUb2l1<int8_t, 32, 512, 33><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTmovUb2l1<int8_t, 64, 96, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTmovUb2l1<half, 64, 64, 65, 48, 48, 16, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTmovUb2l1<int32_t, 128, 128, 129, 64, 64, 64, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTmovUb2l1<int8_t, 256, 256, 256, 32, 32, 224, 224><<<1, nullptr, stream>>>(out, src);
    }
    cout << "launchTmovUb2l1 end!" << endl;
}

template void launchTmovUb2l1<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<3>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<4>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<5>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<6>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<7>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<8>(uint64_t *out, uint64_t *src, void *stream);
template void launchTmovUb2l1<9>(uint64_t *out, uint64_t *src, void *stream);
