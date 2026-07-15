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
#include <pto/common/constants.hpp>

using namespace pto;

template <typename TileDataDst, typename TileDataSrc>
AICORE inline void tf_copy_cbuf_to_ubuf(
    typename TileDataDst::TileDType dst, typename TileDataSrc::TileDType src, int vec_core, int block_count,
    int block_len, int src_stride, int dst_stride)
{
    copy_cbuf_to_ubuf(
        (__ubuf__ void*)__cce_get_tile_ptr(dst), (__cbuf__ void*)__cce_get_tile_ptr(src), vec_core, block_count,
        block_len, src_stride, dst_stride);
}

template <typename DstTileData, typename SrcTileData>
AICORE inline void MovL1ToUbuf(DstTileData& dstTile, SrcTileData& srcTile)
{
    uint16_t blockCount = 1;
    uint16_t blockLen = DstTileData::Rows * DstTileData::Cols * sizeof(typename SrcTileData::DType) / BLOCK_BYTE_SIZE;
    tf_copy_cbuf_to_ubuf<DstTileData, SrcTileData>(
        dstTile.data(), srcTile.data(), 0, blockCount, blockLen, 0,
        0); // move to vector core0
}

template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_ND2NZ(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<
        T, pto::Shape<N1, N2, N3, M, K>, pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>,
        Layout::ND>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    TFILLPAD(aMatTile, aMatTile);
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    MovL1ToUbuf<TileUBData, TileMatAData>(srcTile, aMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_ND2ND(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<
        T, pto::Shape<N1, N2, N3, M, K>, pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>,
        Layout::ND>;
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::NoneBox>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    MovL1ToUbuf<TileUBData, TileMatAData>(srcTile, aMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <
    typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM, int baseK>
AICORE inline void runTLOAD_MIX_NZ2NZ(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<
        T, pto::Shape<N1, N2, N3, M, K>, pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>,
        Layout::NZ>; // [2,2,4,16,8]
    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);
    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, N3 * M, N1 * N2 * K, SLayout::RowMajor, 512>;

    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    TFILLPAD(aMatTile, aMatTile);
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    MovL1ToUbuf<TileUBData, TileMatAData>(srcTile, aMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

// NC1HWC0 or C1HWNC0
template <
    typename T, Layout layout, int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstC0, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void runTLOAD_MIX_5HD(__gm__ T* out, __gm__ T* src)
{
    constexpr int gStride[5] = {
        gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstC0 * sizeof(T);
    constexpr int validRow = dstShape0 * dstShape1 * dstShape2 * dstShape3;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstShape0 * dstShape1 * dstShape2 * dstShape3;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    using ShapeDim5 = pto::Shape<dstShape0, dstShape1, dstShape2, dstShape3, dstC0>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, layout>;

    using TileData = ConvTile<
        TileType::Mat, T, bufferSize, layout, pto::ConvTileShape<dstShape0, dstShape1, dstShape2, dstShape3, dstC0>>;
    TileData srcTile;
    TASSIGN(srcTile, 0x0);

    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>,
        Layout::ND>;
    GlobalDataOut dstGlobal(out);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    MovL1ToUbuf<OutTileData, TileData>(outTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, outTile);
    out = dstGlobal.data();
}

// [C1HW, N/16, 16, C0]
template <
    typename T, int dstShape0, int dstC1HW, int dstShape2, int dstShape3, int dstC0, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void runTLOAD_MIX_FractalZ4D(__gm__ T* out, __gm__ T* src)
{
    constexpr int gStride[5] = {
        gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstC1HW * dstShape2 * dstShape3 * dstC0 * sizeof(T);
    constexpr int validRow = dstC1HW * dstShape2 * dstShape3;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstC1HW * dstShape2 * dstShape3;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    using ShapeDim5 = pto::Shape<1, dstC1HW, dstShape2, dstShape3, dstC0>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, Layout::FRACTAL_Z>;

    using TileData = ConvTile<
        TileType::Mat, T, bufferSize, Layout::FRACTAL_Z, pto::ConvTileShape<dstC1HW, dstShape2, dstShape3, dstC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);
    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, Rows, Cols>, pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>,
        Layout::ND>;
    GlobalDataOut dstGlobal(out);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    MovL1ToUbuf<OutTileData, TileData>(outTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, outTile);
    out = dstGlobal.data();
}

template <
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
__global__ AICORE void TLOAD_MIX_KERNEL(__gm__ uint8_t* out, __gm__ uint8_t* src0, __gm__ uint8_t* src1)
{
    if constexpr (format == 0) { // ND2NZ
        runTLOAD_MIX_ND2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    } else if constexpr (format == 2) { // ND2ND
        runTLOAD_MIX_ND2ND<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    } else if constexpr (format == 4) { // NZ2NZ
        runTLOAD_MIX_NZ2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0), reinterpret_cast<__gm__ T*>(src1));
    } else if constexpr (format == 6) {
        runTLOAD_MIX_5HD<T, Layout::NC1HWC0, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0));
    } else if constexpr (format == 7) {
        runTLOAD_MIX_5HD<T, Layout::FRACTAL_Z, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0));
    } else if constexpr (format == 8) {
        runTLOAD_MIX_FractalZ4D<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(
            reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src0));
    }
}

template <
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void launchTLOADMIX(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    TLOAD_MIX_KERNEL<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

/********************format 0:ND2NZ 2:ND2ND 4 NZ2NZ*****************************/
// 2:ND2ND
template void launchTLOADMIX<int8_t, 2, 1, 2, 3, 33, 99, 1, 2, 3, 33, 99, 198, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 2, 1, 2, 3, 64, 128, 1, 3, 4, 128, 128, 384, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 2, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<float, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 0:ND2NZ
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<float, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 4.NZ2NZ
template void launchTLOADMIX<float, 4, 2, 2, 4, 16, 8, 2, 2, 4, 16, 8, 80, 48>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 4, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 4, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template void launchTLOADMIX<int64_t, 2, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint64_t, 2, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 6 NC1HWC0
template void launchTLOADMIX<int8_t, 6, 1, 3, 16, 128, 32, 3, 4, 1024, 1024, 32, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 6, 3, 2, 128, 8, 32, 3, 2, 128, 128, 32, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 6, 3, 2, 8, 128, 32, 3, 8, 8, 128, 32, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 6, 1, 6, 10, 100, 16, 1, 6, 100, 100, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 6, 10, 16, 16, 2, 16, 256, 16, 100, 16, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 6, 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<float, 6, 1, 1, 56, 112, 8, 2, 3, 224, 224, 8, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 7 FZ2FZ
template void launchTLOADMIX<int8_t, 7, 2, 3, 3, 64, 32, 3, 3, 3, 128, 32, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 7, 8, 5, 5, 32, 32, 8, 5, 5, 128, 32, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 7, 1, 7, 7, 20, 16, 3, 7, 7, 100, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 7, 64, 7, 7, 2, 16, 256, 7, 7, 16, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 7, 96, 3, 3, 8, 16, 256, 3, 3, 8, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<float, 7, 70, 7, 7, 2, 8, 256, 7, 7, 256, 8, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

// 8 FZ4D
template void launchTLOADMIX<uint16_t, 8, 1, 49, 7, 16, 16, 1, 980, 32, 16, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<uint16_t, 8, 1, 81, 3, 16, 16, 1, 90, 3, 16, 16, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 8, 1, 63, 3, 16, 32, 1, 63, 9, 16, 32, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<int8_t, 8, 1, 125, 3, 16, 32, 1, 250, 5, 16, 32, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTLOADMIX<float, 8, 1, 126, 3, 16, 8, 1, 4704, 7, 16, 8, 1, 1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
