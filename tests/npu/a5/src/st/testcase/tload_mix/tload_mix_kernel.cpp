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

using namespace pto;

template <typename DstTileData, typename SrcTileData, uint8_t syncID>
AICORE inline void MovL1ToUbuf(DstTileData &dstTile, SrcTileData &srcTile)
{
    __cbuf__ typename SrcTileData::DType *srcMatAddr = srcTile.data();
    __ubuf__ typename SrcTileData::DType *dstUbAddr = dstTile.data();
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    uint16_t blockLen = DstTileData::Rows * DstTileData::Cols * sizeof(typename SrcTileData::DType) / BLOCK_BYTE_SIZE;
    if constexpr (std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value ||
                  std::is_same<typename SrcTileData::DType, float4_e2m1x2_t>::value) {
        blockLen = DstTileData::Rows * DstTileData::Cols / B4_C0_SIZE;
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0,
                      0); // move to vector
                          // core0
    copy_cbuf_to_ubuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0,
                      0); // move to vector
                          // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);      // veccore0 id0 correspond cubecore id is id0
    set_intra_block(PIPE_MTE1, syncID + 16); // veccore1 id0 correspond cubecore id is 16
#endif
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MIX_ND2NZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 =
        GlobalTensor<T, pto::Shape<N1, N2, N3, M, K>,
                     pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>, Layout::ND>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, baseM, baseK>,
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

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    TFILLPAD(aMatTile, aMatTile);
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, TileMatAData, syncID>(srcTile, aMatTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3,
                     syncID); // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}
template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MIX_DN2NZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 =
        GlobalTensor<T, pto::Shape<N1, N2, N3, M, K>,
                     pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, 1, WN4>, Layout::DN>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, baseM, baseK>,
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

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    TFILLPAD(aMatTile, aMatTile);
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, TileMatAData, syncID>(srcTile, aMatTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3,
                     syncID); // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MIX_ND2ND(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 =
        GlobalTensor<T, pto::Shape<N1, N2, N3, M, K>,
                     pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>, Layout::ND>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::NoneBox>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, TileMatAData, syncID>(srcTile, aMatTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3,
                     syncID); // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MIX_DN2DN(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 =
        GlobalTensor<T, pto::Shape<N1, N2, N3, M, K>,
                     pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, 1, WN4>, Layout::DN>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, baseK, baseM>,
                                       pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseM, 1>,
                                       Layout::ND>; // actually is DN

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::NoneBox>; // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseK, baseM, BLayout::RowMajor, -1, -1>; // DN：baseM need 32Byte aligned
    TileUBData srcTile(baseK, baseM);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, TileMatAData, syncID>(srcTile, aMatTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3,
                     syncID); // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MIX_NZ2NZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<T, pto::Shape<N1, N2, N3, M, K>,
                                        pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>,
                                        Layout::NZ>; // [2,2,4,16,8]
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, baseM, baseK>,
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

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    TFILLPAD(aMatTile, aMatTile);
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, TileMatAData, syncID>(srcTile, aMatTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3,
                     syncID); // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MIX_DN2ZN(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 =
        GlobalTensor<T, pto::Shape<N1, N2, N3, M, K>,
                     pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, 1, WN4>, Layout::DN>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>; // 大Z小N
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    // L1清0 方便测试非对齐场景
#if defined(__DAV_CUBE__)
    uint16_t blockLen = baseM * baseK * sizeof(T) / BLOCK_BYTE_SIZE;
    if constexpr (std::is_same<T, float4_e1m2x2_t>::value || std::is_same<T, float4_e2m1x2_t>::value) {
        blockLen = baseM * baseK / B4_C0_SIZE;
    }
    int64_t repeatBit = (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | 1;
    create_cbuf_matrix((__cbuf__ uint16_t *)srcMatAddr, repeatBit, 0);
#endif
    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, TileMatAData, syncID>(srcTile, aMatTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3,
                     syncID); // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

// NC1HWC0 or C1HWNC0
template <typename T, Layout layout, int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstC0,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void runTLOAD_MIX_5HD(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstC0 * sizeof(T);
    constexpr int validRow = dstShape0 * dstShape1 * dstShape2 * dstShape3;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstShape0 * dstShape1 * dstShape2 * dstShape3;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    using ShapeDim5 = pto::Shape<dstShape0, dstShape1, dstShape2, dstShape3, dstC0>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, layout>;

    using TileData = ConvTile<TileType::Mat, T, bufferSize, layout,
                              pto::ConvTileShape<dstShape0, dstShape1, dstShape2, dstShape3, dstC0>>;
    TileData srcTile;
    TASSIGN(srcTile, 0x0);

    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                       pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<OutTileData, TileData, syncID>(outTile, srcTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, outTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

// [C1HW, N/16, 16, C0]
template <typename T, int dstShape0, int dstC1HW, int dstShape2, int dstShape3, int dstC0, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void runTLOAD_MIX_FractalZ4D(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstC1HW * dstShape2 * dstShape3 * dstC0 * sizeof(T);
    constexpr int validRow = dstC1HW * dstShape2 * dstShape3;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstC1HW * dstShape2 * dstShape3;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    using ShapeDim5 = pto::Shape<1, dstC1HW, dstShape2, dstShape3, dstC0>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, Layout::FRACTAL_Z>;

    using TileData = ConvTile<TileType::Mat, T, bufferSize, Layout::FRACTAL_Z,
                              pto::ConvTileShape<dstC1HW, dstShape2, dstShape3, dstC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);
    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                       pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<OutTileData, TileData, syncID>(outTile, srcTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, outTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

// [N,H,W,C]->[N,C1,H,W,C0]
template <typename T, int dstShape0, int dstC1, int dstShape2, int dstShape3, int dstC0, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void runTLOAD_MIX_NHWC(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstShape0 * dstC1 * dstShape2 * dstShape3 * dstC0 * sizeof(T);
    constexpr int validRow = dstShape0 * dstC1 * dstShape2 * dstShape3;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstShape0 * dstC1 * dstShape2 * dstShape3;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    using ShapeDim5 = pto::Shape < 1, dstShape0, dstShape2, dstShape3,
          gWholeShape4<dstC1 * dstC0 ? gWholeShape4 : dstC1 * dstC0>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, Layout::NHWC>;

    using TileData = ConvTile<TileType::Mat, T, bufferSize, Layout::NC1HWC0,
                              pto::ConvTileShape<dstShape0, dstC1, dstShape2, dstShape3, dstC0>>;
    TileData srcTile;

    TASSIGN(srcTile, 0x0);
    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                       pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<OutTileData, TileData, syncID>(outTile, srcTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, outTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

// [N,C,H,W]->[N,C1,H,W,C0]
template <typename T, int dstShape0, int dstC1, int dstShape2, int dstShape3, int dstC0, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void runTLOAD_MIX_NCHW(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstShape0 * dstC1 * dstShape2 * dstShape3 * dstC0 * sizeof(T);
    constexpr int validRow = dstShape0 * dstC1 * dstShape2 * dstShape3;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstShape0 * dstC1 * dstShape2 * dstShape3;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    // [[1,N,C,H,W]]
    using ShapeDim5 = pto::Shape < 1, dstShape0,
          gWholeShape2<dstC1 * dstC0 ? gWholeShape2 : dstC1 * dstC0, dstShape2, dstShape3>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, Layout::NCHW>;

    using TileData = ConvTile<TileType::Mat, T, bufferSize, Layout::NC1HWC0,
                              pto::ConvTileShape<dstShape0, dstC1, dstShape2, dstShape3, dstC0>>;
    TileData srcTile;

    TASSIGN(srcTile, 0x0);
    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                       pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<OutTileData, TileData, syncID>(outTile, srcTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, outTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

// [N,C,H,W]->[C1HW,N/16,16,C0] [C1HW,N/16,16,C0,srcN,srcC,srcH,srcW,N,C,H,W]
template <typename T, int dstC1HW, int dstN16, int dstShape2, int dstShape3, int srcN, int srcC, int srcH, int srcW,
          int N, int C, int H, int W>
AICORE inline void runTLOAD_MIX_NCHW2FZ4D(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {N * C * H * W, C * H * W, H * W, W, 1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstC1HW * dstN16 * dstShape2 * dstShape3 * sizeof(T);
    constexpr int validRow = dstC1HW * dstN16 * dstShape2;
    constexpr int validCol = dstShape3;
    constexpr int Rows = dstC1HW * dstN16 * dstShape2;
    constexpr int Cols = (dstShape3 + blockSize - 1) / blockSize * blockSize;

    // [[1,N,C,H,W]]
    using ShapeDim5 = pto::Shape<1, srcN, srcC, srcH, srcW>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, Layout::NCHW>;

    using TileData = ConvTile<TileType::Mat, T, bufferSize, Layout::FRACTAL_Z,
                              pto::ConvTileShape<dstC1HW, dstN16, dstShape2, dstShape3>>;
    TileData srcTile;

    TASSIGN(srcTile, 0x0);
    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                       pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<OutTileData, TileData, syncID>(outTile, srcTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, outTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

// [N,C,D,H,W]->[N,D,C1,H,W,C0]
template <typename T, int dstShape0, int dstShape1, int dstC1, int dstShape3, int dstShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void runTLOAD_MIX_NCDHW(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int dstC0 = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstShape0 * dstShape1 * dstC1 * dstShape3 * dstShape4 * dstC0 * sizeof(T);
    constexpr int validRow = dstShape0 * dstShape1 * dstC1 * dstShape3 * dstShape4;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstShape0 * dstShape1 * dstC1 * dstShape3 * dstShape4;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    // [[N,C,D,H,W]]  -> [N, D, C1, H, W, C0]
    using ShapeDim5 = pto::Shape < dstShape0,
          gWholeShape1<dstC1 * dstC0 ? gWholeShape2 : dstC1 * dstC0, dstShape1, dstShape3, dstShape4>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, Layout::NCDHW>;

    using TileData = ConvTile<TileType::Mat, T, bufferSize, Layout::NDC1HWC0,
                              pto::ConvTileShape<dstShape0, dstShape1, dstC1, dstShape3, dstShape4, dstC0>>;
    TileData srcTile;

    TASSIGN(srcTile, 0x0);
    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                       pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<OutTileData, TileData, syncID>(outTile, srcTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, outTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

// [N,C,D,H,W]->[C1DHW,N/16,16,C0] [srcN,srcC,srcD,srcH,srcW,N,C,D,H,W,C1DHW,N/16]
template <typename T, int srcN, int srcC, int srcD, int srcH, int srcW, int N, int C, int D, int H, int W, int dstC1DHW,
          int dstN16>
AICORE inline void runTLOAD_MIX_NCDHW2FZ3D(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int dstC0 = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int gStride[5] = {C * D * H * W, D * H * W, H * W, W, 1};
    constexpr int blockSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int bufferSize = dstC1DHW * dstN16 * 16 * dstC0 * sizeof(T);
    constexpr int validRow = dstC1DHW * dstN16 * 16;
    constexpr int validCol = dstC0;
    constexpr int Rows = dstC1DHW * dstN16 * 16;
    constexpr int Cols = (dstC0 + blockSize - 1) / blockSize * blockSize;

    // [[N,C,D,H,W]]
    using ShapeDim5 = pto::Shape<srcN, srcC, srcD, srcH, srcW>;
    using StridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StridDim5, Layout::NCDHW>;

    using TileData =
        ConvTile<TileType::Mat, T, bufferSize, Layout::FRACTAL_Z_3D, pto::ConvTileShape<dstC1DHW, dstN16, 16, dstC0>>;
    TileData srcTile;

    TASSIGN(srcTile, 0x0);
    GlobalDataIn srcGlobal(src);
    TLOAD(srcTile, srcGlobal);

    using OutTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, validRow, validCol>;
    OutTileData outTile;
    TASSIGN(outTile, 0x0);

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                       pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<OutTileData, TileData, syncID>(outTile, srcTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, outTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
          int BASEM, int BASEK>
__global__ AICORE void TLOAD_MIX_KERNEL(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (format == 0) { // ND2NZ
        runTLOAD_MIX_ND2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 1) { // DN2NZ
        runTLOAD_MIX_DN2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 2) { // ND2ND
        runTLOAD_MIX_ND2ND<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 3) { // DN2DN
        runTLOAD_MIX_DN2DN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 4) { // NZ2NZ
        runTLOAD_MIX_NZ2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 5) { // DN2ZN
        runTLOAD_MIX_DN2ZN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 6) {
        runTLOAD_MIX_5HD<T, Layout::NC1HWC0, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0));
    } else if constexpr (format == 7) {
        runTLOAD_MIX_5HD<T, Layout::FRACTAL_Z, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0));
    } else if constexpr (format == 8) {
        runTLOAD_MIX_FractalZ4D<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(reinterpret_cast<__gm__ T *>(out),
                                                                                reinterpret_cast<__gm__ T *>(src0));
    } else if constexpr (format == 9) {
        runTLOAD_MIX_NHWC<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(reinterpret_cast<__gm__ T *>(out),
                                                                          reinterpret_cast<__gm__ T *>(src0));
    } else if constexpr (format == 10) {
        runTLOAD_MIX_NCHW<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(reinterpret_cast<__gm__ T *>(out),
                                                                          reinterpret_cast<__gm__ T *>(src0));
    } else if constexpr (format == 11) {
        // [C1HW,N/16,16,C0,srcN,srcC,srcH,srcW,N,C,H,W]
        runTLOAD_MIX_NCHW2FZ4D<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0));
    } else if constexpr (format == 12) {
        //[N, D, C1, H, W, C0, WN, WC, WD, WH, WW]
        runTLOAD_MIX_NCDHW<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5>(reinterpret_cast<__gm__ T *>(out),
                                                                           reinterpret_cast<__gm__ T *>(src0));
    } else if constexpr (format == 13) {
        // [srcN,srcC,srcD,srcH,srcW,N,C,D,H,W,C1DHW,N/16]
        runTLOAD_MIX_NCDHW2FZ3D<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0));
    }
}

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
          int BASEM, int BASEK>
void launchTLOADMIX(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    TLOAD_MIX_KERNEL<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

/********************format 0:ND2NZ 1:DN2NZ 2:ND2ND 3:DN2DN 4 NZ2NZ*****************************/
// 2:ND2ND
template void launchTLOADMIX<int8_t, 2, 1, 2, 3, 33, 99, 1, 2, 3, 33, 99, 198, 128>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 2, 1, 2, 3, 64, 128, 1, 3, 4, 128, 128, 384, 128>(uint8_t *out, uint8_t *src0,
                                                                                         uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 2, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);

// 0:ND2NZ
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                          uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>(uint8_t *out, uint8_t *src0,
                                                                                      uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);

// 1:DN2NZ
template void launchTLOADMIX<uint16_t, 1, 1, 1, 1, 64, 128, 1, 1, 1, 64, 128, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 1, 1, 1, 1, 51, 123, 1, 1, 1, 64, 128, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 1, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);

// 3:DN2DN
template void launchTLOADMIX<float, 3, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 3, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 3, 1, 2, 3, 64, 128, 1, 3, 4, 96, 128, 64, 768>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);

// 4.NZ2NZ
template void launchTLOADMIX<float, 4, 2, 2, 4, 16, 8, 2, 2, 4, 16, 8, 80, 48>(uint8_t *out, uint8_t *src0,
                                                                               uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 4, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 4, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

template void launchTLOADMIX<int64_t, 2, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120>(uint8_t *out, uint8_t *src0,
                                                                                      uint8_t *src1, void *stream);
template void launchTLOADMIX<uint64_t, 2, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                         uint8_t *src1, void *stream);

// 5 DN2ZN
template void launchTLOADMIX<uint16_t, 5, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>(uint8_t *out, uint8_t *src0,
                                                                                      uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 5, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);

// 6 NC1HWC0
template void launchTLOADMIX<int8_t, 6, 1, 3, 16, 128, 32, 3, 4, 1024, 1024, 32, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 6, 3, 2, 128, 8, 32, 3, 2, 128, 128, 32, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 6, 3, 2, 8, 128, 32, 3, 8, 8, 128, 32, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 6, 1, 6, 10, 100, 16, 1, 6, 100, 100, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 6, 10, 16, 16, 2, 16, 256, 16, 100, 16, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                         uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 6, 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 6, 1, 1, 56, 112, 8, 2, 3, 224, 224, 8, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);

// 7 FZ2FZ
template void launchTLOADMIX<int8_t, 7, 2, 3, 3, 64, 32, 3, 3, 3, 128, 32, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                 uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 7, 8, 5, 5, 32, 32, 8, 5, 5, 128, 32, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                 uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 7, 1, 7, 7, 20, 16, 3, 7, 7, 100, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 7, 64, 7, 7, 2, 16, 256, 7, 7, 16, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 7, 96, 3, 3, 8, 16, 256, 3, 3, 8, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 7, 70, 7, 7, 2, 8, 256, 7, 7, 256, 8, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                uint8_t *src1, void *stream);

// 8 FZ4D
template void launchTLOADMIX<uint16_t, 8, 1, 49, 7, 16, 16, 1, 980, 32, 16, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                      uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 8, 1, 81, 3, 16, 16, 1, 90, 3, 16, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 8, 1, 63, 3, 16, 32, 1, 63, 9, 16, 32, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 8, 1, 125, 3, 16, 32, 1, 250, 5, 16, 32, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 8, 1, 126, 3, 16, 8, 1, 4704, 7, 16, 8, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);

// 9 : NHWC2NC1HWC0
template void launchTLOADMIX<int8_t, 9, 1, 3, 11, 109, 32, 1, 3, 1023, 1000, 111, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 9, 3, 2, 121, 9, 32, 1, 3, 128, 127, 65, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 9, 1, 6, 10, 100, 16, 1, 1, 100, 100, 96, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 9, 10, 16, 16, 2, 16, 1, 256, 100, 16, 255, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                         uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 9, 1, 1, 56, 112, 8, 1, 2, 224, 224, 25, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 9, 2, 1, 56, 43, 8, 1, 3, 333, 188, 19, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);

// 10 : NCHW2NC1HWC0
template void launchTLOADMIX<int8_t, 10, 1, 3, 11, 109, 32, 1, 3, 111, 1023, 109, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 10, 3, 2, 121, 9, 32, 1, 3, 65, 128, 127, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 10, 1, 6, 10, 100, 16, 1, 1, 96, 100, 100, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 10, 10, 16, 16, 2, 16, 1, 256, 255, 100, 16, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                          uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 10, 1, 1, 56, 112, 8, 1, 2, 25, 224, 112, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 10, 2, 1, 56, 43, 8, 1, 3, 19, 333, 188, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

// 11 : NCHW2FZ4D
template void launchTLOADMIX<int8_t, 11, 75, 3, 16, 32, 48, 95, 5, 5, 50, 111, 5, 5>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 11, 98, 4, 16, 32, 64, 58, 7, 7, 121, 127, 7, 7>(uint8_t *out, uint8_t *src0,
                                                                                      uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 11, 63, 6, 16, 16, 96, 111, 3, 3, 220, 112, 3, 3>(uint8_t *out, uint8_t *src0,
                                                                                         uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 11, 75, 4, 16, 16, 64, 48, 5, 5, 100, 50, 5, 5>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 11, 50, 3, 16, 8, 48, 14, 5, 5, 224, 224, 5, 5>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 11, 27, 2, 16, 8, 32, 24, 3, 3, 333, 188, 3, 3>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
// 12 : NCDHW2NDC1HWC0
template void launchTLOADMIX<int8_t, 12, 1, 2, 3, 11, 109, 3, 111, 2, 1023, 109, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 12, 3, 3, 2, 15, 9, 3, 65, 4, 30, 50, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                 uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 12, 1, 4, 6, 10, 10, 1, 96, 6, 100, 10, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 12, 10, 2, 8, 16, 2, 256, 128, 2, 100, 7, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 12, 1, 5, 1, 25, 31, 2, 25, 7, 112, 31, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 12, 2, 2, 1, 43, 43, 3, 19, 2, 155, 180, 1, 1>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

// 13 : NCDHW2FZ3D
template void launchTLOADMIX<int8_t, 13, 48, 95, 2, 5, 5, 50, 111, 4, 5, 5, 150, 3>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 13, 32, 58, 2, 7, 7, 63, 127, 2, 7, 7, 196, 2>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 13, 48, 111, 2, 3, 3, 110, 112, 2, 3, 3, 126, 3>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 13, 32, 48, 3, 3, 3, 70, 50, 4, 3, 3, 81, 2>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 13, 48, 14, 5, 2, 2, 224, 224, 7, 2, 2, 40, 3>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 13, 32, 24, 2, 3, 3, 333, 188, 2, 3, 3, 54, 2>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

template <typename T, int format, int dtype, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4,
          int WN5, int BASEM, int BASEK>
__global__ AICORE void TLOAD_MIX_KERNEL_B4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (format == 0) { // ND2NZ
        if constexpr (dtype == 0) {
            runTLOAD_MIX_ND2NZ<float4_e1m2x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_ND2NZ<float4_e2m1x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 1) { // DN2NZ
        static_assert(format != 1, "DN2NZ not support if input dtype is fp4");
    } else if constexpr (format == 2) { // ND2ND
        if constexpr (dtype == 0) {
            runTLOAD_MIX_ND2ND<float4_e1m2x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_ND2ND<float4_e2m1x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 3) { // DN2DN
        if constexpr (dtype == 0) {
            runTLOAD_MIX_DN2DN<float4_e1m2x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_DN2DN<float4_e2m1x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 4) { // NZ2NZ
        if constexpr (dtype == 0) {
            runTLOAD_MIX_NZ2NZ<float4_e1m2x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_NZ2NZ<float4_e2m1x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 5) { // DN2ZN
        if constexpr (dtype == 0) {
            runTLOAD_MIX_DN2ZN<float4_e1m2x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_DN2ZN<float4_e2m1x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    }
}

template <typename T, int format, int dtype, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4,
          int WN5, int BASEM, int BASEK>
void launchTLOADMIX_B4(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    TLOAD_MIX_KERNEL_B4<T, format, dtype, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

template void launchTLOADMIX_B4<uint8_t, 2, 0, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>(uint8_t *out,
                                                                                              uint8_t *src0,
                                                                                              uint8_t *src1,
                                                                                              void *stream);
template void launchTLOADMIX_B4<uint8_t, 2, 1, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>(uint8_t *out,
                                                                                              uint8_t *src0,
                                                                                              uint8_t *src1,
                                                                                              void *stream);
template void launchTLOADMIX_B4<uint8_t, 0, 1, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                            uint8_t *src1,
                                                                                            void *stream);
template void launchTLOADMIX_B4<uint8_t, 4, 0, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>(uint8_t *out, uint8_t *src0,
                                                                                          uint8_t *src1, void *stream);
template void launchTLOADMIX_B4<uint8_t, 3, 0, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>(uint8_t *out, uint8_t *src0,
                                                                                            uint8_t *src1,
                                                                                            void *stream);
template void launchTLOADMIX_B4<uint8_t, 5, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                            uint8_t *src1,
                                                                                            void *stream);