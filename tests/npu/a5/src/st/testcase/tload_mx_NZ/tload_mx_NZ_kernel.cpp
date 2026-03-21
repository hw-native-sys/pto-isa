/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void tf_copy_cbuf_to_ubuf(typename TileDataDst::TileDType __out__ dst,
                                              typename TileDataSrc::TileDType __in__ src, int vec_core, int block_count,
                                              int block_len, int src_stride, int dst_stride, int i,
                                              uint32_t tileMatStride, uint32_t tileUbStride)
{
    copy_cbuf_to_ubuf((__ubuf__ void *)(__cce_get_tile_ptr(dst) + i * tileUbStride),
                      (__cbuf__ void *)(__cce_get_tile_ptr(src) + i * tileMatStride), vec_core, block_count, block_len,
                      src_stride, dst_stride);
}

template <typename T, int N0, int N1, int N2, int N3, int N4, int WN0, int WN1, int WN2, int WN3, int WN4, int baseRow,
          int baseCol, int validRow, int validCol, bool isScaleA>
AICORE inline void runTLOAD_SCALE(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // ZZ2ZZ: GM->L1,concatenate N0 in the row direction; L1->UB, Write continuously to UB
    // NN2NN: GM->L1,concatenate N0 in the col direction; L1->UB, Write continuously to UB
    constexpr int validSize = N0 * N1 * N2 * N3 * N4;
    using GlobalDataSrc0 = std::conditional_t<
        isScaleA,
        GlobalTensor<T, pto::Shape<N0, N1, N2, N3, N4>,
                     pto::Stride<WN1 * WN2 * WN3 * WN4, WN2 * WN3 * WN4, WN3 * WN4, WN4, 1>, Layout::MX_A_ZZ>,
        GlobalTensor<T, pto::Shape<N0, N1, N2, N3, N4>,
                     pto::Stride<WN1 * WN2 * WN3 * WN4, WN2 * WN3 * WN4, WN3 * WN4, WN4, 1>, Layout::MX_B_NN>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, 1, validSize>,
                                       pto::Stride<1 * validSize, 1 * validSize, validSize, validSize, 1>, Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);
    using TileMatAData = std::conditional_t<
        isScaleA,
        Tile<TileType::Mat, T, baseRow, baseCol, BLayout::RowMajor, validRow, validCol, SLayout::RowMajor, 32>,
        Tile<TileType::Mat, T, baseRow, baseCol, BLayout::ColMajor, validRow, validCol, SLayout::ColMajor, 32>>;

    using TileUBData = Tile<TileType::Vec, T, 1, validSize, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(1, validSize);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    /*************************************TLOAD****************************************/
    uint8_t syncID = 0;
    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    uint16_t validRowPreN0 = validRow / N0;
    uint16_t validColPreN0 = validCol / N0;
    uint16_t blockCount = isScaleA ? validRowPreN0 / 16 : validColPreN0 / 16;
    uint16_t l12ubBlockLen = isScaleA ? (validCol * 16 * sizeof(T) / 32) : (validRow * 16 * sizeof(T) / 32);
    uint16_t srcStride =
        isScaleA ? ((baseCol - validCol) * 16 * sizeof(T) / 32) : ((baseRow - validRow) * 16 * sizeof(T) / 32);

    uint32_t tileMatStride = isScaleA ? validRowPreN0 * baseCol : baseRow * validColPreN0;
    uint32_t tileUbStride = isScaleA ? validRowPreN0 * validCol : validRow * validColPreN0;
    for (int i = 0; i < N0; i++) {
        tf_copy_cbuf_to_ubuf<TileUBData, TileMatAData>(srcTile.data(), aMatTile.data(), 0, blockCount, l12ubBlockLen,
                                                       srcStride, 0, i, tileMatStride, tileUbStride); // move to vector0
        tf_copy_cbuf_to_ubuf<TileUBData, TileMatAData>(srcTile.data(), aMatTile.data(), 1, blockCount, l12ubBlockLen,
                                                       srcStride, 0, i, tileMatStride, tileUbStride); // move to vector1
    }

    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int format, int N0, int N1, int N2, int N3, int N4, int WN0, int WN1, int WN2, int WN3, int WN4,
          int BASEROW, int BASECOL>
__global__ AICORE void TLOAD_SCALE_KERNEL(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (format == 11) { // ZZ2ZZ
        runTLOAD_SCALE<float8_e8m0_t, N0, N1, N2, N3, N4, WN0, WN1, WN2, WN3, WN4, BASEROW, BASECOL, N0 * N1 * N3,
                       N2 * N4, true>(reinterpret_cast<__gm__ float8_e8m0_t *>(out),
                                      reinterpret_cast<__gm__ float8_e8m0_t *>(src0),
                                      reinterpret_cast<__gm__ float8_e8m0_t *>(src1));
    } else if constexpr (format == 12) { // NN2NN
        runTLOAD_SCALE<float8_e8m0_t, N0, N1, N2, N3, N4, WN0, WN1, WN2, WN3, WN4, BASEROW, BASECOL, N2 * N4,
                       N0 * N1 * N3, false>(reinterpret_cast<__gm__ float8_e8m0_t *>(out),
                                            reinterpret_cast<__gm__ float8_e8m0_t *>(src0),
                                            reinterpret_cast<__gm__ float8_e8m0_t *>(src1));
    }
}

template <typename T, int format, int N0, int N1, int N2, int N3, int N4, int WN0, int WN1, int WN2, int WN3, int WN4,
          int BASEROW, int BASECOL>
void launchTLOADSCALE(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    TLOAD_SCALE_KERNEL<T, format, N0, N1, N2, N3, N4, WN0, WN1, WN2, WN3, WN4, BASEROW, BASECOL>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

/********************format 11:ZZ2ZZ 12:NN2NN*****************************/
// shape[0] == 1, L1Size = [validRow, validCol]
template void launchTLOADSCALE<uint8_t, 11, 1, 1, 2, 16, 2, 1, 2, 3, 16, 2, 16, 4>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

template void launchTLOADSCALE<uint8_t, 12, 1, 2, 1, 16, 2, 1, 3, 2, 16, 2, 2, 32>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

// shape[0] == 1, L1Size > [validRow, validCol]
template void launchTLOADSCALE<uint8_t, 11, 1, 2, 2, 16, 2, 1, 2, 3, 16, 2, 48, 10>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);

template void launchTLOADSCALE<uint8_t, 12, 1, 2, 2, 16, 2, 1, 3, 2, 16, 2, 8, 64>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

template void launchTLOADSCALE<uint8_t, 11, 1, 5, 33, 16, 2, 1, 11, 40, 16, 2, 128, 96>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);

template void launchTLOADSCALE<uint8_t, 12, 1, 64, 29, 16, 2, 1, 65, 59, 16, 2, 58, 1088>(uint8_t *out, uint8_t *src0,
                                                                                          uint8_t *src1, void *stream);

// shape[0] > 1, L1Size = [validRow, validCol]
template void launchTLOADSCALE<uint8_t, 11, 3, 1, 2, 16, 2, 3, 2, 3, 16, 2, 48, 4>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);

template void launchTLOADSCALE<uint8_t, 12, 4, 2, 1, 16, 2, 4, 3, 2, 16, 2, 2, 128>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
// shape[0] > 1, L1Size > [validRow, validCol]
template void launchTLOADSCALE<uint8_t, 11, 4, 3, 3, 16, 2, 4, 10, 5, 16, 2, 192, 10>(uint8_t *out, uint8_t *src0,
                                                                                      uint8_t *src1, void *stream);

template void launchTLOADSCALE<uint8_t, 12, 7, 5, 3, 16, 2, 7, 7, 11, 16, 2, 12, 560>(uint8_t *out, uint8_t *src0,
                                                                                      uint8_t *src1, void *stream);