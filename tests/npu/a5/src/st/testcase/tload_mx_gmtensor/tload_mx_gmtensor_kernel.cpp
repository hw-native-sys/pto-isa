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

template <typename TileData>
__tf__ PTO_INTERNAL void tf_create_cbuf_matrix(typename TileData::TileDType __out__ tile, int64_t repeat_bit, int n)
{
    create_cbuf_matrix((__cbuf__ uint16_t *)__cce_get_tile_ptr(tile), repeat_bit, n);
}

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void tf_copy_cbuf_to_ubuf(typename TileDataDst::TileDType __out__ dst,
                                              typename TileDataSrc::TileDType __in__ src, int vec_core, int block_count,
                                              int block_len, int src_stride, int dst_stride)
{
    copy_cbuf_to_ubuf((__ubuf__ void *)__cce_get_tile_ptr(dst), (__cbuf__ void *)__cce_get_tile_ptr(src), vec_core,
                      block_count, block_len, src_stride, dst_stride);
}

template <typename T, Layout layout, typename GlobalDataSrc0>
AICORE inline auto GetGlobalTensor(__gm__ T *addr, __gm__ T *addr2)
{
    GlobalDataSrc0 tmpGlobal(addr2);
    using DynStrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using DynShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    DynShapeDim5 dynShape(tmpGlobal.GetShape(0), tmpGlobal.GetShape(1), tmpGlobal.GetShape(2), tmpGlobal.GetShape(3),
                          tmpGlobal.GetShape(4));
    DynStrideDim5 dynStride(tmpGlobal.GetStride(0), tmpGlobal.GetStride(1), tmpGlobal.GetStride(2),
                            tmpGlobal.GetStride(3), tmpGlobal.GetStride(4));
    using GlobalData = GlobalTensor<T, decltype(dynShape), decltype(dynStride), layout>;

    GlobalData srcGlobal(addr, dynShape, dynStride);
    return srcGlobal;
}

template <typename T, typename GlobalDataSrc0, typename GlobalDataOut, typename TileMatScaAData, typename TileUBData>
AICORE inline void RunLoadAndStoreDyn(__gm__ T *out, __gm__ T *src0, __gm__ T *src1, uint16_t totalSize)
{
    auto src0Global = GetGlobalTensor<T, GlobalDataSrc0::layout, GlobalDataSrc0>(src0, src1);
    GlobalDataOut dstGlobal(out);

    TileUBData srcTile(1, totalSize);
    TASSIGN(srcTile, 0x0);

    TileMatScaAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    uint8_t syncID = 0;
    // clear memory in L1 to help verify data in unaligned case
#if defined(__DAV_CUBE__)
    uint16_t blockLen = totalSize * sizeof(T) / 32;
    int64_t repeatBit = (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | 1;
    tf_create_cbuf_matrix<TileMatScaAData>(aMatTile.data(), repeatBit, 0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);

    // L1 -> UB : AIC
    uint16_t blockCount = 1;

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    // move to vector    core0
    tf_copy_cbuf_to_ubuf<TileUBData, TileMatScaAData>(srcTile.data(), aMatTile.data(), 0, blockCount, blockLen, 0, 0);
    // move to vector    core1
    tf_copy_cbuf_to_ubuf<TileUBData, TileMatScaAData>(srcTile.data(), aMatTile.data(), 1, blockCount, blockLen, 0, 0);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, typename GlobalDataSrc0, typename GlobalDataOut, typename TileMatScaAData, typename TileUBData>
AICORE inline void RunLoadAndStore(__gm__ T *out, __gm__ T *src0, __gm__ T *src1, uint16_t totalSize)
{
    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    TileUBData srcTile(1, totalSize);
    TASSIGN(srcTile, 0x0);

    TileMatScaAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    uint8_t syncID = 0;
    // clear memory in L1 to help verify data in unaligned case
#if defined(__DAV_CUBE__)
    uint16_t blockLen = totalSize * sizeof(T) / 32;
    int64_t repeatBit = (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | 1;
    tf_create_cbuf_matrix<TileMatScaAData>(aMatTile.data(), repeatBit, 0);

    /*************************************TLOAD****************************************/
    TLOAD<TileMatScaAData, GlobalDataSrc0>(aMatTile, src0Global);

    // L1 -> UB : AIC
    uint16_t blockCount = 1;

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    // move to vector    core0
    tf_copy_cbuf_to_ubuf<TileUBData, TileMatScaAData>(srcTile.data(), aMatTile.data(), 0, blockCount, blockLen, 0, 0);
    // move to vector    core1
    tf_copy_cbuf_to_ubuf<TileUBData, TileMatScaAData>(srcTile.data(), aMatTile.data(), 1, blockCount, blockLen, 0, 0);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MX_AND2ZZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using MxShape = TileShape2D<T, M, K, Layout::MX_A_ND>;
    using MxStride = BaseShape2D<T, WN4, WN5, Layout::MX_A_ND>;
    using GlobalDataSrc0 = GlobalTensor<T, MxShape, MxStride, Layout::MX_A_ND>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, 1, baseM * baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseM * baseK, 1>, Layout::ND>;

    using TileMatScaAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::RowMajor, 32>;
    using TileUBData = Tile<TileType::Vec, T, 1, baseM * baseK, BLayout::RowMajor, -1, -1>;

    RunLoadAndStore<T, GlobalDataSrc0, GlobalDataOut, TileMatScaAData, TileUBData>(out, src0, src1, baseM * baseK);
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MX_ADN2ZZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using MxShape = TileShape2D<T, M, K, Layout::MX_A_DN>;
    using MxStride = BaseShape2D<T, WN4, WN5, Layout::MX_A_DN>;
    using GlobalDataSrc0 = GlobalTensor<T, MxShape, MxStride, Layout::MX_A_DN>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, 1, baseM * baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, 1, 1>, Layout::ND>;

    using TileMatScaAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::RowMajor, 32>; // ZZ format
    using TileUBData = Tile<TileType::Vec, T, 1, baseM * baseK, BLayout::RowMajor, -1, -1>;

    RunLoadAndStore<T, GlobalDataSrc0, GlobalDataOut, TileMatScaAData, TileUBData>(out, src0, src1, baseM * baseK);
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MX_AZZ2ZZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using MxShape = TileShape2D<T, M, K, Layout::MX_A_ZZ>;
    using MxStride = BaseShape2D<T, WN4, WN5, Layout::MX_A_ZZ>;
    using GlobalDataSrc0 = GlobalTensor<T, MxShape, MxStride, Layout::MX_A_ZZ>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, 1, baseM * baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseM * baseK, 1>, Layout::ND>;

    using TileMatScaAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::RowMajor, 32>;
    using TileUBData = Tile<TileType::Vec, T, 1, baseM * baseK, BLayout::RowMajor, -1, -1>;

    RunLoadAndStore<T, GlobalDataSrc0, GlobalDataOut, TileMatScaAData, TileUBData>(out, src0, src1, baseM * baseK);
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MX_BND2NN(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using MxShape = TileShape2D<T, M, K, Layout::MX_B_ND>;
    using MxStride = BaseShape2D<T, WN4, WN5, Layout::MX_B_ND>;
    using GlobalDataSrc0 = GlobalTensor<T, MxShape, MxStride, Layout::MX_B_ND>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, 1, baseM * baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseM * baseK, 1>, Layout::ND>;

    using TileMatScaAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::ColMajor, 32>; // NN format
    using TileUBData = Tile<TileType::Vec, T, 1, baseM * baseK, BLayout::RowMajor, -1, -1>;

    RunLoadAndStore<T, GlobalDataSrc0, GlobalDataOut, TileMatScaAData, TileUBData>(out, src0, src1, baseM * baseK);
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MX_BDN2NN(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using MxShape = TileShape2D<T, M, K, Layout::MX_B_DN>;
    using MxStride = BaseShape2D<T, WN4, WN5, Layout::MX_B_DN>;
    using GlobalDataSrc0 = GlobalTensor<T, MxShape, MxStride, Layout::MX_B_DN>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, 1, baseM * baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, 1, 1>, Layout::ND>;

    using TileMatScaAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::ColMajor, 32>; // NN format
    using TileUBData = Tile<TileType::Vec, T, 1, baseM * baseK, BLayout::RowMajor, -1, -1>;

    RunLoadAndStoreDyn<T, GlobalDataSrc0, GlobalDataOut, TileMatScaAData, TileUBData>(out, src0, src1, baseM * baseK);
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
          int baseK>
AICORE inline void runTLOAD_MX_BNN2NN(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using MxShape = TileShape2D<T, M, K, Layout::MX_B_NN>;
    using MxStride = BaseShape2D<T, WN4, WN5, Layout::MX_B_NN>;
    using GlobalDataSrc0 = GlobalTensor<T, MxShape, MxStride, Layout::MX_B_NN>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, 1, baseM * baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, 1, 1>, Layout::ND>;

    using TileMatScaAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::ColMajor, 32>; // NN format
    using TileUBData = Tile<TileType::Vec, T, 1, baseM * baseK, BLayout::RowMajor, -1, -1>;

    RunLoadAndStoreDyn<T, GlobalDataSrc0, GlobalDataOut, TileMatScaAData, TileUBData>(out, src0, src1, baseM * baseK);
}

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
          int BASEM, int BASEK>
__global__ AICORE void TLOAD_MX_KERNEL(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (format == 0) { // AND2ZZ
        runTLOAD_MX_AND2ZZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 1) { // ADN2ZZ
        runTLOAD_MX_ADN2ZZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 2) { // BND2NN
        runTLOAD_MX_BND2NN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 3) { // BDN2NN
        runTLOAD_MX_BDN2NN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 4) { // AZZ2ZZ
        runTLOAD_MX_AZZ2ZZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 5) { // BNN2NN
        runTLOAD_MX_BNN2NN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    }
}

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
          int BASEM, int BASEK>
void launchTLOADMX(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    TLOAD_MX_KERNEL<float8_e8m0_t, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

// 0:AND2ZZ
template void launchTLOADMX<uint8_t, 0, 1, 1, 1, 16, 4, 1, 1, 1, 16, 4, 16, 4>(uint8_t *out, uint8_t *src0,
                                                                               uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 0, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 32, 158>(uint8_t *out, uint8_t *src0,
                                                                                   uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 0, 1, 1, 1, 32, 128, 1, 1, 1, 160, 128, 64, 1008>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 0, 1, 1, 1, 31, 118, 1, 1, 1, 34, 126, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
// 1:ADN2ZZ
template void launchTLOADMX<uint8_t, 1, 1, 1, 1, 1, 2, 1, 1, 1, 65534, 4, 16, 8>(uint8_t *out, uint8_t *src0,
                                                                                 uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 1, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 16, 64>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 1, 1, 1, 1, 32, 128, 1, 1, 1, 32, 128, 32, 128>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 1, 1, 1, 1, 27, 126, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 1, 1, 1, 1, 31, 118, 1, 1, 1, 34, 126, 64, 128>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
// 2:BND2NN
template void launchTLOADMX<uint8_t, 2, 1, 1, 1, 4, 64, 1, 1, 1, 4, 64, 4, 64>(uint8_t *out, uint8_t *src0,
                                                                               uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 2, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 16, 64>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 2, 1, 1, 1, 32, 127, 1, 1, 1, 32, 128, 32, 256>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 2, 1, 1, 1, 116, 34, 1, 1, 1, 130, 60, 128, 64>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
// 3:BDN2NN
template void launchTLOADMX<uint8_t, 3, 1, 1, 1, 4, 64, 1, 1, 1, 4, 64, 4, 64>(uint8_t *out, uint8_t *src0,
                                                                               uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 3, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 16, 64>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 3, 1, 1, 1, 2, 128, 1, 1, 1, 32, 128, 4, 1088>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 3, 1, 1, 1, 30, 127, 1, 1, 1, 128, 128, 128, 128>(uint8_t *out, uint8_t *src0,
                                                                                       uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 3, 1, 1, 1, 116, 34, 1, 1, 1, 130, 60, 128, 64>(uint8_t *out, uint8_t *src0,
                                                                                     uint8_t *src1, void *stream);
// 4:AZZ2ZZ
template void launchTLOADMX<uint8_t, 4, 1, 1, 1, 16, 4, 1, 1, 1, 16, 4, 16, 4>(uint8_t *out, uint8_t *src0,
                                                                               uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 4, 1, 1, 1, 80, 66, 1, 1, 1, 176, 80, 128, 96>(uint8_t *out, uint8_t *src0,
                                                                                    uint8_t *src1, void *stream);
// 5:BNN2NN
template void launchTLOADMX<uint8_t, 5, 1, 1, 1, 4, 64, 1, 1, 1, 4, 64, 4, 64>(uint8_t *out, uint8_t *src0,
                                                                               uint8_t *src1, void *stream);
template void launchTLOADMX<uint8_t, 5, 1, 1, 1, 58, 1024, 1, 1, 1, 58, 1040, 58, 1088>(uint8_t *out, uint8_t *src0,
                                                                                        uint8_t *src1, void *stream);
