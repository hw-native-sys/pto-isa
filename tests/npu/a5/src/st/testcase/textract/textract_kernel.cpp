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
using fp8_e8m0_t = uint8_t;

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ PTO_INTERNAL void tf_mad_mx(typename TileDataDst::TileDType __out__ dst,
                                   typename TileDataSrc0::TileDType __in__ src0,
                                   typename TileDataSrc1::TileDType __in__ src1, int M, int K, int N, bool a, bool b,
                                   bool c, bool d)
{
    mad_mx(__cce_get_tile_ptr(dst), __cce_get_tile_ptr(src0), __cce_get_tile_ptr(src1), M, K, N, a, b, c, d);
}

template <typename TileDataDst, typename T>
__tf__ PTO_INTERNAL void DynGM2L1(typename TileDataDst::TileDType __out__ dst, __gm__ T *src, unsigned TShape0,
                                  unsigned TShape1)
{
    if (std::is_same<T, float4_e2m1x2_t>::value || std::is_same<T, float4_e1m2x2_t>::value) {
        uint32_t lenBurst = TShape0 * TShape1 * sizeof(T) / 2;
        copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t *)__cce_get_tile_ptr(dst), (__gm__ uint8_t *)src, 0, 1, lenBurst, 0,
                                 0, 0, 0, 0, 0);
    } else {
        uint32_t lenBurst = TShape0 * TShape1 * sizeof(T);
        copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t *)__cce_get_tile_ptr(dst), (__gm__ uint8_t *)src, 0, 1, lenBurst, 0,
                                 0, 0, 0, 0, 0);
    }
}

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void MOV_MX_TOA(typename TileDataDst::TileDType __out__ dst,
                                    typename TileDataSrc::TileDType __in__ src, unsigned dstM, unsigned dstK)
{
    uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(__cce_get_tile_ptr(dst))) / 16;
    constexpr int c0Size = 2;
    uint8_t mStep = dstK / c0Size;
    uint8_t kStep = dstM / 16;
    uint16_t srcStride = dstM / 16;
    uint16_t dstStride = dstM / 16;
    load_cbuf_to_ca_mx(mxDstAddr, static_cast<__cbuf__ void *>(__cce_get_tile_ptr(src)), 0, 0, mStep, kStep, srcStride,
                       dstStride);
}

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void MOV_MX_TOB(typename TileDataDst::TileDType __out__ dst,
                                    typename TileDataSrc::TileDType __in__ src, unsigned dstK, unsigned dstN)
{
    uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(__cce_get_tile_ptr(dst))) / 16;
    constexpr int c0Size = 2;
    uint8_t mStep = dstK / c0Size;
    uint8_t kStep = dstN / 16;
    uint16_t srcStride = dstN / 16;
    uint16_t dstStride = dstN / 16;
    load_cbuf_to_cb_mx(mxDstAddr, static_cast<__cbuf__ void *>(__cce_get_tile_ptr(src)), 0, 0, mStep, kStep, srcStride,
                       dstStride);
}

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexM, uint16_t indexK, uint16_t indexN,
          bool isAtranspose, bool isBtranspose>
AICORE inline void runTEXTRACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose, Tile<TileType::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose, Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, mValid, kValid>;
    using RightTile = TileRight<S, kValid, nValid, kValid, nValid>;
    using ResTile = TileAcc<T, mValid, nValid, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexM, uint16_t indexK, uint16_t indexN,
          bool isAtranspose, bool isBtranspose>
AICORE inline void runTEXTRACT_DYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int k, int n)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;

    using DynShape3Dim5 = pto::Shape<1, 1, 1, mValid, nValid>;
    using DynSTrid3Dim5 = pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShape3Dim5, DynSTrid3Dim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<TileType::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose,
                           Tile<TileType::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, -1, -1>;
    using RightTile = TileRight<S, kValid, nValid, kValid, -1>;
    using ResTile = TileAcc<T, mValid, nValid, -1, nValid>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    int validM = m - indexM;
    int validK = k - indexK;
    int validN = n - indexN;

    LeftTile aTile(validM, validK);
    RightTile bTile(validN);
    ResTile cTile(validM);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename X, int M, int K, int N, uint16_t indexM, uint16_t indexK,
          uint16_t indexN, bool isAtranspose, bool isBtranspose>
AICORE inline void runTEXTRACTMX(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ X *srcMx0, __gm__ X *srcMx1)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose, Tile<TileType::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose, Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>>;

    using TileMatAmxData =
        Tile<TileType::Mat, X, mValid, kValid, BLayout::RowMajor, mValid, kValid, SLayout::RowMajor, 512>;
    using TileMatBmxData =
        Tile<TileType::Mat, X, kValid, nValid, BLayout::ColMajor, kValid, nValid, SLayout::ColMajor, 512>;

    using LeftTile = TileLeft<U, mValid, kValid, mValid, kValid>;
    using RightTile = TileRight<S, kValid, nValid, kValid, nValid>;
    using ResTile = TileAcc<T, mValid, nValid, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatAmxData amxMatTile;
    TileMatBmxData bmxMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(amxMatTile, 0x20000);
    TASSIGN(bmxMatTile, 0x30000);

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    auto &a = aTile.data();
    auto &b = bTile.data();
    auto &c = cTile.data();
    auto &srcAmxAddr = amxMatTile.data();
    auto &srcBmxAddr = bmxMatTile.data();

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    DynGM2L1<TileMatAmxData, X>(srcAmxAddr, srcMx0, mValid, kValid / 32);
    DynGM2L1<TileMatBmxData, X>(srcBmxAddr, srcMx1, kValid / 32, nValid);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
    MOV_MX_TOA<LeftTile, TileMatAmxData>(a, srcAmxAddr, mValid, kValid / 32);
    MOV_MX_TOB<RightTile, TileMatBmxData>(b, srcBmxAddr, kValid / 32, nValid);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    tf_mad_mx<ResTile, LeftTile, RightTile>(c, a, b, mValid, kValid, nValid, false, false, false, true);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTEXTRACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, half, half, 32, 96, 64, 0, 0, 0, false, false>(reinterpret_cast<__gm__ float *>(out),
                                                                      reinterpret_cast<__gm__ half *>(src0),
                                                                      reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, float, float, 128, 48, 64, 0, 0, 0, false, false>(reinterpret_cast<__gm__ float *>(out),
                                                                         reinterpret_cast<__gm__ float *>(src0),
                                                                         reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<int32_t, int8_t, int8_t, 128, 128, 64, 0, 0, 0, false, false>(reinterpret_cast<__gm__ int32_t *>(out),
                                                                              reinterpret_cast<__gm__ int8_t *>(src0),
                                                                              reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, half, half, 64, 96, 64, 32, 16, 16, false, false>(reinterpret_cast<__gm__ float *>(out),
                                                                         reinterpret_cast<__gm__ half *>(src0),
                                                                         reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, float, float, 64, 128, 64, 32, 32, 16, false, false>(reinterpret_cast<__gm__ float *>(out),
                                                                            reinterpret_cast<__gm__ float *>(src0),
                                                                            reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<int32_t, int8_t, int8_t, 128, 128, 64, 32, 64, 32, false, false>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, half, half, 64, 128, 64, 0, 64, 0, true, true>(reinterpret_cast<__gm__ float *>(out),
                                                                      reinterpret_cast<__gm__ half *>(src0),
                                                                      reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, float, float, 64, 64, 128, 0, 0, 32, true, true>(reinterpret_cast<__gm__ float *>(out),
                                                                        reinterpret_cast<__gm__ float *>(src0),
                                                                        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<int32_t, int8_t, int8_t, 128, 64, 128, 32, 0, 0, true, true>(reinterpret_cast<__gm__ int32_t *>(out),
                                                                             reinterpret_cast<__gm__ int8_t *>(src0),
                                                                             reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_10(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, bfloat16_t, bfloat16_t, 64, 128, 64, 16, 0, 0, true, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, float8_e4m3_t, float8_e4m3_t, 64, 128, 64, 0, 32, 0, true, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, float8_e5m2_t, float8_e5m2_t, 64, 128, 64, 0, 0, 32, false, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<float, hifloat8_t, hifloat8_t, 64, 128, 64, 0, 0, 32, false, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_DYNAMIC<int32_t, int8_t, int8_t, 64, 96, 32, 32, 0, 0, true, false>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), 64, 96, 32);
}
extern "C" __global__ AICORE void launchTEXTRACT_15(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_DYNAMIC<float, half, half, 64, 48, 96, 16, 16, 0, true, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), 64, 48, 96);
}
extern "C" __global__ AICORE void launchTEXTRACT_16(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_DYNAMIC<float, float, float, 32, 96, 48, 0, 32, 16, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1), 32, 96, 48);
}
extern "C" __global__ AICORE void launchTEXTRACT_17(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                    __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTEXTRACTMX<float, float4_e2m1x2_t, float4_e2m1x2_t, fp8_e8m0_t, 256, 128, 256, 128, 64, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}
extern "C" __global__ AICORE void launchTEXTRACT_18(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                    __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTEXTRACTMX<float, float4_e1m2x2_t, float4_e1m2x2_t, fp8_e8m0_t, 256, 128, 256, 128, 64, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}
extern "C" __global__ AICORE void launchTEXTRACT_19(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                    __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTEXTRACTMX<float, float4_e2m1x2_t, float4_e2m1x2_t, fp8_e8m0_t, 256, 128, 256, 128, 64, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}
extern "C" __global__ AICORE void launchTEXTRACT_20(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                    __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTEXTRACTMX<float, float4_e1m2x2_t, float4_e1m2x2_t, fp8_e8m0_t, 256, 128, 256, 128, 64, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTEXTRACT_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTEXTRACT_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTEXTRACT_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTEXTRACT_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTEXTRACT_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTEXTRACT_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTEXTRACT_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTEXTRACT_8<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 9) {
        launchTEXTRACT_9<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 10) {
        launchTEXTRACT_10<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTEXTRACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTEXTRACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTEXTRACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 14) {
        launchTEXTRACT_14<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 15) {
        launchTEXTRACT_15<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 16) {
        launchTEXTRACT_16<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <int32_t tilingKey>
void launchTEXTRACTMX(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1, void *stream)
{
    if constexpr (tilingKey == 17) {
        launchTEXTRACT_17<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    } else if constexpr (tilingKey == 18) {
        launchTEXTRACT_18<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    } else if constexpr (tilingKey == 19) {
        launchTEXTRACT_19<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    } else if constexpr (tilingKey == 20) {
        launchTEXTRACT_20<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    }
}
template void launchTEXTRACT<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<15>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<16>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACTMX<17>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                                   void *stream);
template void launchTEXTRACTMX<18>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                                   void *stream);
template void launchTEXTRACTMX<19>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                                   void *stream);
template void launchTEXTRACTMX<20>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                                   void *stream);

template <typename T, typename U, typename S, int M, int K, int N, bool isAtranspose, bool isBtranspose>
AICORE inline void runTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose, Tile<TileType::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose, Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<S, K, N, K, N>;
    using ResTile = TileAcc<T, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, bool isAtranspose, bool isBtranspose>
AICORE inline void runTMOV_DYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int k, int n)
{
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;

    using DynShape3Dim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTrid3Dim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShape3Dim5, DynSTrid3Dim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<TileType::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose,
                           Tile<TileType::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, M, K, -1, -1>;
    using RightTile = TileRight<S, K, N, K, -1>;
    using ResTile = TileAcc<T, M, N, -1, N>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile(m, k);
    RightTile bTile(n);
    ResTile cTile(m);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, bool isAtranspose, bool isBtranspose, int targetM,
          int targetK, int targetN>
AICORE inline void runTMOV_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>,
                     pto::Stride<1 * targetM * targetK, 1 * targetM * targetK, targetM * targetK, 1, targetM>,
                     Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>,
                     pto::Stride<1 * targetM * targetK, 1 * targetM * targetK, targetM * targetK, targetK, 1>,
                     Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>,
                     pto::Stride<1 * targetK * targetN, 1 * targetK * targetN, targetK * targetN, targetN, 1>,
                     Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>,
                     pto::Stride<1 * targetK * targetN, 1 * targetK * targetN, targetK * targetN, 1, targetK>,
                     Layout::DN>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<TileType::Mat, U, targetM, targetK, BLayout::RowMajor, targetM, targetK, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, U, targetM, targetK, BLayout::ColMajor, targetM, targetK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<
        isBtranspose,
        Tile<TileType::Mat, S, targetK, targetN, BLayout::ColMajor, targetK, targetN, SLayout::RowMajor, 512>,
        Tile<TileType::Mat, S, targetK, targetN, BLayout::RowMajor, targetK, targetN, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, targetM, targetK, targetM, K>;
    using RightTile = TileRight<S, targetK, targetN, K, targetN>;
    using ResTile = TileAcc<T, targetM, targetN, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /*********************************TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename X, int M, int K, int N, bool isAtranspose, bool isBtranspose>
AICORE inline void runTMOVMX(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ X *srcMx0, __gm__ X *srcMx1)

{
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose, Tile<TileType::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose, Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>>;

    using TileMatAmxData = Tile<TileType::Mat, X, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>; // temporary
    using TileMatBmxData = Tile<TileType::Mat, X, K, N, BLayout::ColMajor, K, N, SLayout::ColMajor, 512>;

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<S, K, N, K, N>;
    using ResTile = TileAcc<T, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatAmxData amxMatTile;
    TileMatBmxData bmxMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(amxMatTile, 0x20000);
    TASSIGN(bmxMatTile, 0x30000);

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    auto &a = aTile.data();
    auto &b = bTile.data();
    auto &c = cTile.data();
    auto &srcAmxAddr = amxMatTile.data();
    auto &srcBmxAddr = bmxMatTile.data();

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    DynGM2L1<TileMatAmxData, X>(srcAmxAddr, srcMx0, M, K / 32);
    DynGM2L1<TileMatBmxData, X>(srcBmxAddr, srcMx1, K / 32, N);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    MOV_MX_TOA<LeftTile, TileMatAmxData>(a, srcAmxAddr, M, K / 32);
    MOV_MX_TOB<RightTile, TileMatBmxData>(b, srcBmxAddr, K / 32, N);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    tf_mad_mx<ResTile, LeftTile, RightTile>(c, a, b, M, K, N, false, false, false, true);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTMOV_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV<float, half, half, 32, 96, 64, false, false>(reinterpret_cast<__gm__ float *>(out),
                                                         reinterpret_cast<__gm__ half *>(src0),
                                                         reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV<float, float, float, 128, 48, 64, false, false>(reinterpret_cast<__gm__ float *>(out),
                                                            reinterpret_cast<__gm__ float *>(src0),
                                                            reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV<int32_t, int8_t, int8_t, 128, 128, 64, false, false>(reinterpret_cast<__gm__ int32_t *>(out),
                                                                 reinterpret_cast<__gm__ int8_t *>(src0),
                                                                 reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV<float, bfloat16_t, bfloat16_t, 64, 128, 64, true, true>(reinterpret_cast<__gm__ float *>(out),
                                                                    reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                                                    reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV<float, float8_e4m3_t, float8_e4m3_t, 64, 96, 64, true, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV<float, float8_e5m2_t, float8_e5m2_t, 64, 128, 64, false, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV<float, hifloat8_t, hifloat8_t, 128, 128, 64, true, false>(reinterpret_cast<__gm__ float *>(out),
                                                                      reinterpret_cast<__gm__ hifloat8_t *>(src0),
                                                                      reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV_DYNAMIC<int32_t, int8_t, int8_t, 64, 96, 64, true, true>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), 64, 96, 64);
}
extern "C" __global__ AICORE void launchTMOV_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV_DYNAMIC<float, half, half, 64, 128, 64, true, false>(reinterpret_cast<__gm__ float *>(out),
                                                                 reinterpret_cast<__gm__ half *>(src0),
                                                                 reinterpret_cast<__gm__ half *>(src1), 64, 128, 64);
}
extern "C" __global__ AICORE void launchTMOV_10(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV_DYNAMIC<float, float, float, 64, 128, 64, false, true>(reinterpret_cast<__gm__ float *>(out),
                                                                   reinterpret_cast<__gm__ float *>(src0),
                                                                   reinterpret_cast<__gm__ float *>(src1), 64, 128, 64);
}
extern "C" __global__ AICORE void launchTMOV_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV_UNALIGN<int32_t, int8_t, int8_t, 65, 40, 66, true, true, 96, 64, 96>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV_UNALIGN<float, half, half, 65, 40, 66, true, true, 80, 48, 80>(reinterpret_cast<__gm__ float *>(out),
                                                                           reinterpret_cast<__gm__ half *>(src0),
                                                                           reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTMOV_UNALIGN<float, float, float, 65, 40, 66, true, true, 80, 48, 80>(reinterpret_cast<__gm__ float *>(out),
                                                                             reinterpret_cast<__gm__ float *>(src0),
                                                                             reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTMOV_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTMOVMX<float, float4_e2m1x2_t, float4_e2m1x2_t, fp8_e8m0_t, 128, 64, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}
extern "C" __global__ AICORE void launchTMOV_15(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTMOVMX<float, float4_e1m2x2_t, float4_e1m2x2_t, fp8_e8m0_t, 128, 64, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}
extern "C" __global__ AICORE void launchTMOV_16(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTMOVMX<float, float4_e2m1x2_t, float4_e2m1x2_t, fp8_e8m0_t, 128, 64, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}
extern "C" __global__ AICORE void launchTMOV_17(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
                                                __gm__ uint8_t *srcMx0, __gm__ uint8_t *srcMx1)
{
    runTMOVMX<float, float4_e1m2x2_t, float4_e1m2x2_t, fp8_e8m0_t, 128, 64, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
        reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1), reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx0),
        reinterpret_cast<__gm__ fp8_e8m0_t *>(srcMx1));
}

template <int32_t tilingKey>
void launchTMOV(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMOV_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMOV_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMOV_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMOV_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTMOV_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTMOV_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTMOV_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTMOV_8<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 9) {
        launchTMOV_9<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 10) {
        launchTMOV_10<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTMOV_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTMOV_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTMOV_13<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <int32_t tilingKey>
void launchTMOVMX(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1, void *stream)
{
    if constexpr (tilingKey == 14) {
        launchTMOV_14<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    } else if constexpr (tilingKey == 15) {
        launchTMOV_15<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    } else if constexpr (tilingKey == 16) {
        launchTMOV_16<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    } else if constexpr (tilingKey == 17) {
        launchTMOV_17<<<1, nullptr, stream>>>(out, src0, src1, srcMx0, srcMx1);
    }
}

template void launchTMOV<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVMX<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                               void *stream);
template void launchTMOVMX<15>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                               void *stream);
template void launchTMOVMX<16>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                               void *stream);
template void launchTMOVMX<17>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *srcMx0, uint8_t *srcMx1,
                               void *stream);