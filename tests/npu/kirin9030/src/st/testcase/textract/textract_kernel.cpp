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

using namespace pto;

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
    TASSIGN(bMatTile, M * K * sizeof(U));

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

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
    TASSIGN(bMatTile, M * K * sizeof(U));

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
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTEXTRACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<half, half, half, 32, 96, 64, 0, 0, 0, false, false>(reinterpret_cast<__gm__ half *>(out),
                                                                     reinterpret_cast<__gm__ half *>(src0),
                                                                     reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<int32_t, int8_t, int8_t, 128, 128, 64, 0, 0, 0, false, false>(reinterpret_cast<__gm__ int32_t *>(out),
                                                                              reinterpret_cast<__gm__ int8_t *>(src0),
                                                                              reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<half, half, half, 64, 96, 64, 32, 16, 16, false, false>(reinterpret_cast<__gm__ half *>(out),
                                                                        reinterpret_cast<__gm__ half *>(src0),
                                                                        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<int32_t, int8_t, int8_t, 128, 128, 64, 32, 64, 32, false, false>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<half, half, half, 64, 128, 64, 0, 64, 0, true, true>(reinterpret_cast<__gm__ half *>(out),
                                                                     reinterpret_cast<__gm__ half *>(src0),
                                                                     reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT<int32_t, int8_t, int8_t, 128, 64, 128, 32, 0, 0, true, true>(reinterpret_cast<__gm__ int32_t *>(out),
                                                                             reinterpret_cast<__gm__ int8_t *>(src0),
                                                                             reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_DYNAMIC<int32_t, int8_t, int8_t, 64, 96, 32, 32, 0, 0, true, false>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), 64, 96, 32);
}
extern "C" __global__ AICORE void launchTEXTRACT_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_DYNAMIC<half, half, half, 64, 48, 96, 16, 16, 0, true, false>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), 64, 48, 96);
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