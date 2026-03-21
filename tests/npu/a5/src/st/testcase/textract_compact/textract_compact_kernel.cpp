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

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexM, uint16_t indexK, uint16_t indexN,
          int baseM, int baseK, int baseN, bool isAtranspose, bool isBtranspose>
AICORE inline void runTEXTRACT_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, 1, baseM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<1 * baseK * baseN, 1 * baseK * baseN, baseK * baseN, baseN, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<1 * baseK * baseN, 1 * baseK * baseN, baseK * baseN, 1, baseK>, Layout::DN>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose, Tile<TileType::Mat, U, baseM, baseK, BLayout::RowMajor, baseM, baseK, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<
        isBtranspose, Tile<TileType::Mat, S, baseK, baseN, BLayout::ColMajor, baseK, baseN, SLayout::RowMajor, 512>,
        Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeftCompact<U, baseM, baseK, mValid, kValid>;
    using RightTile = TileRightCompact<S, baseK, baseN, kValid, nValid>;
    using ResTile = TileAccCompact<T, baseM, baseN, mValid, nValid>;

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

extern "C" __global__ AICORE void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, half, half, 63, 48, 66, 0, 0, 0, 128, 64, 256, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, half, half, 68, 93, 97, 0, 0, 0, 128, 128, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, half, half, 75, 201, 79, 16, 16, 16, 80, 256, 80, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, half, half, 59, 232, 61, 16, 16, 16, 64, 256, 64, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_21(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float, float, 68, 70, 69, 0, 0, 0, 80, 128, 80, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_22(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float, float, 20, 22, 21, 0, 0, 0, 64, 96, 64, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_23(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float, float, 49, 119, 63, 16, 32, 16, 64, 128, 64, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_24(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float, float, 127, 60, 102, 16, 16, 32, 128, 64, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_31(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, 97, 231, 83, 0, 0, 0, 128, 256, 128, false, false>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_32(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, 71, 188, 82, 0, 0, 0, 128, 256, 128, true, true>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_33(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, 63, 112, 98, 32, 32, 32, 64, 128, 128, false, false>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_34(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, 106, 125, 60, 32, 32, 32, 128, 128, 64, true, true>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_41(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, 23, 24, 25, 0, 0, 0, 96, 64, 96, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_42(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, 23, 24, 25, 0, 0, 0, 96, 64, 96, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_43(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, 39, 40, 41, 16, 16, 16, 96, 64, 96, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_44(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, 39, 40, 41, 16, 16, 16, 96, 64, 96, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_51(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, hifloat8_t, hifloat8_t, 46, 40, 45, 0, 0, 0, 128, 96, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_52(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, hifloat8_t, hifloat8_t, 46, 40, 45, 0, 0, 0, 128, 96, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_53(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, hifloat8_t, hifloat8_t, 78, 72, 77, 32, 32, 32, 128, 96, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_54(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, hifloat8_t, hifloat8_t, 78, 72, 77, 32, 32, 32, 128, 96, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_61(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e4m3_t, float8_e4m3_t, 46, 40, 45, 0, 0, 0, 128, 96, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_62(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e4m3_t, float8_e4m3_t, 46, 40, 45, 0, 0, 0, 128, 96, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_63(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e4m3_t, float8_e4m3_t, 78, 72, 77, 32, 32, 32, 128, 96, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_64(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e4m3_t, float8_e4m3_t, 78, 72, 77, 32, 32, 32, 128, 96, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_71(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e5m2_t, float8_e5m2_t, 46, 40, 45, 0, 0, 0, 128, 96, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_72(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e5m2_t, float8_e5m2_t, 46, 40, 45, 0, 0, 0, 128, 96, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_73(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e5m2_t, float8_e5m2_t, 78, 72, 77, 32, 32, 32, 128, 96, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_74(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float8_e5m2_t, float8_e5m2_t, 78, 72, 77, 32, 32, 32, 128, 96, 128, true, true>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 11) {
        launchTEXTRACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTEXTRACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTEXTRACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 14) {
        launchTEXTRACT_14<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        launchTEXTRACT_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        launchTEXTRACT_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        launchTEXTRACT_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 24) {
        launchTEXTRACT_24<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        launchTEXTRACT_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        launchTEXTRACT_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        launchTEXTRACT_33<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 34) {
        launchTEXTRACT_34<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 41) {
        launchTEXTRACT_41<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 42) {
        launchTEXTRACT_42<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 43) {
        launchTEXTRACT_43<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 44) {
        launchTEXTRACT_44<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 51) {
        launchTEXTRACT_51<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 52) {
        launchTEXTRACT_52<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 53) {
        launchTEXTRACT_53<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 54) {
        launchTEXTRACT_54<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 61) {
        launchTEXTRACT_61<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 62) {
        launchTEXTRACT_62<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 63) {
        launchTEXTRACT_63<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 64) {
        launchTEXTRACT_64<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 71) {
        launchTEXTRACT_71<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 72) {
        launchTEXTRACT_72<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 73) {
        launchTEXTRACT_73<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 74) {
        launchTEXTRACT_74<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTEXTRACT<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<24>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<34>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<42>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<43>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<44>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<51>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<52>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<53>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<54>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<61>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<62>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<63>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<64>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<71>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<72>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<73>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<74>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);