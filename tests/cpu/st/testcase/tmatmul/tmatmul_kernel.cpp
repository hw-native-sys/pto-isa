
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

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename outType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
          bool isBias>
__global__ AICORE void RunTMATMUL(__gm__ outType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ BiasType *src2)
{
    constexpr int blockAlign = (sizeof(AType) == 1) ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, validN>,
                                        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    using GlobalDataOut =
        GlobalTensor<outType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    int offset = 0;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<outType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    if constexpr (isBias) {
        GlobalDataSrc2 src2Global(src2);
        TLOAD(biasTile, src2Global);
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N>
AICORE inline void RunTMATMUL_SPLIT_K(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, uint32_t numRepeats)
{
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // L1上都是大n小z
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<S, K, N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    for (uint32_t i = 0; i < numRepeats; i++) {
        /******************************TLOAD*****************************/
        GlobalDataSrc0 src0Global(src0 + i * M * K);
        GlobalDataSrc1 src1Global(src1 + i * K * N);

        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
    }

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<float, half, half, float, 40, 50, 60, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<int32_t, int8_t, int8_t, int8_t, 6, 7, 8, false>(reinterpret_cast<int32_t *>(out),
                                                                    reinterpret_cast<int8_t *>(src0),
                                                                    reinterpret_cast<int8_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL_SPLIT_K<float, half, half, 128, 128, 64>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), 5);
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<float, float, float, float, 120, 110, 50, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), nullptr);
#ifdef CPU_SIM_BFLOAT_ENABLED
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, float, 40, 50, 60, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0), reinterpret_cast<bfloat16_t *>(src1),
            nullptr);
#endif
    }
}

template void LaunchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTMATMUL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
#endif

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 8, 7, 6, true>(
            reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1),
            reinterpret_cast<int32_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<float, half, half, float, 16, 15, 16, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1),
            reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<float, float, float, float, 127, 128, 63, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1),
            reinterpret_cast<float *>(src2));
#ifdef CPU_SIM_BFLOAT_ENABLED
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, float, 16, 15, 16, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0), reinterpret_cast<bfloat16_t *>(src1),
            reinterpret_cast<float *>(src2));
#endif
    }
}

template void LaunchTMATMULBIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTMATMULBIAS<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
#endif
