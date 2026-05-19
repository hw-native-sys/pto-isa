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

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename OutType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
          bool isBias>
__global__ AICORE void RunTMATMUL(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ BiasType *src2)
{
    constexpr int blockAlign = 32;
    constexpr int M = CeilAlign<int>(validM, 16);
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
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN<0x0>(aTile);
    TASSIGN<0x0>(bTile);
    TASSIGN<0x0>(cTile);
    TASSIGN<0x0>(biasTile);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <typename OutType, typename AType, typename BType, typename BiasType, int M, int K, int N, bool isBias>
__global__ AICORE void RunTMATMUL_SPLIT_K(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
                                          __gm__ BiasType *src2)
{
    constexpr int BASEM = 128;
    constexpr int BASEK = 64;
    constexpr int BASEN = 64;
    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, M, BASEK>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, BASEK, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, BASEM, BASEK, BLayout::ColMajor, M, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, BASEK, BASEN, BLayout::ColMajor, BASEK, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, BASEN, BLayout::RowMajor, 1, BASEN>;

    using LeftTile = TileLeft<AType, BASEM, BASEK, M, BASEK>;
    using RightTile = TileRight<BType, BASEK, BASEN, BASEK, N>;
    using AccTile = TileAcc<OutType, BASEM, BASEN, M, N>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, BASEN, BLayout::RowMajor, 1, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN<0x0>(aTile);
    TASSIGN<0x0>(bTile);
    TASSIGN<0x0>(cTile);
    TASSIGN<0x0>(biasTile);

    constexpr int iter = K / BASEK;

    for (int i = 0; i < iter; i++) { // baseK = 64
        /*************************************TLOAD****************************************/
        GlobalDataSrc0 src0Global(src0 + i * BASEK);
        GlobalDataSrc1 src1Global(src1 + i * BASEK * N);
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);
        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**********************************TMOV && TEXTRACT**********************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);
        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
            } else {
                TMATMUL(cTile, aTile, bTile);
            }
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<half, half, half, half, 40, 50, 60, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 6, 7, 8, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<half, half, half, half, 1, 16, 512, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 26, 15, 27, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 101, 1, 99, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), nullptr);
    }
}

template void LaunchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 8, 7, 6, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<half, half, half, half, 16, 15, 16, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 66, 11, 1, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<half, half, half, half, 1, 16, 1, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<half, half, half, half, 29, 11, 41, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    }
}

template void LaunchTMATMULBIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
