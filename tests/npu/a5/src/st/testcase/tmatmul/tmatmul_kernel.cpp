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

constexpr uint16_t BLOCK_CUBE_M_N = 16;

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
    constexpr int blockAlign = (sizeof(AType) == 1) ? 32 : 16;
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
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasDataTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    /**********************************TMATMUL**********************************/
    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

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
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, BASEN, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<AType, BASEM, BASEK, M, BASEK>;
    using RightTile = TileRight<BType, BASEK, BASEN, BASEK, N>;
    using AccTile = TileAcc<OutType, BASEM, BASEN, M, N>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, BASEN, BLayout::RowMajor, 1, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasDataTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

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

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        /**********************************TMOV && TEXTRACT**********************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);
        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
            } else {
                TMATMUL(cTile, aTile, bTile);
            }
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
#endif
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTGEMV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 =
        GlobalTensor<U, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<S, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);

    constexpr int blockLeft = CUBE_BLOCK_SIZE / sizeof(U);
    constexpr int KLeft = CeilAlign<int>(validK, blockLeft);
    using TileMatAData = Tile<TileType::Mat, U, 1, KLeft, BLayout::RowMajor, 1, validK>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, 1, KLeft, 1, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

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

    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**************************TMOV && TEXTRACT**************************/
    TEXTRACT(aTile, aMatTile, 0, 0);
    TMOV(bTile, bMatTile);

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    if constexpr (isBias) {
        TGEMV_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TGEMV(cTile, aTile, bTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename OutType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
          bool isBias, RoundMode tf32TransMode>
__global__ AICORE void RunTMATMUL_TF32(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
                                       __gm__ BiasType *src2)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);

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
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasDataTile, 0x20000);

    LeftTile aTile;
    aTile.SetMadTF32Mode(tf32TransMode);
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    /**********************************TMATMUL**********************************/
    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }
    aTile.ResetMadMode();
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTGEMV_SPLIT_K(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    constexpr int BASEK = 256;

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, BASEK);

    using GlobalDataSrc0 =
        GlobalTensor<U, pto::Shape<1, 1, 1, validM, BASEK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, BASEK, validN>,
                                        pto::Stride<1 * BASEK * validN, 1 * BASEK * validN, BASEK * validN, validN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatADataGemv = Tile<TileType::Mat, U, 1, BASEK, BLayout::RowMajor, 1, BASEK>;
    using TileMatBData = Tile<TileType::Mat, S, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, 1, BASEK, 1, BASEK>;
    using RightTile = TileRight<S, BASEK, N, BASEK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatADataGemv aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    constexpr int iter = K / BASEK;

    for (int i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(src0 + i * BASEK);
        GlobalDataSrc1 src1Global(src1 + validN * i * BASEK);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        if (i == 0) {
            if constexpr (isBias) {
                TGEMV_BIAS<AccPhase::Partial>(cTile, aTile, bTile, biasTile);
            } else {
                TGEMV<AccPhase::Partial>(cTile, aTile, bTile);
            }
        } else if (i == iter - 1) {
            TGEMV_ACC<AccPhase::Final>(cTile, cTile, aTile, bTile);
        } else {
            TGEMV_ACC<AccPhase::Partial>(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
#endif
    }

    TSTORE<STPhase::Final>(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<float, half, half, float, 40, 50, 60, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<int32_t, int8_t, int8_t, int8_t, 6, 7, 8, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL_SPLIT_K<float, half, half, float, 127, 128, 61, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<float, float, float, float, 120, 110, 50, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, float, 144, 80, 48, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<float, float8_e4m3_t, float8_e4m3_t, float, 32, 64, 96, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL<float, float8_e4m3_t, float8_e5m2_t, float, 128, 96, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 8) {
        RunTMATMUL<float, float8_e5m2_t, float8_e4m3_t, float, 145, 115, 85, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 9) {
        RunTMATMUL<float, float8_e5m2_t, float8_e5m2_t, float, 120, 90, 160, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 10) {
        RunTMATMUL<float, hifloat8_t, hifloat8_t, float, 30, 90, 60, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<hifloat8_t *>(src0),
                                     reinterpret_cast<hifloat8_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 11) {
        RunTGEMV<float, half, half, float, 1, 300, 60, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 12) {
        RunTMATMUL_TF32<float, float, float, float, 16, 32, 64, false, RoundMode::CAST_RINT><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), nullptr);
    } else if constexpr (tilingKey == 13) {
        RunTMATMUL_TF32<float, float, float, float, 128, 96, 64, false, RoundMode::CAST_ROUND><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), nullptr);
    }
}

template void LaunchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 8, 7, 6, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<float, half, half, half, 16, 15, 16, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<float, half, half, bfloat16_t, 112, 127, 80, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<bfloat16_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, bfloat16_t, 80, 112, 63, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1), reinterpret_cast<bfloat16_t *>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL_SPLIT_K<float, float, float, float, 127, 128, 63, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<float, float8_e4m3_t, float8_e4m3_t, float, 120, 90, 160, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL<float, float8_e4m3_t, float8_e5m2_t, float, 32, 64, 96, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 8) {
        RunTMATMUL<float, float8_e5m2_t, float8_e4m3_t, float, 128, 96, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 9) {
        RunTMATMUL<float, float8_e5m2_t, float8_e5m2_t, float, 30, 90, 60, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 10) {
        RunTMATMUL<float, hifloat8_t, hifloat8_t, float, 145, 115, 85, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<hifloat8_t *>(src0),
                                     reinterpret_cast<hifloat8_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 11) {
        RunTGEMV_SPLIT_K<float, half, half, float, 1, 512, 85, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    }
}

template void LaunchTMATMULBIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
