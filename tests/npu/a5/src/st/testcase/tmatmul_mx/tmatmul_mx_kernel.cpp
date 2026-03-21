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

template <typename T>
AICORE inline constexpr T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
AICORE inline constexpr T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T, int format, int M, int KMX>
using GlobalDataSrc2_t = std::conditional_t<
    (format == 0),
    GlobalTensor<T, TileShape2D<T, M, KMX, Layout::MX_A_ZZ>, BaseShape2D<T, M, KMX, Layout::MX_A_ZZ>, Layout::MX_A_ZZ>,
    std::conditional_t<(format == 1),
                       GlobalTensor<T, TileShape2D<T, M, KMX, Layout::MX_A_ND>, BaseShape2D<T, M, KMX, Layout::MX_A_ND>,
                                    Layout::MX_A_ND>,
                       GlobalTensor<T, TileShape2D<T, M, KMX, Layout::MX_A_DN>, BaseShape2D<T, M, KMX, Layout::MX_A_DN>,
                                    Layout::MX_A_DN>>>;

template <typename T, int format, int KMX, int N>
using GlobalDataSrc3_t = std::conditional_t<
    (format == 0),
    GlobalTensor<T, TileShape2D<T, KMX, N, Layout::MX_B_NN>, BaseShape2D<T, KMX, N, Layout::MX_B_NN>, Layout::MX_B_NN>,
    std::conditional_t<(format == 1),
                       GlobalTensor<T, TileShape2D<T, KMX, N, Layout::MX_B_ND>, BaseShape2D<T, KMX, N, Layout::MX_B_ND>,
                                    Layout::MX_B_ND>,
                       GlobalTensor<T, TileShape2D<T, KMX, N, Layout::MX_B_DN>, BaseShape2D<T, KMX, N, Layout::MX_B_DN>,
                                    Layout::MX_B_DN>>>;

template <typename OutType, typename AType, typename BType, typename ScaleType, typename BiasType, int validM,
          int validK, int validN, bool isBias, bool isFp4>
__global__ AICORE void RunTMATMULMX(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ ScaleType *src2,
                                    __gm__ ScaleType *src3, __gm__ BiasType *src4)
{
    constexpr int blockAlign = isFp4 ? 64 : 32; // need to be 32B aligned

    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int kAlign = CeilAlign<int>(validK, 64);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr uint8_t kMX = CeilDiv(kAlign, 32);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using MxShapeA = TileShape2D<ScaleType, M, kMX, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<ScaleType, M, kMX, Layout::MX_A_ZZ>;
    using GlobalDataSrc2 = GlobalTensor<ScaleType, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;

    using MxShapeB = TileShape2D<ScaleType, kMX, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<ScaleType, kMX, N, Layout::MX_B_NN>;
    using GlobalDataSrc3 = GlobalTensor<ScaleType, MxShapeB, MxStrideB, Layout::MX_B_NN>;

    using GlobalDataSrc4 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, validN>,
                                        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataSrc4 src4Global(src4);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, AType, M, kAlign, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, BType, kAlign, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using TileScaleAData =
        Tile<TileType::Mat, ScaleType, M, kMX, BLayout::RowMajor, validM, kMX, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, kMX, N, BLayout::ColMajor, kMX, validN, SLayout::ColMajor, 32>;

    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<AType, M, kAlign, validM, kAlign>;
    using RightTile = TileRight<BType, kAlign, N, kAlign, validN>;
    using LeftScaleTile = TileLeftScale<ScaleType, M, kMX, validM, kMX>;
    using RightScaleTile = TileRightScale<ScaleType, kMX, N, kMX, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * kAlign);
    TASSIGN(aScaleMatTile, M * kAlign + kAlign * N);
    TASSIGN(bScaleMatTile, M * kAlign + kAlign * N + M * kMX);
    TASSIGN(biasDataTile, M * kAlign + kAlign * N + M * kMX + N * kMX);

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

#ifndef __PTO_AUTO__
    uint64_t scaleAAddr = GetScaleAddr(aTile.data());
    uint64_t scaleBAddr = GetScaleAddr(bTile.data());
    TASSIGN(aScaleTile, scaleAAddr);
    TASSIGN(bScaleTile, scaleBAddr);
#endif

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    // Clear L1 buffer
    // Tload will pad to 32B alignment with at most 32B padding
    if constexpr (kAlign - validK >= blockAlign) {
        TFILLPAD(aMatTile, aMatTile);
    }
    TFILLPAD(bMatTile, bMatTile);

    TLOAD<TileScaleAData, GlobalDataSrc2>(aScaleMatTile, src2Global);
    TLOAD<TileScaleBData, GlobalDataSrc3>(bScaleMatTile, src3Global);

    if constexpr (isBias) {
        TLOAD(biasDataTile, src4Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    TMOV(aScaleTile, aScaleMatTile);
    TMOV(bScaleTile, bScaleMatTile);

#ifdef __PTO_AUTO__
    TGET_SCALE_ADDR(aScaleTile, aTile);
    TGET_SCALE_ADDR(bScaleTile, bTile);
#endif

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    /**********************************TMATMUL**********************************/
    if constexpr (isBias) {
        TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile, biasTile);
    } else {
        TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <typename OutType, typename AType, typename BType, typename ScaleType, typename BiasType, int validM,
          int validK, int validN, bool isBias, bool isFp4>
__global__ AICORE void RunTMATMULMX_SPLIT_K(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
                                            __gm__ ScaleType *src2, __gm__ ScaleType *src3, __gm__ BiasType *src4)
{
    constexpr int blockAlign = isFp4 ? 64 : 32; // need to be 32B aligned

    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int K = CeilAlign<int>(validK, 64);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int KMX = CeilDiv(K, 32);

    constexpr int BASEK = 64;
    constexpr int BASEKMX = CeilDiv(BASEK, 32);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, BASEK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, BASEK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using MxShapeA = TileShape2D<ScaleType, M, BASEKMX, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<ScaleType, M, KMX, Layout::MX_A_ZZ>;
    using GlobalDataSrc2 = GlobalTensor<ScaleType, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;

    using MxShapeB = TileShape2D<ScaleType, BASEKMX, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<ScaleType, KMX, N, Layout::MX_B_NN>;
    using GlobalDataSrc3 = GlobalTensor<ScaleType, MxShapeB, MxStrideB, Layout::MX_B_NN>;

    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc4 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, validN>,
                                        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    GlobalDataSrc4 src4Global(src4);

    using TileMatAData = Tile<TileType::Mat, AType, M, BASEK, BLayout::ColMajor, validM, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::RowMajor, 512>;

    using TileScaleAData =
        Tile<TileType::Mat, ScaleType, M, BASEKMX, BLayout::RowMajor, validM, BASEKMX, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, BASEKMX, N, BLayout::ColMajor, BASEKMX, validN, SLayout::ColMajor, 32>;

    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<AType, M, BASEK, validM, BASEK>;
    using RightTile = TileRight<BType, BASEK, N, BASEK, validN>;
    using LeftScaleTile = TileLeftScale<ScaleType, M, BASEKMX, validM, BASEKMX>;
    using RightScaleTile = TileRightScale<ScaleType, BASEKMX, N, BASEKMX, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;
    TileBiasData biasDataTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * BASEK);
    TASSIGN(aScaleMatTile, M * BASEK + N * BASEK);
    TASSIGN(bScaleMatTile, M * BASEK + N * BASEK + M * BASEKMX);
    TASSIGN(biasDataTile, M * BASEK + N * BASEK + M * BASEKMX + N * BASEKMX);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

#ifndef __PTO_AUTO__
    uint64_t scaleAAddr = GetScaleAddr(aTile.data());
    uint64_t scaleBAddr = GetScaleAddr(bTile.data());
    TASSIGN(aScaleTile, scaleAAddr);
    TASSIGN(bScaleTile, scaleBAddr);
#endif

    constexpr int iter = K / BASEK;
    for (int i = 0; i < iter; i++) {
        const int offsetA = (!isFp4) ? (i * BASEK) : (i * BASEK / 2);
        const int offsetB = (!isFp4) ? (validN * i * BASEK) : (validN * i * BASEK / 2);
        GlobalDataSrc0 src0Global(src0 + offsetA);
        GlobalDataSrc1 src1Global(src1 + offsetB);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        const int offsetAMX = i * BASEKMX * 16;
        const int offsetBMX = 16 * i * BASEKMX;
        GlobalDataSrc2 src2Global(src2 + offsetAMX);
        GlobalDataSrc3 src3Global(src3 + offsetBMX);

        TLOAD<TileScaleAData, GlobalDataSrc2>(aScaleMatTile, src2Global);
        TLOAD<TileScaleBData, GlobalDataSrc3>(bScaleMatTile, src3Global);

        if constexpr (isBias) {
            TLOAD(biasDataTile, src4Global);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        TMOV(aScaleTile, aScaleMatTile);
        TMOV(bScaleTile, bScaleMatTile);

#ifdef __PTO_AUTO__
        TGET_SCALE_ADDR(aScaleTile, aTile);
        TGET_SCALE_ADDR(bScaleTile, bTile);
#endif

        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile, biasTile);
            } else {
                TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
            }
        } else {
            TMATMUL_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
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

template <int format, typename OutType, typename AType, typename BType, typename ScaleType, int validM, int validK,
          int validN, bool isFp4>
__global__ AICORE void RunTGEMVMX(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ ScaleType *src2,
                                  __gm__ ScaleType *src3)
{
    constexpr int blockAlign = isFp4 ? 64 : 32; // need to be 32B aligned

    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int kAlign = CeilAlign<int>(validK, 64);
    constexpr int N = CeilAlign<int>(validN, blockAlign);

    constexpr uint8_t kMX = CeilDiv(kAlign, 32);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<ScaleType, pto::Shape<1, 1, 1, 1, kMX>, pto::Stride<kMX, kMX, kMX, kMX, 1>>;

    using GlobalDataSrc3 = GlobalDataSrc3_t<ScaleType, format, kMX, validN>;

    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataOut dstGlobal(out);

    constexpr int blockLeft = isFp4 ? 1024 : 512;
    constexpr int KLeft = CeilAlign<int>(validK, blockLeft);
    using TileMatAData = Tile<TileType::Mat, AType, 1, KLeft, BLayout::RowMajor, 1, validK>;

    using TileScaleAData = Tile<TileType::Mat, ScaleType, 1, kMX, BLayout::RowMajor, 1, kMX, SLayout::RowMajor, 32>;

    using TileMatBData =
        Tile<TileType::Mat, BType, kAlign, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, kMX, N, BLayout::ColMajor, kMX, validN, SLayout::ColMajor, 32>;

    using LeftTile = TileLeft<AType, 1, KLeft, 1, validK>;
    using RightTile = TileRightCompact<BType, kAlign, N, kAlign, validN>;
    using AccTile = TileAccCompact<OutType, M, N, validM, validN>;

    using LeftScaleTile = TileLeftScaleCompact<ScaleType, 1, kMX, 1, kMX>;
    using RightScaleTile = TileRightScaleCompact<ScaleType, kMX, N, kMX, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(aScaleMatTile, 0x20000);
    TASSIGN(bScaleMatTile, 0x30000);

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    AccTile cTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

#ifndef __PTO_AUTO__
    uint64_t scaleAAddr = GetScaleAddr(aTile.data());
    uint64_t scaleBAddr = GetScaleAddr(bTile.data());
    TASSIGN(aScaleTile, scaleAAddr);
    TASSIGN(bScaleTile, scaleBAddr);
#endif

    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TFILLPAD(bMatTile, bMatTile);

    TLOAD<TileScaleAData, GlobalDataSrc2>(aScaleMatTile, src2Global);
    TLOAD<TileScaleBData, GlobalDataSrc3>(bScaleMatTile, src3Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/

    TEXTRACT(aTile, aMatTile, 0, 0);
    TEXTRACT(bTile, bMatTile, 0, 0);

    TMOV(aScaleTile, aScaleMatTile);
    TMOV(bScaleTile, bScaleMatTile);

#ifdef __PTO_AUTO__
    TGET_SCALE_ADDR(aScaleTile, aTile);
    TGET_SCALE_ADDR(bScaleTile, bTile);
#endif

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    /**********************************TMATMUL**********************************/

    TGEMV_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <typename OutType, typename AType, typename BType, typename ScaleType, typename BiasType, int validM,
          int validK, int validN, bool isFp4>
__global__ AICORE void RunTGEMVMX_SPLIT_K(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
                                          __gm__ ScaleType *src2, __gm__ ScaleType *src3, __gm__ BiasType *src4)
{
    constexpr int blockAlign = isFp4 ? 64 : 32; // need to be 32B aligned

    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int kAlign = CeilAlign<int>(validK, 64);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int KMX = CeilDiv(kAlign, 32);

    constexpr int BASEK = 1024;
    constexpr int BASEKMX = CeilDiv(BASEK, 32);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, BASEK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, BASEK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    // SCALEA in GM, ND
    using GlobalDataSrc2 = GlobalTensor<ScaleType, pto::Shape<1, 1, 1, 1, BASEKMX>, pto::Stride<KMX, KMX, KMX, KMX, 1>>;

    using MxShapeB = TileShape2D<ScaleType, BASEKMX, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<ScaleType, KMX, N, Layout::MX_B_NN>;
    using GlobalDataSrc3 = GlobalTensor<ScaleType, MxShapeB, MxStrideB, Layout::MX_B_NN>;

    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc4 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, validN>,
                                        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    GlobalDataSrc4 src4Global(src4);

    constexpr int blockLeft = isFp4 ? 1024 : 512;
    constexpr int KLeft = CeilAlign<int>(validK, blockLeft);
    using TileMatAData = Tile<TileType::Mat, AType, 1, BASEK, BLayout::RowMajor, 1, BASEK>;

    // scale need 32B Align
    using TileScaleAData =
        Tile<TileType::Mat, ScaleType, 1, BASEKMX, BLayout::RowMajor, 1, BASEKMX, SLayout::RowMajor, 32>;

    using TileMatBData = Tile<TileType::Mat, BType, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::RowMajor, 512>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, BASEKMX, N, BLayout::ColMajor, BASEKMX, validN, SLayout::ColMajor, 32>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<AType, 1, BASEK, 1, BASEK>;
    using RightTile = TileRightCompact<BType, BASEK, N, BASEK, validN>;
    using AccTile = TileAccCompact<OutType, M, N, validM, validN>;
    using LeftScaleTile = TileLeftScaleCompact<ScaleType, 1, BASEKMX, 1, BASEKMX>;
    using RightScaleTile = TileRightScaleCompact<ScaleType, BASEKMX, N, BASEKMX, validN>;
    using BiasTile = Tile<TileType::Bias, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(aScaleMatTile, 0x20000);
    TASSIGN(bScaleMatTile, 0x30000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    AccTile cTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

#ifndef __PTO_AUTO__
    uint64_t scaleAAddr = GetScaleAddr(aTile.data());
    uint64_t scaleBAddr = GetScaleAddr(bTile.data());
    TASSIGN(aScaleTile, scaleAAddr);
    TASSIGN(bScaleTile, scaleBAddr);
#endif

    TASSIGN(biasTile, 0x0);
    TASSIGN(cTile, 0x0);

    constexpr int iter = CeilDiv(kAlign, BASEK);
    for (int i = 0; i < iter; i++) {
        const int offsetA = (!isFp4) ? (i * BASEK) : (i * BASEK / 2);
        const int offsetB = (!isFp4) ? (validN * i * BASEK) : (validN * i * BASEK / 2);
        GlobalDataSrc0 src0Global(src0 + offsetA);
        GlobalDataSrc1 src1Global(src1 + offsetB);

        /*************************************TLOAD****************************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);
        const int offsetAMX = i * BASEKMX;
        const int offsetBMX = 16 * i * BASEKMX;
        GlobalDataSrc2 src2Global(src2 + offsetAMX);
        GlobalDataSrc3 src3Global(src3 + offsetBMX);
        TLOAD<TileScaleAData, GlobalDataSrc2>(aScaleMatTile, src2Global);
        TLOAD<TileScaleBData, GlobalDataSrc3>(bScaleMatTile, src3Global);
        if (i == 0) {
            TLOAD(biasDataTile, src4Global);
        }
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        /**********************************TMOV && TEXTRACT**********************************/
        TEXTRACT(aTile, aMatTile, 0, 0);
        TEXTRACT(bTile, bMatTile, 0, 0);
        TMOV(aScaleTile, aScaleMatTile);
        TMOV(bScaleTile, bScaleMatTile);

#ifdef __PTO_AUTO__
        TGET_SCALE_ADDR(aScaleTile, aTile);
        TGET_SCALE_ADDR(bScaleTile, bTile);
#endif

        if (i == 0) {
            TMOV(biasTile, biasDataTile);
        }
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
        /**********************************TMATMUL**********************************/
        if (i == 0) {
            TGEMV_MX(cTile, aTile, aScaleTile, bTile, bScaleTile, biasTile);
        } else {
            TGEMV_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
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

template <int32_t tilingKey>
void LaunchTMATMUL_MX(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULMX<float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, float, 128, 64, 64, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, float, 127, 72, 64, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float, 128, 110, 63, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e2m1x2_t, float8_e8m0_t, float, 128, 64, 64, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTMATMULMX<float, float4_e1m2x2_t, float4_e2m1x2_t, float8_e8m0_t, float, 117, 64, 60, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 6) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 128, 118, 64, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 7) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 115, 64, 30, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 8) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, float, 16, 32, 16, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 9) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float, 10, 50, 54, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 10) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e2m1x2_t, float8_e8m0_t, float, 4, 30, 8, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 11) {
        RunTGEMVMX<1, float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, 1, 128, 62, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 12) {
        RunTGEMVMX<1, float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, 1, 256, 20, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    }
}

template void LaunchTMATMUL_MX<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMATMUL_MX<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMATMUL_MX<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMATMUL_MX<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);

template <int32_t tilingKey>
void LaunchTMATMUL_MX_BIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4,
                           void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULMX<float, float8_e5m2_t, float8_e4m3_t, float8_e8m0_t, float, 115, 64, 30, true, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 2) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, float, 200, 192, 95, true, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 3) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 35, 128, 56, true, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 4) {
        RunTMATMULMX_SPLIT_K<float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 47, 128, 62, true, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 5) {
        RunTMATMULMX_SPLIT_K<float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float, 64, 192, 64, true, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 6) {
        RunTMATMULMX<float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 1, 64, 62, true, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 7) {
        RunTGEMVMX_SPLIT_K<float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 1, 2048, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    }
}

template void LaunchTMATMUL_MX_BIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                       uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                       uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                       uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                       uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                       uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                       uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                       uint8_t *src4, void *stream);