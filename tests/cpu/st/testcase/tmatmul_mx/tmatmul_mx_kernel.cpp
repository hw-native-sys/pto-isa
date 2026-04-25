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
#include "pto/cpu/MXTypes.hpp"

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

template <typename T, int format, int scaleK, int N>
using GlobalDataSrc3_t = std::conditional_t<
    (format == 1),
    GlobalTensor<T, TileShape2D<T, scaleK, N, Layout::ND>, BaseShape2D<T, scaleK, N, Layout::ND>, Layout::ND>,
    GlobalTensor<T, TileShape2D<T, scaleK, N, Layout::DN>, BaseShape2D<T, scaleK, N, Layout::DN>, Layout::DN>>;

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

    constexpr auto scaleK = CeilDiv(validK, 32);
    using MxShapeA = TileShape2D<ScaleType, validM, scaleK, Layout::ND>;
    using MxStrideA = pto::Stride<validM * scaleK, validM * scaleK, validM * scaleK, scaleK, 1>;
    using GlobalDataSrc2 = GlobalTensor<ScaleType, MxShapeA, MxStrideA, Layout::ND>;

    using MxShapeB = TileShape2D<ScaleType, scaleK, validN, Layout::ND>;
    using MxStrideB = pto::Stride<scaleK * validN, scaleK * validN, scaleK * validN, validN, 1>;
    using GlobalDataSrc3 = GlobalTensor<ScaleType, MxShapeB, MxStrideB, Layout::ND>;

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
        Tile<TileType::Mat, ScaleType, M, kMX, BLayout::RowMajor, validM, scaleK, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, kAlign, N, BLayout::ColMajor, scaleK, validN, SLayout::ColMajor, 32>;

    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<AType, M, kAlign, validM, validK>;
    using RightTile = TileRight<BType, kAlign, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<ScaleType, M, kMX, validM, scaleK>;
    using RightScaleTile = TileRightScale<ScaleType, kAlign, N, scaleK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;
    TileBiasData biasDataTile;
    size_t addr = 0;
    TASSIGN(aMatTile, addr);
    addr += TileMatAData::Numel * sizeof(typename TileMatAData::DType);
    TASSIGN(bMatTile, addr);
    addr += TileMatBData::Numel * sizeof(typename TileMatBData::DType);
    TASSIGN(aScaleMatTile, addr);
    addr += TileScaleAData::Numel * sizeof(typename TileScaleAData::DType);
    TASSIGN(bScaleMatTile, addr);
    addr += TileScaleBData::Numel * sizeof(typename TileScaleBData::DType);
    TASSIGN(biasDataTile, addr);
    addr += TileBiasData::Numel * sizeof(typename TileBiasData::DType);

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TASSIGN(aScaleTile, addr);
    addr += LeftScaleTile::Numel * sizeof(typename LeftScaleTile::DType);
    TASSIGN(bScaleTile, addr);
    addr += RightScaleTile::Numel * sizeof(typename RightScaleTile::DType);
    TASSIGN(biasTile, addr);

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
          int validK, int validN, bool isBias, bool isFp4, bool useGEMV = false>
__global__ AICORE void RunTMATMULMX_SPLIT_K(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
                                            __gm__ ScaleType *src2, __gm__ ScaleType *src3, __gm__ BiasType *src4)
{
    constexpr int blockAlign = isFp4 ? 64 : 32; // need to be 32B aligned

    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int K = CeilAlign<int>(validK, 64);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int KMX = CeilDiv(validK, 32);

    constexpr int BASEK = 64;
    constexpr int BASEKMX = CeilDiv(BASEK, 32);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, BASEK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, BASEK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using MxShapeA = TileShape2D<ScaleType, validM, BASEKMX, Layout::ND>;
    using MxStrideA = pto::Stride<validM * KMX, validM * KMX, validM * KMX, KMX, 1>;
    using GlobalDataSrc2 = GlobalTensor<ScaleType, MxShapeA, MxStrideA, Layout::ND>;

    using MxShapeB = TileShape2D<ScaleType, BASEKMX, validN, Layout::ND>;
    using MxStrideB = pto::Stride<KMX * validN, KMX * validN, KMX * validN, validN, 1>;
    using GlobalDataSrc3 = GlobalTensor<ScaleType, MxShapeB, MxStrideB, Layout::ND>;

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
        Tile<TileType::Mat, ScaleType, M, BASEK, BLayout::RowMajor, validM, BASEKMX, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, BASEK, N, BLayout::ColMajor, BASEKMX, validN, SLayout::ColMajor, 32>;

    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<AType, M, BASEK, validM, BASEK>;
    using RightTile = TileRight<BType, BASEK, N, BASEK, validN>;
    using LeftScaleTile = TileLeftScale<ScaleType, M, BASEK, validM, BASEKMX>;
    using RightScaleTile = TileRightScale<ScaleType, BASEK, N, BASEKMX, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;
    TileBiasData biasDataTile;

    size_t addr = 0;
    TASSIGN(aMatTile, addr);
    addr += TileMatAData::Numel * sizeof(typename TileMatAData::DType);
    TASSIGN(bMatTile, addr);
    addr += TileMatBData::Numel * sizeof(typename TileMatBData::DType);
    TASSIGN(aScaleMatTile, addr);
    addr += TileScaleAData::Numel * sizeof(typename TileScaleAData::DType);
    TASSIGN(bScaleMatTile, addr);
    addr += TileScaleBData::Numel * sizeof(typename TileScaleBData::DType);
    TASSIGN(biasDataTile, addr);
    addr += TileBiasData::Numel * sizeof(typename TileBiasData::DType);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TASSIGN(aScaleTile, addr);
    addr += LeftScaleTile::Numel * sizeof(typename LeftScaleTile::DType);
    TASSIGN(bScaleTile, addr);
    addr += RightScaleTile::Numel * sizeof(typename RightScaleTile::DType);
    TASSIGN(biasTile, addr);

    constexpr int iter = K / BASEK;
    for (int i = 0; i < iter; i++) {
        const int offsetA = (!isFp4) ? (i * BASEK) : (i * BASEK / 2);
        const int offsetB = (!isFp4) ? (validN * i * BASEK) : (validN * i * BASEK / 2);
        GlobalDataSrc0 src0Global(src0 + offsetA);
        GlobalDataSrc1 src1Global(src1 + offsetB);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        const int offsetAMX = i * BASEKMX;
        const int offsetBMX = validN * i * BASEKMX;
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

        if constexpr (useGEMV) {
            if (i == 0) {
                if constexpr (isBias) {
                    TGEMV_MX(cTile, aTile, aScaleTile, bTile, bScaleTile, biasTile);
                } else {
                    TGEMV_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
                }
            } else {
                TGEMV_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
            }

        } else {
            if (i == 0) {
                if constexpr (isBias) {
                    TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile, biasTile);
                } else {
                    TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
                }
            } else {
                TMATMUL_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
            }
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
    constexpr auto scaleK = CeilDiv(validK, 32);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<ScaleType, pto::Shape<1, 1, 1, 1, scaleK>, pto::Stride<scaleK, scaleK, scaleK, scaleK, 1>>;

    using GlobalDataSrc3 = GlobalDataSrc3_t<ScaleType, format, scaleK, validN>;

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

    using TileScaleAData = Tile<TileType::Mat, ScaleType, 1, kMX, BLayout::RowMajor, 1, scaleK, SLayout::RowMajor, 32>;

    using TileMatBData =
        Tile<TileType::Mat, BType, kAlign, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, kAlign, N, BLayout::ColMajor, scaleK, validN, SLayout::ColMajor, 32>;

    using LeftTile = TileLeft<AType, 1, KLeft, 1, validK>;
    using RightTile = TileRightCompact<BType, kAlign, N, kAlign, validN>;
    using AccTile = TileAccCompact<OutType, M, N, validM, validN>;

    using LeftScaleTile = TileLeftScaleCompact<ScaleType, 1, kMX, 1, kMX>;
    using RightScaleTile = TileRightScaleCompact<ScaleType, kAlign, N, kMX, validN>;

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

    TASSIGN(aScaleTile, LeftTile::Numel * sizeof(typename LeftTile::DType));
    TASSIGN(bScaleTile, RightTile::Numel * sizeof(typename RightTile::DType));

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

    /**********************************TGEMV***********************************/

    TGEMV_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL_MX(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULMX<float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, float, 128, 64, 64, false, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
            reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, float, 127, 72, 64, false, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
            reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float, 128, 110, 63, false, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
            reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e2m1x2_t, float8_e8m0_t, float, 128, 64, 64, false, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
            reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTMATMULMX<float, float4_e1m2x2_t, float4_e2m1x2_t, float8_e8m0_t, float, 117, 64, 60, false, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
            reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 6) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 128, 118, 64, false, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
            reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 7) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 115, 64, 30, false, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
            reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 8) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, float, 16, 32, 16, false, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
            reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 9) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float, 10, 50, 54, false, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
            reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 10) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e2m1x2_t, float8_e8m0_t, float, 4, 30, 8, false, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
            reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 11) {
        RunTGEMVMX<1, float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, 1, 128, 62, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
            reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 12) {
        RunTGEMVMX<1, float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, 1, 256, 20, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
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
        RunTMATMULMX<float, float8_e5m2_t, float8_e4m3_t, float8_e8m0_t, float, 115, 64, 30, true, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
            reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 2) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, float, 200, 192, 95, true, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
            reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 3) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 35, 128, 56, true, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
            reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 4) {
        RunTMATMULMX_SPLIT_K<float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 47, 128, 62, true, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
            reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 5) {
        RunTMATMULMX_SPLIT_K<float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float, 64, 192, 64, true, false>(
            reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
            reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 6) {
        RunTMATMULMX<float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 1, 64, 62, true, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
            reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
            reinterpret_cast<float8_e8m0_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 7) {
        RunTMATMULMX_SPLIT_K<float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, float, 1, 2048, 64, true, true>(
            reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
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