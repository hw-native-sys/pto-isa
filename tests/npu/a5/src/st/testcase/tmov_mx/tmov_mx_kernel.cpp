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

template <int format, typename OutType, typename AType, typename BType, typename ScaleType, int validM, int validK,
          int validN, bool isFp4>
__global__ AICORE void RunTMOVMX(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ ScaleType *src2,
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
    using GlobalDataSrc2 = GlobalDataSrc2_t<ScaleType, format, validM, kMX>;
    using GlobalDataSrc3 = GlobalDataSrc3_t<ScaleType, format, kMX, validN>;

    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, AType, M, kAlign, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, BType, kAlign, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileScaleAData =
        Tile<TileType::Mat, ScaleType, M, kMX, BLayout::RowMajor, validM, kMX, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, kMX, N, BLayout::ColMajor, kMX, validN, SLayout::ColMajor, 32>;

    using LeftTile = TileLeftCompact<AType, M, kAlign, validM, kAlign>;
    using RightTile = TileRightCompact<BType, kAlign, N, kAlign, validN>;
    using AccTile = TileAccCompact<OutType, M, N, validM, validN>;

    using LeftScaleTile = TileLeftScaleCompact<ScaleType, M, kMX, validM, kMX>;
    using RightScaleTile = TileRightScaleCompact<ScaleType, kMX, N, kMX, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * kAlign);
    TASSIGN(aScaleMatTile, M * kAlign + kAlign * N);
    TASSIGN(bScaleMatTile, M * kAlign + kAlign * N + M * kMX);

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

    if constexpr ((kAlign - validK) * sizeof(AType) >= C0_SIZE_BYTE) {
        TFILLPAD(aMatTile, aMatTile); // TLOAD can only pad to 32B，mmad_mx needs to be aligned to 64 in k direction
    }
    TFILLPAD(bMatTile, bMatTile); // B input is nk,  TLOAD does not pad zeros in k direction

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

    TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <int format, typename OutType, typename AType, typename BType, typename ScaleType, int validM, int validK,
          int validN, uint16_t indexM, uint16_t indexK, uint16_t indexN, bool isFp4>
__global__ AICORE void RunTEXTRACTMX(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
                                     __gm__ ScaleType *src2, __gm__ ScaleType *src3)
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
    using GlobalDataSrc2 = GlobalDataSrc2_t<ScaleType, format, validM, kMX>;
    using GlobalDataSrc3 = GlobalDataSrc3_t<ScaleType, format, kMX, validN>;

    constexpr int mOut = validM - indexM;
    constexpr int kOut = kAlign - indexK;
    constexpr int nOut = validN - indexN;
    constexpr int kMXOut = kMX - indexK / 32;

    using GlobalDataOut = GlobalTensor<OutType, pto::Shape<1, 1, 1, mOut, nOut>,
                                       pto::Stride<1 * mOut * nOut, 1 * mOut * nOut, mOut * nOut, nOut, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, AType, M, kAlign, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, BType, kAlign, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileScaleAData =
        Tile<TileType::Mat, ScaleType, M, kMX, BLayout::RowMajor, validM, kMX, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, kMX, N, BLayout::ColMajor, kMX, validN, SLayout::ColMajor, 32>;

    using LeftTile = TileLeft<AType, M - indexM, kOut, mOut, kOut>;
    using RightTile = TileRight<BType, kOut, N - indexN, kOut, nOut>;
    using LeftScaleTile = TileLeftScale<ScaleType, M - indexM, kMXOut, mOut, kMXOut>;
    using RightScaleTile = TileRightScale<ScaleType, kMXOut, N - indexN, kMXOut, nOut>;
    using AccTile = TileAcc<OutType, M - indexM, N - indexN, mOut, nOut>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * kAlign);
    TASSIGN(aScaleMatTile, M * kAlign + kAlign * N);
    TASSIGN(bScaleMatTile, M * kAlign + kAlign * N + M * kMX);

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

    if constexpr ((kAlign - validK) * sizeof(AType) >= C0_SIZE_BYTE) {
        TFILLPAD(aMatTile, aMatTile);
    }
    TFILLPAD(bMatTile, bMatTile);

    TLOAD<TileScaleAData, GlobalDataSrc2>(aScaleMatTile, src2Global);
    TLOAD<TileScaleBData, GlobalDataSrc3>(bScaleMatTile, src3Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

    TEXTRACT(aScaleTile, aScaleMatTile, indexM, indexK / 32);
    TEXTRACT(bScaleTile, bScaleMatTile, indexK / 32, indexN);

#ifdef __PTO_AUTO__
    TGET_SCALE_ADDR(aScaleTile, aTile);
    TGET_SCALE_ADDR(bScaleTile, bTile);
#endif

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    /**********************************TMATMUL**********************************/

    TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <int format, typename OutType, typename AType, typename BType, typename ScaleType, int validM, int validK,
          int validN, uint16_t indexM, uint16_t indexK, uint16_t indexN, uint16_t baseM, uint16_t baseK, uint16_t baseN,
          bool isFp4>
__global__ AICORE void RunTEXTRACTMX_COMPACT(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
                                             __gm__ ScaleType *src2, __gm__ ScaleType *src3)
{
    constexpr uint8_t baseKmx = CeilDiv<int>(baseK, 32);
    constexpr int kAlign = CeilAlign<int>(validK, 64);
    constexpr uint8_t kMX = CeilDiv<int>(kAlign, 32);

    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, baseM, baseK>,
                                        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<BType, pto::Shape<1, 1, 1, baseK, baseN>,
                                        pto::Stride<1 * baseK * baseN, 1 * baseK * baseN, baseK * baseN, baseN, 1>>;

    using GlobalDataSrc2 = GlobalDataSrc2_t<ScaleType, format, baseM, baseKmx>;
    using GlobalDataSrc3 = GlobalDataSrc3_t<ScaleType, format, baseKmx, baseN>;

    constexpr int mOut = validM - indexM;
    constexpr int kOut = kAlign - indexK;
    constexpr int nOut = validN - indexN;
    constexpr int kMXOut = kMX - indexK / 32;

    using GlobalDataOut = GlobalTensor<OutType, pto::Shape<1, 1, 1, mOut, nOut>,
                                       pto::Stride<1 * mOut * nOut, 1 * mOut * nOut, mOut * nOut, nOut, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, AType, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, BType, baseK, baseN, BLayout::ColMajor, baseK, baseN, SLayout::RowMajor, 512>;
    using TileScaleAData =
        Tile<TileType::Mat, ScaleType, baseM, baseKmx, BLayout::RowMajor, baseM, baseKmx, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, baseKmx, baseN, BLayout::ColMajor, baseKmx, baseN, SLayout::ColMajor, 32>;

    using LeftTile = TileLeftCompact<AType, baseM, baseK, mOut, kOut>;
    using RightTile = TileRightCompact<BType, baseK, baseN, kOut, nOut>;
    using LeftScaleTile = TileLeftScaleCompact<ScaleType, baseM, baseKmx, mOut, kMXOut>;
    using RightScaleTile = TileRightScaleCompact<ScaleType, baseKmx, baseN, kMXOut, nOut>;
    using AccTile = TileAccCompact<OutType, baseM, baseN, mOut, nOut>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, baseM * baseK);
    TASSIGN(aScaleMatTile, baseM * baseK + baseK * baseN);
    TASSIGN(bScaleMatTile, baseM * baseK + baseK * baseN + baseM * baseKmx);

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

    TLOAD<TileScaleAData, GlobalDataSrc2>(aScaleMatTile, src2Global);
    TLOAD<TileScaleBData, GlobalDataSrc3>(bScaleMatTile, src3Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

    TEXTRACT(aScaleTile, aScaleMatTile, indexM, indexK / 32);
    TEXTRACT(bScaleTile, bScaleMatTile, indexK / 32, indexN);

#ifdef __PTO_AUTO__
    TGET_SCALE_ADDR(aScaleTile, aTile);
    TGET_SCALE_ADDR(bScaleTile, bTile);
#endif

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    /**********************************TMATMUL**********************************/

    TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <int32_t tilingKey, int format>
void LaunchTMOV_MX(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVMX<format, float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, 128, 64, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 2) {
        RunTMOVMX<format, float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, 32, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 3) {
        RunTMOVMX<format, float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, 64, 128, 80, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 4) {
        RunTMOVMX<format, float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, 115, 64, 30, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 5) {
        RunTMOVMX<format, float, float8_e5m2_t, float8_e4m3_t, float8_e8m0_t, 64, 120, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 6) {
        RunTMOVMX<format, float, float4_e2m1x2_t, float4_e2m1x2_t, float8_e8m0_t, 48, 192, 96, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 7) {
        RunTMOVMX<format, float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, 128, 64, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 8) {
        RunTMOVMX<format, float, float4_e2m1x2_t, float4_e2m1x2_t, float8_e8m0_t, 95, 12, 90, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 9) {
        RunTMOVMX<format, float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, 4, 30, 8, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 10) {
        RunTEXTRACTMX<format, float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, 128, 32, 64, 64, 0, 32, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 11) {
        RunTEXTRACTMX<format, float, float4_e2m1x2_t, float4_e2m1x2_t, float8_e8m0_t, 128, 98, 64, 32, 64, 0, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 12) {
        RunTEXTRACTMX<format, float, float4_e1m2x2_t, float4_e1m2x2_t, float8_e8m0_t, 128, 60, 254, 16, 0, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 13) {
        RunTEXTRACTMX<format, float, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, 48, 180, 96, 16, 64, 32, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 14) {
        RunTEXTRACTMX<format, float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, 95, 120, 89, 16, 64, 32, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 15) {
        RunTEXTRACTMX<format, float, float4_e1m2x2_t, float4_e2m1x2_t, float8_e8m0_t, 48, 190, 98, 16, 0, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 16) {
        RunTEXTRACTMX_COMPACT<format, float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, 46, 66, 45, 0, 0, 0, 128, 256,
                              128, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 17) {
        RunTEXTRACTMX_COMPACT<format, float, float8_e5m2_t, float8_e5m2_t, float8_e8m0_t, 68, 130, 80, 16, 64, 32, 128,
                              256, 128, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 18) {
        RunTEXTRACTMX_COMPACT<format, float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, 127, 126, 130, 32, 64, 64,
                              256, 128, 256, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 19) {
        RunTEXTRACTMX_COMPACT<format, float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, 80, 96, 192, 48, 0, 64, 128,
                              256, 256, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 20) {
        RunTEXTRACTMX_COMPACT<format, float, float8_e4m3_t, float8_e4m3_t, float8_e8m0_t, 98, 126, 108, 32, 64, 32, 128,
                              256, 128, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 21) {
        RunTEXTRACTMX_COMPACT<format, float, float4_e1m2x2_t, float4_e2m1x2_t, float8_e8m0_t, 68, 196, 80, 0, 64, 64,
                              128, 256, 128, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                                     reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 22) {
        RunTEXTRACTMX_COMPACT<format, float, float8_e5m2_t, float8_e4m3_t, float8_e8m0_t, 32, 64, 108, 16, 0, 32, 128,
                              256, 128, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 23) {
        RunTEXTRACTMX_COMPACT<format, float, float8_e5m2_t, float8_e4m3_t, float8_e8m0_t, 196, 146, 96, 64, 64, 32, 256,
                              256, 128, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    } else if constexpr (tilingKey == 24) {
        RunTEXTRACTMX_COMPACT<format, float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t, 97, 96, 122, 32, 0, 64,
                              128, 256, 128, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                                     reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<float8_e8m0_t *>(src2),
                                     reinterpret_cast<float8_e8m0_t *>(src3));
    }
}

template void LaunchTMOV_MX<1, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<2, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<3, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<4, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<5, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<6, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<7, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<8, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<9, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                  void *stream);
template void LaunchTMOV_MX<10, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<11, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<12, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<13, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<14, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<15, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<16, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<17, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<18, 0>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<19, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<20, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<21, 1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<22, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<23, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);
template void LaunchTMOV_MX<24, 2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream);