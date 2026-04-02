/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

template <typename T, typename U, uint32_t fmapN, uint32_t fmapC1, uint32_t fmapH, uint32_t fmapW, uint32_t fmapC0,
          uint32_t filterC1, uint32_t filterH, uint32_t filterW, uint32_t filterN, uint32_t filterC0, uint32_t outC0,
          uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1, uint8_t padTop = 1,
          uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1>
AICORE inline void runTIMG2COL(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (fmapH + padTop + padBottom - dilationH * (filterH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (fmapW + padLeft + padRight - dilationW * (filterW - 1) - 1) / strideW + 1;
    constexpr uint32_t M = fmapN * heightOut * widthOut;
    constexpr uint32_t SingleCoreM = heightOut * widthOut;
    constexpr uint32_t N = filterN;
    constexpr uint32_t K = filterC1 * filterC0 * filterH * filterW;

    constexpr int gStrideSrc0[5] = {fmapC1 * fmapH * fmapW * fmapC0, fmapH * fmapW * fmapC0, fmapW * fmapC0, fmapC0, 1};
    using ShapeDim5Src0 = pto::Shape<fmapN, fmapC1, fmapH, fmapW, fmapC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {filterH * filterW * filterN * filterC0, filterW * filterN * filterC0,
                                    filterN * filterC0, filterC0, 1};
    using ShapeDim5Src1 = pto::Shape<filterC1, filterH, filterW, filterN, filterC0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc0 src0Global(src0);

    // for auto mode, bufferSize is a misleading variable name in convTile, it shouldn't be number of bytes it should be
    // the number of elements
    constexpr int bufferSizeA = fmapN * fmapC1 * fmapH * fmapW * fmapC0; // * sizeof(U);
    using TileMatAData = ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0,
                                  pto::ConvTileShape<fmapN, fmapC1, fmapH, fmapW, fmapC0>>;
    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);
    static_assert(aMatTile.totalDimCount == 5);

    constexpr int bufferSizeB = filterC1 * filterH * filterW * filterN * filterC0;
    using TileMatBData = ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z,
                                  pto::ConvTileShape<filterC1, filterH, filterW, filterN, filterC0>>;
    TileMatBData bMatTile;
    static_assert(bMatTile.totalDimCount == 5);
    TASSIGN(bMatTile, 0x40000);

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<U, K, N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;
    AccTile cTile;
    LeftTile aTile;
    RightTile bTile;
    TASSIGN(cTile, 0x0);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    uint8_t padList[] = {padLeft, padRight, padTop, padBottom};
    aMatTile.SetFmapH(fmapH);
    aMatTile.SetFmapW(fmapW);
    aMatTile.SetPadListArray(padList);
    aMatTile.SetDilationH(dilationH);
    aMatTile.SetDilationW(dilationW);
    aMatTile.SetFilterH(filterH);
    aMatTile.SetFilterW(filterW);
    aMatTile.SetStrideH(strideH);
    aMatTile.SetStrideW(strideW);
    aMatTile.SetChannelSize(fmapC1 * fmapC0);

    TSETFMATRIX(aMatTile);
    TSET_IMG2COL_PADDING(aMatTile);
    TIMG2COL(aTile, aMatTile, 0, 0);
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
    if constexpr (std::is_same_v<T, int32_t>) {
        using GlobalDataOut =
            GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    } else {
        using GlobalDataOut = GlobalTensor<
            T, pto::Shape<1, filterN / outC0, heightOut, widthOut, outC0>,
            pto::Stride<filterN * heightOut * widthOut, heightOut * widthOut * outC0, widthOut * outC0, outC0, 1>,
            Layout::NC1HWC0>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    }
}

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}
template <typename T, typename U, uint32_t fmapN, uint32_t fmapC1, uint32_t fmapH, uint32_t fmapW, uint32_t fmapC0,
          uint32_t filterC1, uint32_t filterH, uint32_t filterW, uint32_t filterN, uint32_t filterC0, uint32_t outC0,
          uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1, uint8_t padTop = 1,
          uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1>
AICORE inline void runTIMG2COLSplitK(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (fmapH + padTop + padBottom - dilationH * (filterH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (fmapW + padLeft + padRight - dilationW * (filterW - 1) - 1) / strideW + 1;

    constexpr uint32_t validM = fmapN * heightOut * widthOut;
    constexpr uint32_t validN = filterN;
    constexpr uint32_t validK = filterC1 * filterC0 * filterH * filterW;

    constexpr int baseK = 64;
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<uint32_t>(validM, 16);
    constexpr int N = CeilAlign<uint32_t>(validN, blockAlign);
    constexpr int K = CeilAlign<uint32_t>(validK, baseK);

    constexpr int gStrideSrc0[5] = {fmapC1 * fmapH * fmapW * fmapC0, fmapH * fmapW * fmapC0, fmapW * fmapC0, fmapC0, 1};
    using ShapeDim5Src0 = pto::Shape<fmapN, fmapC1, fmapH, fmapW, fmapC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {filterH * filterW * filterN * filterC0, filterW * filterN * filterC0,
                                    filterN * filterC0, filterC0, 1};
    using ShapeDim5Src1 = pto::Shape<filterC1, filterH, filterW, filterN, filterC0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    constexpr int bufferSizeA = fmapN * fmapC1 * fmapH * fmapW * fmapC0;
    using TileMatAData = ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0,
                                  pto::ConvTileShape<fmapN, fmapC1, fmapH, fmapW, fmapC0>>;
    TileMatAData aMatTile;
    static_assert(aMatTile.totalDimCount == 5);
    TASSIGN(aMatTile, 0x0);

    constexpr int bufferSizeB = filterC1 * filterH * filterW * filterN * filterC0;
    using TileMatBData = ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z,
                                  pto::ConvTileShape<filterC1, filterH, filterW, filterN, filterC0>>;
    TileMatBData bMatTile;
    static_assert(bMatTile.totalDimCount == 5);
    TASSIGN(bMatTile, 0x40000);

    using LeftTile = TileLeft<U, 2 * M, 2 * baseK, validM, baseK>; // test compact mode works properly
    using RightTile = TileRight<U, 2 * baseK, 2 * N, baseK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    uint8_t padList[] = {padLeft, padRight, padTop, padBottom};
    aMatTile.SetFmapH(fmapH);
    aMatTile.SetFmapW(fmapW);
    aMatTile.SetPadListArray(padList);
    aMatTile.SetDilationH(dilationH);
    aMatTile.SetDilationW(dilationW);
    aMatTile.SetFilterH(filterH);
    aMatTile.SetFilterW(filterW);
    aMatTile.SetStrideH(strideH);
    aMatTile.SetStrideW(strideW);
    aMatTile.SetChannelSize(fmapC1 * fmapC0);

    TSETFMATRIX<TileMatAData, SetFmatrixMode::FMATRIX_B_MANUAL>(aMatTile);
    TSET_IMG2COL_PADDING(aMatTile);
    constexpr int iter = K / baseK;
    for (int i = 0; i < iter; i++) {
        TIMG2COL<LeftTile, TileMatAData, SetFmatrixMode::FMATRIX_B_MANUAL>(aTile, aMatTile, 0, i * baseK);
        TEXTRACT(bTile, bMatTile, i * baseK, 0);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    if constexpr (std::is_same_v<T, int32_t>) {
        using GlobalDataOut =
            GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                         pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    } else {
        using GlobalDataOut = GlobalTensor<
            T, pto::Shape<1, filterN / outC0, heightOut, widthOut, outC0>,
            pto::Stride<filterN * heightOut * widthOut, heightOut * widthOut * outC0, widthOut * outC0, outC0, 1>,
            Layout::NC1HWC0>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    }
}
template <typename T, typename U, uint32_t fmapN, uint32_t fmapC1, uint32_t fmapH, uint32_t fmapW, uint32_t fmapC0,
          uint32_t filterDim3, uint32_t filterDim2, uint32_t filterDim1, uint32_t filterDim0, uint32_t filterH,
          uint32_t filterW, uint32_t outC0, uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1,
          uint8_t strideW = 1, uint8_t padTop = 1, uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1>
AICORE inline void runTIMG2COLFractalZ4D(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (fmapH + padTop + padBottom - dilationH * (filterH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (fmapW + padLeft + padRight - dilationW * (filterW - 1) - 1) / strideW + 1;

    constexpr uint32_t validM = fmapN * heightOut * widthOut;
    constexpr uint32_t validN = filterDim2 * filterDim1;
    constexpr uint32_t validK = filterDim3 * filterDim0;

    constexpr int baseK = 64;
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<uint32_t>(validM, 16);
    constexpr int N = CeilAlign<uint32_t>(validN, blockAlign);
    constexpr int K = CeilAlign<uint32_t>(validK, baseK);

    constexpr int gStrideSrc0[5] = {fmapC1 * fmapH * fmapW * fmapC0, fmapH * fmapW * fmapC0, fmapW * fmapC0, fmapC0, 1};
    using ShapeDim5Src0 = pto::Shape<fmapN, fmapC1, fmapH, fmapW, fmapC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {filterDim3 * filterDim2 * filterDim1 * filterDim0,
                                    filterDim2 * filterDim1 * filterDim0, filterDim1 * filterDim0, filterDim0, 1};
    using ShapeDim5Src1 = pto::Shape<1, filterDim3, filterDim2, filterDim1, filterDim0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    constexpr int bufferSizeA = sizeof(U) * fmapN * fmapC1 * fmapH * fmapW * fmapC0;
    using TileMatAData = ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0,
                                  pto::ConvTileShape<fmapN, fmapC1, fmapH, fmapW, fmapC0>>;
    TileMatAData aMatTile;
    static_assert(aMatTile.totalDimCount == 5);
    TASSIGN(aMatTile, 0x0);

    constexpr int bufferSizeB = filterDim3 * filterDim2 * filterDim1 * filterDim0;
    using TileMatBData = ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z,
                                  pto::ConvTileShape<filterDim3, filterDim2, filterDim1, filterDim0>>;
    TileMatBData bMatTile;
    static_assert(bMatTile.totalDimCount == 4);
    TASSIGN(bMatTile, 0x40000);

    using LeftTile = TileLeft<U, 2 * M, 2 * baseK, validM, baseK>; // test compact mode works properly
    using RightTile = TileRight<U, 2 * baseK, 2 * N, baseK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    uint8_t padList[] = {padLeft, padRight, padTop, padBottom};
    aMatTile.SetFmapH(fmapH);
    aMatTile.SetFmapW(fmapW);
    aMatTile.SetPadListArray(padList);
    aMatTile.SetDilationH(dilationH);
    aMatTile.SetDilationW(dilationW);
    aMatTile.SetFilterH(filterH);
    aMatTile.SetFilterW(filterW);
    aMatTile.SetStrideH(strideH);
    aMatTile.SetStrideW(strideW);
    aMatTile.SetChannelSize(fmapC1 * fmapC0);

    TSETFMATRIX(aMatTile);
    constexpr int iter = K / baseK;
    for (int i = 0; i < iter; i++) {
        TIMG2COL<LeftTile, TileMatAData, SetFmatrixMode::FMATRIX_B_AUTO>(aTile, aMatTile, 0, i * baseK);
        TEXTRACT(bTile, bMatTile, i * baseK, 0);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    if constexpr (std::is_same_v<T, int32_t>) {
        using GlobalDataOut =
            GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                         pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    } else {
        using GlobalDataOut =
            GlobalTensor<T, pto::Shape<1, filterDim2 * filterDim1 / outC0, heightOut, widthOut, outC0>,
                         pto::Stride<filterDim2 * filterDim1 * heightOut * widthOut, heightOut * widthOut * outC0,
                                     widthOut * outC0, outC0, 1>,
                         Layout::NC1HWC0>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    }
}
extern "C" __global__ AICORE void launchTIMG2COL_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 2;
    constexpr uint32_t fmapH = 4;
    constexpr uint32_t fmapW = 16;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterC1 = 2;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;
    constexpr uint32_t filterN = 16;
    constexpr uint32_t filterC0 = 16;
    constexpr uint32_t outC0 = 16;

    runTIMG2COL<float, bfloat16_t, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN, filterC0,
                outC0>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
                       reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 4;
    constexpr uint32_t fmapW = 16;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterC1 = 4;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;
    constexpr uint32_t filterN = 16;
    constexpr uint32_t filterC0 = 16;
    constexpr uint32_t outC0 = 16;

    runTIMG2COL<float, half, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN, filterC0, outC0,
                2, 1>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
                      reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 8;
    constexpr uint32_t fmapW = 16;
    constexpr uint32_t fmapC0 = 8;

    constexpr uint32_t filterC1 = 4;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;
    constexpr uint32_t filterN = 16;
    constexpr uint32_t filterC0 = 8;
    constexpr uint32_t outC0 = 8;

    runTIMG2COL<float, float, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN, filterC0, outC0,
                1, 1, 2, 2>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
                            reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 1;
    constexpr uint32_t fmapH = 8;
    constexpr uint32_t fmapW = 16;
    constexpr uint32_t fmapC0 = 32;

    constexpr uint32_t filterC1 = 1;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;
    constexpr uint32_t filterN = 16;
    constexpr uint32_t filterC0 = 32;
    constexpr uint32_t outC0 = 16;

    runTIMG2COL<int32_t, int8_t, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN, filterC0,
                outC0>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
                       reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 13;
    constexpr uint32_t fmapW = 57;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterC1 = 4;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;
    constexpr uint32_t filterN = 16;
    constexpr uint32_t filterC0 = 16;
    constexpr uint32_t outC0 = 16;

    runTIMG2COLSplitK<float, bfloat16_t, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN,
                      filterC0, outC0, 2, 2, 2, 2, 1, 2, 1, 2>(reinterpret_cast<__gm__ float *>(out),
                                                               reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                                               reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 25;
    constexpr uint32_t fmapW = 9;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterC1 = 4;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;
    constexpr uint32_t filterN = 16;
    constexpr uint32_t filterC0 = 16;
    constexpr uint32_t outC0 = 16;

    runTIMG2COLSplitK<float, half, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN, filterC0,
                      outC0, 1, 2, 2, 1>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
                                         reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 2;
    constexpr uint32_t fmapH = 14;
    constexpr uint32_t fmapW = 30;
    constexpr uint32_t fmapC0 = 8;

    constexpr uint32_t filterC1 = 2;
    constexpr uint32_t filterH = 4;
    constexpr uint32_t filterW = 4;
    constexpr uint32_t filterN = 16;
    constexpr uint32_t filterC0 = 8;
    constexpr uint32_t outC0 = 8;

    runTIMG2COLSplitK<float, float, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN, filterC0,
                      outC0, 1, 1, 2, 2, 1, 2, 3, 0>(reinterpret_cast<__gm__ float *>(out),
                                                     reinterpret_cast<__gm__ float *>(src0),
                                                     reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 2;
    constexpr uint32_t fmapH = 29;
    constexpr uint32_t fmapW = 60;
    constexpr uint32_t fmapC0 = 32;

    constexpr uint32_t filterC1 = 2;
    constexpr uint32_t filterH = 2;
    constexpr uint32_t filterW = 2;
    constexpr uint32_t filterN = 64;
    constexpr uint32_t filterC0 = 32;
    constexpr uint32_t outC0 = 16;

    runTIMG2COLSplitK<int32_t, int8_t, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterC1, filterH, filterW, filterN,
                      filterC0, outC0, 2, 2, 2, 2, 1, 1, 1, 0>(reinterpret_cast<__gm__ int32_t *>(out),
                                                               reinterpret_cast<__gm__ int8_t *>(src0),
                                                               reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 13;
    constexpr uint32_t fmapW = 57;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterDim3 = 36;
    constexpr uint32_t filterDim2 = 3;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 16;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;

    constexpr uint32_t outC0 = 16;

    runTIMG2COLFractalZ4D<float, bfloat16_t, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterDim3, filterDim2, filterDim1,
                          filterDim0, filterH, filterW, outC0, 2, 2, 2, 2, 1, 2, 1, 2>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_10(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 25;
    constexpr uint32_t fmapW = 9;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterDim3 = 36;
    constexpr uint32_t filterDim2 = 4;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 16;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;

    constexpr uint32_t outC0 = 16;

    runTIMG2COLFractalZ4D<float, half, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterDim3, filterDim2, filterDim1,
                          filterDim0, filterH, filterW, outC0, 1, 2, 2, 1>(reinterpret_cast<__gm__ float *>(out),
                                                                           reinterpret_cast<__gm__ half *>(src0),
                                                                           reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 2;
    constexpr uint32_t fmapH = 14;
    constexpr uint32_t fmapW = 30;
    constexpr uint32_t fmapC0 = 8;

    constexpr uint32_t filterDim3 = 32;
    constexpr uint32_t filterDim2 = 2;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 8;
    constexpr uint32_t filterH = 4;
    constexpr uint32_t filterW = 4;

    constexpr uint32_t outC0 = 8;

    runTIMG2COLFractalZ4D<float, float, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterDim3, filterDim2, filterDim1,
                          filterDim0, filterH, filterW, outC0, 1, 1, 2, 2, 1, 2, 3, 0>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 2;
    constexpr uint32_t fmapH = 29;
    constexpr uint32_t fmapW = 60;
    constexpr uint32_t fmapC0 = 32;

    constexpr uint32_t filterDim3 = 8;
    constexpr uint32_t filterDim2 = 4;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 32;
    constexpr uint32_t filterH = 2;
    constexpr uint32_t filterW = 2;

    constexpr uint32_t outC0 = 16;

    runTIMG2COLFractalZ4D<int32_t, int8_t, fmapN, fmapC1, fmapH, fmapW, fmapC0, filterDim3, filterDim2, filterDim1,
                          filterDim0, filterH, filterW, outC0, 2, 2, 2, 2, 1, 1, 1, 0>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

template <int32_t tilingKey>
void launchTIMG2COL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTIMG2COL_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTIMG2COL_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTIMG2COL_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTIMG2COL_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTIMG2COL_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTIMG2COL_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTIMG2COL_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTIMG2COL_8<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 9) {
        launchTIMG2COL_9<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 10) {
        launchTIMG2COL_10<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTIMG2COL_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTIMG2COL_12<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTIMG2COL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);