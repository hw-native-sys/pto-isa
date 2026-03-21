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
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;
constexpr uint16_t BIAS_ALIGN = 64;
constexpr uint16_t SCALING_ALIGN = 128;
template <typename T>
AICORE inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1>
AICORE inline void L0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src, unsigned GmShape0, unsigned GmShape1,
                              unsigned GmOffset0, unsigned GmOffset1, int uf = 0, uint8_t reluMode = 0)
{
    uint16_t MSize = oriTShape0 < GmShape0 ? oriTShape0 : GmShape0;
    uint16_t NSize = TShape1 < GmShape1 ? TShape1 : GmShape1;
    uint32_t dstStride_dst_D = GmShape1;
    uint16_t srcStride = TShape0;
    uint64_t ndNum = 1;
    uint64_t src_nd_stride = 0;
    uint64_t dst_nd_stride = 0;

    uint8_t UnitFlagMode = uf;
    uint64_t QuantPRE = NoQuant;
    uint8_t ReLUPRE = reluMode;
    bool channelSplit = false;
    bool NZ2ND_EN = true;

    uint64_t config = 0, nd_para = 0;
    nd_para = nd_para | (ndNum & 0xffff);
    nd_para = nd_para | ((src_nd_stride & 0xffff) << 16);
    nd_para = nd_para | ((dst_nd_stride & 0xffff) << 32);
    set_nd_para(nd_para);

    if constexpr (std::is_same<L0CT, float>::value) {
        if constexpr (std::is_same<GMT, int8_t>::value) {
            QuantPRE = QuantMode_t::VQF322B8_PRE;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    } else if constexpr (std::is_same<L0CT, int32_t>::value) {
        if constexpr (std::is_same<GMT, half>::value) {
            QuantPRE = QuantMode_t::VDEQF16;
        } else if constexpr (std::is_same<GMT, int8_t>::value) {
            QuantPRE = QuantMode_t::VREQ8;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    }

    copy_matrix_cc_to_gm((__gm__ GMT *)dst, (__cc__ L0CT *)src, 0, NSize, MSize, dstStride_dst_D, srcStride,
                         UnitFlagMode, QuantPRE, ReLUPRE, channelSplit, NZ2ND_EN);
}

template <typename cType, typename aType, typename bType, typename biasInputType, typename l0cType, int M, int N, int K,
          int isAtranspose, int isBtranspose>
__global__ AICORE void TMOV2BiasKernel(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1,
                                       __gm__ biasInputType *src2)
{
    // The bias addr needs to be 64B aligned.
    constexpr int alignBiasN =
        ((N * sizeof(biasInputType) + BIAS_ALIGN - 1) / BIAS_ALIGN) * BIAS_ALIGN / sizeof(biasInputType);
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataSrc2 = GlobalTensor<biasInputType, Shape<1, 1, 1, 1, alignBiasN>,
                                        Stride<1 * alignBiasN, 1 * alignBiasN, alignBiasN, alignBiasN, 1>>;
    using GlobalDataOut = GlobalTensor<cType, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<TileType::Mat, aType, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose,
                           Tile<TileType::Mat, bType, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>>;
    using TileMatBiasData =
        Tile<TileType::Mat, biasInputType, 1, alignBiasN, BLayout::RowMajor, 1, alignBiasN, SLayout::NoneBox>;

    using LeftTile = Tile<TileType::Left, aType, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, K, N, K, N>;
    using AccTile = TileAcc<l0cType, M, N, M, N>;

    using BiasTile = Tile<TileType::Bias, l0cType, 1, alignBiasN, BLayout::RowMajor, 1, N, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatBiasData biasMatTile;
    uint32_t aMatSize = M * K * sizeof(aType);
    uint32_t bMatSize = K * N * sizeof(bType);
    uint32_t biasMatSize = alignBiasN * sizeof(biasInputType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(biasMatTile, 0x0 + aMatSize + bMatSize);
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(biasMatTile, src2Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMOV(biasTile, biasMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename biasInputType, typename l0cType, int M, int N, int K,
          int isAtranspose, int isBtranspose>
__global__ AICORE void TMOV2BiasDyncmicKernel(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1,
                                              __gm__ biasInputType *src2, int m, int n, int k)
{
    // The bias addr needs to be 64B aligned.
    constexpr int alignN =
        ((N * sizeof(biasInputType) + BIAS_ALIGN - 1) / BIAS_ALIGN) * BIAS_ALIGN / sizeof(biasInputType);
    using DynShapeBiasDim5 = pto::Shape<1, 1, 1, 1, alignN>;
    using DynSTridBiasDim5 = pto::Stride<1 * alignN, 1 * alignN, alignN, alignN, 1>;

    using DynShapeCDim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTridCDim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataSrc2 = GlobalTensor<biasInputType, DynShapeBiasDim5, DynSTridBiasDim5>;
    using GlobalDataOut = GlobalTensor<cType, DynShapeCDim5, DynSTridCDim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<TileType::Mat, aType, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose,
                           Tile<TileType::Mat, bType, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBiasData = Tile<TileType::Mat, biasInputType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    using LeftTile = Tile<TileType::Left, aType, M, K, BLayout::RowMajor, -1, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, K, N, K, -1>;
    using AccTile = TileAcc<l0cType, M, N, M, -1>;
    using BiasTile = Tile<TileType::Bias, l0cType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TileMatBiasData biasMatTile(alignN);

    uint32_t aMatSize = M * K * sizeof(aType);
    uint32_t bMatSize = K * N * sizeof(bType);
    uint32_t biasMatSize = alignN * sizeof(biasInputType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(biasMatTile, 0x0 + aMatSize + bMatSize);

    LeftTile aTile(m);
    RightTile bTile(n);
    AccTile cTile(n);
    BiasTile biasTile(alignN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(biasMatTile, src2Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMOV(biasTile, biasMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename scalingType, typename l0cType, int M, int N, int K,
          int isAtranspose, int isBtranspose, uint8_t reluMode>
__global__ AICORE void TMOV2ScalingKernel(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1,
                                          __gm__ scalingType *src2)
{
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataSrc2 = GlobalTensor<scalingType, Shape<1, 1, 1, 1, N>, Stride<1 * N, 1 * N, N, N, 1>>;
    using GlobalDataOut = GlobalTensor<cType, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<TileType::Mat, aType, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose,
                           Tile<TileType::Mat, bType, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>>;
    using TileMatFbData = Tile<TileType::Mat, scalingType, 1, N, BLayout::RowMajor, 1, N, SLayout::NoneBox>;

    using LeftTile = Tile<TileType::Left, aType, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, K, N, K, N>;
    using AccTile = TileAcc<l0cType, M, N, M, N>;
    using FbTile = Tile<TileType::Scaling, scalingType, 1, N, BLayout::RowMajor, 1, N, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;

    uint32_t aMatSize = M * K * sizeof(aType);
    uint32_t bMatSize = K * N * sizeof(bType);
    uint32_t fbMatSize = N * sizeof(scalingType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(fbMatTile, 0x0 + aMatSize + bMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    FbTile fbTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(fbTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(fbMatTile, src2Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
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
    TMOV(fbTile, fbMatTile);

    TSTORE_FP<AccTile, GlobalDataOut, FbTile>(dstGlobal, cTile, fbTile);

    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename biasInputType, typename scalingType,
          typename l0cType, int M, int N, int K, int isAtranspose, int isBtranspose>
__global__ AICORE void TMOV2BiasAndScalingDyncmicKernel(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1,
                                                        __gm__ biasInputType *src2, __gm__ scalingType *src3, int m,
                                                        int n, int k)
{
    // The bias addr needs to be 64B aligned.
    constexpr int alignBiasN =
        ((N * sizeof(biasInputType) + BIAS_ALIGN - 1) / BIAS_ALIGN) * BIAS_ALIGN / sizeof(biasInputType);
    // The Scaling addr needs to be 128B aligned.
    constexpr int alignScalingN =
        ((N * sizeof(scalingType) + SCALING_ALIGN - 1) / SCALING_ALIGN) * SCALING_ALIGN / sizeof(scalingType);

    using DynShapeBiasDim5 = pto::Shape<1, 1, 1, 1, alignBiasN>;
    using DynSTridBiasDim5 = pto::Stride<1 * alignBiasN, 1 * alignBiasN, alignBiasN, alignBiasN, 1>;

    using DynShapeScalingDim5 = pto::Shape<1, 1, 1, 1, alignScalingN>;
    using DynSTridScalingDim5 = pto::Stride<1 * alignScalingN, 1 * alignScalingN, alignScalingN, alignScalingN, 1>;

    using DynShapeCDim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTridCDim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataSrc2 = GlobalTensor<biasInputType, DynShapeBiasDim5, DynSTridBiasDim5>;
    using GlobalDataSrc3 = GlobalTensor<scalingType, DynShapeScalingDim5, DynSTridScalingDim5>;
    using GlobalDataOut = GlobalTensor<cType, DynShapeCDim5, DynSTridCDim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataOut dstGlobal(out);

    constexpr uint16_t blockCubeK = BLOCK_ALIGN_BYTE / sizeof(aType);
    // A non-transposed input [M, K], with M aligned to 16 and K aligned to 32B;
    // B transposed input [N, K], with N aligned to 16 and K aligned to 32B.
    constexpr uint16_t alignM = (M + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr uint16_t alignN = (N + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr uint16_t alignK = (K + blockCubeK - 1) / blockCubeK * blockCubeK;

    using TileMatAData = std::conditional_t<
        isAtranspose, Tile<TileType::Mat, aType, alignM, alignK, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, aType, alignM, alignK, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<
        isBtranspose, Tile<TileType::Mat, bType, alignK, alignN, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, bType, alignK, alignN, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBiasData =
        Tile<TileType::Mat, biasInputType, 1, alignBiasN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;
    using TileMatScalingData =
        Tile<TileType::Mat, scalingType, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    using LeftTile = Tile<TileType::Left, aType, alignM, alignK, BLayout::RowMajor, -1, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, alignK, alignN, K, -1>;
    using AccTile = TileAcc<l0cType, alignM, alignN, M, -1>;

    using BiasTile = Tile<TileType::Bias, l0cType, 1, alignBiasN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;
    using ScalingTile =
        Tile<TileType::Scaling, scalingType, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    TileMatAData aMatTile(alignM, alignK);
    TileMatBData bMatTile(alignK, alignN);
    TileMatBiasData biasMatTile(alignBiasN);
    TileMatScalingData scalingMatTile(alignScalingN);

    uint32_t aMatSize = alignM * alignK * sizeof(aType);
    uint32_t bMatSize = alignK * alignN * sizeof(bType);
    uint32_t biasMatSize = alignBiasN * sizeof(biasInputType);
    uint32_t fbMatSize = alignScalingN * sizeof(scalingType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(biasMatTile, 0x0 + aMatSize + bMatSize);
    TASSIGN(scalingMatTile, 0x0 + aMatSize + bMatSize + biasMatSize);

    LeftTile aTile(M);
    RightTile bTile(N);
    AccTile cTile(N);
    BiasTile biasTile(N);
    ScalingTile scalingTile(N);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);
    TASSIGN(scalingTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(biasMatTile, src2Global);
    TLOAD(scalingMatTile, src3Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMOV(biasTile, biasMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL_BIAS(cTile, aTile, bTile, biasTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TMOV(scalingTile, scalingMatTile);

    TSTORE_FP<AccTile, GlobalDataOut, ScalingTile>(dstGlobal, cTile, scalingTile);

    out = dstGlobal.data();
}

template <typename AT, typename BT, typename L0CT, typename BiasT, typename GMT, typename ScalingT, int M, int N, int K,
          int isAtranspose, int isBtranspose, int IsBias, int IsQuant, int ReluMode, int Isdynamic, int IsNd = 1>
void LaunchTMOV(GMT *out, AT *src0, BT *src1, BiasT *src2, ScalingT *src3, void *stream)
{
    if constexpr (!Isdynamic) {
        if constexpr (IsBias) {
            if constexpr (std::is_same_v<AT, uint16_t> && std::is_same_v<BiasT, uint16_t>) {
                TMOV2BiasKernel<GMT, half, half, half, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1),
                                             reinterpret_cast<half *>(src2));
            } else if constexpr (std::is_same_v<BiasT, uint16_t>) {
                TMOV2BiasKernel<GMT, AT, BT, half, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, reinterpret_cast<half *>(src2));
            } else if constexpr (std::is_same_v<AT, uint16_t>) {
                TMOV2BiasKernel<GMT, half, half, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), src2);
            } else {
                TMOV2BiasKernel<GMT, AT, BT, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, src2);
            }
        } else if constexpr (IsQuant) {
            if constexpr (std::is_same_v<AT, uint16_t>) {
                TMOV2ScalingKernel<GMT, half, half, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose, ReluMode>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), src3);
            } else if constexpr (std::is_same_v<GMT, uint16_t>) {
                TMOV2ScalingKernel<half, AT, BT, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose, ReluMode>
                    <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), src0, src1, src3);
            } else {
                TMOV2ScalingKernel<GMT, AT, BT, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose, ReluMode>
                    <<<1, nullptr, stream>>>(out, src0, src1, src3);
            }
        }
    } else {
        if constexpr (IsBias && !IsQuant) {
            if constexpr (std::is_same_v<AT, uint16_t> && std::is_same_v<BiasT, uint16_t>) {
                TMOV2BiasDyncmicKernel<GMT, half, half, half, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1),
                                             reinterpret_cast<half *>(src2), M, N, K);
            } else if constexpr (std::is_same_v<BiasT, uint16_t>) {
                TMOV2BiasDyncmicKernel<GMT, AT, BT, half, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, reinterpret_cast<half *>(src2), M, N, K);
            } else if constexpr (std::is_same_v<AT, uint16_t>) {
                TMOV2BiasDyncmicKernel<GMT, half, half, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), src2,
                                             M, N, K);
            } else {
                TMOV2BiasDyncmicKernel<GMT, AT, BT, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, src2, M, N, K);
            }
        } else if constexpr (IsBias && IsQuant) {
            TMOV2BiasAndScalingDyncmicKernel<GMT, AT, BT, BiasT, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose>
                <<<1, nullptr, stream>>>(out, src0, src1, src2, src3, M, N, K);
        }
    }
}

// atype, btype, l0ctype, biastype, gmtype, scalingtype, M, N, K, is_atrans, is_btrans, is_bias, is_quant, relu_mode,
// isdynamic
template void LaunchTMOV<uint16_t, uint16_t, float, float, float, uint64_t, 64, 32, 80, 0, 1, 1, 0, 0, 0>(
    float *out, uint16_t *src0, uint16_t *src1, float *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int32_t, uint64_t, 128, 64, 128, 0, 1, 1, 0, 0, 0>(
    int32_t *out, int8_t *src0, int8_t *src1, int32_t *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<float, float, float, float, float, uint64_t, 128, 48, 64, 0, 1, 1, 0, 0, 0>(
    float *out, float *src0, float *src1, float *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<uint16_t, uint16_t, float, uint16_t, float, uint64_t, 64, 32, 80, 0, 1, 1, 0, 0, 1>(
    float *out, uint16_t *src0, uint16_t *src1, uint16_t *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<float, float, float, uint16_t, float, uint64_t, 112, 48, 96, 0, 1, 1, 0, 0, 1>(
    float *out, float *src0, float *src1, uint16_t *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<float, float, float, uint16_t, float, uint64_t, 64, 128, 96, 0, 1, 1, 0, 0, 0>(
    float *out, float *src0, float *src1, uint16_t *src2, uint64_t *src3, void *stream);

template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 128, 112, 32, 0, 1, 0, 1, 0, 0>(
    int8_t *out, int8_t *src0, int8_t *src1, int32_t *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, uint16_t, uint64_t, 144, 80, 160, 0, 1, 0, 1, 0, 0>(
    uint16_t *out, int8_t *src0, int8_t *src1, int32_t *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<uint16_t, uint16_t, float, float, int8_t, uint64_t, 64, 32, 80, 0, 1, 0, 1, 0, 0>(
    int8_t *out, uint16_t *src0, uint16_t *src1, float *src2, uint64_t *src3, void *stream);

template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 60, 17, 80, 0, 1, 1, 1, 0, 1, 1>(
    int8_t *out, int8_t *src0, int8_t *src1, int32_t *src2, uint64_t *src3, void *stream);
template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 15, 10, 30, 0, 1, 1, 1, 0, 1, 1>(
    int8_t *out, int8_t *src0, int8_t *src1, int32_t *src2, uint64_t *src3, void *stream);