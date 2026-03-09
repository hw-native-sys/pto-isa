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

template <typename cType, typename aType, typename bType, typename biasType, int M, int K, int N, int ValidM,
          int ValidK, int ValidN>
__global__ AICORE void runTMovL12Bias(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ biasType *src2)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
                                        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
                                        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
                                       pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;

    constexpr int alignN = ((N * sizeof(biasType) + 63) / 64) * 64 / sizeof(biasType); // bias aligned to 64 bits

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, ValidM, ValidK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;
    using TileMatBiasData =
        Tile<TileType::Mat, biasType, 1, alignN, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox, 512>;

    using LeftTile = TileLeft<aType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<cType, M, N, ValidM, ValidN>;
    using BiasTile = Tile<TileType::Bias, cType, 1, alignN, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox, 512>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatBiasData biasMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * K * sizeof(aType));
    TASSIGN(biasMatTile, M * K * sizeof(aType) + K * N * sizeof(bType));

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
    Event<Op::TLOAD, Op::TMOV_M2L> evtLoad_Mov = TLOAD(biasMatTile, src2Global);

    /**************************TMOV**************************/
    TMOV(aTile, aMatTile, evtLoad_Mov);
    TMOV(bTile, bMatTile);
    Event<Op::TMOV_M2B, Op::TMATMUL> evtMov_Matmul = TMOV(biasTile, biasMatTile);

    /****************************TMATMUL********************************/
    Event<Op::TMATMUL, Op::TSTORE_ACC> evtMatmul_Store = TMATMUL_BIAS(cTile, aTile, bTile, biasTile, evtMov_Matmul);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile, evtMatmul_Store);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename fbType, typename l0cType, int M, int K, int N,
          int ValidM, int ValidK, int ValidN>
__global__ AICORE void runTMovL12Fb(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    using GlobalDataSrc0 = GlobalTensor<aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
                                        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
                                        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<fbType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
                                       pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, ValidM, ValidK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;
    using TileMatFbData = Tile<TileType::Mat, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    using LeftTile = TileLeft<aType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<l0cType, M, N, ValidM, ValidN>;

    using FbTile = Tile<TileType::Scaling, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * K * sizeof(aType));
    TASSIGN(fbMatTile, M * K * sizeof(aType) + K * N * sizeof(bType));

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    FbTile fbTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(fbTile, 0x0);

    Event<Op::TLOAD, Op::TMOV_M2L> evtLoad_Mov2Left = TLOAD(aMatTile, src0Global);
    Event<Op::TLOAD, Op::TMOV_M2R> evtLoad_Mov2Right = TLOAD(bMatTile, src1Global);
    Event<Op::TLOAD, Op::TMOV_M2S> evtLoad_Mov2Scaling = TLOAD(fbMatTile, src2Global);

    /**************************TMOV**************************/
    Event<Op::TMOV_M2L, Op::TMATMUL> evtMov2L_Matmul = TMOV(aTile, aMatTile, evtLoad_Mov2Left);
    Event<Op::TMOV_M2R, Op::TMATMUL> evtMov2R_Matmul = TMOV(bTile, bMatTile, evtLoad_Mov2Right);
    Event<Op::TMOV_M2S, Op::TSTORE_ACC> evtMov2S_Store = TMOV(fbTile, fbMatTile, evtLoad_Mov2Scaling);

    /**************************TMATMUL**************************/
    Event<Op::TMATMUL, Op::TSTORE_ACC> evtMatmul_Store = TMATMUL(cTile, aTile, bTile, evtMov2L_Matmul, evtMov2R_Matmul);

    /********************************TSTORE****************************/
    TSTORE_FP(dstGlobal, cTile, fbTile, evtMov2S_Store, evtMatmul_Store);
}

template <typename cType, typename aType, typename bType, int M, int K, int N, int ValidM, int ValidK, int ValidN,
          int Block = 1>
__global__ AICORE void runTMovAcc2Vec(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
                                        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
                                        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataOutNd = GlobalTensor<cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
                                         pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;
    using GlobalDataOutNz =
        GlobalTensor<cType, pto::Shape<1, ValidM / Block, ValidN / Block, Block, Block>,
                     pto::Stride<ValidM * ValidN, ValidN * Block, Block * Block, Block, 1>, pto::Layout::NZ>;
    using GlobalDataOut = std::conditional_t<Block == 1, GlobalDataOutNd, GlobalDataOutNz>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, ValidM, ValidK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<aType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<cType, M, N, ValidM, ValidN>;

    using VecTileNd = Tile<TileType::Vec, cType, M, N, BLayout::RowMajor, ValidM, ValidN>;
    using VecTileNz = Tile<TileType::Vec, cType, M, N, BLayout::ColMajor, ValidM, ValidN, SLayout::RowMajor>;
    using VecTile = std::conditional_t<Block == 1, VecTileNd, VecTileNz>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * K * sizeof(aType));

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    VecTile dstTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(dstTile, 0x0);

    /******************************TLOAD*****************************/
    Event<Op::TLOAD, Op::TMOV_M2L> evtLoad_MovL = TLOAD(aMatTile, src0Global);
    Event<Op::TLOAD, Op::TMOV_M2R> evtLoad_MovR = TLOAD(bMatTile, src1Global);

    /**************************TMOV**************************/
    Event<Op::TMOV_M2L, Op::TMATMUL> evtMovL_Matmul = TMOV(aTile, aMatTile, evtLoad_MovL);
    Event<Op::TMOV_M2R, Op::TMATMUL> evtMovR_Matmul = TMOV(bTile, bMatTile, evtLoad_MovR);

    /****************************TMATMUL********************************/
    Event<Op::TMATMUL, Op::TMOV_A2V> evtMatmul_Mov = TMATMUL(cTile, aTile, bTile, evtMovL_Matmul, evtMovR_Matmul);
    /****************************TMOV ACC->VEC**************************/
    Event<Op::TMOV_A2V, Op::TSTORE_VEC> evtMov_Store = TMOV(dstTile, cTile, evtMatmul_Mov);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, dstTile, evtMov_Store);
}

template <int32_t tilingKey>
void launchTMovL12Bias(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *bias, void *stream)
{
    if constexpr (tilingKey == 4) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 128, 96, 64, 128, 96, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(bias));
    } else if constexpr (tilingKey == 5) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 32, 32, 64, 31, 32, 63>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(bias));
    }
}

template <int32_t tilingKey>
void launchTMovL12Fb(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *scaling, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTMovL12Fb<int8_t, int8_t, int8_t, uint64_t, int32_t, 32, 32, 128, 32, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(scaling));
    } else if constexpr (tilingKey == 2) {
        runTMovL12Fb<half, int8_t, int8_t, uint64_t, int32_t, 96, 32, 64, 96, 32, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(scaling));
    }
}

template <int32_t tilingKey>
void launchTMovAcc2Vec(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTMovAcc2Vec<half, half, half, 64, 64, 64, 64, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 2) {
        runTMovAcc2Vec<half, half, half, 64, 64, 64, 64, 64, 64, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    }
}

template void launchTMovL12Bias<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template void launchTMovL12Fb<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Fb<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovAcc2Vec<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMovAcc2Vec<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
