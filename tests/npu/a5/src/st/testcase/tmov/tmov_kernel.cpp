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

template <
    typename cType, typename aType, typename bType, typename biasType, int M, int K, int N, int ValidM, int ValidK,
    int ValidN>
__global__ AICORE void runTMovL12Bias(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1, __gm__ biasType* src2)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<
        aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<
        cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
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
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    Event<Op::TLOAD, Op::TMOV_M2L> evtLoad_Mov;
    Event<Op::TMOV_M2B, Op::TMATMUL> evtTmov_Matmul;
    Event<Op::TMATMUL, Op::TSTORE_ACC> evtMatmul_StoreAcc;

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    evtLoad_Mov = TLOAD(biasMatTile, src2Global);

    /**************************TMOV**************************/
    TMOV(aTile, aMatTile, evtLoad_Mov);
    TMOV(bTile, bMatTile);
    evtTmov_Matmul = TMOV(biasTile, biasMatTile);

    /****************************TMATMUL********************************/
    evtMatmul_StoreAcc = TMATMUL_BIAS(cTile, aTile, bTile, biasTile, evtTmov_Matmul);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile, evtMatmul_StoreAcc);
    out = dstGlobal.data();
}

template <
    typename cType, typename aType, typename bType, typename biasType, int M, int K, int N, int ValidM, int ValidK,
    int ValidN>
__global__ AICORE void runTMovL12BiasDynamic(
    __gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1, __gm__ biasType* src2)
{
    using GlobalDataSrc0 = GlobalTensor<
        aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<
        cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
        pto::Stride<ValidM * ValidN, ValidM * ValidN, ValidM * ValidN, ValidN, 1>>;

    constexpr int alignN = ((N * sizeof(biasType) + 63) / 64) * 64 / sizeof(biasType); // bias aligned to 64 bits

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    using TileMatBiasData = Tile<TileType::Mat, biasType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512>;

    using LeftTile = TileLeft<aType, M, K, -1, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, -1>;
    using AccTile = TileAcc<cType, M, N, ValidM, -1>;

    using BiasTile = Tile<TileType::Bias, cType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512>;

    TileMatAData aMatTile(ValidM, ValidK);
    TileMatBData bMatTile(ValidK, ValidN);
    TileMatBiasData biasMatTile(ValidN);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasMatTile, 0x20000);

    LeftTile aTile(ValidM);
    RightTile bTile(ValidN);
    AccTile cTile(ValidN);
    BiasTile biasTile(ValidN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    Event<Op::TLOAD, Op::TMOV_M2L> evtLoad_Mov;
    Event<Op::TMOV_M2B, Op::TMATMUL> evtTmov_Matmul;
    Event<Op::TMATMUL, Op::TSTORE_ACC> evtMatmul_StoreAcc;

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    evtLoad_Mov = TLOAD(biasMatTile, src2Global);

    /**************************TMOV**************************/
    TMOV(aTile, aMatTile, evtLoad_Mov);
    TMOV(bTile, bMatTile);
    evtTmov_Matmul = TMOV(biasTile, biasMatTile);

    /****************************TMATMUL********************************/
    evtMatmul_StoreAcc = TMATMUL_BIAS(cTile, aTile, bTile, biasTile, evtTmov_Matmul);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile, evtMatmul_StoreAcc);
    out = dstGlobal.data();
}

template <
    typename cType, typename aType, typename bType, typename fbType, typename l0cType, int M, int K, int N, int ValidM,
    int ValidK, int ValidN>
__global__ AICORE void runTMovL12Fb(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1, __gm__ fbType* src2)
{
    using GlobalDataSrc0 = GlobalTensor<
        aType, pto::Shape<1, 1, 1, ValidM, ValidK>,
        pto::Stride<ValidM * ValidK, ValidM * ValidK, ValidM * ValidK, ValidK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        bType, pto::Shape<1, 1, 1, ValidK, ValidN>,
        pto::Stride<ValidK * ValidN, ValidK * ValidN, ValidK * ValidN, ValidN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<fbType, pto::Shape<1, 1, 1, 1, ValidN>, pto::Stride<ValidN, ValidN, ValidN, ValidN, 1>>;
    using GlobalDataOut = GlobalTensor<
        cType, pto::Shape<1, 1, 1, ValidM, ValidN>,
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
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(fbMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    FbTile fbTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(fbTile, 0x0);

    Event<Op::TLOAD, Op::TMOV_M2L> evtLoad_Mov;
    Event<Op::TMOV_M2B, Op::TMATMUL> evtMov_Matmul;
    Event<Op::TMATMUL, Op::TMOV_M2S> evtMatmul_MovM2s;

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    evtLoad_Mov = TLOAD(fbMatTile, src2Global);

    /**************************TMOV & TMATMUL**************************/
    TMOV(aTile, aMatTile, evtLoad_Mov);
    evtMov_Matmul = TMOV(bTile, bMatTile);
    evtMatmul_MovM2s = TMATMUL(cTile, aTile, bTile, evtMov_Matmul);
    TMOV(fbTile, fbMatTile, evtMatmul_MovM2s);

    /********************************TSTORE****************************/
    TSTORE_FP<AccTile, GlobalDataOut, FbTile>(dstGlobal, cTile, fbTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void launchTMovL12Bias(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* bias, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTMovL12Bias<float, half, half, float, 64, 96, 32, 64, 96, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<float*>(bias));
    } else if constexpr (tilingKey == 2) {
        runTMovL12Bias<float, float, float, half, 128, 128, 64, 128, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<half*>(bias));
    } else if constexpr (tilingKey == 3) {
        runTMovL12Bias<float, float, float, bfloat16_t, 64, 80, 32, 64, 80, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<bfloat16_t*>(bias));
    } else if constexpr (tilingKey == 4) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 128, 96, 64, 128, 96, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(bias));
    } else if constexpr (tilingKey == 5) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 32, 32, 64, 31, 32, 63><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(bias));
    } else if constexpr (tilingKey == 6) {
        runTMovL12BiasDynamic<float, half, half, half, 64, 80, 32, 64, 80, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(bias));
    } else if constexpr (tilingKey == 7) {
        runTMovL12BiasDynamic<float, float, float, bfloat16_t, 112, 96, 48, 112, 96, 48><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<bfloat16_t*>(bias));
    } else if constexpr (tilingKey == 8) {
        runTMovL12BiasDynamic<float, float, float, bfloat16_t, 16, 96, 64, 15, 96, 63><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<bfloat16_t*>(bias));
    }
}

template <int32_t tilingKey>
void launchTMovL12Fb(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* scaling, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTMovL12Fb<int8_t, int8_t, int8_t, uint64_t, int32_t, 32, 32, 128, 32, 32, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<uint64_t*>(scaling));
    } else if constexpr (tilingKey == 2) {
        runTMovL12Fb<half, int8_t, int8_t, uint64_t, int32_t, 96, 32, 64, 96, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<uint64_t*>(scaling));
    } else if constexpr (tilingKey == 3) {
        runTMovL12Fb<bfloat16_t, int8_t, int8_t, uint64_t, int32_t, 128, 96, 64, 128, 96, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<uint64_t*>(scaling));
    } else if constexpr (tilingKey == 4) {
        runTMovL12Fb<int8_t, float, float, uint64_t, float, 112, 96, 48, 112, 96, 48><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<uint64_t*>(scaling));
    } else if constexpr (tilingKey == 5) {
        runTMovL12Fb<int8_t, float, float, uint64_t, float, 32, 96, 32, 31, 96, 31><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<uint64_t*>(scaling));
    }
}

template void launchTMovL12Bias<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<6>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<7>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Bias<8>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);

template void launchTMovL12Fb<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Fb<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Fb<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Fb<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
template void launchTMovL12Fb<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);
