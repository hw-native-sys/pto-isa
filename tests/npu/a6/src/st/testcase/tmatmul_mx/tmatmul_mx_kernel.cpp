/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr uint32_t MX_SCALE_GROUP = 32;
constexpr uint32_t HIF4_SCALE_GROUP = 64;
constexpr uint64_t L0C_SIZE_BYTES = 256u * 1024u;
constexpr uint64_t L0A_BUF0 = 0x0u;
constexpr uint64_t L0A_BUF1 = 0x800u;
constexpr uint64_t L0B_BUF0 = 0x0u;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    return (num1 + num2 - 1) / num2 * num2;
}

template <
    typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB, typename LeftTile,
    typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename AccTile, typename GlobalDataA,
    typename GlobalDataB, typename GlobalScaleA, typename GlobalScaleB>
AICORE inline void MxSetupLoadExtractA(
    TileMatA& aMatTile, TileMatB& bMatTile, TileScaleA& aScaleTile, TileScaleB& bScaleTile, LeftTile& al0,
    RightTile& bl0, LeftScaleTile& aScaleL0, RightScaleTile& bScaleL0, AccTile& cTile, GlobalDataA& aDataGm,
    GlobalDataB& bDataGm, GlobalScaleA& aScaleGm, GlobalScaleB& bScaleGm)
{
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);
    TASSIGN(al0, L0A_BUF1);
    TASSIGN(bl0, L0B_BUF0);
    TASSIGN(aScaleL0, GetScaleAddr(al0.data()));
    TASSIGN(bScaleL0, GetScaleAddr(bl0.data()));
    TASSIGN(cTile, 0x0u);

    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(al0, aMatTile, 0, 0);
    TEXTRACT(aScaleL0, aScaleTile, 0, 0);
}

template <typename AccTile, typename LeftTile, typename LeftScaleTile, typename RightTile, typename RightScaleTile>
AICORE inline void MxComputeStore(
    AccTile& cTile, LeftTile& al0, LeftScaleTile& aScaleL0, RightTile& bl0, RightScaleTile& bScaleL0)
{
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL_MX(cTile, al0, aScaleL0, bl0, bScaleL0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
}

template <
    typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB, typename LeftTile,
    typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename AccTile, typename GlobalDataA,
    typename GlobalDataB, typename GlobalScaleA, typename GlobalScaleB, typename GlobalDataOut>
AICORE inline void MxRunMatmul(
    GlobalDataOut& outGm, GlobalDataA& aDataGm, GlobalDataB& bDataGm, GlobalScaleA& aScaleGm, GlobalScaleB& bScaleGm)
{
    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    LeftTile al0;
    RightTile bl0;
    LeftScaleTile aScaleL0;
    RightScaleTile bScaleL0;
    AccTile cTile;

    MxSetupLoadExtractA(
        aMatTile, bMatTile, aScaleTile, bScaleTile, al0, bl0, aScaleL0, bScaleL0, cTile, aDataGm, bDataGm, aScaleGm,
        bScaleGm);
    TEXTRACT(bl0, bMatTile, 0, 0);
    TEXTRACT(bScaleL0, bScaleTile, 0, 0);
    MxComputeStore(cTile, al0, aScaleL0, bl0, bScaleL0);
    TSTORE(outGm, cTile);
}

template <
    typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB, typename LeftTile,
    typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename AccTile, typename GlobalDataA,
    typename GlobalDataB, typename GlobalScaleA, typename GlobalScaleB, int validM, int validN, int tileN, int nTiles>
AICORE inline void MxRunTiledMatmul(
    __gm__ bfloat16_t* out, GlobalDataA& aDataGm, GlobalDataB& bDataGm, GlobalScaleA& aScaleGm, GlobalScaleB& bScaleGm)
{
    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    LeftTile al0;
    LeftScaleTile aScaleL0;
    RightTile bl0;
    RightScaleTile bScaleL0;
    AccTile cTile;

    MxSetupLoadExtractA(
        aMatTile, bMatTile, aScaleTile, bScaleTile, al0, bl0, aScaleL0, bScaleL0, cTile, aDataGm, bDataGm, aScaleGm,
        bScaleGm);

    using GlobalDataOut = GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, validM, pto::DYNAMIC>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    for (int j = 0; j < nTiles; ++j) {
        TEXTRACT(bl0, bMatTile, 0, j * tileN);
        TEXTRACT(bScaleL0, bScaleTile, 0, j * tileN);
        MxComputeStore(cTile, al0, aScaleL0, bl0, bScaleL0);
        GlobalDataOut outGm(out + j * tileN);
        outGm.template SetShape<GlobalTensorDim::DIM_4>(tileN);
        TSTORE(outGm, cTile);
#ifndef __PTO_AUTO__
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
#endif
    }
}

template <typename LeftT, typename RightT, int validM, int validK, int validN>
AICORE inline void RunMxE2m1Impl(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ RightT* bData, __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleK = validK / MX_SCALE_GROUP;

    using TileMatA = Tile<
        TileType::Mat, LeftT, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, RightT, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleK, BLayout::RowMajor, validM, scaleK, SLayout::RowMajor, 32>;
    using TileScaleB =
        Tile<TileType::Mat, uint8_t, scaleK, N, BLayout::ColMajor, scaleK, validN, SLayout::ColMajor, 32>;
    using GlobalDataA = GlobalTensor<
        LeftT, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        RightT, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::MX_B_NN>;
    using GlobalDataOut = GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;
    using LeftTile = TileLeft<LeftT, M, K, validM, validK>;
    using RightTile = TileRight<RightT, K, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleK, validM, scaleK>;
    using RightScaleTile = TileRightScale<uint8_t, scaleK, N, scaleK, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);
    GlobalDataOut outGm(out);
    MxRunMatmul<
        TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile, RightScaleTile, AccTile,
        GlobalDataA, GlobalDataB, GlobalScaleA, GlobalScaleB, GlobalDataOut>(
        outGm, aDataGm, bDataGm, aScaleGm, bScaleGm);
}

template <typename LeftT, int validM, int validK, int validN>
AICORE inline void RunMxHif4BImpl(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleKLeft = validK / MX_SCALE_GROUP;
    constexpr int scaleKRight = validK / HIF4_SCALE_GROUP;
    constexpr int scaleKColsRight = scaleKRight * HIF4_COL_LEN;
    using TileMatA = Tile<
        TileType::Mat, LeftT, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, hifloat4x2_t, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleKLeft, BLayout::RowMajor, validM, scaleKLeft, SLayout::RowMajor, 32>;
    using TileScaleB = Tile<
        TileType::Mat, uint8_t, scaleKColsRight, N, BLayout::ColMajor, scaleKColsRight, validN, SLayout::ColMajor, 32>;
    using GlobalDataA = GlobalTensor<
        LeftT, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        hifloat4x2_t, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleKLeft, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleKLeft, Layout::MX_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleKRight, N, Layout::HIF4_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleKRight, N, Layout::HIF4_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::HIF4_B_NN>;
    using GlobalDataOut = GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;
    using LeftTile = TileLeft<LeftT, M, K, validM, validK>;
    using RightTile = TileRight<hifloat4x2_t, K, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleKLeft, validM, scaleKLeft>;
    using RightScaleTile = TileRightScale<uint8_t, scaleKColsRight, N, scaleKColsRight, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;
    GlobalDataOut outGm(out);
    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleB bScaleGm(bScale);
    GlobalScaleA aScaleGm(aScale);
    MxRunMatmul<
        TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile, RightScaleTile, AccTile,
        GlobalDataA, GlobalDataB, GlobalScaleA, GlobalScaleB, GlobalDataOut>(
        outGm, aDataGm, bDataGm, aScaleGm, bScaleGm);
}

enum class Hif4ScaleGmMode {
    ZZ_NN,
    ND_DN,
};

template <int validM, int validK, int validN, Hif4ScaleGmMode ScaleGmMode = Hif4ScaleGmMode::ZZ_NN>
AICORE inline void RunMxHif4ABImpl(
    __gm__ bfloat16_t* out, __gm__ hifloat4x2_t* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleK = validK / HIF4_SCALE_GROUP;
    constexpr int scaleKCols = scaleK * HIF4_COL_LEN;
    constexpr int tileNRaw = static_cast<int>(L0C_SIZE_BYTES) / (M * 4);
    constexpr int tileNCapped = (tileNRaw < N) ? tileNRaw : N;
    constexpr int tileN = CeilAlign<int>(tileNCapped, 64);
    constexpr int nTiles = (N + tileN - 1) / tileN;
    static_assert(M * tileN * 4 <= static_cast<int>(L0C_SIZE_BYTES), "tiled accumulator exceeds L0C");
    using TileMatA = Tile<
        TileType::Mat, hifloat4x2_t, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, hifloat4x2_t, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleKCols, BLayout::RowMajor, validM, scaleKCols, SLayout::RowMajor, 32>;
    using TileScaleB =
        Tile<TileType::Mat, uint8_t, scaleKCols, N, BLayout::ColMajor, scaleKCols, validN, SLayout::ColMajor, 32>;
    using GlobalDataA = GlobalTensor<
        hifloat4x2_t, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        hifloat4x2_t, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::HIF4_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::HIF4_A_ZZ>;
    using FractalGlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::HIF4_A_ZZ>;
    using NdGlobalScaleA = GlobalTensor<
        hif4_scale, pto::Shape<1, 1, 1, M, scaleK>, pto::Stride<M * scaleK, M * scaleK, M * scaleK, scaleK, 1>,
        Layout::HIF4_SCALE_ND>;
    using GlobalScaleA = std::conditional_t<ScaleGmMode == Hif4ScaleGmMode::ND_DN, NdGlobalScaleA, FractalGlobalScaleA>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::HIF4_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::HIF4_B_NN>;
    using FractalGlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::HIF4_B_NN>;
    using DnGlobalScaleB = GlobalTensor<
        hif4_scale, pto::Shape<1, 1, 1, scaleK, N>, pto::Stride<scaleK * N, scaleK * N, scaleK * N, 1, scaleK>,
        Layout::HIF4_SCALE_DN>;
    using GlobalScaleB = std::conditional_t<ScaleGmMode == Hif4ScaleGmMode::ND_DN, DnGlobalScaleB, FractalGlobalScaleB>;
    using LeftTile = TileLeft<hifloat4x2_t, M, K, validM, validK>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleKCols, validM, scaleKCols>;
    using RightTile = TileRight<hifloat4x2_t, K, tileN, validK, tileN>;
    using RightScaleTile = TileRightScale<uint8_t, scaleKCols, tileN, scaleKCols, tileN>;
    using AccTile = TileAcc<float, M, tileN, validM, tileN>;
    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(reinterpret_cast<typename GlobalScaleA::DType*>(aScale));
    GlobalScaleB bScaleGm(reinterpret_cast<typename GlobalScaleB::DType*>(bScale));
    MxRunTiledMatmul<
        TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile, RightScaleTile, AccTile,
        GlobalDataA, GlobalDataB, GlobalScaleA, GlobalScaleB, validM, validN, tileN, nTiles>(
        out, aDataGm, bDataGm, aScaleGm, bScaleGm);
}

template <typename LeftT, typename RightT, int validM, int validK, int validN>
__global__ AICORE void RunMxE2m1Matmul(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ RightT* bData, __gm__ uint8_t* bScale)
{
    RunMxE2m1Impl<LeftT, RightT, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

template <typename LeftT, int validM, int validK, int validN>
__global__ AICORE void RunMxHif4BMatmul(
    __gm__ bfloat16_t* out, __gm__ LeftT* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    RunMxHif4BImpl<LeftT, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

template <int validM, int validK, int validN>
__global__ AICORE void RunMxHif4ABMatmul(
    __gm__ bfloat16_t* out, __gm__ hifloat4x2_t* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    RunMxHif4ABImpl<validM, validK, validN, Hif4ScaleGmMode::ZZ_NN>(out, aData, aScale, bData, bScale);
}

template <int validM, int validK, int validN>
__global__ AICORE void RunMxHif4ABMatmulWithND(
    __gm__ bfloat16_t* out, __gm__ hifloat4x2_t* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    RunMxHif4ABImpl<validM, validK, validN, Hif4ScaleGmMode::ND_DN>(out, aData, aScale, bData, bScale);
}

namespace TmatmulMxA6 {
template <int caseId>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxA6

#define LAUNCH_E2M1(ID, A_T, B_T, M, K, N)                                                                         \
    template <>                                                                                                    \
    void TmatmulMxA6::Launch<ID>(                                                                                  \
        uint8_t * out, uint8_t * aData, uint8_t * aScale, uint8_t * bData, uint8_t * bScale, void* stream)         \
    {                                                                                                              \
        RunMxE2m1Matmul<A_T, B_T, M, K, N><<<1, nullptr, stream>>>(                                                \
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<A_T*>(aData), reinterpret_cast<uint8_t*>(aScale), \
            reinterpret_cast<B_T*>(bData), reinterpret_cast<uint8_t*>(bScale));                                    \
    }

#define LAUNCH_HIF4B(ID, A_T, M, K, N)                                                                             \
    template <>                                                                                                    \
    void TmatmulMxA6::Launch<ID>(                                                                                  \
        uint8_t * out, uint8_t * aData, uint8_t * aScale, uint8_t * bData, uint8_t * bScale, void* stream)         \
    {                                                                                                              \
        RunMxHif4BMatmul<A_T, M, K, N><<<1, nullptr, stream>>>(                                                    \
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<A_T*>(aData), reinterpret_cast<uint8_t*>(aScale), \
            reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));                           \
    }

#define LAUNCH_HIF4AB(ID, M, K, N)                                                                         \
    template <>                                                                                            \
    void TmatmulMxA6::Launch<ID>(                                                                          \
        uint8_t * out, uint8_t * aData, uint8_t * aScale, uint8_t * bData, uint8_t * bScale, void* stream) \
    {                                                                                                      \
        RunMxHif4ABMatmul<M, K, N><<<1, nullptr, stream>>>(                                                \
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<hifloat4x2_t*>(aData),                    \
            reinterpret_cast<uint8_t*>(aScale), reinterpret_cast<hifloat4x2_t*>(bData),                    \
            reinterpret_cast<uint8_t*>(bScale));                                                           \
    }

#define LAUNCH_HIF4AB_WITH_ND(ID, M, K, N)                                                                 \
    template <>                                                                                            \
    void TmatmulMxA6::Launch<ID>(                                                                          \
        uint8_t * out, uint8_t * aData, uint8_t * aScale, uint8_t * bData, uint8_t * bScale, void* stream) \
    {                                                                                                      \
        RunMxHif4ABMatmulWithND<M, K, N><<<1, nullptr, stream>>>(                                          \
            reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<hifloat4x2_t*>(aData),                    \
            reinterpret_cast<uint8_t*>(aScale), reinterpret_cast<hifloat4x2_t*>(bData),                    \
            reinterpret_cast<uint8_t*>(bScale));                                                           \
    }

LAUNCH_E2M1(1, float4_e1m2x2_t, float4_e1m2x2_t, 128, 128, 128)
LAUNCH_E2M1(2, float4_e1m2x2_t, float4_e1m2x2_t, 64, 64, 64)
LAUNCH_E2M1(3, float4_e2m1x2_t, float4_e2m1x2_t, 128, 128, 128)
LAUNCH_E2M1(4, float4_e1m2x2_t, float4_e2m1x2_t, 128, 128, 128)
LAUNCH_E2M1(5, float4_e2m1x2_t, float4_e1m2x2_t, 128, 128, 128)
LAUNCH_E2M1(6, float8_e4m3_t, float4_e2m1x2_t, 128, 128, 128)
LAUNCH_E2M1(7, float8_e4m3_t, float4_e2m1x2_t, 64, 128, 64)
LAUNCH_E2M1(8, half, float4_e2m1x2_t, 128, 128, 128)
LAUNCH_E2M1(9, half, float4_e2m1x2_t, 64, 128, 64)
LAUNCH_E2M1(10, bfloat16_t, float4_e2m1x2_t, 128, 128, 128)
LAUNCH_E2M1(11, bfloat16_t, float4_e2m1x2_t, 64, 128, 64)
LAUNCH_HIF4B(12, float8_e4m3_t, 128, 128, 128)
LAUNCH_HIF4B(13, float8_e4m3_t, 64, 128, 64)
LAUNCH_HIF4B(14, half, 128, 128, 128)
LAUNCH_HIF4B(15, half, 64, 128, 64)
LAUNCH_HIF4B(16, bfloat16_t, 128, 128, 128)
LAUNCH_HIF4B(17, bfloat16_t, 64, 128, 64)
LAUNCH_HIF4AB(18, 128, 128, 128)
LAUNCH_HIF4AB(19, 128, 256, 128)
LAUNCH_HIF4AB(20, 256, 128, 128)
LAUNCH_HIF4AB(21, 64, 64, 64)
LAUNCH_HIF4AB(22, 256, 256, 256)
LAUNCH_HIF4AB(23, 128, 512, 128)
LAUNCH_HIF4AB(24, 512, 128, 512)
LAUNCH_HIF4AB(25, 128, 128, 256)
LAUNCH_HIF4AB(26, 256, 128, 512)
LAUNCH_E2M1(27, float8_e4m3_t, float4_e2m1x2_t, 1, 256, 64)
LAUNCH_E2M1(28, half, float4_e2m1x2_t, 1, 256, 64)
LAUNCH_HIF4B(29, bfloat16_t, 1, 256, 64)
LAUNCH_E2M1(30, float4_e2m1x2_t, float4_e2m1x2_t, 64, 128, 64)
LAUNCH_E2M1(31, float4_e1m2x2_t, float4_e2m1x2_t, 64, 64, 64)
LAUNCH_E2M1(32, float4_e2m1x2_t, float4_e1m2x2_t, 64, 64, 64)
LAUNCH_E2M1(33, float8_e4m3_t, float4_e2m1x2_t, 128, 256, 128)
LAUNCH_HIF4B(34, half, 128, 256, 128)
LAUNCH_HIF4B(35, float8_e4m3_t, 128, 128, 256)
LAUNCH_HIF4AB_WITH_ND(36, 128, 128, 128)
LAUNCH_HIF4AB_WITH_ND(37, 128, 256, 128)
LAUNCH_HIF4AB_WITH_ND(38, 256, 128, 128)
LAUNCH_HIF4AB_WITH_ND(39, 128, 128, 256)

#undef LAUNCH_E2M1
#undef LAUNCH_HIF4B
#undef LAUNCH_HIF4AB
#undef LAUNCH_HIF4AB_WITH_ND
