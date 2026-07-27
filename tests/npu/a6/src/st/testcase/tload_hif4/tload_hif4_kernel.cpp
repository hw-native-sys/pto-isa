/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// --------------------------------------------------------------------------------
// tload_hif4_kernel.cpp
//
// Minimal A6 TLOAD-only HiF4 testcase. Verifies that BF16 -> HiF4 data +
// scale can be moved from GM to L1 via the pto-isa TLOAD wrapper for the
// hifloat4x2_t dtype and the HIF4_A_ZZ / HIF4_B_NN scale layouts.
//
// This testcase does NOT exercise TEXTRACT, TMATMUL_MX, or TSTORE. The kernel
// issues the four TLOADs (A data, A scale, B data, B scale), then syncs the
// MTE2 pipeline and returns. Correctness is implicit: if TLOAD compiles and
// the simulator does not hang or fault on the DMA, the GM -> L1 path for
// HiF4 works. Byte-level L1 inspection is left to sim log analysis.
// --------------------------------------------------------------------------------

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr uint32_t HIF4_SCALE_GROUP = 64;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    return (num1 + num2 - 1) / num2 * num2;
}

template <int validM, int validK, int validN>
AICORE inline void RunTloadHif4Impl(
    __gm__ hifloat4x2_t* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData, __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleK = validK / HIF4_SCALE_GROUP;
    constexpr int scaleKCols = scaleK * HIF4_COL_LEN;

    // L1 MAT tiles (data). SFractalSize = fractalABSize (512) for 1-byte data.
    using TileMatA = Tile<
        TileType::Mat, hifloat4x2_t, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, hifloat4x2_t, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    // L1 MAT tiles (scale). SFractalSize = 32 (the [16,2]=32B L1 fractal cell — forced by TLoadMxCubeCheck).
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleKCols, BLayout::RowMajor, validM, scaleKCols, SLayout::RowMajor, 32>;
    using TileScaleB =
        Tile<TileType::Mat, uint8_t, scaleKCols, N, BLayout::ColMajor, scaleKCols, validN, SLayout::ColMajor, 32>;

    // GM tensors.
    using GlobalDataA = GlobalTensor<
        hifloat4x2_t, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        hifloat4x2_t, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::HIF4_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::HIF4_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::HIF4_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::HIF4_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::HIF4_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::HIF4_B_NN>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);

    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);

    // Issue the four GM -> L1 TLOADs. MTE2 pipeline.
    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

    // Drain MTE2 so the writes are observable in L1 before the kernel returns.
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE2, EVENT_ID0);
#endif
}

template <int validM, int validK, int validN>
__global__ AICORE void RunTloadHif4(
    __gm__ hifloat4x2_t* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData, __gm__ uint8_t* bScale)
{
    RunTloadHif4Impl<validM, validK, validN>(aData, aScale, bData, bScale);
}

namespace TloadHif4A6 {
template <int validM, int validK, int validN>
void Launch(uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream)
{
    RunTloadHif4<validM, validK, validN><<<1, nullptr, stream>>>(
        reinterpret_cast<hifloat4x2_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
        reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
}
} // namespace TloadHif4A6

template void TloadHif4A6::Launch<128, 128, 128>(
    uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
