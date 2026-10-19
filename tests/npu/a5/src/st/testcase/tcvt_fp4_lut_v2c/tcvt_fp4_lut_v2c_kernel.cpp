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
#include <pto/common/constants.hpp>
#include <pto/common/fifo.hpp>

#include "cases.h"

using namespace pto;

namespace {
#ifdef __DAV_CUBE__
constexpr bool kCube = true;
#else
constexpr bool kCube = false;
#endif
#ifdef __DAV_VEC__
constexpr bool kVec = true;
#else
constexpr bool kVec = false;
#endif

// Check the full physical mapping derived from Tile's boxed-layout header.
template <typename NZ, typename ZN>
AICORE constexpr bool hasExactTransposeAlias()
{
    for (int row = 0; row < NZ::Rows; ++row) {
        for (int col = 0; col < NZ::Cols; ++col) {
            int nzOffset = ((col / NZ::InnerCols) * (NZ::Rows / NZ::InnerRows) + row / NZ::InnerRows) * NZ::InnerNumel +
                           (row % NZ::InnerRows) * NZ::InnerCols + col % NZ::InnerCols;
            int znOffset = ((col / ZN::InnerRows) * (ZN::Cols / ZN::InnerCols) + row / ZN::InnerCols) * ZN::InnerNumel +
                           (row % ZN::InnerCols) * ZN::InnerRows + col % ZN::InnerRows;
            if (nzOffset != znOffset) {
                return false;
            }
        }
    }
    return true;
}

template <int M, int TotalK, int N, int TileK, bool NK>
__global__ AICORE void runFp4LutV2c(__gm__ float* output, __gm__ float8_e4m3_t* a, __gm__ uint8_t* packedB)
{
    constexpr int sourceRows = NK ? N : TileK;
    constexpr int sourceCols = NK ? TileK : N;
    constexpr int sourceBytes = sourceRows * sourceCols / 2;
    constexpr int finalNzBytes = sourceRows * sourceCols;
    constexpr int packedRowStride = NK ? TotalK / 2 : N / 2;
    constexpr int kTiles = TotalK / TileK;
    constexpr uint16_t flagId = 0;
    constexpr uint8_t fifoDepth = 2;
    constexpr uint32_t fifoL1Base = 0x10000;
    static_assert(TotalK % TileK == 0 && TileK % 32 == 0 && N % 32 == 0);
    static_assert(sourceCols % 64 == 0, "Packed FP4 source rows must occupy whole 32-byte blocks");
    static_assert(sourceBytes + finalNzBytes <= 24 * 1024, "Only source and final NZ occupy UB");
    static_assert(M * TileK <= fifoL1Base, "A L1 storage must not overlap the FIFO");

    using Source = Tile<TileType::Vec, float4_e2m1x2_t, sourceRows, sourceCols>;
    using RawSource = Tile<TileType::Vec, uint8_t, sourceRows, sourceCols / 2>;
    using FinalNZ = Tile<
        TileType::Vec, float8_e4m3_t, sourceRows, sourceCols, BLayout::ColMajor, sourceRows, sourceCols,
        SLayout::RowMajor, 512>;
    using MatNZ = Tile<
        TileType::Mat, float8_e4m3_t, sourceRows, sourceCols, BLayout::ColMajor, sourceRows, sourceCols,
        SLayout::RowMajor, 512>;
    using MatZN = Tile<TileType::Mat, float8_e4m3_t, TileK, N, BLayout::RowMajor, TileK, N, SLayout::ColMajor, 512>;
    static_assert(Source::Numel / Source::kPackedElementsPerByte == sourceBytes);
    static_assert(RawSource::Numel == sourceBytes && FinalNZ::Numel == finalNzBytes);
    static_assert(MatNZ::Numel == MatZN::Numel && MatNZ::Loc == MatZN::Loc);
    static_assert(!NK || hasExactTransposeAlias<MatNZ, MatZN>(), "Whole Mat NZ/ZN alias must preserve every byte");

    // Same FIFO and event protocol as tpushpop_vc_nosplit, including deferred TFREE.
    using MatPipe = TPipe<flagId, Direction::DIR_V2C, finalNzBytes, fifoDepth>;
    MatPipe pipe((__gm__ void*)(uint64_t)0, (uint32_t)0, fifoL1Base);

    if constexpr (kVec) {
        if (get_subblockid() == 0) {
            Source source;
            RawSource rawSource;
            FinalNZ finalNZ;
            // Raw uint8 view aliases the packed input; it is not an extra allocation.
            TASSIGN(source, 0);
            TASSIGN(rawSource, 0);
            TASSIGN(finalNZ, sourceBytes);
            using GlobalPackedB = GlobalTensor<
                uint8_t, Shape<1, 1, 1, sourceRows, sourceCols / 2>, pto::Stride<1, 1, 1, packedRowStride, 1>>;
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            for (int tile = 0; tile < kTiles; ++tile) {
                // KN: k0*(N/2). NK: k0/2, preserving TotalK/2 GM row stride.
                size_t packedOffset = NK ? tile * TileK / 2 : tile * TileK * (N / 2);
                GlobalPackedB globalPackedB(packedB + packedOffset);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                TLOAD(rawSource, globalPackedB);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                TCVT(finalNZ, source, RoundMode::CAST_RINT);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TPUSH<MatPipe, FinalNZ, TileSplitAxis::TILE_NO_SPLIT>(pipe, finalNZ);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            pipe_barrier(PIPE_ALL);
        }
    }

    if constexpr (kCube) {
        using GlobalA = GlobalTensor<
            float8_e4m3_t, Shape<1, 1, 1, M, TileK>, pto::Stride<M * TotalK, M * TotalK, M * TotalK, TotalK, 1>>;
        using GlobalC = GlobalTensor<float, Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
        using MatA = Tile<TileType::Mat, float8_e4m3_t, M, TileK, BLayout::ColMajor, M, TileK, SLayout::RowMajor, 512>;
        using Left = TileLeft<float8_e4m3_t, M, TileK>;
        using Right = TileRight<float8_e4m3_t, TileK, N>;
        using Acc = TileAcc<float, M, N>;
        MatA aMat;
        MatNZ bMat;
        Left aTile;
        Right bTile;
        Acc cTile;
        TASSIGN(aMat, 0);
        TASSIGN(aTile, 0);
        TASSIGN(bTile, 0);
        TASSIGN(cTile, 0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        for (int tile = 0; tile < kTiles; ++tile) {
            GlobalA globalA(a + tile * TileK);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            TLOAD(aMat, globalA);
            TPOP<MatPipe, MatNZ, TileSplitAxis::TILE_NO_SPLIT>(pipe, bMat);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            TMOV(aTile, aMat);
            if constexpr (NK) {
                MatZN bMatZN;
                // Whole FIFO tile alias after every pop, including wrapped slots.
                TRESHAPE(bMatZN, bMat);
                TMOV(bTile, bMatZN);
            } else {
                TMOV(bTile, bMat);
            }
            // Existing free publishes on MTE1, after the L1 -> L0 reads above.
            TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            if (tile == 0) {
                TMATMUL(cTile, aTile, bTile);
            } else {
                TMATMUL_ACC(cTile, cTile, aTile, bTile);
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        GlobalC globalC(output);
        TSTORE(globalC, cTile);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }
}
} // namespace

void launchFp4LutV2c(int caseId, unsigned char* output, unsigned char* a, unsigned char* packedB, void* stream)
{
    switch (static_cast<Fp4LutV2cCaseId>(caseId)) {
#define FP4_LUT_V2C_LAUNCH(Name, M, TotalK, N, TileK, NK)                                                             \
    case Fp4LutV2cCaseId::Name:                                                                                       \
        runFp4LutV2c<M, TotalK, N, TileK, NK>                                                                         \
            <<<1, nullptr, stream>>>(reinterpret_cast<float*>(output), reinterpret_cast<float8_e4m3_t*>(a), packedB); \
        break;
        FP4_LUT_V2C_CASES(FP4_LUT_V2C_LAUNCH)
#undef FP4_LUT_V2C_LAUNCH
    }
}
