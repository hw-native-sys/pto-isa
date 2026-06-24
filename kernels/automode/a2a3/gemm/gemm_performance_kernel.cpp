/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include "multiBuffer.hpp"

using namespace pto;
using namespace pto_auto;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024; // L0A/L0B ping-pong split (32 KiB per buffer)

// Pipeline mental model (instruction-level):
// - TLOAD     (GM -> L1):   fill aMatTile/bMatTile
// - TEXTRACT  (L1 -> L0):   slice aMatTile/bMatTile into aTile/bTile for the current baseK
// - TMATMUL   (Cube):       cTile = A*B (accumulated over K)
// - TSTORE    (L0C -> GM):  write cTile back to GM
//
// The code still uses PIPE_MTE* events for synchronization because those are the underlying hardware pipes;
// comments refer to the high-level PTO instructions to make tuning easier.

template <typename T, typename U, typename S, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN>
AICORE inline void InitGMOffsets(__gm__ U *&currentSrc0, __gm__ S *&currentSrc1, __gm__ T *&currentDst, __gm__ T *out,
                                 __gm__ U *src0, __gm__ S *src1)
{
    // Work partition (SPMD-style):
    // - Each core owns a contiguous C tile of shape [singleCoreM, singleCoreN].
    // - It reads the corresponding A panel [singleCoreM, K] and B panel [K, singleCoreN].
    constexpr uint32_t mIter = m / singleCoreM;
    uint32_t mIterIdx = get_block_idx() % mIter; // get current launch core idx
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
    uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
    uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentDst = out + gmOffsetC;
}

template <typename T, typename U, typename S, int m, int n, uint32_t baseM, uint32_t baseN, uint32_t singleCoreK,
          typename ResTile>
AICORE inline void StoreResult(ResTile &cTile, __gm__ T *currentDst, uint32_t i, uint32_t j)
{
    // TSTORE stage: write the finished C tile [baseM, baseN] back to GM.
    // the data size read from L0C after single k loop is [baseM, baseN]
    using NDValidShapeC = TileShape2D<T, baseM, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<T, m, n, Layout::ND>; // stride use global C m n
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;

    GlobalDataOut dstGlobal(currentDst + i * baseM * n + j * baseN);
    TSTORE<STPhase::Final>(dstGlobal, cTile);
}

template <typename T, typename U, typename S, typename B, uint32_t blockDim, int m, int k, int n, int validM,
          int validK, int validN, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM,
          uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunGemmE2E(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    __gm__ U *currentSrc0 = nullptr;
    __gm__ S *currentSrc1 = nullptr;
    __gm__ T *currentDst = nullptr;
    InitGMOffsets<T, U, S, m, k, n, singleCoreM, singleCoreK, singleCoreN>(currentSrc0, currentSrc1, currentDst, out,
                                                                           src0, src1);

    using TileMatA =
        Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>;
    using TileMatB =
        Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>;

    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, m, k, Layout::ND>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

    using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, k, n, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    ResTile cTile;

    constexpr uint32_t mLoop = singleCoreM / baseM;
    constexpr uint32_t nLoop = singleCoreN / baseN;
    constexpr uint32_t kLoop = singleCoreK / baseK;

    MultiBuffered<BUFFER_NUM> db;
    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    for (uint32_t i = 0; i < mLoop; i++) {
        for (uint32_t j = 0; j < nLoop; j++) {
            constexpr int NumIters = kLoop / stepKa;
            db.loop<Range<NumIters>>([&](auto outer_ctx) {
                int outer_iter = outer_ctx.iter;
                int outer_buf = outer_ctx.bufferId;

                TileMatA aMatTile;
                TileMatB bMatTile;
                GlobalDataSrcA gmA(currentSrc0 + i * singleCoreK * baseM + outer_iter * stepKa * baseK);
                GlobalDataSrcB gmB(currentSrc1 + j * singleCoreK * baseN + outer_iter * stepKa * baseK);

                TLOAD(aMatTile, gmA);
                TLOAD(bMatTile, gmB);

                MultiBuffered<BUFFER_NUM> inner_db;

                inner_db.loop<Range<stepKa>>([&](auto inner_ctx) {
                    int inner_iter = inner_ctx.iter;
                    int inner_buf = inner_ctx.bufferId;
                    int general_iter = outer_iter * stepKa + inner_iter;
                    bool first = inner_iter == 0 && inner_buf == 0 && outer_iter == 0 && outer_buf == 0;
                    bool last = inner_iter == stepKa - 1 && outer_iter == NumIters - 1 && inner_buf == BUFFER_NUM - 1 &&
                                outer_buf == BUFFER_NUM - 1;

                    TEXTRACT(aTile[inner_buf], aMatTile, 0, (general_iter % stepKa) * baseK);
                    TEXTRACT(bTile[inner_buf], bMatTile, (general_iter % stepKb) * baseK, 0);

                    if (first) {
                        TMATMUL<AccPhase::Partial>(cTile, aTile[inner_buf], bTile[inner_buf]);
                    } else if (last) {
                        TMATMUL_ACC<AccPhase::Final>(cTile, cTile, aTile[inner_buf], bTile[inner_buf]);
                    } else {
                        TMATMUL_ACC<AccPhase::Partial>(cTile, cTile, aTile[inner_buf], bTile[inner_buf]);
                    }
                });
            });

            StoreResult<T, U, S, m, n, baseM, baseN, singleCoreK, ResTile>(cTile, currentDst, i, j);
        }
    }
}

template <typename T, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
          uint32_t stepKb, uint32_t stepN>
__global__ AICORE void GemmPerformance(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    RunGemmE2E<float, half, half, float, blockDim, m, k, n, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM,
               baseK, baseN, stepM, stepKa, stepKb, stepN>(reinterpret_cast<__gm__ float *>(out),
                                                           reinterpret_cast<__gm__ half *>(src0),
                                                           reinterpret_cast<__gm__ half *>(src1));
}

template <typename T>
void LaunchGEMME2E(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    constexpr uint32_t blockDim = 24;
    constexpr uint32_t m = 6144;
    constexpr uint32_t n = 6144;
    constexpr uint32_t k = 6144;
    constexpr uint32_t singleCoreM = 1536;
    constexpr uint32_t singleCoreN = 1024;
    constexpr uint32_t singleCoreK = 6144;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseN = 256;
    constexpr uint32_t baseK = 64;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 4;
    constexpr uint32_t stepKb = 4;
    constexpr uint32_t stepN = 1;
    GemmPerformance<T, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa,
                    stepKb, stepN><<<blockDim, nullptr, stream>>>(out, src0, src1);
}

template void LaunchGEMME2E<uint16_t>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
