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
#include "gemm_config.hpp"
#include "ready_queue.hpp"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;
constexpr uint32_t G_STEP_KA = 4;
constexpr uint32_t G_STEP_KB = 4;

static_assert(G_BASE_N == G_BASE_K * G_STEP_KA, "Expect one comm K-block equals one compute step pack");
static_assert(G_BASE_M > 0, "G_BASE_M must be positive");
static_assert(G_BASE_N > 0, "G_BASE_N must be positive");
static_assert(G_BASE_K > 0, "G_BASE_K must be positive");

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 24
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;

#ifdef __CCE_AICORE__
using namespace pto;

using TileMatAData = Tile<TileType::Mat, half, G_BASE_M, G_BASE_K * G_STEP_KA, BLayout::ColMajor, G_BASE_M,
                          G_BASE_K * G_STEP_KA, SLayout::RowMajor>;
using TileMatBData = Tile<TileType::Mat, half, G_BASE_K * G_STEP_KB, G_BASE_N, BLayout::RowMajor, G_BASE_K * G_STEP_KB,
                          G_BASE_N, SLayout::ColMajor>;
using LeftTile = TileLeft<half, G_BASE_M, G_BASE_K, G_BASE_M, G_BASE_K>;
using RightTile = TileRight<half, G_BASE_K, G_BASE_N, G_BASE_K, G_BASE_N>;
using ResTile = TileAcc<float, G_BASE_M, G_BASE_N, G_BASE_M, G_BASE_N>;

using NDValidShapeC = TileShape2D<float, G_BASE_M, G_BASE_N>;
using NDWholeShapeC = BaseShape2D<float, G_M, G_N>;
using GlobalDataOut = GlobalTensor<float, NDValidShapeC, NDWholeShapeC>;

// ---------------------------------------------------------------------------
// ProcessKIterationContinuous: 单次 K-iteration 的 L1 load + L0 extract + matmul
// ---------------------------------------------------------------------------
template <typename T, typename U, typename S, int M, int K, int N, uint32_t baseM, uint32_t baseK, uint32_t baseN,
          uint32_t stepKa, uint32_t stepKb>
AICORE inline void ProcessKIterationContinuous(
    uint32_t localKIter, uint32_t globalKIter, __gm__ U *currentSrc0, __gm__ S *currentSrc1,
    Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>
        aMatTile[BUFFER_NUM],
    Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>
        bMatTile[BUFFER_NUM],
    TileLeft<U, baseM, baseK, baseM, baseK> aTile[BUFFER_NUM],
    TileRight<S, baseK, baseN, baseK, baseN> bTile[BUFFER_NUM], TileAcc<T, baseM, baseN, baseM, baseN> &cTile,
    uint8_t &mte2DBFlag, uint8_t &mte1DBFlag)
{
    using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, M, K, Layout::ND>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

    using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, K, N, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    const uint32_t kModStepKa = localKIter % stepKa;

    if (kModStepKa == 0) {
        GlobalDataSrcA gmA(currentSrc0 + localKIter * baseK);
        GlobalDataSrcB gmB(currentSrc1 + localKIter * baseK);

        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    wait_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);

    if (kModStepKa == 0) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * baseK);

    if (kModStepKa == 0) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (localKIter % stepKb) * baseK, 0);

    if ((localKIter + 1) % stepKa == 0) {
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)currMte2Idx);
    }

    set_flag(PIPE_MTE1, PIPE_M, (event_t)mte1DBFlag);
    wait_flag(PIPE_MTE1, PIPE_M, (event_t)mte1DBFlag);

    if (globalKIter == 0) {
        TMATMUL(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    }

    set_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// 同步 flags 初始化 / 清理 / 写回辅助函数
// ---------------------------------------------------------------------------
AICORE inline void InitPipelineFlags()
{
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
}

AICORE inline void DrainPipelineFlags()
{
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
}

AICORE inline void StoreCTile(__gm__ float *tileDst, ResTile &cTile)
{
    GlobalDataOut dstGlobal(tileDst);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    // Drain the outstanding writeback before the next N-tile reuses L0C on A5.
    pipe_barrier(PIPE_ALL);
}

// ---------------------------------------------------------------------------
// RunKBlockRange: 对连续 K-block 范围 [kb_start, kb_end) 执行矩阵乘
// ---------------------------------------------------------------------------
AICORE inline void RunKBlockRange(int kb_start, int kb_end, __gm__ half *aRowBase, __gm__ half *bColBase,
                                  TileMatAData aMatTile[BUFFER_NUM], TileMatBData bMatTile[BUFFER_NUM],
                                  LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], ResTile &cTile,
                                  uint8_t &mte2DBFlag, uint8_t &mte1DBFlag, uint32_t &globalKIter)
{
    constexpr uint32_t k_iters_per_block = G_BASE_N / G_BASE_K;
    for (int kb = kb_start; kb < kb_end; ++kb) {
        int k_col_offset = kb * static_cast<int>(G_BASE_N);
        __gm__ half *aSrc = aRowBase + k_col_offset;
        __gm__ half *bSrc = bColBase + k_col_offset;
        for (uint32_t kIter = 0; kIter < k_iters_per_block; ++kIter) {
            ProcessKIterationContinuous<float, half, half, G_M, G_K, G_N, G_BASE_M, G_BASE_K, G_BASE_N, G_STEP_KA,
                                        G_STEP_KB>(kIter, globalKIter, aSrc, bSrc, aMatTile, bMatTile, aTile, bTile,
                                                   cTile, mte2DBFlag, mte1DBFlag);
            globalKIter++;
        }
    }
}

// ---------------------------------------------------------------------------
// ProcessStreamingTiles: tile-ready 轮询 + K-block 迭代，处理单个 N-tile
//   在 ComputeRowGroupStreaming 的 ni 循环内调用
// ---------------------------------------------------------------------------
AICORE inline void ProcessStreamingTiles(volatile __gm__ TileFlagMatrix *flags, volatile __gm__ int32_t *summary_base,
                                         int src_rank, int first_streaming_tile, int last_streaming_tile,
                                         int block_start, int block_end, __gm__ half *aRowBase, __gm__ half *bColBase,
                                         TileMatAData aMatTile[BUFFER_NUM], TileMatBData bMatTile[BUFFER_NUM],
                                         LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], ResTile &cTile,
                                         uint8_t &mte2DBFlag, uint8_t &mte1DBFlag, uint32_t &globalKIter)
{
    int num_st = (first_streaming_tile <= last_streaming_tile) ? (last_streaming_tile - first_streaming_tile + 1) : 0;

    int32_t epoch_base = (flags->epoch - 1) * flags->num_tiles_per_src;

    uint64_t done = 0;
    int processed_count = 0;
    int next_st = first_streaming_tile;

    while (processed_count < num_st) {
        int32_t ready_count = GetReadyCountFromSrc(summary_base, src_rank);
        if (ready_count <= epoch_base + processed_count) {
            WaitReadyCountFromSrc(summary_base, src_rank, epoch_base + processed_count + 1);
        }

        for (int scan_cnt = 0; scan_cnt < num_st && processed_count < num_st; ++scan_cnt) {
            int st = next_st;
            next_st = (next_st >= last_streaming_tile) ? first_streaming_tile : (next_st + 1);

            int idx = st - first_streaming_tile;
            if ((done & (1ULL << idx)) != 0)
                continue;
            if (!IsTileReady(flags, src_rank, st))
                continue;

            done |= (1ULL << idx);
            processed_count++;

            int tile_start_blk = st * flags->tile_size;
            int tile_end_blk =
                (tile_start_blk + flags->tile_size > block_end) ? block_end : (tile_start_blk + flags->tile_size);
            int kb_start = (tile_start_blk > block_start) ? (tile_start_blk - block_start) : 0;
            int kb_end = tile_end_blk - block_start;
            if (kb_end <= kb_start)
                continue;

            RunKBlockRange(kb_start, kb_end, aRowBase, bColBase, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag,
                           mte1DBFlag, globalKIter);
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeRowGroupStreaming: 远程 rank 数据，等 tile 就绪信号后流式计算
// ---------------------------------------------------------------------------
AICORE inline void ComputeRowGroupStreaming(__gm__ float *output, __gm__ half *shmem_input, __gm__ half *src1,
                                            __gm__ TileFlagMatrix *tile_flags, int mi, int m_tiles_per_rank,
                                            int k_chunks, TileMatAData aMatTile[BUFFER_NUM],
                                            TileMatBData bMatTile[BUFFER_NUM], LeftTile aTile[BUFFER_NUM],
                                            RightTile bTile[BUFFER_NUM], ResTile &cTile)
{
    constexpr uint32_t n_tiles = G_N / G_BASE_N;

    volatile __gm__ TileFlagMatrix *flags = reinterpret_cast<volatile __gm__ TileFlagMatrix *>(tile_flags);
    volatile __gm__ int32_t *summary_base = GetSummaryBase(flags);

    int src_rank = mi / m_tiles_per_rank;
    int mi_local = mi % m_tiles_per_rank;
    int block_start = mi_local * k_chunks;
    int block_end = (mi_local + 1) * k_chunks;

    int tile_size = flags->tile_size;
    int num_tiles_per_src = flags->num_tiles_per_src;
    int first_streaming_tile = 0;
    int last_streaming_tile = -1;
    if (block_end > block_start && num_tiles_per_src > 0 && tile_size > 0) {
        first_streaming_tile = block_start / tile_size;
        last_streaming_tile = (block_end - 1) / tile_size;
        if (last_streaming_tile >= num_tiles_per_src) {
            last_streaming_tile = num_tiles_per_src - 1;
        }
    }

    __gm__ half *aRowBase = shmem_input + static_cast<uint64_t>(mi) * G_BASE_M * G_K;
    __gm__ float *outRowBase = output + static_cast<uint64_t>(mi) * G_BASE_M * G_N;

    for (uint32_t ni = 0; ni < n_tiles; ++ni) {
        __gm__ float *tileDst = outRowBase + ni * G_BASE_N;
        __gm__ half *bColBase = src1 + static_cast<uint64_t>(ni) * G_BASE_N * G_K;

        uint8_t mte2DBFlag = 0;
        uint8_t mte1DBFlag = 0;
        uint32_t globalKIter = 0;

        InitPipelineFlags();

        ProcessStreamingTiles(flags, summary_base, src_rank, first_streaming_tile, last_streaming_tile, block_start,
                              block_end, aRowBase, bColBase, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag,
                              mte1DBFlag, globalKIter);

        DrainPipelineFlags();
        StoreCTile(tileDst, cTile);
    }

    pipe_barrier(PIPE_ALL);
}

// ---------------------------------------------------------------------------
// ComputeRowGroupDirect: 本地 rank 数据已就绪，直接计算
// ---------------------------------------------------------------------------
AICORE inline void ComputeRowGroupDirect(__gm__ float *output, __gm__ half *shmem_input, __gm__ half *src1, int mi,
                                         int k_chunks, TileMatAData aMatTile[BUFFER_NUM],
                                         TileMatBData bMatTile[BUFFER_NUM], LeftTile aTile[BUFFER_NUM],
                                         RightTile bTile[BUFFER_NUM], ResTile &cTile)
{
    constexpr uint32_t n_tiles = G_N / G_BASE_N;

    __gm__ half *aRowBase = shmem_input + static_cast<uint64_t>(mi) * G_BASE_M * G_K;
    __gm__ float *outRowBase = output + static_cast<uint64_t>(mi) * G_BASE_M * G_N;

    for (uint32_t ni = 0; ni < n_tiles; ++ni) {
        __gm__ float *tileDst = outRowBase + ni * G_BASE_N;
        __gm__ half *bColBase = src1 + static_cast<uint64_t>(ni) * G_BASE_N * G_K;

        uint8_t mte2DBFlag = 0;
        uint8_t mte1DBFlag = 0;
        uint32_t globalKIter = 0;

        InitPipelineFlags();

        RunKBlockRange(0, k_chunks, aRowBase, bColBase, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag, mte1DBFlag,
                       globalKIter);

        DrainPipelineFlags();
        StoreCTile(tileDst, cTile);
    }

    pipe_barrier(PIPE_ALL);
}

// ---------------------------------------------------------------------------
// AllocateComputeTiles: 初始化 L1/L0 tile buffer 地址映射
// ---------------------------------------------------------------------------
AICORE inline void AllocateComputeTiles(TileMatAData aMatTile[BUFFER_NUM], TileMatBData bMatTile[BUFFER_NUM],
                                        LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], ResTile &cTile)
{
    constexpr size_t l1ASize = G_BASE_M * G_BASE_K * G_STEP_KA * sizeof(half);
    constexpr size_t l1BSize = G_BASE_K * G_STEP_KB * G_BASE_N * sizeof(half);
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + l1ASize);
    TASSIGN(bMatTile[0], 0x0 + BUFFER_NUM * l1ASize);
    TASSIGN(bMatTile[1], 0x0 + BUFFER_NUM * l1ASize + l1BSize);

    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(cTile, 0x0);
}

// ---------------------------------------------------------------------------
// AllGatherGemmComputeStreamingImpl
//   Phase 1: 本地 rank row-group 直接计算
//   Phase 2: 远程 rank row-group streaming 等待后计算
// ---------------------------------------------------------------------------
AICORE inline void AllGatherGemmComputeStreamingImpl(__gm__ float *output, __gm__ half *shmem_input, __gm__ half *src1,
                                                     __gm__ TileFlagMatrix *tile_flags, int launch_blocks)
{
    const int block_id = get_block_idx();

    volatile __gm__ TileFlagMatrix *flags = reinterpret_cast<volatile __gm__ TileFlagMatrix *>(tile_flags);

    int n_ranks = flags->num_ranks;
    if (n_ranks <= 0) {
        return;
    }
    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    int m_tiles_per_rank = m_tiles / n_ranks;
    int k_chunks = static_cast<int>(G_K / G_BASE_N);

    TileMatAData aMatTile[BUFFER_NUM];
    TileMatBData bMatTile[BUFFER_NUM];
    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    ResTile cTile;
    AllocateComputeTiles(aMatTile, bMatTile, aTile, bTile, cTile);

    int my_rank = flags->my_rank;

    if (my_rank >= 0 && my_rank < n_ranks) {
        int local_mi_start = my_rank * m_tiles_per_rank;
        int local_mi_end = local_mi_start + m_tiles_per_rank;
        for (int mi = local_mi_start + block_id; mi < local_mi_end; mi += launch_blocks) {
            ComputeRowGroupDirect(output, shmem_input, src1, mi, k_chunks, aMatTile, bMatTile, aTile, bTile, cTile);
        }
    }

    for (int mi = block_id; mi < m_tiles; mi += launch_blocks) {
        int src_rank = mi / m_tiles_per_rank;
        if (src_rank == my_rank)
            continue;
        ComputeRowGroupStreaming(output, shmem_input, src1, tile_flags, mi, m_tiles_per_rank, k_chunks, aMatTile,
                                 bMatTile, aTile, bTile, cTile);
    }
}

#endif // __CCE_AICORE__

__global__ AICORE void AllGatherGemmComputeStreamingKernel(__gm__ uint8_t *output, __gm__ uint8_t *shmem_input,
                                                           __gm__ uint8_t *src1, __gm__ uint8_t *tile_flags,
                                                           int launch_blocks)
{
#ifdef __CCE_AICORE__
    AllGatherGemmComputeStreamingImpl(
        reinterpret_cast<__gm__ float *>(output), reinterpret_cast<__gm__ half *>(shmem_input),
        reinterpret_cast<__gm__ half *>(src1), reinterpret_cast<__gm__ TileFlagMatrix *>(tile_flags), launch_blocks);
#endif
}

void launchAllGatherGemmComputeStreaming(uint8_t *output, uint8_t *shmem_input, uint8_t *src1, uint8_t *tile_flags,
                                         void *stream, int launch_blocks = COMPUTE_BLOCK_NUM)
{
    AllGatherGemmComputeStreamingKernel<<<launch_blocks, nullptr, stream>>>(output, shmem_input, src1, tile_flags,
                                                                            launch_blocks);
}
