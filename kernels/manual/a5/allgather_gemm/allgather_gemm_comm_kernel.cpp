/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>

#include <pto/pto-inst.hpp>

#ifdef __CCE_AICORE__
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#endif
#include "common.hpp"
#include "gemm_config.hpp"
#include "ready_queue.hpp"

#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 4
#endif
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;

#ifdef __CCE_AICORE__

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using TileData = pto::Tile<pto::TileType::Vec, half, G_BASE_M, G_BASE_N, pto::BLayout::RowMajor, -1, -1>;

constexpr size_t TILE_UB_BYTES = ((G_BASE_M * G_BASE_N * sizeof(half) + 1023) / 1024) * 1024;

struct CommParams {
    int myRank;
    int nRanks;
    int numRemoteRanks;
    int mTilesLocal;
    int kTiles;
    int numTilesPerSrc;
    int chunkSize;
    int numChunks;
};

AICORE inline CommParams BuildCommParams(__gm__ HcclDeviceContext *hcclCtx, volatile __gm__ ChunkFlagMatrix *flags)
{
    CommParams p;
    p.myRank = static_cast<int>(hcclCtx->rankId);
    p.nRanks = static_cast<int>(hcclCtx->rankNum);
    if (p.nRanks <= 0) {
        p.nRanks = 1;
    }
    p.numRemoteRanks = p.nRanks - 1;
    int mTiles = static_cast<int>(G_M / G_BASE_M);
    p.mTilesLocal = mTiles / p.nRanks;
    p.kTiles = static_cast<int>(G_K / G_BASE_N);
    p.numTilesPerSrc = p.mTilesLocal * p.kTiles;
    p.chunkSize = flags->chunk_size;
    p.numChunks = flags->num_chunks_per_src;
    return p;
}

AICORE inline void SetupLocalFlags(volatile __gm__ ChunkFlagMatrix *flags, volatile __gm__ int32_t *summaryBase,
                                   int myRank, int numChunks)
{
    for (int c = 0; c < numChunks; ++c) {
        SetChunkFlagReady(flags, myRank, c);
    }
    // Use TNOTIFY AtomicAdd (matching TWAIT) instead of direct store,
    // so that TWAIT's hardware signal mechanism is properly triggered.
    volatile __gm__ int32_t *ptr = summaryBase + myRank;
    pto::comm::Signal sig(reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ int32_t *>(ptr)));
    pto::comm::TNOTIFY(sig, numChunks, pto::comm::NotifyOp::AtomicAdd);
}

AICORE inline void InitTileBuffers(TileData &pingTile, TileData &pongTile)
{
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, TILE_UB_BYTES);
}

AICORE inline int DestIdxToRank(int destIdx, int myRank)
{
    return (destIdx < myRank) ? destIdx : (destIdx + 1);
}

struct RemoteEndpoint {
    __gm__ ChunkFlagMatrix *chunkFlags;
    __gm__ int32_t *summarySrc;
};

AICORE inline RemoteEndpoint GetRemoteEndpoint(__gm__ HcclDeviceContext *hcclCtx, __gm__ ChunkFlagMatrix *flagsMut,
                                               volatile __gm__ ChunkFlagMatrix *flags, int destRank, int myRank)
{
    RemoteEndpoint ep;
    ep.chunkFlags = reinterpret_cast<__gm__ ChunkFlagMatrix *>(HcclRemotePtr(hcclCtx, flagsMut, destRank));
    ep.summarySrc = reinterpret_cast<__gm__ int32_t *>(
                        reinterpret_cast<__gm__ uint8_t *>(HcclRemotePtr(hcclCtx, flagsMut, destRank)) +
                        ChunkFlagMatrixBytes(flags)) +
                    myRank;
    return ep;
}

struct DispatchContext {
    __gm__ half *shmemInput;
    __gm__ HcclDeviceContext *hcclCtx;
    volatile __gm__ ChunkFlagMatrix *flags;
    const ShapeDyn *tileShape;
    const StrideDyn *tileStride;
    TileData *pingTile;
    TileData *pongTile;
    CommParams p;
    int blockIdx;
    int numBlocks;
};

// 将一个 chunk 对应的所有 tile 通过 TPUT 传输到远端，完成后设置远端 flag
AICORE inline void TransferChunkToRemote(__gm__ half *shmemInput, __gm__ HcclDeviceContext *hcclCtx,
                                         __gm__ ChunkFlagMatrix *remoteChunkFlags, __gm__ int32_t *remoteSummarySrc,
                                         const ShapeDyn &tileShape, const StrideDyn &tileStride, TileData &pingTile,
                                         TileData &pongTile, const CommParams &p, int destRank, int chunkIdx)
{
    int tileStart = chunkIdx * p.chunkSize;
    int tileEnd = tileStart + p.chunkSize;
    if (tileEnd > p.numTilesPerSrc) {
        tileEnd = p.numTilesPerSrc;
    }

    __gm__ half *remoteInput = HcclRemotePtr(hcclCtx, shmemInput, destRank);
    for (int t = tileStart; t < tileEnd; ++t) {
        int miLocal = t / p.kTiles;
        int kt = t % p.kTiles;
        int miGlobal = p.myRank * p.mTilesLocal + miLocal;
        uint64_t colOff = static_cast<uint64_t>(kt * static_cast<int>(G_BASE_N));
        uint64_t rowOff = static_cast<uint64_t>(miGlobal) * G_BASE_M;
        uint64_t offset = rowOff * G_K + colOff;
        Global srcG(shmemInput + offset, tileShape, tileStride);
        Global dstG(remoteInput + offset, tileShape, tileStride);
        pto::comm::TPUT(dstG, srcG, pingTile, pongTile);
    }

    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    SetRemoteChunkFlagReady(remoteChunkFlags, p.myRank, chunkIdx, remoteSummarySrc);
}

// 硬件 block 数少于 dest 数：按 work_id 均分
AICORE inline void DispatchFewBlocks(const DispatchContext &ctx)
{
    __gm__ ChunkFlagMatrix *flagsMut = const_cast<__gm__ ChunkFlagMatrix *>(ctx.flags);
    int totalWork = ctx.p.numRemoteRanks * ctx.p.numChunks;

    for (int workId = ctx.blockIdx; workId < totalWork; workId += ctx.numBlocks) {
        int destIdx = workId / ctx.p.numChunks;
        int chunkIdx = workId % ctx.p.numChunks;
        int destRank = DestIdxToRank(destIdx, ctx.p.myRank);

        RemoteEndpoint ep = GetRemoteEndpoint(ctx.hcclCtx, flagsMut, ctx.flags, destRank, ctx.p.myRank);

        TransferChunkToRemote(ctx.shmemInput, ctx.hcclCtx, ep.chunkFlags, ep.summarySrc, *ctx.tileShape,
                              *ctx.tileStride, *ctx.pingTile, *ctx.pongTile, ctx.p, destRank, chunkIdx);
    }
}

// 硬件 block 数 >= dest 数：每 dest 分配连续 block 区间
AICORE inline void DispatchManyBlocks(const DispatchContext &ctx)
{
    __gm__ ChunkFlagMatrix *flagsMut = const_cast<__gm__ ChunkFlagMatrix *>(ctx.flags);
    if (ctx.p.numRemoteRanks <= 0) {
        return;
    }
    int blocksPerDest = ctx.numBlocks / ctx.p.numRemoteRanks;
    if (blocksPerDest <= 0) {
        blocksPerDest = 1;
    }
    int destIdx = ctx.blockIdx / blocksPerDest;
    int localIdx = ctx.blockIdx % blocksPerDest;
    if (destIdx >= ctx.p.numRemoteRanks) {
        return;
    }

    int destRank = DestIdxToRank(destIdx, ctx.p.myRank);
    RemoteEndpoint ep = GetRemoteEndpoint(ctx.hcclCtx, flagsMut, ctx.flags, destRank, ctx.p.myRank);

    int chunksPerBlock = (ctx.p.numChunks + blocksPerDest - 1) / blocksPerDest;
    int chunkStart = localIdx * chunksPerBlock;
    int chunkEnd = chunkStart + chunksPerBlock;
    if (chunkEnd > ctx.p.numChunks) {
        chunkEnd = ctx.p.numChunks;
    }

    for (int chunkIdx = chunkStart; chunkIdx < chunkEnd; ++chunkIdx) {
        TransferChunkToRemote(ctx.shmemInput, ctx.hcclCtx, ep.chunkFlags, ep.summarySrc, *ctx.tileShape,
                              *ctx.tileStride, *ctx.pingTile, *ctx.pongTile, ctx.p, destRank, chunkIdx);
    }
}

AICORE inline void CommAIVRoleStreamingParallel(__gm__ half *shmemInput, __gm__ ChunkFlagMatrix *chunkFlags,
                                                __gm__ HcclDeviceContext *hcclCtx, int blockIdx, int numBlocks)
{
    volatile __gm__ ChunkFlagMatrix *flags = reinterpret_cast<volatile __gm__ ChunkFlagMatrix *>(chunkFlags);
    CommParams p = BuildCommParams(hcclCtx, flags);

    if (p.numRemoteRanks <= 0 || numBlocks <= 0) {
        return;
    }

    volatile __gm__ int32_t *summaryBase = GetSummaryBase(flags);
    if (blockIdx == 0) {
        SetupLocalFlags(flags, summaryBase, p.myRank, p.numChunks);
    }

    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    InitTileBuffers(pingTile, pongTile);

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_K, G_BASE_M * G_K, G_BASE_M * G_K, G_K, 1);

    DispatchContext ctx{shmemInput, hcclCtx,   flags, &tileShape, &tileStride,
                        &pingTile,  &pongTile, p,     blockIdx,   numBlocks};

    if (numBlocks < p.numRemoteRanks) {
        DispatchFewBlocks(ctx);
    } else {
        DispatchManyBlocks(ctx);
    }
}

#endif // __CCE_AICORE__

__global__ AICORE void RingCommStreamingKernel(__gm__ uint8_t *shmemInput, __gm__ uint8_t *chunkFlags,
                                               __gm__ uint8_t *hcclCtxRaw, int blockNum)
{
#ifdef __CCE_AICORE__
    int blockIdx = get_block_idx();
    __gm__ HcclDeviceContext *hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtxRaw);
    int nRanks = static_cast<int>(hcclCtx->rankNum);

    if (nRanks <= 1) {
        return;
    }

    CommAIVRoleStreamingParallel(reinterpret_cast<__gm__ half *>(shmemInput),
                                 reinterpret_cast<__gm__ ChunkFlagMatrix *>(chunkFlags), hcclCtx, blockIdx, blockNum);
#endif
}

void launchRingCommStreaming(uint8_t *shmemInput, uint8_t *chunkFlags, uint8_t *hcclCtx, int nRanks, void *stream)
{
    int numRemoteRanks = nRanks - 1;
    if (numRemoteRanks <= 0) {
        return;
    }
    int blocksPerDest = COMM_BLOCK_NUM / numRemoteRanks;
    if (blocksPerDest < 1) {
        blocksPerDest = 1;
    }
    int totalBlocks = numRemoteRanks * blocksPerDest;
    RingCommStreamingKernel<<<totalBlocks, nullptr, stream>>>(shmemInput, chunkFlags, hcclCtx, totalBlocks);
}
